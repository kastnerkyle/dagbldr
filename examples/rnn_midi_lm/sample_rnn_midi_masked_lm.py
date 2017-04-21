#!/usr/bin/env python
from dagbldr import fetch_checkpoint_dict
from dagbldr.datasets import fetch_bach_chorales_music21
from dagbldr.datasets import list_of_array_iterator
import numpy as np

import argparse
import cPickle as pickle
import os
from dagbldr.nodes import np_softmax_activation
import copy

# sample13 gotten from
# leto52:/Tmp/kastner/dagbldr_models/rnn_midi_masked_lm_13-49-00_2017-20-04_45c5e7/45c5e7_model_checkpoint_2350.pkl
def duration_and_pitch_to_pretty_midi(durations, pitches, add_to_name=0):
    import pretty_midi
    # BTAS mapping
    voice_mappings = ["Sitar", "Orchestral Harp", "Acoustic Guitar (nylon)", "Pan Flute"]
    voice_velocity = [60, 100, 100, 50]
    voice_offset = [0, 0, 0, 12]
    len_durations = len(durations)
    order = durations.shape[-1]
    n_samples = durations.shape[1]
    assert len(durations) == len(pitches)
    for ss in range(n_samples):
        pm_obj = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a cello instrument
        def mkpm(name):
            return pretty_midi.instrument_name_to_program(name)

        def mki(p):
            return pretty_midi.Instrument(program=p)

        pm_programs = [mkpm(n) for n in voice_mappings]
        pm_instruments = [mki(p) for p in pm_programs]
        time_offset = np.zeros((order,))
        for ii in range(len_durations):
            for jj in range(order):
                pitches_isj = pitches[ii, ss, jj]
                durations_isj = durations[ii, ss, jj]
                p = int(pitches_isj)
                d = durations_isj
                if d == -1:
                    continue
                if p == -1:
                    continue
                s = time_offset[jj]
                e = time_offset[jj] + d
                time_offset[jj] += d
                note = pretty_midi.Note(velocity=voice_velocity[jj],
                                        pitch=p + voice_offset[jj],
                                        start=s, end=e)
                # Add it to our cello instrument
                pm_instruments[jj].notes.append(note)
        # Add the cello instrument to the PrettyMIDI object
        for pm_instrument in pm_instruments:
            pm_obj.instruments.append(pm_instrument)
        # Write out the MIDI data
        pm_obj.write('sample_{}.mid'.format(ss + add_to_name))

parser = argparse.ArgumentParser(description="Sample audio from saved model")
args = parser.parse_args()

bach = fetch_bach_chorales_music21()

# 4 pitches 4 durations
order = 4
n_in = 2 * order
n_pitch_emb = 20
n_dur_emb = 4
n_hid = 64
minibatch_size = 2
n_reps = 10
sm = lambda x: np_softmax_activation(x, temperature)
max_step = 70
max_note = order
prime_step = 10
temperature = .1
deterministic = False

n_pitches = len(bach["pitch_list"])
n_durations = len(bach["duration_list"])

random_state = np.random.RandomState(1999)

lp = bach["list_of_data_pitch"]
ld = bach["list_of_data_duration"]

checkpoint_dict = fetch_checkpoint_dict(["rnn_midi_masked_lm"])
predict_function = checkpoint_dict["predict_function"]

train_itr = list_of_array_iterator([lp, ld], minibatch_size, stop_index=.9,
                                   randomize=True, random_state=random_state)
valid_itr = list_of_array_iterator([lp, ld], minibatch_size, start_index=.9,
                                   randomize=True, random_state=random_state)

for i in range(n_reps):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(train_itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)
    h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
    h0_i = h0_init

    mb_o = copy.deepcopy(mb)
    mb = np.zeros((max_step, mb.shape[1], mb.shape[2])).astype("float32")
    mb[0, :, :] = mb_o[0, :, :]
    mask = mb[:, :, 0] * 0. + 1.

    for n_t in range(1, max_step - 1):
        print("Sampling timestep %i" % n_t)
        for n_n in range(max_note):
            r = predict_function(mb[n_t - 1:n_t + 1], mask[n_t - 1:n_t + 1], h0_i)
            pitch_lins = r[:4]
            dur_lins = r[4:8]
            pitch_preds = [sm(pl) for pl in pitch_lins]
            dur_preds = [sm(dl) for dl in dur_lins]

            # deterministic
            if deterministic:
                pitch_pred = pitch_preds[n_n].argmax(axis=-1)[0]
                dur_pred = dur_preds[n_n].argmax(axis=-1)[0]
            else:
                shp = pitch_preds[n_n].shape
                pitch_pred = pitch_preds[n_n].reshape((-1, shp[-1]))
                shp = dur_preds[n_n].shape
                dur_pred = dur_preds[n_n].reshape((-1, shp[-1]))

                def rn(pp, eps=1E-6):
                    return pp / (pp.sum() + eps)
                s_p = [random_state.multinomial(1, rn(pitch_pred[m_m])).argmax()
                       for m_m in range(pitch_pred.shape[0])]
                s_d = [random_state.multinomial(1, rn(dur_pred[m_m])).argmax()
                       for m_m in range(dur_pred.shape[0])]

                s_p = np.array(s_p).reshape((shp[0], shp[1]))
                s_d = np.array(s_d).reshape((shp[0], shp[1]))

                pitch_pred = s_p
                dur_pred = s_d
            if n_t > prime_step:
                mb[n_t, :, n_n] = pitch_pred
                mb[n_t, :, n_n + max_note] = dur_pred
            else:
                mb[n_t, :, n_n] = mb_o[n_t, :, n_n]
                mb[n_t, :, n_n + max_note] = mb_o[n_t, :, n_n + max_note]
        r = predict_function(mb[n_t - 1:n_t + 1], mask[n_t - 1:n_t + 1], h0_i)
        pitch_lins = r[:4]
        dur_lins = r[4:8]
        h0 = r[-1]
        h0_i = h0[-1]

    pitch_where = []
    duration_where = []
    pl = bach['pitch_list']
    dl = bach['duration_list']
    pitch_mb = mb[:, :, :order]
    duration_mb = mb[:, :, order:]
    for n, pli in enumerate(pl):
        pitch_where.append(np.where(pitch_mb == n))

    for n, dli in enumerate(dl):
        duration_where.append(np.where(duration_mb == n))

    for n, pw in enumerate(pitch_where):
        pitch_mb[pw] = pl[n]

    for n, dw in enumerate(duration_where):
        duration_mb[dw] = dl[n]

    if not os.path.exists("masked_samples"):
        os.mkdir("masked_samples")
    os.chdir("masked_samples")
    duration_and_pitch_to_pretty_midi(duration_mb, pitch_mb, add_to_name=i * mb.shape[1])
    os.chdir("..")
