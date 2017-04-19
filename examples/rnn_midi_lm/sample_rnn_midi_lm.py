#!/usr/bin/env python
from dagbldr import fetch_checkpoint_dict
from dagbldr.datasets import fetch_bach_chorales_music21
from dagbldr.datasets import contiguous_list_of_array_iterator
import numpy as np

import argparse
import cPickle as pickle

parser = argparse.ArgumentParser(description="Sample audio from saved model")
parser.add_argument("-e", "--epoch", help="Epoch to match",
                    default=None, required=False)
args = parser.parse_args()
epoch_arg = args.epoch

bach = fetch_bach_chorales_music21()

n_timesteps = 50
minibatch_size = 5
# 4 pitches 4 durations
order = 4
n_in = 2 * order
n_pitch_emb = 20
n_dur_emb = 4
n_hid = 64
minibatch_size = 32
truncation_length = 30
n_epochs = 500

n_pitches = len(bach["pitch_list"])
n_durations = len(bach["duration_list"])

random_state = np.random.RandomState(1999)

lp = bach["list_of_data_pitch"]
ld = bach["list_of_data_duration"]

train_itr = contiguous_list_of_array_iterator([lp, ld], minibatch_size,
                                              truncation_length, stop_index=.9)
valid_itr = contiguous_list_of_array_iterator([lp, ld], minibatch_size,
                                              truncation_length, start_index=.9)
pitch_mb, dur_mb = next(train_itr)
train_itr.reset()
mb = np.concatenate((pitch_mb, dur_mb), axis=-1)

h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")

checkpoint_dict = fetch_checkpoint_dict(["rnn_midi_lm"])

predict_function = checkpoint_dict["predict_function"]

from dagbldr.nodes import np_softmax_activation
sm = np_softmax_activation
max_step = 100
max_note = 4
prime_step = 5
h0_i = h0_init

def duration_and_pitch_to_pretty_midi(durations, pitches):
    import pretty_midi
    # Create a PrettyMIDI object
    len_durations = len(durations)
    order = durations.shape[-1]
    n_samples = durations.shape[1]
    assert len(durations) == len(pitches)
    for ss in range(n_samples):
        pm_obj = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a cello instrument
        pm_program = pretty_midi.instrument_name_to_program('Cello')
        pm_instrument = pretty_midi.Instrument(program=pm_program)
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
                note = pretty_midi.Note(velocity=100,
                                        pitch=p, start=s, end=e)
                # Add it to our cello instrument
                pm_instrument.notes.append(note)
        # Add the cello instrument to the PrettyMIDI object
        pm_obj.instruments.append(pm_instrument)
        # Write out the MIDI data
        pm_obj.write('sample_{}.mid'.format(ss))

import copy
mb_o = copy.deepcopy(mb)
mb = np.zeros((max_step, mb.shape[1], mb.shape[2])).astype("float32")
mb[0, :, :] = mb_o[0, :, :]

for n_t in range(1, max_step - 1):
    print("Sampling timestep %i" % n_t)
    for n_n in range(max_note):
        r = predict_function(mb[n_t - 1:n_t + 1], h0_i)
        pitch_lins = r[:4]
        dur_lins = r[4:8]
        pitch_preds = [sm(pl) for pl in pitch_lins]
        dur_preds = [sm(dl) for dl in dur_lins]

        pitch_pred = pitch_preds[n_n].argmax(axis=-1)[0]
        dur_pred = dur_preds[n_n].argmax(axis=-1)[0]
        if n_t > prime_step:
            mb[n_t, :, n_n] = pitch_pred
            mb[n_t, :, n_n + max_note] = dur_pred
        else:
            mb[n_t, :, n_n] = mb_o[n_t, :, n_n]
            mb[n_t, :, n_n + max_note] = mb_o[n_t, :, n_n + max_note]
    r = predict_function(mb[n_t - 1:n_t + 1], h0_i)
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

duration_and_pitch_to_pretty_midi(duration_mb, pitch_mb)
