#!/usr/bin/env python
from dagbldr import fetch_checkpoint_dict
from dagbldr.datasets import fetch_bach_chorales_music21
from dagbldr.datasets import fetch_symbtr_music21
from dagbldr.datasets import list_of_array_iterator
from dagbldr.datasets import pitches_and_durations_to_pretty_midi
from dagbldr.datasets import dump_midi_player_template
import numpy as np

import argparse
import cPickle as pickle
import os
from dagbldr.nodes import np_softmax_activation
import copy

# sample13 gotten from
# leto52:/Tmp/kastner/dagbldr_models/rnn_midi_masked_lm_13-49-00_2017-20-04_45c5e7/45c5e7_model_checkpoint_2350.pkl

# major model, bach
# /u/kastner/dagbldr_lookup/1d6ba2_rnn_midi_masked_lm.json (06-43-35_2017-21-04)

# minor mode, bach
# /u/kastner/dagbldr_lookup/1bd8b6_rnn_midi_masked_lm.json (06-42-55_2017-21-04)

parser = argparse.ArgumentParser(description="Sample audio from saved model")
args = parser.parse_args()

mu = fetch_bach_chorales_music21()
#mu = fetch_symbtr_music21()

order = mu["list_of_data_pitch"][0].shape[-1]
n_in = 2 * order
n_pitch_emb = 20
n_dur_emb = 4
n_hid = 64
minibatch_size = 2
n_reps = 5
max_step = 70
max_len = 150
max_note = order
prime_step = 25
temperature = .01
sm = lambda x: np_softmax_activation(x, temperature)
if temperature == 0.:
    deterministic = True
    temperature = 1.
else:
    deterministic = False

n_pitches = len(mu["pitch_list"])
n_durations = len(mu["duration_list"])

random_state = np.random.RandomState(1999)

lp = mu["list_of_data_pitch"]
ld = mu["list_of_data_duration"]

lp2 = [lpi[:max_len] for n, lpi in enumerate(lp)]
ld2 = [ldi[:max_len] for n, ldi in enumerate(ld)]

lp = lp2
ld = ld2

# key can be major minor none
key = None
if key is not None:
    lip = []
    lid = []
    for n, k in enumerate(mu["list_of_data_key"]):
        if key in k:
            lip.append(lp[n])
            lid.append(ld[n])
    lp = lip
    ld = lid

lpn = np.concatenate(lp, axis=0)
lpn = lpn - lpn[:, 0][:, None]

ldn = np.concatenate(ld, axis=0)
ldn = ldn - ldn[:, 0][:, None]

lpnu = np.vstack({tuple(row) for row in lpn})
ldnu = np.vstack({tuple(row) for row in ldn})

step_lookups_pitch = []
step_lookups_duration = []
from collections import defaultdict
for i in range(ldn.shape[-1]):
    if i == 0:
        lup = defaultdict(lambda: np.arange(2 * n_pitches, dtype="float32") - n_pitches)
        lud = defaultdict(lambda: np.arange(2 * n_durations, dtype="float32") - n_durations)
    elif i < ldn.shape[-1]:
        lup = {}
        keyset = {tuple(row) for row in lpnu[:, :i]}
        for k in keyset:
            ii = np.where(lpnu[:, :i] == k)[0]
            v = lpnu[ii, i]
            vset = np.array(sorted(list(set(v))))
            lup[k] = vset
        lud = {}
        keyset = {tuple(row) for row in ldnu[:, :i]}
        for k in keyset:
            ii = np.where(ldnu[:, :i] == k)[0]
            v = ldnu[ii, i]
            vset = np.array(sorted(list(set(v))))
            lud[k] = vset
    step_lookups_pitch.append(lup)
    step_lookups_duration.append(lud)


def make_markov_mask(mb, mb_mask, limit, step_lookups):
    pre_mb = mb.copy()
    mb = mb - mb[:, :, 0][:, :, None]
    markov_masks = []
    for ii in range(mb.shape[2]):
        markov_masks.append(np.zeros((mb.shape[0], mb.shape[1], limit), dtype="float32"))
    for j in range(mb.shape[1]):
        for i in range(mb.shape[0]):
            if mb_mask[i, j] > 0:
                for k in range(mb.shape[2]):
                    try:
                        tt = step_lookups[k][tuple(mb[i, j, :k])]
                    except:
                        # todo, fix this????
                        tt = step_lookups[0][tuple(mb[i, j, :k])]
                    subidx = tt[(pre_mb[i, j, k] + tt) >= 0] + pre_mb[i, j, k]
                    subidx = subidx[subidx < limit]
                    subidx = subidx.astype("int32")
                    tmp = markov_masks[k][i, j].copy()
                    for si in subidx:
                        tmp[si] = 1.
                    markov_masks[k][i, j, :] = tmp
            else:
                for k in range(mb.shape[2]):
                    markov_masks[k][i, j, :] *= 0.
    return markov_masks

checkpoint_dict = fetch_checkpoint_dict(["rnn_midi_markov_masked_lm"])
predict_function = checkpoint_dict["predict_function"]

train_itr = list_of_array_iterator([lp, ld], minibatch_size, stop_index=.9,
                                   randomize=True, random_state=random_state)
valid_itr = list_of_array_iterator([lp, ld], minibatch_size, start_index=.9,
                                   randomize=True, random_state=random_state)

use_itr = valid_itr
for i in range(n_reps):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(use_itr)
    pitch_where = []
    dur_where = []
    pl = mu['pitch_list']
    dl = mu['duration_list']
    for n, pli in enumerate(pl):
        pitch_where.append(np.where(pitch_mb == n))

    for n, dli in enumerate(dl):
        dur_where.append(np.where(dur_mb == n))

    for n, pw in enumerate(pitch_where):
        pitch_mb[pw] = pl[n]

    for n, dw in enumerate(dur_where):
        dur_mb[dw] = dl[n]

    # print(mu["filename_list"][:2 * minibatch_size])
    pitches_and_durations_to_pretty_midi(pitch_mb, dur_mb,
                                         save_dir="samples/samples",
                                         add_to_name=i * minibatch_size)
dump_midi_player_template("samples")
use_itr.reset()


if not os.path.exists("samples"):
    os.mkdir("samples")

dump_midi_player_template("samples")

for i in range(n_reps):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(use_itr)
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
            # needs to be 3D
            dur_mb = mb[n_t, :, -order:][None]
            pitch_mb = mb[n_t, :, :order][None]
            pitch_mb_mask = mask[n_t, :][None]
            dur_mb_mask = mask[n_t, :][None]

            cur_pitch_markov_mask = make_markov_mask(pitch_mb, pitch_mask, n_pitches, step_lookups_pitch)
            cur_dur_markov_mask = make_markov_mask(dur_mb, dur_mask, n_durations, step_lookups_duration)

            r = predict_function(mb[n_t - 1:n_t + 1], mask[n_t - 1:n_t + 1], h0_i)
            pitch_lins = r[:4]
            dur_lins = r[4:8]
            pitch_preds = [sm(pl) for pl in pitch_lins]
            dur_preds = [sm(dl) for dl in dur_lins]

            pitch_preds = [pp * ppm for pp, ppm in zip(pitch_preds, cur_pitch_markov_mask)]
            dur_preds = [dp * dpm for dp, dpm in zip(dur_preds, cur_dur_markov_mask)]

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
                    pp[pp > eps] = pp[pp > eps] - eps
                    pp[pp <= eps] = 0.
                    return pp

                s_p = []
                for m_m in range(pitch_pred.shape[0]):
                    s_pi = random_state.multinomial(1, rn(pitch_pred[m_m])).argmax()
                    s_p.append(s_pi)

                s_d = []
                for m_m in range(dur_pred.shape[0]):
                    s_di = random_state.multinomial(1, rn(dur_pred[m_m])).argmax()
                    s_d.append(s_di)

                #s_p = [random_state.multinomial(1, rn(pitch_pred[m_m])).argmax()
                #       for m_m in range(pitch_pred.shape[0])]
                #s_d = [random_state.multinomial(1, rn(dur_pred[m_m])).argmax()
                #       for m_m in range(dur_pred.shape[0])]

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
    pl = mu['pitch_list']
    dl = mu['duration_list']
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

    pitch_mb = pitch_mb[prime_step:]
    duration_mb = duration_mb[prime_step:]
    pitches_and_durations_to_pretty_midi(pitch_mb, duration_mb,
                                         save_dir="samples/samples",
                                         name_tag="masked_sample_{}.mid",
                                         voice_params="woodwinds",
                                         add_to_name=i * mb.shape[1])
