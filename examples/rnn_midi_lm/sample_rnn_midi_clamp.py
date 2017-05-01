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
n_reps = 10
max_step = 70
max_len = 150
max_note = order
prime_step = 25
temperature = .00
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


# trying per-piece note clamping
def make_oh(arr, oh_size):
    oh_arr = np.zeros((oh_size,)).astype(np.float32)
    for ai in arr:
        oh_arr[int(ai)] = 1.
    return oh_arr


def make_mask_lookups(lp, ld):
    lookups_pitch = {}
    lookups_duration = {}
    assert len(lp) == len(ld)
    for i in range(len(lp)):
        # they might overwrite but it doesn't matter
        lpi = lp[i]
        ldi = ld[i]
        lpiq = np.unique(lpi)
        ldiq = np.unique(ldi)
        lpioh = make_oh(lpiq, n_pitches)
        ldioh = make_oh(ldiq, n_durations)
        lpik = tuple(lpiq)
        ldik = tuple(ldiq)
        lookups_pitch[lpik] = (lpiq, lpioh)
        lookups_duration[ldik] = (ldiq, ldioh)
    return lookups_pitch, lookups_duration


def mask_lookup(mb, lookup):
    all_att = []
    for i in range(mb.shape[1]):
        uq = np.unique(mb[:, i])
        try:
            att = lookup[tuple(uq)]
        except KeyError:
            # 0 masking
            att = lookup[tuple(uq[1:])]
        all_att.append(att)
    return all_att

lookups_pitch, lookups_duration = make_mask_lookups(lp, ld)

checkpoint_dict = fetch_checkpoint_dict(["rnn_midi_clamp"])
predict_function = checkpoint_dict["predict_function"]

train_itr = list_of_array_iterator([lp, ld], minibatch_size, stop_index=.9,
                                   randomize=True, random_state=random_state)
valid_itr = list_of_array_iterator([lp, ld], minibatch_size, start_index=.9,
                                   randomize=True, random_state=random_state)
use_itr = valid_itr

if not os.path.exists("samples"):
    os.mkdir("samples")

dump_midi_player_template("samples")

for i in range(n_reps):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(use_itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)

    pitch_att = mask_lookup(pitch_mb, lookups_pitch)
    duration_att = mask_lookup(dur_mb, lookups_duration)
    mask_pitch_mbs = np.concatenate([pa[1][None] for pa in pitch_att], axis=0)
    mask_duration_mbs = np.concatenate([da[1][None] for da in duration_att], axis=0)
    extra_mask_mbs = np.concatenate((mask_pitch_mbs, mask_duration_mbs), axis=-1)

    h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
    h0_i = h0_init

    mb_o = copy.deepcopy(mb)
    mb = np.zeros((max_step, mb.shape[1], mb.shape[2])).astype("float32")
    mb[0, :, :] = mb_o[0, :, :]
    mask = mb[:, :, 0] * 0. + 1.

    for n_t in range(1, max_step - 1):
        print("Sampling timestep %i" % n_t)
        for n_n in range(max_note):
            r = predict_function(mb[n_t - 1:n_t + 1], mask[n_t - 1:n_t + 1], h0_i, extra_mask_mbs)
            pitch_lins = r[:4]
            dur_lins = r[4:8]

            pitch_preds = [sm(pl) * mask_pitch_mbs for pl in pitch_lins]
            dur_preds = [sm(dl) * mask_duration_mbs for dl in dur_lins]

            # deterministic
            if deterministic:
                pitch_pred = pitch_preds[n_n].argmax(axis=-1)[0]
                dur_pred = dur_preds[n_n].argmax(axis=-1)[0]
            else:
                shp = pitch_preds[n_n].shape
                pitch_pred = pitch_preds[n_n].reshape((-1, shp[-1]))
                shp = dur_preds[n_n].shape
                dur_pred = dur_preds[n_n].reshape((-1, shp[-1]))

                # todo, fix sampler to not put mass on highest
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
        r = predict_function(mb[n_t - 1:n_t + 1], mask[n_t - 1:n_t + 1], h0_i, extra_mask_mbs)
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
