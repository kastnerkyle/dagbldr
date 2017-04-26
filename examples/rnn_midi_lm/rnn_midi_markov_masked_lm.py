#!/usr/bin/env python
import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import linear
from dagbldr.nodes import lstm_fork
from dagbldr.nodes import lstm
from dagbldr.nodes import slice_state
from dagbldr.nodes import multiembed
from dagbldr.nodes import automask
from dagbldr.nodes import softmax_activation

from dagbldr.nodes import categorical_crossentropy
from dagbldr.nodes import masked_cost

from dagbldr import get_params
from dagbldr.utils import create_checkpoint_dict

from dagbldr.optimizers import adam
from dagbldr.optimizers import gradient_clipping

from dagbldr.training import TrainingLoop
from dagbldr.datasets import list_of_array_iterator

from dagbldr.datasets import fetch_symbtr_music21
from dagbldr.datasets import fetch_bach_chorales_music21
from dagbldr.datasets import fetch_wikifonia_music21

mu = fetch_bach_chorales_music21()
#mu = fetch_symbtr_music21()
#mu = fetch_wikifonia_music21()

#n_epochs = 500
n_epochs = 2350
minibatch_size = 2
order = mu["list_of_data_pitch"][0].shape[-1]
n_in = 2 * order

n_pitches = len(mu["pitch_list"])
n_durations = len(mu["duration_list"])

random_state = np.random.RandomState(1999)
n_pitch_emb = 20
n_dur_emb = 4
n_hid = 64
max_len = 150

lp = mu["list_of_data_pitch"]
ld = mu["list_of_data_duration"]

lp2 = [lpi[:max_len] for n, lpi in enumerate(lp)]
ld2 = [ldi[:max_len] for n, ldi in enumerate(ld)]

lp = lp2
ld = ld2

"""
key = "major"
lip = []
lid = []
for n, k in enumerate(mu["list_of_data_key"]):
    if key in k:
        lip.append(lp[n])
        lid.append(ld[n])
lp = lip
ld = lid
"""

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


train_itr = list_of_array_iterator([lp, ld], minibatch_size, stop_index=.9,
                                   randomize=True, random_state=random_state)

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
                    tt = step_lookups[k][tuple(mb[i, j, :k])]
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

valid_itr = list_of_array_iterator([lp, ld], minibatch_size, start_index=.9,
                                   randomize=True, random_state=random_state)

pitch_mb, pitch_mask, dur_mb, dur_mask = next(train_itr)
m_p = make_markov_mask(pitch_mb, pitch_mask, n_pitches, step_lookups_pitch)
m_d = make_markov_mask(dur_mb, dur_mask, n_durations, step_lookups_duration)
from IPython import embed; embed(); raise ValueError()

train_itr.reset()
mb = np.concatenate((pitch_mb, dur_mb), axis=-1)
h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")

A_sym = tensor.tensor3()
A_sym.tag.test_value = mb
A_mask_sym = tensor.fmatrix()
A_mask_sym.tag.test_value = pitch_mask

h0 = tensor.fmatrix()
h0.tag.test_value = h0_init

random_state = np.random.RandomState(1999)

pitch_e = multiembed([A_sym[:, :, :order]], order, n_pitches, n_pitch_emb,
                     name="pitch_embed", random_state=random_state)
duration_e = multiembed([A_sym[:, :, order:]], order, n_durations, n_dur_emb,
                        name="duration_embed", random_state=random_state)

X_pitch_e_sym = pitch_e[:-1]
X_dur_e_sym = duration_e[:-1]

X_mask_sym = A_mask_sym[:-1]

X_fork = lstm_fork([X_pitch_e_sym, X_dur_e_sym],
                   [order * n_pitch_emb, order * n_dur_emb],
                   n_hid, name="lstm_fork_1", random_state=random_state)


def step(in_t, in_mask_t, h_tm1):
    h_t = lstm(in_t, h_tm1, [n_in], n_hid, name="lstm_1",
               mask=in_mask_t,
               random_state=random_state)
    return h_t

h, _ = theano.scan(step,
                   sequences=[X_fork, X_mask_sym],
                   outputs_info=[h0])

h_o = slice_state(h, n_hid)

y_pitch_e_sym = pitch_e[1:]
y_dur_e_sym = duration_e[1:]

ar_y_pitch = automask([y_pitch_e_sym], order)
ar_y_dur = automask([y_dur_e_sym], order)

y_pitch_sym = A_sym[1:, :, :order]
y_dur_sym = A_sym[1:, :, order:]

pitch_lins = []
dur_lins = []
costs = []

for i in range(order):
    y_dur_lin = linear([h_o, ar_y_pitch[i], ar_y_dur[i]],
                       [n_hid, order * n_pitch_emb, order * n_dur_emb],
                       n_durations,
                       name="pred_dur_%i" % i,
                       random_state=random_state)
    dur_lins.append(y_dur_lin)

    y_dur_pred = softmax_activation(y_dur_lin)
    dur_weight = float(n_pitches) / (n_pitches + n_durations)

    dur_cost = dur_weight * categorical_crossentropy(y_dur_pred, y_dur_sym[..., i])
    dur_cost = masked_cost(dur_cost, X_mask_sym).sum() / (X_mask_sym.sum() + 1.)

    y_pitch_lin = linear([h_o, ar_y_pitch[i], ar_y_dur[i]],
                         [n_hid, order * n_pitch_emb, order * n_dur_emb],
                         n_pitches,
                         name="pred_pitch_%i" % i,
                         random_state=random_state)
    pitch_lins.append(y_pitch_lin)

    y_pitch_pred = softmax_activation(y_pitch_lin)
    pitch_weight = float(n_durations) / (n_pitches + n_durations)

    pitch_cost = pitch_weight * categorical_crossentropy(y_pitch_pred,
                                                         y_pitch_sym[..., i])
    pitch_cost = masked_cost(pitch_cost, X_mask_sym).sum() / (X_mask_sym.sum() + 1.)

    costs.append(dur_cost)
    costs.append(pitch_cost)

cost = sum(costs) / float(order * 2)
params = list(get_params().values())
params = params
grads = tensor.grad(cost, params)

clip = 5.0
learning_rate = 0.0001
grads = gradient_clipping(grads, clip)
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([A_sym, A_mask_sym, h0], [cost, h],
                               updates=updates)
cost_function = theano.function([A_sym, A_mask_sym, h0], [cost, h])
predict_function = theano.function([A_sym, A_mask_sym, h0],
                                   pitch_lins + dur_lins + [h])

def train_loop(itr, info):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)
    mask = pitch_mask
    h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
    cost, _ = fit_function(mb, mask, h0_init)
    return [cost]


def valid_loop(itr, info):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)
    mask = pitch_mask
    h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
    cost, _ = cost_function(mb, mask, h0_init)
    return [cost]


checkpoint_dict = create_checkpoint_dict(locals())

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=n_epochs,
                  checkpoint_every_n_seconds=30 * 60 * 60,
                  checkpoint_every_n_epochs=n_epochs // 10,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True,
                  skip_most_recents=False)
epoch_results = TL.run()
