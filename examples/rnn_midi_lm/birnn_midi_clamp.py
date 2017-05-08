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
from dagbldr.nodes import reverse_with_mask

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
#n_epochs = 2350
#n_epochs = 3000
#n_epochs = 300
n_epochs = 850
minibatch_size = 2
order = mu["list_of_data_pitch"][0].shape[-1]
n_in = 2 * order

n_pitches = len(mu["pitch_list"])
n_durs = len(mu["duration_list"])

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
        ldioh = make_oh(ldiq, n_durs)
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

lookups_pitch, lookups_dur = make_mask_lookups(lp, ld)

train_itr = list_of_array_iterator([lp, ld], minibatch_size, stop_index=.9,
                                   randomize=True, random_state=random_state)

valid_itr = list_of_array_iterator([lp, ld], minibatch_size, start_index=.9,
                                   randomize=True, random_state=random_state)

pitch_mb, pitch_mask, dur_mb, dur_mask = next(train_itr)
train_itr.reset()

mb = np.concatenate((pitch_mb, dur_mb), axis=-1)

pitch_att = mask_lookup(pitch_mb, lookups_pitch)
dur_att = mask_lookup(dur_mb, lookups_dur)
mask_pitch_mbs = np.concatenate([pa[1][None] for pa in pitch_att], axis=0)
mask_dur_mbs = np.concatenate([da[1][None] for da in dur_att], axis=0)
extra_mask_mbs = np.concatenate((mask_pitch_mbs, mask_dur_mbs), axis=-1)

h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
h0_r_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")

A_sym = tensor.tensor3()
A_sym.tag.test_value = mb
A_mask_sym = tensor.fmatrix()
A_mask_sym.tag.test_value = pitch_mask

A_extra_mask_sym_2 = tensor.fmatrix()
A_extra_mask_sym_2.tag.test_value = extra_mask_mbs
A_extra_mask_sym = A_extra_mask_sym_2[None, :, :]
pitch_extra_mask_sym = A_extra_mask_sym[..., :n_pitches]
dur_extra_mask_sym = A_extra_mask_sym[..., -n_durs:]

h0 = tensor.fmatrix()
h0.tag.test_value = h0_init

h0_r = tensor.fmatrix()
h0_r.tag.test_value = h0_r_init

random_state = np.random.RandomState(1999)

pitch_e = multiembed([A_sym[:, :, :order]], order, n_pitches, n_pitch_emb,
                     name="pitch_embed", random_state=random_state)
dur_e = multiembed([A_sym[:, :, order:]], order, n_durs, n_dur_emb,
                        name="dur_embed", random_state=random_state)

gap = 4
X_pitch_e_sym = pitch_e[:-2 * gap]
X_dur_e_sym = dur_e[:-2 * gap]
X_mask_sym = A_mask_sym[:-2 * gap]

X_pitch_e_next_sym = pitch_e[2 * gap:]
X_dur_e_next_sym = dur_e[2 * gap:]
X_mask_next_sym = A_mask_sym[2 * gap:]

X_fork = lstm_fork([X_pitch_e_sym, X_dur_e_sym, X_pitch_e_next_sym, X_dur_e_next_sym],
                   [order * n_pitch_emb, order * n_dur_emb, order * n_pitch_emb, order * n_dur_emb],
                   n_hid, name="fork_1", random_state=random_state)

X_pitch_e_r_sym = reverse_with_mask(X_pitch_e_sym, X_mask_sym, minibatch_size)
X_dur_e_r_sym = reverse_with_mask(X_dur_e_sym, X_mask_sym, minibatch_size)

X_pitch_e_next_r_sym = reverse_with_mask(X_pitch_e_next_sym, X_mask_next_sym, minibatch_size)
X_dur_e_next_r_sym = reverse_with_mask(X_dur_e_next_sym, X_mask_next_sym, minibatch_size)

X_r_fork = lstm_fork([X_pitch_e_r_sym, X_dur_e_r_sym, X_pitch_e_next_r_sym, X_dur_e_next_r_sym],
                     [order * n_pitch_emb, order * n_dur_emb, order * n_pitch_emb, order * n_dur_emb],
                     n_hid, name="fork_r_1", random_state=random_state)


def step(in_t, mask_t, in_r_t, mask_r_t, h_tm1, h_r_tm1):
    h_t = lstm(in_t, h_tm1, [n_hid], n_hid, name="rnn_1",
               mask=mask_t,
               random_state=random_state)
    h_r_t = lstm(in_r_t, h_r_tm1, [n_hid], n_hid, name="rnn_r_1",
                 mask=mask_r_t,
                 random_state=random_state)
    return h_t, h_r_t


(h, h_r), _ = theano.scan(step,
                          sequences=[X_fork, X_mask_sym, X_r_fork, X_mask_sym],
                          outputs_info=[h0, h0_r])

h_f_o = slice_state(h, n_hid)
h_r_o = slice_state(h_r, n_hid)

h_r_o_r = reverse_with_mask(h_r_o, X_mask_sym, minibatch_size)

h_o = tensor.concatenate((h_f_o, h_r_o_r), axis=2)

extra_mask_proj = linear([A_extra_mask_sym[0]],
                         [n_pitches + n_durs],
                         2 * n_hid,
                         name="extra_mask_proj",
                         random_state=random_state)

h_o = h_o + extra_mask_proj

# 1:-2 * gap + 1 is one step ahead with extra backwards context
# gap:-gap is gapped context, predict center
y_pitch_e_sym = pitch_e[1:-2 * gap + 1]
y_dur_e_sym = dur_e[1:-2 * gap + 1]

y_pitch_sym = A_sym[1:-2 * gap + 1, :, :order]
y_dur_sym = A_sym[1:-2 * gap + 1, :, order:]
"""
y_pitch_e_sym = pitch_e[gap:-1 * gap]
y_dur_e_sym = dur_e[gap:-1 * gap]

y_pitch_sym = A_sym[gap:-1 * gap, :, :order]
y_dur_sym = A_sym[gap:-gap, :, order:]
"""

ar_y_pitch = automask([y_pitch_e_sym], order)
ar_y_dur = automask([y_dur_e_sym], order)

pitch_lins = []
dur_lins = []
costs = []

for i in range(order):
    y_dur_lin = linear([h_o, ar_y_pitch[i], ar_y_dur[i]],
                       [2 * n_hid, order * n_pitch_emb, order * n_dur_emb],
                       n_durs,
                       name="pred_dur_%i" % i,
                       random_state=random_state)

    dur_lins.append(y_dur_lin)

    y_dur_pred = softmax_activation(y_dur_lin)
    y_dur_pred = dur_extra_mask_sym * y_dur_pred
    dur_weight = float(n_pitches) / (n_pitches + n_durs)

    dur_cost = dur_weight * categorical_crossentropy(y_dur_pred,
                                                     y_dur_sym[..., i], 1E-8)
    dur_cost = masked_cost(dur_cost, X_mask_sym).sum() / (X_mask_sym.sum() + 1.)

    y_pitch_lin = linear([h_o, ar_y_pitch[i], ar_y_dur[i]],
                         [2 * n_hid, order * n_pitch_emb, order * n_dur_emb],
                         n_pitches,
                         name="pred_pitch_%i" % i,
                         random_state=random_state)

    pitch_lins.append(y_pitch_lin)

    y_pitch_pred = softmax_activation(y_pitch_lin)
    y_pitch_pred = pitch_extra_mask_sym * y_pitch_pred
    pitch_weight = float(n_durs) / (n_pitches + n_durs)

    pitch_cost = pitch_weight * categorical_crossentropy(y_pitch_pred,
                                                         y_pitch_sym[..., i],
                                                         1E-8)
    pitch_cost = masked_cost(pitch_cost,
                             X_mask_sym).sum() / (X_mask_sym.sum() + 1.)

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

fit_function = theano.function([A_sym, A_mask_sym, h0, h0_r, A_extra_mask_sym_2],
                               [cost, h, h_r], updates=updates)
cost_function = theano.function([A_sym, A_mask_sym, h0, h0_r, A_extra_mask_sym_2],
                                [cost, h, h_r])
predict_function = theano.function([A_sym, A_mask_sym, h0, h0_r, A_extra_mask_sym_2],
                                   pitch_lins + dur_lins + [h, h_r])


def train_loop(itr, info):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)

    pitch_att = mask_lookup(pitch_mb, lookups_pitch)
    dur_att = mask_lookup(dur_mb, lookups_dur)
    mask_pitch_mbs = np.concatenate([pa[1][None] for pa in pitch_att], axis=0)
    mask_dur_mbs = np.concatenate([da[1][None] for da in dur_att],
                                       axis=0)
    extra_mask_mbs = np.concatenate((mask_pitch_mbs, mask_dur_mbs),
                                    axis=-1)
    mask = pitch_mask
    h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
    h0_r_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")

    cost, _1, _2 = fit_function(mb, mask, h0_init, h0_r_init, extra_mask_mbs)
    return [cost]


def valid_loop(itr, info):
    pitch_mb, pitch_mask, dur_mb, dur_mask = next(itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)

    pitch_att = mask_lookup(pitch_mb, lookups_pitch)
    dur_att = mask_lookup(dur_mb, lookups_dur)
    mask_pitch_mbs = np.concatenate([pa[1][None] for pa in pitch_att], axis=0)
    mask_dur_mbs = np.concatenate([da[1][None] for da in dur_att],
                                       axis=0)
    extra_mask_mbs = np.concatenate((mask_pitch_mbs, mask_dur_mbs),
                                    axis=-1)

    mask = pitch_mask
    h0_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
    h0_r_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")

    cost, _1, _2 = cost_function(mb, mask, h0_init, h0_r_init, extra_mask_mbs)
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
