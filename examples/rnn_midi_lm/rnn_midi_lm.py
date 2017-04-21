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

from dagbldr import get_params
from dagbldr.utils import create_checkpoint_dict

from dagbldr.optimizers import adam
from dagbldr.optimizers import gradient_clipping

from dagbldr.training import TrainingLoop
from dagbldr.datasets import contiguous_list_of_array_iterator

from dagbldr.datasets import fetch_bach_chorales_music21
bach = fetch_bach_chorales_music21()

n_pitches = len(bach["pitch_list"])
n_durations = len(bach["duration_list"])

random_state = np.random.RandomState(1999)

# 4 pitches 4 durations
order = 4
n_in = 2 * order
n_pitch_emb = 20
n_dur_emb = 4
n_hid = 64
minibatch_size = 32
truncation_length = 30
n_epochs = 500

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

A_sym = tensor.tensor3()
A_sym.tag.test_value = mb

h0 = tensor.fmatrix()
h0.tag.test_value = h0_init

random_state = np.random.RandomState(1999)

pitch_e = multiembed([A_sym[:, :, :order]], order, n_pitches, n_pitch_emb,
                     name="pitch_embed", random_state=random_state)
duration_e = multiembed([A_sym[:, :, order:]], order, n_durations, n_dur_emb,
                        name="duration_embed", random_state=random_state)

X_pitch_e_sym = pitch_e[:-1]
X_dur_e_sym = duration_e[:-1]

X_fork = lstm_fork([X_pitch_e_sym, X_dur_e_sym],
                   [order * n_pitch_emb, order * n_dur_emb],
                   n_hid, name="lstm_fork_1", random_state=random_state)


def step(in_t, h_tm1):
    h_t = lstm(in_t, h_tm1, [n_in], n_hid, name="lstm_1",
               random_state=random_state)
    return h_t

h, _ = theano.scan(step,
                   sequences=[X_fork],
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

    dur_cost = dur_weight * categorical_crossentropy(y_dur_pred, y_dur_sym[..., i]).mean()

    y_pitch_lin = linear([h_o, ar_y_pitch[i], ar_y_dur[i]],
                         [n_hid, order * n_pitch_emb, order * n_dur_emb],
                         n_pitches,
                         name="pred_pitch_%i" % i,
                         random_state=random_state)
    pitch_lins.append(y_pitch_lin)

    y_pitch_pred = softmax_activation(y_pitch_lin)
    pitch_weight = float(n_durations) / (n_pitches + n_durations)

    pitch_cost = pitch_weight * categorical_crossentropy(y_pitch_pred,
                                                         y_pitch_sym[..., i]).mean()

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

fit_function = theano.function([A_sym, h0], [cost, h],
                               updates=updates)
cost_function = theano.function([A_sym, h0], [cost, h])
predict_function = theano.function([A_sym, h0],
                                   pitch_lins + dur_lins + [h])


def train_loop(itr, info):
    global h0_init
    pitch_mb, dur_mb = next(itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)
    cost, h0_next = fit_function(mb, h0_init)
    h0_init = h0_next[-1]
    return [cost]


def valid_loop(itr, info):
    global h0_init
    pitch_mb, dur_mb = next(itr)
    mb = np.concatenate((pitch_mb, dur_mb), axis=-1)
    cost, h0_next = cost_function(mb, h0_init)
    h0_init = h0_next[-1]
    return [cost]


checkpoint_dict = create_checkpoint_dict(locals())

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=n_epochs,
                  checkpoint_every_n_seconds=30 * 60,
                  checkpoint_every_n_epochs=50,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True,
                  skip_most_recents=False)
epoch_results = TL.run()
