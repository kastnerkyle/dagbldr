#!/usr/bin/env python
from extras import synthesis_sequence_iterator, pe
import os

import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import linear
from dagbldr.nodes import gru
from dagbldr.nodes import gru_fork
from dagbldr.nodes import slice_state

from dagbldr import get_params
from dagbldr.utils import create_checkpoint_dict

from dagbldr.optimizers import adam
from dagbldr.training import TrainingLoop


filedir = "/Tmp/kastner/vctk_American_speakers/numpy_features/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
    nfsdir = "/data/lisatmp4/kastner/vctk_American_speakers/numpy_features/"
    cmd = "rsync -avhp %s %s" % (nfsdir, fd)
    pe(cmd, shell=True)

files = [filedir + fs for fs in os.listdir(filedir)]
truncation = 500
minibatch_size = 128
n_hid = 1024
train_itr = synthesis_sequence_iterator(files, minibatch_size, truncation,
                                        stop_index=.9)

valid_itr = synthesis_sequence_iterator(files, minibatch_size, truncation,
                                        start_index=.9)
X_mb, y_mb = next(train_itr)
train_itr.reset()

from extras import generate_merlin_wav

"""
import numpy as np
a = np.load("/Tmp/kastner/vctk_American_speakers/numpy_features/p294_010.npz")
generate_merlin_wav(a["audio_features"], do_post_filtering=False)
raise ValueError()

y_itf = train_itr.inverse_transform(y_mb)
generate_merlin_wav(y_itf[:, 0, :], do_post_filtering=False)
raise ValueError()
"""
raise ValueError()

n_ins = X_mb.shape[-1]
n_outs = y_mb.shape[-1]

train_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
train_h2_init = np.zeros((minibatch_size, n_hid)).astype("float32")
train_h3_init = np.zeros((minibatch_size, n_hid)).astype("float32")

valid_h1_init = np.zeros((minibatch_size, n_hid)) .astype("float32")
valid_h2_init = np.zeros((minibatch_size, n_hid)) .astype("float32")
valid_h3_init = np.zeros((minibatch_size, n_hid)) .astype("float32")

X_sym = tensor.tensor3()
y_sym = tensor.tensor3()
h1_0 = tensor.fmatrix()
h2_0 = tensor.fmatrix()
h3_0 = tensor.fmatrix()

X_sym.tag.test_value = X_mb
y_sym.tag.test_value = y_mb

h1_0.tag.test_value = train_h1_init
h2_0.tag.test_value = train_h2_init
h3_0.tag.test_value = train_h3_init

random_state = np.random.RandomState(1999)

l1 = linear([X_sym], [n_ins], n_hid, name="linear_l",
            random_state=random_state)


def step(in_t, h1_tm1, h2_tm1, h3_tm1):
    h1_fork = gru_fork([in_t, h3_tm1], [n_hid, n_hid], n_hid, name="h1_fork",
                       random_state=random_state)
    h1_t = gru(h1_fork, h1_tm1, [n_hid], n_hid, name="rec_l1",
               random_state=random_state)

    h2_fork = gru_fork([in_t, h1_t], [n_hid, n_hid], n_hid, name="h2_fork",
                       random_state=random_state)
    h2_t = gru(h2_fork, h2_tm1, [n_hid], n_hid, name="rec_l2",
               random_state=random_state)

    h3_fork = gru_fork([in_t, h2_t], [n_hid, n_hid], n_hid, name="h3_fork",
                       random_state=random_state)
    h3_t = gru(h3_fork, h3_tm1, [n_hid], n_hid, name="rec_l3",
               random_state=random_state)
    return h1_t, h2_t, h3_t

(h1, h2, h3), _ = theano.scan(step,
                              sequences=[l1],
                              outputs_info=[h1_0, h2_0, h3_0])

y_pred = linear([l1, h1, h2, h3], [n_hid, n_hid, n_hid, n_hid],
                n_outs, name="out_l",
                random_state=random_state)
loss = ((y_pred - y_sym) ** 2)
cost = loss.mean()

params = list(get_params().values())
grads = tensor.grad(cost, params)

learning_rate = 0.002
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym, h1_0, h2_0, h3_0],
                               [cost, h1, h2, h3], updates=updates)
cost_function = theano.function([X_sym, y_sym, h1_0, h2_0, h3_0],
                                [cost, h1, h2, h3])
predict_function = theano.function([X_sym, h1_0, h2_0, h3_0],
                                   [y_pred, h1, h2, h3])


def train_loop(itr):
    X_mb, y_mb = next(itr)
    cost, h1, h2, h3 = fit_function(X_mb, y_mb,
                                    train_h1_init, train_h2_init, train_h3_init)
    train_h1_init[:] = h1[-1, :]
    train_h2_init[:] = h2[-1, :]
    train_h3_init[:] = h3[-1, :]
    return [cost]


def valid_loop(itr):
    X_mb, y_mb = next(itr)
    cost, h1, h2, h3 = cost_function(X_mb, y_mb,
                                     valid_h1_init, valid_h2_init, valid_h3_init)
    valid_h1_init[:] = h1[-1, :]
    valid_h2_init[:] = h2[-1, :]
    valid_h3_init[:] = h3[-1, :]
    return [cost]

checkpoint_dict = create_checkpoint_dict(locals())

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=200,
                  checkpoint_every_n_epochs=1,
                  checkpoint_every_n_seconds=15 * 60,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True)
epoch_results = TL.run()
