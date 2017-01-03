#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe
from extras import jose_masked_synthesis_sequence_iterator
import os

import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import linear, embed
from dagbldr.nodes import gru
from dagbldr.nodes import gru_fork
from dagbldr.nodes import gaussian_attention
from dagbldr.nodes import masked_cost

from dagbldr import get_params
from dagbldr.utils import create_checkpoint_dict

from dagbldr.optimizers import adam
from dagbldr.optimizers import gradient_norm_rescaling
from dagbldr.training import TrainingLoop

'''
filedir = "/Tmp/kastner/vctk_American_speakers/norm_info/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
    nfsdir = "/data/lisatmp4/kastner/vctk_American_speakers/norm_info/"
    cmd = "rsync -avhp %s %s" % (nfsdir, fd)
    pe(cmd, shell=True)

filedir = "/Tmp/kastner/vctk_American_speakers/numpy_features/"
#if not os.path.exists(filedir):
if filedir[-1] != "/":
    fd = filedir + "/"
else:
    fd = filedir
if not os.path.exists(fd):
    os.makedirs(fd)
nfsdir = "/data/lisatmp4/kastner/vctk_American_speakers/numpy_features/"
cmd = "rsync -avhp %s %s" % (nfsdir, fd)
pe(cmd, shell=True)

files = [filedir + fs for fs in os.listdir(filedir)]
minibatch_size = 8
n_hid = 1024
random_state = np.random.RandomState(1999)

train_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_phonemes",
                                               stop_index=.9,
                                               randomize=True,
                                               random_state=random_state)

valid_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_phonemes",
                                               start_index=.9,
                                               randomize=True,
                                               random_state=random_state)
'''

filedir = "/Tmp/kastner/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
filep = filedir + "vctk.hdf5"
nfsp = "/data/lisatmp4/kastner/vctk_American_speakers/vctk.hdf5"
cmd = "rsync -avhp %s %s" % (nfsp, filep)
pe(cmd, shell=True)

random_state = np.random.RandomState(1999)
minibatch_size = 8
n_hid = 1024
train_itr = jose_masked_synthesis_sequence_iterator("/Tmp/kastner/vctk.hdf5",
                                                    minibatch_size=minibatch_size,
                                                    stop_index=.9)
valid_itr = jose_masked_synthesis_sequence_iterator("/Tmp/kastner/vctk.hdf5",
                                                    minibatch_size=minibatch_size,
                                                    start_index=.9)
X_mb, y_mb, X_mb_mask, y_mb_mask = next(valid_itr)

y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))
train_itr.reset()

n_text_ins = X_mb.shape[-1]
n_audio_ins = y_mb.shape[-1]
n_audio_outs = y_mb.shape[-1]
att_dim = 20
train_noise_pwr = 4.
valid_noise_pwr = train_noise_pwr

"""
from extras import generate_merlin_wav

import numpy as np
a = np.load("/Tmp/kastner/vctk_American_speakers/numpy_features/p294_010.npz")
generate_merlin_wav(a["audio_features"], do_post_filtering=False)
raise ValueError()

y_itf = train_itr.inverse_transform(y_mb)
generate_merlin_wav(y_itf[:, 0, :], do_post_filtering=False)
raise ValueError()
"""

train_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
train_h2_init = np.zeros((minibatch_size, n_hid)).astype("float32")
train_h3_init = np.zeros((minibatch_size, n_hid)).astype("float32")

valid_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
valid_h2_init = np.zeros((minibatch_size, n_hid)).astype("float32")
valid_h3_init = np.zeros((minibatch_size, n_hid)).astype("float32")

train_w1_init = np.zeros((minibatch_size, n_text_ins)).astype("float32")
valid_w1_init = np.zeros((minibatch_size, n_text_ins)).astype("float32")

train_k1_init = np.zeros((minibatch_size, att_dim)).astype("float32")
valid_k1_init = np.zeros((minibatch_size, att_dim)).astype("float32")

X_sym = tensor.tensor3()
y_sym = tensor.tensor3()
X_mask_sym = tensor.fmatrix()
y_mask_sym = tensor.fmatrix()

noise_pwr = tensor.fscalar()
noise_pwr.tag.test_value = 1.

h1_0 = tensor.fmatrix()
h2_0 = tensor.fmatrix()
h3_0 = tensor.fmatrix()

w1_0 = tensor.fmatrix()
w1_0.tag.test_value = train_w1_init

k1_0 = tensor.fmatrix()
k1_0.tag.test_value = train_k1_init

X_sym.tag.test_value = X_mb
y_sym.tag.test_value = y_mb
X_mask_sym.tag.test_value = X_mb_mask
y_mask_sym.tag.test_value = y_mb_mask

h1_0.tag.test_value = train_h1_init
h2_0.tag.test_value = train_h2_init
h3_0.tag.test_value = train_h3_init

y_tm1_sym = y_sym[:-1]
y_tm1_mask_sym = y_mask_sym[:-1]

# how to do noise?
srng = theano.tensor.shared_randomstreams.RandomStreams(0)
noise = srng.normal(y_tm1_sym.shape)
y_tm1_sym = y_tm1_sym + noise_pwr * noise

y_t_sym = y_sym[1:]
y_t_mask_sym = y_mask_sym[1:]


init = "normal"

def step(in_t, mask_t, h1_tm1, h2_tm1, h3_tm1, k_tm1, w_tm1,
         ctx, ctx_mask):
    h1_t, k1_t, w1_t = gaussian_attention([in_t], [n_audio_ins],
                                          h1_tm1, k_tm1, w_tm1,
                                          ctx, n_text_ins, n_hid,
                                          att_dim=att_dim,
                                          average_step=0.05,
                                          cell_type="gru",
                                          conditioning_mask=ctx_mask,
                                          step_mask=mask_t, name="rec_gauss_att",
                                          random_state=random_state)
    h2_fork = gru_fork([in_t, h1_t, w1_t, h3_tm1], [n_audio_ins, n_hid, n_text_ins, n_hid], n_hid,
                       name="h2_fork",
                       random_state=random_state, init_func=init)
    h2_t = gru(h2_fork, h2_tm1, [n_hid], n_hid, mask=mask_t, name="rec_l2",
               random_state=random_state, init_func=init)

    h3_fork = gru_fork([in_t, h1_t, w1_t, h2_t], [n_audio_ins, n_hid, n_text_ins, n_hid], n_hid, name="h3_fork",
                       random_state=random_state, init_func=init)
    h3_t = gru(h3_fork, h3_tm1, [n_hid], n_hid, mask=mask_t, name="rec_l3",
               random_state=random_state, init_func=init)
    return h1_t, h2_t, h3_t, k1_t, w1_t

(h1, h2, h3, k, w), _ = theano.scan(step,
                                    sequences=[y_tm1_sym, y_tm1_mask_sym],
                                    outputs_info=[h1_0, h2_0, h3_0, k1_0, w1_0],
                                    non_sequences=[X_sym, X_mask_sym])
comb = h1 + h2 + h3
y_pred = linear([comb], [n_hid],
                n_audio_outs, name="out_l",
                random_state=random_state, init_func=init)

loss = masked_cost(((y_pred - y_t_sym) ** 2), y_t_mask_sym)
cost = loss.sum() / (y_mask_sym.sum() + 1E-5)

params = list(get_params().values())
grads = tensor.grad(cost, params)
grads = gradient_norm_rescaling(grads)

learning_rate = 0.0002
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)


fit_function = theano.function([X_sym, y_sym, X_mask_sym, y_mask_sym,
                                h1_0, h2_0, h3_0, k1_0, w1_0, noise_pwr],
                               [cost, h1, h2, h3], updates=updates)
cost_function = theano.function([X_sym, y_sym, X_mask_sym, y_mask_sym,
                                 h1_0, h2_0, h3_0, k1_0, w1_0, noise_pwr],
                                [cost, h1, h2, h3])
predict_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym,
                                    h1_0, h2_0, h3_0, k1_0, w1_0, noise_pwr],
                                   [y_pred, h1, h2, h3, k, w])


def train_loop(itr):
    X_mb, y_mb, X_mask, y_mask = next(itr)
    y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
    y_mask = np.concatenate((1. * y_mask[0, :][None], y_mask))
    cost, h1, h2, h3 = fit_function(X_mb, y_mb, X_mask, y_mask,
                                    train_h1_init, train_h2_init, train_h3_init,
                                    train_k1_init, train_w1_init, train_noise_pwr)
    return [cost]


def valid_loop(itr):
    X_mb, y_mb, X_mask, y_mask = next(itr)
    y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
    y_mask = np.concatenate((1. * y_mask[0, :][None], y_mask))
    cost, h1, h2, h3 = cost_function(X_mb, y_mb, X_mask, y_mask,
                                     valid_h1_init, valid_h2_init, valid_h3_init,
                                     valid_k1_init, valid_w1_init, valid_noise_pwr)
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
