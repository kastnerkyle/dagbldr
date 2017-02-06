#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe
from extras import jose_masked_synthesis_sequence_iterator
import os

import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import linear
from dagbldr.nodes import gru
from dagbldr.nodes import gru_fork
from dagbldr.nodes import gaussian_attention
from dagbldr.nodes import masked_cost

from dagbldr import get_params, get_logger
from dagbldr.utils import create_checkpoint_dict
from dagbldr.utils import numpy_one_hot

from dagbldr.optimizers import adam
from dagbldr.optimizers import gradient_norm_rescaling
from dagbldr.training import TrainingLoop

filedir = "/Tmp/kastner/romanian_multi_speakers/norm_info/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
    #nfsdir = "/data/lisatmp4/kastner/vctk_American_speakers/norm_info/"
    sdir = "marge:" + filedir
    cmd = "rsync -avhp %s %s" % (sdir, fd)
    pe(cmd, shell=True)

filedir = "/Tmp/kastner/romanian_multi_speakers/numpy_features/"
#if not os.path.exists(filedir):
if filedir[-1] != "/":
    fd = filedir + "/"
else:
    fd = filedir
if not os.path.exists(fd):
    os.makedirs(fd)
sdir = "marge:" + filedir
cmd = "rsync -avhp %s %s" % (sdir, fd)
pe(cmd, shell=True)

files = [filedir + fs for fs in os.listdir(filedir)]
final_files = []
final_ids = []
for f in files:
    ext = f.split("/")[-1]
    if "train" == ext[:5]:
        final_ids.append(0)
        final_files.append(f)
    elif "ele" == ext[:3]:
        final_ids.append(1)
        final_files.append(f)
    elif "geo" == ext[:3]:
        final_ids.append(2)
        final_files.append(f)

files = final_files
file_ids = final_ids
assert len(files) == len(file_ids)

minibatch_size = 8
n_hid = 1024
random_state = np.random.RandomState(1999)


train_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="romanian_chars",
                                               extra_ids=file_ids,
                                               stop_index=.9,
                                               randomize=True,
                                               random_state=random_state)

valid_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="romanian_chars",
                                               extra_ids=file_ids,
                                               start_index=.9,
                                               randomize=True,
                                               random_state=random_state)

"""
x_tot = 0
y_tot = 0
try:
    while True:
        X_mb, y_mb, X_mb_mask, y_mb_mask = next(train_itr)
        for i in range(X_mb.shape[1]):
            x = X_mb_mask[:, i]
            y = y_mb_mask[:, i]
            x = x[x != 0]
            y = y[y != 0]
            x_tot += len(x)
            y_tot += len(y)
except:
    ratio = float(x_tot) / float(y_tot)
    from IPython import embed; embed()
    raise ValueError()
"""

X_mb, y_mb, X_mb_mask, y_mb_mask, id_mb = next(train_itr)
train_itr.reset()

y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))

n_ids = len(list(set(file_ids)))

id_mb = numpy_one_hot(id_mb, n_ids).astype("float32")

n_text_ins = X_mb.shape[-1]
n_audio_ins = y_mb.shape[-1]
n_audio_outs = y_mb.shape[-1]
n_ctx_ins = 2 * n_hid


att_dim = 20
emb_dim = 128
train_noise_pwr = 4.
valid_noise_pwr = 0.


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

train_enc_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
train_enc_h1_r_init = np.zeros((minibatch_size, n_hid)).astype("float32")

train_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
train_h2_init = np.zeros((minibatch_size, n_hid)).astype("float32")
train_h3_init = np.zeros((minibatch_size, n_hid)).astype("float32")

valid_enc_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
valid_enc_h1_r_init = np.zeros((minibatch_size, n_hid)).astype("float32")

valid_h1_init = np.zeros((minibatch_size, n_hid)).astype("float32")
valid_h2_init = np.zeros((minibatch_size, n_hid)).astype("float32")
valid_h3_init = np.zeros((minibatch_size, n_hid)).astype("float32")

train_w1_init = np.zeros((minibatch_size, n_ctx_ins)).astype("float32")
valid_w1_init = np.zeros((minibatch_size, n_ctx_ins)).astype("float32")

train_k1_init = np.zeros((minibatch_size, att_dim)).astype("float32")
valid_k1_init = np.zeros((minibatch_size, att_dim)).astype("float32")

X_sym = tensor.tensor3()
y_sym = tensor.tensor3()
X_mask_sym = tensor.fmatrix()
y_mask_sym = tensor.fmatrix()

id_sym = tensor.fmatrix()

noise_pwr = tensor.fscalar()
noise_pwr.tag.test_value = train_noise_pwr

enc_h1_0 = tensor.fmatrix()
enc_h1_r_0 = tensor.fmatrix()
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
id_sym.tag.test_value = id_mb

enc_h1_0.tag.test_value = train_enc_h1_init
enc_h1_r_0.tag.test_value = train_enc_h1_r_init
h1_0.tag.test_value = train_h1_init
h2_0.tag.test_value = train_h2_init
h3_0.tag.test_value = train_h3_init

y_tm1_sym = y_sym[:-1]
y_tm1_mask_sym = y_mask_sym[:-1]

srng = theano.tensor.shared_randomstreams.RandomStreams(0)
noise = srng.normal(y_tm1_sym.shape)
y_tm1_sym = y_tm1_sym + noise_pwr * noise
#y_tm1_sym = y_tm1_sym + noise_pwr * y_tm1_sym

y_t_sym = y_sym[1:]
y_t_mask_sym = y_mask_sym[1:]


init = "normal"

def encoder_step(in_t, mask_t, in_r_t, mask_r_t, h1_tm1, h1_r_tm1):
    enc_h1_fork = gru_fork([in_t], [n_text_ins], n_hid,
                           name="enc_h1_fork",
                           random_state=random_state, init_func=init)
    enc_h1_t = gru(enc_h1_fork, h1_tm1, [n_hid], n_hid, mask=mask_t, name="enc_h1",
                   random_state=random_state, init_func=init)

    enc_h1_r_fork = gru_fork([in_r_t], [n_text_ins], n_hid,
                       name="enc_h1_r_fork",
                       random_state=random_state, init_func=init)
    enc_h1_r_t = gru(enc_h1_r_fork, h1_r_tm1, [n_hid], n_hid, mask=mask_r_t, name="enc_h1_r",
               random_state=random_state, init_func=init)
    return enc_h1_t, enc_h1_r_t


(enc_h1, enc_h1_r), _ = theano.scan(encoder_step,
                          sequences=[X_sym, X_mask_sym, X_sym[::-1], X_mask_sym[::-1]],
                          outputs_info=[enc_h1_0, enc_h1_r_0])

enc_ctx = enc_h1 + enc_h1_r
enc_ctx = tensor.concatenate((enc_h1, enc_h1_r[::-1]), axis=2)

'''
# unidirectional
def encoder_step(in_t, mask_t, h1_tm1):
    enc_h1_fork = gru_fork([in_t], [n_text_ins], n_hid,
                           name="enc_h1_fork",
                           random_state=random_state, init_func=init)
    enc_h1_t = gru(enc_h1_fork, h1_tm1, [n_hid], n_hid, mask=mask_t, name="enc_h1",
                   random_state=random_state, init_func=init)
    return enc_h1_t


enc_h1, _ = theano.scan(encoder_step,
                          sequences=[X_sym, X_mask_sym],
                          outputs_info=[enc_h1_0])

enc_ctx = enc_h1 #+ enc_h1_r #tensor.concatenate((enc_h1, enc_h1_r[::-1]), axis=2)
'''

id_emb = linear([id_sym], [n_ids], emb_dim, name="id_emb",
                random_state=random_state)

average_step = 0.0809
#min_step = .9 * average_step
#max_step = 1.25 * average_step
def step(in_t, mask_t, h1_tm1, h2_tm1, h3_tm1, k_tm1, w_tm1,
         ctx, ctx_mask, id_c):
    h1_t, k1_t, w1_t = gaussian_attention([in_t, id_c], [n_audio_ins, emb_dim],
                                          h1_tm1, k_tm1, w_tm1,
                                          ctx, n_ctx_ins, n_hid,
                                          att_dim=att_dim,
                                          average_step=average_step,
                                          #min_step=min_step,
                                          #max_step=max_step,
                                          cell_type="gru",
                                          conditioning_mask=ctx_mask,
                                          step_mask=mask_t, name="rec_gauss_att",
                                          random_state=random_state)
    h2_fork = gru_fork([in_t, h1_t, w1_t, h3_tm1, id_c], [n_audio_ins, n_hid, n_ctx_ins, n_hid, emb_dim], n_hid,
                       name="h2_fork",
                       random_state=random_state, init_func=init)
    h2_t = gru(h2_fork, h2_tm1, [n_hid], n_hid, mask=mask_t, name="rec_l2",
               random_state=random_state, init_func=init)

    h3_fork = gru_fork([in_t, h1_t, w1_t, h2_t, id_c], [n_audio_ins, n_hid, n_ctx_ins, n_hid, emb_dim], n_hid, name="h3_fork",
                       random_state=random_state, init_func=init)
    h3_t = gru(h3_fork, h3_tm1, [n_hid], n_hid, mask=mask_t, name="rec_l3",
               random_state=random_state, init_func=init)
    return h1_t, h2_t, h3_t, k1_t, w1_t

(h1, h2, h3, k, w), _ = theano.scan(step,
                                    sequences=[y_tm1_sym, y_tm1_mask_sym],
                                    outputs_info=[h1_0, h2_0, h3_0, k1_0, w1_0],
                                    non_sequences=[enc_ctx, X_mask_sym, id_emb])
comb = h1 + h2 + h3
y_pred = linear([comb], [n_hid],
                n_audio_outs, name="out_l",
                random_state=random_state, init_func=init)

loss = masked_cost(((y_pred - y_t_sym) ** 2), y_t_mask_sym)
cost = loss.sum() / (y_mask_sym.sum() + 1E-5)

params = list(get_params().values())
grads = tensor.grad(cost, params)
grads = gradient_norm_rescaling(grads, 10.)

learning_rate = 0.0001
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)


in_args = [X_sym, y_sym, X_mask_sym, y_mask_sym, id_sym]
in_args += [enc_h1_0, enc_h1_r_0, h1_0, h2_0, h3_0, k1_0, w1_0, noise_pwr]
out_args = [cost, enc_h1, enc_h1_r, h1, h2, h3]
pred_out_args= [y_pred, enc_h1, enc_h1_r, h1, h2, h3, k, w]


fit_function = theano.function(in_args, out_args, updates=updates)
cost_function = theano.function(in_args, out_args)
predict_function = theano.function(in_args, pred_out_args)

n_epochs = 120

def train_loop(itr):
    X_mb, y_mb, X_mask, y_mask, id_mb = next(itr)
    y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
    y_mask = np.concatenate((1. * y_mask[0, :][None], y_mask))
    id_mb = numpy_one_hot(id_mb, n_ids).astype("float32")
    cost, enc_h1, enc_h1_r, h1, h2, h3 = fit_function(X_mb, y_mb, X_mask, y_mask, id_mb,
                                            train_enc_h1_init, train_enc_h1_r_init,
                                            train_h1_init, train_h2_init, train_h3_init,
                                            train_k1_init, train_w1_init, train_noise_pwr)
    return [cost]


def valid_loop(itr):
    X_mb, y_mb, X_mask, y_mask, id_mb = next(itr)
    y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
    y_mask = np.concatenate((1. * y_mask[0, :][None], y_mask))
    id_mb = numpy_one_hot(id_mb, n_ids).astype("float32")
    cost, enc_h1, enc_h1_r, h1, h2, h3 = cost_function(X_mb, y_mb, X_mask, y_mask, id_mb,
                                             valid_enc_h1_init, valid_enc_h1_r_init,
                                             valid_h1_init, valid_h2_init, valid_h3_init,
                                             valid_k1_init, valid_w1_init, valid_noise_pwr)
    return [cost]

checkpoint_dict = create_checkpoint_dict(locals())

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=n_epochs,
                  checkpoint_every_n_epochs=1,
                  checkpoint_every_n_seconds=5 * 60 * 60,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True)
epoch_results = TL.run()
