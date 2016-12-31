#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe, generate_merlin_wav
import os
import numpy as np
#import theano

def rsync_sub(filedir):
    # Only works on /Tmp/kastner, for me, on lisa servers
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    if not os.path.exists(fd):
        os.makedirs(fd)
    # assumes first part is /Tmp/kastner
    sp = filedir.split("/")
    post = "/".join(sp[3:])
    nfsdir = "/data/lisatmp4/kastner/" + post #vctk_American_speakers/norm_info/"
    cmd = "rsync -avhp %s %s" % (nfsdir, fd)
    pe(cmd, shell=True)

filedir = "/Tmp/kastner/vctk_American_speakers/norm_info/"
rsync_sub(filedir)

numpy_filedir = "/Tmp/kastner/vctk_American_speakers/numpy_features/"
rsync_sub(numpy_filedir)

files = [numpy_filedir + fs for fs in os.listdir(numpy_filedir)]
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
X_mb, y_mb, X_mb_mask, y_mb_mask = next(train_itr)
y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))
train_itr.reset()

n_text_ins = X_mb.shape[-1]
n_audio_ins = y_mb.shape[-1]
n_audio_outs = y_mb.shape[-1]
att_dim = 20


import argparse
import cPickle as pickle
parser = argparse.ArgumentParser(description="Sample audio from saved model")
parser.add_argument("pkl_or_npz_file", help="Stored pkl file")
args = parser.parse_args()
filepath = args.pkl_or_npz_file
ext = filepath.split(".")[-1]

if ext == "pkl":
    print("Found checkpoint pkl file, doing sampling")
    pickle_path = filepath
    with open(pickle_path) as f:
        checkpoint_dict = pickle.load(f)

    predict_function = checkpoint_dict["predict_function"]

    # skip some of the shorter ones
    for i in range(70):
        X_mb, y_mb, X_mb_mask, y_mb_mask = next(valid_itr)
    y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
    y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))

    i_h1 = np.zeros((minibatch_size, n_hid)).astype("float32")
    i_h2 = np.zeros((minibatch_size, n_hid)).astype("float32")
    i_h3 = np.zeros((minibatch_size, n_hid)).astype("float32")

    i_w1 = np.zeros((minibatch_size, n_text_ins)).astype("float32")
    i_k1 = np.zeros((minibatch_size, att_dim)).astype("float32")

    sample_length = len(y_mb)
    res = np.zeros((sample_length, minibatch_size, y_mb.shape[-1])).astype("float32")
    w_res = np.zeros((sample_length, minibatch_size, n_text_ins)).astype("float32")
    inmask = np.ones_like(res[:, :, 0])
    noise_pwr = 0.
    for i in range(2, sample_length):
        r = predict_function(X_mb, X_mb_mask,
                             #y_mb[i - 2:i], y_mb_mask[i - 2:i],
                             res[i - 2:i], inmask[i - 2:i],
                             i_h1, i_h2, i_h3, i_k1, i_w1, noise_pwr)
        y_p = r[0]
        o_h1 = r[1]
        o_h2 = r[2]
        o_h3 = r[3]
        o_k1 = r[4]
        o_w1 = r[5]
        res[i, :, :] = y_p
        w_res[i, :, :] = o_w1
        i_h1, i_h2, i_h3 = (o_h1[-1], o_h2[-1], o_h3[-1])
        i_k1 = o_k1[-1]
        i_w1 = o_w1[-1]

    # TODO: Figure out how to avoid this :(
    out = {"audio_results": res,
           "associated_phones": X_mb,
           "attention_weights": w_res,
           "code2phone": valid_itr._code2phone}
    savename = "sampled.npz"
    np.savez_compressed(savename, **out)
    print("Sampling complete, saved to %s" % savename)
elif ext == "npz":
    print("Found file path extension npz, reconstructing")
    npz_path = filepath
    d = np.load(npz_path)
    r = d["audio_results"]
    #r = valid_itr.inverse_transform(r)
    c = d["associated_phones"]
    lu = d["code2phone"]
    att = d["attention_weights"]
    for i in range(r.shape[1]):
        c_str = "".join([lu[ci] for ci in c[:, i, :].argmax(axis=-1)])
        non_a = [n for n, ci in enumerate(c_str) if ci != "a"]
        if len(non_a) != 0:
            last_non_a = non_a[-1] + 1
            c_str = c_str[:last_non_a]
        name = "mygen_%s_%i" % (c_str, i)
        generate_merlin_wav(r[:, i, :], file_basename=name,
                            do_post_filtering=False)
    print("Reconstruction complete")
else:
    print("Unrecognized extension, exiting")

'''
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
X_mask_sym = tensor.fmatrix()
y_mask_sym = tensor.fmatrix()

h1_0 = tensor.fmatrix()
h2_0 = tensor.fmatrix()
h3_0 = tensor.fmatrix()

X_sym.tag.test_value = X_mb
y_sym.tag.test_value = y_mb
X_mask_sym.tag.test_value = X_mb_mask
y_mask_sym.tag.test_value = y_mb_mask

h1_0.tag.test_value = train_h1_init
h2_0.tag.test_value = train_h2_init
h3_0.tag.test_value = train_h3_init

random_state = np.random.RandomState(1999)

l1 = linear([X_sym], [n_ins], n_hid, name="linear_l",
            random_state=random_state)


def step(in_t, mask_t, h1_tm1, h2_tm1, h3_tm1):
    h1_fork = gru_fork([in_t, h3_tm1], [n_hid, n_hid], n_hid, name="h1_fork",
                       random_state=random_state)
    h1_t = gru(h1_fork, h1_tm1, [n_hid], n_hid, mask=mask_t, name="rec_l1",
               random_state=random_state)

    h2_fork = gru_fork([in_t, h1_t], [n_hid, n_hid], n_hid, name="h2_fork",
                       random_state=random_state)
    h2_t = gru(h2_fork, h2_tm1, [n_hid], n_hid, mask=mask_t, name="rec_l2",
               random_state=random_state)

    h3_fork = gru_fork([in_t, h2_t], [n_hid, n_hid], n_hid, name="h3_fork",
                       random_state=random_state)
    h3_t = gru(h3_fork, h3_tm1, [n_hid], n_hid, mask=mask_t, name="rec_l3",
               random_state=random_state)
    return h1_t, h2_t, h3_t

(h1, h2, h3), _ = theano.scan(step,
                              sequences=[l1, X_mask_sym],
                              outputs_info=[h1_0, h2_0, h3_0])

y_pred = linear([l1, h1, h2, h3], [n_hid, n_hid, n_hid, n_hid],
                n_outs, name="out_l",
                random_state=random_state)

loss = masked_cost(((y_pred - y_sym) ** 2), y_mask_sym)
cost = loss.sum(axis=2).mean(axis=1).mean(axis=0)

params = list(get_params().values())
grads = tensor.grad(cost, params)

learning_rate = 0.0002
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym, X_mask_sym, y_mask_sym,
                                h1_0, h2_0, h3_0],
                               [cost, h1, h2, h3], updates=updates)
cost_function = theano.function([X_sym, y_sym, X_mask_sym, y_mask_sym,
                                 h1_0, h2_0, h3_0],
                                [cost, h1, h2, h3])
predict_function = theano.function([X_sym, X_mask_sym,
                                    h1_0, h2_0, h3_0],
                                   [y_pred, h1, h2, h3])


def train_loop(itr):
    X_mb, y_mb, X_mask, y_mask = next(itr)
    cost, h1, h2, h3 = fit_function(X_mb, y_mb, X_mask, y_mask,
                                    train_h1_init, train_h2_init, train_h3_init)
    #train_h1_init[:] = h1[-1, :]
    #train_h2_init[:] = h2[-1, :]
    #train_h3_init[:] = h3[-1, :]
    return [cost]


def valid_loop(itr):
    X_mb, y_mb, X_mask, y_mask = next(itr)
    cost, h1, h2, h3 = cost_function(X_mb, y_mb, X_mask, y_mask,
                                     valid_h1_init, valid_h2_init, valid_h3_init)
    #valid_h1_init[:] = h1[-1, :]
    #valid_h2_init[:] = h2[-1, :]
    #valid_h3_init[:] = h3[-1, :]
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
'''
