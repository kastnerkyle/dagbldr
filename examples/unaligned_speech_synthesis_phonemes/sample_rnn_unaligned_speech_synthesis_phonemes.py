#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe, generate_merlin_wav
from extras import jose_masked_synthesis_sequence_iterator
import os
import numpy as np
#import theano

'''
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
'''

filedir = "/Tmp/kastner/"
nfsdir = "/data/lisatmp4/kastner/vctk_American_speakers/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
filep = filedir + "vctk.hdf5"
nfsp = nfsdir + "vctk.hdf5"
cmd = "rsync -avhp %s %s" % (nfsp, filep)
pe(cmd, shell=True)

filep = filedir + "norm_info_mgc_lf0_vuv_bap_63_MVN.dat"
nfsp = nfsdir + "norm_info/norm_info_mgc_lf0_vuv_bap_63_MVN.dat"
cmd = "rsync -avhp %s %s" % (nfsp, filep)
pe(cmd, shell=True)
norm_info_file = filep


random_state = np.random.RandomState(1999)
minibatch_size = 8
n_hid = 1024
train_itr = jose_masked_synthesis_sequence_iterator("/Tmp/kastner/vctk.hdf5",
                                                    minibatch_size=minibatch_size,
                                                    stop_index=.9)
valid_itr = jose_masked_synthesis_sequence_iterator("/Tmp/kastner/vctk.hdf5",
                                                    minibatch_size=minibatch_size,
                                                    start_index=.9)
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
    print("Finished unpickling file")

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
        print("Iteration %i" % i)
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
           "groundtruth": y_mb,
           "attention_weights": w_res}
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
    with open(norm_info_file, 'rb') as f:
        cmp_info = np.fromfile(f, dtype=np.float32)
    cmp_info = cmp_info.reshape((2, -1))
    cmp_mean = cmp_info[0]
    cmp_std = cmp_info[1]

    r = r * cmp_std + cmp_mean

    for i in range(r.shape[1]):
        name = "mygen_%i" % i
        generate_merlin_wav(r[:, i, :], file_basename=name,
                            do_post_filtering=False)
    print("Reconstruction complete")
else:
    print("Unrecognized extension, exiting")
