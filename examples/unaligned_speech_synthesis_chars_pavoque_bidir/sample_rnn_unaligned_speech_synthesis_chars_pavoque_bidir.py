#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe, generate_merlin_wav
import os
import numpy as np
from scipy.io import wavfile

filedir = "/Tmp/kastner/pavoque_all_speakers/norm_info/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
    sdir = "marge:" + filedir
    cmd = "rsync -avhp %s %s" % (sdir, fd)
    pe(cmd, shell=True)

filedir = "/Tmp/kastner/pavoque_all_speakers/numpy_features/"
if filedir[-1] != "/":
    fd = filedir + "/"
else:
    fd = filedir
if not os.path.exists(fd):
    os.makedirs(fd)
sdir = "marge:" + filedir
cmd = "rsync -avhp %s %s" % (sdir, fd)
pe(cmd, shell=True)

norm_info_file = "/Tmp/kastner/pavoque_all_speakers/norm_info/"
# try with vctk??? at least the stats are probably close...
# norm_info_file = "/Tmp/kastner/vctk_American_speakers/norm_info/"
norm_info_file += "norm_info_mgc_lf0_vuv_bap_63_MVN.dat"

files = [filedir + fs for fs in os.listdir(filedir)]
files = [f for f in files if "neutral" in f]
"""
calculate the text -> audio ratio
ratios = []
for fi in files:
    aa = np.load(fi)
    tl = len(str(aa["text"]))
    al = len(aa["audio_features"])
    if al != 0:
        ratios.append(float(tl) / al)
print(np.mean(ratios))
from IPython import embed; embed()
raise ValueError()
"""

random_state = np.random.RandomState(1999)
minibatch_size = 8
n_hid = 1024


train_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="german_chars",
                                               stop_index=.9,
                                               randomize=True,
                                               random_state=random_state)


valid_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="german_chars",
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
n_ctx_ins = n_hid
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
    all_results = []
    all_groundtruth = []
    all_phones = []
    all_attention = []

    # Surely 40 samples is enough for anyone...
    n_rounds = 5

    for n in range(n_rounds):
        X_mb, y_mb, X_mb_mask, y_mb_mask = next(valid_itr)
        y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
        y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))

        i_enc_h1 = np.zeros((minibatch_size, n_hid)).astype("float32")
        i_enc_h1_r = np.zeros((minibatch_size, n_hid)).astype("float32")

        i_h1 = np.zeros((minibatch_size, n_hid)).astype("float32")
        i_h2 = np.zeros((minibatch_size, n_hid)).astype("float32")
        i_h3 = np.zeros((minibatch_size, n_hid)).astype("float32")

        i_w1 = np.zeros((minibatch_size, n_ctx_ins)).astype("float32")
        i_k1 = np.zeros((minibatch_size, att_dim)).astype("float32")

        sample_length = len(y_mb)
        res = np.zeros((sample_length, minibatch_size, y_mb.shape[-1])).astype("float32")
        w_res = np.zeros((sample_length, minibatch_size, n_ctx_ins)).astype("float32")
        inmask = np.ones_like(res[:, :, 0])
        noise_pwr = 0.
        for i in range(2, sample_length):
            print("Round %i, Iteration %i" % (n, i))
            r = predict_function(X_mb, res[i - 2:i],
                                 X_mb_mask, inmask[i - 2:i],
                                 #y_mb[i - 2:i], inmask[i - 2:i],
                                 i_enc_h1, i_enc_h1_r,
                                 i_h1, i_h2, i_h3, i_k1, i_w1, noise_pwr)
            y_p = r[0]
            o_enc_h1 = r[1]
            o_enc_h1_r = r[2]
            o_h1 = r[3]
            o_h2 = r[4]
            o_h3 = r[5]
            o_k1 = r[6]
            o_w1 = r[7]

            thresh = 0 #sample_length // 2
            if i < thresh:
                res[i, :, :] = y_mb[i]
            else:
                res[i, :, :] = y_p
            w_res[i, :, :] = o_w1
            #i_enc_h1 never changes
            #i_enc_h1_r never changes
            i_h1, i_h2, i_h3 = (o_h1[-1], o_h2[-1], o_h3[-1])
            i_k1 = o_k1[-1]
            i_w1 = o_w1[-1]
        all_results.append(res)
        all_phones.append(X_mb)
        all_groundtruth.append(y_mb)
        all_attention.append(w_res)

    out = {}
    for n in range(n_rounds):
        out["audio_results_%i" % n] = all_results[n]
        out["associated_phones_%i" % n] = all_phones[n]
        out["groundtruth_%i" % n] = all_groundtruth[n]
        out["attention_%i" % n] = all_attention[n]
    savename = "sampled.npz"
    np.savez_compressed(savename, **out)
    print("Sampling complete, saved to %s" % savename)
elif ext == "npz":
    print("Found file path extension npz, reconstructing")
    with open(norm_info_file, 'rb') as f:
        cmp_info = np.fromfile(f, dtype=np.float32)
    cmp_info = cmp_info.reshape((2, -1))
    cmp_mean = cmp_info[0]
    cmp_std = cmp_info[1]
    npz_path = filepath
    d = np.load(npz_path)

    n_rounds = len([dk for dk in d.keys() if "audio_results" in dk])
    removed_count = 0
    total_count = 0
    for n in range(n_rounds):
        r = d["audio_results_%i" % n]
        r = r * cmp_std + cmp_mean

        genfile = "%i_%i_gen"
        truefile = "%i_%i_true"
        for i in range(r.shape[1]):
            total_count += 1
            name = genfile % (n, i)
            generate_merlin_wav(r[:, i, :], file_basename=name,
                                do_post_filtering=False)
            wavpath = "gen/" + name + ".wav"
            sr, wav = wavfile.read(wavpath)
            # remove DC
            wav = wav - np.mean(wav.astype("float32"))
            mn = np.mean(np.abs(wav))
            md = np.median(np.abs(wav))
            heuristic = mn
            if heuristic < 40:
                print("Detected insufficient activity in heuristic - removing %s" % wavpath)
                removed_count += 1
                os.remove(wavpath)

        gtr = d["groundtruth_%i" % n]
        gtr = gtr * cmp_std + cmp_mean

        for i in range(gtr.shape[1]):
            genpart = genfile % (n, i)
            if os.path.exists("gen/" + genpart + ".wav"):
                # Only get GT for files that made it past heuristic check
                truename = truefile % (n, i)
                generate_merlin_wav(gtr[:, i, :], file_basename=truename,
                                    do_post_filtering=False)
    print("Reconstruction complete")
    print("Accepted ratio %s" % str((total_count - removed_count) / float(total_count)))
else:
    print("Unrecognized extension, exiting")
