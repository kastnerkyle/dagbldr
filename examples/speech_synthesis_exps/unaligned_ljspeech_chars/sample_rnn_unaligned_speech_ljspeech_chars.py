#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe, generate_merlin_wav, ltsd_vad
from dagbldr import fetch_checkpoint_dict
from dagbldr.utils import numpy_one_hot
import os
import sys
import numpy as np
from scipy.io import wavfile
import copy
import time

import argparse
import cPickle as pickle

parser = argparse.ArgumentParser(description="Sample audio from saved model")
parser.add_argument("npz_file", help="Stored file", default="none.none",
                    nargs="?")
parser.add_argument("-s", "--sample_text", help="file with sample text",
                    default=None, required=False)
parser.add_argument("-g", "--ground_truth", help="npz file is groundtruth",
                    action="store_true")
args = parser.parse_args()
filepath = args.npz_file
is_groundtruth = args.ground_truth
arg_ext = filepath.split(".")[-1]

norm_info_file = "/Tmp/kastner/lj_speech_speakers/norm_info/"
# try with vctk??? at least the stats are probably close...
# norm_info_file = "/Tmp/kastner/vctk_American_speakers/norm_info/"
norm_info_file += "norm_info_mgc_lf0_vuv_bap_63_MVN.dat"

def fetch_from_base(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    if not os.path.exists(fd):
        os.makedirs(fd)
    sdir = "leto01:" + fd
    cmd = "rsync -avhp %s %s" % (sdir, fd)
    pe(cmd, shell=True)

filedir = "/Tmp/kastner/lj_speech_speakers/norm_info/"
fetch_from_base(filedir)

if arg_ext == "npz":
    print("Found file path extension npz, reconstructing")
    with open(norm_info_file, 'rb') as f:
        cmp_info = np.fromfile(f, dtype=np.float32)
    cmp_info = cmp_info.reshape((2, -1))
    cmp_mean = cmp_info[0]
    cmp_std = cmp_info[1]
    npz_path = filepath
    d = np.load(npz_path)
    if is_groundtruth:
        n_rounds = 1
    else:
        n_rounds = len([dk for dk in d.keys() if "audio_results" in dk])
    removed_count = 0
    total_count = 0
    start_time = time.time()
    for n in range(n_rounds):
        if is_groundtruth:
            # need to add the dimension
            r = d["audio_features"][:, None]
        else:
            r = d["audio_results_%i" % n]
        r = r * cmp_std + cmp_mean

        genfile = "%i_%i_gen"
        truefile = "%i_%i_true"
        for i in range(r.shape[1]):
            total_count += 1
            name = genfile % (n, i)
            if not os.path.exists("gen"):
                os.mkdir("gen")
            if not os.path.exists("gen/trash"):
                os.mkdir("gen/trash")

            generate_merlin_wav(r[:, i, :], file_basename=name,
                                do_post_filtering=False)
            wavpath = "gen/" + name + ".wav"
            sr, iwav = wavfile.read(wavpath)
            # remove DC
            wav = iwav - np.mean(iwav.astype("float32"))
            vad_wav, vad = ltsd_vad(wav, sr, threshold=9)
            # Try to get 2 seconds total above the threshold
            heuristic = sum(vad.astype("int32")) / float(sr)
            thresh = 2

            if heuristic < thresh:
                print("Detected insufficient activity in heuristic - moving to trash %s" % wavpath)
                removed_count += 1
                pe("mv %s gen/trash/%s.wav" % (wavpath, name), shell=True)

        if is_groundtruth:
            gtr = d["audio_features"][:, None]
        else:
            gtr = d["groundtruth_%i" % n]
        gtr = gtr * cmp_std + cmp_mean

        for i in range(gtr.shape[1]):
            genpart = genfile % (n, i)
            # Only get GT for files that made it past heuristic check
            truename = truefile % (n, i)
            generate_merlin_wav(gtr[:, i, :], file_basename=truename,
                                do_post_filtering=False)
            if not os.path.exists("gen/" + genpart + ".wav"):
                truepath = "gen/" + truename + ".wav"
                pe("mv %s gen/trash/%s.wav" % (truepath, truename), shell=True)
    print("Reconstruction complete")
    print("Accepted ratio %s" % str((total_count - removed_count) / float(total_count)))
    sys.exit()

filedir = "/Tmp/kastner/lj_speech_speakers/numpy_features/"
fetch_from_base(filedir)
files = [filedir + fs for fs in sorted(os.listdir(filedir))]

fixed_sample_length = 2250
random_state = np.random.RandomState(1999)
minibatch_size = 8
n_hid = 1024

train_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="ljspeech_chars",
                                               stop_index=.9,
                                               randomize=False)

valid_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="ljspeech_chars",
                                               start_index=.9,
                                               randomize=False)

sample_filedir = "/Tmp/kastner/lj_speech_speakers/text_to_sample/"
if not os.path.exists(sample_filedir):
    os.mkdir(sample_filedir)
sample_files = [sample_filedir + fs for fs in os.listdir(sample_filedir)]
if len(sample_files) != 0:
    print("WARNING: text_to_sample not empty but will be overwritten!")
    print("WARNING: This is normal, but be aware.")
    rm_glob = sample_filedir + "*.npz"
    pe("rm %s" % rm_glob, shell=True)

if args.sample_text is not None:
    fpath = args.sample_text
    if not os.path.exists(fpath):
        raise ValueError("Unable to find file at %s, exiting." % fpath)
    with open(fpath, "r") as f:
        lines = f.readlines()
    # allow comment lines with #
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len("".join(l.split("#")[0].split())) != 0]
    lines = [l for l in lines if len("".join(l.split())) != 0]
else:
    raise ValueError("Must pass sentences file with -s")

if sample_filedir[-1] != "/":
    sample_filedir += "/"

for n, i_file in enumerate(files[:len(lines)]):
    file_ext = "sample_%i.npz" % n
    new_fname = sample_filedir + file_ext
    pe("cp %s %s" % (i_file, new_fname), shell=True)

sample_files = [sample_filedir + fs for fs in os.listdir(sample_filedir)]
sample_files = sorted([s for s in sample_files if s[-4:] == ".npz"])
for n, sf in enumerate(sample_files):
    d = np.load(sf)
    new_file_id = sf.split("/")[-1][:-4]
    new_text = np.array(lines[n])
    # paranoia
    # these will not be right!
    # FIXME: get rid of these entries in general
    new_durations = d["durations"]
    new_phonemes = d["phonemes"]
    new_audio_features = d["audio_features"] * 0
    new_d = {k: v.copy() for k, v in d.items()}
    new_d["audio_features"] = new_audio_features
    new_d["text"] = new_text
    new_d["phonemes"] = new_phonemes
    new_d["durations"] = new_durations
    np.savez(sf, **new_d)
assert len(sample_files) >= minibatch_size

valid_itr = masked_synthesis_sequence_iterator(sample_files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="ljspeech_chars",
                                               randomize=False)

X_mb, y_mb, X_mb_mask, y_mb_mask = next(valid_itr)
y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))

valid_itr.reset()

n_text_ins = X_mb.shape[-1]
n_audio_ins = y_mb.shape[-1]
n_audio_outs = y_mb.shape[-1]
n_ctx_ins = 2 * n_hid
att_dim = 20

if arg_ext == "none":
    print("Fetching and doing sampling")
    checkpoint_dict = fetch_checkpoint_dict(["ljspeech_chars"])
    print("Finished unpickling file")

    predict_function = checkpoint_dict["predict_function"]
    all_results = []
    all_groundtruth = []
    all_phones = []
    all_attention = []

    # A way to increase the number of samples
    n_rounds = 1
    start_time = time.time()
    for n in range(n_rounds):
        X_mb, y_mb, X_mb_mask, y_mb_mask = next(valid_itr)
        y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
        y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))
        valid_itr.reset()
        i_enc_h1 = np.zeros((minibatch_size, n_hid)).astype("float32")
        i_enc_h1_r = np.zeros((minibatch_size, n_hid)).astype("float32")

        i_h1 = np.zeros((minibatch_size, n_hid)).astype("float32")
        i_h2 = np.zeros((minibatch_size, n_hid)).astype("float32")
        i_h3 = np.zeros((minibatch_size, n_hid)).astype("float32")

        i_w1 = np.zeros((minibatch_size, n_ctx_ins)).astype("float32")
        i_k1 = np.zeros((minibatch_size, att_dim)).astype("float32")

        sample_length = fixed_sample_length
        res = np.zeros((sample_length, minibatch_size, y_mb.shape[-1])).astype("float32")
        w_res = np.zeros((sample_length, minibatch_size, n_ctx_ins)).astype("float32")
        inmask = np.ones_like(res[:, :, 0])
        noise_pwr = 0.
        pre = 2
        for i in range(pre, sample_length):
            print("Round %i, Iteration %i" % (n, i))
            #r = predict_function(X_mb, y_mb,
            #                     X_mask, y_mask,
            r = predict_function(X_mb, res[i - pre:i],
                                 X_mb_mask, inmask[i - pre:i],
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
            w_res[i, :, :] = o_w1[-1]
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
        out["associated_text_%i" % n] = all_phones[n]
        out["groundtruth_%i" % n] = all_groundtruth[n]
        out["attention_%i" % n] = all_attention[n]
    savename = "sampled.npz"
    np.savez_compressed(savename, **out)
    print("Sampling complete, saved to %s" % savename)
    end_time = time.time()
    print("Number of rounds sampled %i" % int(n_rounds))
    print("Total time to sample %f" % (end_time - start_time))
