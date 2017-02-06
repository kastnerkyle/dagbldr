#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe, generate_merlin_wav, simple_energy_vad
from dagbldr import fetch_checkpoint_dict
from dagbldr.utils import numpy_one_hot
import os
import numpy as np
from scipy.io import wavfile

filedir = "/Tmp/kastner/romanian_multi_speakers/norm_info/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
    sdir = "marge:" + filedir
    cmd = "rsync -avhp %s %s" % (sdir, fd)
    pe(cmd, shell=True)

filedir = "/Tmp/kastner/romanian_multi_speakers/numpy_features/"
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

norm_info_file = "/Tmp/kastner/romanian_speakers/norm_info/"
# try with vctk??? at least the stats are probably close...
# norm_info_file = "/Tmp/kastner/vctk_American_speakers/norm_info/"
norm_info_file += "norm_info_mgc_lf0_vuv_bap_63_MVN.dat"
fixed_sample_length = 2250

random_state = np.random.RandomState(1999)
minibatch_size = 8
n_hid = 1024


train_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="romanian_chars",
                                               extra_ids=file_ids,
                                               stop_index=.9,
                                               randomize=False,
                                               random_state=random_state)


valid_itr = masked_synthesis_sequence_iterator(files, minibatch_size,
                                               itr_type="unaligned_text",
                                               class_set="romanian_chars",
                                               extra_ids=file_ids,
                                               start_index=.9,
                                               randomize=False,
                                               random_state=random_state)
X_mb, y_mb, X_mb_mask, y_mb_mask, id_mb = next(valid_itr)
y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))

n_ids = len(list(set(file_ids)))
id_mb = numpy_one_hot(id_mb, n_ids).astype("float32")

valid_itr.reset()

n_text_ins = X_mb.shape[-1]
n_audio_ins = y_mb.shape[-1]
n_audio_outs = y_mb.shape[-1]
n_ctx_ins = 2 * n_hid
att_dim = 20

import argparse
import cPickle as pickle
parser = argparse.ArgumentParser(description="Sample audio from saved model")
parser.add_argument("npz_file", help="Stored file", default="none.none",
                    nargs="?")
args = parser.parse_args()
filepath = args.npz_file
ext = filepath.split(".")[-1]

if ext == "none":
    print("Fetching and doing sampling")
    checkpoint_dict = fetch_checkpoint_dict(["multispeaker", "romanian"])
    print("Finished unpickling file")

    predict_function = checkpoint_dict["predict_function"]
    all_results = []
    all_groundtruth = []
    all_phones = []
    all_attention = []

    # A way to increase the number of samples
    n_rounds = min([n_ids, 5])
    for n in range(n_rounds):
        X_mb, y_mb, X_mb_mask, y_mb_mask, id_mb = next(valid_itr)
        y_mb = np.concatenate((0. * y_mb[0, :, :][None], y_mb))
        y_mb_mask = np.concatenate((1. * y_mb_mask[0, :][None], y_mb_mask))
        # hack for multispeaker
        id_mb = 0 * id_mb + n
        valid_itr.reset()
        id_mb = numpy_one_hot(id_mb, n_ids).astype("float32")

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
            r = predict_function(X_mb, res[i - pre:i],
                                 X_mb_mask, inmask[i - pre:i],
                                 id_mb,
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
            heuristic = np.mean(np.abs(wav)) / np.median(np.abs(wav))
            thresh = 1.5

            if heuristic < thresh:
                print("Detected insufficient activity in heuristic - moving to trash %s" % wavpath)
                removed_count += 1
                pe("mv %s gen/trash/%s.wav" % (wavpath, name), shell=True)
            """
            normwav = wav / np.iinfo(iwav.dtype).max
            s_vad, speech_presence = simple_energy_vad(normwav, sr,
                                                       theta_main=30,
                                                       theta_min=-50)
            inds = np.arange(len(speech_presence))[speech_presence == True]
            if len(inds) == 0:
                heuristic = 0
            else:
                start = inds[0]
                end = inds[-1]
                heuristic = (end - start) / float(sr)
            print(heuristic)
            if heuristic < 1:
                print("Detected insufficient activity in heuristic - removing %s" % wavpath)
                removed_count += 1
                os.remove(wavpath)
            else:
                # FIXME: HACK TO AVOID ISSUES WITH ATTENTION START
                print("Performing voice detection cleanup...")
                iwav = iwav[start:end]
                wavfile.write(wavpath, sr, iwav)
            """

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
else:
    print("Unrecognized extension, exiting")
