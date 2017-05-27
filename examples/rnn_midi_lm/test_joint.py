#!/usr/bin/env python
import numpy as np

from dagbldr.datasets import pitches_and_durations_to_pretty_midi
from dagbldr.datasets import list_of_array_iterator

from dagbldr.datasets import fetch_symbtr_music21
from dagbldr.datasets import fetch_bach_chorales_music21
from dagbldr.datasets import fetch_wikifonia_music21
from dagbldr.datasets import fetch_haralick_midi_music21
from dagbldr.datasets import fetch_lakh_midi_music21

from dagbldr.utils import minibatch_kmedians, beamsearch

import os
import cPickle as pickle
import copy

#mu = fetch_lakh_midi_music21(subset="pop")
mu = fetch_bach_chorales_music21()
#mu = fetch_haralick_midi_music21(subset="mozart_piano")
#mu = fetch_symbtr_music21()
#mu = fetch_wikifonia_music21()

#n_epochs = 500
#n_epochs = 2350
#n_epochs = 3000
minibatch_size = 10
n_iter = 0

pitch_clusters = 32000
dur_clusters = 4000
from_scratch = False

pitch_oh_size = len(mu["pitch_list"])
dur_oh_size = len(mu["duration_list"])

order = mu["list_of_data_pitch"][0].shape[-1]

n_pitches = len(mu["pitch_list"])
n_dur = len(mu["duration_list"])
random_state = np.random.RandomState(1999)

lp = mu["list_of_data_pitch"]
ld = mu["list_of_data_duration"]
lql = mu["list_of_data_quarter_length"]

train_itr = list_of_array_iterator([lp, ld], minibatch_size,
                                   list_of_extra_info=[lql],
                                   stop_index=.9,
                                   randomize=True, random_state=random_state)

valid_itr = list_of_array_iterator([lp, ld], minibatch_size,
                                   list_of_extra_info=[lql],
                                   start_index=.9,
                                   randomize=True, random_state=random_state)

r = next(train_itr)
pitch_mb, pitch_mask, dur_mb, dur_mask = r[:4]
train_itr.reset()


def oh_3d(a, oh_size):
    return (np.arange(oh_size) == a[:, :, None] - 1).astype(int)


def get_codebook(list_of_arr, n_components, n_iter, oh_size):
    j = np.vstack(list_of_arr)
    oh_j = oh_3d(j, oh_size=oh_size)
    shp = oh_j.shape
    oh_j2 = oh_j.reshape(-1, shp[1] * shp[2])

    codebook = minibatch_kmedians(oh_j2, n_components=n_components,
                                  n_iter=n_iter,
                                  random_state=random_state, verbose=True)
    return codebook


def quantize(list_of_arr, codebook, oh_size):
    from scipy.cluster.vq import vq
    quantized_arr = []
    list_of_codes = []
    for arr in list_of_arr:
        oh_a = oh_3d(arr, oh_size)
        shp = oh_a.shape
        oh_a2 = oh_a.reshape(-1, shp[1] * shp[2])

        codes, _ = vq(oh_a2, codebook)
        list_of_codes.append(codes)

        q_oh_a = codebook[codes]
        q_oh_a = q_oh_a.reshape(-1, shp[1], shp[2]).argmax(axis=-1)
        quantized_arr.append(q_oh_a)
    return quantized_arr, list_of_codes


def codebook_lookup(list_of_code_arr, codebook, last_shape=4):
    reconstructed_arr = []
    for arr in list_of_code_arr:
        pitch_slices = []
        oh_codes = codebook[arr]
        pitch_size = codebook.shape[1] // last_shape
        boundaries = np.arange(1, last_shape + 1, 1) * pitch_size
        for i in range(len(oh_codes)):
            pitch_slice = np.where(oh_codes[i] == 1)[0]
            for n, bo in enumerate(boundaries):
                if len(pitch_slice) <= n:
                    pitch_slice = np.insert(pitch_slice, len(pitch_slice), 0)
                elif pitch_slice[n] >= bo:
                    pitch_slice = np.insert(pitch_slice, n, 0)
                else:
                    pass
            pitch_slices.append(pitch_slice % pitch_size)
        new_arr = np.array(pitch_slices).astype("float32")
        reconstructed_arr.append(new_arr)
    return reconstructed_arr


def fixup_dur_list(dur_list):
    new = []
    dl = mu["duration_list"]

    for ldi in dur_list:
        ldi = ldi.copy()
        dur_where = []
        for n, dli in enumerate(dl):
            dur_where.append(np.where(ldi == dli))

        for n, dw in enumerate(dur_where):
            ldi[dw] = n
        new.append(ldi)
    return new


def unfixup_dur_list(dur_list, hack=True):
    new = []
    dl = mu["duration_list"]
    for ldi in dur_list:
        ldi = ldi.copy().astype("float32")
        dur_where = []
        for n, dli in enumerate(dl):
            dur_where.append(np.where(ldi == n))

        if hack:
            for n, dw in enumerate(dur_where[:-2]):
                ldi[dw] = dl[n] if n <= 1 else dl[n + 2]
            new.append(ldi)
        else:
            for n, dw in enumerate(dur_where):
                ldi[dw] = dl[n]
            new.append(ldi)
    return new


def fixup_pitch_list(pitch_list):
    new = []
    pl = mu["pitch_list"]

    for lpi in pitch_list:
        lpi = lpi.copy()
        pitch_where = []
        for n, pli in enumerate(pl):
            pitch_where.append(np.where(lpi == pli))

        for n, pw in enumerate(pitch_where):
            lpi[pw] = n
        new.append(lpi)
    return new


def unfixup_pitch_list(pitch_list):
    new = []
    pl = mu["pitch_list"]
    for lpi in pitch_list:
        lpi = lpi.copy().astype("float32")
        pitch_where = []
        for n, pli in enumerate(pl):
            pitch_where.append(np.where(lpi == n))

        for n, pw in enumerate(pitch_where):
            lpi[pw] = pl[n]
        new.append(lpi)
    return new


ld = fixup_dur_list(ld)
lp = fixup_pitch_list(lp)
from IPython import embed; embed(); raise ValueError()


if from_scratch or not os.path.exists("dur_codebook.npy"):
    dur_codebook = get_codebook(ld, n_components=dur_clusters, n_iter=n_iter, oh_size=dur_oh_size)
    np.save("dur_codebook.npy", dur_codebook)
else:
    dur_codebook = np.load("dur_codebook.npy")


if from_scratch or not os.path.exists("pitch_codebook.npy"):
    pitch_codebook = get_codebook(lp, n_components=pitch_clusters, n_iter=n_iter, oh_size=pitch_oh_size)
    np.save("pitch_codebook.npy", pitch_codebook)
else:
    pitch_codebook = np.load("pitch_codebook.npy")


def pre_d(dmb, quantize_it=True):
    list_of_dur = [dmb[:, i, :] for i in range(dmb.shape[1])]
    o_list_of_dur = list_of_dur
    list_of_dur = fixup_dur_list(list_of_dur)

    if quantize:
        q_list_of_dur, q_list_of_dur_codes = quantize(list_of_dur, dur_codebook, dur_oh_size)
        o_q_list_of_dur = q_list_of_dur
        q_list_of_dur = unfixup_dur_list(q_list_of_dur, hack=True)
    else:
        q_list_of_dur = list_of_dur
        # garbage
        q_list_of_dur_codes = list_of_dur
        q_list_of_dur = unfixup_dur_list(q_list_of_dur, hack=False)

    q_list_of_dur = [qld[:, None, :] for qld in q_list_of_dur]
    q_list_of_dur_codes = [qldc[:, None, None] for qldc in q_list_of_dur_codes]
    q_dur_mb = np.concatenate(q_list_of_dur, axis=1)
    q_code_mb = np.concatenate(q_list_of_dur_codes, axis=1).astype("float32")
    return q_dur_mb, q_code_mb


def pre_p(pmb, quantize_it=True):
    list_of_pitch = [pmb[:, i, :] for i in range(pmb.shape[1])]
    o_list_of_pitch = list_of_pitch
    list_of_pitch = fixup_pitch_list(list_of_pitch)

    if quantize:
        q_list_of_pitch, q_list_of_pitch_codes = quantize(list_of_pitch, pitch_codebook, pitch_oh_size)
        o_q_list_of_pitch = q_list_of_pitch
    else:
        q_list_of_pitch = list_of_pitch
        # garbage
        q_list_of_pitch_codes = list_of_pitch
    q_list_of_pitch = unfixup_pitch_list(q_list_of_pitch)

    q_list_of_pitch = [qlp[:, None, :] for qlp in q_list_of_pitch]
    q_list_of_pitch_codes = [qlpc[:, None, None] for qlpc in q_list_of_pitch_codes]
    q_pitch_mb = np.concatenate(q_list_of_pitch, axis=1)
    q_code_mb = np.concatenate(q_list_of_pitch_codes, axis=1).astype("float32")
    return q_pitch_mb, q_code_mb


def accumulate(mb, counter_dict, order):
    counter_dict = copy.deepcopy(counter_dict)
    for mi in range(mb.shape[1]):
        si = order
        for ni in range(len(mb) - order - 1):
            se = si - order
            ee = si
            prefix = tuple(mb[se:ee, mi].ravel())
            next_i = mb[ee, mi].ravel()[0]
            if prefix not in counter_dict.keys():
                counter_dict[prefix] = {}

            if next_i not in counter_dict[prefix].keys():
                counter_dict[prefix][next_i] = 1
            else:
                counter_dict[prefix][next_i] += 1
            si += 1
    return counter_dict


def normalize(counter_dict):
    counter_dict = copy.deepcopy(counter_dict)
    for k in counter_dict.keys():
        sub_d = copy.deepcopy(counter_dict[k])

        tot = 0.
        for sk in sub_d.keys():
            tot += sub_d[sk]

        for sk in sub_d.keys():
            sub_d[sk] /= float(tot)

        counter_dict[k] = sub_d
    return counter_dict


joint_order = 10
p_total_frequency = {}
d_total_frequency = {}

for r in train_itr:
    pitch_mb, pitch_mask, dur_mb, dur_mask = r[:4]
    q_pitch_mb, q_pitch_code_mb = pre_p(pitch_mb)
    q_dur_mb, q_dur_code_mb = pre_d(dur_mb)
    p_frequency = copy.deepcopy(p_total_frequency)
    d_frequency = copy.deepcopy(d_total_frequency)
    p_frequency = accumulate(q_pitch_code_mb, p_frequency, joint_order)
    d_frequency = accumulate(q_dur_code_mb, d_frequency, joint_order)
    p_total_frequency.update(p_frequency)
    d_total_frequency.update(d_frequency)

p_total_frequency = normalize(p_total_frequency)
d_total_frequency = normalize(d_total_frequency)


def rolloff_lookup(lookup_dict, lookup_key):
    """ roll off lookups n, n-1, n-2, n-3, down to random choice at 0 """
    lk = lookup_key
    ld = lookup_dict
    try:
        dist_lookup = lookup_dict[lk]
    except KeyError:
        for oi in range(1, len(lookup_key)):
            if oi == (len(lookup_key) - 1):
               # choose one of the elements of the lookup...
               sub_keys = sorted(list(set(lookup_key)))
               dist_lookup = {sk: 1. / len(sub_keys) for sk in sub_keys}
               break
            else:
                sub_keys = [ki for ki in lookup_dict.keys() if lk[oi:] == ki[oi:]]
                if len(sub_keys) > 0:
                   random_state.shuffle(sub_keys)
                   dist_lookup = lookup_dict[sub_keys[0]]
                   break
    return dist_lookup


def prob_func(prefix):
    history = prefix[-joint_order:]
    pitch_history = [h[0] for h in history]
    dur_history = [h[1] for h in history]
    p_lu = tuple(pitch_history)
    d_lu = tuple(dur_history)
    pitch_dist = rolloff_lookup(p_total_frequency, p_lu)
    dur_dist = rolloff_lookup(d_total_frequency, d_lu)
    dist = []
    # model as p(x, y) = p(x) * p(y)
    for pk in pitch_dist.keys():
        for dk in dur_dist.keys():
            dist.append((pitch_dist[pk] * dur_dist[dk], (pk, dk)))
    return dist


a = next(valid_itr)
pitch_mb, pitch_mask, dur_mb, dur_mask = a[:4]
pitch_mb, _ = pre_p(pitch_mb, quantize_it=False)
dur_mb, _ = pre_d(dur_mb, quantize_it=False)

n_pitch_mb, n_pitch_mask, n_dur_mb, n_dur_mask = a[:4]
q_pitch_mb, q_pitch_code_mb = pre_p(n_pitch_mb)
q_dur_mb, q_dur_code_mb = pre_d(n_dur_mb)
qpms = r[-1]

final_pitches = []
final_durs = []
for mbi in range(minibatch_size):
    start_pitch_token = [int(qp) for qp in list(q_pitch_code_mb[:joint_order, mbi, 0])]
    start_dur_token = [int(dp) for dp in list(q_dur_code_mb[:joint_order, mbi, 0])]
    start_token = [tuple([qp, dp]) for qp, dp in zip(start_pitch_token, start_dur_token)]
    end_token = [(start_token[0][0], None)]
    stochastic = True
    beam_width = 10
    clip = 75
    random_state = np.random.RandomState(90210)
    b = beamsearch(prob_func, beam_width,
                   start_token=start_token,
                   end_token=end_token,
                   clip_len=clip,
                   stochastic=stochastic,
                   random_state=random_state)

    # for all beams, take the sequence (p[0]) and the respective type (ip[0] for pitch, ip[1] for dur)
    # last number (4) for reconstruction to actual data (4 voices)
    quantized_pitch_seqs = codebook_lookup([np.array([ip[0] for ip in p[0]]).astype("int32") for p in b], pitch_codebook, 4)
    quantized_dur_seqs = codebook_lookup([np.array([ip[1] for ip in p[0]]).astype("int32") for p in b], dur_codebook, 4)

    # take top beams
    final_pitches.append(quantized_pitch_seqs[0])
    final_durs.append(quantized_dur_seqs[0])

# make into a minibatch
pad_size = max([len(fp) for fp in final_pitches])
new_qps = np.zeros((pad_size, len(final_pitches), final_pitches[0].shape[1])).astype("float32")
for n, fp in enumerate(final_pitches):
    new_qps[:len(fp), n] = fp

pad_size = max([len(fd) for fd in final_durs])
new_qds = np.zeros((pad_size, len(final_durs), final_durs[0].shape[1])).astype("float32")
for n, fd in enumerate(final_durs):
    new_qds[:len(fd), n] = fd

q_pitch_mb = new_qps
q_dur_mb = new_qds

min_len = min([q_pitch_mb.shape[0], q_dur_mb.shape[0]])
q_pitch_mb = q_pitch_mb[:min_len]
q_dur_mb = q_dur_mb[:min_len]

pitch_where = []
duration_where = []
pl = mu['pitch_list']
dl = mu['duration_list']


for n, pli in enumerate(pl):
    pitch_where.append(np.where(q_pitch_mb == n))

for n, dli in enumerate(dl):
    duration_where.append(np.where(q_dur_mb == n))

for n, pw in enumerate(pitch_where):
    q_pitch_mb[pw] = pl[n]

for n, dw in enumerate(duration_where):
    q_dur_mb[dw] = dl[n]


i = 0
# ext in order to avoid influence of "priming"
ext = int(joint_order // 2) + 1
pitch_mb = pitch_mb[joint_order + ext:]
dur_mb = dur_mb[joint_order + ext:]
q_pitch_mb = q_pitch_mb[joint_order + ext:]
q_dur_mb = q_dur_mb[joint_order + ext:]
pitches_and_durations_to_pretty_midi(pitch_mb, dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_sample_{}.mid",
                                     #list_of_quarter_length=[int(.5 * qpm) for qpm in qpms],
                                     voice_params="woodwinds",
                                     add_to_name=i * pitch_mb.shape[1])

pitches_and_durations_to_pretty_midi(q_pitch_mb, q_dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_quantized_sample_{}.mid",
                                     #list_of_quarter_length=qpms,
                                     voice_params="woodwinds",
                                     add_to_name=i * q_pitch_mb.shape[1])
