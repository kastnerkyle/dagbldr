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

pitch_clusters = 8192
dur_clusters = 1024
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


def unfixup_dur_list(dur_list):
    new = []
    dl = mu["duration_list"]
    for ldi in dur_list:
        ldi = ldi.copy().astype("float32")
        dur_where = []
        for n, dli in enumerate(dl):
            dur_where.append(np.where(ldi == n))

        # gross hackz needed since -1 and 0 perform the same function...
        # fix it?
        for n, dw in enumerate(dur_where[:-1]):
            ldi[dw] = dl[n] if n == 0 else dl[n + 1]
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


def pre_d(dmb):
    list_of_dur = [dmb[:, i, :] for i in range(dmb.shape[1])]
    o_list_of_dur = list_of_dur
    list_of_dur = fixup_dur_list(list_of_dur)

    q_list_of_dur, q_list_of_dur_codes = quantize(list_of_dur, dur_codebook, dur_oh_size)
    o_q_list_of_dur = q_list_of_dur
    q_list_of_dur = unfixup_dur_list(q_list_of_dur)

    q_list_of_dur = [qld[:, None, :] for qld in q_list_of_dur]
    q_list_of_dur_codes = [qldc[:, None, None] for qldc in q_list_of_dur_codes]
    q_dur_mb = np.concatenate(q_list_of_dur, axis=1)
    q_code_mb = np.concatenate(q_list_of_dur_codes, axis=1).astype("float32")
    return q_dur_mb, q_code_mb


def pre_p(pmb):
    list_of_pitch = [pmb[:, i, :] for i in range(pmb.shape[1])]
    o_list_of_pitch = list_of_pitch
    list_of_pitch = fixup_pitch_list(list_of_pitch)

    q_list_of_pitch, q_list_of_pitch_codes = quantize(list_of_pitch, pitch_codebook, pitch_oh_size)
    o_q_list_of_pitch = q_list_of_pitch
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


from collections import Counter, defaultdict

pitch_order = 2
dur_order = 1
p_total_frequency = {}
d_total_frequency = {}

for r in train_itr:
    pitch_mb, pitch_mask, dur_mb, dur_mask = r[:4]
    q_pitch_mb, q_pitch_code_mb = pre_p(pitch_mb)
    q_dur_mb, q_dur_code_mb = pre_d(dur_mb)
    # add 2 for start and eos
    q_pitch_code_mb += 2
    q_dur_code_mb += 2
    # start is 0, end is 1
    st_p = q_pitch_code_mb[0][None] * 0.
    e_p = q_pitch_code_mb[0][None] * 0. + 1
    st_d = q_dur_code_mb[0][None] * 0.
    e_d = q_dur_code_mb[0][None] * 0. + 1
    q_pitch_code_mb = np.concatenate((st_p, q_pitch_code_mb, e_p))
    q_dur_code_mb = np.concatenate((st_d, q_dur_code_mb, e_d))
    p_frequency = copy.deepcopy(p_total_frequency)
    d_frequency = copy.deepcopy(d_total_frequency)
    p_frequency = accumulate(q_pitch_code_mb, p_frequency, pitch_order)
    d_frequency = accumulate(q_dur_code_mb, d_frequency, dur_order)
    p_total_frequency.update(p_frequency)
    d_total_frequency.update(d_frequency)

p_total_frequency = normalize(p_total_frequency)
d_total_frequency = normalize(d_total_frequency)

def p_prob_func(prefix):
    history = prefix[-pitch_order:]
    lu = tuple(history)
    dist_lookup = p_total_frequency[lu]
    dist = [(v, k) for k, v in dist_lookup.items()]
    return dist

def d_prob_func(prefix):
    history = prefix[-1]
    lu = tuple(history)
    dist_lookup = d_total_frequency[lu]
    dist = [(v, k) for k, v in dist_lookup.items()]
    return dist

a = next(valid_itr)
pitch_mb, pitch_mask, dur_mb, dur_mask = a[:4]
q_pitch_mb, q_pitch_code_mb = pre_p(pitch_mb)
q_dur_mb, q_dur_code_mb = pre_d(dur_mb)
qpms = r[-1]

start_token = [0, int(q_pitch_code_mb[0, 0, 0])]
end_token = 1
stochastic = True
beam_width = 10
clip = 50
random_state = np.random.RandomState(90210)
db = beamsearch(d_prob_func, beam_width,
                start_token=start_token,
                end_token=end_token,
                clip_len=clip,
                stochastic=stochastic,
                random_state=random_state)

quantized_pitch_seqs = codebook_lookup([np.array(pbi[0]).astype("int32") for pbi in pb], pitch_codebook, 4)
quantized_dur_seqs = codebook_lookup([np.array(dbi[0]).astype("int32") for dbi in db], dur_codebook, 4)
from IPython import embed; embed(); raise ValueError()

start_token = 0
end_token = 1
stochastic = True
beam_width = 10
clip = 50
random_state = np.random.RandomState(90210)
pb = beamsearch(p_prob_func, beam_width,
                start_token=start_token,
                end_token=end_token,
                clip_len=clip,
                stochastic=stochastic,
                random_state=random_state)

quantized_pitch_seqs = codebook_lookup([np.array(pbi[0]).astype("int32") for pbi in pb], pitch_codebook, 4)


from IPython import embed; embed(); raise ValueError()

sampled_dur = dur_mb[:, 0, :]

i = 0
pitches_and_durations_to_pretty_midi(pitch_mb, dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_sample_{}.mid",
                                     #list_of_quarter_length = qpms,
                                     voice_params="woodwinds",
                                     add_to_name=i * pitch_mb.shape[1])

pitches_and_durations_to_pretty_midi(q_pitch_mb, q_dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_quantized_sample_{}.mid",
                                     #list_of_quarter_length = qpms,
                                     voice_params="woodwinds",
                                     add_to_name=i * q_pitch_mb.shape[1])
