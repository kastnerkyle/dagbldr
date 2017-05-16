#!/usr/bin/env python
import numpy as np

from dagbldr.datasets import pitches_and_durations_to_pretty_midi
from dagbldr.datasets import list_of_array_iterator

from dagbldr.datasets import fetch_symbtr_music21
from dagbldr.datasets import fetch_bach_chorales_music21
from dagbldr.datasets import fetch_wikifonia_music21
from dagbldr.datasets import fetch_haralick_midi_music21
from dagbldr.datasets import fetch_lakh_midi_music21

from dagbldr.utils import minibatch_kmedians

import os
import cPickle as pickle

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
from_scratch = True

pitch_oh_size = 89
dur_oh_size = 12

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

r = next(valid_itr)
pitch_mb, pitch_mask, dur_mb, dur_mask = r[:4]
qpms = r[-1]


def oh_3d(a, oh_size="max"):
    if oh_size == "max":
        oh_size = a.max()
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

q_pitch_mb, q_pitch_code_mb = pre_p(pitch_mb)
q_dur_mb, q_dur_code_mb = pre_d(dur_mb)

i = 0
pitches_and_durations_to_pretty_midi(pitch_mb, dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_sample_{}.mid",
                                     list_of_quarter_length = qpms,
                                     voice_params="woodwinds",
                                     add_to_name=i * pitch_mb.shape[1])

pitches_and_durations_to_pretty_midi(q_pitch_mb, q_dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_quantized_sample_{}.mid",
                                     list_of_quarter_length = qpms,
                                     voice_params="woodwinds",
                                     add_to_name=i * q_pitch_mb.shape[1])
