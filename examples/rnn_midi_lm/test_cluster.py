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

n_iter = 1


def oh_3d(a, oh_size="max"):
    if oh_size == "max":
        oh_size = a.max()
    return (np.arange(oh_size) == a[:, :, None] - 1).astype(int)


def get_codebook(list_of_arr, n_components, n_iter):
    j = np.vstack(list_of_arr)
    oh_j = oh_3d(j)
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


# map to positive ints
#ld_new = []
#dl = mu["duration_list"]
#for ldi in ld:
#    ldi = ldi.copy()
#    dur_where = []
#    for n, dli in enumerate(dl):
#        dur_where.append(np.where(ldi == dli))
#
#for n, dw in enumerate(dur_where):
#        ldi[dw] = n
#    ld_new.append(ldi)
#ld = ld_new
#
#if not os.path.exists("dur_codebook.npy"):
#    dur_codebook = get_codebook(ld, n_components=500, n_iter=1)
#    np.save("dur_codebook.npy", dur_codebook)
#else:
#    dur_codebook = np.load("dur_codebook.npy")
#
#list_of_dur = [dur_mb[:, i, :] for i in range(dur_mb.shape[1])]
#q_list_of_dur, q_list_of_dur_codes = quantize(list_of_dur, dur_codebook, 12)
#def _dur_fixup(arr):
#    dl = mu["duration_list"]
#    arr = arr.copy()
#    dur_where = []
#    for n, _ in enumerate(dl):
#        dur_where.append(np.where(arr == n))
#    for n, dw in enumerate(dur_where):
#        arr[dw] = dl[n]
#    return arr
#q_list_of_dur = [_dur_fixup(qld) for qld in q_list_of_dur]
#q_list_of_dur = [qld[:, None, :] for qld in q_list_of_dur]
#q_dur_mb = np.concatenate(q_list_of_dur, axis=1)

if not os.path.exists("pitch_codebook.npy"):
    pitch_codebook = get_codebook(lp, n_components=4096, n_iter=1)
    np.save("pitch_codebook.npy", pitch_codebook)
else:
    pitch_codebook = np.load("pitch_codebook.npy")

list_of_pitch = [pitch_mb[:, i, :] for i in range(pitch_mb.shape[1])]
q_list_of_pitch, q_list_of_pitch_codes = quantize(list_of_pitch, pitch_codebook, 88)

q_list_of_pitch = [qlp[:, None, :] for qlp in q_list_of_pitch]

q_pitch_mb = np.concatenate(q_list_of_pitch, axis=1)

i = 0
pitches_and_durations_to_pretty_midi(pitch_mb, dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_sample_{}.mid",
                                     list_of_quarter_length = qpms,
                                     voice_params="woodwinds",
                                     add_to_name=i * pitch_mb.shape[1])

pitches_and_durations_to_pretty_midi(q_pitch_mb, dur_mb,
                                     save_dir="samples/samples",
                                     name_tag="test_quantized_sample_{}.mid",
                                     list_of_quarter_length = qpms,
                                     voice_params="woodwinds",
                                     add_to_name=i * q_pitch_mb.shape[1])
