from collections import defaultdict
import numpy as np


def make_markov_mask(mb, mb_mask, limit, step_lookups, go_through=None,
                     warn=False):
    pre_mb = mb.copy()
    mb = mb - mb[:, :, 0][:, :, None]
    markov_masks = []
    for ii in range(mb.shape[2]):
        markov_masks.append(np.zeros((mb.shape[0], mb.shape[1], limit), dtype="float32"))
    for j in range(mb.shape[1]):
        for i in range(mb.shape[0]):
            if mb_mask[i, j] > 0:
                range_for_k = range(mb.shape[2])
                if go_through != None:
                    range_for_k = range(go_through + 1)
                for k in range_for_k:
                    if k == 0:
                        tt = step_lookups[k]["init"]
                    else:
                        if warn:
                            try:
                                tt = step_lookups[k][tuple(mb[i, j, :k])]
                            except:
                                tt = step_lookups[0]["init"]
                                print("WARNING: Key not found, falling back to all ones mask")
                        else:
                            tt = step_lookups[k][tuple(mb[i, j, :k])]

                    subidx = tt[(pre_mb[i, j, k] + tt) >= 0] + pre_mb[i, j, k]
                    subidx = subidx[subidx < limit]
                    subidx = subidx.astype("int32")
                    tmp = markov_masks[k][i, j].copy()
                    for si in subidx:
                        tmp[si] = 1.
                    markov_masks[k][i, j, :] = tmp
            else:
                for k in range(mb.shape[2]):
                    markov_masks[k][i, j, :] *= 0.
    return markov_masks


def preproc_and_make_lookups(mu, max_len=150, key=None):
    lp = mu["list_of_data_pitch"]
    ld = mu["list_of_data_duration"]

    lp2 = [lpi[:max_len] for n, lpi in enumerate(lp)]
    ld2 = [ldi[:max_len] for n, ldi in enumerate(ld)]

    lp = lp2
    ld = ld2

    # key can be major minor none
    if key is not None:
        lip = []
        lid = []
        for n, k in enumerate(mu["list_of_data_key"]):
            if key in k:
                lip.append(lp[n])
                lid.append(ld[n])
        lp = lip
        ld = lid

    lpn = np.concatenate(lp, axis=0)
    lpn = lpn - lpn[:, 0][:, None]

    ldn = np.concatenate(ld, axis=0)
    ldn = ldn - ldn[:, 0][:, None]

    lpnu = np.vstack({tuple(row) for row in lpn})
    ldnu = np.vstack({tuple(row) for row in ldn})

    n_pitches = len(mu["pitch_list"])
    n_durations = len(mu["duration_list"])

    step_lookups_pitch = []
    step_lookups_duration = []
    for i in range(ldn.shape[-1]):
        if i == 0:
            # default dict trickery backfired :|
            lup = {"init": np.arange(2 * n_pitches, dtype="float32") - n_pitches for k in range(n_pitches)}
            lud = {"init": np.arange(2 * n_durations, dtype="float32") - n_durations for k in range(n_durations)}
        elif i < ldn.shape[-1]:
            lup = {}
            keyset = {tuple(row) for row in lpnu[:, :i]}
            for k in keyset:
                ii = np.where(lpnu[:, :i] == k)[0]
                v = lpnu[ii, i]
                vset = np.array(sorted(list(set(v))))
                lup[k] = vset
            lud = {}
            keyset = {tuple(row) for row in ldnu[:, :i]}
            for k in keyset:
                ii = np.where(ldnu[:, :i] == k)[0]
                v = ldnu[ii, i]
                vset = np.array(sorted(list(set(v))))
                lud[k] = vset
        step_lookups_pitch.append(lup)
        step_lookups_duration.append(lud)
    return lp, ld, step_lookups_pitch, step_lookups_duration
