# -*- coding: utf-8 -*-
from __future__ import print_function, division
from collections import defaultdict
import warnings
import numpy as np
import os
import shutil
import stat
import subprocess
import h5py
import wave
import scipy as sp


english_charset = ['\t', '!', ' ', "'", '-', ',', '.', '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'J', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'Z', 'a', '`', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']
german_charset = ['\x87', '\x93', '\x99', '\x9b', '\x9f', '\xa1', ' ', '\xa7', '\xa9', '(', '\xad', ',', '.', '\xb3', ':', '\xc3', 'B', '\xc5', 'D', 'F', 'H', 'J', 'L', 'N', 'P', 'R', 'T', 'V', 'X', 'Z', 'b', 'd', 'f', 'h', 'j', 'l', 'n', 'p', 'r', 't', 'v', 'x', 'z', '\x80', '\x84', '\x8e', '\x96', '\x9c', '\x9e', '!', '\xa0', '\xa2', '\xa4', "'", ')', '\xa8', '\xaa', '-', '/', '1', '\xb4', '\xb6', ';', '\xbc', '?', 'A', 'C', '\xc2', 'E', '\xc4', 'G', 'I', 'K', 'M', 'O', 'Q', 'S', 'U', 'W', 'Y', 'a', 'c', '\xe2', 'e', 'g', 'i', 'k', 'm', 'o', 'q', 's', 'u', 'w', 'y']
romanian_charset = [' ', ',', '.', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\x83', '\x8e', '\x9e', '\x9f', '\xa2', '\xa3', '\xae', '\xc3', '\xc4', '\xc5']
arabic_charset = [' ', '$', '&', "'", '*', '-', '.', '<', '>', 'A', 'D', 'E', 'F', 'H', 'K', 'N', 'S', 'T', 'Y', 'Z', '^', 'a', 'b', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z', '|', '}', '~']
french_charset = [' ', "'", ',', '-', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xa0', '\xa2', '\xa7', '\xa8', '\xa9', '\xaa', '\xab', '\xae', '\xaf', '\xb4', '\xb9', '\xbb', '\xbc', '\xbf', '\xc3', '\xef']
bangla_charset = [' ', "'", '.', ';', 'A', 'D', 'H', 'I', 'J', 'N', 'R', 'S', 'T', 'U', 'Y', '`', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'y', '~']

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

# Convenience function to reuse the defined env
def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=False):
    """
    Print and execute command on system
    """
    for line in execute(cmd, shell=shell):
        print(line, end="")


# As originally seen in sklearn.utils.extmath
# Credit to the sklearn team
def _incremental_mean_and_var(X, last_mean=.0, last_variance=None,
                              last_sample_count=0):
    """Calculate mean update and a Youngs and Cramer variance update.
    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    last_mean : array-like, shape: (n_features,)
    last_variance : array-like, shape: (n_features,)
    last_sample_count : int
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
        If None, only mean is computed
    updated_sample_count : int
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = X.sum(axis=0)

    new_sample_count = X.shape[0]
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = X.var(axis=0) * new_sample_count
        if last_sample_count == 0:  # Avoid division by 0
            updated_unnormalized_variance = new_unnormalized_variance
        else:
            last_over_new_count = last_sample_count / new_sample_count
            last_unnormalized_variance = last_variance * last_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance +
                new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


class masked_synthesis_sequence_iterator(object):
    def __init__(self, filename_list, minibatch_size,
                 start_index=0,
                 stop_index=np.inf,
                 normalized=True,
                 normalization_file="default",
                 itr_type="unaligned_text",
                 class_set="english_chars",
                 extra_options="lower",
                 extra_ids=None,
                 randomize=False,
                 random_state=None):
        self.minibatch_size = minibatch_size
        self.normalized = normalized
        self.extra_options = extra_options

        n_files = len(filename_list)
        if start_index != 0:
            if start_index < 0:
                start_index = n_files + start_index
            elif start_index < 1:
                start_index = int(n_files * start_index)
            else:
                start_index = start_index

        if stop_index != np.inf:
            if stop_index < 0:
                start_index = n_files + stop_index
            elif stop_index < 1:
                stop_index = int(n_files * stop_index)
            else:
                stop_index = stop_index
        else:
            stop_index = None

        if extra_ids is not None:
            assert len(extra_ids) == len(filename_list)
            sort_idx = sorted(range(len(filename_list)),
                              key=filename_list.__getitem__)
            extra_ids = [extra_ids[i] for i in sort_idx]
            filename_list = [filename_list[i] for i in sort_idx]
        else:
            filename_list = sorted(filename_list)
            filename_list = filename_list[start_index:stop_index]
        if (len(filename_list) % minibatch_size) != 0:
            new_len = len(filename_list) - len(filename_list) % minibatch_size
            filename_list = filename_list[:new_len]
            if extra_ids is not None:
                extra_ids = extra_ids

        self.start_index = start_index
        self.stop_index = stop_index
        self.filename_list = filename_list
        self.extra_ids = extra_ids

        self.file_count = len(filename_list)
        self.n_audio_features = 63

        self.class_set = class_set

        if normalized:
            # should give a unique hash for same files and start, stop index
            hashi = hash(str(start_index) + "_" + str(stop_index))
            for f in sorted(filename_list):
                hashi ^= hash(f)
            stats_file_name = "_stored_stats_%s.npz" % hashi
            if os.path.exists(stats_file_name):
                ss = np.load(stats_file_name)
                audio_normalization_mean = ss["audio_normalization_mean"]
                audio_normalization_std = ss["audio_normalization_std"]
                audio_sample_count = ss["audio_sample_count"]
                audio_min = ss["audio_min"]
                audio_max = ss["audio_max"]
                file_count = ss["file_count"]
                file_sample_lengths = ss["file_sample_lengths"]
            else:
                print("Calculating statistics")
                # get iterator length and normalization constants
                audio_normalization_mean = None
                audio_normalization_std = None
                audio_sample_count = np.zeros((1,))
                audio_min = None
                audio_max = None
                file_count = np.zeros((1,))
                file_sample_lengths = np.zeros((len(filename_list),))
                for n, f in enumerate(filename_list):
                    # continuous vs discrete check?
                    # not needed, for now...
                    print("Loading file %i" % (n + 1))
                    a = np.load(f)
                    af = a["audio_features"]
                    if af.shape[0] < 1:
                        continue
                    file_sample_lengths[n] = len(af)

                    if n == 0:
                        audio_normalization_mean = np.mean(af, axis=0)
                        audio_normalization_std = np.std(af, axis=0)
                        audio_sample_count[0] = len(af)

                        audio_min = np.min(af, axis=0)
                        audio_max = np.max(af, axis=0)
                    else:
                        aumean, austd, aucount = _incremental_mean_and_var(
                            af, audio_normalization_mean, audio_normalization_std,
                            audio_sample_count[0])

                        audio_normalization_mean = aumean
                        audio_normalization_std = austd
                        audio_sample_count[0] = aucount

                        audio_min = np.minimum(af.min(axis=0), audio_min)
                        audio_max = np.maximum(af.max(axis=0), audio_max)

                    file_count[0] = n + 1
                save_dict = {"audio_normalization_mean": audio_normalization_mean,
                             "audio_normalization_std": audio_normalization_std,
                             "audio_sample_count": audio_sample_count,
                             "audio_min": audio_min,
                             "audio_max": audio_max,
                             "file_count": file_count,
                             "file_sample_lengths": file_sample_lengths}
                np.savez_compressed(stats_file_name, **save_dict)

            if normalization_file != "default":
                raise ValueError("Not yet supporting other placement of norm file")
            # This is assumes all the files are in the same directory...
            subdir = "/".join(filename_list[0].split("/")[:-2])
            norm_info_dir = subdir + "/norm_info/"
            audio_norm_file = "norm_info_mgc_lf0_vuv_bap_%s_MVN.dat" % str(self.n_audio_features)
            with open(norm_info_dir + audio_norm_file) as fid:
                cmp_info = np.fromfile(fid, dtype=np.float32)
            cmp_info = cmp_info.reshape((2, -1))
            audio_norm = cmp_info

        self.audio_mean = audio_norm[0, ]
        self.audio_std = audio_norm[1, ]

        self.slice_start_ = start_index
        self.file_sample_lengths = file_sample_lengths
        if random_state is not None:
            self.random_state = random_state
        self.randomize = randomize
        if self.randomize and random_state is None:
            raise ValueError("random_state must be given for randomize=True")

        def reorder_assign():
            # reorder files according to length - shortest to longest
            s_to_l = np.argsort(self.file_sample_lengths)
            reordered_filename_list = []
            for i in s_to_l:
                reordered_filename_list.append(self.filename_list[i])

            if self.extra_ids is not None:
                reordered_extra_ids = []
                for i in s_to_l:
                    reordered_extra_ids.append(self.extra_ids[i])

            # THIS TRUNCATES! Should already be handled by truncation in
            # filename splitting, but still something to watch out for
            reordered_splitlist = list(zip(*[iter(reordered_filename_list)] * self.minibatch_size))

            random_index = list(range(len(reordered_splitlist)))

            if self.randomize:
                self.random_state.shuffle(random_index)
            reordered_splitlist = [reordered_splitlist[i] for i in random_index]
            reordered_filename_list = [item for sublist in reordered_splitlist
                                       for item in sublist]
            self.filename_list = reordered_filename_list

            if self.extra_ids is not None:
                reordered_ids_splitlist = list(zip(*[iter(reordered_extra_ids)] * self.minibatch_size))
                reordered_ids_splitlist = [reordered_ids_splitlist[i] for i in random_index]
                reordered_extra_ids = [item for sublist in reordered_ids_splitlist
                                       for item in sublist]
                self.extra_ids = reordered_extra_ids

        self._shuffle = reorder_assign
        self._shuffle()

        self.itr_type = itr_type
        allowed_itr_types =["aligned", "unaligned_phonemes", "unaligned_text"]
        if itr_type not in allowed_itr_types:
            raise AttributeError("Unknown itr_type %s, allowable types %s" % (itr_type, allowed_itr_types))

        allowed_class_sets = ["english_chars", "german_chars",
                              "romanian_chars", "arabic_chars", "french_chars",
                              "bangla_chars"]
        if self.class_set not in allowed_class_sets:
            raise ValueError("class_set argument %s not currently supported!" % class_set,
                             "Allowed types are %s" % str(allowed_class_sets))

        if self.class_set == "english_chars":
            cs = english_charset
        elif self.class_set == "german_chars":
            cs = german_charset
        elif self.class_set == "romanian_chars":
            cs = romanian_charset
        elif self.class_set == "arabic_chars":
            cs = arabic_charset
        elif self.class_set == "french_chars":
            cs = french_charset
        elif self.class_set == "bangla_chars":
            cs = bangla_charset

        if self.extra_options == "lower":
            cs = sorted(list(set([csi.lower() for csi in cs])))

        self._rlu = {k: v for k, v in enumerate(cs)}
        self._lu = {v: k for k, v in self._rlu.items()}

        def npload(f):
            d = np.load(f)
            #self._phoneme_utts = self._phoneme_utts.append(d["phonemes"])
            # FIXME: Why are there filenames in some entries of code2phone...
            # FIXME: Standardize feature making between phones and text
            if self.itr_type == "unaligned_phonemes":
                return (d["phonemes"], d["audio_features"])
            elif self.itr_type == "unaligned_text":
                txt = [di for di in str(d["text"])]
                if self.extra_options == "lower":
                    txt = [ti.lower() for ti in txt]
                if all([ti in self._lu.keys() for ti in txt]):
                    txt = np.array([self._lu[ti] for ti in txt])
                else:
                    print("WARNING: Some keys not found in lookup, skipping them!")
                    txt = np.array([self._lu[ti] for ti in txt if ti in self._lu.keys()])
                return (txt, d["audio_features"])
            else:
                return np.hstack((d["text_features"], d["audio_features"]))

        self._load_file = npload

        def rg():
            self._current_file_idx = 0

        self.n_epochs_seen_ = 0
        self.n_iterations_seen_ = 0
        self.current_file_ids_ = [""] * self.minibatch_size
        rg()
        self.reset_gens = rg

    def reset(self, internal_reset=False):
        if internal_reset:
            self.n_epochs_seen_ += 1
        self.reset_gens()
        if self.randomize:
            self._shuffle()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            out = []
            for i in range(self.minibatch_size):
                fpath = self.filename_list[self._current_file_idx]
                r = self._load_file(fpath)
                if self.extra_ids is not None:
                    r = (r[0], r[1], self.extra_ids[self._current_file_idx])
                out.append(r)
                self._current_file_idx += 1
                self.current_file_ids_[i] = fpath

            if self.itr_type == "aligned":
                raise ValueError("UNSUPPORTED")
            elif self.itr_type == "unaligned_phonemes" or self.itr_type == "unaligned_text":
                mtl = max([len(o[0]) for o in out])
                mal = max([len(o[1]) for o in out])
                text = [o[0] for o in out]
                oh_size = len(self._lu)
                aud = [o[1] for o in out]
                text_arr = np.zeros((mtl, self.minibatch_size, oh_size)).astype("float32")
                text_mask_arr = np.zeros_like(text_arr[:, :, 0])
                audio_arr = np.zeros((mal, self.minibatch_size, self.n_audio_features)).astype("float32")
                audio_mask_arr = np.zeros_like(audio_arr[:, :, 0])
                for i in range(self.minibatch_size):
                    aud_i = aud[i]
                    text_i = text[i]
                    audio_arr[:len(aud_i), i, :] = aud_i
                    audio_mask_arr[:len(aud_i), i] = 1.
                    for n, j in enumerate(text_i):
                        text_arr[n, i, j] = 1.
                    text_mask_arr[:len(text_i), i] = 1.
                self.n_iterations_seen_ += 1
                if self.extra_ids is not None:
                    ids = np.array([o[2] for o in out], dtype="int32")
                    return text_arr, audio_arr, text_mask_arr, audio_mask_arr, ids
                else:
                    return text_arr, audio_arr, text_mask_arr, audio_mask_arr
        except StopIteration:
            self.reset(internal_reset=True)
            raise StopIteration("Stop index reached")
        except IndexError:
            self.reset(internal_reset=True)
            raise StopIteration("End of file list reached")

    def transform(self, audio_features, text_features=None):
        raise ValueError("Not defined")

    def inverse_transform(self, audio_features, text_features=None):
        if text_features is not None:
            raise ValueError("NYI")
        af = audio_features
        am = self.audio_mean
        astd = self.audio_std
        if audio_features.ndim == 2:
            am = am[None, :]
            astd = astd[None, :]
            return af * astd + am
        elif audio_features.ndim == 3:
            am = am[None, None, :]
            astd = astd[None, None, :]
        return af * astd + am


class jose_masked_synthesis_sequence_iterator(object):
    def __init__(self, vctk_hdf5, minibatch_size,
                 start_index=0,
                 stop_index=np.inf,
                 randomize=False,
                 random_state=None):
        self.minibatch_size = minibatch_size
        self.start_index = start_index
        self.stop_index = stop_index

        self.n_audio_features = 63
        self.n_text_features = 420
        self.n_features = self.n_text_features + self.n_audio_features

        self.slice_start_ = start_index
        if random_state is not None:
            self.random_state = random_state
        self.randomize = randomize
        if self.randomize and random_state is None:
            raise ValueError("random_state must be given for randomize=True")

        self._text_utts = []
        self._phoneme_utts = []
        self._code2phone = None
        self._code2char = None
        self._current_idx = self.start_index

        hf = h5py.File(vctk_hdf5, 'r')
        self._file = hf
        self._n_file = len(hf['features'])
        self.phone_oh_size = 44

        if stop_index == np.inf:
            self.stop_index = self._n_file

        elif stop_index < 1:
            self.stop_index = int(stop_index * self._n_file)

        if start_index < 1:
            self.start_index = int(start_index * self._n_file)

        """
        'features'
        'features_shape_labels'
        'features_shapes'
        'full_labels'
        'full_labels_shape_labels'
        'full_labels_shapes'
        'phonemes'
        'speaker_index'
        'text'
        'unaligned_phonemes'
        """

        def load(i):
            features = self._file['features'][i]
            fshp = self._file['features_shapes'][i]
            features = features.reshape(fshp)
            phones = self._file['unaligned_phonemes'][i]
            return (phones, features)

        self._load_file = load

        def rg():
            self._current_idx = self.start_index

        rg()
        self.reset_gens = rg

    def reset(self):
        self.reset_gens()
        if self.randomize:
            self._shuffle()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            out = []
            for i in range(self.minibatch_size):
                if self._current_idx >= self.stop_index:
                    raise ValueError()
                r = self._load_file(self._current_idx)
                out.append(r)
                self._current_idx += 1

            mtl = max([len(o[0]) for o in out])
            mal = max([len(o[1]) for o in out])
            text = [o[0] for o in out]
            phone_oh_size = self.phone_oh_size
            aud = [o[1] for o in out]
            text_arr = np.zeros((mtl, self.minibatch_size, phone_oh_size)).astype("float32")
            text_mask_arr = np.zeros_like(text_arr[:, :, 0])
            audio_arr = np.zeros((mal, self.minibatch_size, self.n_audio_features)).astype("float32")
            audio_mask_arr = np.zeros_like(audio_arr[:, :, 0])
            for i in range(self.minibatch_size):
                aud_i = aud[i]
                text_i = text[i]
                audio_arr[:len(aud_i), i, :] = aud_i
                audio_mask_arr[:len(aud_i), i] = 1.
                for n, j in enumerate(text_i):
                    text_arr[n, i, j] = 1.
                text_mask_arr[:len(text_i), i] = 1.
            return text_arr, audio_arr, text_mask_arr, audio_mask_arr
        except ValueError:
            self.reset()
            raise StopIteration("End of dataset iterator reached")

# Source the tts_env_script
env_script = "tts_env.sh"
if os.path.isfile(env_script):
    command = 'env -i bash -c "source %s && env"' % env_script
    for line in execute(command, shell=True):
        key, value = line.split("=")
        # remove newline
        value = value.strip()
        os.environ[key] = value
else:
    raise IOError("Cannot find file %s" % env_script)

festdir = os.environ["FESTDIR"]
festvoxdir = os.environ["FESTVOXDIR"]
estdir = os.environ["ESTDIR"]
sptkdir = os.environ["SPTKDIR"]
# generalize to more than VCTK when this is done...

vctkdir = os.environ["VCTKDIR"]
htkdir = os.environ["HTKDIR"]
merlindir = os.environ["MERLINDIR"]

# from merlin
def load_binary_file(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    features = features[:(dimension * (features.size / dimension))]
    features = features.reshape((-1, dimension))
    return features


def array_to_binary_file(data, output_file_name):
    data = np.array(data, 'float32')
    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()


def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    frame_number = features.size / dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return  features, frame_number


def generate_merlin_wav(
        data, gen_dir=None, file_basename=None, #norm_info_file,
        do_post_filtering=True, mgc_dim=60, fl=1024, sr=16000):
    # Made from Jose's code and Merlin
    if gen_dir is None:
        gen_dir = "gen/"
    gen_dir = os.path.abspath(gen_dir) + "/"
    if file_basename is None:
        base = "tmp_gen_wav"
    else:
        base = file_basename
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)

    file_name = os.path.join(gen_dir, base + ".cmp")
    """
    fid = open(norm_info_file, 'rb')
    cmp_info = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_info = cmp_info.reshape((2, -1))
    cmp_mean = cmp_info[0, ]
    cmp_std = cmp_info[1, ]

    data = data * cmp_std + cmp_mean
    """

    array_to_binary_file(data[:8000], file_name)
    # This code was adapted from Merlin. All licenses apply

    out_dimension_dict = {'bap': 1, 'lf0': 1, 'mgc': 60, 'vuv': 1}
    stream_start_index = {}
    file_extension_dict = {
        'mgc': '.mgc', 'bap': '.bap', 'lf0': '.lf0',
        'dur': '.dur', 'cmp': '.cmp'}
    gen_wav_features = ['mgc', 'lf0', 'bap']

    dimension_index = 0
    for feature_name in out_dimension_dict.keys():
        stream_start_index[feature_name] = dimension_index
        dimension_index += out_dimension_dict[feature_name]

    dir_name = os.path.dirname(file_name)
    file_id = os.path.splitext(os.path.basename(file_name))[0]
    features, frame_number = load_binary_file_frame(file_name, 63)

    for feature_name in gen_wav_features:

        current_features = features[
            :, stream_start_index[feature_name]:
            stream_start_index[feature_name] +
            out_dimension_dict[feature_name]]

        gen_features = current_features

        if feature_name in ['lf0', 'F0']:
            if 'vuv' in stream_start_index.keys():
                vuv_feature = features[
                    :, stream_start_index['vuv']:stream_start_index['vuv'] + 1]

                for i in range(int(frame_number)):
                    if vuv_feature[i, 0] < 0.5:
                        gen_features[i, 0] = -1.0e+10  # self.inf_float

        new_file_name = os.path.join(
            dir_name, file_id + file_extension_dict[feature_name])

        array_to_binary_file(gen_features, new_file_name)

    pf_coef = 1.4
    fw_alpha = 0.58
    co_coef = 511

    sptkdir = merlindir + "tools/bin/SPTK-3.9/"
    #sptkdir = os.path.abspath("latest_features/merlin/tools/bin/SPTK-3.9") + "/"
    sptk_path = {
        'SOPR': sptkdir + 'sopr',
        'FREQT': sptkdir + 'freqt',
        'VSTAT': sptkdir + 'vstat',
        'MGC2SP': sptkdir + 'mgc2sp',
        'MERGE': sptkdir + 'merge',
        'BCP': sptkdir + 'bcp',
        'MC2B': sptkdir + 'mc2b',
        'C2ACR': sptkdir + 'c2acr',
        'MLPG': sptkdir + 'mlpg',
        'VOPR': sptkdir + 'vopr',
        'B2MC': sptkdir + 'b2mc',
        'X2X': sptkdir + 'x2x',
        'VSUM': sptkdir + 'vsum'}

    #worlddir = os.path.abspath("latest_features/merlin/tools/bin/WORLD") + "/"
    worlddir = merlindir + "tools/bin/WORLD/"
    world_path = {
        'ANALYSIS': worlddir + 'analysis',
        'SYNTHESIS': worlddir + 'synth'}

    fw_coef = fw_alpha
    fl_coef = fl

    files = {'sp': base + '.sp',
             'mgc': base + '.mgc',
             'f0': base + '.f0',
             'lf0': base + '.lf0',
             'ap': base + '.ap',
             'bap': base + '.bap',
             'wav': base + '.wav'}

    mgc_file_name = files['mgc']
    cur_dir = os.getcwd()
    os.chdir(gen_dir)

    #  post-filtering
    if do_post_filtering:
        line = "echo 1 1 "
        for i in range(2, mgc_dim):
            line = line + str(pf_coef) + " "

        pe(
            '{line} | {x2x} +af > {weight}'
            .format(
                line=line, x2x=sptk_path['X2X'],
                weight=os.path.join(gen_dir, 'weight')), shell=True)

        pe(
            '{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | '
            '{c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
            .format(
                freqt=sptk_path['FREQT'], order=mgc_dim - 1,
                fw=fw_coef, co=co_coef, mgc=files['mgc'],
                c2acr=sptk_path['C2ACR'], fl=fl_coef,
                base_r0=files['mgc'] + '_r0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{freqt} -m {order} -a {fw} -M {co} -A 0 | '
            '{c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                freqt=sptk_path['FREQT'], fw=fw_coef, co=co_coef,
                c2acr=sptk_path['C2ACR'], fl=fl_coef,
                base_p_r0=files['mgc'] + '_p_r0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{mc2b} -m {order} -a {fw} | '
            '{bcp} -n {order} -s 0 -e 0 > {base_b0}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                mc2b=sptk_path['MC2B'], fw=fw_coef,
                bcp=sptk_path['BCP'], base_b0=files['mgc'] + '_b0'), shell=True)

        pe(
            '{vopr} -d < {base_r0} {base_p_r0} | '
            '{sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
            .format(
                vopr=sptk_path['VOPR'],
                base_r0=files['mgc'] + '_r0',
                base_p_r0=files['mgc'] + '_p_r0',
                sopr=sptk_path['SOPR'],
                base_b0=files['mgc'] + '_b0',
                base_p_b0=files['mgc'] + '_p_b0'), shell=True)

        pe(
            '{vopr} -m -n {order} < {mgc} {weight} | '
            '{mc2b} -m {order} -a {fw} | '
            '{bcp} -n {order} -s 1 -e {order} | '
            '{merge} -n {order2} -s 0 -N 0 {base_p_b0} | '
            '{b2mc} -m {order} -a {fw} > {base_p_mgc}'
            .format(
                vopr=sptk_path['VOPR'], order=mgc_dim - 1,
                mgc=files['mgc'],
                weight=os.path.join(gen_dir, 'weight'),
                mc2b=sptk_path['MC2B'], fw=fw_coef,
                bcp=sptk_path['BCP'],
                merge=sptk_path['MERGE'], order2=mgc_dim - 2,
                base_p_b0=files['mgc'] + '_p_b0',
                b2mc=sptk_path['B2MC'],
                base_p_mgc=files['mgc'] + '_p_mgc'), shell=True)

        mgc_file_name = files['mgc'] + '_p_mgc'

    # Vocoder WORLD

    soprcmd1 = '{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'
    soprcmd1 = soprcmd1.format(sopr=sptk_path['SOPR'], lf0=files['lf0'], x2x=sptk_path['X2X'], f0=files['f0'])

    pe(soprcmd1, shell=True)

    soprcmd2 = '{sopr} -c 0 {bap} | {x2x} +fd > {ap}'
    soprcmd2 = soprcmd2.format(sopr=sptk_path['SOPR'], bap=files['bap'], x2x=sptk_path['X2X'], ap=files['ap'])
    pe(soprcmd2, shell=True)

    mgc2spcmd='{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
    mgc2spcmd=mgc2spcmd.format(mgc2sp=sptk_path['MGC2SP'], alpha=fw_alpha,
                               order=mgc_dim - 1, fl=fl, mgc=mgc_file_name,
                               sopr=sptk_path['SOPR'], x2x=sptk_path['X2X'],
                               sp=files['sp'])
    pe(mgc2spcmd, shell=True)

    syncmd='{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'
    syncmd=syncmd.format(synworld=world_path['SYNTHESIS'], fl=fl, sr=sr,
                         f0=files['f0'], sp=files['sp'], ap=files['ap'],
                         wav=files['wav'])
    pe(syncmd, shell=True)


    pe(
        'rm -f {ap} {sp} {f0} {bap} {lf0} {mgc} {mgc}_b0 {mgc}_p_b0 '
        '{mgc}_p_mgc {mgc}_p_r0 {mgc}_r0 {cmp} weight'.format(
            ap=files['ap'], sp=files['sp'], f0=files['f0'],
            bap=files['bap'], lf0=files['lf0'], mgc=files['mgc'],
            cmp=base + '.cmp'),
    shell=True)
    os.chdir(cur_dir)


'''
def get_reconstructions():
    features_dir = "latest_features/numpy_features/"
    for fp in os.listdir(features_dir):
        print("Reconstructing %s" % fp)
        a = np.load(features_dir + fp)
        generate_merlin_wav(a["audio_features"], "latest_features/gen",
                    file_basename=fp.split(".")[0],
                            do_post_filtering=False)
'''


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    Originally from scikits.talkbox, author Anne Archibald
    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:
            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value
    endvalue    The value to use for end='pad'
    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must "\
                          "be positive")

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            if end in ['pad','wrap']:
                b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b
        elif end == 'delay':
            s = list(a.shape)
            l_orig = l
            l += overlap
            # if l not divisible by length, pad last frame with zeros
            if l_orig % (length-overlap):
                roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            else:
                roundup = l
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)

            b[..., :(overlap)] = endvalue
            b[..., (overlap):(l_orig+overlap)] = a
            b[..., (l_orig+overlap):] = endvalue
            a = b
        else:
            raise ValueError("end has to be either 'cut', 'pad', 'wrap', or 'delay'.")

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'")
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


def simple_energy_vad(x, fs, framelen=0.02, theta_main=30, theta_min=-55):
    '''Simple energy voice activity detection algorithm based on energy
    thresholds as described in Tomi Kinnunen and Padmanabhan Rajan, "A
    practical, self-adaptive voice activity detector for speaker verification
    with noisy telephone and microphone data", ICASSP 2013, Vancouver (NOTE:
    this is the benchmark method, not the method proposed by the authors).

    By Joao Felipe Santos, as implemented in SRMRPy
    '''
    # Split signal in frames
    framelen = int(framelen * fs)
    frames = segment_axis(x, length=framelen, overlap=0, end='pad')
    frames_zero_mean = frames - frames.mean(axis=0)
    frame_energy = 10*np.log10(1/(framelen-1) * (frames_zero_mean**2).sum(axis=1) + 1e-6)
    max_energy = max(frame_energy)
    speech_presence = (frame_energy > max_energy - theta_main) & (frame_energy > theta_min)
    x_vad = np.zeros_like(x, dtype=bool)
    for idx, frame in enumerate(frames):
        if speech_presence[idx]:
            x_vad[idx*framelen:(idx+1)*framelen] = True
        else:
            x_vad[idx*framelen:(idx+1)*framelen] = False
    return x[x_vad], x_vad


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def laglead_energy_vad(x, fs, small_win_size=0.25, large_win_size=1.,
                       up_threshold=1E-4, down_threshold=-1E-3):
    '''Simple energy voice activity detection algorithm using lag-lead windows
    '''
    sz = x.itemsize
    shp = x.shape
    if len(shp) != 1:
        raise ValueError("Input should be a 1D array")
    small_win_samples = int(small_win_size * fs)
    large_win_samples = int(large_win_size * fs)
    small_overlap = rolling_window(np.abs(x), small_win_samples)
    small_mean = np.mean(small_overlap, axis=-1)
    large_overlap = rolling_window(np.abs(x), large_win_samples)
    large_mean = np.mean(large_overlap, axis=-1)
    lead_diff = large_mean - small_mean[:len(large_mean)]
    lag_diff = large_mean - small_mean[-len(large_mean):]
    ups = np.arange(len(lead_diff))[lead_diff > up_threshold]
    downs = np.arange(len(lag_diff))[lag_diff < down_threshold] + (len(small_mean) - len(large_mean))
    vad = []
    is_up = False
    ups = list(ups)
    downs = list(downs)
    for i in range(len(x)):
        if is_up:
            removing = True
            while removing:
                if len(ups) == 0:
                    break
                if ups[0] < i:
                    ups.pop(0)
                else:
                    removing = False

            if len(downs) == 0:
                vad.append(True)
                continue

            if i < downs[0]:
                vad.append(True)
            else:
                vad.append(False)
                is_up = True
        else:
            removing = True
            while removing:
                if len(downs) == 0:
                    break
                if downs[0] < i:
                    downs.pop(0)
                else:
                    removing = False

            if len(ups) == 0:
                vad.append(False)
                continue

            if i < ups[0]:
                vad.append(False)
            else:
                vad.append(True)
                is_up = True
    vad = np.array(vad)
    return x[vad], vad


def read_signal(filename, winsize):
    wf=wave.open(filename,'rb')
    n=wf.getnframes()
    str=wf.readframes(n)
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen=((int )(len(str)/2/winsize) + 1) * winsize
    signal=sp.zeros(siglen, sp.int16)
    signal[0:len(str)/2] = sp.fromstring(str,sp.int16)
    return [signal, params]

def get_frame(signal, winsize, no):
    shift=winsize/2
    start=no*shift
    end = start+winsize
    return signal[start:end]


# from jfsantos
class LTSD():
    def __init__(self,winsize,window,order):
        self.winsize = int(winsize)
        self.window = window
        self.order = order
        self.amplitude = {}

    def get_amplitude(self,signal,l):
        if self.amplitude.has_key(l):
            return self.amplitude[l]
        else:
            amp = sp.absolute(sp.fft(get_frame(signal, self.winsize,l) * self.window))
            self.amplitude[l] = amp
            return amp

    def compute_noise_avg_spectrum(self, nsignal):
        windownum = int(len(nsignal)//(self.winsize//2) - 1)
        avgamp = np.zeros(self.winsize)
        for l in range(windownum):
            avgamp += sp.absolute(sp.fft(get_frame(nsignal, self.winsize,l) * self.window))
        return avgamp/float(windownum)

    def compute(self,signal):
        self.windownum = int(len(signal)//(self.winsize//2) - 1)
        ltsds = np.zeros(self.windownum)
        #Calculate the average noise spectrum amplitude based 20 frames in the head parts of input signal.
        self.avgnoise = self.compute_noise_avg_spectrum(signal[0:self.winsize*20])**2
        for l in xrange(self.windownum):
            ltsds[l] = self.ltsd(signal,l,5)
        return ltsds

    def ltse(self,signal,l,order):
        maxamp = np.zeros(self.winsize)
        for idx in range(l-order,l+order+1):
            amp = self.get_amplitude(signal,idx)
            maxamp = np.maximum(maxamp,amp)
        return maxamp

    def ltsd(self,signal,l,order):
        if l < order or l+order >= self.windownum:
            return 0
        return 10.0*np.log10(np.sum(self.ltse(signal,l,order)**2/self.avgnoise)/float(len(self.avgnoise)))


def ltsd_vad(x, fs, threshold=9, winsize=8192):
    window = sp.hanning(winsize)
    ltsd = LTSD(winsize, window, 5)
    s_vad = ltsd.compute(x)
    # LTSD is 50% overlap, so each "step" covers 4096 samples
    # +1 to cover the extra edge window
    n_samples = int(((len(s_vad) + 1) * winsize) // 2)
    time_s = n_samples / float(fs)
    time_points = np.linspace(0, time_s, len(s_vad))
    time_samples = (fs * time_points).astype(np.int32)
    time_samples = time_samples
    f_vad = np.zeros_like(x, dtype=np.bool)
    offset = winsize
    for n, (ss, es) in enumerate(zip(time_samples[:-1], time_samples[1:])):
        sss = ss - offset
        if sss < 0:
            sss = 0
        ses = es - offset
        if ses < 0:
            ses = 0
        if s_vad[n + 1] < threshold:
            f_vad[sss:ses] = False
        else:
            f_vad[sss:ses] = True
    f_vad[ses:] = False
    return x[f_vad], f_vad


if __name__=="__main__":
    WINSIZE=8192
    sound='sound.wav'
    signal, params = read_signal(sound,WINSIZE)
    res, vad = ltsd_vad(signal, 16000)
    """
    window = sp.hanning(WINSIZE)
    ltsd = LTSD(WINSIZE, window, 5)
    res =  ltsd.compute(signal)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(signal)
    ax.plot(np.max(signal) * (2 * vad.astype("int32") - 1))
    plt.savefig('res.png')
