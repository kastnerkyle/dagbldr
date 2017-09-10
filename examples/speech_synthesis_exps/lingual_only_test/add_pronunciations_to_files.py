#!/usr/bin/env python
from extras import masked_synthesis_sequence_iterator, pe
import os
import copy

import numpy as np
import theano
from theano import tensor
from simple_caverphone import caverphone
import json
import re

filedir = "/Tmp/kastner/vctk_American_speakers/norm_info/"
if not os.path.exists(filedir):
    if filedir[-1] != "/":
        fd = filedir + "/"
    else:
        fd = filedir
    os.makedirs(fd)
    #nfsdir = "/data/lisatmp4/kastner/vctk_American_speakers/norm_info/"
    sdir = "leto01:" + filedir
    cmd = "rsync -avhp %s %s" % (sdir, fd)
    pe(cmd, shell=True)

filedir = "/Tmp/kastner/vctk_American_speakers/numpy_features/"
#if not os.path.exists(filedir):
if filedir[-1] != "/":
    fd = filedir + "/"
else:
    fd = filedir
if not os.path.exists(fd):
    os.makedirs(fd)
sdir = "leto01:" + filedir
cmd = "rsync -avhp %s %s" % (sdir, fd)
pe(cmd, shell=True)

files = [filedir + fs for fs in sorted(os.listdir(filedir))]

final_files = []
final_ids = []
speaker_ids = {}
idx = 0
for f in files:
    ext = f.split("/")[-1]
    speaker_tag = ext.split("_")[0]
    if speaker_tag not in speaker_ids:
        speaker_ids[speaker_tag] = idx
        idx += 1
    final_ids.append(speaker_ids[speaker_tag])
    final_files.append(f)

files = final_files
file_ids = final_ids
assert len(files) == len(file_ids)

# new keys are
# caverphone_text
# cmudict_text
# need to get cmudict...

# get cmudict from https://github.com/hyperreality/Poetry-Tools/tree/master/poetrytools/cmudict
with open(os.path.join(os.path.dirname(__file__), 'cmudict.json')) as json_file:
    cmu = json.load(json_file)

cleaner = re.compile('[^a-zA-Z ]')
ws = re.compile('\s+')

for n, f in enumerate(files):
    print("Adding lingual features to file {} of {}".format(n, len(files)))
    d = dict(np.load(f))
    if "kwds" in d:
        d = dict(d["kwds"].item())
        np.savez_compressed(f, **d)
        continue
    new_d = copy.copy(d)
    text = str(d["text"])
    clean_text = cleaner.sub("", text)
    # also remove trailing...
    clean_text = ws.sub(" ", clean_text).strip()
    if "caverphone_text" not in d.keys():
        try:
            cvr = " ".join([caverphone(str(dd)) for dd in clean_text.split(" ")])
        except:
            from IPython import embed; embed(); raise ValueError()
        new_d["caverphone_text"] = np.array(cvr)
    if True or "cmudict_text" not in d.keys():
        words = clean_text.split(" ")
        pron = []
        for word in words:
            try:
                pron_word = "".join(cmu[word.lower()][0])
                pron.append(pron_word)
            except KeyError:
                pron.append("-")
        cmu_pron = np.array(str(" ".join(pron)))
        new_d["cmudict_text"] = cmu_pron
    np.savez_compressed(f, **new_d)
