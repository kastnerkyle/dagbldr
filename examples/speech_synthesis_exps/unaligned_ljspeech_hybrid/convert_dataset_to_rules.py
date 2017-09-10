from eng_rules import hybrid_g2p
import os
import numpy as np
import copy

filedir = "/Tmp/kastner/lj_speech_hybrid_speakers/numpy_features/"
files = [filedir + fs for fs in sorted(os.listdir(filedir))]
all_chars = set()
for n, f in enumerate(files):
    print("Processing file {}".format(n))
    di = np.load(f)
    if "old_text" not in di.keys():
        d = {k: copy.deepcopy(v) for k, v in di.items()}
        d["old_text"] = d["text"]
        if len(str(d["text"])) == 0:
            print("Empty input string detected in {}".format(f))
            os.remove(f)
            continue
        try:
            r = hybrid_g2p(str(d["text"]))
        except:
            print("Error occured - check it!")
            from IPython import embed; embed(); raise ValueError()

        rules = [ri[0] for ri in r]
        prons = [ri[1] for ri in r]
        fprons = " ".join(["".join(p).replace(" ", "-") for p in prons])
        fprons = fprons.replace("<", "")
        fprons = fprons.replace(">", "")
        d["text"] = fprons
        np.savez(f, **d)
        all_chars |= set(fprons)
    else:
        all_chars |= set(str(di["text"]))
from IPython import embed; embed(); raise ValueError()
