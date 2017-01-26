import os
import sys
import numpy as np

if len(sys.argv) < 2:
    raise ValueError("Need path to numpy_features")

numpy_features_dir = os.path.abspath(sys.argv[1])

if numpy_features_dir[-1] != "/":
    numpy_features_dir = numpy_features_dir + "/"

charset = set()
for nf in [i for i in os.listdir(numpy_features_dir) if ".npz" in i]:
    fpath = numpy_features_dir + nf
    d = np.load(fpath)
    t = set(list(str(d["text"])))
    charset = charset | t
final_charset = sorted(list(charset))

with open("final_charset.txt", "w") as f:
    f.writelines([str(final_charset),])
