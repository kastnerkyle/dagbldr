import sys
import os
import numpy as np

dirname = sys.argv[1]
if dirname[-1] != "/":
    dirname = dirname + "/"

files = [f for f in os.listdir(dirname) if f[-4:] == ".npz"]

all_chars = set()
all_phonemes = set()
for fi in files:
    d = np.load(dirname + fi)
    print(str(d["text"]))
    chars_in_fi = set([di for di in str(d["text"])])
    phones_in_fi = set([pi for pi in str(d["phonemes"])])
    all_chars = all_chars | chars_in_fi
    all_phonemes = all_phonemes | phones_in_fi
    print("Processed %s" % fi)

with open("charset.txt", "w") as f:
    f.write(repr(list(all_chars)))

with open("phoneset.txt", "w") as f:
    f.write(repr(list(all_phonemes)))
