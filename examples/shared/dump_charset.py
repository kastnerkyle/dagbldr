import os
import sys
import numpy as np
import copy
import argparse

parser = argparse.ArgumentParser(description="Get a charset from a set of numpy features (from extract_feats.py)",
                                 epilog="Example usage: python dump_charset.py path_to_numpy_features")

parser.add_argument("numpy_features_path",
                    help="filepath for directory of numpy feature files")
parser.add_argument("--inplace_fix", "-f",
                    help="cleanup broken text, you will need to modify the script carefully for new data!",
                    action="store_true",
                    default=False,
                    required=False)


args = parser.parse_args()
inplace_fix = args.inplace_fix
numpy_features_dir = os.path.abspath(args.numpy_features_path)


if numpy_features_dir[-1] != "/":
    numpy_features_dir = numpy_features_dir + "/"

common = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
common += common.lower()
common += " "
common += "'"
# bangla specific...
#common += "~"
#common += "`"
#common += "."
#common += ";"

charset = set()
for ni, nf in enumerate([i for i in os.listdir(numpy_features_dir) if ".npz" in i]):
    fpath = numpy_features_dir + nf
    d = np.load(fpath)
    l = str(d["text"])
    if any([li not in common for li in l]):
        print("Noncommon chars in %s" % nf)
        print(l)
        l = l.replace("(2)", "`")
        l = l.replace("(3)", "```")
        #dd_orig = copy.deepcopy(dict(d))

        # Overwriting... very dangerous
        dd = dict(d)
        dd["text"] = np.array(l)
        if inplace_fix:
            np.savez(fpath, **dd)
            d = np.load(fpath)

    t = set(list(str(d["text"])))
    charset = charset | t
final_charset = sorted(list(charset))

with open("final_charset.txt", "w") as f:
    f.writelines([str(final_charset),])
