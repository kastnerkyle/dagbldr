# *-* encoding: utf-8 *-*
import numpy as np
import os
import shutil

supertest_files_path = "supertest/numpy_features/"


minibatch_size = 8

"""
text_replacements = ["domnul Trump vÄ rugÄm sÄ dÄrÃ¢me acest zid.",]
                     # mister Trump please tear down this wall.
text_replacements *= 8
"""
# Copy pasted from Google Translate
text_replacements = ["domnul Trump vÄ rugÄm sÄ dÄrÃ¢me acest zid.",
                     # mister Trump please tear down this wall.
                     "BunÄ ziua lume este frumos sÄ vÄ Ã®ntÃ¢mpine.",
                     # Hello world it is nice to greet you.
                     "Nu am un om, ci un om-maÈinÄ.",
                     # I have not a human, but a human machine.
                     "fÄrÄ gaurÄ am nici un suflet.",
                     # without a hole I have no soul.
                     "Am sÄpat Ã®n, Èi nu voi schimba.",
                     # I'm dug in, and I'll never change.
                     "probe este egal ca probe nu fie.",
                     # samples is as samples do be.
                     "Iron Maiden concerteazÄ Ã®n Ã®ntreaga lume.",
                     # Iron Maiden is touring worldwide.
                     "In nici un caz va atÃ¢rna un maniac afarÄ, la barul nostru."]
                     # No way will a maniac hang out at our bar.

np_files = os.listdir(supertest_files_path)
np_files = [supertest_files_path + f for f in np_files if ".bak" not in f]
np_files = sorted(np_files)

for n, npf in enumerate(np_files):
    bak = ".".join(npf.split(".")[:-1]) + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(npf, bak)
    if n >= minibatch_size:
        os.remove(npf)

np_files = [f for f in np_files if ".bak" not in f]
np_files = sorted(np_files)

if len(text_replacements) != len(np_files):
    raise ValueError("Not enough text replacement strings")

for n, npf in enumerate(np_files):
    res = {}
    a = np.load(npf)
    r = {k: v for k, v in a.items()}
    r["text"] = np.array(text_replacements[n])
    r["durations"] *= 0
    r["audio_features"] *= 0
    np.savez(npf, **r)
