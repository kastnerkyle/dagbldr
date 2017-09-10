from collections import OrderedDict
import itertools
import re

# order of rules is very important
# this + ordered dict guarantees iteration order
def add_to(od, tups):
    for tup in tups:
        od.update({tup[0]: tup[1]})
    return od

r3 = OrderedDict()
kv3 = [("cough", "cou2f"),
       ("rough", "rou2f"),
       ("tough", "tou2f"),
       ("enough", "enou2f"),
       ("gn", "2n"),
       ("mb", "m2")]
r3 = add_to(r3, kv3)

r4 = OrderedDict()
kv4 = [("cq", "2q"),
       ("ci", "si"),
       ("ce", "se"),
       ("cy", "sy"),
       ("tch", "2ch"),
       ("c", "k"),
       ("q", "k"),
       ("x", "k"),
       ("v", "f"),
       ("dg", "2g"),
       ("tio", "sio"),
       ("tia", "sia"),
       ("d", "t"),
       ("ph", "fh"),
       ("b", "p"),
       ("sh", "s2"),
       ("z", "s")]
r4 = add_to(r4, kv4)

r6 = OrderedDict()
kv6 = [("j", "y"),
       ("^y3", "Y3"),
       ("^y", "A"),
       ("y", "3"),
       ("3gh3", "3kh3"),
       ("gh", "22"),
       ("g", "k"),
       ("s+", "S"),
       ("t+", "T"),
       ("p+", "P"),
       ("k+", "K"),
       ("f+", "F"),
       ("m+", "M"),
       ("n+", "N"),
       ("w3", "W3"),
       ("wh3", "Wh3"),
       ("w$", "3"),
       ("w", "2"),
       ("^h", "A"),
       ("h", "2"),
       ("r3", "R3"),
       ("r$", "3"),
       ("r", "2"),
       ("l3", "L3"),
       ("l$", "3"),
       ("l", "2")]
r6 = add_to(r6, kv6)


# x in dict is O(1)
vowels = {"a":None, "e":None, "i":None, "o":None, "u":None}


def caverphone(inp):
    """
    Original writeup by David Hood, with tests and Python code:
    http://caversham.otago.ac.nz/files/working/ctp150804.pdf

    See this site for more details, and related algorithms:
    http://ntz-develop.blogspot.ca/2011/03/phonetic-algorithms.html

    Example output
    Maclaverty: MKLFTA
    """
    # step 1. lower
    s1 = inp.lower()
    s = s1[::-1]

    # step 2. remove end e
    if s[0] == "e":
        s2 = ""
        is_end_e = True
        for n in range(len(s)):
            if s[n] == "e" and is_end_e:
                continue
            is_end_e = False
            s2 += s[n]
        s2 = s2[::-1] + "e"
    else:
        s2 = s1

    # step 3. tranform beginning of word
    s3 = s2
    for k in r3.keys():
        if s2[:len(k)] == k:
            s3 = r3[s2[:len(k)]] + s2[len(k):]

    # step 4. more replacements
    s4 = s3
    for k in r4.keys():
        s4 = s4.replace(k, r4[k])

    # step 5. vowel at beginning with A
    s5 = ""
    for n in range(len(s4)):
        if n == 0 and s4[n] in vowels:
            s5 += "A"
        elif s4[n] in vowels:
            s5 += "3"
        else:
            s5 += s4[n]

    # step 6. more replacements
    s6 = s5
    for k in r6.keys():
        if "^" in k:
            if k[1:] == s6[:len(k[1:])]:
                s6 = r6[k] + s6[len(k[1:]):]
        elif "$" in k:
            if k[:-1] == s6[-len(k[:-1]):]:
                s6 = s6[:-len(k[:-1])] + r6[k]
        elif "+" in k:
            s6 = re.sub(k, k.upper().replace("+", ""), s6)
        else:
            s6 = s6.replace(k, r6[k])

    # step 7. if last is 3, replace with A and remove all 2, 3
    s7 = s6
    if s7[-1] == "3":
        s7 = s7[:-1] + "A"
    s7 = s7.replace("2", "")
    s7 = s7.replace("3", "")
    return s7
