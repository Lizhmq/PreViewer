import os, json
import random
import numpy as np


def removestrange(s):
    lines = s.split("\n")
    toremove = r"\ No newline at end of file"
    lines = [line for line in lines if line != toremove]
    return "\n".join(lines)

def regurize(dic):
    dic["patch"] = removestrange(dic["patch"])
    diff = dic["patch"]
    oldf = dic["oldf"]
    if not diff.startswith("@@ "):
        difflines = diff.split("\n")[1:-1]   # remove last empty line
        removecnt = sum([1 if not line.startswith("+") else 0 for line in difflines])
        addcnt = sum([1 if not line.startswith("-") else 0 for line in difflines])
        oldcnt = removecnt
        newcnt = addcnt
        difflines = ["" if len(line) == 0 else line[1:] for line in difflines if len(line) == 0 or line[0] != "+"]
        newdiff = "\n".join(difflines)
        idx = oldf.find(newdiff)
        if idx == -1:
            print("not found")
            # open("diff.txt", "w").write(diff)
            # open("oldf.txt", "w").write(oldf)
            # open("newf.txt", "w").write(newdiff)
            # exit(0)
            return {}
        else:
            # open("diff.txt", "w").write(diff)
            # open("oldf.txt", "w").write(oldf)
            # open("newf.txt", "w").write(newdiff)
            linenum = oldf[:idx].count("\n") + 1
            prefix = f"@@ -{linenum},{oldcnt} +{linenum},{newcnt} @@"
            dic["patch"] = prefix + diff
            # print(prefix)
            # exit(0)
    return dic

def cntline(fp):
    cnt = 0
    while True:
        try:
            next(fp)
            cnt += 1
        except StopIteration:
            break
    return cnt


dirname = "../../../lzzz/processed"
outfile = "chunk_"
# files = ["ruby_cls.jsonl", "ruby_gen.jsonl"]
files = os.listdir(dirname)
langs = [file[:4] for file in files]
files = [os.path.join(dirname, file) for file in files if file.endswith(".jsonl")]
fps = [open(file, "r") for file in files]
lens = np.array(list(map(cntline, fps)))
fps = [open(file, "r") for file in files]

outlist = []
breakcnt = 50000
outidx = 0
while True:
    idx = np.random.choice(range(len(lens)), 1, p=lens/np.sum(lens))[0]
    lens[idx] -= 1
    if sum(lens) == 0:
        break
    fp, lang = fps[idx], langs[idx]
    try:
        dic = json.loads(next(fp))
    except:
        print("JSON decoding error.")
        continue
    dic = regurize(dic)
    if dic == {}:
        continue
    dic["lang"] = lang
    outlist.append(json.dumps(dic))
    # print(len(outlist))
    if len(outlist) == breakcnt:
        with open(os.path.join(dirname, outfile + str(outidx) + ".jsonl"), "w") as fp:
            fp.write("\n".join(outlist))
        outlist = []
        outidx += 1
        print(f"{outidx} files written.")

with open(os.path.join(dirname, outfile + str(outidx) + ".jsonl"), "w") as fp:
    fp.write("\n".join(outlist))
outlist = []
outidx += 1
print(f"{outidx} files written.")

print("Done.")
