import os, json
import math
import numpy as np


def removestrange(s):
    lines = s.split("\n")
    toremove = r"\ No newline at end of file"
    lines = [line for line in lines if line != toremove]
    return "\n".join(lines)

def regularize(dic):
    dic["patch"] = removestrange(dic["patch"])
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
outfile = "genchunk_train"
# files = ["ruby_cls.jsonl", "ruby_gen.jsonl"]
files = os.listdir(dirname)
# langs = [file[:4] for file in files]
files.remove("chunk_8.jsonl")
files.remove("chunk_9.jsonl")
files = [os.path.join(dirname, file) for file in files if file.startswith("chunk") and file.endswith(".jsonl")]
fps = [open(file, "r") for file in files]
lens = np.array(list(map(cntline, fps)))
fps = [open(file, "r") for file in files]

# totl = sum(lens)
totl = 815493 #magic number
gpus = 8
breakcnt = math.ceil(totl / gpus)

outlist = []
outidx = 0
while True:
    idx = np.random.choice(range(len(lens)), 1, p=lens/np.sum(lens))[0]
    lens[idx] -= 1
    if sum(lens) == 0:      # this seems drop the last sample
        break
    fp = fps[idx]
    try:
        line = next(fp)
        line = line.encode("ascii", "ignore").decode()
        dic = json.loads(line)
        if "msg" not in dic or len(dic["msg"]) == 0:
            continue
    except:
        print("JSON decoding error.")
        continue
    outlist.append(json.dumps(dic))
    # print(len(outlist))
    if len(outlist) == breakcnt:
        with open(os.path.join(dirname, outfile + str(outidx) + ".jsonl"), "w") as fp:
            fp.write("\n".join(outlist) + "\n")
        outlist = []
        outidx += 1
        print(f"{outidx} files written.")

with open(os.path.join(dirname, outfile + str(outidx) + ".jsonl"), "w") as fp:
    fp.write("\n".join(outlist))
outlist = []
outidx += 1
print(f"{outidx} files written.")

print("Done.")
