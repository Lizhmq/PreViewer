import os, json
import random
import math
import numpy as np


def cntgenline(fp):
    cnt = 0
    while True:
        try:
            line = next(fp)
            dic = json.loads(line)
            if "msg" not in dic or len(dic["msg"]) == 0:
                continue
            cnt += 1
        except StopIteration:
            break
    return cnt


dirname = "../../../lzzz/processed"
files = os.listdir(dirname)
files.remove("chunk_8.jsonl")
files.remove("chunk_9.jsonl")
files = [os.path.join(dirname, file) for file in files if file.startswith("chunk") and file.endswith(".jsonl")]
fps = [open(file, "r") for file in files]
lens = np.array(list(map(cntgenline, fps)))
fps = [open(file, "r") for file in files]
print(sum(lens))