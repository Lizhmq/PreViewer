import re, json
import random
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from transformers import RobertaTokenizer


class TextDataset(Dataset):
    def __init__(self, tokenizer, pool, args, file_path=None):
        self.examples = read_review_examples(file_path, 100)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def convert_examples_to_features(item):
    example, tokenizer, args = item
    # [1:-1] to remove <s> and </s>
    def encode_remove(tokenizer, text):
        text = tokenizer.encode(text)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        else:
            raise NotImplementedError
    prevlines = [encode_remove(tokenizer, source_str) for source_str in example.prevlines]
    afterlines = [encode_remove(tokenizer, source_str) for source_str in example.afterlines]
    lines = [encode_remove(tokenizer, source_str) for source_str in example.lines]
    labels = list(example.labels)
    inputl = len(lines)
    inputl += sum(map(len, lines))
    prev_after_len = max(len(prevlines), len(afterlines))
    left, right = 0, len(lines)
    while inputl > args.max_source_length - 2:
        if left % 2 == 0:
            left += 1
            inputl -= len(lines[left]) + 1
        else:
            right -= 1
            inputl -= len(lines[right]) + 1
    lines = lines[left:right]
    i = 0
    while inputl < args.max_source_length - 2 and i < prev_after_len:
        if i < len(prevlines):
            newl = inputl + len(prevlines[-1-i]) + 1
            if newl > args.max_source_length - 2:
                break
            lines.insert(0, prevlines[-1-i])
            labels.insert(0, -100)
            inputl = newl  # tag
        if i < len(afterlines):
            newl = inputl + len(afterlines[i]) + 1
            if newl > args.max_source_length - 2:
                break
            lines.append(afterlines[i])
            labels.append(-100)
            inputl = newl    # tag
        i += 1
    assert inputl <= args.max_source_length - 2, "Too long inputs."
    source_ids, input_labels, target_ids = [], [], []
    SPECIAL_ID = 0
    mask_idxs = random.choices(range(len(lines)), k=int(len(lines) * args.mask_rate))
    for i, (line, label) in enumerate(zip(lines, labels)):
        source_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
        input_labels.append(label)
        if i in mask_idxs:
            source_ids.append(tokenizer.mask_id)
            input_labels.append(-100)
            target_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
            target_ids.extend(line)
        else:
            source_ids.extend(line)
            input_labels.extend([-100] * len(line))
        if SPECIAL_ID < 99:     # only 0-99 ids in vocab
            SPECIAL_ID += 1
    if example.msg != "":
        target_ids.append(tokenizer.msg_id)
        target_ids.extend(encode_remove(tokenizer, example.msg))
    assert len(input_labels) == len(source_ids), "Not equal length."
    assert len(input_labels) <= args.max_source_length - 2, "Too long inputs."
    input_labels = [-100] + input_labels + [-100]
    source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
    pad_len = args.max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_id] * pad_len
    input_labels += [-100] * pad_len
    target_ids = target_ids[:args.max_target_length - 2]
    target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
    pad_len = args.max_target_length - len(target_ids)
    target_ids += [tokenizer.pad_id] * pad_len
    return ReviewFeatures(example.idx, source_ids, input_labels, target_ids)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, example_id, source_ids, target_ids, url=None):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class ReviewFeatures(object):
    def __init__(self, example_id, source_ids, source_labels, target_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_labels = source_labels
        self.target_ids = target_ids


class Example(object):
    """A single training/test example."""

    def __init__(self, idx, source, target, url=None, task="", sub_task=""):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class ReviewExample(object):
    """A single training/test example."""

    def __init__(
        self, idx, oldf, diff, msg, cmtid,
    ):
        self.idx = idx
        self.oldf = oldf
        self.diff = diff
        self.msg = msg
        self.cmtid = cmtid
        self.prevlines = []
        self.afterlines = []
        self.lines = []
        self.labels = []
        self.avail = False
        self.align_and_clean()

    def remove_space(self, line):
        rep = " \t\r"
        totallen = len(line)
        i = 0
        while i < totallen and line[i] in rep:
            i += 1
        j = totallen - 1
        while j >= 0 and line[j] in rep:
            j -= 1
        return line[i : j + 1]

    def align_and_clean(self):
        oldflines = self.oldf.split("\n")
        difflines = self.diff.split("\n")
        first_line = difflines[0]
        difflines = difflines[1:]
        regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
        matchres = re.match(regex, first_line)
        if matchres:
            startline, endline, startpos, endpos = matchres.groups()
            self.avail = True
        else:
            return
        startline, endline = int(startline) - 1, int(endline) - 1
        self.prevlines = oldflines[:startline]
        self.afterlines = oldflines[endline + 1 :]
        for line in difflines:
            if line.startswith("-"):
                self.lines.append(line[1:])
                self.labels.append(0)
            elif line.startswith("+"):
                self.lines.append(line[1:])
                self.labels.append(1)
            else:
                self.lines.append(line)
                self.labels.append(2)
        self.prevlines = [self.remove_space(line) for line in self.prevlines]
        self.afterlines = [self.remove_space(line) for line in self.afterlines]
        self.lines = [self.remove_space(line) for line in self.lines]
        self.prevlines = [line for line in self.prevlines if len(line) > 0]
        self.afterlines = [line for line in self.afterlines if len(line) > 0]
        self.lines, self.labels = list(
            zip(
                *[
                    (line, label)
                    for line, label in zip(self.lines, self.labels)
                    if len(line) > 0
                ]
            )
        )


def read_review_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    idx = 0
    with open(filename) as f:
        for line in f:
            js = json.loads(line.strip())
            example = ReviewExample(
                        idx=idx,
                        oldf=js["oldf"],
                        diff=js["patch"],
                        msg=js["msg"] if "msg" in js else "",
                        cmtid=js["cmtid"] if "cmtid" in js else "",
                    )
            if example.avail:
                examples.append(example)
                idx += 1
                if idx == data_num:
                    break
            else:
                print("Passing invalid diff.")
                
    return examples
