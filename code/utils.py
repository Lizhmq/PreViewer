import re, json
import os, random
import torch, logging
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import T5Tokenizer
from transformers import RobertaTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



class MyTokenizer(object):
    """
    Wrapper for ByteLevelBPETokenizer
    """
    def __init__(self, vocab=None, merges=None, **kwargs):
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges, **kwargs)
        self.update_id2token()

    @staticmethod
    def from_pretrained(path):
        vocabp = os.path.join(path, "vocab.json")
        mergesp = os.path.join(path, "merges.txt")
        mytoken = MyTokenizer(vocabp, mergesp)
        return mytoken

    def update_id2token(self):
        vocab = self.tokenizer.get_vocab()
        self.id2token = {vocab[token]: token for token in vocab}

    def add_special_tokens(self, dic):
        for values in dic.values():
            self.tokenizer.add_special_tokens(values)
        self.update_id2token()

    def convert_ids_to_tokens(self, ids):
        vocab = self.id2token
        return [vocab[i] for i in ids]
    
    def decode(self, ids, **kwargs):    ##### to be update
        tokens = self.convert_ids_to_tokens(ids)
        return " ".join(tokens)

    def encode(self, text, **kwargs):
        text = text.encode("ascii", errors="ignore").decode("ascii")
        return self.tokenizer.encode(text).ids

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def __len__(self):
        return len(self.tokenizer.get_vocab())


class TextDataset(Dataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.cnt = 0
        self.tokenizer = tokenizer
        if isinstance(tokenizer, MyTokenizer):
            tokenizer_type = "mytok"
        elif isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".exps")
        # savep = "/home/v-zhuoli1/lzzz/processed/chunk_25.exps"
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(file_path, samplenum, tokenizer)
            logger.info(f"Tokenize examples: {file_path}")
            examples = pool.map(self.tokenize, \
                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)
        logger.info("Convert examples to features...")
        self.set_start_end_ids(examples)
        self.featss = pool.map(self.convert_examples_to_features, \
            [(example, tokenizer, args) for example in examples])
        self.feats = [feat for feats in self.featss for feat in feats]  # expand the lists

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

    def reset_len(self, data_len):
        assert len(self.feats) >= data_len
        self.feats = self.feats[:data_len]

    def set_start_end_ids(self, examples):
        for example in examples:
            labels = example.labels
            start_id = 0
            end_id = len(labels) - 1
            for i, label in enumerate(labels):
                if label != -100:               # find the first label
                    start_id = i
                    break
            for i in range(len(labels) - 1, -1, -1):
                label = labels[i]
                if label != -100:
                    end_id = i
                    break
            example.start_id = start_id
            example.end_id = end_id

    def tokenize(self, item):
        example, tokenizer, args = item
        example.input = self.encode_remove(tokenizer, example.input, args)
        e0id = tokenizer.special_dict["<e0>"]
        inputs = " ".join(str(id) for id in example.input)
        lines = inputs.split(" " + str(e0id) + " ")
        lines = [
            [int(v) for v in line.split(" ") if len(v) > 0] for line in lines
        ]
        lens = [len(line) for line in lines]
        if 0 in lens:
            logger.info("Warning: empty line in an example.")
        lens = list(map(len, lines))
        curlen = len(lens) + sum(lens)
        left, right = 0, len(lines)
        while curlen > args.max_source_length - 2:
            if left % 2 == 0:
                curlen -= 1 + len(lines[left])
                left += 1
            else:
                right -= 1
                curlen -= 1 + len(lines[right])
        lines = lines[left:right]
        labels = example.labels[left:right]
        assert len(lines) + sum(map(len, lines)) <= args.max_source_length - 2, "Too long inputs in TextDataset.tokenize."
        if len(lines) != len(labels):
            logger.info("Not equal length in TextDataset.tokenize.")
            lines = lines[:len(labels)]
            labels = labels[:len(lines)]
        example.lines = lines
        example.labels = labels
        example.msg = self.encode_remove(tokenizer, example.msg, args)
        return example

    def convert_examples_to_features(self, item):
        example, _, _ = item
        if len(example.msg) > 0:
            exs = []
            for _ in range(2):  # up sampling
                if random.random() < 5 / 6:
                    exs.append(self.msg_example(item))
                elif random.random() < 0.5:
                    exs.append(self.encoder_example(item))
                else:
                    exs.append(self.decoder_example(item))
            return exs
        if random.random() < 0.5:
            return [self.encoder_example(item)]
        return [self.decoder_example(item)]

    def encoder_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        target_ids = [tokenizer.pad_id] * args.max_target_length
        source_ids, input_labels = [], []
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
                input_labels.append(-100)
            if label != -100:       # only insert special tokens at diffs, not context
                source_ids.append(tokenizer.mask_id)
                input_labels.append(label)
            source_ids.extend(line)
            input_labels.extend([-100] * len(line))
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
                input_labels.append(-100)
        assert len(input_labels) == len(source_ids), "Not equal length."
        assert len(input_labels) <= args.max_source_length, f"Too long inputs: {len(input_labels)}."
        source_ids = source_ids[:args.max_source_length - 2]
        input_labels = input_labels[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        input_labels = [-100] + input_labels + [-100]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        input_labels += [-100] * pad_len

        new_input_labels = []
        map_dict = {0: tokenizer.del_id, 1: tokenizer.add_id, 2: tokenizer.keep_id}
        for label in input_labels:
            if label == -100:
                new_input_labels.append(-100)
            else:
                new_input_labels.append(map_dict[label])
        input_labels = new_input_labels
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(input_labels) == args.max_source_length, "Not equal length."
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="label")

    def decoder_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels

        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        SPECIAL_ID = 0
        mask_idxs = random.choices(range(len(lines)), k=int(len(lines) * args.mask_rate))
        id_dict = {0: tokenizer.del_id, 1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label in id_dict:
                source_ids.append(id_dict[label])
            if i in mask_idxs:
                source_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.extend(line)
                if SPECIAL_ID < 99:     # only 0-99 ids in vocab
                    SPECIAL_ID += 1
            else:
                source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 2]
        target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="line")

    def msg_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        id_dict = {0: tokenizer.del_id, 1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label != -100:
                source_ids.append(id_dict[label])
            source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        target_ids.append(tokenizer.msg_id)
        target_ids.extend(example.msg)
        assert len(source_ids) <= args.max_source_length, f"Too long inputs: {len(source_ids)}."
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 2]
        target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="msg")

    def encode_remove(self, tokenizer, text, args):
        text = tokenizer.encode(text, max_length=args.max_source_length, truncation=True)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        elif type(tokenizer) == MyTokenizer:
            return text
        else:
            raise NotImplementedError


class CommentGenDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, MyTokenizer):
            tokenizer_type = "mytok"
        elif isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".exps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(file_path, samplenum, tokenizer)
            logger.info(f"Tokenize examples: {file_path}")
            examples = pool.map(self.tokenize, \
                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)
        logger.info("Convert examples to features...")
        self.set_start_end_ids(examples)
        self.feats = pool.map(self.convert_examples_to_features, \
            [(example, tokenizer, args) for example in examples])
        self.feats = [feat for feat in self.feats if feat is not None]

    def convert_examples_to_features(self, item):
        example, tokenizer, args = item
        if len(example.msg) == 0:
            return None
        return self.msg_example(item)




class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, example_id, source_ids, target_ids, url=None):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class ReviewFeatures(object):
    def __init__(self, example_id, source_ids, source_labels, target_ids, type):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_labels = source_labels
        self.target_ids = target_ids
        assert type in ("label", "line", "msg")
        self.type = type

class ReviewExample(object):
    """A single training/test example."""

    def __init__(
        self, idx, oldf, diff, msg, cmtid, max_len
    ):
        self.idx = idx      # idx is useless yet
        self.oldf = oldf
        self.diff = diff
        self.msg = msg
        self.cmtid = cmtid
        self.max_len = max_len
        self.prevlines = []
        self.afterlines = []
        self.lines = []
        self.labels = []
        self.avail = False
        self.input = ""
        self.align_and_clean()
        self.postprocess()

    def postprocess(self):
        if not self.avail:
            return
        # Warning: lines is not self.lines
        # lines for rough length estimation
        lines = [source_str.split() for source_str in self.lines]
        inputl = len(lines) # line tag
        inputl += sum(map(len, lines))
        left, right = 0, len(lines)
        while inputl > self.max_len:
            if left % 2 == 0:
                inputl -= len(lines[left]) + 1
                left += 1
            else:
                right -= 1
                inputl -= len(lines[right]) + 1
        lines = lines[left:right]
        self.lines = self.lines[left:right]
        self.labels = self.labels[left:right]
        prevlines = self.prevlines
        afterlines = self.afterlines
        prev_after_len = max(len(prevlines), len(afterlines))
        i = 0
        while inputl < self.max_len and i < prev_after_len:
            if i < len(prevlines):
                newl = inputl + len(prevlines[-1-i].split()) + 1
                if newl > self.max_len:
                    break
                self.lines.insert(0, prevlines[-1-i])
                self.labels.insert(0, -100)
                inputl = newl  # tag
            if i < len(afterlines):
                newl = inputl + len(afterlines[i].split()) + 1
                if newl > self.max_len:
                    break
                self.lines.append(afterlines[i])
                self.labels.append(-100)
                inputl = newl    # tag
            i += 1
        assert inputl <= self.max_len, "Too long inputs."
        assert len(self.lines) == len(self.labels), "Not equal length."
        self.input = "<e0>".join(self.lines)
        self.prevlines, self.lines, self.afterlines = [], [], []

    def remove_space_clean(self, line):
        rep = " \t\r"
        totallen = len(line)
        i = 0
        while i < totallen and line[i] in rep:
            i += 1
        j = totallen - 1
        while j >= 0 and line[j] in rep:
            j -= 1
        line = line[i : j + 1]
        # keep ascii chars only
        line = line.encode("ascii", errors="ignore").decode("ascii")
        return line

    def align_and_clean(self):
        oldflines = self.oldf.split("\n")
        difflines = self.diff.split("\n")
        first_line = difflines[0]
        difflines = difflines[1:]
        difflines = [line for line in difflines if line != "\ No newline at end of file"]
        regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
        matchres = re.match(regex, first_line)
        if matchres:
            startline, rangelen, startpos, endpos = matchres.groups()
            self.avail = True
        else:
            self.avail = False
            return
        startline, rangelen = int(startline) - 1, int(rangelen)
        endline = startline + rangelen
        self.prevlines = oldflines[:startline]
        self.afterlines = oldflines[endline:]
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
        self.prevlines = [self.remove_space_clean(line) for line in self.prevlines]
        self.afterlines = [self.remove_space_clean(line) for line in self.afterlines]
        self.lines = [self.remove_space_clean(line) for line in self.lines]
        self.msg = self.remove_space_clean(self.msg)
        self.prevlines = [line for line in self.prevlines if len(line) > 0]
        self.afterlines = [line for line in self.afterlines if len(line) > 0]
        # print("\n".join(self.prevlines))
        # print("\n\n\n\n")
        # print("\n".join(self.lines))
        # print("\n\n\n\n")
        # print("\n".join(self.afterlines))
        # print("\n\n\n\n")
        assert len(self.lines) == len(self.labels), "Not equal length in align."
        topack = list(
            zip(
                *[
                    (line, label)
                    for line, label in zip(self.lines, self.labels)
                    if len(line) > 0
                ]
            )
        )
        if topack == []:
            self.avail = False
            return
        else:
            self.lines, self.labels = topack
        # tuple->list, convenient for later operation
        self.lines = list(self.lines)
        self.labels = list(self.labels)


def read_review_examples(filename, data_num=-1, tokenizer=None):
    """Read examples from filename."""
    examples = []
    idx = 0
    with open(filename) as f:
        for line in f:
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            maxl = 128
            if isinstance(tokenizer, MyTokenizer):
                maxl = 178
            example = ReviewExample(
                        idx=idx,
                        oldf=js["oldf"],
                        diff=js["patch"],
                        msg=js["msg"] if "msg" in js else "",
                        cmtid=js["cmtid"] if "cmtid" in js else "",
                        max_len=maxl
                    )
            if example.avail:
                examples.append(example)
                idx += 1
                if idx == data_num:
                    break
            else:
                # print(f"Passing {idx} because of invalid diff.")
                idx += 1 
                if idx == data_num:
                    break
                
    return examples
