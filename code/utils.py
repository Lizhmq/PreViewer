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

    @staticmethod
    def from_pretrained(path):
        vocabp = os.path.join(path, "vocab.json")
        mergesp = os.path.join(path, "merges.txt")
        mytoken = MyTokenizer(vocabp, mergesp)
        return mytoken

    def add_special_tokens(self, dic):
        self.tokenizer.add_special_tokens(dic.values())

    def convert_ids_to_tokens(self, ids):
        vocab = self.tokenizer.get_vocab()
        return [vocab[i] for i in ids]

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text).ids



class TextDataset(Dataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        savep = file_path.replace(".jsonl", ".exps")
        # savep = "/home/v-zhuoli1/lzzz/processed/chunk_25.exps"
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(file_path, samplenum)
            logger.info(f"Tokenize examples: {file_path}")
            examples = pool.map(self.tokenize, \
                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)
        logger.info("Convert examples to features...")
        self.feats = pool.map(self.convert_examples_to_features, \
            [(example, tokenizer, args) for example in examples])

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

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
        assert 0 not in lens, "Empty line."
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
        assert len(lines) == len(labels), "Not equal length in TextDataset.tokenize."
        example.lines = lines
        example.labels = labels
        example.msg = self.encode_remove(tokenizer, example.msg, args)
        return example

    def convert_examples_to_features(self, item):
        """
        Mask and padding.
        """
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels

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
        # if len(example.msg) > 0:
        target_ids.append(tokenizer.msg_id)
        target_ids.extend(example.msg)
        assert len(input_labels) == len(source_ids), "Not equal length."
        assert len(input_labels) <= args.max_source_length - 2, f"Too long inputs: {len(input_labels)}."
        input_labels = [-100] + input_labels + [-100]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        input_labels += [-100] * pad_len
        target_ids = target_ids[:args.max_target_length - 2]
        target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(input_labels) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids)

    def encode_remove(self, tokenizer, text, args):
        text = tokenizer.encode(text, max_length=args.max_source_length - 2, truncation=True)
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
        savep = file_path.replace(".jsonl", ".exps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(file_path, samplenum)
            logger.info(f"Tokenize examples: {file_path}")
            examples = pool.map(self.tokenize, \
                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)
        logger.info("Convert examples to features...")
        self.feats = pool.map(self.convert_examples_to_features, \
            [(example, tokenizer, args) for example in examples])
        self.feats = [feat for feat in self.feats if feat is not None]
    
    def convert_examples_to_features(self, item):
        """
        Mask and padding.
        """
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels

        if len(example.msg) == 0:
            return None

        source_ids, input_labels, target_ids = [], [], []
        SPECIAL_ID = 0
        ADD_ID = tokenizer.encode("+")[0]
        SUB_ID = tokenizer.encode("-")[0]
        for i, (line, label) in enumerate(zip(lines, labels)):
            source_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
            input_labels.append(label)
            if label == 1:
                source_ids.append(ADD_ID)
                input_labels.append(-100)
            elif label == 0:
                source_ids.append(SUB_ID)
                input_labels.append(-100)
            source_ids.extend(line)
            input_labels.extend([-100] * len(line))
            if SPECIAL_ID < 99:     # only 0-99 ids in vocab
                SPECIAL_ID += 1
        assert len(example.msg) > 0, "Empty message."
        target_ids.append(tokenizer.msg_id)
        target_ids.extend(example.msg)
        assert len(input_labels) == len(source_ids), "Not equal length."
        # assert len(input_labels) <= args.max_source_length - 2, f"Too long inputs: {len(input_labels)}."
        input_labels = [-100] + input_labels[:args.max_source_length - 2] + [-100]
        source_ids = [tokenizer.bos_id] + source_ids[:args.max_source_length - 2] + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        input_labels += [-100] * pad_len
        target_ids = target_ids[:args.max_target_length - 2]
        target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(input_labels) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
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


class ReviewExample(object):
    """A single training/test example."""

    def __init__(
        self, idx, oldf, diff, msg, cmtid,
    ):
        self.idx = idx      # idx is useless yet
        self.oldf = oldf
        self.diff = diff
        self.msg = msg
        self.cmtid = cmtid
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
        while inputl > 512 - 2:
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
        while inputl < 512 - 2 and i < prev_after_len:
            if i < len(prevlines):
                newl = inputl + len(prevlines[-1-i].split()) + 1
                if newl > 512 - 2:
                    break
                self.lines.insert(0, prevlines[-1-i])
                self.labels.insert(0, -100)
                inputl = newl  # tag
            if i < len(afterlines):
                newl = inputl + len(afterlines[i].split()) + 1
                if newl > 512 - 2:
                    break
                self.lines.append(afterlines[i])
                self.labels.append(-100)
                inputl = newl    # tag
            i += 1
        assert inputl <= 512 - 2, "Too long inputs."
        assert len(self.lines) == len(self.labels), "Not equal length."
        # self.lines is about 512 now, let's tokenize it
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
        line = "".join([ch for ch in line if ord(ch) < 128])
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


def read_review_examples(filename, data_num=-1):
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
                # print(f"Passing {idx} because of invalid diff.")
                idx += 1 
                if idx == data_num:
                    break
                
    return examples
