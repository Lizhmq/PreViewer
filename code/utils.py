import json



def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )



class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class ReviewExample(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 oldf,
                 diff,
                 msg,
                 cmtid,
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
        self.align_and_clean()
    
    def remove_space(self, line):
        rep = " \t\r\n"
        totallen = len(line)
        i = 0
        while i < totallen and line[i] in rep:
            i += 1
        j = totallen - 1
        while j >= 0 and line[j] in rep:
            j -= 1
        return line[i:j+1]

    def align_and_clean(self):
        oldflines = self.oldf.split('\n')
        difflines = self.diff.split('\n')
        first_line = difflines[0]
        difflines = difflines[1:]
        regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
        startline, endline, startpos, endpos = re.match(regex, first_line).groups()
        startline, endline = int(startline) - 1, int(endline) - 1
        self.prevlines = oldflines[:startline]
        self.afterlines = oldflines[endline+1:]
        for line in difflines:
            if line.startswith('-'):
                self.lines.append(line[1:])
                self.labels.append(0)
            elif line.startswith('+'):
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
        self.lines, self.labels = list(zip(*[(line, label) for line, label in zip(self.lines, self.labels) if len(line) > 0]))


def read_review_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    idx = 0
    with open(filename) as f:
        for line in f:
            js = json.loads(line.strip())
            examples.append(
                ReviewExample(
                    idx=idx,
                    oldf=js['oldf'],
                    diff=js['patch'],
                    msg=js['msg'] if 'msg' in js else '',
                    cmtid=js['cmtid'] if 'cmtid' in js else '',
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples