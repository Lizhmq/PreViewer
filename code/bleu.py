import os
import argparse
from evaluator.smooth_bleu import bleu_fromstr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    ref = os.path.join(args.path, 'golds.txt')
    hyp = os.path.join(args.path, 'preds.txt')
    with open(ref, 'r') as f:
        refs = f.readlines()
    with open(hyp, 'r') as f:
        hyps = f.readlines()
    refs = [ref.strip().lower() for ref in refs]
    hyps = [hyp.strip().lower() for hyp in hyps]
    bleu = bleu_fromstr(hyps, refs)
    print(bleu)
    stopwords = open("stopwords.txt").readlines()
    stopwords = [stopword.strip() for stopword in stopwords]
    refs = [" ".join([word for word in ref.lower().split() if word not in stopwords]) for ref in refs]
    hyps = [" ".join([word for word in hyp.lower().split() if word not in stopwords]) for hyp in hyps]
    bleu = bleu_fromstr(hyps, refs)
    print(bleu)

if __name__ == '__main__':
    main()