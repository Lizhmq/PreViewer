from numpy import outer
from evaluator import smooth_bleu
from evaluator.bleu import _bleu

pred_file = open('sh/test.output.30k-60k', 'r').readlines()
gold_fn = 'sh/test.gold'
golds = open(gold_fn, 'r').readlines()

predictions = [pred.strip() for pred in pred_file]
golds = [gold.strip().replace("`", "") for gold in golds]
print(f"EM: {sum([a == b for (a, b) in zip(predictions, golds)]) / len(predictions)}")

predictions = [str(i) + "\t" + pred.replace("\t", " ") for (i, pred) in enumerate(predictions)]
new_gold_fn = 'sh/test.gold.new'
gold_file = open(gold_fn, 'r').readlines()
gold_file = [gold.strip().replace("`", "") for gold in gold_file]
gold_file = [str(i) + "\t" + gold.replace("\t", " ") for (i, gold) in enumerate(gold_file)]
open(new_gold_fn, 'w').write('\n'.join(gold_file))

(goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, new_gold_fn)
bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
print(bleu)

# gold_fn = "sh/test.gold"
# output_fn = "sh/test.output"
# print(round(_bleu(gold_fn, output_fn), 2))