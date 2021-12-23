from evaluator import smooth_bleu

pred_file = open('prediction/summarize_java.output', 'r').readlines()
gold_fn = 'prediction/summarize_java.gold'

predictions = [pred.strip() for pred in pred_file]

(goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

print(bleu)