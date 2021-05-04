from ast import literal_eval
import pandas as pd
from scipy.stats import sem

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)


if __name__ == '__main__':
    
    tsd = pd.read_csv("toxic_spans/data/tsd_test.csv") 
    tsd.spans = tsd.spans.apply(literal_eval)
    
    spans_pred = []
    with open("spans-pred.txt") as f:
        for line in f:
            spans_pred.append(line.strip().split('\t')[1])
            
    tsd['preds'] = spans_pred
    tsd.preds = tsd.preds.apply(literal_eval)
    tsd["f1_scores"] = tsd.apply(lambda row: f1(row.preds, row.spans), axis=1)
    print("Evaluate result: {:.2f}%".format(tsd.f1_scores.mean()*100))
    