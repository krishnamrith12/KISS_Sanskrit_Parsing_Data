import sys

names = ['tense', 'case', 'num', 'gen', 'lemma']

gold_index = [2, 3, 4, 5, 6]
ann_index = [0, 1, 2, 3, 4]

def load_from_file(filename, delimiter="\t", gold=False):

    words  = []
    tags = []

    with open(filename, 'r') as fp:

        for line in fp:
            text = line.strip()
            if len(text) ==0 or "-DOCSTART-" in text:
                continue
            else:
                word = text.split(delimiter)[0]
                words.append(word)
                if gold:
                    # Last column in CNG composite
                    all_tags = text.split(delimiter)[1:]
                    tag = {}
                    for i, name in enumerate(names):
                        tag[name] = all_tags[gold_index[i]]
                    tags.append(tag)
                else:
                    all_tags = text.split(delimiter)[1:]
                    tag = {}
                    for i, name in enumerate(names):
                         tag[name] = all_tags[ann_index[i]]
                    tags.append(tag)

    print("Number of words loaded :  {0}".format(len(tags)))

    return words, tags



def computeF1(hyps, golds, prefix, labels_to_ix=None, write_results=False):


    f1_precision_scores = {}
    f1_precision_total = {}
    f1_recall_scores = {}
    f1_recall_total = {}
    f1_average = 0.0

    # Precision
    for i, word_tags in enumerate(hyps, start=0):
        for k, v in word_tags.items():
            if v=="999":
                continue
            if k not in f1_precision_scores:
                f1_precision_scores[k] = 0
                f1_precision_total[k] = 0
            if k in golds[i]:
                if v==golds[i][k]:
                    f1_precision_scores[k] += 1
            f1_precision_total[k] += 1

    #print(f1_precision_scores)
    #print(f1_precision_total)
    f1_micro_precision = sum(f1_precision_scores.values())/sum(f1_precision_total.values())

    for k in f1_precision_scores.keys():
        f1_precision_scores[k] = f1_precision_scores[k]/f1_precision_total[k]

    # Recall
    for i, word_tags in enumerate(golds, start=0):
        for k, v in word_tags.items():
            if v=="999":
                continue
            if k not in f1_recall_scores:
                f1_recall_scores[k] = 0
                f1_recall_total[k] = 0
            if k in hyps[i]:
                if v==hyps[i][k]:
                    f1_recall_scores[k] += 1
            f1_recall_total[k] += 1

    #print(f1_recall_scores)
    #print(f1_recall_total)
    f1_micro_recall = sum(f1_recall_scores.values())/sum(f1_recall_total.values())

    f1_scores = {}

    for k in f1_recall_scores.keys():
        f1_recall_scores[k] = f1_recall_scores[k]/f1_recall_total[k]
        if f1_recall_scores[k]==0 or k not in f1_precision_scores:
            f1_scores[k] = 0
        else:
            f1_scores[k] = 2 * (f1_precision_scores[k] * f1_recall_scores[k]) / (f1_precision_scores[k] + f1_recall_scores[k])

        f1_average += f1_recall_total[k] * f1_scores[k]

    f1_average /= sum(f1_recall_total.values())
    f1_micro_score = 2 * (f1_micro_precision * f1_micro_recall) / (f1_micro_precision + f1_micro_recall)

    #print(f1_precision_scores)
    #print(f1_recall_scores)
    #print(f1_average, f1_micro_score)
    return f1_average, f1_micro_score


def get_exact_accuracy(gold_tags, pred_tags):

    # inputs: list of dicts
    correct = 0
    total = 0
    for i, g in enumerate(gold_tags):
        p = pred_tags[i]
        total +=1
        if p == g:
            correct +=1

    print("--"*40)
    print("Total Correct :" + str(correct))
    print("Total         :" + str(total))
    print("Exact Token Accuracy :" + str(correct/(float(total))))
    print("--"*40)

if __name__=="__main__":


    ann_file = sys.argv[1]
    gold_file = sys.argv[2]

    _, tags_ann = load_from_file(ann_file, delimiter='\t', gold=False)
    _, tags_gold = load_from_file(gold_file, delimiter='\t', gold=True)

    get_exact_accuracy(tags_gold, tags_ann)

    f1_average, f1_micro_score = computeF1(tags_ann, tags_gold, "")

    print("--"*40 )
    print("Average F1 or Macro F1 : " + str(f1_average) + "\n")
    print("--"*40)
    print("Micro F1 : " + str(f1_micro_score) + "\n")
    print("--"*40)
