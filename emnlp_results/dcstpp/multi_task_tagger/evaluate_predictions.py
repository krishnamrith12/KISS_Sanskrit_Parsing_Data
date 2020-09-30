import sys
import argparse

def load_sentences(filename, delimiter="\t", gold=False):

    sentences  = []
    tags = []
    sent = []
    tag = []

    with open(filename, 'r') as fp:

        for line in fp:
            text = line.strip()
            if len(text) ==0 or "-DOCSTART-" in text:
                if len(sent) != 0:
                    sentences.append(sent)
                    tags.append(tag)
                    sent = []
                    tag = []
            else:
                word = text.split(delimiter)[0]
                sent.append(word)
                if gold:
                    # Last column in CNG composite
                    all_tags = text.split(delimiter)[1:-1]
                    tag.append(tuple(all_tags))
                else:
                    tag.append(text.split(delimiter)[-1])

    if len(sent) != 0:
        sentences.append(sent)
        tags.append(tag)

    print("Number of sentences loaded :  {0}".format(len(sentences)))

    return sentences, tags


def get_exact_acc(pred, gold):
    # pred : each element a list of list
    # gold : list of list of tuples
    num_sentences = len(gold)
    num_tags = len(pred)

    # For each of the tag types
    pred_tags = {}
    gold_tags = {}

    for i in range(num_tags):
        pred_tags[i] = []
        gold_tags[i] = []

    correct = 0
    total = 0
    for i in range(num_sentences):
        for j in range(len(gold[i])):
            gold_tup = gold[i][j]
            pred_tup = list()
            for t in range(num_tags):
                pred_tup.append(pred[t][i][j])
                pred_tags[t].append(pred[t][i][j])
                gold_tags[t].append(gold[i][j][t])
            pred_tup = tuple(pred_tup)

            ## Add tags to respective type lists
            #for t in range(num_tags):
            #    pred_tags[t].append(pred[t][i][j])
            #    gold_tags[t].append(gold[i][j][t])

            if pred_tup == gold_tup:
                correct +=1
            total +=1

    print("Printing Exact Accuracy Stats : ")
    print("Correct : {0}".format(correct))
    print("Total : {0}".format(total))
    print("Exact Accuracy : {0}".format(correct/(float(total))))

    return pred_tags, gold_tags


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='For Evaluating generated tags. Please provide files in the order that tags are present in the gold file.')
    parser.add_argument('--files', nargs='+', default='./data/ner2003/eng.train.iobes', help='path to annotated files')
    parser.add_argument('--gold', default='', help='Gold file')
    parser.add_argument('--names', nargs='+', default='', help='Name of the tags in gold file columns in order')

    args = parser.parse_args()

    tags = []
    for i in range(len(args.files)):
        _, t = load_sentences(args.files[i], delimiter=' ', gold=False)
        tags.append(t)

    _, gold_labels = load_sentences(args.gold, gold=True)

    # gold_labels : list of list of tuples
    # tags : each element a list of list

    types = len(tags)
    assert types == len(gold_labels[0][0])

    print("--"*40)
    print("Going to calculate accuracy -- Token wise -- all tags should match for each token")

    pred_tags, gold_tags = get_exact_acc(tags, gold_labels)

    print("--"*40)
    acc = {}
    for i in range(types):
        corr = 0
        tot = 0
        for j in range(len(pred_tags[i])):
            if pred_tags[i][j] == gold_tags[i][j]:
                corr +=1
            tot +=1
        print("Printing stats for tag index {0} with name {1} ".format(i, args.names[i]))
        print("Correct : {0}".format(corr))
        print("Total : {0}".format(tot))
        print("Exact Accuracy : {0}".format(corr/(float(tot))))

    print("\n")
    print("--"*40)


