import sys

def load_dcs_file(filename):

    mapping = {}

    with open(filename, 'r') as fp:
        for line in fp:
            if len(line.strip()) == 0:
                continue

            arr = line.strip().split('\t')

            dcs_id = arr[2]
            if dcs_id not in mapping:
                mapping[dcs_id] = set()

            # tense, case, num, gen
            tags = (arr[3], arr[4] , arr[5], arr[6] )
            mapping[dcs_id].add(tags)

    print('Keys in mapping ' + str(len(mapping)))
    return mapping


def load_pred_file(filename):

    sentences = []
    sent = []

    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            if len(line.strip()) == 0:
                if len(sent) != 0:
                    sentences.append(sent)
                    sent = []
                continue
            arr = line.strip().split('\t')
            tag = (arr[1], arr[2], arr[3], arr[4])
            sent.append(tag)


    if len(sent) != 0:
        sentences.append(sent)
        sent = []

    print("No of sentences: " + str(len(sentences)))

    return sentences



def load_gold_file(filename):

    sentences = []
    sent = []
    with open(filename, 'r') as fp:
        for line in fp:
            if 'word' in line:
                if len(sent) != 0:
                    sentences.append(sent)
                    sent = []
                continue
            if len(line.strip()) == 0:
                if len(sent) != 0:
                    sentences.append(sent)
                    sent = []
                continue

            dcs_id = line.strip().split('\t')[-1]

            sent.append(dcs_id)

    if len(sent) != 0:
        sentences.append(sent)
        sent = []

    print("No of sentences: " + str(len(sentences)))

    return sentences

if __name__=="__main__":

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    all_file = sys.argv[3]

    mapping = load_dcs_file(all_file)


    pred_sentences = load_pred_file(pred_file)

    gold_sentences = load_gold_file(gold_file)

    assert len(pred_sentences) == len(gold_sentences)

    correct = 0.0
    incorrect = 0.0
    not_matched = 0.0
    total = 0.0
    for i in range(len(gold_sentences)):

        p = pred_sentences[i]
        g = gold_sentences[i]

        assert len(p) == len(g)

        for j in range(len(p)):
            pred = p[j]
            gold = g[j]

            print(pred)
            print(gold)

            if gold not in mapping:
                not_matched +=1
            else:
                if pred in mapping[gold]:
                    correct +=1
                else:
                    incorrect +=1

            total +=1

    print(correct)
    print(incorrect)
    print(not_matched)
    print(total)

    print('Matched accuracy ' + str(correct/(total-not_matched)))
    print('Actual accuracy ' + str(correct/(total)))


