import sys

def load_sentences(filename):

    sentences = []

    sent = []

    with open(filename, 'r') as fp:
        for line in fp:
            text = line.strip()
            if len(text) == 0:
                if len(sent) != 0:
                    sentences.append(sent)
                    sent = []
                continue

            arr = text.split('\t')
            wid = arr[0].strip()
            w = arr[1].strip()
            pos = arr[2].strip()
            morph = arr[3].strip()
            head_idx = arr[4].strip()
            head_label = arr[5].strip()

            sent.append((wid, w, pos, morph, head_idx, head_label))


    if len(sent) != 0:
        sentences.append(sent)

    print('Total Sentences Loaded : ' + str(len(sentences)))

    return sentences

def write_sentences(filename, gold_sentences, pred_sentences):

    assert len(gold_sentences) == len(pred_sentences)

    fp = open(filename, 'w')
    fp.write('word_id\tword\tpostag\tlemma\tgold_head\tgold_label\tpred_head\tpred_label\n')
    fp.write('\n')

    for i in range(len(gold_sentences)):
        gold = gold_sentences[i]
        pred = pred_sentences[i]
        for j in range(len(gold)):
            fp.write(gold[j][0] + '\t' + gold[j][1] + '\t' + gold[j][3] + '\t' +  gold[j][1] + '\t' + gold[j][4] + '\t' + gold[j][5] + '\t' + pred[j][4] + '\t' + pred[j][5] + '\n')
        fp.write('\n')

    fp.close()



if __name__=="__main__":

    gold_sentences = load_sentences(sys.argv[1])
    pred_sentences = load_sentences(sys.argv[2])

    write_sentences(sys.argv[3], gold_sentences, pred_sentences)
