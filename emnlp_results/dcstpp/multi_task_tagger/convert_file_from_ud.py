import sys

def load_ud_file(filename):
    sentences = []
    sent = []
    with open(filename, 'r') as fp:
        for line in fp:
            text = line.strip()
            if len(text) == 0:
                if len(sent) != 0:
                    sentences.append(sent)
                    sent = []
            else:
                sent.append(text.split('\t')[1])


    if len(sent) != 0:
        sentences.append(sent)
    print('Number of sentences ' + str(len(sentences)))
    return sentences


def write_to_format(filename, sentences):

    fp = open(filename, 'w')
    for sent in sentences:
        for word in sent:
            fp.write(word + '\n')

        fp.write('\n')

    fp.close()


if __name__=="__main__":

    sentences = load_ud_file(sys.argv[1])


    write_to_format(sys.argv[2], sentences)
