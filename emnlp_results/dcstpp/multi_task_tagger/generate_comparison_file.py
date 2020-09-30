import sys

order = ['Tense' , 'Case', 'Number', 'Gender' , 'LemmaLastCharacter']
index = [2, 3, 4, 5, 6]

def load_from_file(filename, delimiter=' ', gold=False):

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
                    arr = text.split(delimiter)[1:]
                    t = [arr[idx] for idx in index]
                    tag.append(tuple(t))
                else:
                    tag.append(text.split(delimiter)[-1])

    if len(sent) != 0:
        sentences.append(sent)
        tags.append(tag)

    print("Number of sentences loaded :  {0}".format(len(sentences)))

    return sentences, tags


if __name__=="__main__":

    gold_file = sys.argv[1]

    num_files = int(sys.argv[2])

    directory = sys.argv[3]

    file_names = []
    for i in range(num_files):
        file_names.append(sys.argv[i+4])

    print(file_names)

    list_sentences = []
    list_tags = []

    for name in file_names:
        sentences, tags = load_from_file(directory + '/' + name)
        list_sentences.append(sentences)
        list_tags.append(tags)

    gold_sentences, gold_tags = load_from_file(gold_file, delimiter='\t', gold=True)

    fp = open('comparison_file_new.conll', 'w')
    fp.write('word' + "\t" + '\t'.join(order) + "\n")
    fp.write("\n")

    for i, sent in enumerate(sentences):
        for j, word in enumerate(sent):
            tag_str = ""
            gold_t = gold_tags[i][j]
            for k in range(num_files):
                tag_str+= str(gold_t[k]) + ":" + str(list_tags[k][i][j] + "\t")
            if tag_str[-1] == "\t":
                tag_str = tag_str[:-1]
            fp.write(word + "\t" + tag_str + "\n")
        fp.write("\n")

    fp.close()
