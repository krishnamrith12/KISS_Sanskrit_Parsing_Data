import sys


def load_results(filename, return_len_wise=False):

    results = []
    sent = []
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            splits = line.strip().split('\t')
            if len(line.strip()) == 0:
                if len(sent) != 0:
                    results.append(sent)
                    sent = []
                continue
            gold_head = splits[-4]
            gold_label = splits[-3]
            pred_head = splits[-2]
            pred_label = splits[-1]
            sent.append((gold_head, gold_label, pred_head, pred_label))
    print('Total Number of sentences ' + str(len(results)))

    len_wise = {}
    for i in range(len(results)):
        l = len(results[i])
        if l not in len_wise:
            len_wise[l] = []
        len_wise[l].append(results[i])

    if return_len_wise:
        return results, len_wise
    return results

def calculate_las_uas(gold_heads, gold_labels, pred_heads, pred_labels):

    u_correct = 0
    l_correct = 0
    u_total = 0
    l_total = 0

    for i in range(len(gold_heads)):
        if gold_heads[i] == pred_heads[i]:
            u_correct +=1
        u_total +=1
        l_total +=1
        if gold_heads[i] == pred_heads[i] and gold_labels[i] == pred_labels[i]:
            l_correct +=1
    return u_correct, u_total, l_correct, l_total


def calculate_stats(results, fp, filename, length):
    u_correct = 0
    l_correct = 0
    u_total = 0
    l_total = 0

    sent_uas = []
    sent_las = []

    for i in range(len(results)):
        gold_heads, gold_labels, pred_heads, pred_labels = zip(*results[i])
        u_c, u_t, l_c, l_t = calculate_las_uas(gold_heads, gold_labels, pred_heads, pred_labels)
        if u_t >0:
            uas = float(u_c)/u_t
            las = float(l_c)/l_t
            sent_uas.append(uas)
            sent_las.append(las)
        u_correct += u_c
        l_correct += l_c
        u_total += u_t
        l_total += l_t

    if u_total == 0:
        UAS = 0.0
        LAS = 0.0
    else:
        UAS = float(u_correct)/u_total
        LAS = float(l_correct)/l_total

    fp.write(filename + '\t' + str(length) + '\t' + str(len(results)) + '\t' + str(UAS) + '\t' + str(LAS) + '\n')

    return sent_uas, sent_las, UAS, LAS

def write_results(sent_uas, sent_las, filename_uas, filename_las):

    fp_uas = open(filename_uas, 'w')
    fp_las = open(filename_las, 'w')

    for i in range(len(sent_uas)):
        fp_uas.write(str(sent_uas[i]) + '\n')
        fp_las.write(str(sent_las[i]) + '\n')

    fp_uas.close()
    fp_las.close()


if __name__=="__main__":

    num_files = int(sys.argv[1])
    files = []
    fp = open('all_results.tsv', 'w')
    fp.write('Filename\tLength\tNum.Sentences\tUAs\tLAS\n')
    for i in range(num_files):
        res, len_wise = load_results(sys.argv[i + 2], return_len_wise=True)
        lens = sorted(len_wise.keys())
        for k in lens:
            calculate_stats(len_wise[k], fp, sys.argv[i + 2], k)
        fp.write('\n')

    fp.close()
