from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.mtl_deep_shortcut import *
import model.utils as utils
from model.evaluator import eval_wc
from model.predictor import predict_wc #NEW

import argparse
import json
import os
import sys, pickle
from tqdm import tqdm
import itertools
import functools
import random

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def reorder_list(old_list, reorder_index):

    new_list = []
    for i in reorder_index:
        new_list.append(old_list[i])

    return new_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    #parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    #parser.add_argument('--emb_file', default='./embedding/glove.6B.100d.txt', help='path to pre-trained embedding')
    #parser.add_argument('--train_file', nargs='+', default='./data/ner2003/eng.train.iobes', help='path to training file')
    #parser.add_argument('--dev_file', nargs='+', default='./data/ner2003/eng.testa.iobes', help='path to development file')
    #parser.add_argument('--test_file', nargs='+', default='./data/ner2003/eng.testb.iobes', help='path to test file')
    #parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    #parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_hidden', type=int, default=300, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=300, help='dimension of word-level layers')
    #parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio')
    #parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    #parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='./clean_models/', help='checkpoint path')
    #parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    parser.add_argument('--word_dim', type=int, default=100, help='dimension of word embedding')
    parser.add_argument('--char_layers', type=int, default=1, help='number of char level layers')
    parser.add_argument('--word_layers', type=int, default=1, help='number of word level layers')
    #parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    #parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    #parser.add_argument('--fine_tune', action='store_false', help='fine tune the diction of word embedding or not')
    #parser.add_argument('--load_check_point', default='', help='path previous checkpoint that want to be loaded')
    #parser.add_argument('--load_opt', action='store_true', help='also load optimizer from the checkpoint')
    #parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer choice')
    #parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    #parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--small_crf', action='store_false', help='use small crf instead of large crf, refer model.crf module for more details')
    #parser.add_argument('--mini_count', type=float, default=0, help='thresholds to replace rare words with <unk>')
    #parser.add_argument('--lambda0', type=float, default=1, help='lambda0')
    #parser.add_argument('--co_train', action='store_true', help='cotrain language model')
    #parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    #parser.add_argument('--high_way', action='store_true', help='use highway layers')
    #parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    #parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    #parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    #parser.add_argument('--shrink_embedding', action='store_true', help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    #parser.add_argument('--output_annotation', action='store_true', help='output annotation results or not')
    #parser.add_argument('--label_file', nargs='+', default='./data/ner2003/eng.train.iobes', help='path to file containing set of labels')


    #parser.add_argument('--train_pkl_file', nargs='+', default='./data/ner2003/eng.train.iobes', help='path to file containing set of labels')

    parser.add_argument('--output_directory', default='./', help='directory for storing output annotation files.')
    parser.add_argument('--num_tasks', type=int, default=4, help='number of tasks')
    parser.add_argument('--out_files', nargs='+', default='./data/.txt', help='Apply output annotation to files. Should be in the same order as the pkl file order')
    parser.add_argument('--args_file', default='./data/ner2003/eng.train.iobes', help='path to file containing set of labels')
    parser.add_argument('--prefix', default='./data/ner2003/eng.train.iobes', help='prefix of the saved pkl files.')
    parser.add_argument('--order', default='cas-gen-num-lem', help='specify the order of the tasks')
    parser.add_argument('--order_pkl_file', default='cas-gen-num-lem', help='specify the order of the tasks in pkl file')
    parse_args = parser.parse_args()

    #save_filename=sys.argv[1]

    print(parse_args)
    out_files = parse_args.out_files


    assert len(parse_args.order_pkl_file.split('-')) == len(parse_args.order.split('-'))

    pkl_order_dict = {}
    pkl_orders = parse_args.order_pkl_file.split('-')
    for i, o in enumerate(pkl_orders):
        pkl_order_dict[o] = i

    order_list = parse_args.order.split('-')

    reorder_index = []
    for o in order_list:
        reorder_index.append(pkl_order_dict[o])

    print("Re-ordering list is ")
    print(reorder_index)

    output_directory = parse_args.output_directory

    save_filename = parse_args.args_file

    args_file = parse_args.args_file

    num_tasks = parse_args.num_tasks

    num_files = num_tasks

    print("Number of tasks : " + str(num_tasks))
    print("Order of the tasks is " + parse_args.order)

    print("CRF type -- " + str(parse_args.small_crf))
    file_prefix = parse_args.prefix

    new_dataset_loader = []
    new_dev_dataset_loader = []
    new_test_dataset_loader = []

    test_word = []
    test_lines = []
    print('Loading --out files for annotation....')
    for i in range(num_tasks):
        with codecs.open(parse_args.out_files[i], 'r', 'utf-8') as f:
            test_lines0 = f.readlines()
        test_lines.append(test_lines0)

    for i in range(num_tasks):
        test_word0 = utils.read_features_sentences(test_lines[i])
        print(test_word0[0])
        print("Number of docs : " + str(len(test_word0)))
        test_word.append(test_word0)

    test_word = reorder_list(test_word, reorder_index)

    #for i in range(num_files):
    for i in reorder_index:
        dataset, forw_corp, back_corp = utils.load_data_pkl_file(file_prefix +  'train_' + str(i) + ".pkl")
        dev_dataset, forw_dev, back_dev = utils.load_data_pkl_file(file_prefix + 'dev_' + str(i) + ".pkl")
        test_dataset, forw_test, back_test = utils.load_data_pkl_file(file_prefix + 'test_' + str(i) + ".pkl")


        new_dataset_loader.append([torch.utils.data.DataLoader(tup, parse_args.batch_size, shuffle=True, drop_last=False, num_workers=40) for tup in dataset])
        new_dev_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset])
        new_test_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset])
    print('Loading data dictionary from file ' + save_filename)

    with open(save_filename, 'rb') as fp:
        d = pickle.load(fp)

    args = d['args']
    args.gpu = 0
    label_maps = d['label_maps']
    char_map = d['char_map']
    f_map = d['f_map']
    file_num = d['file_num']
    in_doc_words = d['in_doc_words']
    embedding_tensor = d['embedding_tensor']
    dataset_loader = d['dataset_loader']
    dev_dataset_loader = d['dev_dataset_loader']
    test_dataset_loader = d['test_dataset_loader']
    forw_corp = d['forw_corp']
    back_corp = d['back_corp']
    forw_dev = d['forw_dev']
    back_dev = d['back_dev']
    forw_test = d['forw_test']
    back_test = d['back_test']


    # Reorder label_maps
    label_maps = reorder_list(label_maps, reorder_index)
    args.checkpoint = parse_args.checkpoint
    # Set args

    args.word_hidden = parse_args.word_hidden
    args.char_hidden = parse_args.char_hidden
    args.word_dim = parse_args.word_dim
    args.char_dim = parse_args.char_dim
    args.char_layers = parse_args.char_layers
    args.word_layers = parse_args.word_layers
    args.small_crf = parse_args.small_crf

    print("Will save checkpoint in " + str(args.checkpoint))

    inv_f_map = {}
    for k, v in f_map.items():
        inv_f_map[v] = k
    print(inv_f_map[6430])
    #print(f_map['<pad>'])

    args.output_annotation = True

    print("Number of files : " + str(file_num))

    dataset_loader = new_dataset_loader
    dev_dataset_loader = new_dev_dataset_loader
    test_dataset_loader = new_test_dataset_loader

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print(args)
    args.batch_size = 10
    # build model
    print('building model')
    print(label_maps)
    label_maps_sizes = [len(lmap) for lmap in label_maps]
    print(label_maps_sizes)
    ner_model = LM_LSTM_CRF(label_maps_sizes, len(char_map), args.char_dim, args.char_hidden, args.char_layers, args.word_dim, args.word_hidden, args.word_layers, len(f_map), args.drop_out, file_num, large_CRF=args.small_crf, if_highway=args.high_way, in_doc_words=in_doc_words, highway_layers = args.highway_layers)

    if args.load_check_point:
        ner_model.load_state_dict(checkpoint_file['state_dict'])
    else:
        if not args.rand_embedding:
            ner_model.load_pretrained_word_embedding(embedding_tensor)
        ner_model.rand_init(init_word_embedding=args.rand_embedding)

    if args.update == 'sgd':
        optimizer = optim.SGD(ner_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(ner_model.parameters(), lr=args.lr)

    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    crit_lm = nn.CrossEntropyLoss()
    crit_ner_list = nn.ModuleList()
    for i in range(file_num):
        ith_label_map = label_maps[i]
        crit_ner = CRFLoss_vb(len(ith_label_map), ith_label_map['<start>'], ith_label_map['<pad>'])
        crit_ner_list.append(crit_ner)

    if args.gpu >= 0:
        if_cuda = True
        print('device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        crit_lm.cuda()
        for i in range(file_num):
            crit_ner_list[i].cuda()
        ner_model.cuda()
        packer_list = []
        for i in range(file_num):
            packer = CRFRepack_WC(len(label_maps[i]), True)
            packer_list.append(packer)
    else:
        if_cuda = False
        packer_list = []
        for i in range(file_num):
            packer = CRFRepack_WC(len(label_maps[i]), False)
            packer_list.append(packer)

    tot_length = sum(map(lambda t: len(t), dataset_loader))

    best_f1 = []
    for i in range(file_num):
        best_f1.append(float('-inf'))

    best_pre = []
    for i in range(file_num):
        best_pre.append(float('-inf'))

    best_rec = []
    for i in range(file_num):
        best_rec.append(float('-inf'))

    best_acc = []
    for i in range(file_num):
        best_acc.append(float('-inf'))

    track_list = list()
    start_time = time.time()
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    evaluator_list = []
    predictor_list = []
    for i in range(file_num):
        evaluator = eval_wc(packer_list[i], label_maps[i], args.eva_matrix)

        predictor = predict_wc(if_cuda, f_map, char_map, label_maps[i], f_map['<eof>'], char_map['\n'], label_maps[i]['<pad>'], label_maps[i]['<start>'], True, args.batch_size, args.caseless) #NEW

        evaluator_list.append(evaluator)
        predictor_list.append(predictor)

    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        sample_num = 1

        epoch_loss = 0
        ner_model.train()

        for sample_id in tqdm( range(sample_num) , mininterval=2,
                desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stdout):

            file_no = random.randint(0, file_num-1)
            print("Sampling from task : {0} - name {1}".format(file_no, order_list[file_no]))
            cur_dataset = dataset_loader[file_no]

            for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v in itertools.chain.from_iterable(cur_dataset):

                f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v)

                ner_model.zero_grad()
                scores = ner_model(f_f, f_p, b_f, b_p, w_f, file_no)
                loss = crit_ner_list[file_no](scores, tg_v, mask_v)

                epoch_loss += utils.to_scalar(loss)
                if args.co_train:
                    cf_p = f_p[0:-1, :].contiguous()
                    cb_p = b_p[1:, :].contiguous()
                    cf_y = w_f[1:, :].contiguous()
                    cb_y = w_f[0:-1, :].contiguous()
                    cfs, _ = ner_model.word_pre_train_forward(f_f, cf_p)
                    loss = loss + args.lambda0 * crit_lm(cfs, cf_y.view(-1))
                    cbs, _ = ner_model.word_pre_train_backward(b_f, cb_p)
                    loss = loss + args.lambda0 * crit_lm(cbs, cb_y.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm(ner_model.parameters(), args.clip_grad)
                optimizer.step()

        epoch_loss /= tot_length

        # update lr
        utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        # eval & save check_point
        if 'f' in args.eva_matrix:
            dev_f1, dev_pre, dev_rec, dev_acc = evaluator_list[file_no].calc_score(ner_model, dev_dataset_loader[file_no], file_no)

            if dev_f1 > best_f1[file_no]:
                patience_count = 0
                best_f1[file_no] = dev_f1
                best_pre[file_no] = dev_pre
                best_rec[file_no] = dev_rec

                test_f1, test_pre, test_rec, test_acc = evaluator_list[file_no].calc_score(ner_model, test_dataset_loader[file_no], file_no)

                track_list.append(
                    {'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'test_f1': test_f1,
                     'test_acc': test_acc})

                print(
                    '(loss: %.4f, epoch: %d, dataset: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f, F1 on test = %.4f, pre on test= %.4f, rec on test= %.4f), saving...' %
                    (epoch_loss,
                     args.start_epoch,
                     file_no,
                     dev_f1,
                     dev_pre,
                     dev_rec,
                     test_f1,
                     test_pre,
                     test_rec))

                if args.output_annotation: #NEW
                    print('annotating')
                    with open(output_directory + 'output'+str(file_no)+'.txt', 'w') as fout:
                        predictor_list[file_no].output_batch(ner_model, test_word[file_no], fout, file_no)

                try:
                    utils.save_checkpoint({
                        'epoch': args.start_epoch,
                        'state_dict': ner_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'l_map': label_maps[file_no],
                        'c_map': char_map,
                        'in_doc_words': in_doc_words
                    }, {'track_list': track_list,
                        'args': vars(args)
                        }, args.checkpoint + 'cwlm_lstm_crf' + str(file_no))
                except Exception as inst:
                    print(inst)

            else:
                patience_count += 1
                print('(loss: %.4f, epoch: %d, dataset: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f)' %
                      (epoch_loss,
                       args.start_epoch,
                       file_no,
                       dev_f1,
                       dev_pre,
                       dev_rec))
                track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})

        else:

            dev_acc = evaluator_list[file_no].calc_score(ner_model, dev_dataset_loader[file_no], file_no)

            if dev_acc > best_acc[file_no]:
                patience_count = 0
                best_acc[file_no] = dev_acc

                test_acc = evaluator_list[file_no].calc_score(ner_model, test_dataset_loader[file_no], file_no)

                track_list.append(
                    {'loss': epoch_loss, 'dev_acc': dev_acc, 'test_acc': test_acc})

                print(
                    '(loss: %.4f, epoch: %d, dataset : %d, dev acc = %.4f, acc on test= %.4f), saving...' %
                    (epoch_loss,
                     args.start_epoch,
                      file_no,
                     dev_acc,
                     test_acc))
                if args.output_annotation: #NEW
                    print('annotating')
                    with open(output_directory + 'output_best_'+ str(file_no)+ "_" + str(order_list[file_no]) +'_.txt', 'w') as fout:
                        predictor_list[file_no].output_batch(ner_model, test_word[file_no], fout, file_no)

                try:
                    utils.save_checkpoint({
                        'epoch': args.start_epoch,
                        'state_dict': ner_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'l_map': label_maps[file_no],
                        'c_map': char_map,
                        'in_doc_words': in_doc_words
                    }, {'track_list': track_list,
                        'args': vars(args)
                        }, args.checkpoint + 'cwlm_lstm_crf_' + str(order_list[file_no]) + "_" + str(file_no))
                except Exception as inst:
                    print(inst)

            else:
                patience_count += 1
                print('(loss: %.4f, epoch: %d, dev acc = %.4f)' %
                      (epoch_loss,
                       args.start_epoch,
                       dev_acc))
                track_list.append({'loss': epoch_loss, 'dev_acc': dev_acc})

        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        if patience_count >= args.patience and args.start_epoch >= args.least_iters:
            break

    #print best
    # if 'f' in args.eva_matrix:
    #     eprint(args.checkpoint + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc))
    # else:
    #     eprint(args.checkpoint + ' dev_acc: %.4f test_acc: %.4f\n' % (dev_acc, test_acc))

    # printing summary
    # print('setting:')
    # print(args)
