from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf_mtl import *
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    parser.add_argument('--emb_file', default='./embedding/glove.6B.100d.txt', help='path to pre-trained embedding')
    parser.add_argument('--train_file', nargs='+', default='./data/ner2003/eng.train.iobes', help='path to training file')
    parser.add_argument('--dev_file', nargs='+', default='./data/ner2003/eng.testa.iobes', help='path to development file')
    parser.add_argument('--test_file', nargs='+', default='./data/ner2003/eng.testb.iobes', help='path to test file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_hidden', type=int, default=300, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=300, help='dimension of word-level layers')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    parser.add_argument('--word_dim', type=int, default=100, help='dimension of word embedding')
    parser.add_argument('--char_layers', type=int, default=1, help='number of char level layers')
    parser.add_argument('--word_layers', type=int, default=1, help='number of word level layers')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', action='store_false', help='fine tune the diction of word embedding or not')
    parser.add_argument('--load_check_point', default='', help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_opt', action='store_true', help='also load optimizer from the checkpoint')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer choice')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--small_crf', action='store_false', help='use small crf instead of large crf, refer model.crf module for more details')
    parser.add_argument('--mini_count', type=float, default=0, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--lambda0', type=float, default=1, help='lambda0')
    parser.add_argument('--co_train', action='store_true', help='cotrain language model')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--high_way', action='store_true', help='use highway layers')
    parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', action='store_true', help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    parser.add_argument('--output_annotation', action='store_true', help='output annotation results or not')
    parser.add_argument('--label_file', nargs='+', default='./data/ner2003/eng.train.iobes', help='path to file containing set of labels')
    parser.add_argument('--prefix', default='data/sanskrit/data_save.pkl', help='path to save preprocessed data file')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('setting:')
    print(args)
    label_file = args.label_file

    # load corpus
    print('--'*50)
    print('\n')
    print('loading corpus')
    file_num = len(args.train_file)
    lines = []
    dev_lines = []
    test_lines = []
    for i in range(file_num):
        with codecs.open(args.train_file[i], 'r', 'utf-8') as f:
            lines0 = f.readlines()
        lines.append(lines0)
    for i in range(file_num):
        with codecs.open(args.dev_file[i], 'r', 'utf-8') as f:
            dev_lines0 = f.readlines()
        dev_lines.append(dev_lines0)
    for i in range(file_num):
        with codecs.open(args.test_file[i], 'r', 'utf-8') as f:
            test_lines0 = f.readlines()
        test_lines.append(test_lines0)

    dataset_loader = []
    dev_dataset_loader = []
    test_dataset_loader = []
    f_map = dict()
    l_map = dict()
    char_count = dict()
    train_features = []
    dev_features = []
    test_features = []
    train_labels = []
    dev_labels = []
    test_labels = []
    train_features_tot = []
    test_word = []

    label_maps = [] # One for each task/dataset

    print('--'*50)
    print('\n')
    print('Reading corpus...')
    for i in range(file_num):
        l_map = dict()
        dev_features0, dev_labels0 = utils.read_corpus(dev_lines[i])
        test_features0, test_labels0 = utils.read_corpus(test_lines[i])

        print('For task {0}, number of sentences in dev set : {1}'.format(i, len(dev_labels0)))
        print('For task {0}, number of sentences in test set : {1}'.format(i, len(test_labels0)))

        dev_features.append(dev_features0)
        test_features.append(test_features0)
        dev_labels.append(dev_labels0)
        test_labels.append(test_labels0)

        if args.output_annotation: #NEW
            test_word0 = utils.read_features(test_lines[i])
            test_word.append(test_word0)

        if args.load_check_point:
            if os.path.isfile(args.load_check_point):
                print("loading checkpoint: '{}'".format(args.load_check_point))
                checkpoint_file = torch.load(args.load_check_point)
                args.start_epoch = checkpoint_file['epoch']
                f_map = checkpoint_file['f_map']
                l_map = checkpoint_file['l_map']
                c_map = checkpoint_file['c_map']
                in_doc_words = checkpoint_file['in_doc_words']
                train_features, train_labels = utils.read_corpus(lines[i])
            else:
                print("no checkpoint found at: '{}'".format(args.load_check_point))
        else:
            print('constructing coding table')
            train_features0, train_labels0, f_map, l_map, char_count = utils.generate_corpus_char(lines[i], f_map, l_map, char_count, c_thresholds=args.mini_count, if_shrink_w_feature=False)
        label_maps.append(l_map)
        train_features.append(train_features0)
        train_labels.append(train_labels0)

        train_features_tot += train_features0

    print('Going to shrink character map')
    shrink_char_count = [k for (k, v) in iter(char_count.items()) if v >= args.mini_count]
    char_map = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}

    char_map['<u>'] = len(char_map)  # unk for char
    char_map[' '] = len(char_map)  # concat for char
    char_map['\n'] = len(char_map)  # eof for char

    f_set = {v for v in f_map}
    dt_f_set = f_set
    f_map = utils.shrink_features(f_map, train_features_tot, args.mini_count)

    l_set = set()
    label_sets = [] # Only dev and test set


    for i in range(file_num):
        l_set = set()
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features[i]), dt_f_set)
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features[i]), dt_f_set)

    for i in range(file_num):
        l_set = set()
        for k,j in enumerate(dev_labels[i]):
            l_set.update(j)
        for k,j in enumerate(test_labels[i]):
            l_set.update(j)
        label_sets.append(l_set)

    if not args.rand_embedding:
        print("feature size: '{}'".format(len(f_map)))
        print('loading embedding')
        if args.fine_tune:  # which means does not do fine-tune
            f_map = {'<eof>': 0}
        f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', f_map, dt_f_set, args.caseless, args.unk, args.word_dim, shrink_to_corpus=args.shrink_embedding)
        print("embedding size: '{}'".format(len(f_map)))

    else:
        f_map, in_doc_words = utils.create_word_dict(f_map)
        embedding_tensor = None

    for i in range(file_num):
        print('Loading label set from file -- ' + label_file[i])
        loaded_label_set = utils.load_label_set_from_file(label_file[i])
        for label in list(loaded_label_set):
            if label not in label_maps[i]:
                label_maps[i][label] = len(label_maps[i])

    for i in range(file_num):
        for label in label_sets[i]:
            if label not in label_maps[i]:
                label_maps[i][label] = len(label_maps[i])

    #file_prefix = 'data/pkl_files/tense_cas_num_gen/50k_'
    file_prefix = args.prefix
    print('--'*40)
    print('Going to print the label maps ....')
    print(label_maps)
    print('constructing dataset')
    load_from_pkl = False
    args.caseless = False # Always set False
    for i in range(file_num):

        if load_from_pkl:
            dataset, forw_corp, back_corp = utils.load_data_pkl_file(file_prefix + 'train_' + str(i) + ".pkl")
            dev_dataset, forw_dev, back_dev = utils.load_data_pkl_file(file_prefix + 'dev_' + str(i) + ".pkl")
            test_dataset, forw_test, back_test = utils.load_data_pkl_file(file_prefix + 'test_' + str(i) + ".pkl")

        else:
            print('will save data to files ')
            # construct dataset
            dataset, forw_corp, back_corp = utils.construct_bucket_mean_vb_wc(train_features[i], train_labels[i], label_maps[i], char_map, f_map, args.caseless)
            dev_dataset, forw_dev, back_dev = utils.construct_bucket_mean_vb_wc(dev_features[i], dev_labels[i], label_maps[i], char_map, f_map, args.caseless)
            test_dataset, forw_test, back_test = utils.construct_bucket_mean_vb_wc(test_features[i], test_labels[i], label_maps[i], char_map, f_map, args.caseless)
            dict_maps = {}
            dict_maps['label_maps'] = label_maps[i]
            dict_maps['char_map'] = char_map
            dict_maps['f_map'] = f_map
            dict_maps['caseless'] = args.caseless

            utils.save_dict_pkl_file(dict_maps, file_prefix + str(i) + "_maps.pkl")
            utils.save_data_pkl_file(dataset, forw_corp, back_corp, file_prefix +  'train_' + str(i) + ".pkl")
            utils.save_data_pkl_file(dev_dataset, forw_dev, back_dev, file_prefix +  'dev_' + str(i) + ".pkl")
            utils.save_data_pkl_file(test_dataset, forw_test, back_test, file_prefix + 'test_' + str(i) + ".pkl")

        dataset_loader.append([torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset])
        dev_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset])
        test_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset])


    d = {}

    d['args'] = args
    d['label_maps'] = label_maps
    d['char_map'] = char_map
    d['f_map'] = f_map
    d['file_num'] = file_num
    d['in_doc_words'] = in_doc_words
    d['embedding_tensor'] = embedding_tensor
    d['dataset_loader'] = dataset_loader
    d['dev_dataset_loader'] = dev_dataset_loader
    d['test_dataset_loader'] = test_dataset_loader
    d['forw_corp'] = forw_corp
    d['back_corp'] = back_corp
    d['forw_dev'] = forw_dev
    d['back_dev'] = back_dev
    d['forw_test'] = forw_test
    d['back_test'] = back_test

    with open(file_prefix + "all_data.pkl", 'wb') as fp:
        pickle.dump(d, fp)
