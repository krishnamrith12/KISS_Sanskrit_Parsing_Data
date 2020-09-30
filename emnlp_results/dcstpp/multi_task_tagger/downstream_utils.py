from __future__ import print_function
import sys
sys.path.insert(0, 'multi_task_tagger')
sys.path.insert(0, 'multi_task_tagger/model')
import datetime
import model
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
import pickle
from tqdm import tqdm
import itertools
import functools
import random

random.seed(1234)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def reorder_list(old_list, reorder_index):

    new_list = []
    for i in reorder_index:
        new_list.append(old_list[i])

    return new_list


def do_eval_and_output_annotation(args, ner_model, predictor_list, order_list, output_directory, test_word):

    num_files = len(test_word)

    for file_no in range(num_files):
        print('annotating')
        with open(output_directory + 'output_best_'+ str(file_no)+ "_" + str(order_list[file_no]) +'_.txt', 'w') as fout:
            predictor_list[file_no].output_batch(ner_model, test_word[file_no], fout, file_no)


#def get_mtl_predictor(args):




def get_multi_task_model(prefix='multi_task_tagger/data/pkl_files/num_gen_cas_tense_lem/50k_',
                        args_file='multi_task_tagger/data/pkl_files/num_gen_cas_tense_lem/50k_all_data.pkl',
                        order_pkl_file='num-gen-cas-tense-lem',
                        order='num-gen-cas-tense-lem',
                        num_tasks=5, output_directory= 'test_data/poetry/', eva_matrix='a', word_dim=200, char_dim=30,
                        char_layers=1, word_layers=1, small_crf=True,
                        checkpoint='multi_task_tagger/checkpoints_new/layered_num_gen_cas_tense_lem_15_May/', word_hidden=256,
                        batch_size=16, char_hidden=100,
                        load_check_point=True, do_eval=True,
                        checkpoint_file='multi_task_tagger/checkpoints_new/layered_num_gen_cas_tense_lem_15_May/cwlm_lstm_crf_cas_2.model',
                        out_files=None):

    #out_files = out_files

    assert len(order_pkl_file.split('-')) == len(order.split('-'))

    pkl_order_dict = {}
    pkl_orders = order_pkl_file.split('-')
    for i, o in enumerate(pkl_orders):
        pkl_order_dict[o] = i

    order_list = order.split('-')

    reorder_index = []
    for o in order_list:
        reorder_index.append(pkl_order_dict[o])

    print("Re-ordering list is ")
    print(reorder_index)

    output_directory = output_directory

    save_filename = args_file

    args_file = args_file

    num_tasks = num_tasks

    num_files = num_tasks

    print("Number of tasks : " + str(num_tasks))
    print("Order of the tasks is " + order)

    print("CRF type -- " + str(small_crf))
    file_prefix = prefix

    new_dataset_loader = []
    new_dev_dataset_loader = []
    new_test_dataset_loader = []

    test_word = []
    test_lines = []
    #print('Loading --out files for annotation....')
    #for i in range(num_tasks):
    #    with codecs.open(out_files[i], 'r', 'utf-8') as f:
    #        test_lines0 = f.readlines()
    #    test_lines.append(test_lines0)

    #for i in range(num_tasks):
    #    test_word0 = utils.read_features_sentences(test_lines[i])
    #    print(test_word0[0])
    #    print("Number of docs : " + str(len(test_word0)))
    #    test_word.append(test_word0)

    #test_word = reorder_list(test_word, reorder_index)

    #for i in range(num_files):
    for i in reorder_index:
        dataset, forw_corp, back_corp = utils.load_data_pkl_file(file_prefix +  'train_' + str(i) + ".pkl")
        dev_dataset, forw_dev, back_dev = utils.load_data_pkl_file(file_prefix + 'dev_' + str(i) + ".pkl")
        test_dataset, forw_test, back_test = utils.load_data_pkl_file(file_prefix + 'test_' + str(i) + ".pkl")


        new_dataset_loader.append([torch.utils.data.DataLoader(tup, batch_size, shuffle=True, drop_last=False, num_workers=40) for tup in dataset])
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
    file_num = num_tasks

    print('Shape of Embedding Tensor')
    print(embedding_tensor.shape)

    # Reorder label_maps
    label_maps = reorder_list(label_maps, reorder_index)
    args.checkpoint = checkpoint
    # Set args

    args.word_hidden = word_hidden
    args.char_hidden = char_hidden
    args.word_dim = word_dim
    args.char_dim = char_dim
    args.char_layers = char_layers
    args.word_layers = word_layers
    args.small_crf = small_crf
    args.eva_matrix = eva_matrix
    args.load_check_point = load_check_point
    args.do_eval = do_eval
    args.checkpoint_file = checkpoint_file

    print(args.word_hidden)

    print("Will save checkpoint in " + str(args.checkpoint))

    inv_f_map = {}
    for k, v in f_map.items():
        inv_f_map[v] = k
    #print(inv_f_map[6430])
    print(f_map['<unk>'])

    args.output_annotation = True

    print("Number of files : " + str(file_num))

    dataset_loader = new_dataset_loader
    dev_dataset_loader = new_dev_dataset_loader
    test_dataset_loader = new_test_dataset_loader

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print(args)
    args.batch_size = batch_size
    # build model
    print('building model')
    print(label_maps)
    label_maps_sizes = [len(lmap) for lmap in label_maps]
    print(label_maps_sizes)
    print('File_num' + str(file_num))
    ner_model = LM_LSTM_CRF(label_maps_sizes, len(char_map), args.char_dim, args.char_hidden, args.char_layers, args.word_dim, args.word_hidden, args.word_layers, len(f_map), args.drop_out, file_num, large_CRF=args.small_crf, if_highway=args.high_way, in_doc_words=in_doc_words, highway_layers = args.highway_layers)

    if args.load_check_point:
        print(args.checkpoint_file)
        if os.path.isfile(args.checkpoint_file):
            print("loading checkpoint: '{}'".format(args.checkpoint_file))
            checkpoint_file = torch.load(args.checkpoint_file)
        else:
            raise FileNotFoundError('File not found')
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

    print('Num of epochs : ' + str(args.epoch))

    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    evaluator_list = []
    predictor_list = []
    for i in range(file_num):
        evaluator = eval_wc(packer_list[i], label_maps[i], args.eva_matrix)

        predictor = predict_wc(if_cuda, f_map, char_map, label_maps[i], f_map['<eof>'], char_map['\n'], label_maps[i]['<pad>'], label_maps[i]['<start>'], True, args.batch_size, args.caseless) #NEW

        evaluator_list.append(evaluator)
        predictor_list.append(predictor)

    return predictor_list, ner_model

    #if args.do_eval:
    #    # If only evaluation and output annotation

    #    do_eval_and_output_annotation(args, ner_model, predictor_list, order_list, output_directory, test_word)

    #    print('Done with output annotation. Exiting...')
    #    sys.exit(0)
    #else:
    #    raise Error("Don't know what to do")
