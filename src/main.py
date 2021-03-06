# coding = utf-8
import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
import os
import datetime
import traceback
import model



# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# self define
from utils import preprocess_write, get_pretrained_word_embed, preprocess_pos_neg
from dataload import load_data, load_pos_neg_data
from train import eval, trainRGL, show_reconstruct_results, style_transfer# , show_reconstruct_results_f11, show_reconstruct_results_f22
from model import RGLIndividualSaperateSC

# paths
TRAIN_PATH     = '../data/train.ft.txt'
TEST_PATH      = '../data/test.ft.txt'
TEST_PRE_PATH  = '../data/t.tsv'
POS_TEST_PATH  = '../data/test.pos'
NEG_TEST_PATH  = '../data/test.neg'
POS_TRAIN_PATH = '../data/train.pos'
NEG_TRAIN_PATH = '../data/train.neg'
GLOVE_PATH     = '../data/glove.42B.300d.txt'

small_pos_path   = '../data/amazon_small.pos'
small_neg_path   = '../data/amazon_small.neg'
small_pos        = '../data/amazon_small.pos'
small_neg        = '../data/amazon_small.neg'

small_glove_path = '../data/wordvec.txt'
small_path       = '../data/small.txt'
small_pre_path   = '../data/small_preprocess.tsv'


amazon_train = '../data/amazon_train.tsv'
amazon_test = '../data/amazon_test.tsv'

parser = argparse.ArgumentParser(description='')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-train', action='store_true', default=True, help='train or test')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()


# Parameters setting
args.grad_clip    = 2
args.embed_dim    = 300
args.hidden_dim   = 100
args.batch_size   = 32
args.lr           = 0.0001
args.num_epoch    = 200
args.num_class    = 2
args.max_length   = 20
args.lamda        = 1.0
args.device       = torch.device('cuda')
args.kernel_num   = 100
args.kernel_sizes = '3,4,5'
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.dropout      = 0.1

# Preprocess
if not os.path.exists(TEST_PRE_PATH):
    logger.info('Preprocessing begin...')
    preprocess_write(TEST_PATH, TEST_PRE_PATH)
else:
    logger.info('No need to preprocess!')

# Load data
logger.info('Loading data begin...')
text_field, label_field, train_data, train_iter, dev_data, dev_iter = load_data(amazon_train, amazon_test, args)
text_field.build_vocab(train_data, dev_data, min_freq=10)
label_field.build_vocab(train_data)
logger.info('Length of vocab is: ' + str(len(text_field.vocab)))


args.vocab_size = len(text_field.vocab)
args.word_2_index = text_field.vocab.stoi # tuple of dict({word: index})
args.index_2_word = text_field.vocab.itos # only list of words

# Initial word embedding
logger.info('Getting pre-trained word embedding ...')
args.pretrained_weight = get_pretrained_word_embed(small_glove_path, args, text_field)  


# Build model and train
rgl_net = RGLIndividualSaperateSC(args.vocab_size, args.embed_dim, args.num_class, 
    args.hidden_dim, args.pretrained_weight.numpy(), args.word_2_index['<SOS>'], args).cuda()

if args.snapshot is not None:
    logger.info('Load model from' + args.snapshot)
    rgl_net.load_state_dict(torch.load(args.snapshot))
    if not os.path.exists(small_pos):
        preprocess_pos_neg(small_pos_path, small_pos)
        preprocess_pos_neg(small_neg_path, small_neg)
    pos_iter, neg_iter = load_pos_neg_data(small_pos, small_neg, text_field, args)
    style_transfer(pos_iter, neg_iter, rgl_net, args)
else:
    logger.info('Train model begin...')

    try:
        trainRGL(train_iter=train_iter, dev_iter=dev_iter, train_data=train_data, model=rgl_net, args=args)
    except KeyboardInterrupt:
        print(traceback.print_exc())
        print('\n' + '-' * 89)
        print('Exiting from training early')


# python main.py -snapshot RGLModel/epoch_10_batch_254000_acc_85.2_bestmodel.pt



### python main.py -snapshot RGLModel/IndSep/epoch_18_batch_1900_acc_99.9005634736_bestmodel.pt

# Build s2s model and train
# s2s_model = Seq2Seq(src_nword=args.vocab_size, 
#                     trg_nword=args.vocab_size, 
#                     num_layer=2, 
#                     embed_dim=args.embed_dim, 
#                     hidden_dim=args.hidden_dim, 
#                     max_len=args.max_length, 
#                     trg_soi=args.word_2_index['<SOS>'], 
#                     args=args)
# s2s_model.cuda()
# if args.snapshot is not None:
#     logger.info('Load model from' + args.snapshot)
#     s2s_model.load_state_dict(torch.load(args.snapshot))
#     show_reconstruct_results(dev_iter, s2s_model, args)
#     # if not os.path.exists(small_pos):
#     #     preprocess_pos_neg(small_pos_path, small_pos)
#     #     preprocess_pos_neg(small_neg_path, small_neg)
#     # pos_iter, neg_iter = load_pos_neg_data(small_pos, small_neg, text_field, args)
#     # style_transfer(pos_iter, neg_iter, rgl_net, args)
# else:
#     logger.info('Train model begin...')

#     try:
#         trainS2S(train_iter=train_iter, dev_iter=dev_iter, train_data=train_data, model=s2s_model, args=args)
#     except KeyboardInterrupt:
#         print(traceback.print_exc())
#         print('\n' + '-' * 89)
#         print('Exiting from training early')


















