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
from utils import preprocess_write, get_pretrained_word_embed
from dataload import load_data
from train import eval, trainRGL, demo_model
from model import RGLIndividualSaperateSC

# paths
TRAIN_PATH     = '../data/train.ft.txt'
TEST_PATH      = '../data/test.ft.txt'
TEST_PRE_PATH  = '../data/test_preprocess.tsv'
POS_TEST_PATH  = '../data/test.pos'
NEG_TEST_PATH  = '../data/test.neg'
POS_TRAIN_PATH = '../data/train.pos'
NEG_TRAIN_PATH = '../data/train.neg'
GLOVE_PATH     = '../data/glove.42B.300d.txt'

small_pos_path   = '../data/small.pos'
small_neg_path   = '../data/small.neg'
small_glove_path = '../data/small_glove.txt'
small_path       = '../data/small.txt'
small_pre_path   = '../data/small_preprocess.txt'


parser = argparse.ArgumentParser(description='')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-train', action='store_true', default=True, help='train or test')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()


# Parameters setting
args.embed_dim  = 300
args.hidden_dim = 200
args.batch_size = 32
args.lr         = 0.0001
args.num_epoch  = 20
args.num_class  = 2
args.max_length = 200
args.lamda      = 1.0
args.device     = torch.device('cuda')


# Preprocess
if not os.path.exists(small_pre_path):
    logger.info('Preprocessing begin...')
    preprocess_write(small_path, small_pre_path)
else:
    logger.info('No need to preprocess!')

# Load data
logger.info('Loading data begin...')
text_field, label_field, train_data, train_iter, dev_data, dev_iter = load_data(small_pre_path, small_pre_path, args)
text_field.build_vocab(train_data, min_freq=5)
logger.info('Length of vocab is: ' + str(len(text_field.vocab)))

args.vocab_size = len(text_field.vocab)

args.word_2_index = text_field.vocab.stoi
args.index_2_word = text_field.vocab.itos



# Initial word embedding
logger.info('Getting pre-trained word embedding ...')
args.pretrained_weight = get_pretrained_word_embed(small_glove_path, args, text_field)  


# Build model and train
rgl_net = RGLIndividualSaperateSC(args.vocab_size, args.embed_dim, args.num_class, 
    args.hidden_dim, args.pretrained_weight, args.word_2_index).cuda()

if args.snapshot is not None:
    logger.info('Load model from' + args.snapshot)
    pass
else:
    logger.info('Train model begin...')
    try:
        trainRGL(train_iter=train_iter, vali_iter=vali_iter, model=rgl_net, args=args)
    except KeyboardInterrupt:
        print(traceback.print_exc())
        print('\n' + '-' * 89)
        print('Exiting from training early')

# # Train RGL()
# if args.train:
#     logger.info('Training begin...')
#     trainRGL(train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
#             dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
#             test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
#             vocab, w2i, embedding)
# else:
#     logger.info('Test begin...')
#     model = RGLIndividualSaperateSC(len(vocab), 300, 2, 200, embedding, w2i).cuda()
#     model = load_state_dict(torch.load(args.snapshot))
#     demo_model(sent1, sent2, model, w2i)











