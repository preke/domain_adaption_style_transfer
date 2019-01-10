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
from utils import preprocess, initialWordEmbedding, preprocess_write
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
args.embedding_dim = 300
args.embedding_num = 0
args.hidden_dim    = 200
args.batch_size    = 32
args.device = torch.device('cuda')

# Preprocess
if not os.path.exists(small_pre_path):
    logger.info('Preprocessing begin...')
    preprocess_write(small_path, small_pre_path)
    # preprocess(TRAIN_PATH, POS_TRAIN_PATH, NEG_TRAIN_PATH)
    # preprocess(TEST_PATH, POS_TEST_PATH, NEG_TEST_PATH)
else:
    logger.info('No need to preprocess!')

# Load data
logger.info('Loading data begin...')
text_field, label_field, train_data, train_iter, dev_data, dev_iter = load_data(small_pre_path, Tsmall_pre_path, args)
logger.info('Loading data Done!')
# Load data
# logger.info('Loading data begin...')
# train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
# dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
# test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
# vocab, w2i = get_batches(POS_TEST_PATH, NEG_TEST_PATH)



# # Initial word embedding
# logger.info('Initial word embedding begin...')
# embedding = initialWordEmbedding(small_glove_path, w2i)    
# # embedding = initialWordEmbedding(GLOVE_PATH, w2i)    

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











