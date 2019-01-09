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
from utils import preprocess, initialWordEmbedding
from dataload import get_batches
from train import eval, trainRGL


# paths
TRAIN_PATH     = '../data/train.ft.txt'
TEST_PATH      = '../data/test.ft.txt'
POS_TEST_PATH  = '../data/test.pos'
NEG_TEST_PATH  = '../data/test.neg'
POS_TRAIN_PATH = '../data/train.pos'
NEG_TRAIN_PATH = '../data/train.neg'
GLOVE_PATH     = '../data/glove.42B.300d.txt'

small_pos_path   = '../data/small.pos'
small_neg_path   = '../data/small.neg'
small_glove_path = '../data/small_glove.txt'
# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# args = parser.parse_args()




# Preprocess
if not os.path.exists(POS_TRAIN_PATH):
    logger.info('Preprocessing begin...')
    preprocess(TRAIN_PATH, POS_TRAIN_PATH, NEG_TRAIN_PATH)
    preprocess(TEST_PATH, POS_TEST_PATH, NEG_TEST_PATH)
else:
    logger.info('No need to preprocess!')

# Load data
logger.info('Loading data begin...')
train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
vocab, w2i = get_batches(small_pos_path, small_neg_path)

# Initial word embedding
logger.info('Initial word embedding begin...')
embedding = initialWordEmbedding(GLOVE_PATH, w2i)    

# Train RGL()
logger.info('Training RGL begin...')
trainRGL(train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
        dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
        test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
        vocab, w2i, embedding)













