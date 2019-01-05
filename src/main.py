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
from utils import preprocess
from dataload import get_batches

# logging
import logging
from logging.config import dictConfig
from log_config import logging_config
dictConfig(logging_config)
logger = logging.getLogger("default_handlers")

TRAIN_PATH = '../data/train.ft.txt'
TEST_PATH = '../data/test.ft.txt'
POS_TEST_PATH = '../data/test.pos'
NEG_TEST_PATH = '../data/test.neg'
POS_TRAIN_PATH = '../data/train.pos'
NEG_TRAIN_PATH = '../data/train.neg'
# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# args = parser.parse_args()




# Preprocess
if not os.path.exists(POS_TRAIN_PATH):
    preprocess(TRAIN_PATH, POS_TRAIN_PATH, NEG_TRAIN_PATH)
    preprocess(TEST_PATH, POS_TEST_PATH, NEG_TEST_PATH)
else:
	logger.info('No need to preprocess!')

# load data
train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
vocab, w2i = get_batches(POS_TEST_PATH, NEG_TEST_PATH)

