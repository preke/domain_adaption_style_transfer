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
from load_data import preprocess



TRAIN_PATH = '../data/train.ft.txt'
TEST_PATH = '../data/test.ft.txt'
POS_TEST_PATH = '../data/test.pos'
NEG_TEST_PATH = '../data/test.neg'
# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
# args = parser.parse_args()



# Preprocess
preprocess(TEST_PATH, POS_TEST_PATH, NEG_TEST_PATH)

