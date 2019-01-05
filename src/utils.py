import re
import os
import torch
import torch.nn as nn
import torchtext.data as data
import torchtext.datasets as datasets
from nltk.corpus import sentiwordnet as swn
import pickle
import numpy as np
import codecs
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


def preprocess(in_path, pos_output_paths, neg_output_paths):
    '''
    Remove labels and split data into 2 files(.pos and .neg)
    '''
    
    pos_writer = open(pos_output_paths, 'w')
    neg_writer = open(neg_output_paths, 'w')
    with open(in_path, 'r') as reader:
        for line in reader:
            if line[9] == '1': # negative
                text = line[11:]
                # neg_writer.write(text.split(': ')[0].lower()) # title
                # neg_writer.write('\t')
                neg_writer.write(text.split(': ')[1].lower())
            if line[9] == '2': # positive
                text = line[11:]
                # pos_writer.write(text.split(': ')[0].lower()) # title
                # pos_writer.write('\t')
                pos_writer.write(text.split(': ')[1].lower())
    pos_writer.close()
    neg_writer.close()

def initialWordEmbedding(fileName,stoi):
    embedding = np.random.random((len(stoi),300))
    with codecs.open(fileName, encoding="utf-8") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            #print("The " + str(i) + " line: " + line)
            line = line.strip().split()
            if len(line) < 10:
                continue
            if line[0] in stoi:
                embedding[stoi[line[0]]] = np.array([float(val) for val in line[-300:]])
    return embedding

