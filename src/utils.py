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

def preprocess_write(in_path, out_path):
    '''
    Generate tsv file to load easier
    '''
    writer = open(out_path, 'w')
    with open(in_path, 'r') as reader:
        for line in reader:
            if line[9] == '1': # negative
                text = line[11:]
                writer.write('0\t')
                writer.write(text.split(': ')[1].lower())
            if line[9] == '2': # positive
                text = line[11:]
                writer.write('1\t')
                writer.write(text.split(': ')[1].lower())                
    writer.close()

def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec

def get_pretrained_word_embed(glova_path, args, text_field):
    embedding_dict = load_glove_as_dict(glove_path)
    word_vec_list = []
    for idx, word in enumerate(text_field.vocab.itos):
        if word in embedding_dict:
            try:
                vector = np.array(embedding_dict[word], dtype=float).reshape(1, args.embed_dim)
            except:
                vector = np.random.rand(1, args.embed_dim)
        else:
            vector = np.random.rand(1, args.embed_dim)
        word_vec_list.append(torch.from_numpy(vector))
    wordvec_matrix = torch.cat(word_vec_list)
    return wordvec_matrix







