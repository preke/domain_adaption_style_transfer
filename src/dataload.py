# coding = utf-8
import re
from nltk.tokenize import word_tokenize
import codecs
import random
import torch
import torchtext.data as data
import torchtext.datasets as datasets

# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def gen_iter(path, text_field, label_field, args):
    '''
    Load TabularDataset from path,
    then convert it into a iterator
    return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(path=path, format='tsv', fields=[('label', label_field), ('text', text_field)])
    tmp_iter = data.BucketIterator(tmp_data,
                    batch_size        = args.batch_size,
                    sort_key          = lambda x: len(x.text),
                    sort_within_batch = True,
                    device            = args.device,
                    repeat            = False)
    return tmp_data, tmp_iter

def load_data(train_path, dev_path, args):
    text_field  = data.Field(sequential=True, use_vocab=True, batch_first=True, 
            lower=True, include_lengths=True, preprocessing=data.Pipeline(clean_str), fix_length=args.max_length,
            pad_token='<PAD>', unk_token='<UNK>', init_token='<SOS>', eos_token='<EOS>')
    label_field = data.Field(batch_first=True, sequential=False, pad_token=None, unk_token=None)
    logger.info('Loading Train data begin...')
    train_data, train_iter = gen_iter(train_path, text_field, label_field, args)
    logger.info('Loading Validation data begin...')
    dev_data, dev_iter = gen_iter(dev_path, text_field, label_field, args)
    return text_field, label_field, train_data, train_iter, dev_data, dev_iter







def gen_pos_neg_iter(path, text_field, args):
    tmp_data = data.TabularDataset(path=path, format='tsv', fields=[('text', text_field)])
    tmp_iter = data.BucketIterator(tmp_data,
                    batch_size        = args.batch_size,
                    sort_key          = lambda x: len(x.text),
                    sort_within_batch = True,
                    device            = args.device,
                    repeat            = False)
    return tmp_iter

def load_pos_neg_data(pos_path, neg_path, args):
    text_field  = data.Field(sequential=True, use_vocab=True, batch_first=True, 
            lower=True, include_lengths=True, preprocessing=data.Pipeline(clean_str), fix_length=args.max_length,
            pad_token='<PAD>', unk_token='<UNK>', init_token='<SOS>', eos_token='<EOS>')
    logger.info('Loading pos data begin...')
    pos_iter = gen_pos_neg_iter(pos_path, text_field, args)
    logger.info('Loading Validation data begin...')
    neg_iter = gen_pos_neg_iter(neg_path, text_field, args)
    return pos_iter, neg_iter






