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
            lower=True, include_lengths=True, preprocessing=data.Pipeline(clean_str),
            pad_token='<PAD>', unk_token='<UNK>', init_token='<SOS>', eos_token='<EOS>')
    label_field = data.Field(batch_first=True, sequential=False, pad_token=None, unk_token=None)
    train_data, train_iter = gen_iter(train_path, text_field, label_field, args)
    dev_data, dev_iter = gen_iter(dev_path, text_field, label_field, args)
    return text_field, label_field, train_data, train_iter, dev_data, dev_iter




# Average = True



# def readSent(fileName,flag):
#     sent_list = []
#     with codecs.open(fileName,"r",encoding = 'utf-8', errors = "ignore") as f:
#         lines = f.readlines()
    
#     for line in lines:
#         line = clean_str(line)
#         words = word_tokenize(line)
#         sent_list.append((words,flag))
    
#     # lenth = 
#     train = sent_list[:int(0.8 * len(sent_list))]
#     dev   = sent_list[int(0.8 * len(sent_list)) :]
#     test  = sent_list[int(0.8 * len(sent_list)) :]
#     return train, dev, test

# def sortSamples(sentence,w2i):
#     sent2idx = []
#     for sent,l in sentence:
#         temp = [w2i[word] for word in sent if word in w2i]
#         if len(temp) > 0:
#             sent2idx.append((len(temp),temp,l))
        
#     sentence = sorted(sent2idx,reverse = True)
    
#     return sentence

# def buildVocab(train_sent):
#     vocab = set()
#     for sent in train_sent:
#         vocab.update(sent[0])
    
#     vocab = list(vocab)
#     logger.info('size of vocab: ' + str(len(vocab)))
#     vocab = ["<pad>"] + vocab
#     w2i = {}
#     i2w = {}
#     for i,w in enumerate(vocab):
#         w2i[w] = i
#         i2w[i] = w
    
#     return vocab,w2i,i2w

# def paddingSentence(batch_sentence,w2i,label,lenths):
    
#     max_lenth = max(lenths)
#     padding_sentence = [sent + [w2i["<pad>"]] * (max_lenth - len(sent)) for sent in batch_sentence]
#     if Average:
#         masks = [[1] * len(sent) + [0] * (max_lenth - len(sent)) for sent in batch_sentence]
#     else:
#         masks = [[0] * (len(sent) - 1) + [1] + [0] * (max_lenth - len(sent)) for sent in batch_sentence]
    
#     return padding_sentence,lenths,label,masks
    
        
# def generateBatch(sort_sent,w2i,batch_size):
#     samples = []
#     labels = []
#     lenths = []
#     for sent in sort_sent:
#         lenths.append(sent[0])
#         samples.append(sent[1])
#         labels.append(sent[2])
#     samples_batch = []
#     lenth_batch = []
#     labels_batch = []
#     mask_batch = []
#     for i in range(0,len(samples),batch_size):
#         current_batch = samples[i : i + batch_size]
#         padding_sentence,lenth,label,mask = paddingSentence(current_batch,w2i,labels[i : i + batch_size],lenths[i : i + batch_size])
#         samples_batch.append(padding_sentence)
#         lenth_batch.append(lenth)
#         labels_batch.append(label)
#         mask_batch.append(mask)
    
#     return samples_batch,lenth_batch,labels_batch,mask_batch



# def get_batches(POS_PATH, NEG_PATH):
#     '''
#     original:  getMRBatch()
#     Get batches of 32
#     '''
#     batch_length = 3
#     train_pos,dev_pos,test_pos = readSent(POS_PATH,1)
#     train_neg,dev_neg,test_neg = readSent(NEG_PATH,0)

#     train_sentence = train_pos + train_neg
#     # vocab, w2i, i2w = buildVocab(train_sentence)
#     vocab, w2i, i2w = torchtext.build_vocab(train_sentence)
#     train_sentence = sortSamples(train_sentence,w2i)
#     train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch = generateBatch(train_sentence, w2i, batch_length)
    
#     dev_sent = dev_pos + dev_neg
#     dev_sentence = sortSamples(dev_sent,w2i)
#     dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch = generateBatch(dev_sentence,w2i,batch_length)
    
#     test_sent = test_pos + test_neg
#     test_sentence = sortSamples(test_sent,w2i)
#     test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch = generateBatch(test_sentence,w2i,batch_length)
    
#     return train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
#             dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
#             test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
#             vocab, w2i







