import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets
import RGL

import codecs
from nltk.corpus import sentiwordnet as swn
import pickle
import numpy as np
import pandas as pd
import codecs
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import traceback

import dataload
from model import *


# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

best_results = 0


def eval(dev_iter, model, alpha):
    flag = 0
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0

    for batch in dev_iter:
        sample  = batch.text[0]
        length  = batch.text[1]
        label   = batch.label         
        mask    = generate_mask(torch.max(length), length)
        mask    = Variable(torch.FloatTensor(mask).cuda())
        feature = Variable(sample)
        target  = Variable(label)
        logit,_,_,reconstruct_out = model(feature, length, alpha, mask, is_train=False)
        loss                      = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        size += len(sample)
    avg_loss /= size
    accuracy = 100.0 * float(corrects)/float(size)
    global best_results
    if accuracy > best_results:
        flag = 1
        best_results = accuracy
        logger.info('Evaluation - loss: {:.6f}  acc: {:.1f}%({}/{}) \n'.format(avg_loss, 
                                                                           accuracy, 
                                                                           corrects, 
                                                                           size))
    return accuracy, flag



def generate_mask(max_length, length):
    mask_batch = [ [1]*int(i)+[0]*(int(max_length)-int(i)) for i in list(length)]
    return mask_batch

def trainRGL(train_iter, dev_iter, train_data, model, args):    
    save_dir = "RGLModel/Newdata/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    optimizer        = optim.Adam(model.parameters(), lr=args.lr)
    loss_class       = nn.CrossEntropyLoss().cuda()
    loss_domain      = nn.CrossEntropyLoss().cuda()
    loss_reconstruct = nn.NLLLoss()
    n_epoch          = args.num_epoch
    lamda            = args.lamda
    len_iter         = int(len(train_data)/args.batch_size) + 1
    

    for epoch in range(n_epoch):
        i = 0
        logger.info('In ' + str(i) + ' batch...')
        for batch in train_iter:
            model.train()
            sample  = batch.text[0]
            length  = batch.text[1]
            label   = batch.label            
            p       = float(i + epoch * len_iter) / n_epoch / len_iter
            alpha   = 2. / (1. + np.exp(-10 * p)) - 1
            feature = Variable(sample)
            target  = Variable(label)
            mask    = generate_mask(torch.max(length), length)
            mask    = Variable(torch.FloatTensor(mask).cuda())
            
            model.zero_grad()
            class_out, domain_out, out, reconstruct_out = model(feature, length, alpha, mask)
            feature_iow      = Variable(feature.contiguous().view(-1)).cuda()
            reconstruct_loss = loss_reconstruct(reconstruct_out, feature_iow)
            
            
            ## begin print reconstruct result
            ## end

            err_label   = loss_class(class_out, target)
            err_domain  = loss_domain(class_out, target)
            
            err = err_domain + err_label + lamda * out + reconstruct_loss
            err.backward()
            optimizer.step()
            if i % 100 == 0:
                show_reconstruct_results(dev_iter, model, args, i)
                # acc, flag = eval(dev_iter, model, alpha)
                # save_path = save_dir + "epoch_" + str(epoch) + "_batch_" + str(i) + "_acc_" + str(acc) +"_bestmodel.pt"
                # if flag:
                #     torch.save(model.state_dict(), save_path)
                #     logger.info('Save model to ' + save_path)
                #     logger.info('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, err_ae: %f' \
                #       % (epoch, i, len_iter, err_label.cpu().data.numpy(),
                #          err_domain.cpu().data.numpy(), out, reconstruct_loss))
            i += 1



def show_reconstruct_results(dev_iter, model, args， cnt):
    writer = open('logs_'+str(cnt)+'_.txt', 'w')
    cnt_batch = 0
    for batch in dev_iter:
        logger.info('In ' + str(cnt_batch) + '  batch...')
        sample  = batch.text[0]
        length  = batch.text[1]
        mask    = generate_mask(torch.max(length), length)
        mask    = Variable(torch.FloatTensor(mask).cuda())
        feature = Variable(sample)
        feature01, feature02 = model.extractFeature(feature, length, mask)
        reconstruct_out = model.reconstruct(feature01, feature02, feature, length, is_train=False)
        out_in_batch = reconstruct_out.view(len(length), torch.max(length), args.vocab_size)
        k = 0 
        for i in out_in_batch:
            writer.write(' '.join([args.index_2_word[int(l)] for l in sample[k]]))
            # writer.write('\n')
            writer.write('\n=============\n')
            writer.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(i, dim=1)]))
            writer.write('\n************\n')
            k = k + 1
        cnt_batch += 1
    writer.close()


def style_transfer(pos_iter, neg_iter, model, args):
    
    total_cnt = 0
    model.eval()
    pos_df = [] # id, length, feature, feature1, feature2
    neg_df = [] # id, length, feature, feature1, feature2
    '''
    <type 'int'>
    <class 'torch.Tensor'>;  tensor(65, device='cuda:0')
    <class 'torch.Tensor'>;  torch.Size([100])
    <class 'torch.Tensor'>;  torch.Size([200])
    <class 'torch.Tensor'>;  torch.Size([200])
    '''
    cnt_batch = 0
    for batch in pos_iter:
        logger.info('In ' + str(cnt_batch) + '  pos batch...')
        sample  = batch.text[0]
        length  = batch.text[1]
        mask    = generate_mask(torch.max(length), length)
        mask    = Variable(torch.FloatTensor(mask).cuda())
        feature = Variable(sample)
        feature01, feature02 = model.extractFeature(feature, length, mask)
        for i in range(len(length)):
            pos_df.append([ total_cnt, length[i], feature[i], feature01[i], feature02[i] ])
            total_cnt += 1
        cnt_batch += 1
    
    for batch in neg_iter:
        logger.info('In ' + str(cnt_batch) + '  batch...')
        sample  = batch.text[0]
        length  = batch.text[1]
        feature = Variable(sample)
        mask    = generate_mask(torch.max(length), length)
        mask    = Variable(torch.FloatTensor(mask).cuda())
        feature01, feature02 = model.extractFeature(feature, length, mask)
        for i in range(len(length)):
            neg_df.append([ total_cnt, length[i], feature[i], feature01[i], feature02[i] ])
            total_cnt += 1
        cnt_batch += 1

    pos_df = pd.DataFrame(pos_df, columns=['id', 'length', 'feature', 'feature1', 'feature2'])
    neg_df = pd.DataFrame(neg_df, columns=['id', 'length', 'feature', 'feature1', 'feature2'])

    print pos_df.shape
    print neg_df.shape
    writer = open('pos_neg_log.txt', 'w')
    for index, row in pos_df.iterrows():
        pos = row['feature1']
        sim = []
        for neg in neg_df['feature1']:
            print neg
            sim.append(F.cosine_similarity(pos.unsqueeze(0), neg.unsqueeze(0)))
        max_index = int(np.argmax(np.array(sim)))
        reconstruct_out = model.reconstruct(
                            pos.unsqueeze(0), 
                            neg_df['feature2'][max_index].unsqueeze(0), 
                            row['feature'].unsqueeze(0), 
                            row['length'].unsqueeze(0),
                            is_train=False)
        out_in_batch = reconstruct_out.view(1, args.max_length, args.vocab_size)
        k = 0 
        '''
         Still have problems
        '''
        sample = row['feature']
        for i in out_in_batch:
            writer.write(' '.join([args.index_2_word[int(l)] for l in sample]))
            writer.write('\n\n')
            writer.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(i, dim=1)]))
            writer.write('\n************\n')
            k = k + 1

    
    
    writer.close()

def demo_style_transfer(sent1, sent2, model, args):
    '''
        Input sent1 and sent2,
        Then get the generated sentence with sent1's semantic feature and sent2's style.
    '''
    # content_1, style_1 = model.extract_feature()
    pass




