import os

import torch
import torch.nn as nn
import torchtext.data as data
import torchtext.datasets as datasets
import RGL

import codecs
from nltk.corpus import sentiwordnet as swn
import pickle
import numpy as np
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


def eval(samples,lenth,labels, model,alpha, masks, test = False):
    flag = 0
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for s,le,l,m in zip(samples,lenth,labels,masks):
        feature                   = Variable(torch.LongTensor(s).cuda())
        target                    = Variable(torch.LongTensor(l).cuda())
        mask                      = Variable(torch.FloatTensor(m).cuda())
        logit,_,_,reconstruct_out = model(feature,le,alpha,mask)
        loss                      = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        size += len(s)

    avg_loss /= size
    accuracy = 100.0 * float(corrects)/float(size)
    if not test:
        global best_results
        if accuracy > best_results:
            flag = 1
            best_results = accuracy
            print('\nEvaluation - loss: {:.6f}  acc: {:.1f}%({}/{}) \n'.format(avg_loss, 
                                                                           accuracy, 
                                                                           corrects, 
                                                                           size))
    return accuracy, flag



def generate_mask(max_length, length):
    mask_batch = [ [1]*int(i)+[0]*(int(max_length)-int(i)) for i in list(length)]
    return mask_batch
'''
def trainRGL(train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
            dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
            test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
            vocab, w2i, embedding):
'''
def trainRGL(train_iter, dev_iter, train_data, model, args):    
    optimizer        = optim.Adam(model.parameters(), lr=args.lr)
    loss_class       = nn.CrossEntropyLoss().cuda()
    loss_domain      = nn.CrossEntropyLoss().cuda()
    loss_reconstruct = nn.NLLLoss()
    n_epoch          = args.num_epoch
    lamda            = args.lamda
    len_iter         = int(len(train_data)/args.batch_size) + 1
    

    for epoch in range(n_epoch):
        
        # for i, sample, lenth, label, mask in zip(range(len_iter),train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch):
        i = 0
        for batch in train_iter:
            logger.info('Batch ' + str(i))
            model.train()
            sample  = batch.text[0]
            length  = batch.text[1]
            label   = batch.label            
            p       = float(i + epoch * len_iter) / n_epoch / len_iter
            alpha   = 2. / (1. + np.exp(-10 * p)) - 1
            feature = Variable(sample)
            target  = Variable(label)
            
            mask    = generate_mask(sample.size()[1], length)
            mask    = Variable(torch.FloatTensor(mask).cuda())
            # print feature.size()
            # print feature

            model.zero_grad()
            # reconstruct parts
            class_out, domain_out, out, reconstruct_out = model(feature, length, alpha, mask)
            feature_iow     = Variable(feature.contiguous().view(-1)).cuda()
            # reconstruct_out = Variable(reconstruct_out.view(batch_size, max(lenth)).cuda())
            # print(reconstruct_out.size())

            
            loss = loss_reconstruct(reconstruct_out, feature_iow)
            
   
            err_label   = loss_class(class_out, target)
            err_domain  = loss_domain(class_out, target)
            
            #domain_out = F.log_softmax(domain_out)
            err = err_domain + err_label + lamda * out + loss
            # #err = err_label
            # err.backward()
            # optimizer.step()
            # acc, flag = eval(dev_samples_batch, dev_lenth_batch, dev_labels_batch, model, alpha, dev_mask_batch)
            
            # save_path = "RGLModel/IndSep/"
            # if not os.path.exists(save_path):
            #     os.mkdir(save_path)
            # save_path += " epoch " + str(epoch) + " batch " + str(i) + " bestmodel.pt"
            # if flag:
            #     torch.save(model.state_dict(), save_path)
                
            #     logger.info('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
            #       % (epoch, i, len_iter, err_label.cpu().data.numpy(),
            #          err_domain.cpu().data.numpy(), out))
                
                
            #     acc, flag = eval(test_samples_batch, test_lenth_batch, test_labels_batch, model,alpha, test_mask_batch, True)
            #     logger.info("The test accuracy is " + str(acc))
            i += 1


def demo_model(sent1, sent2, model, w2i):
    '''
    '''
    pass
