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
        logit,_,_,reconstruct_out = model(feature,length,alpha,mask)
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
    save_dir = "RGLModel/IndSep/"
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
            print reconstruct_out.size()
            reconstruct_out = reconstruct_out.view(args.batch_size, args.max_length, args.vocab_size)
            
            print [args.index_2_word[i] for i in j[i] for j in reconstruct_out]
            ## end

            err_label   = loss_class(class_out, target)
            err_domain  = loss_domain(class_out, target)
            
            # err = err_domain + err_label + lamda * out + reconstruct_loss
            # err.backward()
            # optimizer.step()
            # if i % 100 == 0:
            #     acc, flag = eval(dev_iter, model, alpha)
            #     save_path = save_dir + "epoch_" + str(epoch) + "_batch_" + str(i) + "_acc_" + str(acc) +"_bestmodel.pt"
            #     if flag:
            #         torch.save(model.state_dict(), save_path)
            #         logger.info('Save model to ' + save_path)
            #         logger.info('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, err_ae: %f' \
            #           % (epoch, i, len_iter, err_label.cpu().data.numpy(),
            #              err_domain.cpu().data.numpy(), out, reconstruct_loss))
            i += 1


def demo_model(sent1, sent2, model, w2i):
    '''
        Input sent1 and sent2,
        Then get the generated sentence with sent1's semantic feature and sent2's style.
    '''
    # content_1, style_1 = model.extract_feature()
    pass
