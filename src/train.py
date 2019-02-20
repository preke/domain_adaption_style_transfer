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
from utils import preprocess_write, get_pretrained_word_embed, preprocess_pos_neg
from dataload import load_data, load_pos_neg_data



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
        # mask    = generate_mask(torch.max(length), length)
        # mask    = Variable(torch.FloatTensor(mask).cuda())
        feature = Variable(sample)
        target  = Variable(label)
        
        logit, _, _, reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()], alpha, is_train=False)
        loss                      = F.cross_entropy(logit, target, size_average=False)

        feature_iow      = Variable(feature.contiguous().view(-1)).cuda()
        loss_reconstruct = nn.NLLLoss()
        reconstruct_loss = loss_reconstruct(reconstruct_out, feature_iow)


        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        size += len(sample)


    avg_loss /= size
    accuracy = 100.0 * float(corrects)/float(size)
    global best_results
    logger.info('Evaluation - loss: {:.6f}  acc: {:.1f}%({}/{}) err_ae: {:.6f}\n'.format(avg_loss, 
                                                                           accuracy, 
                                                                           corrects, 
                                                                           size,
                                                                           reconstruct_loss))
    if accuracy > best_results:
        flag = 1
        best_results = accuracy
        
    return accuracy, flag, reconstruct_loss

# def generate_mask(max_length, length):
#     mask_batch = [ [1]*int(i)+[0]*(int(max_length)-int(i)) for i in list(length)]
#     return mask_batch

def trainRGL(train_iter, dev_iter, train_data, model, args):    
    save_dir = "RGLModel/concat/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    optimizer        = optim.Adam(model.parameters(), lr=args.lr)
    loss_class       = nn.CrossEntropyLoss().cuda()
    loss_domain      = nn.CrossEntropyLoss().cuda()
    loss_reconstruct = nn.NLLLoss()
    n_epoch          = args.num_epoch
    lamda            = args.lamda
    len_iter         = int(len(train_data)/args.batch_size) + 1
    
    cnt_epoch = 0
    cnt_batch = 0
    for epoch in range(n_epoch): 
        logger.info('In ' + str(cnt_epoch) + ' epoch... ')
        for batch in train_iter:
            model.train()
            sample  = batch.text[0]
            length  = batch.text[1]
            label   = batch.label            
            p       = float(cnt_batch + epoch * len_iter) / n_epoch / len_iter
            alpha   = 2. / (1. + np.exp(-10 * p)) - 1
            feature = Variable(sample)
            target  = Variable(label)
            # mask    = generate_mask(torch.max(length), length)
            # mask    = Variable(torch.FloatTensor(mask).cuda())
            
            
            # class_out, domain_out, out, reconstruct_out = model(feature, length, alpha, mask)
            class_out, domain_out, out, reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()], alpha)
            feature_iow     = Variable(feature[:,1:].contiguous().view(-1)).cuda()


            optimizer.zero_grad()
            reconstruct_loss = loss_reconstruct(reconstruct_out, feature_iow)
            
            err_label   = loss_class(class_out, target)
            err_domain  = loss_domain(class_out, target)
            

            '''
            How to give weights to each loss
            '''
            err = err_domain + err_label + lamda * out + 5*reconstruct_loss
            err.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()
            if cnt_batch % 2000 == 0:
                
                acc, flag, eval_aeloss = eval(dev_iter, model, alpha)
                show_reconstruct_results(dev_iter, model, args, cnt_batch, eval_aeloss)
                save_path = save_dir + "epoch_" + str(epoch) + "_batch_" + str(cnt_batch) + "_acc_" + str(acc) +"_bestmodel.pt"
                if flag:
                    torch.save(model.state_dict(), save_path)
                    logger.info('Save model to ' + save_path)
                    logger.info('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, err_ae: %f' \
                      % (epoch, cnt_batch, len_iter, err_label.cpu().data.numpy(),
                         err_domain.cpu().data.numpy(), out, reconstruct_loss))

                
                # Try to illustrate the Style Transfer
                # small_pos_path   = '../data/amazon_small.pos'
                # small_neg_path   = '../data/amazon_small.neg'
                # small_pos        = '../data/amazon_small.pos'
                # small_neg        = '../data/amazon_small.neg'
                
                # if not os.path.exists(small_pos):
                #     preprocess_pos_neg(small_pos_path, small_pos)
                #     preprocess_pos_neg(small_neg_path, small_neg)
                # pos_iter, neg_iter = load_pos_neg_data(small_pos, small_neg, text_field, args)
                # style_transfer(pos_iter, neg_iter, model, args, cnt_batch)

            cnt_batch += 1
        cnt_epoch += 1

def show_reconstruct_results(dev_iter, model, args, cnt=0, reconstruct_loss=0.0):
    writer = open('f12_logs_'+str(cnt)+'_' + str(float(reconstruct_loss)) + '_.txt', 'w')
    cnt_batch = 0
    for batch in dev_iter:
        # logger.info('In ' + str(cnt_batch) + '  batch...')
        sample  = batch.text[0]
        length  = batch.text[1]
        # mask    = generate_mask(torch.max(length), length)
        # mask    = Variable(torch.FloatTensor(mask).cuda())
        feature = Variable(sample)
        


        feature01, feature02, output = model.extractFeature(feature[:, :-1], [i-1 for i in length.tolist()])
        reconstruct_out = model.reconstruct(feature01, feature02, output, feature, [i-1 for i in length.tolist()], is_train=False)
        out_in_batch = reconstruct_out.contiguous().view(len(length), args.max_length, args.vocab_size)
        k = 0 
        for i in out_in_batch:
            writer.write(' '.join([args.index_2_word[int(l)] for l in sample[k]]))
            # writer.write('\n')
            writer.write('\n=============\n')
            writer.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(i, dim=-1)]))
            writer.write('\n\n')
            k = k + 1
        cnt_batch += 1
    writer.close()

def style_transfer(pos_iter, neg_iter, model, args):
    
    total_cnt = 0
    model.eval()
    pos_df = [] # id, length, feature, feature1, feature2
    neg_df = [] # id, length, feature, feature1, feature2
    
    cnt_batch = 0
    writer1 = open('pos_batch'+'_.txt', 'w')
    writer2 = open('pos_single'+'_.txt', 'w')
    
    for batch in pos_iter:
        sample  = batch.text[0]
        length  = batch.text[1]
        feature = Variable(sample)
        feature01, feature02, output = model.extractFeature(feature[:, :-1], [i-1 for i in length.tolist()])
        
        reconstruct_out = model.reconstruct(feature01.data, feature02.data, output.data, feature.data, [i-1 for i in length.tolist()], is_train=False)
        out_in_batch = reconstruct_out.contiguous().view(len(length), args.max_length, args.vocab_size)
        k = 0


        for i in out_in_batch:
            writer1.write(' '.join([args.index_2_word[int(l)] for l in sample[k]]))
            # writer.write('\n')
            writer1.write('\n=============\n')
            writer1.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(i, dim=-1)]))
            writer1.write('\n\n')
            k = k + 1
        
        tmp_out_in_batch = output
        for tmp in range(len(length)):
            pos           = feature01.data[tmp].unsqueeze(0)
            neg           = feature02.data[tmp].unsqueeze(0)
            # pos_attention = output.data[tmp].unsqueeze(0)
            # feature       = feature.data[tmp].unsqueeze(0)
            # length        = (length[tmp]-1).unsqueeze(0)
            for i in range(5): # batch size 32 (2^5)
                pos           = torch.cat((pos, pos))
                neg           = torch.cat((neg, neg))
                # feature       = torch.cat((feature, feature))
                # pos_attention = torch.cat((pos_attention, pos_attention))
                # length        = torch.cat((length, length))

            reconstruct_out = model.reconstruct(pos, neg, output.data, feature.data, [i-1 for i in length.tolist()], is_train=False)
            out_in_batch = reconstruct_out.contiguous().view(32, args.max_length, args.vocab_size)
            if tmp_out_in_batch == out_in_batch:
                print 'same' + str(tmp)
            tmp_out_in_batch = out_in_batch    
            for i in out_in_batch[:1]:
                writer2.write(str())
                writer2.write(' '.join([args.index_2_word[int(l)] for l in sample[tmp]]))
                # writer.write('\n')
                writer2.write('\n=============\n')
                writer2.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(i, dim=-1)]))
                writer2.write('\n\n')
                #k = k + 1
    
        writer1.close()    
        writer2.close()
        break
    # for batch in neg_iter:
    #     sample  = batch.text[0]
    #     length  = batch.text[1]
    #     feature = Variable(sample)
    #     feature01, feature02, output = model.extractFeature(feature[:, :-1], [i-1 for i in length.tolist()])
    #     for i in range(len(length)):
    #         neg_df.append([ total_cnt, length[i], feature[i], feature01[i], feature02[i], output[i] ])
    #         total_cnt += 1
    #     cnt_batch += 1

    # pos_df = pd.DataFrame(pos_df, columns=['id', 'length', 'feature', 'feature1', 'feature2', 'hiddens'])
    # neg_df = pd.DataFrame(neg_df, columns=['id', 'length', 'feature', 'feature1', 'feature2', 'hiddens'])

    # print pos_df.shape
    # print neg_df.shape
    '''
    writer = open('pos_pos_log_'+'_.txt', 'w')
    for index, row in pos_df[:100].iterrows():
        pos           = row['feature1'].unsqueeze(0)
        pos_attention = row['hiddens'].unsqueeze(0)
        feature       = row['feature'].unsqueeze(0)
        length        = row['length'].unsqueeze(0)
        # pos_f2        = row['length'].unsqueeze(0)
        sim           = []
        for neg in neg_df['feature1']:
            sim.append(F.cosine_similarity(pos, neg.unsqueeze(0)))
        max_index = int(np.argmax(np.array(sim)))
        # neg = neg_df['feature2'][max_index].unsqueeze(0)
        neg = row['feature2'].unsqueeze(0)

        for i in range(5): # batch size 32 (2^5)
            pos           = torch.cat((pos, pos))
            neg           = torch.cat((neg, neg))
            feature       = torch.cat((feature, feature))
            pos_attention = torch.cat((pos_attention, pos_attention))
            length        = torch.cat((length, length))

        # print '++++++++'
        # print pos.size()
        # print neg.size()
        # print pos_attention.size()

        reconstruct_out = model.reconstruct(
                        pos, 
                        neg, 
                        pos_attention,
                        feature, 
                        [i-1 for i in length.tolist()],
                        is_train = False)

        out_in_batch = reconstruct_out.contiguous().view(32, args.max_length, args.vocab_size)
        k = 0 
        sample = row['feature']
        neg_sample = neg_df['feature'][max_index]
        for generated in out_in_batch:
            writer.write(' '.join([args.index_2_word[int(l)] for l in sample]))
            writer.write('\n\n')

            # writer.write(' '.join([args.index_2_word[int(l)] for l in neg_sample]))
            # writer.write('\n\n')
            
            writer.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(generated, dim=1)]))
            writer.write('\n************\n')
            k = k + 1

    writer.close()
    '''

def demo_style_transfer(sent1, sent2, model, args):
    '''
        Input sent1 and sent2,
        Then get the generated sentence with sent1's semantic feature and sent2's style.
    '''
    # content_1, style_1 = model.extract_feature()
    pass


def trainS2S(train_iter, dev_iter, train_data, model, args):
    save_dir = "RGLModel/Newdata/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    optimizer        = optim.Adam(model.parameters(), lr=args.lr)
    loss_reconstruct = nn.NLLLoss()
    n_epoch          = args.num_epoch
    lamda            = args.lamda
    len_iter         = int(len(train_data)/args.batch_size) + 1
    cnt_epoch = 0
    cnt_batch = 0
    for epoch in range(n_epoch): 
        logger.info('In ' + str(cnt_epoch) + ' epoch... ')
        for batch in train_iter:
            model.train()
            sample  = batch.text[0]
            length  = batch.text[1]
            p       = float(cnt_batch + epoch * len_iter) / n_epoch / len_iter
            alpha   = 2. / (1. + np.exp(-10 * p)) - 1
            feature = Variable(sample)

            reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()], feature[:, :-1])
            feature_iow     = Variable(feature[:,1:].contiguous().view(-1)).cuda()

            optimizer.zero_grad()
            reconstruct_loss = loss_reconstruct(reconstruct_out, feature_iow)
            err = reconstruct_loss
            err.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()
            if cnt_batch % 100 == 0:
                # logger.info('Train_loss:{:.6f}'.format(reconstruct_loss))
                avg_loss = eval_S2S(dev_iter, model)
            if cnt_batch % 1000 == 0:                
                show_reconstruct_results_S2S(dev_iter, model, args, cnt_batch, avg_loss)
            cnt_batch += 1
        cnt_epoch += 1

def eval_S2S(dev_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for batch in dev_iter:
        sample  = batch.text[0]
        length  = batch.text[1]
        feature = Variable(sample)
        
        reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()])
        feature_iow      = Variable(feature.contiguous().view(-1)).cuda() # the whole sentence

        loss_reconstruct = nn.NLLLoss()
        reconstruct_loss = loss_reconstruct(reconstruct_out, feature_iow)


        avg_loss += reconstruct_loss.data
        size += len(sample)

    avg_loss /= size
    logger.info('Evaluation - Train_loss:{:.6f}, eva_loss: {:.6f}\n'.format(reconstruct_loss, avg_loss))
    return avg_loss

def show_reconstruct_results_S2S(dev_iter, model, args, cnt, avg_loss):
    writer = open('s2s_logs_'+str(cnt) + '__' + str(float(avg_loss)) + '_.txt', 'w')
    cnt_batch = 0
    for batch in dev_iter:
        sample  = batch.text[0]
        length  = batch.text[1]
        feature = Variable(sample)
        
        reconstruct_out = model(feature[:, :-1], [i-1 for i in length.tolist()])
        out_in_batch = reconstruct_out.contiguous().view(len(length), args.max_length, args.vocab_size)
        k = 0 
        for i in out_in_batch:
            writer.write(' '.join([args.index_2_word[int(l)] for l in sample[k]]))
            writer.write('\n=============\n')
            writer.write(' '.join([args.index_2_word[int(j)] for j in torch.argmax(i, dim=-1)]))
            writer.write('\n************\n')
            k = k + 1
        cnt_batch += 1
    writer.close()

