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

import dataload

best_results = 0
KL = True
lr = 1e-3

train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch, \
            dev_samples_batch,dev_lenth_batch,dev_labels_batch,dev_mask_batch, \
            test_samples_batch,test_lenth_batch,test_labels_batch,test_mask_batch, \
            vocab, w2i = dataload.getMRBatch()



def eval(samples,lenth,labels, model,alpha, masks, test = False):
    flag = 0
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for s,le,l,m in zip(samples,lenth,labels,masks):
        feature = Variable(torch.LongTensor(s).cuda())
        target = Variable(torch.LongTensor(l).cuda())
        mask = Variable(torch.FloatTensor(m).cuda())
        logit,_,_ = model(feature,le,alpha,mask)
        loss = F.cross_entropy(logit, target, size_average=False)

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
    return accuracy,flag

def  trainRGL():
    rgl_net = RGL.RGLIndividualSaperateSC(len(vocab),300,2,300,embedding).cuda()
    #rgl_net = RGL.RGLCommonSaperateSC(len(vocab),300,2,300,embedding).cuda()
    #rgl_net = RGL.RGLIndividualSingleSC(len(vocab),300,2,300,embedding).cuda()
    #rgl_net = RGL.RGLCommonSingleSC(len(vocab),300,2,300,embedding).cuda()
    optimizer = optim.Adam(rgl_net.parameters(), lr=lr)
    loss_class = nn.CrossEntropyLoss().cuda()
    loss_domain = nn.CrossEntropyLoss().cuda() #nn.MSELoss().cuda()  #nn.KLDivLoss().cuda() #nn.CrossEntropyLoss().cuda()
    
    n_epoch = 100
    lamda = 1.0
    len_iter = len(train_samples_batch)
    for epoch in range(n_epoch):
        for i,sample,lenth,label,mask in zip(range(len_iter),train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch):
            rgl_net.train()
            p = float(i + epoch * len_iter) / n_epoch / len_iter
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            feature = Variable(torch.LongTensor(sample).cuda())
            target = Variable(torch.LongTensor(label).cuda())
            mask = Variable(torch.FloatTensor(mask).cuda())
            rgl_net.zero_grad()
            
            class_out,domain_out,out = rgl_net(feature,lenth,alpha,mask)
            class_out,domain_out,out = rgl_net(feature,lenth,alpha,mask)
    
            err_label = loss_class(class_out,target)
            err_domain = loss_domain(class_out,target)
            #domain_out = F.log_softmax(domain_out)
            err = err_domain + err_label + lamda * out
            #err = err_label
            err.backward()
            optimizer.step()
            acc,flag = eval(dev_samples_batch,dev_lenth_batch,dev_labels_batch,rgl_net,alpha,dev_mask_batch)
            save_path = "RGLModel/IndSep"
            save_path += " epoch " + str(epoch) + " batch " + str(i) + " bestmodel.pt"
            if flag:
                torch.save(rgl_net.state_dict(), save_path)
                #'''
                print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, i, len_iter, err_label.cpu().data.numpy(),
                     err_domain.cpu().data.numpy(), out))
                '''
                print('epoch: %d, [iter: %d / all %d], err_s_label: %f' \
                  % (epoch, i, len_iter, err_label.cpu().data.numpy()))
                '''
                acc,flag = eval(test_samples_batch,test_lenth_batch,test_labels_batch,rgl_net,alpha,test_mask_batch,True)
                print("The test accuracy is " + str(acc))
            i += 1

# def  trainEquProb():
#     rgl_net = RGL.IndividualSingleSC(len(vocab),300,2,300,embedding).cuda()
#     #rgl_net = RGL.LSTMSC(len(vocab),300,2,300,embedding).cuda()
#     optimizer = optim.Adam(rgl_net.parameters(), lr=lr)
#     loss_class = nn.CrossEntropyLoss().cuda()
#     if KL:
#         loss_domain = nn.KLDivLoss().cuda()#nn.CrossEntropyLoss().cuda() #nn.MSELoss().cuda()  #nn.KLDivLoss().cuda() #nn.CrossEntropyLoss().cuda()
#     else:
#         loss_domain = nn.MSELoss().cuda()
    
#     n_epoch = 100
#     alpha = 1.0
#     lamda = 1.0
#     len_iter = len(train_samples_batch)
#     for epoch in range(n_epoch):
#         for i,sample,lenth,label,mask in zip(range(len_iter),train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch):
#             rgl_net.train()
#             feature = Variable(torch.LongTensor(sample).cuda())
#             target = Variable(torch.LongTensor(label).cuda())
#             mask = Variable(torch.FloatTensor(mask).cuda())
#             rgl_net.zero_grad()
#             class_out,domain_out,out = rgl_net(feature,lenth,alpha,mask)
    
#             err_label = loss_class(class_out,target)
#             #err_domain = loss_domain(class_out,target)
#             if KL:
#                 domain_out = F.log_softmax(domain_out)
#             else:
#                 domain_out = F.softmax(domain_out)
#             target_dis = torch.FloatTensor([0.5,0.5]).cuda().unsqueeze(0).expand(domain_out.size(0),
#                         2).contiguous()
#             err_domain = loss_domain(domain_out,Variable(target_dis))
    
#             err = err_domain + err_label + lamda * out
#             #err = err_label
#             err.backward()
#             optimizer.step()
#             acc,flag = eval(dev_samples_batch,dev_lenth_batch,dev_labels_batch,rgl_net,alpha,dev_mask_batch)
#             save_path = "EquProbModel/IndSing"
#             if KL:
#                 save_path += "KLD"
#             else:
#                 save_path += "MSE"
#             save_path += " epoch " + str(epoch) + " batch " + str(i) + " bestmodel.pt"
#             if flag:
#                 rgl_net.train()
#                 torch.save(rgl_net.state_dict(), save_path)
#                 #'''
#                 print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
#                   % (epoch, i, len_iter, err_label.cpu().data.numpy(),
#                      err_domain.cpu().data.numpy(), out))
#                 '''
#                 print('epoch: %d, [iter: %d / all %d], err_s_label: %f' \
#                   % (epoch, i, len_iter, err_label.cpu().data.numpy()))
#                 '''
#                 acc,flag = eval(test_samples_batch,test_lenth_batch,test_labels_batch,rgl_net,alpha,test_mask_batch,True)
#                 print("The test accuracy is " + str(acc))
#             i += 1

# def  trainLSTMSC():
#     rgl_net = RGL.LSTMSC(len(vocab),300,2,300,embedding).cuda()
#     #rgl_net = RGL.LSTMSC(len(vocab),300,2,300,embedding).cuda()
#     optimizer = optim.Adam(rgl_net.parameters(), lr=lr)
#     loss_class = nn.CrossEntropyLoss().cuda()
    
#     n_epoch = 100
#     alpha = 1.0
#     len_iter = len(train_samples_batch)
#     for epoch in range(n_epoch):
#         for i,sample,lenth,label,mask in zip(range(len_iter),train_samples_batch,train_lenth_batch,train_labels_batch,train_mask_batch):
#             rgl_net.train()
#             feature = Variable(torch.LongTensor(sample).cuda())
#             target = Variable(torch.LongTensor(label).cuda())
#             mask = Variable(torch.FloatTensor(mask).cuda())
#             rgl_net.zero_grad()
#             class_out,_,_ = rgl_net(feature,lenth,alpha,mask)
    
#             err = loss_class(class_out,target)
#             err.backward()
#             optimizer.step()
#             acc,flag = eval(dev_samples_batch,dev_lenth_batch,dev_labels_batch,rgl_net,alpha,dev_mask_batch)
#             save_path = "LSTMSCModel/IndSing"
#             save_path += " epoch " + str(epoch) + " batch " + str(i) + " bestmodel.pt"
#             if flag:
#                 rgl_net.train()
#                 torch.save(rgl_net.state_dict(), save_path)
#                 #'''
#                 print('epoch: %d, [iter: %d / all %d], err_label: %f ' \
#                   % (epoch, i, len_iter, err.cpu().data.numpy()))
#                 '''
#                 print('epoch: %d, [iter: %d / all %d], err_s_label: %f' \
#                   % (epoch, i, len_iter, err_label.cpu().data.numpy()))
#                 '''
#                 acc,flag = eval(test_samples_batch,test_lenth_batch,test_labels_batch,rgl_net,alpha,test_mask_batch,True)
#                 print("The test accuracy is " + str(acc))
#             i += 1

def getTestFeature(model,test_samples_batch,test_lenth_batch,test_mask_batch):
    attr_indep_feature = []
    attr_dep_feature = []
    for s,le,l,m in zip(test_samples_batch,test_lenth_batch,test_mask_batch):
        fea01,fea02 = model.extractFeature(s,l,m)
        attr_indep_feature.append(fea01)
        attr_dep_feature.append(fea02)
    return attr_indep_feature,attr_dep_feature

#trainRGL()
#trainEquProb()
#trainLSTMSC()


