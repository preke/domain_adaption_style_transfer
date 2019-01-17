import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

from utils import *

# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.enc_h_in = nn.Linear(hidden_dim*2, hidden_dim)
        self.prev_s_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, enc_h, prev_s):
        '''
        enc_h  : B x S x 2*H 
        prev_s : B x 1 x H 
        '''
        seq_len = enc_h.size(1) 

        enc_h_in = self.enc_h_in(enc_h) # B x S x H
        prev_s = self.prev_s_in(prev_s).unsqueeze(1)  # B x 1 x H

        h = F.tanh(enc_h_in + prev_s.expand_as(enc_h_in)) # B x S x H
        h = self.linear(h)  # B x S x 1

        alpha = F.softmax(h)
        ctx = torch.bmm(alpha.transpose(2,1), enc_h).squeeze(1) # B x 1 x 2*H

        return ctx 

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len, trg_soi, pre_embedding):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.trg_soi = trg_soi
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.data.copy_(torch.from_numpy(pre_embedding))
        
        self.attention = Attention(hidden_dim) 
        self.decodercell = DecoderCell(embed_dim, hidden_dim)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)


    def forward(self, content, sentiment, target, length, is_train=True):
        # logger.info('Is train: ' + str(is_train))
        if is_train:
            batch_size, target_len = target.size(0), target.size(1)
            dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))

            if torch.cuda.is_available():
                dec_h = dec_h.cuda()

            target = self.embed(target)
            
            for i in range(target_len):                     
                prev_s       = self.decodercell(target[:, i], content, sentiment)
                dec_h[:,i,:] = prev_s # .unsqueeze(1)
            outputs = self.dec2word(dec_h)
        else:
            batch_size = len(length)
            target = Variable(torch.LongTensor([self.trg_soi] * batch_size)).view(batch_size, 1)
            outputs = Variable(torch.zeros(batch_size, int(torch.max(length)), self.vocab_size))

            if torch.cuda.is_available():
                target = target.cuda()
                outputs = outputs.cuda()
            
            for i in range(int(torch.max(length))):
                target = self.embed(target).squeeze(1)                             
                prev_s = self.decodercell(target, content, sentiment)
                output = self.dec2word(prev_s)
                print output
                outputs[:,i,:] = output
                target = output.topk(1)[1]

        return outputs


class DecoderCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(DecoderCell, self).__init__()

        self.input_weights = nn.Linear(embed_dim, hidden_dim*2)
        self.hidden_weights = nn.Linear(hidden_dim, hidden_dim*2)

        self.input_in = nn.Linear(embed_dim, hidden_dim)
        self.hidden_in = nn.Linear(hidden_dim, hidden_dim)

        self.combine_hidden = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, trg_word, content, sentiment):
        '''
        trg_word : B x E
        content   : B x H 
        sentiment      : B x H
        '''
        prev_s = torch.cat((content, sentiment), 1)
        prev_s = self.combine_hidden(prev_s)
        gates = self.input_weights(trg_word) + self.hidden_weights(prev_s)
        reset_gate, update_gate = gates.chunk(2, 1)

        reset_gate = F.sigmoid(reset_gate)
        update_gate = F.sigmoid(update_gate)

        prev_s_tilde = self.input_in(trg_word) + self.hidden_in(prev_s)
        prev_s_tilde = F.tanh(prev_s_tilde)

        prev_s = torch.mul((1-reset_gate), prev_s) + torch.mul(reset_gate, prev_s_tilde)
        return prev_s
        
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class LSTMSC(nn.Module):
    def __init__(self, embedding_num, embedding_size, num_class, hidden_size, pre_embedding):
        super(LSTMSC,self).__init__()
        self.embedding_num = embedding_num
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers = 3
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        self.bi_encoder01 = nn.LSTM(
            self.embedding_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.encoder01 = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=0.2
        )




    def get_state(self,input_line):
        batch_size = input_line.size(0)
        h0_encoder_bi01 = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        c0_encoder_bi01 = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        h0_encoder01 = Variable(torch.zeros(
            self.layers - 1,
            batch_size,
            self.hidden_size
        ))

        c0_encoder01 = Variable(torch.zeros(
            self.layers - 1,
            batch_size,
            self.hidden_size
        ))
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()),(h0_encoder01.cuda(), c0_encoder01.cuda())
    
    def getFeatureDistribution(self,feature):
        return F.softmax(self.linear(feature))
    
    def forward(self,input_line,lenth,alpha,mask):
        embed = self.embedding (input_line)
        hidden_bi01, hidden_01 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output,batch_first = True)
        
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        mask = mask.unsqueeze(2)
        #print(mask.size())
        feature = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        out = self.linear(feature)
        return out,None,None
        
class RGLIndividualSaperateSC(nn.Module):
    def __init__(self, embedding_num, embedding_size, num_class, hidden_size, pre_embedding, w2i, args):
        super(RGLIndividualSaperateSC, self).__init__()
        self.args           = args
        self.embedding_num  = embedding_num
        self.embedding_size = embedding_size
        self.embedding      = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size    = hidden_size
        self.linear         = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers         = 4
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        self.w2i = w2i
        self.bi_encoder01 = nn.LSTM(
            self.embedding_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.bi_encoder02 = nn.LSTM(
            self.embedding_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.encoder01 = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=0.2
        )
        self.encoder02 = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=0.2
        )
        self.class_classifier = nn.Linear(hidden_size,num_class)
        self.class_classifier.weight.data.normal_(0, 0.01)
        self.class_classifier.bias.data.fill_(0)
        self.domain_classifier = nn.Linear(hidden_size,num_class)
        self.domain_classifier.weight.data.normal_(0, 0.01)
        self.domain_classifier.bias.data.fill_(0)
        self.decoder = Decoder(self.embedding_num, self.embedding_size, self.hidden_size, 50, self.w2i, pre_embedding)

    def get_state(self, input_line):
        batch_size = input_line.size(0)
        h0_encoder_bi01 = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        c0_encoder_bi01 = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        
        h0_encoder_bi02 = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        c0_encoder_bi02 = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))

        h0_encoder01 = Variable(torch.zeros(
            self.layers - 1,
            batch_size,
            self.hidden_size
        ))

        c0_encoder01 = Variable(torch.zeros(
            self.layers - 1,
            batch_size,
            self.hidden_size
        ))
        h0_encoder02 = Variable(torch.zeros(
            self.layers - 1,
            batch_size,
            self.hidden_size
        ))

        c0_encoder02 = Variable(torch.zeros(
            self.layers - 1,
            batch_size,
            self.hidden_size
        ))
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()), (h0_encoder_bi02.cuda(), c0_encoder_bi02.cuda()),\
            (h0_encoder01.cuda(), c0_encoder01.cuda()),(h0_encoder02.cuda(), c0_encoder02.cuda())
    
    def extractFeature(self, input_line, lenth, mask):
        embed = self.embedding(input_line)
        hidden_bi01,hidden_bi02, hidden_01,hidden_02 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output01, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output01,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output01, batch_first = True)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output02, (_, _) = self.bi_encoder02(pack_embed, hidden_bi02)
        unpacked_output02,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output02, batch_first = True)
        
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        mask = mask.unsqueeze(2)
        feature01 = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output02,unpacked_len,batch_first = True)
        output02, (src_h_t02, src_c_t) = self.encoder02(pack_output, hidden_02)
        output02,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output02,batch_first = True)
        
        feature02 = torch.sum(output02 * mask, 1) / torch.sum(mask, 1) 

        return feature01,feature02
    
    def reconstruct(self, content, style, input_line, length, is_train=True):
        out = self.decoder(content, style, input_line, length, is_train)
        out = F.log_softmax(out.contiguous().view(-1, self.embedding_num))
        return out


    def forward(self, input_line, lenth, alpha, mask, is_train=True):
        feature01, feature02 = self.extractFeature(input_line, lenth, mask)
        
        reconstruction_out = self.reconstruct(feature01, feature02, input_line, lenth, is_train)
        
        class_out = self.class_classifier(feature02)
        
        reverse_feature = ReverseLayerF.apply(feature01, alpha)
        class_out   = self.class_classifier(feature02)
        domain_out  = self.domain_classifier(reverse_feature)
        feature_out = feature01.mm(feature02.t())
        feature_out = feature_out ** 2
        feature_out = torch.mean(feature_out)
        return class_out, domain_out, feature_out, reconstruction_out




    
