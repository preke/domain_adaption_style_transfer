import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F


class lstm(nn.Module):
    def __init__(self, args):
        super(lstm, self).__init__()
        self.args = args

        # for word embeddings

        self.V = args.word_embedding_num
        self.D = args.word_embedding_length
        self.word_embedding = nn.Embedding(self.V, self.D)
        self.bidirectional = False
        # use pre-trained
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        self.lstm = nn.LSTM(input_size=self.D,
                            hidden_size=self.hidden_dim,
                            batch_first=True,
                            num_layers=self.num_layers)
        # In train function, set args.batch_size as ba.size()[0]
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)

    def init_hidden(self, batch_size, device):
        '''
            Initialize hidden layers with random nums
        '''
        layer_num = 2 if self.bidirectional else 1
        if device == -1:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num)), \
                    Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num)))
        else:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num).cuda()), \
                    Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num).cuda()))

    def forward(self, sentence, hidden):
        sentence = self.word_embedding(sentence)
        # Outputs: output, (h_n, c_n)

        lstm_out, (lstm_h, lstm_c) = self.lstm(sentence, hidden)
        return lstm_h


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
    def __init__(self,embedding_num,embedding_size,num_class,hidden_size,pre_embedding):
        super(RGLIndividualSaperateSC, self).__init__()
        self.embedding_num  = embedding_num
        self.embedding_size = embedding_size
        self.embedding      = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size    = hidden_size
        self.linear         = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers         = 4
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
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
        self.class_classifier.weight.data.normal_(0,0.01)
        self.class_classifier.bias.data.fill_(0)
        self.domain_classifier = nn.Linear(hidden_size,num_class)
        self.domain_classifier.weight.data.normal_(0,0.01)
        self.domain_classifier.bias.data.fill_(0)

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
        #h0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        #c0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()), (h0_encoder_bi02.cuda(), c0_encoder_bi02.cuda()),\
            (h0_encoder01.cuda(), c0_encoder01.cuda()),(h0_encoder02.cuda(), c0_encoder02.cuda())
    
    def extractFeature(self,input_line,lenth,mask):
        embed = self.embedding (input_line)
        hidden_bi01,hidden_bi02, hidden_01,hidden_02 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output01, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output01,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output01,batch_first = True)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output02, (_, _) = self.bi_encoder02(pack_embed, hidden_bi02)
        unpacked_output02,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output02,batch_first = True)
        
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        
        mask = mask.unsqueeze(2)
        #print(mask.size())
        feature01 = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        
        #bilstm_output02, (_, _) = self.bi_encoder02(embed, hidden_bi02)
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output02,unpacked_len,batch_first = True)
        output02, (src_h_t02, src_c_t) = self.encoder02(pack_output, hidden_02)
        output02,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output02,batch_first = True)

         #output01.mean(dim = 1) # feature01  learn no sentiment feature
        
        feature02 = torch.sum(output02 * mask, 1) / torch.sum(mask, 1) #output02.mean(dim = 1) #feature02 learn sentiment feature

        return feature01,feature02
    
    def forward(self,input_line,lenth,alpha,mask):
        feature01,feature02 = self.extractFeature(input_line, lenth, mask)
        class_out = self.class_classifier(feature02)
        domain_out = self.domain_classifier(reverse_feature)
        
        # before 
        reverse_feature = ReverseLayerF.apply(feature01,alpha)
        class_out = self.class_classifier(feature02)
        #domain_out = self.class_classifier(reverse_feature)
        domain_out = self.domain_classifier(reverse_feature)

        # 2 classifier, no need to reverse first

        feature_out = feature01.mm(feature02.t())
        feature_out = feature_out ** 2
        feature_out = torch.mean(feature_out)
        return class_out, domain_out, feature_out







class RGLCommonSaperateSC(nn.Module):
    def __init__(self,embedding_num,embedding_size,num_class,hidden_size,pre_embedding):
        super(RGLCommonSaperateSC,self).__init__()
        self.embedding_num = embedding_num
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers = 4
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
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
        self.class_classifier.weight.data.normal_(0,0.01)
        self.class_classifier.bias.data.fill_(0)
        self.domain_classifier = nn.Linear(hidden_size,num_class)
        self.domain_classifier.weight.data.normal_(0,0.01)
        self.domain_classifier.bias.data.fill_(0)

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
        #h0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        #c0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()), (h0_encoder_bi02.cuda(), c0_encoder_bi02.cuda()),\
            (h0_encoder01.cuda(), c0_encoder01.cuda()),(h0_encoder02.cuda(), c0_encoder02.cuda())
    
    def extractFeature(self,input_line,lenth,mask):
        embed = self.embedding (input_line)
        hidden_bi01,hidden_bi02, hidden_01,hidden_02 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output01, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output01,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output01,batch_first = True)
        '''
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output02, (_, _) = self.bi_encoder02(pack_embed, hidden_bi02)
        unpacked_output02,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output02,batch_first = True)
        '''
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        
        mask = mask.unsqueeze(2)
        #print(mask.size())
        feature01 = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        
        #bilstm_output02, (_, _) = self.bi_encoder02(embed, hidden_bi02)
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output02, (src_h_t02, src_c_t) = self.encoder02(pack_output, hidden_02)
        output02,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output02,batch_first = True)

         #output01.mean(dim = 1) # feature01  learn no sentiment feature
        
        feature02 = torch.sum(output02 * mask, 1) / torch.sum(mask, 1) #output02.mean(dim = 1) #feature02 learn sentiment feature

        return feature01,feature02
    
    def forward(self,input_line,lenth,alpha,mask):
        feature01,feature02 = self.extractFeature(input_line,lenth,mask)
        reverse_feature = ReverseLayerF.apply(feature01,alpha)
        class_out = self.class_classifier(feature02)
        #domain_out = self.class_classifier(reverse_feature)
        domain_out = self.domain_classifier(reverse_feature)
        feature_out = feature01.mm(feature02.t())
        feature_out = feature_out ** 2
        feature_out = torch.mean(feature_out)
        return class_out,domain_out,feature_out

class RGLIndividualSingleSC(nn.Module):
    def __init__(self,embedding_num,embedding_size,num_class,hidden_size,pre_embedding):
        super(RGLIndividualSingleSC,self).__init__()
        self.embedding_num = embedding_num
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers = 4
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
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
        self.class_classifier.weight.data.normal_(0,0.01)
        self.class_classifier.bias.data.fill_(0)
        self.domain_classifier = nn.Linear(hidden_size,num_class)
        self.domain_classifier.weight.data.normal_(0,0.01)
        self.domain_classifier.bias.data.fill_(0)

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
        #h0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        #c0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()), (h0_encoder_bi02.cuda(), c0_encoder_bi02.cuda()),\
            (h0_encoder01.cuda(), c0_encoder01.cuda()),(h0_encoder02.cuda(), c0_encoder02.cuda())
    
    def extractFeature(self,input_line,lenth,mask):
        embed = self.embedding (input_line)
        hidden_bi01,hidden_bi02, hidden_01,hidden_02 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output01, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output01,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output01,batch_first = True)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output02, (_, _) = self.bi_encoder02(pack_embed, hidden_bi02)
        unpacked_output02,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output02,batch_first = True)
        
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        
        mask = mask.unsqueeze(2)
        #print(mask.size())
        feature01 = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        
        #bilstm_output02, (_, _) = self.bi_encoder02(embed, hidden_bi02)
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output02,unpacked_len,batch_first = True)
        output02, (src_h_t02, src_c_t) = self.encoder02(pack_output, hidden_02)
        output02,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output02,batch_first = True)

         #output01.mean(dim = 1) # feature01  learn no sentiment feature
        
        feature02 = torch.sum(output02 * mask, 1) / torch.sum(mask, 1) #output02.mean(dim = 1) #feature02 learn sentiment feature

        return feature01,feature02
    
    def forward(self,input_line,lenth,alpha,mask):
        feature01,feature02 = self.extractFeature(input_line,lenth,mask)
        reverse_feature = ReverseLayerF.apply(feature01,alpha)
        class_out = self.class_classifier(feature02)
        domain_out = self.class_classifier(reverse_feature)
        #domain_out = self.domain_classifier(reverse_feature)
        feature_out = feature01.mm(feature02.t())
        feature_out = feature_out ** 2
        feature_out = torch.mean(feature_out)
        return class_out,domain_out,feature_out

class RGLCommonSingleSC(nn.Module):
    def __init__(self,embedding_num,embedding_size,num_class,hidden_size,pre_embedding):
        super(RGLCommonSingleSC,self).__init__()
        self.embedding_num = embedding_num
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers = 4
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
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
        self.class_classifier.weight.data.normal_(0,0.01)
        self.class_classifier.bias.data.fill_(0)
        self.domain_classifier = nn.Linear(hidden_size,num_class)
        self.domain_classifier.weight.data.normal_(0,0.01)
        self.domain_classifier.bias.data.fill_(0)

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
        #h0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        #c0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()), (h0_encoder_bi02.cuda(), c0_encoder_bi02.cuda()),\
            (h0_encoder01.cuda(), c0_encoder01.cuda()),(h0_encoder02.cuda(), c0_encoder02.cuda())
    
    def extractFeature(self,input_line,lenth,mask):
        embed = self.embedding (input_line)
        hidden_bi01,hidden_bi02, hidden_01,hidden_02 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output01, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output01,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output01,batch_first = True)
        '''
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output02, (_, _) = self.bi_encoder02(pack_embed, hidden_bi02)
        unpacked_output02,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output02,batch_first = True)
        '''
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        
        mask = mask.unsqueeze(2)
        #print(mask.size())
        feature01 = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        
        #bilstm_output02, (_, _) = self.bi_encoder02(embed, hidden_bi02)
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output02, (src_h_t02, src_c_t) = self.encoder02(pack_output, hidden_02)
        output02,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output02,batch_first = True)

         #output01.mean(dim = 1) # feature01  learn no sentiment feature
        
        feature02 = torch.sum(output02 * mask, 1) / torch.sum(mask, 1) #output02.mean(dim = 1) #feature02 learn sentiment feature

        return feature01,feature02
    
    def forward(self,input_line,lenth,alpha,mask):
        feature01,feature02 = self.extractFeature(input_line,lenth,mask)
        reverse_feature = ReverseLayerF.apply(feature01,alpha)
        class_out = self.class_classifier(feature02)
        domain_out = self.class_classifier(reverse_feature)
        #domain_out = self.domain_classifier(reverse_feature)
        feature_out = feature01.mm(feature02.t())
        feature_out = feature_out ** 2
        feature_out = torch.mean(feature_out)
        return class_out,domain_out,feature_out

class IndividualSingleSC(nn.Module):
    def __init__(self,embedding_num,embedding_size,num_class,hidden_size,pre_embedding):
        super(IndividualSingleSC,self).__init__()
        self.embedding_num = embedding_num
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers = 4
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
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
        self.class_classifier.weight.data.normal_(0,0.01)
        self.class_classifier.bias.data.fill_(0)
        self.domain_classifier = nn.Linear(hidden_size,num_class)
        self.domain_classifier.weight.data.normal_(0,0.01)
        self.domain_classifier.bias.data.fill_(0)

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
        #h0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        #c0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()), (h0_encoder_bi02.cuda(), c0_encoder_bi02.cuda()),\
            (h0_encoder01.cuda(), c0_encoder01.cuda()),(h0_encoder02.cuda(), c0_encoder02.cuda())
    
    def extractFeature(self,input_line,lenth,mask):
        embed = self.embedding (input_line)
        hidden_bi01,hidden_bi02, hidden_01,hidden_02 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output01, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output01,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output01,batch_first = True)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output02, (_, _) = self.bi_encoder02(pack_embed, hidden_bi02)
        unpacked_output02,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output02,batch_first = True)
        
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        
        mask = mask.unsqueeze(2)
        #print(mask.size())
        feature01 = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        
        #bilstm_output02, (_, _) = self.bi_encoder02(embed, hidden_bi02)
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output02,unpacked_len,batch_first = True)
        output02, (src_h_t02, src_c_t) = self.encoder02(pack_output, hidden_02)
        output02,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output02,batch_first = True)

         #output01.mean(dim = 1) # feature01  learn no sentiment feature
        
        feature02 = torch.sum(output02 * mask, 1) / torch.sum(mask, 1) #output02.mean(dim = 1) #feature02 learn sentiment feature

        return feature01,feature02

    def forward(self,input_line,lenth,alpha,mask):
        feature01,feature02 = self.extractFeature(input_line,lenth,mask)
        class_out = self.class_classifier(feature02)
        domain_out = self.class_classifier(feature01)
        #domain_out = self.domain_classifier(reverse_feature)
        feature_out = feature01.mm(feature02.t())
        feature_out = feature_out ** 2
        feature_out = torch.mean(feature_out)
        
        return class_out,domain_out,feature_out

class CommonSingleSC(nn.Module):
    def __init__(self,embedding_num,embedding_size,num_class,hidden_size,pre_embedding):
        super(CommonSingleSC,self).__init__()
        self.embedding_num = embedding_num
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_num,embedding_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(embedding_size,num_class)
        self.linear.weight.data.normal_(0,0.01)
        self.linear.bias.data.fill_(0)
        self.layers = 4
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
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
        self.class_classifier.weight.data.normal_(0,0.01)
        self.class_classifier.bias.data.fill_(0)
        self.domain_classifier = nn.Linear(hidden_size,num_class)
        self.domain_classifier.weight.data.normal_(0,0.01)
        self.domain_classifier.bias.data.fill_(0)

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
        #h0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        #c0_encoder_bi = Variable(torch.zeros(4,batch_size,self.hidden_size // 2).cuda())
        return (h0_encoder_bi01.cuda(), c0_encoder_bi01.cuda()), (h0_encoder_bi02.cuda(), c0_encoder_bi02.cuda()),\
            (h0_encoder01.cuda(), c0_encoder01.cuda()),(h0_encoder02.cuda(), c0_encoder02.cuda())
    
    def extractFeature(self,input_line,lenth,mask):
        embed = self.embedding (input_line)
        hidden_bi01,hidden_bi02, hidden_01,hidden_02 = self.get_state(input_line)
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output01, (_, _) = self.bi_encoder01(pack_embed, hidden_bi01)
        unpacked_output01,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output01,batch_first = True)
        '''
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed,lenth,batch_first = True)
        packed_output02, (_, _) = self.bi_encoder02(pack_embed, hidden_bi02)
        unpacked_output02,unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output02,batch_first = True)
        '''
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output01, (src_h_t01, src_c_t) = self.encoder01(pack_output, hidden_01)
        output01,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output01,batch_first = True)
        
        mask = mask.unsqueeze(2)
        #print(mask.size())
        feature01 = torch.sum(output01 * mask, 1) / torch.sum(mask, 1)
        
        #bilstm_output02, (_, _) = self.bi_encoder02(embed, hidden_bi02)
        pack_output = torch.nn.utils.rnn.pack_padded_sequence(unpacked_output01,unpacked_len,batch_first = True)
        output02, (src_h_t02, src_c_t) = self.encoder02(pack_output, hidden_02)
        output02,uppacked_lenth = torch.nn.utils.rnn.pad_packed_sequence(output02,batch_first = True)

         #output01.mean(dim = 1) # feature01  learn no sentiment feature
        
        feature02 = torch.sum(output02 * mask, 1) / torch.sum(mask, 1) #output02.mean(dim = 1) #feature02 learn sentiment feature

        return feature01,feature02

    def forward(self,input_line,lenth,alpha,mask):
        feature01,feature02 = self.extractFeature(input_line,lenth,mask)
        class_out = self.class_classifier(feature02)
        domain_out = self.class_classifier(feature01)
        #domain_out = self.domain_classifier(reverse_feature)
        feature_out = feature01.mm(feature02.t())
        feature_out = feature_out ** 2
        feature_out = torch.mean(feature_out)
        
        return class_out,domain_out,feature_out
    
