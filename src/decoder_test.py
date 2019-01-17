class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.hidden_dim  = args.hidden_dim
        self.max_len     = args.max_len
        self.vocab_size  = args.vocab_size
        
        self.embed       = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.data.copy_(torch.from_numpy(pre_embedding))
        
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)
        self.GRU = nn.GRU


    def forward(self, content, sentiment, target, length, is_train=True):
        if is_train:
            batch_size, target_len = target.size(0), target.size(1)
            dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))

            if torch.cuda.is_available():
                dec_h = dec_h.cuda()

            target = self.embed(target)
            
            for i in range(target_len):   
                print 'train'
                print target[:, i]

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
                print 'evaluate'
                print target
                prev_s = self.decodercell(target, content, sentiment)
                output = self.dec2word(prev_s)
                outputs[:,i,:] = output
                target = output.topk(1)[1]

        return outputs