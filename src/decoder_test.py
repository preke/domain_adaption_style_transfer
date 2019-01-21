import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from params import Params
from utils import Vocab, Hypothesis, word_detector
from typing import Union, List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-31


class EncoderRNN(nn.Module):

  def __init__(self, embed_size, hidden_size, bidi=True, *, rnn_drop: float=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_directions = 2 if bidi else 1
    self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)

  def forward(self, embedded, hidden, input_lengths=None):
    """
    :param embedded: (src seq len, batch size, embed size)
    :param hidden: (num directions, batch size, encoder hidden size)
    :param input_lengths: list containing the non-padded length of each sequence in this batch;
                          if set, we use `PackedSequence` to skip the PAD inputs and leave the
                          corresponding encoder states as zeros
    :return: (src seq len, batch size, hidden size * num directions = decoder hidden size)
    Perform multi-step encoding.
    """
    if input_lengths is not None:
      embedded = pack_padded_sequence(embedded, input_lengths)

    output, hidden = self.gru(embedded, hidden)

    if input_lengths is not None:
      output, _ = pad_packed_sequence(output)

    if self.num_directions > 1:
      # hidden: (num directions, batch, hidden) => (1, batch, hidden * 2)
      batch_size = hidden.size(1)
      hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size,
                                                        self.hidden_size * self.num_directions)
    return output, hidden

  def init_hidden(self, batch_size):
    return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=DEVICE)



