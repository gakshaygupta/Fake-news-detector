# This code is heavily modified version of the code presented in the paper "Unsupervised Neural Machine Translation" by Mikel artetxe

from undreamt import data

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, bidirectional=False, layers=1, dropout=0):
        super(RNNEncoder, self).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even for bidirectional encoders')
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size // self.directions
        self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, embedding_size, padding_idx=0)
        self.rnn = nn.GRU(embedding_size, self.hidden_size, bidirectional=bidirectional, num_layers=layers,
                          dropout=dropout)

    def forward(self, ids, lengths, word_embeddings, hidden):
        sorted_lengths = sorted(lengths, reverse=True)
        is_sorted = sorted_lengths == lengths
        is_varlen = sorted_lengths[0] != sorted_lengths[-1]
        if not is_sorted:
            true2sorted = sorted(range(len(lengths)), key=lambda x: -lengths[x])
            sorted2true = sorted(range(len(lengths)), key=lambda x: true2sorted[x])
            ids = torch.stack([ids[:, i] for i in true2sorted], dim=1)
            lengths = [lengths[i] for i in true2sorted]
        
        embeddings = word_embeddings(data.word_ids(ids)) + self.special_embeddings(data.special_ids(ids))
        if is_varlen:
            embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths)
        output, hidden = self.rnn(embeddings, hidden)
        if self.bidirectional:
            hidden = torch.stack([torch.cat((hidden[2*i], hidden[2*i+1]), dim=1) for i in range(self.layers)])
        if is_varlen:
            output = nn.utils.rnn.pad_packed_sequence(output)[0]
        if not is_sorted:
            hidden = torch.stack([hidden[:, i, :] for i in sorted2true], dim=1)
            output = torch.stack([output[:, i, :] for i in sorted2true], dim=1)
        return hidden, output   #[seq_len,batch,hidden_size]

    def initial_hidden(self, batch_size):
        return Variable(torch.zeros(self.layers*self.directions, batch_size, self.hidden_size), requires_grad=False)
