# This code is heavily modified version of the code presented in the paper "Unsupervised Neural Machine Translation" by Mikel artetxe https://github.com/artetxem/undreamt

from undreamt import data, devices

import torch
import torch.nn as nn
from torch.autograd import Variable
class Translator:
    def __init__(self, encoder_embeddings, encoder, output_layer,src_dict,device=devices.default):
        self.encoder_embeddings = encoder_embeddings
        self.encoder = encoder
        self.device = device
        self.output_layer = output_layer
        self.src_dictionary = src_dict
        self.criterion = nn.NLLLoss(size_average=False)

    def _train(self, mode):
        self.encoder.train(mode)
        self.output_layer.train(mode)
        self.criterion.train(mode)

    def encode(self, sentences, train=False):
        self._train(train)
        ids, lengths = self.src_dictionary.sentences2ids(sentences, eos=True)#

        #might have to replace this with our corruption function
        with torch.no_grad():
            varids = self.device(Variable(torch.LongTensor(ids), requires_grad=False))  # might have to remove  volatile flag
        hidden = self.device(self.encoder.initial_hidden(len(sentences)))
        hidden, context = self.encoder(ids=varids, lengths=lengths, word_embeddings=self.encoder_embeddings, hidden=hidden)
        return hidden, context, lengths

    def mask(self, lengths):
        batch_size = len(lengths)
        max_length = max(lengths)
        if max_length == min(lengths):
            return None
        mask = torch.ByteTensor(batch_size, max_length).fill_(0)
        for i in range(batch_size):
            for j in range(lengths[i], max_length):
                mask[i, j] = 1
        return self.device(mask)

    def predict(self, sentences,NormSenti):
        hidden, context, context_lengths = self.encode(sentences, train = False)
        context_mask = self.mask(context_lengths)
        output = self.output_layer(context,NormSenti,context_mask)
        return torch.argmax(output,dim=1)


    def score(self, src,NormSenti,out, train=False):
        self._train(train)

        # Check batch sizes
        if len(src) != len(out):
            raise Exception('Sentence and hypothesis lengths do not match')

        # Encode
        hidden, context, context_lengths = self.encode(src, train)
        context_mask = self.mask(context_lengths)
        # Output
        output = self.device(self.output_layer(context,NormSenti,context_mask))
        #print("output_raw:-",output)
        lossEG = self.criterion(output, out)

        return lossEG #
