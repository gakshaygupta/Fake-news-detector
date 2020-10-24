# This code is heavily modified version of the code presented in the paper "Unsupervised Neural Machine Translation" by Mikel artetxe https://github.com/artetxem/undreamt


from undreamt import devices
from undreamt.encoder import RNNEncoder
from undreamt.translator import Translator
from undreamt import data
from undreamt.LIAR import LIARPlusDataset
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from undreamt.softlayer import Output
from sklearn.metrics import accuracy_score
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
import os
def main_train():
    # Build argument parser
    parser = argparse.ArgumentParser(description='Train a neural model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora', 'Corpora related arguments')
    corpora_group.add_argument('--input', help='dataframe containing the input ')
    corpora_group.add_argument('--output', help='dataframe containing the output')
    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments;  give pre-trained  embeddings and vocabulary')
    embedding_group.add_argument('--src_embeddings', help='the source language word embeddings')
    embedding_group.add_argument('--src_vocabulary', help='the source language vocabulary')

    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--hidden', type=int, default=600, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--disable_bidirectional', action='store_true', help='use a single direction encoder')
    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch', type=int, default=50, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.3, help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--epochs', type=int, default=300000, help='the number of training iterations for initialization phase (defaults to 300000)')

    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=0, help='save intermediate models at this interval')

    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=1000, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--validation_input', default=(), help='validation input dataframe')
    logging_group.add_argument('--validation_output', default=(), help='validation output dataframe ')
    # change el2s to el2es and add el2hs also we have to validate the translation of the english if parallel data is not available
    # conditions to add for arguments:- el2es,el2hs,s_validation,discriminator_hidden_size,discriminator_type
    # Other
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')

    # Parse arguments
    args = parser.parse_args()
    # Validate arguments
    if args.src_embeddings is None:
        print('Provide src embeddings path')
        sys.exit(-1)

    # Select device
    device = devices.gpu if args.cuda else devices.cpu

    # Create optimizer lists
    src2out_optimizers = []

    # Method to create a module optimizer and add it to the given lists
    def add_optimizer(module, module_name, direction):
        if args.param_init != 0.0:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate)
        direction.append([optimizer,module_name])

    # Load word embeddings
    src_embeddings = src_dictionary = None
    embedding_size = 0

    if args.src_embeddings is not None:
        glove_input_file = r'{0}'.format(args.src_embeddings)
        filename = word2vec_output_file = r'{0}.word2vec'.format(args.src_embeddings)
        l=os.listdir()
        if not filename in l:
            glove2word2vec(glove_input_file, word2vec_output_file)
        gensim_embeddings = KeyedVectors.load_word2vec_format(filename, binary=False)
        print("*********************************************************************Embeddings loaded successfully*********************************************************************")
        src_dictionary = data.Dictionary(gensim_embeddings)
        src_embeddings = nn.Embedding(gensim_embeddings.vectors.shape[0]+1,gensim_embeddings.vectors.shape[1],padding_idx=0)
        src_embeddings.weight.data.copy_(torch.from_numpy(np.concatenate((np.zeros(shape=[1,gensim_embeddings.vectors.shape[1]],dtype="float32"),gensim_embeddings.vectors),axis=0)))
        del gensim_embeddings
        src_embeddings = device(src_embeddings)
        src_embeddings.requires_grad = False
        if embedding_size == 0:
            embedding_size = src_embeddings.weight.data.size()[1]
        if embedding_size != src_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)
    # Build encoder
    encoder = device(RNNEncoder(embedding_size=embedding_size, hidden_size=args.hidden,
                                bidirectional=not args.disable_bidirectional, layers=args.layers, dropout=args.dropout))
    add_optimizer(encoder, 'encoder', src2out_optimizers)

    # Build decoders
    output = device(Output(hidden_size=args.hidden))
    add_optimizer(output, 'output', src2out_optimizers)#



    # Build translators src_encoder_embeddings,src_decoder_embeddings,src_generator,trg_encoder_embeddings,trg_decoder_embeddings,trg_generator
    src2out_translator = Translator(encoder_embeddings=src_embeddings,encoder=encoder,output_layer=output,src_dict = src_dictionary, device=device)#

    #print("optimizers",src2out_optimizers)
    # Build trainers
    trainers = []
    src2out_trainer = None #might have to be changed
    dataset = LIARPlusDataset(args.input,args.output)
    data_loader = DataLoader(dataset,batch_size=args.batch,shuffle=True,num_workers=4)
    src2out_trainer = Trainer(translator=src2out_translator, optimizers=src2out_optimizers, batch_size=args.batch,data_size=dataset.__len__())
    trainers.append(src2out_trainer)



    # Build validators
    src2out_validators = []

    validation_data = LIARPlusDataset(args.validation_input,args.validation_output)
    validation_loader = DataLoader(validation_data,batch_size=args.batch,shuffle=True,num_workers=4)
    src2out_validators.append(Validator(src2out_translator, validation_loader, args.batch, accuracy_fn=accuracy_score))
        # Build loggers
    loggers = []



    loggers.append(Logger('Fake news classification', src2out_trainer, args.log_interval, (src2out_validators), args.encoding))

    def save_models(name ,initialization = True):
            torch.save(src2out_translator, '{0}.{1}.src2out.pth'.format(name, args.save))

    def training(l,data):
            loggers = l
            for step in range(1, args.epochs + 1):
                for batch_id, batch in enumerate(data):
                    for trainer in trainers:
                        trainer.step(batch)


                if args.save is not None and args.save_interval > 0 and step % args.save_interval == 0:
                    save_models('epoch{0}'.format(step))

                if step % args.log_interval == 0:
                    print('EPOCH {0}'.format(step))
                    for logger in loggers:
                        logger.log(step)

            save_models('final_FAKE_LIAR')

    training(loggers,data_loader)

class Trainer:
    def __init__(self, optimizers, translator, batch_size, data_size):
        self.translator = translator
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.reset_stats()
        self.count = 0
        print("number of Optimizers:-",len(self.optimizers))
    def step(self,batch):
        # Reset gradients
        for optimizer in self.optimizers:
            optimizer[0].zero_grad()

        # Read input sentences

        t = time.time()

        src, NormSenti, out = batch["sentence"], batch["NormSenti"], batch["output"]
        self.sentence_count+= len(src)
        self.src_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in src])  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t
        #tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        #print("Read sentence",tot_m,used_m,free_m)
        # Compute loss
        t = time.time()
        loss = self.translator.score(src, NormSenti, out, train=True)
        self.loss += loss.item()
        self.forward_time += time.time() - t
        self.count += 1

        # Backpropagate error + optimize
        t = time.time()

        loss.div(self.batch_size).backward()
        #print(self.translator.output_layer.h2_layer.weight.grad[0])
        for optimizer in self.optimizers:
            optimizer[0].step()
        self.backward_time += time.time() - t

    def reset_stats(self):
        self.src_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.loss = 0
        self.sentence_count = 0


    def avg_loss(self):

        return  self.loss/self.sentence_count

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / self.total_time()


class Validator:
    def __init__(self, translator, validator_loader,batch_size,accuracy_fn=accuracy_score):
        self.translator = translator
        self.loader = validator_loader
        self.batch_size = batch_size
        self.accuracy_fn = accuracy_fn


    def accuracy(self):
        count = 0
        loss = 0
        acc = 0
        for batch_id, sample_batch in enumerate(self.loader):
            sentences = sample_batch['sentence']
            count+=len(sentences)
            NormSenti = sample_batch["NormSenti"]
            output = sample_batch["output"]
            loss += self.translator.score(sentences, NormSenti, output, train=False)
            predicted = self.translator.predict(sentences,NormSenti)
            #print("output and prediction:-",output,predicted)
            acc += self.accuracy_fn(y_pred=output.view(-1),y_true=predicted.view(-1))
        return loss/count ,acc*self.batch_size/count



class Logger:
    def __init__(self, name, trainer, log_interval, validators=(), encoding='utf-8'):
        self.name = name
        self.trainer = trainer
        self.validators = validators
        self.encoding = encoding
        self.log_interval = log_interval
    def log(self, step=0):
        if self.trainer is not None or len(self.validators) > 0:
            print('{0}'.format(self.name))
        if self.trainer is not None:
                lossEG = self.trainer.avg_loss()
                print('  - Training:   {0:10.2f}  ({1:.2f}s: {2:.2f}tok/s src),'.format(lossEG ,self.trainer.total_time(), self.trainer.words_per_second()))  # have to log generator and discriminator loss also
                self.trainer.reset_stats()
        for id, validator in enumerate(self.validators):
            t = time.time()
            loss,accuracy = validator.accuracy()
            print('  - Validation: loss: {0:10.2f} acc: {1:10.2f}     ({2:.2f}s)'.format(loss ,accuracy ,time.time() - t))
