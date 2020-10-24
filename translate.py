# This code is heavily modified version of the code used in the paper "Unsupervised Neural Machine Translation" by Mikel artetxe https://github.com/artetxem/undreamt

import argparse
import sys
import torch
from undreamt.LIAR import LIARPlusDataset
from torch.utils.data import  DataLoader
from sklearn.metrics import accuracy_score
import pickle
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate using a pre-trained model')
    parser.add_argument('--model', help='a model previously trained with train.py')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', default="", help='the input file ')
    parser.add_argument('-o', '--output', default="", help='the output file ')
    args = parser.parse_args()

    # Load model
    translator = torch.load(args.model)
    dataset = LIARPlusDataset(args.input,args.output)
    data_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
    with open("onehot_encoder_decoder.plk","rb") as a :
        enc = pickle.load(a)
    def accuracy(loader = data_loader,accuracy_fn=accuracy_score,batch_size=args.batch_size):
        count = 0
        acc_multi = 0
        acc_binary = 0
        cat2idx = {enc.categories_[0][i]:i for i in range(0,6)}
        for batch_id, sample_batch in enumerate(loader):
            sentences = sample_batch['sentence']
            count+=len(sentences)
            NormSenti = sample_batch["NormSenti"]
            output = sample_batch["output"]
            predicted = translator.predict(sentences,NormSenti)
            #print("output and prediction:-",output,predicted)
            new_out = (output.view(-1)==cat2idx["true"]).int()+(output.view(-1)==cat2idx["mostly-true"]).int()+(output.view(-1)==cat2idx["half-true"]).int()
            new_pred = (predicted.view(-1)==cat2idx["true"]).int()+(output.view(-1)==cat2idx["mostly-true"]).int()+(output.view(-1)==cat2idx["half-true"]).int()
            acc_binary+= accuracy_fn(y_true=new_out,y_pred=new_pred )
            acc_multi += accuracy_fn(y_true=output.view(-1),y_pred=predicted.view(-1))
        return acc_multi*batch_size/count,acc_binary*batch_size/count
    multi,binary=accuracy()
    print("accuracy for the given dataset is :- multi :- {0} binary:- {1}".format(multi,binary))

if __name__ == '__main__':
    main()
