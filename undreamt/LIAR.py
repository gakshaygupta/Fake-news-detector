from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
class LIARPlusDataset(Dataset):


    def __init__(self, input_file,output_file):
        self.LIAR = pd.read_csv(input_file)
        self.output_data = np.load(output_file)

    def __len__(self):
        return len(self.LIAR)

    def __getitem__(self, idx):
        #print("idx:-{0}".format(idx))
        sentence = self.LIAR.iloc[idx]["16"]
        NormSenti = torch.Tensor(np.array(self.LIAR.iloc[0][["9","10","11","12","13","polarity","subjectivity"]],dtype="float32"))
        output = torch.argmax(torch.LongTensor(self.output_data[idx]),dim=0)# to be filled
        sample = {'sentence': sentence, 'NormSenti': NormSenti, "output":output}
        #sample = {'NormSenti': NormSenti,"output":output}
        return sample
