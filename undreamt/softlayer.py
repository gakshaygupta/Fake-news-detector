import torch
import torch.nn as nn
from undreamt.attention import WeightedSum


class Output(nn.Module):
    def __init__(self,hidden_size):
        super(Output, self).__init__()
        self.hidden_size=hidden_size
        self.attention = WeightedSum(hidden_size)
        self.h1_layer = nn.Linear(hidden_size+7,100,bias=True)
        self.h2_layer = nn.Linear(100,6,bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self,enc_output,NormSenti,context_mask):
        #print("Parameters:-",self.h2_layer.weight[1])
        weightedSum=self.attention(enc_output,context_mask) #batch*dim
        concat=torch.cat((weightedSum,NormSenti),1) #batch*(dim+7)
        h1_out = self.h1_layer(concat) #batch*100 #can use ReLU
        h2_out = self.h2_layer(self.relu(h1_out)) #batch*6
        soft_out = self.log_softmax(h2_out) # batch*6
        return soft_out
