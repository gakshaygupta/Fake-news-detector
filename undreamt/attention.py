
import torch.nn as nn


class WeightedSum(nn.Module):
    def __init__(self, dim):
        super(WeightedSum, self).__init__()
        self.linear_layer = nn.Linear(dim,dim,bias=True)
        self.linear_align = nn.Linear(dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, context, mask):
        # query: batch*dim
        # context: length*batch*dim
        # ans: batch*dim

        context_t = context.transpose(0, 1)  # batch*length*dim
        # Compute Hi score
        layer_first = self.relu(self.linear_layer(context_t)) # batch*length*dim
        align = self.linear_align(layer_first) # batch*length*1

        # Mask alignment scores
        if mask is not None:
            align.squeeze(2).data.masked_fill_(mask, -float('inf'))

        
        attention = self.softmax(align)  # batch*length
        #print("attention shape:-",attention.size())
        #print("context_shape:- ",context_t.size())
        # Computed weighted context
        weighted_context = attention.squeeze(2).unsqueeze(1).bmm(context_t).squeeze(1)  # batch*dim

        return weighted_context
