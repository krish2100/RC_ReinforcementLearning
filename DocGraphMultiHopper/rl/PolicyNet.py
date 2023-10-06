import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):

    def __init__(self, state_dim=768):
        super(PolicyNet, self).__init__()
        self.h = nn.Linear(2*state_dim,state_dim)
        self.out = nn.Linear(state_dim,state_dim)
        
    def forward(self, context, choices):
        x = F.relu(self.h(context))
        x = self.out(x)
        x = F.softmax(torch.matmul(choices,x), dim=0) #, dim=1)
        return x