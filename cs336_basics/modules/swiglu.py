import torch
import torch.nn as nn
import math


class swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.w1=nn.Parameter(torch.empty((self.d_model,self.d_ff),**self.factory_kwargs))
        self.w2=nn.Parameter(torch.empty((self.d_ff,self.d_model),**self.factory_kwargs))
        self.w3=nn.Parameter(torch.empty((self.d_model,self.d_ff),**self.factory_kwargs))
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # torch.mul(((x@self.w1)/nn.Sigmoid(x)),x@self.w3)@self.w2
        print(self.w1.shape,self.w2.shape,self.w3.shape,x.shape)
        return torch.mul(((x@self.w1)/torch.sigmoid(x@self.w1)),x@self.w3)@self.w2
