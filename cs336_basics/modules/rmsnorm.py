import torch
import torch.nn as nn
import math
import numpy as np

class RMS_Norm(nn.Module):
    def __init__(self, d_model:int,eps:float=1e-5,device:torch.device|None=None,dtype:torch.dtype|None=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.weight=nn.Parameter(torch.ones(self.d_model,**self.factory_kwargs))
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        
        v=x.pow(2).mean(dim=-1,keepdim=True)
        
        x_normed=x*torch.rsqrt(v+self.eps)
        
        result=self.weight*x_normed
        
        return result.to(in_dtype)
                    
        