import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight=nn.Parameter(torch.empty((self.num_embeddings,self.embedding_dim),**factory_kwargs))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3.0,
            b=3.0
        )
    
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        # print(token_ids.shape)
        return self.weight[token_ids]