import torch
import torch.nn as nn
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features:int,out_features:int,device:torch.device|None=None,dtype:torch.dtype|None=None):
        """Construct a linear transformation module.

        Args:
            in_features (int):final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None, optional):  Device to store the parameters on
            dtype (torch.dtype | None, optional):  Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialization based on the formula:
        Normal distribution N(0, sigma^2) truncated at [-3*sigma, 3*sigma]
        where sigma = sqrt( 2 / (d_in + d_out) )
        """
        d_in = self.in_features
        d_out = self.out_features
        

        std = math.sqrt(2.0 / (d_in + d_out))
        

        limit = 3.0 * std
        

        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=std, 
            a=-limit, 
            b=limit
        )
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input.

        Args:
            x (torch.Tensor): input ...,in_dim

        Returns:
            torch.Tensor: output ...,out_dim
        """
        # print(self.weight.shape,x.shape)
        return x@self.weight.T