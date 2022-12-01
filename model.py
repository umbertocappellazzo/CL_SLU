#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:27:31 2022

@author: umbertocappellazzo
"""

from torch import nn
import torch


"""
PyTorch implementation of the TCN backbone, which is the feature extractor of our CL model (ENC_θ in our paper notation). 

"""

class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


EPS = 1e-8

class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())
        




class Conv1DBlock(nn.Module):

    """One dimensional convolutional block, as proposed in [1].
Args:
    :param in_chan (int): Number of input channels.
    :param hid_chan (int): Number of hidden channels in the depth-wise
        convolution.
    :param kernel_size (int): Size of the depth-wise convolutional kernel.
    :param padding (int): Padding of the depth-wise convolution.
    :param dilation (int): Dilation of the depth-wise convolution.
    
References:
    [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
    for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
    https://arxiv.org/abs/1809.07454
"""
    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, ):
        super(Conv1DBlock, self).__init__()
        conv_norm = GlobLN
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(), conv_norm(hid_chan), depth_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out
    

class TCN(nn.Module):
    
    """
    PyTorch implementation of the TCN backbone, which is the feature extractor of our CL model (ENC_θ in our paper notation). 
    """
    
    
    
    # The default parameters of the constructor are the ones used in our experiment.
    def __init__(self, in_chan=40, n_blocks=5, n_repeats=2, out_chan=64, hid_chan=128,
                 kernel_size=3, ):
        super(TCN, self).__init__()
        
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.out_chan = out_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        
        

        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, out_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):  
            for x in range(n_blocks):  
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(out_chan, hid_chan, kernel_size, padding=padding, dilation=2 ** x))

    
    def forward(self, mixture_w):
        
        output = self.bottleneck(mixture_w)
        
        for i in range(len(self.TCN)):
            
           residual = self.TCN[i](output)
           output = output + residual
        
        output = output.mean(-1)   # MAX pooling
        
        return output
        

"""
PyTorch implementation of the incremental classifier (FC_φ in our paper notation).
"""
    
class ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)

    def reset_parameters(self):
        self.head.reset_parameters()

    def forward(self, x):
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n


"""
Our whole CL model that comprises the feature extractor (TCN) and the continual classifier.

"""

class CL_model(nn.Module):
    """
    :param nb_classes: The initial # of classes (task 0).
    For the other parameters, see the TCN and Conv1DBlock classes.
    """
    # The default parameters of the constructor are the ones used in our experiment.
    def __init__(self, nb_classes, in_chan=40, n_blocks=5, n_repeats=2, out_chan=64, hid_chan=128,
                 kernel_size=3, device="cpu", ):
        super().__init__()
        
        self.nb_classes = nb_classes
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.out_chan = out_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        
        
        
        
        self.encoder = TCN(in_chan=self.in_chan, n_blocks=self.n_blocks, n_repeats=self.n_repeats, 
                           out_chan=self.out_chan, hid_chan=self.hid_chan,
                           kernel_size=self.kernel_size).to(device)
        
        self.classif = ContinualClassifier(self.out_chan, self.nb_classes).to(device)
        
        
    def forward_features(self, x):
        """ 
        This method computes the task feature embeddings (through the TCN enc) that will be used by 
        the task classifier (ENC_θ). This method will be called when we only want to compute the
        features for the MSE loss.
        """
        
        x_feat = self.encoder(x)
        
        return x_feat
    
    def forward_classifier(self, x):
        """
        Linear classifier operation (FC_φ).
        """
        
        return self.classif(x)
        
        
    def forward(self, x):
        enc_out = self.forward_features(x)
        return self.forward_classifier(enc_out)    