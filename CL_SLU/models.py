import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


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
    in_chan (int): Number of input channels.
    hid_chan (int): Number of hidden channels in the depth-wise
        convolution.
    skip_out_chan (int): Number of channels in the skip convolution.
        If 0 or None, `Conv1DBlock` won't have any skip connections.
        Corresponds to the the block in v1 or the paper. The `forward`
        return res instead of [res, skip] in this case.
    kernel_size (int): Size of the depth-wise convolutional kernel.
    padding (int): Padding of the depth-wise convolution.
    dilation (int): Dilation of the depth-wise convolution.
    norm_type (str, optional): Type of normalization to use. To choose from
        -  ``'gLN'``: global Layernorm
        -  ``'cLN'``: channelwise Layernorm
        -  ``'cgLN'``: cumulative global Layernorm
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
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation,groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),conv_norm(hid_chan), depth_conv1d,nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out


class TCN(nn.Module):
    # n blocks --> receptive field increases , n_repeats increases capacity mostly
    def __init__(self, in_chan=40, n_src=1, out_chan=(6, 14, 4), n_blocks=5, n_repeats=2, bn_chan=64, hid_chan=128,kernel_size=3, ):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size

        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):  # ripetizioni 2
            for x in range(n_blocks):  # 5 layers convoluzionali
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan,kernel_size, padding=padding,dilation=2 ** x))

        self.out = nn.ModuleList()
        for o in out_chan:
            ##Gestisce multitask or intent classification
            out_conv = nn.Linear(bn_chan, n_src * o)
            self.out.append(nn.Sequential(nn.PReLU(), out_conv))

    # Get activation function.
    def forward(self, mixture_w):
        output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual


        logits = [out(output.mean(-1)) for out in self.out]
        #return tuple(logits)
        return logits

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config


if __name__ == "__main__":
    inp = torch.rand(16, 40, 600)
    m = TCN(in_chan=40,out_chan = (33,))
    o = m(inp)
    #print(type(o))
    #print(o)
    #model = TCN()
    #print(model)
    #print(type(model))
    #num_list = [p.numel() for p in model.parameters() if p.requires_grad==True] 
    #print(sum(num_list))
    #num_list = [p.numel() for p in model.parameters()] 
    #print(sum(num_list))
    
    

