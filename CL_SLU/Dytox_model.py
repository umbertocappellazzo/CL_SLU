#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:31:02 2022

@author: umbertocappellazzo
"""

from torch import nn
import torch
from transformers import Wav2Vec2Model,WavLMModel,Wav2Vec2Processor
from utils import trunc_normal_, freeze_parameters, DropPath
import copy
from torch.utils import data
import numpy as np
import soundfile as sf
import librosa
import os
from scipy import signal
import torch.optim as optim


class fsc_data(data.Dataset):
    def __init__(self, csvfilename, max_len=64000, win_len=0.02, signaltype='wavscut'):
        self.max_len = max_len
        self.audioid = []
        self.transcriptions = []
        self.intent = []
        self.subintent = [[] for i in range(3)]
        self.win_len = win_len
        self.eps = np.finfo(np.float64).eps
        self.signaltype = signaltype

        with open(csvfilename, encoding="utf-8") as fcsv:
            lines = fcsv.readlines()
            for l in lines[1:]:
                items = l[:-1].split(',')
                self.audioid.append(items[1])
                if (len(items)) == 7:
                    self.transcriptions.append(items[3])
                else:
                    self.transcriptions.append((" ").join(items[3:5]))
                self.intent.append(tuple(items[-3:]))
                for i in range(3):
                    self.subintent[i].append(self.intent[-1][i])

            utteranceset = sorted(list(set(self.transcriptions)))
            self.sentence_labels = [utteranceset.index(t) for t in self.transcriptions]
            intentset = sorted(list(set(self.intent)))
            self.intent_labels = [intentset.index(t) for t in self.intent]
            subintent_sets = [sorted(list(set(self.subintent[i]))) for i in range(3)]
            self.subintent_labels = []
            for i in range(3):
                self.subintent_labels.append([subintent_sets[i].index(t) for t in self.subintent[i]])

    def __len__(self):
        return len(self.audioid)

    def __getitem__(self, index):
        audiofile = self.audioid[index]
        #print(audiofile)
        #audioin = audiofile.replace('wavs',self.signaltype); print(audioin)
        f, sr = sf.read("/data/cappellazzo/CL_SLU/fluent_speech_commands_dataset/" + audiofile)

        if len(f) > self.max_len:
            f = f[len(f)//2-self.max_len//2:len(f)//2+self.max_len//2]

        n_fft = int(self.win_len * sr)
        if len(f) < self.max_len:
            ff = np.pad(f, [(0, self.max_len - f.shape[0]), ], mode='constant')
            f = ff

        label = (self.intent_labels[index], [self.subintent_labels[i][index] for i in range(3)])

        # extracting Mel filters
        filters = librosa.filters.mel(sr, n_fft, n_mels=40)
        window = signal.hamming(n_fft, sym=False)
        spectrogram = np.abs(
            librosa.stft(y=f + self.eps, n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 2, center=True, window=window))
        melspectrum = np.log(np.dot(filters, spectrogram) + self.eps)
        return torch.from_numpy(melspectrum), label

    def getsets(self):
        return sorted(list(set(self.intent))), [sorted(list(set(self.subintent[i]))) for i in range(3)]



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
    # n blocks --> receptive field increases , n_repeats increases capacity mostly
    def __init__(self, in_chan=40, n_src=1, out_chan=(6, 14, 4), n_blocks=5, n_repeats=2, bn_chan=64, hid_chan=128,
                 kernel_size=3, ):
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
        
        #self.pretrain_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
        #self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        #self.pretrain_model.freeze_feature_encoder()

        layer_norm = GlobLN(in_chan)
        #layer_norm = nn.LayerNorm(401)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):  # ripetizioni 2
            for x in range(n_blocks):  # 5 layers convoluzionali
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2 ** x))

        #self.out = nn.ModuleList()
        #for o in out_chan:
       #     ##Gestisce multitask or intent classification
        #    out_conv = nn.Linear(bn_chan, n_src * o)
        #    self.out.append(nn.Sequential(nn.PReLU(), out_conv))
        #self.classif = ContinualClassifier(bn_chan ,out_chan[0])
        #self.act = nn.PReLU()

    # Get activation function.
    def forward(self, mixture_w):
        
        #proc = self.processor(mixture_w.cuda(),sampling_rate=16000,return_tensors="pt").input_values
        #y = self.pretrain_model(proc.squeeze(0).cuda())[0]
        #output = self.bottleneck(y)
        output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            
           residual = self.TCN[i](output)
           output = output + residual

        ###provare max pool2D su ouput seguito de reshape .view(-1,1)
        #logits = [out(output.mean(-1)) for out in self.out]
        #logits = output.mean(-1)   # LOGITS for TCNwithDytox.
        #logits = self.classif(logits)
        # logits = self.out(output)
        #return tuple(logits)
        #return logits[0]
        return output
        #return logits   # Used for standard TCN Brutti.
        
    def update(self,new_classes):
        new_head = nn.Linear(self.bn_chan,self.out[0][1].out_features+new_classes)
        new_head.weight.data[:-new_classes] = self.out[0][1].weight.data
        
        new_head.cuda()
        
        self.out[0][1] = new_head
        






class ClassAttention(nn.Module):
    """taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
     with slight modifications to do CA
     """
    
    def __init__(self, dim, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0., fc=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q = fc(dim, dim, bias= qkv_bias)
        self.k = fc(dim, dim, bias= qkv_bias)
        self.v = fc(dim, dim, bias= qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        #self.apply(self._init_weights)
        
    def reset_parameters(self):
        self.apply(self.init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask_heads=None, **kwargs):
        # B = batch_size, N = # of features, C = size of each feature, or embed_dim, or hidden_size. These 
        # letters are a bit misleading imo, I'd change then, but I wanna be consistent with the original nomenclature, 
        #thus I leave them unaltered.
        
        B, N, C = x.shape  
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if mask_heads is not None:
            mask_heads = mask_heads.expand(B, self.num_heads, -1, N)
            attn = attn * mask_heads
        
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls, attn, v
    


class MHSA(nn.Module):
    """
    Multi-head Self Attention (MHSA) implementation.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N**.5)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        distances = indd**.5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, v


class Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type=MHSA,
                 fc=nn.Linear, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, fc=fc, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, fc=fc)

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.reset_parameters()
        self.mlp.apply(self.mlp._init_weights)

    def forward(self, x, task_index=1):
        if isinstance(self.attn, ClassAttention):  # Like in CaiT
            
            
            task_token = x[:,:task_index]
            #print(f"task token shape in the TA block: {task_token.shape}")
            xx = self.norm1(x)
            
            xx, attn, v = self.attn(xx)
            #print(f"x vector after the class attention: {xx.shape}")
            
            task_token = self.drop_path(xx[:,:task_index]) + task_token
            task_token = self.drop_path(self.mlp(self.norm2(task_token))) + task_token
            
            return task_token, attn, v

        xx = self.norm1(x)
        xx, attn, v = self.attn(xx)

        x = self.drop_path(xx) + x
        x = self.drop_path(self.mlp(self.norm2(x))) + x

        return x, attn, v




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., fc=nn.Linear):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = fc(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = fc(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


    
    
class ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        #self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        #x = self.norm(x)
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n




class DyTox_slu(nn.Module):
    """ DyTox class definition.
    :param nb_classes: The initial # of classes (task 0).
    :param pretrain_model_name: The pretrained model used for extracting the features from the raw
    speech signals. It representes the "ENCODER". If wav2vec 2.0 is used, the encoder consists of 
    a CNN-based latent feature encoder + a Transformer encoder. Note that the CNN encoder is kep
    frozen, while the Transformer is finetuned.
    :param individual_classifier: Classifier config, DyTox is in `1-1` by default.
    :param num_heads: # of heads for the attention operation. Bear in mind that embed_dim % num_heads must be equal to 0.
    :param act_layer: Activation function to be used in the MLP block. Default: "GELU".
    :param norm_layer: Normalization type. Default: Layer Normalization.
    :param fc: Fully connected layer definition that will be used by the MLP block. 
    :param drop_path: Whether to apply Drop Path, a.k.a. Stochastic Depth. Default: 0., i.e., not applied.
    :param nb_SA: The number of transformer blocks (MHSA) for the encoder. Default: 3.
    :param nb_TA: The number of transformer blocks (Class_Attention) for the decoder. Default: 1.
    """
    
    def __init__(self, nb_classes, individual_classifier = '1-1', num_heads = 8, embed_dim = 400, nb_SA=0, nb_TA =1,drop =0.,
                   drop_path=0., attn_drop=0., mlp_ratio = 4,pretrain_model_name=''):
        super().__init__()
        
        self.nb_classes = nb_classes
        self.nb_classes_per_task = [nb_classes]
        self.individual_classifier = individual_classifier
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.pretrain_model_name = pretrain_model_name
        
        
        # Add other pretrained models.
        #if self.pretrain_model_name == '':
        #    self.pretrain_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        #    self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            #self.pretrain_model.freeze_feature_encoder()
        #    for p in self.pretrain_model.parameters():
        #        p.requires_grad = False
            
        #    self.pretrain_model.cuda()
                
        #self.embed_dim = self.pretrain_model.config.hidden_size
        self.embed_dim = embed_dim
        self.encoder = TCN(in_chan=40,out_chan = (4,)).cuda()
        self.task_tokens = nn.ParameterList([trunc_normal_(nn.Parameter(torch.zeros(1, 1, self.embed_dim).cuda()),std=.02)])
        
        
        
        
        #1-1 config: one indipendente classifer for each task that receives the embedding from that task token 
        #(default config. in DyTox).
        in_dim, out_dim = self._get_ind_clf_dim()
        
        self.head = nn.ModuleList([ContinualClassifier(in_dim, out_dim).cuda()])
        
        blocks = []
        #for layer_index in range(nb_SA):
        #    block = Block(dim=self.embed_dim, num_heads=self.num_heads,  mlp_ratio=self.mlp_ratio, qkv_bias=False, qk_scale=None, drop=drop, attn_drop=attn_drop,
        #                 drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type=MHSA,
        #                 fc=nn.Linear)
        #    blocks.append(block)
        
        for layer_index in range(nb_TA):
            block = Block(dim=self.embed_dim, num_heads=self.num_heads,  mlp_ratio=self.mlp_ratio, qkv_bias=False, qk_scale=None, drop=drop, attn_drop=attn_drop,
                         drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type=ClassAttention,
                         fc=nn.Linear)
            blocks.append(block)
        
        blocks = nn.ModuleList(blocks)
        
        #self.sabs = blocks[:nb_SA]
        #self.tabs = blocks[-nb_TA:]
        self.tabs = blocks         # In case I use transformers as encoder, use the 2 lines of code above.
            
        #Definition of the Task-Attention BLOCK: LN1, TA, LN2, MLP.
        #self.norm1 = norm_layer(self.embed_dim)
        #self.class_attention = ClassAttention(self.embed_dim, num_heads=self.num_heads)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm2 = norm_layer(self.embed_dim)
        #mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        
        #MLP layer inp_dim = out_dim.
        #self.mlp = Mlp(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, fc=fc)
        
        #self.classif = ContinualClassifier(768, 4)
        
        
        
    def update_model(self, nb_new_classes):
        """ This method expands the DyTox model to support the new task. In fact,
        the updated model has to grapple with the new experience. It does so by adding a new 
        task token and a new classifier specific for the new experience.
        :param nb_new_classes: Number of new classes brought by the new task.
        """
            
        self.nb_classes_per_task.append(nb_new_classes)
            
        # Task token update: ----------------------------------------------
        
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        
        self.task_tokens.append(new_task_token)
        # -----------------------------------------------------------------
        
        # Diversity head update: ------------------------------------------
        
        # TBD if divergence loss is to be used.
        # -----------------------------------------------------------------
        
        
        # Classifier update: ----------------------------------------------
        in_dim, out_dim = self._get_ind_clf_dim()
        
        self.head.append(ContinualClassifier(in_dim, out_dim).cuda())
            
    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        
        requires_grad = False
        freeze_parameters(self, requires_grad=not requires_grad)
        self.train()
        
        for name in names:
            if name == 'old_task_tokens':
                freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                freeze_parameters(self.head[:-1], requires_grad=requires_grad)
                
    def _get_ind_clf_dim(self):
        """ Compute the input and output dimensions of the classifier depending on its config.
        By default, 1-1 mode is employed.
        """
        
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            in_dim = len(self.task_tokens)*self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            in_dim = len(self.task_tokens)*self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown individual classifier {self.individual_classifier}.')
        
        return in_dim, out_dim
            
        
    
    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters(),
            'new_head': self.head[-1].parameters(),
            'head': self.head.parameters(),
            'TCN': self.encoder.parameters(),
            'TA_block': self.class_attention.parameters()
            }
    
    
    def task_attention_block(self,x):
        """ This function computes the task attention transformer: x --> LN1 
        --> CA --> RES --> LN2 --> MLP --> RES.
        """
        
        task_token = x[:,:1]
        #print(f"task token shape in the TA block: {task_token.shape}")
        xx = self.norm1(x)
        
        xx, attn, v = self.class_attention(xx)
        #print(f"x vector after the class attention: {xx.shape}")
        
        task_token = self.drop_path(xx[:,:1]) + task_token
        task_token = self.drop_path(self.mlp(self.norm2(task_token))) + task_token
        
        return task_token, attn, v
            
            
        
    def forward_features(self, x):
        """ This method compute the task embeddings that will be used by 
        the task classifiers.
        """
        B = x.shape[0]    # Batch size
        
        # If I use wav2vec 2.0 encoder.
        
        #x_proc = self.processor(x,sampling_rate=16000,return_tensors="pt").input_values
        #x_proc = self.pretrain_model(x_proc.squeeze(0).cuda())[0]
        #print(x_proc.shape)
        
        
        x_proc = self.encoder(x)[:,:,:self.embed_dim]   #X_proc would have embed_dim+1 as dimension, but we need embed_dim%num_heads = 0, so I drop the last feature.
        #x_proc = x[:,:,:self.embed_dim]
        
        #s_e, s_a, s_v = [], [], []
        
        #for blk in self.sabs:
        #    x_proc, attn, v = blk(x_proc)
        #    s_e.append(x_proc)
        #    s_a.append(attn)
        #    s_v.append(v)
            
        
        
        #print(f"input to task attention: {x_proc.shape}")
        
        tokens_emb = []
        attentions = []
        
        for task_token in self.task_tokens:
            task_token = task_token.expand(B,-1,-1)
            #print(f"task token shape: {task_token.shape}")
            
            for blk in self.tabs:
                
                task_token, attn, v = blk(torch.cat((task_token, x_proc), dim=1))
            #print(f"task token shape after att block: {task_token.shape}")
            
            attentions.append(attn)
            
            tokens_emb.append(task_token[:, 0])
        
        return tokens_emb,tokens_emb[-1], attentions               
                
        
        
        
    def forward_classifier(self, tokens_emb, last_tok_emb):
        """ Once all task embeddings e_1, ..., e_t are extracted, classify.
        Classifier has different modes based on a pattern x-y:
        - x means the number of task embeddings in input
        - y means the number of task to predict
        So:
        - n-n: predicts all task given all embeddings
        But:
        - 1-1: predict 1 task given 1 embedding, which is the 'independent classifier' used in the paper.
        By default this work exploits 1-1 setting. 
        """
        logits = []
        
        for i, head in enumerate(self.head):
            if self.individual_classifier in ('1-n', '1-1'):
                logits.append(head(tokens_emb[i]))
            else: # n-1, n-n
                logits.append(head(torch.cat(tokens_emb[:i+1], dim=1)))
        
        if self.individual_classifier in ('1-1', 'n-1'):
            logits = torch.cat(logits, dim=1)
        else: # 1-n, n-n
            final_logits = torch.zeros_like(logits[-1])
            for i in range(len(logits)):
                final_logits[:, :logits[i].shape[1]] += logits[i]
            
            for i, c in enumerate(self.nb_classes_per_task):
                final_logits[:, :c] /= len(self.nb_classes_per_task) - i
            logits = final_logits
        
        #print(f"logits shape: {logits.shape}")
        return {'logits': logits, 'tokens': tokens_emb}
                
        
        
    def forward(self, x):
        tokens_emb, last_tok_emb, _ = self.forward_features(x)
        return self.forward_classifier(tokens_emb, last_tok_emb)
    
    # def forward(self, x): 
        
    #     x_proc = self.processor(x,sampling_rate=16000,return_tensors="pt").input_values
        
    #     x_proc = self.pretrain_model(x_proc.squeeze(0).cuda())[0]
        
    #     return self.classif(x_proc[:,0])
        
#if __name__ == "__main__":
    
    #inp = torch.rand(16, 40, 600)
    #m = TCN(in_chan=40,out_chan = (33,))
    #o = m(inp)
    
    # mode = "test"
    
    # if mode == "test":
    #     #/Users/umbertocappellazzo/Desktop/PHD
    #     #/data/cappellazzo/CL_SLU/fluent_speech_commands_dataset/data/train_data.csv
    #     train_data = fsc_data("/data/cappellazzo/CL_SLU/fluent_speech_commands_dataset/data/train_data.csv", max_len = 64000)
    #     params = {'batch_size': 100,'shuffle': True,'num_workers':10}
    #     train_set_generator = data.DataLoader(train_data, **params)
        
    #     valid_data = fsc_data("/data/cappellazzo/CL_SLU/fluent_speech_commands_dataset/data/test_data.csv", max_len = 64000)
    #     params = {'batch_size':100,'shuffle': False,'num_workers':10}
    #     valid_set_generator = data.DataLoader(valid_data, **params)
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     modelname = "best_model_inverted.pkl"
        
    #     model = TCN(in_chan = 40, n_blocks=5, n_repeats=2, out_chan=(31, )).to(device)
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     criterion = torch.nn.CrossEntropyLoss()
    #     best_accuracy = 0
        
    #     intentSet, subIntentSet = train_data.getsets()
    #     actionSet, objectSet, locationSet = subIntentSet
        
    #     for e in range(100):
    #         for i, d in enumerate(train_set_generator):
    #             model.train()
    #             f, l = d
    
    #             y = model(f.float().to(device))
                
                
    #             loss = criterion(y[0], l[0].to(device))
    #             print("Iteration %d in epoch%d--> loss = %f"%(i, e, loss.item()), end='\r')
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             if i % 100 == 0:
                    
            
    #                 model.eval()
    #                 correct = []
            
    #                 for j, eval_data in enumerate(valid_set_generator):
    #                     feat, label = eval_data
    #                     y_eval = model(feat.float().to(device))
    #                     pred = torch.argmax(y_eval[0].detach().cpu(), dim=1)
    #                     intent_pred = pred
    #                     correct.append((intent_pred == label[0]).float())
            
    #                 acc = np.mean(np.hstack(correct))
    #                 intent_acc = acc
    #                 iter_acc = '\n iteration %d epoch %d-->' % (i, e)
    #                 print(iter_acc, acc, best_accuracy)
    #                 if intent_acc > best_accuracy:
    #                     improved_accuracy = 'Current accuracy = %f (%f), updating best model' % (intent_acc, best_accuracy)
    #                     print(improved_accuracy)
    #                     best_accuracy = intent_acc
    #                     best_epoch = e
    #                     torch.save(model.state_dict(), modelname)
    #     print(f"The best epoch is: {best_epoch}")
    
    # else:
    #     test_data = fsc_data("/data/cappellazzo/CL_SLU/fluent_speech_commands_dataset/data/valid_data.csv", max_len = 64000)
    #     params = {'batch_size': 100,'shuffle': True,'num_workers':10}
    #     test_set_generator = data.DataLoader(test_data, **params)
        
    #     valid_data = fsc_data("/data/cappellazzo/CL_SLU/fluent_speech_commands_dataset/data/test_data.csv", max_len = 64000)
    #     params = {'batch_size':100,'shuffle': False,'num_workers':10}
    #     valid_set_generator = data.DataLoader(valid_data, **params)
        
        
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     print('Using device %s' % device)
        
    #     model = TCN(in_chan = 40, n_blocks=5, n_repeats=2, out_chan=(31, )).to(device)
    #     modelname = "best_model_inverted.pkl"
        
    #     model.load_state_dict(torch.load(modelname))
    #     model.eval()
    #     model.to(device)
        
    #     intentSet, subIntentSet = test_data.getsets()
    #     actionSet, objectSet, locationSet = subIntentSet
        
    #     correct_test = []
        
    #     for i, d in enumerate(test_set_generator):
    #         feat, label = d
    #         print('Iter %d (%d/%d)' % (i, i * 100, len(test_data)), end='\r')
    #         z_eval = model(feat.float().to(device))
    #         pred = [torch.max(z.detach().cpu(), dim=1)[1] for z in z_eval]
    #         pred_test = pred[0]
    #         correct_test.append(np.array(pred_test == label[0], dtype=float))
        
    #     acc_test = (np.mean(np.hstack(correct_test)))
    #     print("The accuracy on test set is %f" % (acc_test))
        
    #     correct_valid = []
        
    #     for i, d in enumerate(valid_set_generator):
            
    #         print('Iter %d (%d/%d)' % (i, i * 100, len(valid_data)), end='\r')
    #         feat, label = d
            
    #         a_eval = model(feat.float().to(device))
    #         pred = [torch.max(a.detach().cpu(), dim=1)[1] for a in a_eval]
            
    #         pred_test = pred[0]
            
    #         correct_valid.append(np.array(pred_test == label[0], dtype=float))
        
    #     acc_val = (np.mean(np.hstack(correct_valid)))
    #     print("The accuracy on the validation set is %f" % (acc_val))