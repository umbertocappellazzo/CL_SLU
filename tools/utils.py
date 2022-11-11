#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:14:13 2022

@author: umbertocappellazzo
"""
from torch.nn import functional as F
from torchaudio import transforms as t
import numpy as np



def trunc(x, max_len):
    l = len(x)
    if l > max_len:
        x = x[l//2-max_len//2:l//2+max_len//2]
    if l < max_len:
        x = F.pad(x, (0, max_len-l), value=0.)
    
    eps = np.finfo(np.float64).eps
    sample_rate = 16000
    n_mels = 40
    win_len = 25
    hop_len= 10
    win_len = int(sample_rate/1000*win_len)
    hop_len = int(sample_rate/1000*hop_len)
    mel_spectr = t.MelSpectrogram(sample_rate=16000,
            win_length=win_len, hop_length=hop_len, n_mels=n_mels)
    
    return np.log(mel_spectr(x)+eps)  
    

def freeze_parameters(m, requires_grad=False):
    for p in m.parameters():
        p.requires_grad = requires_grad


def get_kdloss(predictions,predictions_old,current_loss,tau,is_both_kds=False):
    logits_for_distil = predictions[:, :predictions_old.shape[1]]
    alpha = np.log((predictions_old.shape[1] / predictions.shape[1])+1)
    
    _kd_loss = F.kl_div(
        F.log_softmax(logits_for_distil / tau, dim=1),
        F.log_softmax(predictions_old / tau, dim=1),
        reduction='mean',
        log_target=True) * (tau ** 2)
    
    if is_both_kds: return current_loss + alpha*_kd_loss
    else: return (1-alpha)*current_loss + alpha*_kd_loss
    
    
def get_kdloss_onlyrehe(predictions,predictions_old,current_loss,tau,alpha,is_both_kds=False):
    logits_for_distil = predictions[:, :predictions_old.shape[1]]
    
    _kd_loss = F.kl_div(
        F.log_softmax(logits_for_distil / tau, dim=1),
        F.log_softmax(predictions_old / tau, dim=1),
        reduction='mean',
        log_target=True) * (tau ** 2)
    if is_both_kds: return current_loss + alpha*_kd_loss
    else: return (1-alpha)*current_loss + alpha*_kd_loss