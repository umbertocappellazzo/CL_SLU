#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:19:15 2022

@author: umbertocappellazzo
"""
from continuum.datasets import FluentSpeech
import torch
from torch.utils import data
import soundfile as sf
import numpy as np
from scipy import signal
import librosa
import os
from continuum import ClassIncremental
from avalanche.benchmarks.generators import nc_benchmark
from tqdm import tqdm
from torch.utils.data import Dataset
from torchaudio import transforms
import torchaudio
import os
import json
import torch
from avalanche.benchmarks.generators import nc_benchmark
from models import TCN
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.logging import InteractiveLogger
#from avalanche.training.plugins import EvaluationPlugin
#from avalanche.evaluation.metrics import accuracy_metrics, \
#    loss_metrics, timing_metrics, forgetting_metrics

import torch.nn as nn
from torch.utils import data
#from avalanche.logging import TextLogger
from tqdm import tqdm
import pickle   
from transformers import Wav2Vec2Model,HubertModel
#from avalanche.training.plugins import StrategyPlugin
#from avalanche.training.strategies import BaseStrategy
from torch.optim.lr_scheduler import StepLR 
from continuum.metrics import Logger
from GSC_CL_scenario import GSC_CL




class fsc_data(data.Dataset):
    def __init__(self, path_to_dataset,max_len=64000,mode='train'):
        self.max_len = max_len
        self.audioid = []
        self.transcriptions = []
        self.intents = []
        self.subintent = [[] for i in range(3)]
        #self.win_len = win_len
        #self.eps = np.finfo(np.float64).eps
        self.path_to_dataset = path_to_dataset
        self.path_to_csvfile = path_to_dataset + 'data/' + mode + '_data.csv'
         

        with open(self.path_to_csvfile, encoding="utf-8") as fcsv:
            lines = fcsv.readlines()
            for l in lines[1:]:
                items = l[:-1].split(',')
                self.audioid.append(items[1])
                if (len(items)) == 7:
                    self.transcriptions.append(items[3])
                else:
                    self.transcriptions.append((" ").join(items[3:5]))
                self.intents.append(tuple(items[-3:]))
                for i in range(3):
                    self.subintent[i].append(self.intents[-1][i])

            utteranceset = sorted(list(set(self.transcriptions)))
            self.sentence_labels = [utteranceset.index(t) for t in self.transcriptions]
            intentset = sorted(list(set(self.intents)))
            self.targets = [intentset.index(t) for t in self.intents]
            subintent_sets = [sorted(list(set(self.subintent[i]))) for i in range(3)]
            self.subintent_labels = []
            for i in range(3):
                self.subintent_labels.append([subintent_sets[i].index(t) for t in self.subintent[i]])
            self.concat_labels = [str([self.subintent_labels[i][j] for i in range(3)]) for j in range(len(self.subintent_labels[0]))]
            self.unique_labels = sorted(list((set([self.concat_labels[i] for i in range(len(self.concat_labels))]))))
            self.y = [self.unique_labels.index(l) for l in self.concat_labels]
            
    def __len__(self):
        return len(self.audioid)

    def __getitem__(self, index):
        audiofile = self.audioid[index]
        
        f, sr = sf.read(self.path_to_dataset + audiofile)
        
        if len(f) > self.max_len:
            f = f[len(f)//2-self.max_len//2:len(f)//2+self.max_len//2]

        #n_fft = int(self.win_len * sr)
        if len(f) < self.max_len:
            ff = np.pad(f, [(0, self.max_len - f.shape[0]), ], mode='constant')
            f = ff

        label = (self.targets[index],[self.subintent_labels[i][index] for i in range(3)])

        #extracting Mel filters
        #filters = librosa.filters.mel(sr, n_fft, n_mels=40)
        #window = signal.hamming(n_fft, sym=False)
        #spectrogram = np.abs(
        #    librosa.stft(y=f + self.eps, n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 2, center=True, window=window))
        #melspectrum = np.log(np.dot(filters, spectrogram) + self.eps)
        #return torch.from_numpy(melspectrum), label
        return torch.FloatTensor(f), label

    def getsets(self):
        return sorted(list(set(self.intent))), [sorted(list(set(self.subintent[i]))) for i in range(3)]
    
    
    
    
    
    
if __name__ == "__main__":
    #path_to_csv = "/Users/umbertocappellazzo/Desktop/PHD/fluent_speech_commands_dataset/data/train_data.csv"
    path_to_dataset = '/Users/umbertocappellazzo/Desktop/PHD/fluent_speech_commands_dataset/'
    fsc_train = fsc_data(path_to_dataset)
    # fsc_test = fsc_data(path_to_dataset,mode='test')
    f,idd = fsc_train[0]
    #logger = Logger(list_subsets=['train', 'test'])
    
    # leng = len(fsc)
    # maxx = 0
    # minn = 1e10
    # max_id = ""
    # vec = []
    # id_vec = []
    # for x in range(leng):
    #     f,idd = fsc[x]
    #     vec.append(len(f))
    #     id_vec.append(idd)
    #     if len(f) > maxx:
    #         maxx = len(f)
    #         max_id = idd
    #     if len(f) < minn:
    #         minn = len(f)
    #scenario = ClassIncremental(fsc_train,initial_increment=5,increment=2)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = TCN(in_chan=49,out_chan=(31,)).to(device)
    #model_wav = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    # criterion = CrossEntropyLoss()
    # learning_rate = 0.001
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # #params = {'batch_size': 256,'shuffle': True}
    # #params = {'batch_size': 128,'shuffle': True}
    # #train_set_generator = data.DataLoader(fsc_train, **params)
    # #test_set_generator = data.DataLoader(fsc_test, batch_size=16,shuffle=True)
    
    
    
    
    #classes =  ["backward", "bed", "bird", "cat", "dog", "down", "eight", 
                    # "five", "follow", "forward", "four", "go", "happy", "house", 
                    # "left", "marvin", "nine", "no", "off", "on", "one", 
                    # "right", "seven", "sheila", "six", "stop", "three", 
                    # "tree", "two", "up", "wow", "yes", "zero"]
    
    #classes =  ["backward", "bed", "bird", "cat", "dog", "down", "eight", 
                 #   "five", "follow"]
    
    # kwargs = {"path_to_data": "speech_commands_v0.02","test_list_path": 
    #         os.path.join(os.getcwd(),"test_list.json"), "classes": classes}
    # n_exp = 11
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # #device = torch.device("cpu")
    # print(device)
    
    # train_dataset = GSC_CL(**kwargs,mode="train")
    # #print(len(train_dataset))
    # test_dataset = GSC_CL(**kwargs,mode="test")
    # scenario = nc_benchmark(train_dataset,test_dataset,n_exp,task_labels=False,seed=1234,
    #                       fixed_class_order=list(range(len(classes))))
    # model = TCN(in_chan=train_dataset.n_mels,out_chan=(len(classes),)).to(device)
    # #emb_mod = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    # #model = Fuse(emb_mod,model)
    # criterion = CrossEntropyLoss()
    # learning_rate = 0.001
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # # scenario = nc_benchmark(fsc_train,fsc_test,10,task_labels=True,seed=1234,per_exp_classes={0:4})
    # for i, exp in enumerate(scenario.train_stream):
    #     print(f"Processing exp #: {i}" )
    #     datas = exp.dataset
    #     datass = data.DataLoader(datas, batch_size=16,shuffle=True)
    #     for l,x in enumerate(datass):
    #         print(f"Processing batch #: {l}")
    #         f,d = x[0],x[1]
    #         print(d.shape)
    #         y = model(model_wav(f)[0])
    #         print(y.shape)
    #         loss = criterion(y,d)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         logger.add([(y.argmax(1)).detach().numpy(),d.detach().numpy(),x[2].detach().numpy()],subset='train')
    #     logger.end_task()
        
    

    