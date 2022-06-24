#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:13:55 2022

@author: umbertocappellazzo
"""

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
#from avalanche.training.strategies import Naive, GEM, Cumulative, \
#    JointTraining, GDumb, LwF, EWC, AGEM, CWRStar, Replay, SynapticIntelligence
#import torch.nn as nn
#from torch.utils import data
#from avalanche.logging import TextLogger
#from tqdm import tqdm
import pickle   
from transformers import Wav2Vec2Model,WavLMModel,Wav2Vec2Processor
#from avalanche.training.plugins import StrategyPlugin
#from avalanche.training.strategies import BaseStrategy
import torch.nn as nn


CLASS_INDEX = { "backward" : 0, "bed" : 1, "bird" : 2, "cat" : 3, "dog" : 4, "down" : 5
               ,"eight" : 6, 
                "five" : 7, "follow" : 8, "forward" : 9, "four" : 10, "go" : 11, "happy": 12, "house" : 13, 
                "left": 14, "marvin" : 15, "nine" : 16, "no" : 17, "off" : 18, "on" : 19, "one" : 20, 
                "right" : 21, "seven" : 22, "sheila" : 23, "six" : 24, "stop" : 25, "three" : 26, 
                "tree" : 27, "two" : 28, "up" : 29, "wow" : 30, "yes" : 31, "zero" : 32}




# class LwF_emb(StrategyPlugin):
#     def __init__(self):
#         super().__init__()
#         #self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
#         #self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
#         self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
#         self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#         #self.model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
#         self.model.cuda()
#         #self.processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        
#         self.model.freeze_feature_encoder()
#     def before_forward(self, strategy:"BaseStrategy", **kwargs):
#         #print(strategy.mbatch[0].shape)
        
#         proc = self.processor(strategy.mbatch[0],sampling_rate=16000,return_tensors="pt").input_values
#         strategy.mbatch[0] = self.model(proc.squeeze(0).cuda())[0]
#         #print(strategy.mbatch[0].shape)
#         #strategy.mbatch[0] = self.model(strategy.mbatch[0])[0]
#     def before_eval_forward(self, strategy:"BaseStrategy", **kwargs):
#         #strategy.mbatch[0] = self.model(strategy.mbatch[0])[0]
#         #print(strategy.mbatch[0].shape)
#         #strategy.mbatch[0] = self.model(self.processor(strategy.mbatch[0],sampling_rate=16000,return_tensors="pt").input_values)[0]
#         #print(strategy.mbatch[0].shape)
#         proc = self.processor(strategy.mbatch[0],sampling_rate=16000,return_tensors="pt").input_values
#         strategy.mbatch[0] = self.model(proc.squeeze(0).cuda())[0]


# class Fuse(nn.Module):
#     def __init__(self, embedding_model, model):
#         super().__init__()
#         self.embedding = embedding_model.cuda()
#         self.model = model

#     def forward(self, x):
#         emb = self.embedding(x)[0]
#         out = self.model(emb)
#         return out




class GSC_CL(Dataset):
    def __init__(self, path_to_data, test_list_path, classes, win_len = 25, hop_len = 5, mode="train",max_len=16000):
        super().__init__()
        self.path = path_to_data
        self.test_list_path = test_list_path
        self.classes = classes
        self.mode = mode
        self.n_mels = 49
        self.sample_rate = 16000
        self.max_len = max_len
        
        self.win_len = int(self.sample_rate/1000*win_len)
        self.hop_len = int(self.sample_rate/1000*hop_len)
        
        self.mel_spectr = transforms.MelSpectrogram(sample_rate=self.sample_rate,
                win_length=self.win_len, hop_length=self.hop_len, n_mels=self.n_mels)
        
        
        
        self.class_index = CLASS_INDEX
        
        # Creation of lists containing audio ids and respective labels. 
        self.audio_id = []
        self.targets = []
        #self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        #self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-100h")
        #self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        with open(self.test_list_path, "r") as f:
            test_list = json.load(f)
        for class_ in self.classes:
            files = [file for file in os.listdir(os.path.join(self.path,class_))]
            
            for x in files:
                if self.mode == "train" and x in test_list[class_]:
                    continue
                elif self.mode == "test" and x not in test_list[class_]:
                    continue
                self.audio_id.append(x)
                self.targets.append(torch.ones(1).long()*self.class_index[class_])
                
        
        
    def __getitem__(self,index):
        audio, sample_rate = torchaudio.load(os.path.join(self.path,list(self.class_index.keys())[list(self.class_index.values()).index(int(self.targets[index]))],self.audio_id[index]))
        assert sample_rate == self.sample_rate
        
        # Pad audios that are shorter than 16000 samples (1sec).
        if audio.size(1) < self.max_len:
            padded_audio = torch.zeros(1,self.max_len)
            padded_audio[:,:audio.size(1)] = audio
            audio = padded_audio
            
        
        #features = self.model(audio)
        #features = self.mel_spectr(audio)
        
        #return features[0].detach().squeeze(0), self.targets[index]
        #return features.squeeze(0), self.targets[index]
        return audio.squeeze(0), self.targets[index]
    
    def __len__(self):
        return len(self.audio_id)



if __name__ == "__main__":
    
    classes =  ["backward", "bed", "bird", "cat", "dog", "down", "eight", 
                    "five", "follow", "forward", "four", "go", "happy", "house", 
                    "left", "marvin", "nine", "no", "off", "on", "one", 
                    "right", "seven", "sheila", "six", "stop", "three", 
                    "tree", "two", "up", "wow", "yes", "zero"]
    
    #classes =  ["backward", "bed", "bird", "cat", "dog", "down", "eight", 
                 #   "five", "follow"]
    
    kwargs = {"path_to_data": os.path.join(os.getcwd(),"speech_commands_v0.02"),"test_list_path": 
            os.path.join(os.getcwd(),"test_list.json"), "classes": classes}
    n_exp = 11
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    
    train_dataset = GSC_CL(**kwargs,mode="train")
    #print(len(train_dataset))
    test_dataset = GSC_CL(**kwargs,mode="test")
    
    
    
    results = []
    for fold in range(3):
        print("Cross-validation on fold #", fold)
        
        scenario = nc_benchmark(train_dataset,test_dataset,n_exp,task_labels=False,seed=1234,
                              fixed_class_order=list(range(len(classes))))
        #,per_exp_classes= {0:30}
        #model = MLP(40*11,128,33)
        model = TCN(in_chan=train_dataset.n_mels,out_chan=(len(classes),)).to(device)
        #emb_mod = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        #model = Fuse(emb_mod,model)
        criterion = CrossEntropyLoss()
        learning_rate = 0.001
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        
        
        loggers = []
        loggers.append(InteractiveLogger())
        #f = open(os.path.join(os.getcwd(), 'text_logger_GEM.txt'), 'w')
        #text_logger = TextLogger(f)
        #loggers.append(text_logger)
        
        eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True), 
                                          loss_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                          timing_metrics(epoch=True,experience=True,stream=True),
                                          forgetting_metrics(experience=True, stream=True),loggers=loggers)
        
        
        #strategy = Naive(model,optimizer,criterion,train_mb_size=64,train_epochs=6,eval_mb_size=32,device=device,plugins=[LwF_emb()],evaluator=eval_plugin, eval_every=1)
        #strategy = GEM(model,optimizer,criterion,patterns_per_exp=512,train_mb_size=128,train_epochs=3,eval_mb_size=32,device=device,evaluator=eval_plugin,eval_every=1)
        #strategy = Cumulative(model,optimizer,criterion,train_mb_size=256,train_epochs=10,eval_mb_size=32,device=device,evaluator=eval_plugin, eval_every=1)
        #strategy = JointTraining(model,optimizer,criterion,train_mb_size=256,train_epochs=15,eval_mb_size=64,device=device,plugins=[LwF_emb()],evaluator=eval_plugin, eval_every=10)
        #strategy = GDumb(model,optimizer,criterion,mem_size=5000,train_mb_size=256,train_epochs=3,eval_mb_size=32,device=device,evaluator=eval_plugin, eval_every=1,)
        #strategy = LwF(model,optimizer,criterion,alpha=1,temperature=1,train_mb_size=128,train_epochs=6,eval_mb_size=64,device=device,plugins=[LwF_emb()],evaluator=eval_plugin, eval_every=10)
        #strategy = EWC(model,optimizer,criterion,ewc_lambda=1,train_mb_size=256,train_epochs=3,eval_mb_size=32,device=device,evaluator=eval_plugin, eval_every=1)
        #strategy = AGEM(model,optimizer,criterion,patterns_per_exp=512,sample_size=512,train_mb_size=256,train_epochs=3,eval_mb_size=32,device=device,evaluator=eval_plugin,eval_every=1)
        #strategy = CWRStar(model,optimizer,criterion,cwr_layer_name=None,train_mb_size=256,train_epochs=3,eval_mb_size=32,device=device,evaluator=eval_plugin, eval_every=1)
        strategy = Replay(model,optimizer,criterion,mem_size=1000,train_mb_size=150,train_epochs=6,eval_mb_size=64,device=device,plugins=[LwF_emb()],evaluator=eval_plugin, eval_every=10)
        #strategy = SynapticIntelligence(model,optimizer,criterion,si_lambda=1,train_mb_size=256,train_epochs=3,eval_mb_size=32,device=device,evaluator=eval_plugin, eval_every=1)
        #plugins=[LwF_emb()]
        
        print('Starting experiment...')
        results_fold = []
        for i, exp in enumerate(scenario.train_stream):
            print("Start training on experience ", exp.current_experience)
        #strategy.train(scenario.train_stream,num_workers=12)    
        
            strategy.train(exp,num_workers=12)
            #strategy.train(exp, eval_streams=[scenario.test_stream[i]])
            print('Training completed')
            print('Computing accuracy on the whole test set')
            results_fold.append(strategy.eval(scenario.test_stream,num_workers=8))
            #results_fold.append(strategy.eval(scenario.test_stream))
            #results.append(strategy.eval(scenario.test_stream))
        
        
        results.append(results_fold)
        #f.close()
        
    with open('metrics_replay_wav_updated','wb') as handle:
        pickle.dump(results,handle)

        
            
                    
# IMPLEMENTATION FOR LOADING ALL DATA TOGETHER (issues if audios have quite different lengths, e.g. temporal dimension differs).        
            
# class GSC_CL():
#     def __init__(self, path_to_data, test_list_path, classes, win_len = 25, hop_len = 10, mode="train"):
#         super().__init__()
#         self.path = path_to_data
#         self.test_list_path = test_list_path
#         self.classes = classes
#         self.mode = mode
#         self.n_mels = 40
#         self.sample_rate = 16000
        
#         win_len = int(self.sample_rate/1000*win_len)
#         hop_len = int(self.sample_rate/1000*hop_len)
        
#         self.mel_spectr = transforms.MelSpectrogram(sample_rate=self.sample_rate,
#                 win_length=win_len, hop_length=hop_len, n_mels=self.n_mels)
        
#         self.classes = list(CLASS_INDEX.keys())
        
#         self.class_index = CLASS_INDEX
        
#         self.data = None
#         self.targets = None
#         self._load_data()
    
#     def _load_data(self):
        
#         with open(self.test_list_path, "r") as f:
#             test_list = json.load(f)
        
#         data = []
#         targets = []
        
#         for class_ in self.classes:
#             files = [file for file in os.listdir(os.path.join(self.path,class_))]
            
#             features = []
#             for x in files:
#                 if self.mode == "train" and x in test_list[class_]:
#                     continue
#                 elif self.mode == "test" and x not in test_list[class_]:
#                     continue
#                 audio, sample_rate = torchaudio.load(os.path.join(self.path,class_,x))
#                 assert sample_rate == self.sample_rate
#                 features.append(self.mel_spectr(audio[:,0:3000]))
                
#             data.append(torch.cat(features,dim=0))
#             targets.append(torch.ones(data[-1].size(0)).long() *self.class_index[class_])
#         self.data = torch.cat(data,dim=0)
#         self.targets = torch.cat(targets,dim=0)
        
#     def __getitem__(self,index):
#         return self.data[index], self.targets[index]
    
    
#     def __len__(self):
#         return self.data.size(0)
        

#if __name__ == "__main__":
    
    #classes =  ["backward", "bed", "bird", "cat", "dog", "down", "eight", 
    #                "five", "follow", "forward", "four", "go", "happy", "house", 
    #                "left", "marvin", "nine", "no", "off", "on", "one", 
    #                "right", "seven", "sheila", "six", "stop", "three", 
    #                "tree", "two", "up", "wow", "yes", "zero"]
    
    #kwargs = {"path_to_data": os.path.join(os.getcwd(),"speech_commands_v0.02"),"test_list_path": 
    #        os.path.join(os.getcwd(),"test_list.json"), "classes": classes}
    #n_exp = 11
    
    #train_dataset = GSC_CL(**kwargs,mode="train")
    #test_dataset = GSC_CL(**kwargs,mode="test")
    
    #scenario = nc_benchmark(train_dataset,test_dataset,n_exp,task_labels=False,seed=1234,
    #                        fixed_class_order=list(range(len(classes))))
        

    
