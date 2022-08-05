#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:50:14 2022

@author: umbertocappellazzo
"""

import copy
import os
from torch.utils.data import DataLoader
from functools import partial
from torch.optim import Adam, AdamW
from continuum import ClassIncremental
from continuum.datasets import FluentSpeech
import torch
from torch.nn import functional as F
import argparse
from continuum.metrics import Logger
import numpy as np
#from DyTox_TCNonly import DyTox_slu, TCN
from Dytox_model import DyTox_slu, TCN
#from DyTox_AST import DyTox_slu_AST
from tools.utils import trunc, get_kdloss,get_finetuning_dataset, SpecAugment,collate_batch
import time
import datetime
import json
import pytorch_warmup as warmup
import wandb
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils import data
from continuum import rehearsal
import matplotlib.pyplot as plt
import librosa
#import torch_optimizer as optim

def get_args_parser():
    parser = argparse.ArgumentParser('DyTox bare-version for SLU (FSC) train and evaluation', add_help=False)
    
    # Dataset parameters.
    parser.add_argument('--data_path', type=str, default='/data/cappellazzo/CL_SLU/',help='path to dataset')
    parser.add_argument('--max_len', type=int, default=64000, help='max length for the audio signal --> it will be cut')
    parser.add_argument('--download_dataset', default=False, help='whether to download the FSC dataset or not')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type= str, default='cuda', help='device to use for training/testing')
   
    
    
    
    # Training/inference parameters.
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--eval_every", type=int, default=1, help="Eval model every X epochs, if None only eval at the task end")
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--label_smoothing', type=float, default=0., help='Label smoothing for the CE loss')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    
    
    parser.add_argument('--output_basedir', default='./checkpoints/',
                        help='path where to save, empty for no saving')
    
    # Rehearsal memory
    parser.add_argument('--memory_size', default=930, type=int,
                        help='Total memory size in number of stored (image, label).')
    parser.add_argument('--fixed_memory', default=True, action='store_true',
                        help='Dont fully use memory when no all classes are seen as in Hou et al. 2019')
    parser.add_argument('--rehearsal', default="random",
                        choices=[
                            'random',
                            'closest_token', 'closest_all',
                            'icarl_token', 'icarl_all',
                            'furthest_token', 'furthest_all'
                        ],
                        help='Method to herd sample for rehearsal.')
    parser.add_argument('--sep_memory', default=False, action='store_true',
                        help='Dont merge memory w/ task dataset but keep it alongside')
    
    
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='',
                        help='Dont use that')
    
    # DISTILLATION
    
    parser.add_argument('--auto-kd', default=False, action='store_true',
                        help='Balance kd factor as WA https://arxiv.org/abs/1911.07053')
    parser.add_argument('--kd', default=0., type=float)
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')
    
    # DIVERSITY.
    parser.add_argument('--head-div-coeff', default=0., type=float,
                        help='Use a divergent head to predict among new classes + 1 using last token')
    
    # FINETUNING
    parser.add_argument('--finetuning', default='', choices=['balanced'],
                        help='Whether to do a finetuning after each incremental task. Backbone are frozen.')
    parser.add_argument('--finetuning-epochs', default=10, type=int,
                        help='Number of epochs to spend in finetuning.')
    
    
    #Optimizer
    # parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
    #                     help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--weight-decay', type=float, default=0.02,
    #                     help='weight decay (default: 0.05)')
    
    #Learning rate schedule parameters
    # parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "cosine"')
    # parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
    #                     help='learning rate (default: 5e-4)')
    # parser.add_argument("--incremental-lr", default=None, type=float,
    #                     help="LR to use for incremental task (t > 0)")
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    # parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6)')
    # parser.add_argument('--incremental-warmup-lr', type=float, default=None, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6) for task T > 0')
    # parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
    #                     help='epoch interval to decay LR')
    # parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    # parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')
    
    
    # Continual learning parameters.
    parser.add_argument('--increment', type=int, default=3, help='# of classes per task/experience')
    parser.add_argument('--initial_increment', type=int, default=4, help='# of classes for the 1st task/experience')
    parser.add_argument('--nb_tasks', type=int, default=10, help='the scenario number of tasks')
    parser.add_argument('--max_task', type=int, default=None, help='Max task id to train on')
    
    #DyTox model parameters.
    parser.add_argument('--ind_clf', default='1-1', choices=['1-1', '1-n', 'n-n', 'n-1'],
                        help='Independent classifier per task but predicting all seen classes')
    
    # What to freeze during the training of experience i.
    parser.add_argument('--freeze_task', type=str, default=['old_task_tokens','old_heads'], nargs="*", help='What to freeze before every incremental task (t > 0)')
    
    
    return parser
    




def bce_with_logits(x, y):
    
    return F.binary_cross_entropy_with_logits(
        x,
        torch.eye(x.shape[1])[y].to(y.device)
    )


    

def main(args):
    out_file = open("logs_metrics_DyTox_rehe_fixedmem.json", 'w')
    wandb.init(project="CL FSC (DyTox)", name="DyTox_rehe_fixedmemory",entity="umbertocappellazzo",config = {"lr": args.lr, "weight_decay":args.weight_decay, "epochs":args.epochs, "batch size": args.batch_size})
    
    print(args)
    
    
    
    
    
    # Create the logger for tracking and computing the metrics thoughout the training and test phases.
    logger = Logger(list_subsets=['train','test'])
    use_distillation = args.auto_kd
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create the train and test dataset splits + corresponding CI scenarios. 
    dataset_train = FluentSpeech(args.data_path,train=True,download=False)
    dataset_valid = FluentSpeech(args.data_path,train="valid",download=False)
    dataset_test = FluentSpeech(args.data_path,train=False,download=False)
    
    
    #scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    #scenario_valid = ClassIncremental(dataset_valid,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    scenario_test = ClassIncremental(dataset_test,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    
    # Policy 1: one Frequency and one Time maskings for SpecAugment. Policy 2: two Frequency and two Time maskings.
    
    #scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len),partial(SpecAugment,F=10,T=100,double=False)])
    scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    #scenario_train_policy1 = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len),partial(SpecAugment,F=10,T=100,double=False)])
    #scenario_train_policy2 = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len),partial(SpecAugment,F=10,T=100,double=True)])
    
    #scenario_train = ClassIncremental(dataset_train,nb_tasks=1,transformations=[partial(trunc, max_len=args.max_len)])
    #scenario_test = ClassIncremental(dataset_test,nb_tasks=1,transformations=[partial(trunc, max_len=args.max_len)])
    
    
   
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = bce_with_logits
    #criterion_test = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    
    
    
   # KD DISTILLATION. 
    teacher_model = None
    
    
    # Memory for rehearsal
    memory = None
    if args.memory_size > 0:
        memory = rehearsal.RehearsalMemory(args.memory_size, herding_method= 'random', fixed_memory=True, nb_total_classes=scenario_train.nb_classes)
        
        #memory = Memory(
        #    args.memory_size, scenario_train.nb_classes, args.rehearsal, args.fixed_memory
        #)
    
    nb_classes = args.initial_increment
    lr = args.lr
    
    start_time = time.time()
    
    epochs = args.epochs
    
   
    if use_distillation:
        print("Knowledge Distillation loss exploited in this experiment")
    if args.head_div_coeff > 0:
        print("Divergence loss exploited in this experiment")
    
    ###########################################################################
    #                                                                         #
    # Begin of the task loop                                                  #
    #                                                                         #
    ###########################################################################
    #out_file = open("logs_metrics_DyTox_CL_fixfreeze_norehear_bce.json", 'w')
    
    #dytox_weightdecay_31classes_valid
    #wandb.init(project="CL FSC (DyTox)", name="DyToxwithTCN_CL",entity="umbertocappellazzo",config = {"lr": args.lr, "weight_decay":args.weight_decay, "epochs":args.epochs, "batch size": args.batch_size})
    #wandb.config = {"lr": args.lr, "weight_decay":args.weight_decay, "epochs":args.epochs, "batch size": args.batch_size}
    
    for task_id, exp_train in enumerate(scenario_train):
        
        
        print("Shape of exp_train: ",len(exp_train))
        if args.max_task == task_id:
            print("Stop training because of max task")
            break
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")
        
        #out_file.write(f'Metrics result for task #{task_id}' + '\n' + '\n')
    # In case of distillation
        if use_distillation and task_id > 0:
            teacher_model = copy.deepcopy(model)
            teacher_model.freeze(['all'])
            teacher_model.eval()
        
        # Definition of the model Dytox: if task_id == 0, first initialization; 
        # else, the model must be updated (classifiers + task tokens).
        
        
        if task_id > 0 and memory is not None and not args.sep_memory:
            
            
            
            previous_size = len(exp_train)
            
            exp_train.add_samples(*memory.get())   # Standard command w/ou augment.
            print(f"{len(exp_train) - previous_size} samples added from memory.")
        
        
        if task_id == 0:
            print('Creating DyTox model for SLU (FSC)!')
            #model = DyTox_slu_AST(nb_classes,head_div=args.head_div_coeff>0).to(device)  # DyTox AST.
            model = DyTox_slu(nb_classes,head_div=args.head_div_coeff>0).to(device)
            #model = DyTox_slu(nb_classes).to(device)    # For TCN inside DyTox
            #model = TCN(in_chan=40,out_chan = (nb_classes,)).cuda()
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
            #for name, param in model.named_parameters(): 
            #    print(name,param.shape) if param.requires_grad == True else print(f"{name} param has no grad set to True with size {param.size}")
            print('number of params of overall DyTox:', n_parameters)
            #n_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            #print('number of params of the TCN:', n_parameters)
        else:
            print(f'Updating DyTox, {args.increment} new classes + 1 new task token for the {task_id}-th task.')
            model.update_model(args.increment)
            #model.classif.add_new_outputs(args.increment)   # For TCN inside DyTox
            #model.update(args.increment)   # For TCN
            
            #for name, param in model.named_parameters(): 
            #    print(name,param.shape)
            
            
        #wandb.watch(model)
        # Freeze all task tokens and heads/classifiers before the actual task token.
        if task_id > 0:
            print("Freezing past tokens and heads")
            model.freeze(args.freeze_task)
       
    # Optimizer definition: a new optimizer for each task since the model is extended by the time we 
    # enter a new task.
        
        #optimizer = create_optimizer(args, model)
        
        #lr_scheduler, _ = create_scheduler(args, optimizer)
        
        
        #optimizer = Adam(model.parameters(),lr=lr,weight_decay=args.weight_decay)
        #optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr,weight_decay=args.weight_decay)
        optimizer = AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=lr,weight_decay=args.weight_decay)
        #optimizer = AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay)
        
        
        test_taskset = scenario_test[:task_id+1]    # Evaluation on all seen tasks.
        #concat_exps_test = torch.utils.data.ConcatDataset([test_taskset,scenario_test_aug[:task_id+1]])  # FOR augmented case.
        
        # For scenario w/ augmentation using collate_fn.
        train_loader = DataLoader(exp_train, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True,drop_last=False)
        #valid_loader = DataLoader(scenario_valid[:task_id+1], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_taskset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        # For scenario w/ augmentation using 3 datasets (no collate_fn).
        
        #train_loader = DataLoader(exp_train, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        #train_loader_policy1 = DataLoader(scenario_train_policy1[task_id], batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        #train_loader_policy2 = DataLoader(scenario_train_policy2[task_id], batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        #test_loader = DataLoader(test_taskset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        
        # mean,std = [], []
        # print(f"Normalizing train task {task_id} samples:")
        # for x_,_,_ in train_loader:
        #     cur_mean = torch.mean(x_)
        #     cur_std = torch.std(x_)
        #     mean.append(cur_mean)
        #     std.append(cur_std)
            
        # print(np.mean(mean),np.mean(std))
        # mean = np.mean(mean)
        # std = np.mean(std)
        
        # mean_test,std_test = [], []
        # print(f"Normalizing test task {task_id} samples:")
        # for x_,_,_ in test_loader:
        #     cur_mean = torch.mean(x_)
        #     cur_std = torch.std(x_)
        #     mean_test.append(cur_mean)
        #     std_test.append(cur_std)
            
        # print(np.mean(mean_test),np.mean(std_test))
        # mean_test = np.mean(mean_test)
        # std_test = np.mean(std_test)
        
        
        
        
        
        #best_accuracy = 0
        #print(len(train_loader))
        
        #num_steps = len(train_loader) *epochs
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        
        #warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
        #warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=105)
        
    
    ###########################################################################
    #                                                                         #
    # Begin of the train and test loops                                       #
    #                                                                         #
    ###########################################################################
        print(f"Start training for {epochs} epochs")
        #max_accuracy = 0.0
        for epoch in range(epochs):
            
            
            model.train()
            train_loss = 0.
            print(f"Epoch #: {epoch}")
            #train_loader_iter = iter(train_loader)
            #train_loader_iter_1 = iter(train_loader_policy1)
            #train_loader_iter_2 = iter(train_loader_policy2)
            
            for x,y,t in train_loader:
            #for minibatch in range(len(train_loader)):
            
                
                
                #x,y,t = next(train_loader_iter)
                #x_1,y_1,t_1 = next(train_loader_iter_1)
                #x_2,y_2,t_2 = next(train_loader_iter_2)
                #x,y,t = torch.cat([x,x_1, x_2]), torch.cat([y,y_1, y_2]), torch.cat([t,t_1, t_2])
                
                x = x.to(device); #x = (x-mean)/std
                y = y.to(device)
                #print(x.shape)
                #print(y.shape)
                
                optimizer.zero_grad()
                predict_dict = model(x)#['logits']
                predictions = predict_dict['logits']
                
                
                #print(predictions)
                #print(y[:,0])
                #print(y)
                #print(predictions.detach().cpu().numpy().size)
                #print(y[:,0].detach().cpu().numpy().size)
                #print(predictions.shape)
                #print(y.shape)
                #print(y[:,0].shape)
                #print(predictions[:5,:].argmax(dim=1))
                #print(y[:5,0])
                loss = criterion(predictions,y[:,0])  #y[:,0]
                
                
                if teacher_model is not None:
                    with torch.no_grad():
                        predictions_old = teacher_model(x)['logits']
                
                    loss = get_kdloss(predictions,predictions_old,loss,args.distillation_tau)
                    
                if model.head_div is not None:
                    total_classes = predictions.shape[1]
                    nb_new_classes = predict_dict['div'].shape[1] - 1
                    nb_old_classes = total_classes - nb_new_classes
                    div_targets = torch.clone(y[:,0])
                    mask_old_cls = div_targets < nb_old_classes
                    mask_new_cls = ~mask_old_cls
                    div_targets[mask_old_cls] = 0
                    div_targets[mask_new_cls] -= nb_old_classes - 1
                    div_loss = args.head_div_coeff * criterion(predict_dict['div'], div_targets)
                    loss  += div_loss
                
                
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                #lr_scheduler.step(epoch)
                
                
                #with warmup_scheduler.dampening():
                #    lr_scheduler.step()
                
                
                #print(f"train loss for each batch: {loss.item()}")
                #break
                logger.add([predictions.cpu().argmax(dim=1),y[:,0].cpu(),t], subset= 'train')
                
                #if args.eval_every and (epoch+1) % args.eval_every == 0:
                    
                    #train_loss /= len(train_loader)
                    #print(f"train_loss: {train_loss}")
                    #print(predictions)
                    #print(y[:,0])
                
            
            
            
            
            # Test phase
            if args.eval_every and (epoch+1) % args.eval_every == 0:
                model.eval()
                test_loss = 0.
                train_loss /= len(train_loader)
                print(f"train_loss: {train_loss}")
                
                #print(predictions)
                #print(y[:,0])
                #with torch.inference_mode(): 
                with torch.no_grad():
                    for x_valid, y_valid, t_valid in test_loader:
                        predic_valid = model(x_valid.cuda())['logits']
                        #predic_valid = (predic_valid - mean_test)/std_test
                        test_loss += criterion(predic_valid,y_valid[:,0].cuda()).item()
                        #print(predic_valid[:5,:].argmax(dim=1))
                        #print(y_valid[:5,0])
                        
                        logger.add([predic_valid.cpu().argmax(dim=1),y_valid[:,0].cpu(),t_valid], subset = 'test')
                    test_loss /= len(test_loader)
                    wandb.log({"train_loss": train_loss, "valid_loss": test_loss,"train_acc": logger.online_accuracy,"valid_acc": logger.accuracy })
                    print(f"Train accuracy: {logger.online_accuracy}")
                    print(f"Valid accuracy: {logger.accuracy}")
                    print(f"Valid loss at epoch {epoch} and task {task_id}: {test_loss}")
                    
            
            
            json.dump({"task": task_id, "epoch": epoch, #"current lr": lr_scheduler.get 
                       "valid_acc": round(100*logger.accuracy,2), "train_acc": round(100*logger.online_accuracy,2),
                       "avg_acc": round(100 * logger.average_incremental_accuracy, 2),"online_cum_perf": round(100*logger.online_cumulative_performance,2),
                        'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task], 'bwt': round(100 * logger.backward_transfer, 2),'forgetting': round(100 * logger.forgetting, 6),
                        'train_loss': round(train_loss, 5), 'valid_loss': round(test_loss, 5)}, out_file,ensure_ascii = False )
            out_file.write('\n')
            
            logger.end_epoch()
            
            
        if memory is not None:
            #task_memory_path = os.path.join(args.resume, f'memory_{task_id}.npz')
            #if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                # Resuming this task step, thus reloading saved memory samples
                # without needing to re-compute them
                #memory.load(task_memory_path)
            #else:
            #    memory.add(scenario_train[task_id], model, args.initial_increment if task_id == 0 else args.increment)
            #    print(len(memory))
            #    if args.resume != '':
            #        memory.save(task_memory_path)
            #    else:
            #        memory.save(os.path.join(args.output_dir, f'memory_{task_id}.npz'))
            memory.add(*scenario_train[task_id].get_raw_samples(),z=None)   # FOr normal scenario w/out augmentation.
            #concat_datasets = torch.utils.data.ConcatDataset([scenario_train[task_id],scenario_train_aug[task_id]])
            #concat_datasets = 
            #memory.add(*concat_datasets.get_raw_samples(),z=None)
            print(len(memory))
            
            assert len(memory) <= args.memory_size    
            
          
            
        if args.finetuning and memory and task_id > 0:
            
            dataset_finetune = get_finetuning_dataset(exp_train, memory)
            print(f'Finetuning phase with {len(dataset_finetune)} samples.')
            loader_finetune = DataLoader(dataset_finetune, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
            loader_test_finetune = DataLoader(test_taskset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
            print(f'Train-ft and val loaders of lengths: {len(loader_finetune)} and {len(loader_test_finetune)}.')
            model.freeze('tcn')
            optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr,weight_decay=args.weight_decay)
            for epoch in range(args.finetuning_epochs):
                
                model.train()
                train_loss = 0.
                print(f"Finetune Epoch #: {epoch}")
                for x,y,t in loader_finetune:
                    x = x.to(device)
                    y = y.to(device)
                    
                    optimizer.zero_grad()
                    predict_dict = model(x)#['logits']
                    predictions = predict_dict['logits']
                    loss = criterion(predictions,y[:,0])
                    
                    if teacher_model is not None:
                        with torch.no_grad():
                            predictions_old = teacher_model(x)['logits']
                    
                        loss = get_kdloss(predictions,predictions_old,loss,args.distillation_tau)
                    
                    if model.head_div is not None:
                        total_classes = predictions.shape[1]
                        nb_new_classes = predict_dict['div'].shape[1] - 1
                        nb_old_classes = total_classes - nb_new_classes
                        div_targets = torch.clone(y[:,0])
                        mask_old_cls = div_targets < nb_old_classes
                        mask_new_cls = ~mask_old_cls
                        div_targets[mask_old_cls] = 0
                        div_targets[mask_new_cls] -= nb_old_classes - 1
                        div_loss = args.head_div_coeff * criterion(predict_dict['div'], div_targets)
                        loss  += div_loss
                    
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    
                    logger.add([predictions.cpu().argmax(dim=1),y[:,0].cpu(),t], subset= 'train')
                    
                if args.eval_every and (epoch+1) % args.eval_every == 0:
                    model.eval()
                    test_loss = 0.
                    train_loss /= len(train_loader)
                    print(f"train_loss: {train_loss}")
                    with torch.no_grad():
                        for x_valid, y_valid, t_valid in loader_test_finetune:
                            predic_valid = model(x_valid.cuda())['logits']
                            
                            test_loss += criterion(predic_valid,y_valid[:,0].cuda()).item()
                            logger.add([predic_valid.cpu().argmax(dim=1),y_valid[:,0].cpu(),t_valid], subset = 'test')
                        test_loss /= len(test_loader)
                        wandb.log({"train_loss": train_loss, "valid_loss": test_loss,"train_acc": logger.online_accuracy,"valid_acc": logger.accuracy })
                        print(f"Train accuracy: {logger.online_accuracy}")
                        print(f"Valid accuracy: {logger.accuracy}")
                        print(f"Valid loss at epoch {epoch} and task {task_id}: {test_loss}")
                        
                    
                
                
                
                
            
           
           
           
            # CHECK WHETHER TOKENS AND HEADS ARE CORRECTLY FROZEN OR NOT.
            #for id,tok in enumerate(model.task_tokens):
                #print(tok.requires_grad)
            #    print(f"Token of task {id} and epoch {epoch}: {tok[:,:,0:5]}")
            # for id, heads in enumerate(model.head):
            #     print(f"Head of task {id} and epoch {epoch}: {heads.head.weight.data[0:1,0:10]}")
            #     for p in heads.parameters():
            #         print(p.requires_grad)  
                
            
            #print(logger.average_incremental_accuracy)
            #print([round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task])
                json.dump({"task": task_id, "epoch": epoch, #"current lr": lr_scheduler.get 
                       "valid_acc": round(100*logger.accuracy,2), "train_acc": round(100*logger.online_accuracy,2),
                       "avg_acc": round(100 * logger.average_incremental_accuracy, 2),"online_cum_perf": round(100*logger.online_cumulative_performance,2),
                        'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task], 'bwt': round(100 * logger.backward_transfer, 2),'forgetting': round(100 * logger.forgetting, 6),
                        'train_loss': round(train_loss, 5), 'valid_loss': round(test_loss, 5)}, out_file,ensure_ascii = False )
                out_file.write('\n')
            
            
            #         json.dump({"task": task_id, "epoch": epoch, "acc": round(100*logger.accuracy,2), "online_acc": round(100*logger.online_accuracy,2),
            #            'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task], 
            #            'train_loss': round(loss.item(), 5), 'test_loss': round(test_loss.item(), 5)}, out_file,ensure_ascii = False )
            
                #     out_file.write('/n')
                # model.train()
            #"avg_acc": round(100 * logger.average_incremental_accuracy, 2)
           # "online_cum_perf": round(100*logger.online_cumulative_performance,2),
            
        
        

            
        
        
        out_file.write('\n')
        
        
        #if memory is not None:
            #task_memory_path = os.path.join(args.resume, f'memory_{task_id}.npz')
            #if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                # Resuming this task step, thus reloading saved memory samples
                # without needing to re-compute them
                #memory.load(task_memory_path)
            #else:
            #    memory.add(scenario_train[task_id], model, args.initial_increment if task_id == 0 else args.increment)
            #    print(len(memory))
            #    if args.resume != '':
            #        memory.save(task_memory_path)
            #    else:
            #        memory.save(os.path.join(args.output_dir, f'memory_{task_id}.npz'))
         #   memory.add(*scenario_train[task_id].get_raw_samples(),z=None)
            #print(len(memory))
            
          #  assert len(memory) <= args.memory_size
        
                    
        
        logger.end_task()   
        
       
        print(f'task id: {task_id}')
        
        
    out_file.write('Parameters: \n')
    json.dump({"starting lr": args.lr,"epochs": args.epochs, "batch_size": args.batch_size, "initial_increment": args.initial_increment,
               "optimizer":"Adam", "label_smoothing":args.label_smoothing, "loss_type": "cross_entropy loss", "weight_decay": args.weight_decay},out_file,ensure_ascii = False )
    #torch.save(logger,'logger.pt')   
    
              
    out_file.close()            
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.finish()     
            
            
            
            
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DyTox bare-version for SLU (FSC) train and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
    #wandb.init(project="CL FSC (DyTox)", name="DyTox_CL_WD_adamW_sweep_hyperparam",entity="umbertocappellazzo")
    
    # sweep_config = {'method': 'grid'}
    
    # parameters_dict = {
    #     'weight_decay': {
    #         'values': [0.1,0.02,0.05]
    #         },
    #     'lr': {
    #         'values': [1e-4,5e-4,1e-3]
    #         },
    #     }
    
    # sweep_config['parameters'] = parameters_dict
    # parameters_dict.update({
    #     'epochs': {
    #         'value':75
    #         }
    #     })    
    # sweep_id = wandb.sweep(sweep_config,entity="umbertocappellazzo",project="CL FSC (DyTox)")
    
    # def train(config=None):
    #     with wandb.init(config=config):
    #         config = wandb.config
    #         main(args,config)
            
    # wandb.agent(sweep_id,train,entity="umbertocappellazzo",project="CL FSC (DyTox)")
    
    
    #from torchaudio import transforms
    
   #  def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
   #      fig, axs = plt.subplots(1, 1)
   #      axs.set_title(title or 'Spectrogram (db)')
   #      axs.set_ylabel(ylabel)
   #      axs.set_xlabel('frame')
   #      im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
   #      if xmax:
   #          axs.set_xlim((0, xmax))
   #      fig.colorbar(im, ax=axs)
   #      plt.show(block=False)
        
    
    
    
    # def collate_batch(data):
    #     spectrogram_data = []
        
    #     x,y,t = zip(*data)
    #     y = torch.tensor(y)
        
    #     for x,_,_ in data:
    #         spec = SpecAugment(x,F=10,T=100,double=False)
    #         spectrogram_data.append(x)
    #         spectrogram_data.append(spec)
    #         spec = SpecAugment(x,F=14,T=140,double=True)
    #         spectrogram_data.append(spec)
    #     x = torch.stack(spectrogram_data)
    #     idx = torch.randperm(x.size(0))
    #     return x[idx,:,:].float(),y,torch.tensor(t)
    
    #from torchaudio import transforms as T
    #data_path ='/Users/umbertocappellazzo/Desktop/PHD'  #'/data/cappellazzo/CL_SLU/' 
    #dataset = FluentSpeech(data_path,train=True,download=False)
     #scenario = ClassIncremental(dataset, nb_tasks=31, transformations=[partial(trunc, max_len=64000)])
    #scenario = ClassIncremental(dataset,increment=3,initial_increment=4,transformations=[partial(trunc, max_len=64000)])
    #scenario_aug = ClassIncremental(dataset,increment=3,initial_increment=4,transformations=[partial(trunc, max_len=64000),partial(SpecAugment,F=10,T=100,double=True)])
     #out_file = open("FSC_class_distribution.json", 'w')
     #masking = T.FrequencyMasking(freq_mask_param=13)
     #time = T.TimeMasking(time_mask_param=100)
     #warp = T.TimeStretch()
    #for task_id, exp_train in enumerate(scenario):
        
        #train_loader = DataLoader(torch.utils.data.ConcatDataset([exp_train,scenario_aug[task_id]]), batch_size=10,shuffle=False,num_workers=0,pin_memory=True)
    #    train_loader = DataLoader(exp_train, batch_size=10,collate_fn=collate_batch,shuffle=False,num_workers=0,pin_memory=True)
    #    print(len(train_loader))
    #    for x,y,t in train_loader:
    #        pass
    
            
    
    #         plot_spectrogram(x[0,:,:])
             #mask = masking(x[0,:,:])
             #warped_sig = warp(x[0,:,:],1.2)
             #plot_spectrogram(warped_sig)
            
    #         break
            
            #             if count ==0:
    #                 print(x[0,:4,:4])
    #                 count +=1
        
        #print(scenario.classes)
    #    print(len(train_loader))
    #    json.dump({f'Class id: {task_id}, class name: {list(dataset.class_ids.keys())[task_id]}': len(train_loader)},out_file,ensure_ascii = False)
    #    out_file.write('\n')
        #if task_id ==2:
        #    break
    #out_file.close()  
    
    # mode = "test"
    
    # if mode == "valid":
    #     dataset = FluentSpeech(data_path,train=True,download=False)
    #     dataset_valid = FluentSpeech(data_path,train="valid",download=False)
    #     #params = {'batch_size': 100,'shuffle': True}
    #     #train_set_generator = data.DataLoader(dataset.get_data(), **params)
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     modelname = "best_model_continuum_alessiocode.pkl"
    
        
    #     #scenario = ClassIncremental(dataset, increment=3, initial_increment=4, transformations=[partial(trunc, max_len=64000)])
    #     #scenario_test = ClassIncremental(dataset_test, increment=3, initial_increment=4, transformations=[partial(trunc, max_len=64000)])
    
    #     scenario = ClassIncremental(dataset, nb_tasks=1, transformations=[partial(trunc_new, max_len=64000)])
    #     scenario_valid = ClassIncremental(dataset_valid,nb_tasks=1, transformations=[partial(trunc_new, max_len=64000)])
    #     best_accuracy = 0
    #     criterion = torch.nn.CrossEntropyLoss()
    #     for task_id, exp_train in enumerate(scenario):
    #         model = TCN(in_chan = 40, n_blocks=5, n_repeats=2, out_chan=(31, )).to(device)
            
    #         optimizer = Adam(model.parameters(), lr=0.001)
    #         valid_taskset = scenario_valid[:task_id+1] 
    #         train_loader = DataLoader(exp_train, batch_size=100,shuffle=True,num_workers=10, pin_memory=True)
    #         valid_loader = DataLoader(valid_taskset, batch_size=100, num_workers=10,shuffle=True, pin_memory=True)
            
    #         for e in range(100):
    #             for (c, (i,d,t)) in enumerate(train_loader):
    #                 model.train()
    #                 #f, l = d
    #                 f = i; l = d
    #                 y = model(f.float().to(device))
                    
    #                 loss = criterion(y[0], l[:,0].to(device))
    #                 print("Iteration %d in epoch%d--> loss = %f"%(c, e, loss.item()), end='\r')
    #                 loss.backward()
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 if c % 100 == 0:
                        
    #                     model.eval()
    #                     correct = []
                
    #                     for j, eval_data,t1 in valid_loader:
    #                         feat, label = j,eval_data
    #                         y_eval = model(feat.float().to(device))
    #                         pred = torch.argmax(y_eval[0].detach().cpu(), dim=1)
    #                         intent_pred = pred
    #                         correct.append((intent_pred == label[:,0]).float())
                        
    #                     acc = np.mean(np.hstack(correct))
    #                     intent_acc = acc
    #                     iter_acc = '\n iteration %d epoch %d-->' % (c, e)
    #                     print(iter_acc, acc, best_accuracy)
    #                     if intent_acc > best_accuracy:
    #                         improved_accuracy = 'Current accuracy = %f (%f), updating best model' % (intent_acc, best_accuracy)
    #                         print(improved_accuracy)
    #                         best_accuracy = intent_acc
    #                         best_epoch = e
    #                         torch.save(model.state_dict(), modelname)
    #         print(f"The best epoch is: {best_epoch}")
            
    # else:
            
    #     dataset_test = FluentSpeech(data_path,train=False,download=False)
    #     dataset_valid = FluentSpeech(data_path,train="valid",download=False)
            
    #     scenario_test = ClassIncremental(dataset_test, nb_tasks=1, transformations=[partial(trunc_new, max_len=64000)])
    #     scenario_valid = ClassIncremental(dataset_valid,nb_tasks=1, transformations=[partial(trunc_new, max_len=64000)])
        
        
        
        
        
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     print('Using device %s' % device)
        
    #     model = TCN(in_chan = 40, n_blocks=5, n_repeats=2, out_chan=(31, )).to(device)
    #     modelname = "best_model_continuum_alessiocode.pkl"
        
    #     model.load_state_dict(torch.load(modelname))
    #     model.eval()
    #     model.to(device)
        
        
    #     for task_id, exp_train in enumerate(scenario_test):
            
    #         test_loader = DataLoader(exp_train, batch_size=100,shuffle=True, num_workers=10, pin_memory=True)
            
            
            
            
    #         correct_test = []
        
    #         for i, d,t in test_loader:
    #             feat, label = i,d
    #             #print('Iter %d (%d/%d)' % (i, i * 100, len(test_data)), end='\r')
    #             z_eval = model(feat.float().to(device))
    #             pred = [torch.max(z.detach().cpu(), dim=1)[1] for z in z_eval]
    #             pred_test = pred[0]
    #             correct_test.append(np.array(pred_test == label[:,0], dtype=float))
            
    #         acc_test = (np.mean(np.hstack(correct_test)))
    #         print("The accuracy on test set is %f" % (acc_test))
        
        
        
    #     for task_id, exp_train in enumerate(scenario_valid):
        
    #         correct_valid = []
    #         valid_loader = DataLoader(exp_train, batch_size=100,shuffle=True, num_workers=10, pin_memory=True)
        
    #         for i, d,t in valid_loader:
            
            
    #             feat, label = i,d
            
    #             a_eval = model(feat.float().to(device))
    #             pred = [torch.max(a.detach().cpu(), dim=1)[1] for a in a_eval]
            
    #             pred_test = pred[0]
            
    #             correct_valid.append(np.array(pred_test == label[:,0], dtype=float))
        
    #     acc_val = (np.mean(np.hstack(correct_valid)))
    #     print("The accuracy on the validation set is %f" % (acc_val))
            
    
        