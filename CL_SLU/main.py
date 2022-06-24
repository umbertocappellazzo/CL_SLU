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
from utils import trunc, Memory, InfiniteLoader, trunc_new
import time
import datetime
import json
#import pytorch_warmup as warmup
import wandb
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils import data
from continuum import rehearsal


def get_args_parser():
    parser = argparse.ArgumentParser('DyTox bare-version for SLU (FSC) train and evaluation', add_help=False)
    
    # Dataset parameters.
    parser.add_argument('--data_path', type=str, default='/data/cappellazzo/CL_SLU/',help='path to dataset')
    parser.add_argument('--max_len', type=int, default=64000, help='max length for the audio signal --> it will be cut')
    parser.add_argument('--download_dataset', default=False, help='whether to download the FSC dataset or not')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type= str, default='cuda', help='device to use for training/testing')
    
    
    
    
    # Training/inference parameters.
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--eval_every", type=int, default=1, help="Eval model every X epochs, if None only eval at the task end")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--label_smoothing', type=float, default=0., help='Label smoothing for the CE loss')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    
    
    parser.add_argument('--output_basedir', default='./checkponts/',
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
    
    #Optimizer
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
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
    
    # Learning rate schedule parameters
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
    out_file = open("logs_metrics_DyTox_CL_fixfreeze_norehear_bce.json", 'w')
    wandb.init(project="CL FSC (DyTox)", name="DyTox_CL_4%_layernorm_notglobal",entity="umbertocappellazzo",config = {"lr": args.lr, "weight_decay":args.weight_decay, "epochs":args.epochs, "batch size": args.batch_size})
    
    print(args)
    
    
    
    
    
    # Create the logger for tracking and computing the metrics thoughout the training and test phases.
    logger = Logger(list_subsets=['train','test'])
    
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    #seed = args.seed
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    
    # Create the train and test dataset splits + corresponding CI scenarios. 
    dataset_train = FluentSpeech(args.data_path,train=True,download=False)
    dataset_valid = FluentSpeech(args.data_path,train="valid",download=False)
    dataset_test = FluentSpeech(args.data_path,train=False,download=False)
    scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    scenario_valid = ClassIncremental(dataset_valid,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    scenario_test = ClassIncremental(dataset_test,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    #scenario_train = ClassIncremental(dataset_train,nb_tasks=1,transformations=[partial(trunc, max_len=args.max_len)])
    #scenario_test = ClassIncremental(dataset_test,nb_tasks=1,transformations=[partial(trunc, max_len=args.max_len)])
    
    
   
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = bce_with_logits
    #criterion_test = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    
    
    
    #USE DISTILLATION HERE:
    #teacher_model = None
    #Code below goes inside task loop
    #if use_distillation and task_id > 0:
        #teacher_model = copy.deepcopy(model_without_ddp)
        #teacher_model.freeze(['all'])
        #teacher_model.eval()
    
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
        
        
        if args.max_task == task_id:
            print("Stop training because of max task")
            break
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")
        
        #out_file.write(f'Metrics result for task #{task_id}' + '\n' + '\n')
    # In case of distillation
    # if use_distillation and task_id > 0:
    #     teacher_model = copy.deepcopy(model_without_ddp)
    #     teacher_model.freeze(['all'])
    #     teacher_model.eval()
        
        # Definition of the model Dytox: if task_id == 0, first initialization; 
        # else, the model must be updated (classifiers + task tokens).
        
        loader_memory = None
        if task_id > 0 and memory is not None and not args.sep_memory:
            #dataset_memory = memory.get_dataset(dataset_train)
            #data_loader = DataLoader(dataset_memory,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,drop_last=True)
            #loader_memory = InfiniteLoader(data_loader)
            #if not args.sep_memory:
            previous_size = len(exp_train)
            exp_train.add_samples(*memory.get())
            print(f"{len(exp_train) - previous_size} samples added from memory.")
        
        
        if task_id == 0:
            print('Creating DyTox model for SLU (FSC)!')
            model = DyTox_slu(nb_classes).to(device)
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
            #model.classif.add_new_outputs(args.increment)
            #model.update(args.increment)   # For TCN
            
            #for name, param in model.named_parameters(): 
            #    print(name,param.shape)
            
            
        #wandb.watch(model)
        # Freeze all task tokens and heads/classifiers before the actual task token.
        if task_id > 0:
            print("Freezing past tokens and heads")
            model.freeze(args.freeze_task)
        #model = TCN(in_chan=40,out_chan = (31,)).cuda()
    # Optimizer definition: a new optimizer for each task since the model is extended by the time we 
    # enter a new task.
        
        #optimizer = create_optimizer(args, model)
        
        #lr_scheduler, _ = create_scheduler(args, optimizer)
        
        
        #optimizer = Adam(model.parameters(),lr=lr,weight_decay=args.weight_decay)
        optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr,weight_decay=args.weight_decay)
        #optimizer = AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=lr,weight_decay=args.weight_decay)
        #optimizer = AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay)
        test_taskset = scenario_test[:task_id+1]    # Evaluation on all seen tasks.
        #train_taskset = scenario_train[:task_id+1]
        #train_loader = DataLoader(train_taskset, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        train_loader = DataLoader(exp_train, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        #valid_loader = DataLoader(scenario_valid[:task_id+1], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_taskset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
        #best_accuracy = 0
        
        #num_steps = len(train_loader) *epochs
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        #warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    
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
            for x,y,t in train_loader:
                x = x.to(device)#[0,:].unsqueeze(0)
                y = y.to(device)#[0,:].unsqueeze(0)
                #print(x.shape)
                #print(y.shape)
                
                optimizer.zero_grad()
                predictions = model(x)['logits']
                
                
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
                        
                        test_loss += criterion(predic_valid,y_valid[:,0].cuda()).item()
                        #print(predic_valid[:5,:].argmax(dim=1))
                        #print(y_valid[:5,0])
                        
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
            logger.end_epoch()
        
        

            
        
        
        out_file.write('\n')
        
        
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
            memory.add(*scenario_train[task_id].get_raw_samples(),z=None)
            #print(len(memory))
            
            assert len(memory) <= args.memory_size
        
                    
        
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
    
    
    
    
    #data_path = '/Users/umbertocappellazzo/Desktop/PHD' #'/data/cappellazzo/CL_SLU/' 
    #dataset = FluentSpeech(data_path,train=True,download=False)
    #scenario = ClassIncremental(dataset, nb_tasks=31, transformations=[partial(trunc, max_len=64000)])
    #out_file = open("FSC_class_distribution.json", 'w')
    
    #for task_id, exp_train in enumerate(scenario):
        
    #    train_loader = DataLoader(exp_train, batch_size=1,shuffle=False)
        
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
            
            










#for i,taskset in enumerate(scenario_test):
    #    loader = DataLoader(taskset, batch_size=200,shuffle=True)
    #     if i == 1:
    #         break
    #    count = 0
    #    for x, y, t in loader:
    #        print(len(x))
    #        count += len(x)
            #         print(x.shape, y[:,0])
    # for i,taskset in enumerate(scenario):
    #     loader = DataLoader(taskset, batch_size=10,shuffle=True)
    #     if i ==1:
    #         break
    #     for x, y, t in loader:
    #         print(x.shape, y[:,0])
            
        
        
            
        
        
    
        
        
        