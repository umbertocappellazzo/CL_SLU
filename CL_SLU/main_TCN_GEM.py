#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:30:11 2022

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
from DyTox_TCNonly import DyTox_slu, TCN
#from Dytox_model import DyTox_slu, TCN
from tools.utils import trunc, get_kdloss,get_finetuning_dataset, SpecAugment,collate_batch
import time
import datetime
import json
#import pytorch_warmup as warmup
import wandb
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils import data
from continuum import rehearsal
import matplotlib.pyplot as plt
import librosa
import quadprog

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
    parser.add_argument('--weight_decay', type=float, default=0.)
    
    
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




def solve_quadprog(g,G,memory_strength=0.5):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
    



    

def main(args):
    out_file = open("logs_metrics_DyTox_CL_fixfreeze_norehear_bce.json", 'w')
    wandb.init(project="CL FSC (DyTox)", name="TCN_rehe_GEM_KD_fixedmemory",entity="umbertocappellazzo",config = {"lr": args.lr, "weight_decay":args.weight_decay, "epochs":args.epochs, "batch size": args.batch_size})
    
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
    #dataset_valid = FluentSpeech(args.data_path,train="valid",download=False)
    dataset_test = FluentSpeech(args.data_path,train=False,download=False)
    
    
    #scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    #scenario_valid = ClassIncremental(dataset_valid,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    scenario_test = ClassIncremental(dataset_test,increment=args.increment,initial_increment=args.initial_increment,transformations=[partial(trunc, max_len=args.max_len)])
    
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
        memory = rehearsal.RehearsalMemory(args.memory_size, herding_method= 'random', fixed_memory=args.fixed_memory, nb_total_classes=scenario_train.nb_classes)
        
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
        
        #concat_exps_train = torch.utils.data.ConcatDataset([exp_train,scenario_train_aug[task_id]])   # Used when we wanna merge normal and augmented datasets.
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
            #dataset_memory = memory.get_dataset(dataset_train)
            #data_loader = DataLoader(dataset_memory,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=True,drop_last=True)
            #loader_memory = InfiniteLoader(data_loader)
            #if not args.sep_memory:
            previous_size = len(exp_train)
            #concat_exps_train.add_samples(memory.get())   # For data aug.
            #exp_train.add_samples(*memory.get())   # Standard command w/ou augment.
            #print(f"{len(exp_train) - previous_size} samples added from memory.")
        
        
        if task_id == 0:
            print('Creating DyTox model for SLU (FSC)!')
            #model = DyTox_slu(nb_classes,head_div=args.head_div_coeff>0).to(device)
            model = DyTox_slu(nb_classes).to(device)      # For TCN inside DyTox
            #model = TCN(in_chan=40,out_chan = (nb_classes,)).cuda()
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
            #for name, param in model.named_parameters(): 
            #    print(name,param.shape) if param.requires_grad == True else print(f"{name} param has no grad set to True with size {param.size}")
            print('number of params of overall DyTox:', n_parameters)
            #n_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            #print('number of params of the TCN:', n_parameters)
        else:
            print(f'Updating DyTox, {args.increment} new classes + 1 new task token for the {task_id}-th task.')
            #model.update_model(args.increment)
            model.classif.add_new_outputs(args.increment)   # For TCN inside DyTox
            #model.update(args.increment)   # For TCN
            
            #for name, param in model.named_parameters(): 
            #    print(name,param.shape)
            
            
        #wandb.watch(model)
        # Freeze all task tokens and heads/classifiers before the actual task token.
        #if task_id > 0:
        #    print("Freezing past tokens and heads")
        #    model.freeze(args.freeze_task)
        #model = TCN(in_chan=40,out_chan = (31,)).cuda()
    # Optimizer definition: a new optimizer for each task since the model is extended by the time we 
    # enter a new task.
        
        #optimizer = create_optimizer(args, model)
        
        #lr_scheduler, _ = create_scheduler(args, optimizer)
        
        
        
        optimizer = Adam(model.parameters(),lr=lr,weight_decay=args.weight_decay)
        #optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr,weight_decay=args.weight_decay)
        #optimizer = AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=lr,weight_decay=args.weight_decay)
        #optimizer = AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay)
        
        test_taskset = scenario_test[:task_id+1]    # Evaluation on all seen tasks.
        #concat_exps_test = torch.utils.data.ConcatDataset([test_taskset,scenario_test_aug[:task_id+1]])  # FOR augmented case.
        
        # For normal scenario w/out augmentation.
        train_loader = DataLoader(exp_train, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,pin_memory=True, drop_last=False)
        #valid_loader = DataLoader(scenario_valid[:task_id+1], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_taskset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        #train_loader_policy1 = DataLoader(scenario_train_policy1[task_id], batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        #train_loader_policy2 = DataLoader(scenario_train_policy2[task_id], batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        
            
            
        # For augmentation case.
        
        #train_loader = DataLoader(concat_exps_train, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        #valid_loader = DataLoader(scenario_valid[:task_id+1], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        #test_loader = DataLoader(concat_exps_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        
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
            #for minibatch in range(len(train_loader)):
            for x,y,t in train_loader:
                
                
                
                x = x.to(device)
                y = y.to(device)
                if task_id > 0:
                    G = []
                    model.train()
                    for count_task in range(task_id):
                        model.train()
                        optimizer.zero_grad()
                        x_ref, y_ref, _ = memory.slice(keep_tasks=[count_task])
                        x_ref = torch.tensor(x_ref).to(device)
                        y_ref = torch.tensor(y_ref).to(device)
                        out = model(x_ref)
                        loss = criterion(out,y_ref[:,0])
                        loss.backward()
                        G.append(
                            torch.cat(
                                [
                                 p.grad.flatten() 
                                 if p.grad is not None
                                 else torch.zeros(p.numel(),device=device)
                                 for p in model.parameters()
                                ],
                                dim=0,
                            )
                        )
                    G = torch.stack(G)  # (experiences, parameters)
                
                        
                        
                
                optimizer.zero_grad()
                predict_dict = model(x)#['logits']
                predictions = predict_dict#['logits']
                
                
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
                        predictions_old = teacher_model(x)#['logits']
                
                    loss = get_kdloss(predictions,predictions_old,loss,args.distillation_tau)
                    
                # if model.head_div is not None:
                #     total_classes = predictions.shape[1]
                #     nb_new_classes = predict_dict['div'].shape[1] - 1
                #     nb_old_classes = total_classes - nb_new_classes
                #     div_targets = torch.clone(y[:,0])
                #     mask_old_cls = div_targets < nb_old_classes
                #     mask_new_cls = ~mask_old_cls
                #     div_targets[mask_old_cls] = 0
                #     div_targets[mask_new_cls] -= nb_old_classes - 1
                #     div_loss = args.head_div_coeff * criterion(predict_dict['div'], div_targets)
                #     loss  += div_loss
                
                
                train_loss += loss.item()
                loss.backward()
                
                
                with torch.no_grad():
                    if task_id > 0:
                        g = torch.cat(
                            [
                             p.grad.flatten() 
                             if p.grad is not None
                             else torch.zeros(p.numel(),device=device)
                             for p in model.parameters()
                            ],
                            dim=0,
                        )
                        to_project = (torch.mv(G, g) < 0).any()
                    else:
                        to_project = False
                    if to_project:
                        v_star = solve_quadprog(g,G).to(device)
                        num_pars = 0  # reshape v_star into the parameter matrices
                        for p in model.parameters():
                            curr_pars = p.numel()
                            if p.grad is not None:
                                p.grad.copy_(
                                    v_star[num_pars : num_pars + curr_pars].view(p.size())
                                    )
                            num_pars += curr_pars
                        assert num_pars == v_star.numel(), "Error in projecting gradient"
                        
                
                
                
                
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
                        predic_valid = model(x_valid.cuda())#['logits']
                        
                        
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
            memory.add(*scenario_train[task_id].get_samples(range(len(scenario_train[task_id]))),z=None)   # FOr normal scenario w/out augmentation.
            #memory.add(*scenario_train[task_id].get_raw_samples(),z=None)
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