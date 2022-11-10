#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 08:51:04 2022

@author: umbertocappellazzo
"""

import copy
from torch.utils.data import DataLoader
from functools import partial
from torch.optim import Adam
from continuum import ClassIncremental
from continuum.datasets import FluentSpeech
import torch
import argparse
from continuum.metrics import Logger
import numpy as np
from model import CL_model
from tools.utils import trunc, get_kdloss,get_kdloss_onlyrehe, freeze_parameters
import time
import datetime
import json
import wandb
from continuum import rehearsal
from statistics import mean
import math


def get_args_parser():
    parser = argparse.ArgumentParser('CiCL for Spoken Language Understandig (Intent classification) on FSC: train and evaluation',
                                     add_help=False)
    
    # Dataset parameters.
    
    parser.add_argument('--data_path', type=str, default='/data/cappellazzo/CL_SLU/',help='path to dataset')
    parser.add_argument('--max_len', type=int, default=64000, 
                        help='max length for the audio signal --> it will be cut')
    parser.add_argument('--download_dataset', default=False, 
                        help='whether to download the FSC dataset or not')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type= str, default='cuda', 
                        help='device to use for training/testing')
    
    
    # Training/inference parameters.
    
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--eval_every", type=int, default=1, 
                        help="Eval model every X epochs")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--label_smoothing', type=float, default=0., 
                        help='Label smoothing for the CE loss')
    parser.add_argument('--weight_decay', type=float, default=0.)
    
    
    # Rehearsal memory.
    
    parser.add_argument('--memory_size', default=0, type=int,
                        help='Total memory size in number of stored samples.')
    parser.add_argument('--fixed_memory', default=True, action='store_true',
                        help='Dont fully use memory when no all classes are seen')
    parser.add_argument('--herding', default="random",
                        choices=[
                            'random',
                            'cluster', 'barycenter',
                        ],
                        help='Method to herd sample for rehearsal.')
    
    # DISTILLATION parameters.
    
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')
    parser.add_argument('--feat_space_kd', default='None', choices=[None,'only_rehe','all'])
    parser.add_argument('--preds_space_kd', default='None', choices=[None,'only_rehe','all'])
    
    # Continual learning parameters.
    
    parser.add_argument('--increment', type=int, default=3, 
                        help='# of classes per task/experience')
    parser.add_argument('--initial_increment', type=int, default=4, 
                        help='# of classes for the 1st task/experience')
    parser.add_argument('--nb_tasks', type=int, default=10, 
                        help='the scenario number of tasks')
    parser.add_argument('--offline_train', type=bool, default=True, 
                        help='whether to train in an offline fashion (i.e., no CL setting)')
    parser.add_argument('--total_classes', type=int, default= 31, 
                        help='The total number of classes when we train in an offline i.i.d. fashion. Set to None otherwise.')
    
    # WANDB parameters.
    
    parser.add_argument('--use_wandb', type=bool, default=True, 
                        help='whether to track experiments with wandb')
    parser.add_argument('--project_name', type=str, default='ICASSP_paper_experiments')
    parser.add_argument('--exp_name', type=str, default='prova')
    
    
    return parser
    

    

def main(args):
    out_file = open("logs_metrics.json", 'w')
    
    if args.use_wandb:
        
        wandb.init(project=args.project_name, name=args.exp_name,entity="umbertocappellazzo",
                   config = {"lr": args.lr, "weight_decay":args.weight_decay, 
                   "epochs":args.epochs, "batch size": args.batch_size})
    
    print(args)
    
    
    # Create the Continuum logger for tracking and computing the metrics throughout the training and test phases.
    
    logger = Logger(list_subsets=['train','test'])
    
    feat_space_kd = args.feat_space_kd
    preds_space_kd = args.preds_space_kd
    
    device = torch.device(args.device)
   
    # Fix the seed for reproducibility (if desired).
    #seed = args.seed
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    
    # Create the train and test dataset splits + corresponding CiCL scenarios. 
    
    dataset_train = FluentSpeech(args.data_path,train=True,download=False)
    #dataset_valid = FluentSpeech(args.data_path,train="valid",download=False)
    dataset_test = FluentSpeech(args.data_path,train=False,download=False)
    
    
    # Define the order in which the classes will be spread through the CL tasks.
    # In my experiments, I use this config and the [0,1,2,3,...] config. Just remove the 
    # class_order parameter from the scenario definition to get the latter config.
    
    class_order = [19, 27, 30, 28, 15,  4,  2,  9, 10, 22, 11,  7,  1, 25, 16, 14,  5,
             8, 29, 12, 21, 17,  3, 20, 23,  6, 18, 24, 26,  0, 13]
    
    
    
    if args.offline_train:   # Create just 1 task with all classes.
       
        scenario_train = ClassIncremental(dataset_train,nb_tasks=1,transformations=[partial(trunc, max_len=args.max_len)])
        scenario_test = ClassIncremental(dataset_test,nb_tasks=1,transformations=[partial(trunc, max_len=args.max_len)])
        
    else:
        
        scenario_train = ClassIncremental(dataset_train,increment=args.increment,initial_increment=args.initial_increment,
                                          transformations=[partial(trunc, max_len=args.max_len)],class_order=class_order)
        scenario_test = ClassIncremental(dataset_test,increment=args.increment,initial_increment=args.initial_increment,
                                         transformations=[partial(trunc, max_len=args.max_len)],class_order=class_order)
    
    # Losses employed: CE + MSE.
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_mse = torch.nn.MSELoss()
    
    
   # Prepare the teacher (previous) model if KD is employed. 
    
    teacher_model = None
    use_distillation = True if (feat_space_kd or preds_space_kd) else False
    use_both_kds = True if (feat_space_kd and preds_space_kd) else False
        
    # Memory for rehearsal
    
    memory = None
    if args.memory_size > 0:
        memory = rehearsal.RehearsalMemory(args.memory_size, herding_method= args.herding, 
                                           fixed_memory=args.fixed_memory, nb_total_classes=scenario_train.nb_classes)
        
    
    
    # Use all the classes for offline training.
    
    initial_classes = args.total_classes if args.offline_train else args.initial_increment
    
    
    start_time = time.time()
    
    epochs = args.epochs
    
   
    ###########################################################################
    #                                                                         #
    # Begin of the task loop                                                  #
    #                                                                         #
    ###########################################################################
    
    
    global_average = []
    for task_id, exp_train in enumerate(scenario_train):
        
        print("Shape of exp_train: ",len(exp_train))
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")
        
        
        # In case of distillation
        
        if use_distillation and task_id > 0:
            
            teacher_model = copy.deepcopy(model)
            freeze_parameters(teacher_model)
            teacher_model.eval()
        
        
        if task_id > 0 and memory is not None:
            
            exp_train.add_samples(*memory.get())   
        
        # Definition of the CL model: if task_id == 0, first initialization; 
        # else, the model must be updated (classifier).
        
        if task_id == 0:
            
            print('Creating the CL model:')
            model = CL_model(initial_classes,device=device).to(device)     
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of params of the model:', n_parameters)
            
        else:
            
            print(f'Updating the CL model, {args.increment} new classes for the classifier.')
            model.classif.add_new_outputs(args.increment)   
            
        
        
        optimizer = Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        
        
        test_taskset = scenario_test[:task_id+1]    # Evaluation on all seen tasks.
        
        train_loader = DataLoader(exp_train, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers,pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_taskset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)
       
        
    ###########################################################################
    #                                                                         #
    # Begin of the train and test loops                                       #
    #                                                                         #
    ###########################################################################
        print(f"Start training for {epochs} epochs")
        
        if task_id >0 and feat_space_kd == 'all':
            
            alpha = teacher_model.classif.nb_classes/model.classif.nb_classes
        
        if args.memory_size > 0:
            seen_classes = list(memory.seen_classes)
        
        # For each iteration, we consider the last 5 epochs and we take their average for better stability.
        
        last_5_epochs = []
        
        for epoch in range(epochs):
            
            
            model.train()
            train_loss = 0.
            
            print(f"Epoch #: {epoch}")
           
            for x,y,t in train_loader:
                
                optimizer.zero_grad()
                
                
                # Find samples from the current batch that correspond to the past classes (i.e., buffer memory samples).
                
                loss = 0.
                mse_loss = None
                
                # KD on the feature space applied to only rehearsal data (R).
                if task_id >0 and feat_space_kd == 'only_rehe':
                    
                    indexes_batch = []
                    for seen_class in seen_classes:
                        indexes_class = np.where(y.numpy()==seen_class)[0]
                        
                        indexes_batch.append(indexes_class)
                    indexes_batch = np.concatenate(indexes_batch)
                    
                    if len(indexes_batch) == 0:  # No rehe samples in the minibatch. Go on.
                        pass
                    else:
                    
                        x_memory = x[indexes_batch].to(device)
                        
                        current_features = model.forward_features(x_memory)
                        past_features = teacher_model.forward_features(x_memory)
                        
                        alpha = math.sqrt(len(indexes_batch)/len(x))
                        
                        mse_loss = alpha*criterion_mse(current_features,past_features)
                        loss += mse_loss
                        
                
                # KD on the feature space applied to rehearsal data + current data (D∪R).
                
                if task_id > 0 and feat_space_kd == 'all':
                                    
                    past_features = teacher_model.forward_features(x)
                    current_features = model.forward_features(x)
                    
                    alpha = np.log(1+alpha)
                    
                    mse_loss = alpha*criterion_mse(current_features,past_features)
                    loss += mse_loss
                    
                
                
                x = x.to(device)
                y = y.to(device)

                predictions = model(x)#['logits']
                    
                
                if task_id > 0 and mse_loss: # MSE KD.
                    loss += (1-alpha)*criterion(predictions,y)
                    
                else: # It handles the case for task_id == 0 (no CL) and when KD on feature space is not used.
                    loss += criterion(predictions,y)
                    
                
                # KD preds on D∪R.
                if task_id > 0 and preds_space_kd == 'all':
                    with torch.no_grad():
                        predictions_old = teacher_model(x)
                
                    loss = get_kdloss(predictions,predictions_old,loss,args.distillation_tau,use_both_kds)
                    
                
                
                if task_id > 0 and preds_space_kd == 'only_rehe':
                    
                    if not use_both_kds:   # Find the rehe samples among the training batch. When we use both KDs, 
                                           # this operation has been already performed.
                        
                        indexes_batch = []
                        for seen_class in seen_classes:
                            
                            indexes_class = np.where(y.cpu().numpy()==seen_class)[0]
                            indexes_batch.append(indexes_class)
                                
                        indexes_batch = np.concatenate(indexes_batch)
                    
                    if len(indexes_batch) == 0:
                        pass
                    else:
                        
                        x_memory = x[indexes_batch].to(device)
                                    
                        current_predictions = predictions[indexes_batch]
                        past_predictions = teacher_model(x_memory)
                    
                        kd_weight = math.sqrt(len(indexes_batch)/len(x))
                    
                        loss = get_kdloss_onlyrehe(current_predictions,past_predictions,loss,
                                                   args.distillation_tau,kd_weight,use_both_kds)
                
                
                
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                logger.add([predictions.cpu().argmax(dim=1),y.cpu(),t], subset= 'train')
            
            
            # Test phase
            if args.eval_every and (epoch+1) % args.eval_every == 0:
                model.eval()
                test_loss = 0.
                train_loss /= len(train_loader)
                
                with torch.inference_mode():
                    for x_valid, y_valid, t_valid in test_loader:
                        
                        predic_valid = model(x_valid.cuda())
                        
                        test_loss += criterion(predic_valid,y_valid.cuda()).item()  
                        logger.add([predic_valid.cpu().argmax(dim=1),y_valid.cpu(),t_valid], subset = 'test')
                    
                    if epoch in range(epochs-5,epochs):
                        last_5_epochs.append(logger.accuracy)
                    test_loss /= len(test_loader)
                    if args.use_wandb:
                        wandb.log({"train_loss": train_loss, "valid_loss": test_loss,"train_acc": 
                                   logger.online_accuracy,"valid_acc": logger.accuracy })
                    
                    print(f"Train accuracy: {logger.online_accuracy}")
                    print(f"Valid accuracy: {logger.accuracy}")
                    print(f"Valid loss at epoch {epoch} and task {task_id}: {test_loss}")
                    
            
            
            json.dump({"task": task_id, "epoch": epoch,
                        "valid_acc": round(100*logger.accuracy,2), "train_acc": round(100*logger.online_accuracy,2),
                        "avg_acc": round(100 * logger.average_incremental_accuracy, 2),"online_cum_perf": round(100*logger.online_cumulative_performance,2),
                        'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task], 'bwt': round(100 * logger.backward_transfer, 2),
                        'forgetting': round(100 * logger.forgetting, 6),'train_loss': round(train_loss, 5), 
                        'valid_loss': round(test_loss, 5)}, out_file,ensure_ascii = False )
            
            out_file.write('\n')
            logger.end_epoch()
            
        print(f"Mean of last 5 epochs of task {task_id}: ",mean(last_5_epochs)) 
        global_average.append(mean(last_5_epochs))
        
        if memory is not None:
            
            if args.herding == 'random':
                memory.add(*scenario_train[task_id].get_raw_samples(),z=None) 
            
            # For herding == 'cluster' or 'barycenter', we need to extract the feature embeddings.
            
            else:
            
                loader = DataLoader(scenario_train[task_id], batch_size=args.batch_size,shuffle=False, 
                                    num_workers=2,pin_memory=True, drop_last=False)
                
                features, targets = [], []
                
                with torch.no_grad():
                    for x, y, _ in loader:
                        feats = model.forward_features(x.cuda())
                        feats = feats.cpu().numpy()
                        y = y.numpy()
                        features.append(feats)
                        targets.append(y)
                
                features = np.concatenate(features)
                targets = np.concatenate(targets)
                
                memory.add(*scenario_train[task_id].get_raw_samples(),z=features) 
            
            
            
            assert len(memory) <= args.memory_size  
          
 
        out_file.write('\n')
 
        logger.end_task()   
    
              
    out_file.close()            
    
    print(global_average)
    print("Mean of tasks accuracy: ",mean(global_average))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.use_wandb:
        wandb.finish()     
            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser('CiCL for Spoken Language Understandig (Intent classification) on FSC: train and evaluation',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)