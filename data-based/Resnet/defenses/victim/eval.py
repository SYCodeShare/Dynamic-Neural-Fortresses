#!/usr/bin/python

import argparse
import os.path as osp
import os
import sys
import json
from datetime import datetime
 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.optim as optim
from gan import GeneratorA

sys.path.append(os.getcwd())
from defenses import datasets
import defenses.utils.model as model_utils
import defenses.utils.utils as knockoff_utils
import defenses.config as cfg

from defenses.victim import *


def train_model(model, dataloader,testloader, optimizer, scheduler,num_epochs=10, exit_weights=[1.0, 1.0, 1.0, 1.0, 1.0]): #exit_weights=[0.4, 0.4, 0.1, 0.1, 1.0]
       
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        nz = 256
        G_activation = torch.tanh  
        generator_ood = GeneratorA(nz=nz, nc=3, img_size=224, activation=G_activation)
        device = 'cuda'
        generator_ood.apply(weights_init)
        generator_ood.to(device)
        gan_batch_size = 32

        def entropy(logits):
            probabilities = F.softmax(logits, dim=1)
            log_probabilities = F.log_softmax(logits, dim=1)
            return -(probabilities * log_probabilities).sum(dim=1)
        
        def custom_loss(outputs, labels, outputs_ood, labels_ood, exit_weights=None):
            loss = 0.0
            if exit_weights is None:
                exit_weights = [1.0 for _ in range(len(outputs))]
            entropy_weight =0.2         
            final_exit_index = len(outputs) - 1
            for exit_index, output in enumerate(outputs):
                loss += exit_weights[exit_index] * F.cross_entropy(output, labels)
                entropy_loss = entropy(output).mean()
                if exit_index != final_exit_index:
                    loss -= entropy_weight * entropy_loss  
                else:
                    loss += entropy_weight * entropy_loss  
            beta = 1  
            for exit_index, output in enumerate(outputs_ood):         
                loss += beta*exit_weights[exit_index] * F.cross_entropy(output, labels_ood)
                entropy_loss = entropy(output).mean()
                if exit_index != final_exit_index:
                    loss += beta * entropy_weight * entropy_loss  
                else:
                    loss -= beta * entropy_weight * entropy_loss  
            return loss
        def withoutood_custom_loss(outputs, labels, exit_weights=None):
            loss = 0.0
            if exit_weights is None:
                exit_weights = [1.0 for _ in range(len(outputs))]
            entropy_weight = 0.2 
            final_exit_index = len(outputs) - 1
            for exit_index, output in enumerate(outputs):          
                loss += exit_weights[exit_index] * F.cross_entropy(output, labels)
                entropy_loss = entropy(output).mean()
                if exit_index != final_exit_index:
                    loss -= entropy_weight * entropy_loss  
                else:
                    loss += entropy_weight * entropy_loss  
            return loss

        model.train()
        model.train_EE = True
        model.inference_EE = False
        use_ood = True
        every_ten_ood = True
        if use_ood:
            z = torch.randn((gan_batch_size, nz)).to(device)
            fake_ood = generator_ood(z).detach()
            if every_ten_ood:
                for epoch in range(num_epochs):
                    adjust_learning_rate(optimizer, epoch)
                    running_loss = 0.0
                    if epoch/2==0:
                        for inputs, labels in dataloader:
                            optimizer.zero_grad()
                            outputs_ood = model(fake_ood.to(device))             
                            labels_ood = outputs_ood[0].argmax(dim=1)                 
                            outputs = model(inputs.to(device))                 
                            loss = custom_loss(outputs, labels.to(device),outputs_ood,labels_ood,exit_weights)                
                            loss.backward()
                            optimizer.step()     
                            running_loss += loss.item() 
                    else:    
                        for inputs, labels in dataloader:
                            optimizer.zero_grad()   
                            outputs = model(inputs.to(device))                 
                            loss = withoutood_custom_loss(outputs, labels.to(device),exit_weights)                
                            loss.backward()
                            optimizer.step()     
                            running_loss += loss.item()      
                    epoch_loss = running_loss / len(dataloader.dataset)
                    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")
                    model_save_path = f"EE_{epoch}.pth"
                    torch.save(model.state_dict(), model_save_path)
                    model.train_EE = False
                    model.inference_EE = True
                    model.recount_exits()
                    test_loss, test_acc, _ = model_utils.test_step(model, testloader, nn.CrossEntropyLoss(), device,
                                                            epoch=epoch,min_max_values=True)
                    model.train()
                    model.train_EE = True
                    model.inference_EE = False
                    model.recount_exits()
                    print(f"test_acc : {test_acc}")
                print("Training complete!")       



def load_EarlyExit(model):
    path = 'EE_6.pth'
    model.load_state_dict(torch.load(path))
    model.eval()
    model.train_EE = False
    model.inference_EE = True
    print(f"Loaded model from {path}")


def freeze_model_except_early_exits(model): 
    for param in model.base_model.parameters():
        param.requires_grad = False
    for base_model_layer in model.base_model_layers:
        for param in base_model_layer.parameters():
            param.requires_grad = False
    for early_exit in model.early_exit_blocks:
        for param in early_exit.parameters():
            param.requires_grad = True


def print_trainable_layers(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Trainable: {param.requires_grad}")


def train_EarlyExit(model,train_loader,testloader):
    freeze_model_except_early_exits(model)
    print_trainable_layers(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    num_epochs = 10 
    scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    print(model)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    train_model(model, train_loader,testloader, optimizer, scheduler,num_epochs=num_epochs)
    end.record()
    torch.cuda.synchronize()
    print("TIME: ",start.elapsed_time(end)/1000)



def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.01
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                        choices=knockoff_utils.BBOX_CHOICES)
    parser.add_argument('defense_args', metavar='STR', type=str, help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--quantize',type=int,help="Whether using quantized defense",default=0)
    parser.add_argument('--quantize_args',type=str,help='Quantization arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']
    BB = Blackbox
    defense_kwargs = knockoff_utils.parse_defense_kwargs(params['defense_args'])
    defense_kwargs['log_prefix'] = 'test'
    print('=> Initializing BBox with defense {} and arguments: {}'.format(defense_type, defense_kwargs))
    blackbox = BB.from_modeldir(blackbox_dir, device, **defense_kwargs)
    if params['quantize']:
        quantize_kwargs = knockoff_utils.parse_defense_kwargs(params['quantize_args'])
        if quantize_kwargs['epsilon'] > 0.0:
            print('=> Initializing Quantizer with arguments: {}'.format(quantize_kwargs))
            blackbox = incremental_kmeans(blackbox,**quantize_kwargs)

    for k, v in defense_kwargs.items():
        params[k] = v

    # ----------- Set up queryset
    with open(osp.join(blackbox_dir, 'params.json'), 'r') as rf:
        bbox_params = json.load(rf)
    testset_name = bbox_params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if testset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = datasets.__dict__[testset_name](train=False, transform=transform)
    print('=> Evaluating on {} ({} samples)'.format(testset_name, len(testset)))
  # ----------- Evaluate
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    epoch = bbox_params['epochs']
    testloader = DataLoader(testset, num_workers=nworkers, shuffle=False, batch_size=batch_size)
    
    train_EE = False
    if train_EE:
        print("==> train EE")
        trainset_name = bbox_params['dataset']
        valid_datasets = datasets.__dict__.keys()
        if trainset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        modelfamily = datasets.dataset_to_modelfamily[trainset_name]
        transform = datasets.modelfamily_to_transforms[modelfamily]['train']
        trainset = datasets.__dict__[trainset_name](train=True, transform=transform)
        print('=> Training on {} ({} samples)'.format(trainset_name, len(trainset)))
        num_classes = len(trainset.classes)
        batch_size = params['batch_size']
        nworkers = params['nworkers']
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=nworkers
        )
        train_EarlyExit(blackbox.model,train_loader,testloader)
    else:
        load_EarlyExit(blackbox.model)


 
    test_loss, test_acc, _ = model_utils.test_step(blackbox, testloader, nn.CrossEntropyLoss(), device,
                                                epoch=epoch,min_max_values=True)
    print(f"Test_acc: {test_acc}")

if __name__ == '__main__':
    main()
