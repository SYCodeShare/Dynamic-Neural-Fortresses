#!/usr/bin/python

import os.path as osp
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from defenses.utils.type_checks import TypeCheck
import defenses.models.zoo as zoo
from defenses import datasets
from timm.models import create_model


def load_EarlyExit(model):
    path = 'EE.pth'
    model.load_state_dict(torch.load(path))
    model.eval()
    model.train_EE = False
    model.inference_EE = True
    print(f"Loaded model from {path}")


class EarlyExitBlock(nn.Module):
    def __init__(self, planes, num_classes):
        super(EarlyExitBlock, self).__init__()
        
        self.ee_avgpool = nn.AdaptiveAvgPool1d(1)
        self.ee_norm=nn.LayerNorm(planes)

        self.fc1 = nn.Linear(planes, num_classes*10)
        self.fc2 = nn.Linear(num_classes*10, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.ee_norm(x)  
        x = self.ee_avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        probs = self.softmax(x)
        return x, probs

class Net_EarlyExit(nn.Module):
    def __init__(self, model=None,num_classes=200, exit_thresholds=[0.9, 0.92, 0.94, 0.96],planes=[384,768,1536,1536]):
        super(Net_EarlyExit, self).__init__()
        self.base_model = model
        self.num_classes = num_classes
        self.early_exit_blocks = nn.ModuleList([
            EarlyExitBlock(planes[0], num_classes),
            EarlyExitBlock(planes[1], num_classes),
            EarlyExitBlock(planes[2], num_classes),
            EarlyExitBlock(planes[3], num_classes)
        ])
        self.exit_thresholds = exit_thresholds
        self.use_early_exit = True
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.exits_count = [0,0,0,0,0]
        self.times = [0.0,0.0,0.0,0.0,0.0]
        self.train_EE = False
        self.inference_EE = False
        print("exit_thresholds ",self.exit_thresholds)
    
    def forward(self, x):
        
        if self.inference_EE:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()     
            exit_indices = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
            outputs = torch.zeros((x.size(0), self.num_classes), device=x.device)
            not_exited = torch.ones(x.size(0), dtype=torch.bool, device=x.device)      
            early_exit_outputs = []
            x = self.base_model.patch_embed(x)
            x = self.base_model.pos_drop(x)
            for i, layer in enumerate(self.base_model.layers):
                x = layer(x)
                if i < len(self.early_exit_blocks):
                    early_exit_output, probs = self.early_exit_blocks[i](x)
                    end.record()            
                    torch.cuda.synchronize()
                    self.times[i] = self.times[i] + (start.elapsed_time(end)/1000)              
                    early_exit_outputs.append(early_exit_output)  
                    not_exited = (exit_indices == -1)  
                    if not_exited.any():
                        max_probs, _ = torch.max(probs, dim=1)
                        exit_now = max_probs > self.exit_thresholds[i]  
                        exit_now = exit_now & not_exited 
                        exit_indices[exit_now] = i  
            x = self.base_model.norm(x)  
            x = self.avgpool(x.transpose(1, 2))  
            x = torch.flatten(x, 1)
            x = self.base_model.head(x)  
            end.record()
            torch.cuda.synchronize()
            self.times[len(self.early_exit_blocks)] = self.times[len(self.early_exit_blocks)] + (start.elapsed_time(end)/1000)
            early_exit_outputs.append(x) 
            exit_indices[exit_indices == -1] = len(self.early_exit_blocks)
            output = early_exit_outputs
            exit_th = exit_indices
            num_exits = len(output)
            batch_size = len(output[0])
            sample_indices = torch.arange(batch_size)
            stacked_output = torch.stack(output, dim=0)
            selected_outputs = stacked_output[exit_th, sample_indices]
            output=selected_outputs
            counts = torch.bincount(exit_th, minlength=5)
            for j in range(counts.shape[0]):
                self.exits_count[j] = self.exits_count[j] + counts[j] 
            return output 
        if self.train_EE: 
            early_exit_outputs = []
            x = self.base_model.patch_embed(x)
            x = self.base_model.pos_drop(x)
            for i, layer in enumerate(self.base_model.layers):
                x = layer(x)
                if i < len(self.early_exit_blocks):
                    early_exit_output, probs = self.early_exit_blocks[i](x)
                    early_exit_outputs.append(early_exit_output)  
            x = self.base_model.norm(x)  
            x = self.avgpool(x.transpose(1, 2))  
            x = torch.flatten(x, 1)
            x = self.base_model.head(x)  
            early_exit_outputs.append(x)    
            return early_exit_outputs
    def enable_early_exit(self):
        self.use_early_exit = True
    def disable_early_exit(self):
        self.use_early_exit = False
    def recount_exits(self):
        self.exits_count = [0,0,0,0,0]

def model_wrapper(model):
    model = Net_EarlyExit(model)
    return model

def load_model(model_name, evaluate):
    if model_name == 'cait':
        model = create_model('cait_s36_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 128
    elif model_name == 'deit':
        model = create_model('deit_base_distilled_patch16_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 64
    elif model_name == 'swin':
        model = create_model('swin_large_patch4_window12_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 32
    elif model_name == 'vit':
        model = create_model('vit_large_patch16_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 64
    else:
        logger.error('Invalid model name, please use either cait, deit, swin, or vit')
        sys.exit(1)

    for param in model.parameters():
        param.requires_grad = False
    model.reset_classifier(num_classes=200)

    if evaluate:
        if evaluate.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(evaluate)
        else:
            checkpoint = torch.load(evaluate)

        print("LOADING MODEL")
        print(f"Checkpoint : {evaluate}")
        model.load_state_dict(checkpoint['state_dict'])

    return model, batch_size

class Blackbox(object):
    def __init__(self, model, device=None, output_type='probs', topk=None, rounding=None, dataset_name=None,
                 modelfamily=None, model_arch=None, num_classes=None, model_dir=None, out_path=None, log_prefix=''):
        print('=> Blackbox ({})'.format([model.__class__, device, output_type, topk, rounding]))
        self.device = torch.device('cuda') if device is None else device
        self.output_type = output_type
        self.topk = topk
        self.rounding = rounding
        self.require_xinfo = False
        self.top1_preserve = True 
        self.dataset_name = dataset_name
        self.modelfamily = modelfamily
        self.model_arch = model_arch
        self.num_classes = num_classes
        self.model = model.to(device)
        self.output_type = output_type
        self.model.eval()
        self.model_dir = model_dir
        self.call_count = 0

        if self.topk is not None or self.rounding is not None:
            print('Blackbox with defense: topk={}\trounding={}'.format(self.topk, self.rounding))

        self.out_path = out_path
        self.log_prefix = log_prefix
        self.queries = [] 
        if self.out_path is not None:
            self.log_path = osp.join(self.out_path, 'distance{}.log.tsv'.format(self.log_prefix))
            if not osp.exists(self.log_path):
                with open(self.log_path, 'w') as wf:
                    columns = ['call_count', 'l1_max', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                    wf.write('\t'.join(columns) + '\n')
        else:
            self.log_path = None

    @classmethod
    def from_modeldir(cls, model_dir, device=None, output_type='probs', **kwargs):
        device = torch.device('cuda') if device is None else device
        params_path = osp.join(model_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        if 'queryset' in params:
            dataset_name = params['queryset']
        elif 'testdataset' in params:
            dataset_name = params['testdataset']
        elif 'dataset' in params:
            dataset_name = params['dataset']
        else:
            raise ValueError('Unable to determine model family')
        modelfamily = datasets.dataset_to_modelfamily[dataset_name]
        model = zoo.get_net(model_arch, modelfamily, num_classes=num_classes)
        model = model.to(device)
        checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))
        blackbox = cls(model=model, device=device, output_type=output_type, dataset_name=dataset_name,
                       modelfamily=modelfamily, model_arch=model_arch, num_classes=num_classes, model_dir=model_dir,
                       **kwargs)
        return blackbox

    @classmethod
    def from_modeldirVIT(cls, model_dir, VITmodel, evaluate, device=None, output_type='probs', **kwargs):
        print(cls)
        device = torch.device('cuda') if device is None else device
        params_path = osp.join(model_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        if 'queryset' in params:
            dataset_name = params['queryset']
        elif 'testdataset' in params:
            dataset_name = params['testdataset']
        elif 'dataset' in params:
            dataset_name = params['dataset']
        else:
            raise ValueError('Unable to determine model family')
        modelfamily = datasets.dataset_to_modelfamily[dataset_name]
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, batch_size = load_model(VITmodel, evaluate)
        model = model.to(device)
        true_batch_size = 128
        update_freq = true_batch_size // batch_size
        img_size = 384
        checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))
        model = model_wrapper(model)
        train_EE = False
        if not train_EE:
            load_EarlyExit(model)
        blackbox = cls(model=model, device=device, output_type=output_type, dataset_name=dataset_name,
                       modelfamily=modelfamily, model_arch=model_arch, num_classes=num_classes, model_dir=model_dir,
                       **kwargs)
        return blackbox

    @staticmethod
    def truncate_output(y_t_probs, topk=None, rounding=None):
        if topk is not None:
            topk_vals, indices = torch.topk(y_t_probs, topk)
            newy = torch.zeros_like(y_t_probs)
            if rounding == 0:
                newy = newy.scatter(1, indices, torch.ones_like(topk_vals))
            else:
                newy = newy.scatter(1, indices, topk_vals)
            y_t_probs = newy

        # Rounding of decimals
        if rounding is not None and rounding > 0:
            y_t_probs = torch.Tensor(np.round(y_t_probs.cpu().numpy(), decimals=rounding)).to(y_t_probs)

        return y_t_probs

    @staticmethod
    def make_one_hot(labels, K):
        return torch.zeros(labels.shape[0], K, device=labels.device).scatter(1, labels.unsqueeze(1), 1)
    def calc_query_distances(self,queries):
        l1s, l2s, kls = [], [], []
        for i in range(len(queries)):
            y_v, y_prime, *_ = queries[i]
            y_v, y_prime = torch.tensor(y_v), torch.tensor(y_prime)
            l1s.append((y_v - y_prime).norm(p=1,dim=1))
            l2s.append((y_v - y_prime).norm(p=2,dim=1))
            kls.append(torch.sum(F.kl_div((y_v+1e-6).log(), y_prime, reduction='none'),dim=1))
        l1s = torch.cat(l1s).cpu().numpy()
        l2s = torch.cat(l2s).cpu().numpy()
        kls = torch.cat(kls).cpu().numpy()
        l1_max, l1_mean, l1_std = np.amax(l1s), np.mean(l1s), np.std(l1s)
        l2_mean, l2_std = np.mean(l2s), np.std(l2s)
        kl_mean, kl_std = np.mean(kls), np.std(kls)

        return l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std

    def calc_distance(self,y, ytilde, ydist, device=torch.device('cuda')):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        # assert len(y.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']
        ytilde = ytilde.to(device)
        if ydist == 'l1':
            return (ytilde - y).norm(p=1)
        elif ydist == 'l2':
            return (ytilde - y).norm(p=2)
        elif ydist == 'kl':
            return F.kl_div((y+1e-6).log(), ytilde, reduction='batchmean')
        else:
            raise ValueError('Unrecognized ydist contraint')

    def is_in_dist_ball(self,y, ytilde, ydist, epsilon, tolerance=1e-4):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        # assert len(y.shape) == 1, 'Does not support batching'
        return (self.calc_distance(y, ytilde, ydist) - epsilon).clamp(min=0.) <= tolerance

    def is_in_simplex(self,ytilde, tolerance=1e-4):
        # assert len(ytilde.shape) == 1, 'Does not support batching'
        return torch.sum(torch.abs(ytilde.clamp(min=0., max=1.).sum(dim=1) - 1.) <= tolerance).item()==len(ytilde)

    def __call__(self, query_input, stat = True, return_origin = False):
        TypeCheck.multiple_image_blackbox_input_tensor(query_input)
        with torch.no_grad():
            query_input = query_input.to(self.device)
            query_output = self.model(query_input)
            self.call_count += query_input.shape[0]
            y_v = F.softmax(query_output, dim=1)

        y_prime = self.truncate_output(y_v, topk=self.topk, rounding=self.rounding)

        if stat:
            
            self.queries.append((y_v.cpu().detach().numpy(), y_prime.cpu().detach().numpy()))

            if self.call_count % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)

                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.calc_query_distances(self.queries)

                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        if return_origin:
            return y_prime,y_v
        else:
            return y_prime

    def eval(self):
        self.model.eval()

    def get_yprime(self,y,x_info=None):
        return self.truncate_output(y, topk=self.topk, rounding=self.rounding)
    
    def get_xinfo(self,x):
        return None
