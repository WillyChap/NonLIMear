"""
Author: wchapman@ucar.edu
Will Chapman
"""

import argparse
import json
import os
import time
import copy
import warnings
import numpy as np
import sys


from torch.utils.tensorboard import SummaryWriter
#from torch.cuda import is_available,device_count,current_device,device,get_device_name
import torch
from tqdm import tqdm

# Training settings
# explore this:
# from eval_gcn import ensemble_performance
# !!!

from deeplim.training import evaluate, train_epoch, train_epoch_LIM, get_dataloaders, evaluate_LIM
from deeplim.GCN.GCN_model import GCN
from deeplim.DLIM.deeplim_model import nlim
from utilities.utils import set_gpu, set_seed
from utilities.model_logging import update_tqdm, save_model
from utilities.optimization import get_optimizer, get_loss, CRPSloss

print('torch-version: ',torch.__version__)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # or use: "once"
    parser = argparse.ArgumentParser(description='PyTorch ENSO Time series forecasting')
    parser.add_argument("--gpu_id", default=-1, type=int)
    parser.add_argument("--horizon", default=3, type=int, help='how many months to forecast')
    parser.add_argument("--out", default='out', type=str)
    parser.add_argument("--optim", default='adam', type=str, help='supply an optimizer')
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--lr", type=float,help='algorithm learning rate')
    parser.add_argument("--epochs", default=15, type=int,help='number of epochs to train')  #
    parser.add_argument("--data_dir", default='Data/', type=str)
    parser.add_argument("--grid_edges", default='false', type=str)
    parser.add_argument("--seed", default=42, type=int,help='provide a seed to train with')
    parser.add_argument("--numeof", type=int, help='number of eofs')
    args = parser.parse_args()

    print(str(args))

    if args.gpu_id >= 0:
        device = "cuda"
        set_gpu(args.gpu_id)
        print('device available :', torch.cuda.is_available())
        print('device count: ', torch.cuda.device_count())
        print('current device: ',torch.cuda.current_device())
        print('device name: ',torch.cuda.get_device_name())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    print('using device: ', device)

    base_dir = f'{args.out}/{args.horizon}lead/'
    adj = None
    config_files = ['DLIM_config_bias_expand.json']
    ID = str(time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'))

    for i, config_file in enumerate(config_files):
        with open(f'configs/DLIM_{config_file}.json', 'r') as f:
            config = json.load(f)

        params, net_params = config['params'], config['net_params']
        params['horizon'] = args.horizon
        params['data_dir'] = args.data_dir + '/'
        params['optimizer'] = args.optim
        params['weight_decay'] = args.weight_decay or params['weight_decay']
        params['lr'] = args.lr or params['lr']
        params['epochs'] =args.epochs or params['epochs']
        params['grid_edges'] = True if args.grid_edges.lower() == 'true' else False
        params['seed'] = args.seed
        net_params['num_eofs'] = args.numeof or net_params['num_eofs']
        set_seed(params['seed'])

        ##create out directory if it doesn't exist
        outbase_dir = f'{args.out}/'+str(params['horizon'])+'lead/'
        isExist = os.path.exists(outbase_dir)
        if not isExist:
            os.makedirs(outbase_dir)
            print("The new directory is created!")

        print(outbase_dir)
        
        if args.horizon in [1,2]:
            params['lat_min']=-10
            params['lat_max']=10
        elif args.horizon in [3,4]:
            params['lat_min']=-15
            params['lat_max']=15
        elif args.horizon in [5,6]:
            params['lat_min']=-30
            params['lat_max']=30
        elif args.horizon in [9,12,23]:
            params['lat_min']=-55
            params['lat_max']=60
        
        # #set up data and G matrix
        (adj, static_feats, _), (trainloader,valloader,testloader) = get_dataloaders(params, net_params)
        static_feats = static_feats[:,2:]
        static_feats = np.concatenate([static_feats[:,:int(static_feats.shape[1]/2)],static_feats[:,int(static_feats.shape[1]/2):]])
        print('shape static feats', static_feats.shape)
        #
        # # model and optmizer
        if params['loss'] in ['gauss','laplace','cauchy','crps']:
            'You are running a probabilistic model, the output will be 2 nodes (mean,scale)'
            outsize=1
        else:
            'You are running a probabilistic model, the output will be 1 nodes (mean,scale)'
            outsize=1

        model = nlim(net_params, params,static_feat=static_feats, adj=adj,outsize=outsize,device=device)
        optimizer = get_optimizer(params['optimizer'], model, lr=params['lr'],weight_decay=params['weight_decay'], nesterov=params['nesterov'])
        criterion = get_loss(params['loss'])

        # Train model# Train model# Train model# Train model# Train model# Train model
        # Train model# Train model# Train model# Train model# Train model# Train model

        t_total = time.time()
        model = model.to(device)
        val_stats = None
        best_val_loss = cur_val = 1000
        #
        with tqdm(range(1, params['epochs'] + 1)) as t:
            for epoch in t:
                start_t = time.time()
                loss, num_edges = train_epoch_LIM(trainloader, model, criterion, optimizer, device, epoch)
                duration = time.time() - start_t
                if valloader is not None:
                    # Note that the default 'validation set' is included in the training set (=SODA),
                    # and is not used at all.
                      
                    _, val_stats = evaluate_LIM(valloader, model, device=device)
                    _, train_stats = evaluate_LIM(trainloader, model, device=device)
                    
                    if params['loss'] in ['gauss','laplace','cauchy','crps']:
                        _, val_stats_away,truer,preder,scales = evaluate_LIM_prob(valloader,model, device=device,return_preds=True)
                        val_stats['crps'] = CRPSloss(preder, truer, scales, eps=1e-06, reduction='mean')
                    
                    print('validation: ',val_stats)
                    print('train: ',train_stats)

                update_tqdm(t, loss, n_edges=num_edges, time=duration, val_stats=val_stats)
                if params['loss'] in ['gauss','laplace','cauchy','crps']:
                    #save the best model....
                    if epoch == 1:
                        epoch_best=1
                        best_accuracy = val_stats['crps']
                        best_model_lim = copy.deepcopy(model)
                    else:
                        print(epoch)
                        if best_accuracy < val_stats['crps']:
                            continue
                        else:
                            print('new best')
                            epoch_best = epoch
                            best_accuracy = val_stats['crps']
                            best_model_lim = copy.deepcopy(model)
                    
                else:
                    #save the best model....
                    if epoch == 1:
                        epoch_best=1
                        best_accuracy = val_stats['corrcoef']
                        best_model_lim = copy.deepcopy(model)
                    else:
                        print(epoch)
                        if best_accuracy > val_stats['corrcoef']:
                            continue
                        else:
                            print('new best')
                            epoch_best = epoch
                            best_accuracy = val_stats['corrcoef']
                            best_model_lim = copy.deepcopy(model)

        ##create out directory if it doesn't exist
        out_mod_dir = outbase_dir+'/'+config_file+'/'
        isExist = os.path.exists(out_mod_dir)
        if not isExist:
            os.makedirs(out_mod_dir)
            print("The new directory is created!")
        mod_name =out_mod_dir+'/LIM'+'_numeofs_' + f'{net_params["num_eofs"]:03}'+'_seed_' + f'{params["seed"]:03}' +'_optimizer_'+params["optimizer"]+'_loss_'+params["loss"]+'_epochs_'+f'{params["epochs"]:03}'+'.pth'
        print('this is your model:',best_model_lim)
        save_model(epoch_best,best_model_lim,optimizer,criterion,mod_name,str(args))
