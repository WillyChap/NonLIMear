"""
Author: wchapman@ucar.edu
Will Chapman
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
# from eval_gcn import evaluate_preds
from utilities.data_wrangling import load_cnn_data, to_dataloaders
from utilities.utils import get_euclidean_adj
from sklearn.metrics import mean_squared_error


def get_dataloaders(params, net_params):
    # Load data
    load_data_kwargs = {
        'window': params['window'], 'lead_months': params['horizon'], 'lon_min': params['lon_min'],
        'lon_max': params['lon_max'], 'lat_min': params['lat_min'], 'lat_max': params['lat_max'],
        'data_dir': params['data_dir'], 'use_heat_content': params['use_heat_content'],
        'add_index_node': net_params['index_node']
    }
    cmip5, SODA, GODAS, cords, cnn_mask = load_cnn_data(**load_data_kwargs, return_new_coordinates=True, return_mask=True)
    net_params['num_nodes'] = SODA[0].shape[3]
    if 'grid_edges' in params and params['grid_edges']:
        print('Using grid edges, i.e. based on spatial proximity!!!! ')
        adj = get_euclidean_adj(GODAS[0], radius_lat=5, radius_lon=5, self_loop=True)
        static_feats = None
    else:
        adj = None
        # Static features for adj learning
        static_feats = get_static_feats(params, net_params, cords, SODA[0])
        assert SODA[0].shape[3] == cmip5[0].shape[3] and SODA[0].shape[3] == GODAS[0].shape[3]

    trainloader, valloader, testloader = \
        to_dataloaders(cmip5, SODA, GODAS, batch_size=params['batch_size'],
                       valid_split=params['validation_frac'], concat_cmip5_and_soda=True,
                       shuffle_training=params['shuffle'], validation=params['validation_set'])
    del cmip5, SODA, GODAS
    return (adj, static_feats, cords), (trainloader, valloader, testloader)

def get_static_feats(params, net_params, coordinates, trainset):
    max_lat = max(params['lat_max'], params['lat_min'])
    static_feats = np.array([
        [lat / max_lat, (lon - 180) / 360] for lat, lon in coordinates
    ])  # (#nodes, 2) = (#nodes (lat, lon))
    trainset_sst = trainset[:, 0, 0, :].squeeze()  # take SSTs of the first timestep before prediction
    static_feats = np.concatenate((static_feats, trainset_sst.T), axis=1)  # (#nodes, 2 + len(trainset))
    if trainset.shape[1] == 2:
        trainset_hc = trainset[:, 1, 0, :].squeeze()  # take SSTs of the first timestep before prediction
        static_feats = np.concatenate((static_feats, trainset_hc.T), axis=1)  # (#nodes, 2 + 2*len(trainset))
    return static_feats


def train_epoch(dataloader, model, criterion, optims, device, epoch, nth_step=100):
    if not isinstance(optims, list):
        optims = [optims]
    model.train()
    total_loss = 0
    for iter, (X, Y) in enumerate(dataloader, 1):
        X, Y = X.to(device), Y.to(device)
        for optim in optims:
            optim.zero_grad()
        X = X.reshape((X.shape[0], -1, X.shape[3])).transpose(1, 2)  # shape = (batch_size x #features x #nodes)

        preds = model(X)
        loss = criterion(preds, Y)
        loss.backward()

        for optim in optims:
            optim.step()
        total_loss += loss.item()
    num_edges = torch.count_nonzero(model.adj.detach()).item()
    return total_loss / iter, num_edges


def train_epoch_LIM(dataloader, model, criterion, optims, device, epoch, nth_step=100):
    if not isinstance(optims, list):
        optims = [optims]
    model.train()
    total_loss = 0
    for iter, (X, Y) in enumerate(dataloader, 1):
        X, Y = X.to(device), Y.to(device)
        Y=Y.unsqueeze(dim=1)
        for optim in optims:
            optim.zero_grad()
#         X = X.reshape((X.shape[0], -1, X.shape[3])).transpose(1, 2)  # shape = (batch_size x #features x #nodes)
        preds = model(X)
        loss = criterion(preds, Y)
        loss.backward()

        for optim in optims:
            optim.step()
        total_loss += loss.item()
    num_edges = torch.count_nonzero(model.adj.detach()).item()
    return total_loss / iter, num_edges


def evaluate(dataloader, model, device, return_preds=False):
    model.eval()
    total_loss_l2 = 0
    total_loss_l1 = 0
    preds = None
    Ytrue = None
    for i, (X, Y) in enumerate(dataloader, 1):
        assert len(X.size()) == 4, "Expected X to have shape (batch_size, #channels, window, #nodes)"
        X, Y = X.to(device), Y.to(device)
        X = X.reshape((X.shape[0], -1, X.shape[3])).transpose(1, 2)
        with torch.no_grad():
            output = model(X)
        if preds is None:
            preds = output
            Ytrue = Y
        else:
            preds = torch.cat((preds, output))
            Ytrue = torch.cat((Ytrue, Y))
        total_loss_l2 += F.mse_loss(output, Y).item()
        total_loss_l1 += F.l1_loss(output, Y).item()

    preds = preds.data.cpu().numpy()
    Ytest = Ytrue.data.cpu().numpy()
    oni_stats = evaluate_preds(Ytest, preds, return_dict=True)
    oni_stats['mae'] = total_loss_l1
    if return_preds:
        return total_loss_l2 / i, oni_stats, Ytest, preds
    else:
        return total_loss_l2 / i, oni_stats


def evaluate_LIM(dataloader, model, device, return_preds=False):
    model.eval()
    total_loss_l2 = 0
    total_loss_l1 = 0
    preds = None
    Ytrue = None
    for i, (X, Y) in enumerate(dataloader, 1):
        assert len(X.size()) == 4, "Expected X to have shape (batch_size, #channels, window, #nodes)"
        X, Y = X.to(device), Y.to(device)
        Y=Y.unsqueeze(dim=1)
#         X = X.reshape((X.shape[0], -1, X.shape[3])).transpose(1, 2)
        with torch.no_grad():
            output = model(X)
        if preds is None:
            preds = output
            Ytrue = Y
        else:
            preds = torch.cat((preds, output))
            Ytrue = torch.cat((Ytrue, Y))
        total_loss_l2 += F.mse_loss(output, Y).item()
        total_loss_l1 += F.l1_loss(output, Y).item()

    preds = preds.data.cpu().numpy().squeeze()
    Ytest = Ytrue.data.cpu().numpy().squeeze()
    oni_stats = evaluate_preds(Ytest, preds, return_dict=True)
    oni_stats['mae'] = total_loss_l1
    if return_preds:
        return total_loss_l2 / i, oni_stats, Ytest, preds
    else:
        return total_loss_l2 / i, oni_stats


def evaluate_preds(Ytrue, preds, **kwargs):
#     print('true shape:',Ytrue.shape)
#     print('preds shape:',preds.shape)

    Ytrue=Ytrue.squeeze()
    preds=preds.squeeze()

    oni_corr = np.corrcoef(Ytrue, preds)[0, 1]
    try:
        rmse_val = rmse(Ytrue, preds)
    except ValueError as e:
        print(e)
        rmse_val = -1
    # r, p = pearsonr(Ytrue, preds)   # same as using np.corrcoef(y, yhat)[0, 1]
    oni_stats = {"corrcoef": oni_corr, "rmse": rmse_val}  # , "Pearson_r": r, "Pearson_p": p}

    try:
        # All season correlation skill = Mean of the corrcoefs for each target season
        # (whereas the corrcoef above computes it directly on the whole timeseries).
        predsTS = preds.reshape((-1, 12))
        YtestTT = Ytrue.reshape((-1, 12))
        all_season_cc = 0
        for target_mon in range(12):
            all_season_cc += np.corrcoef(predsTS[:, target_mon], YtestTT[:, target_mon])[0, 1]
        all_season_cc /= 12
        oni_stats['all_season_cc'] = all_season_cc
    except ValueError:
        pass
    return oni_stats

def rmse(y, preds):
    """
    :return:  The root-mean-squarred error (RMSE)  value
    """
    return np.sqrt(mean_squared_error(y, preds))



def predict_ts(dataloader, model, device, return_preds=False):
    model.eval()
    preds = None
    Ytrue = None
    for i, (X, Y) in enumerate(dataloader, 1):
        assert len(X.size()) == 4, "Expected X to have shape (batch_size, #channels, window, #nodes)"
        X, Y = X.to(device), Y.to(device)
        X = X.reshape((X.shape[0], -1, X.shape[3])).transpose(1, 2)
        with torch.no_grad():
            output = model(X)
        if preds is None:
            preds = output
            Ytrue = Y
        else:
            preds = torch.cat((preds, output))
            Ytrue = torch.cat((Ytrue, Y))
    return preds,Ytrue

def update_tqdm(tq, train_loss, val_stats=None, test_stats=None, **kwargs):
    def get_stat_dict(dictio, prefix, all=False):
        dict_two = dict()
        set_if_exists(dictio, dict_two, 'rmse', prefix)
        set_if_exists(dictio, dict_two, 'corrcoef', prefix)
        set_if_exists(dictio, dict_two, 'all_season_cc', prefix)

        if all:
            set_if_exists(dictio, dict_two, 'mae', prefix)
        return dict_two

    if val_stats is None:
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **test_print, **kwargs)
    else:
        val_print = get_stat_dict(val_stats, 'val', all=True)
        if test_stats is None:
            tq.set_postfix(train_loss=train_loss, **val_print, **kwargs)
        else:
            test_print = get_stat_dict(test_stats, 'test')
            tq.set_postfix(train_loss=train_loss, **val_print, **test_print, **kwargs)


def set_if_exists(dictio_from, dictio_to, key, prefix):
    if key in dictio_from:
        dictio_to[f'{prefix}_{key}'.lstrip('_')] = dictio_from[key]
