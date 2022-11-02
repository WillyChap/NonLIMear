
import torch
import torch.nn as nn
from deeplim.readout_MLP import ONI_MLP,LIM_MLP,LIM_MLP_GaussLL
from deeplim.GCN.graph_conv_layer import GraphConvolution
from deeplim.structure_learner import EdgeStructureLearner,LIMG_PC,LIMG
from utilities.utils import get_activation_function
import random
import numpy as np

class nlim_old(nn.Module):
    def __init__(self, net_params,params, static_feat=None, adj=None, device="cuda", outsize=1, verbose=True):
        super().__init__()
        self.L = net_params['L']
        assert self.L > 1
        self.act = net_params['activation']
        self.out_dim = self.mlp_input_dim = net_params['out_dim']
        self.batch_norm = net_params['batch_norm']
        self.graph_pooling = net_params['readout'].lower()
        self.jumping_knowledge = net_params['jumping_knowledge']
        self.tau = params['horizon']
        self.device=device
        self.numpreds = static_feat.shape[0]
        self.num_eofs = net_params["num_eofs"]
        self.add_bias = net_params["LIM_Bias"]

        self.Gbias = nn.Parameter(torch.rand(self.num_eofs,self.num_eofs),requires_grad=True)
        dropout = net_params['dropout']
        hid_dim = net_params['hidden_dim']
        num_nodes = net_params['num_nodes']
        activation = get_activation_function(self.act, functional=True, num=1, device=device)
        conv_kwargs = {'activation': activation, 'batch_norm': self.batch_norm,
                       'residual': net_params['residual'], 'dropout': dropout}


        layers = [GraphConvolution(net_params['in_dim'], hid_dim, **conv_kwargs)]
        layers += [GraphConvolution(hid_dim, hid_dim, **conv_kwargs) for _ in range(self.L - 2)]
        layers.append(GraphConvolution(hid_dim, self.out_dim, **conv_kwargs))

        self.layers = nn.ModuleList(layers)

        if self.jumping_knowledge:
            self.mlp_input_dim = self.mlp_input_dim + hid_dim * (self.L - 1)
        if self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            self.mlp_input_dim = self.mlp_input_dim * 2
        self.MLP_layer = LIM_MLP(static_feat.shape[0], outsize, act_func=self.act, batch_norm=net_params['mlp_batch_norm'],
                                    dropout=dropout, device=device,L=4)
        if adj is None:
            self.adj, self.learn_adj = None, True
            max_num_edges = int(net_params['avg_edges_per_node'] * num_nodes)
            self.LIM_learner = LIMG_PC(50,50,50,device=device, static_feat=static_feat,num_eofs=self.num_eofs)
        else:
            print('Using a static connectivity structure !!!')
            self.adj, self.learn_adj = torch.from_numpy(adj).float().to(device), False

        if verbose:
            print([x for x in self.layers])

    def get_adj(self):
        if self.learn_adj:
            return self.LIM_learner.forward()
        return self.adj


    def forward(self, input, readout=True):
        if self.learn_adj:
            # Generate an adjacency matrix/connectivity structure for the graph convolutional forward pass
            self.adj = self.LIM_learner.forward()
            self.Gpow_pure = torch.linalg.matrix_power(self.adj, self.tau)
            self.Gpow = torch.linalg.matrix_power(self.adj, self.tau)

            if self.add_bias:
                self.Gpow = self.Gpow + self.Gbias
            else:
                self.Gpow = self.Gpow

            #project input space onto pcs.
            Forecast_inits =torch.cat((input[:,0,-1,:].to(torch.float64).to(self.device),input[:,1,-1,:].to(torch.float64).to(self.device)),1)
            proj_pcs = Forecast_inits@self.LIM_learner.eofs.T.to(torch.float64).to(self.device) #project the dataset onto the EOFS using a matrix mult.
            for_pcs_space = torch.mm(self.Gpow,proj_pcs.T)
            self.xfor = for_pcs_space.T@self.LIM_learner.eofs.to(torch.float64).to(self.device)
            out = self.MLP_layer(self.xfor.float())

        return  out
    
    
    

    
    
class nlim(nn.Module):
    def __init__(self, net_params,params, static_feat=None, adj=None, device="cuda", outsize=1, verbose=True):
        super().__init__()
        self.L = net_params['L']
        assert self.L > 1
        self.act = net_params['activation']
        self.out_dim = self.mlp_input_dim = net_params['out_dim']
        outsize = self.out_dim
        self.batch_norm = net_params['batch_norm']
        self.graph_pooling = net_params['readout'].lower()
        self.jumping_knowledge = net_params['jumping_knowledge']
        self.tau = params['horizon']
        self.device=device
        self.numpreds = static_feat.shape[0]
        self.num_eofs = net_params["num_eofs"]
        self.add_bias = net_params["LIM_Bias"]
        self.loss = params['loss'].lower().strip()
        
        
        self.Gbias = nn.Parameter(torch.rand(self.num_eofs,self.num_eofs),requires_grad=True)
        dropout = net_params['dropout']
        hid_dim = net_params['hidden_dim']
        num_nodes = net_params['num_nodes']
        activation = get_activation_function(self.act, functional=True, num=1, device=device)
        conv_kwargs = {'activation': activation, 'batch_norm': self.batch_norm,
                       'residual': net_params['residual'], 'dropout': dropout}
        
        
        layers = [GraphConvolution(net_params['in_dim'], hid_dim, **conv_kwargs)]
        layers += [GraphConvolution(hid_dim, hid_dim, **conv_kwargs) for _ in range(self.L - 2)]
        layers.append(GraphConvolution(hid_dim, self.out_dim, **conv_kwargs))
        
        self.layers = nn.ModuleList(layers)
        
        if self.jumping_knowledge:
            self.mlp_input_dim = self.mlp_input_dim + hid_dim * (self.L - 1)
        if self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            self.mlp_input_dim = self.mlp_input_dim * 2
        
        if self.loss in ['gauss','laplace','cauchy','crps']:
            self.MLP_layer = LIM_MLP_GaussLL(static_feat.shape[0], outsize, act_func=self.act, batch_norm=net_params['mlp_batch_norm'],
                                    dropout=dropout, device=device,L=net_params['L'])
        else:
            self.MLP_layer = LIM_MLP(static_feat.shape[0], outsize, act_func=self.act, batch_norm=net_params['mlp_batch_norm'],
                                    dropout=dropout, device=device,L=net_params['L'])
        if adj is None:
            self.adj, self.learn_adj = None, True
            max_num_edges = int(net_params['avg_edges_per_node'] * num_nodes)
            self.LIM_learner = LIMG_PC(50,50,50,device=device, static_feat=static_feat,num_eofs=self.num_eofs)
        else:
            print('Using a static connectivity structure !!!')
            self.adj, self.learn_adj = torch.from_numpy(adj).float().to(device), False

        if verbose:
            print([x for x in self.layers])

    def get_adj(self):
        if self.learn_adj:
            return self.LIM_learner.forward()
        return self.adj


    def forward(self, input, readout=True):
        if self.learn_adj:
            # Generate an adjacency matrix/connectivity structure for the graph convolutional forward pass
            self.adj = self.LIM_learner.forward()
            self.Gpow_pure = torch.linalg.matrix_power(self.adj, self.tau)
            self.Gpow = torch.linalg.matrix_power(self.adj, self.tau)
            
            if self.add_bias:
                self.Gpow = self.Gpow + self.Gbias
            else:
                self.Gpow = self.Gpow
            
            #project input space onto pcs. 
            Forecast_inits =torch.cat((input[:,0,-1,:].to(torch.float64).to(self.device),input[:,1,-1,:].to(torch.float64).to(self.device)),1)

            proj_pcs = Forecast_inits@self.LIM_learner.eofs.T.to(torch.float64).to(self.device) #project the dataset onto the EOFS using a matrix mult.
            for_pcs_space = torch.mm(self.Gpow,proj_pcs.T)
            self.xfor = for_pcs_space.T@self.LIM_learner.eofs.to(torch.float64).to(self.device)
            
            if self.loss in ['gauss','laplace','cauchy','crps']:
                out,var = self.MLP_layer(self.xfor.float())
                return  out,var
            else:
                out = self.MLP_layer(self.xfor.float())
                return  out

