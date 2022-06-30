"""
Author: wchapman@ucar.edu
Will Chapman
"""
import torch
import torch.nn as nn
import numpy as np


class EdgeStructureLearner(nn.Module):
    def __init__(self, num_nodes, max_num_edges, dim, static_feat, device='cuda', alpha1=0.1, alpha2=2.0,
                 self_loops=True):
        super().__init__()
        if static_feat is None:
            raise ValueError("Please give static node features (e.g. part of the timeseries)")
        self.num_nodes = num_nodes
        xd = static_feat.shape[1]
        self.lin1 = nn.Linear(xd, dim)
        self.lin2 = nn.Linear(xd, dim)

        self.static_feat = static_feat if isinstance(static_feat, torch.Tensor) else torch.from_numpy(static_feat)
        self.static_feat = self.static_feat.float().to(device)

        self.device = device
        self.dim = dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_edges = max_num_edges
        self.self_loops = self_loops
        self.diag = torch.eye(self.num_nodes).bool().to(device)

    def forward(self):
        nodevec1 = torch.tanh(self.alpha1 * self.lin1(self.static_feat))
        nodevec2 = torch.tanh(self.alpha1 * self.lin2(self.static_feat))

        adj = torch.sigmoid(self.alpha2 * nodevec1 @ nodevec2.T)
        adj = adj.flatten()
        mask = torch.zeros(self.num_nodes * self.num_nodes).to(self.device)
        _, strongest_idxs = torch.topk(adj, self.num_edges)  # Adj to get the strongest weight value indices
        mask[strongest_idxs] = 1
        adj = adj * mask
        adj = adj.reshape((self.num_nodes, self.num_nodes))
        if self.self_loops:
            adj[self.diag] = adj[self.diag].clamp(min=0.5)

        return adj

class LIMG_PC(nn.Module):
    def __init__(self, num_nodes, max_num_edges, dim, static_feat, device='cuda', alpha1=0.1, alpha2=2.0,
                 self_loops=True,num_eofs=100):
        super().__init__()
        if static_feat is None:
            raise ValueError("Please give static node features (e.g. part of the timeseries)")
        self.num_nodes = num_nodes

        A, Lh, E = torch.linalg.svd(torch.from_numpy(static_feat).T)
        A = A[:, :len(Lh)] #trim the time field (in case time > space)
        A = A[:, :len(Lh)] #trim the time field (in case time > space)
        PCs = A*Lh
        select_neofs=22
        self.select_neofs=num_eofs

        # normalize time series and scale in singular values to retain variance
        self.eofs = E[:self.select_neofs,:]
        n=torch.var(self.eofs,axis=1)
        self.eofs=self.eofs/torch.unsqueeze(n,dim=1)

        self.PCS=PCs[:,:self.select_neofs]
        self.PCS = self.PCS*torch.unsqueeze(n,dim=1).T*Lh[:self.select_neofs]
        self.eig_vals=Lh

        A, Lh, E = torch.linalg.svd(torch.from_numpy(static_feat).T)
        A = A[:, :len(Lh)] #trim the time field (in case time > space)
        A = A[:, :len(Lh)] #trim the time field (in case time > space)

        self.var_explained=(Lh*Lh)/(torch.sum(Lh*Lh))
        tau0_data = (self.PCS[:,:self.select_neofs].T)

        tlag = 1
        self.tlag = tlag
        x0=tau0_data[:,0:-tlag]
        x1=tau0_data[:,tlag:]

        #covariance structrure
        self.static_feat = x0 if isinstance(x0, torch.Tensor) else torch.from_numpy(x0)
        self.static_feat = self.static_feat.to(torch.float64).to(device)

        #lagged feature
        self.static_feat1 = x1 if isinstance(x1, torch.Tensor) else torch.from_numpy(x1)
        self.static_feat1 = self.static_feat1.to(torch.float64).to(device)

        self.device = device
        self.dim = dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_edges = max_num_edges
        self.self_loops = self_loops
        self.diag = torch.eye(self.num_nodes).bool().to(device)

    def forward(self):
        #covariance structure
        nodevec1 = self.static_feat
        nodevec2 = self.static_feat
        #lagged covariance structure
        nodevec11 = self.static_feat
        nodevec21 = self.static_feat1
        adj = (nodevec21 @ nodevec11.T)@(torch.linalg.pinv(nodevec1 @ nodevec1.T))
        return adj

    def forward_params(self):
        #covariance structure
        nodevec1 = self.static_feat
        nodevec2 = self.static_feat
        #lagged covariance structure
        nodevec11 = self.static_feat
        nodevec21 = self.static_feat1


        cT = (nodevec21 @ nodevec11.T)/(nodevec11.shape[1]-1)
        c0 = (nodevec1 @ nodevec1.T)/(nodevec11.shape[1]-1)
        G = (cT.numpy())@(np.linalg.pinv(c0.numpy()))

        g, u = np.linalg.eig(G)
        #sort the eigenvecs
        iSort = g.argsort()[::-1]    #Sort the eigen values and vectors in order
        g     = g[iSort]
        u     = u[:,iSort]


        # Define the adjoints (v) based on the transpose of G
        eigVal_T, v = np.linalg.eig(np.transpose(G))
        iSortT = eigVal_T.argsort()[::-1]
        eigVal_T    = eigVal_T[iSortT]
        v           = v[:,iSortT]

        # But modes should ultimately be sorted by decreasing decay time (i.e., decreasing values of 1/beta.real)
        # Compute Beta
        b_tau   = np.log(g)
        b_alpha = b_tau/self.tlag

        # Sort data by decreasing decay time
        sortVal = -1/b_alpha.real              #Decay time

        iSort2 = sortVal.argsort()[::-1]
        u      = u[:,iSort2]
        v      = v[:,iSort2]
        g      = g[iSort2]
        b_alpha = b_alpha[iSort2]

        nDat = G.shape[0]
        # Make diagonal array of Beta (values should be negative)
        # beta = torch.zeros((nDat, nDat), dtype=torch.cdouble)
        # beta.fill_diagonal_(b_alpha)

        beta = np.zeros((nDat, nDat), dtype=complex)
        mask = np.diag(np.ones_like(b_alpha))
        beta = mask*np.diag(b_alpha) + (1. - mask)*beta

        #Need to normalize u so that u_transpose*v = identitity matrix, and u*v_transpose = identity matrix as well
        normFactors = (u.T@v)
        normU       = (u@np.linalg.inv(normFactors))

        # STEP 3: Compute L and Q matrices
        # Compute L matrix as normU * beta * v_transpose
        L = np.dot(normU, np.dot(beta, np.transpose(v)))

        # Compute Q matrix
        Q_negative = np.dot(L, c0) + np.dot(c0, np.transpose(L))
        Q = -Q_negative

        # Also define the periods and decay times
        # Also define the periods and decay times
        periods = (2 * np.pi) / b_alpha.imag
        decayT  = -1 / b_alpha.real

        device =self.device
        b_alpha=torch.from_numpy(b_alpha).cdouble().to(device)
        L=torch.from_numpy(L).cdouble().to(device)
        Q=torch.from_numpy(Q).cdouble().to(device)
        G=torch.from_numpy(G).cdouble().to(device)
        normU=torch.from_numpy(normU).cdouble().to(device)
        v=torch.from_numpy(v).cdouble().to(device)
        g=torch.from_numpy(g).cdouble().to(device)
        periods=torch.from_numpy(periods).float().to(device)
        decayT=torch.from_numpy(decayT).float().to(device)

        return b_alpha, L, Q, G, c0, cT, normU, v, g, periods, decayT, nodevec11, nodevec21


class LIMG(nn.Module):
    def __init__(self, num_nodes, max_num_edges, dim, static_feat, device='cuda', alpha1=0.1, alpha2=2.0,
                 self_loops=True):
        super().__init__()
        if static_feat is None:
            raise ValueError("Please give static node features (e.g. part of the timeseries)")
        self.num_nodes = num_nodes
        xd = static_feat.shape[1]
        tlag = 1
        self.tlag = tlag
        x0=static_feat[:,0:-tlag]
        x1=static_feat[:,tlag:]

        xd = x0.shape[1]
        #covariance structrure
        self.lin1 = nn.Linear(xd, dim)
        self.lin2 = nn.Linear(xd, dim)
        self.linlag1 = nn.Linear(xd, dim)
        self.linlag2 = nn.Linear(xd, dim)

        self.static_feat = x0 if isinstance(x0, torch.Tensor) else torch.from_numpy(x0)
        self.static_feat = self.static_feat.to(torch.float64).to(device)

        #lagged feature
        self.static_feat1 = x1 if isinstance(x1, torch.Tensor) else torch.from_numpy(x1)
        self.static_feat1 = self.static_feat1.to(torch.float64).to(device)

        self.device = device
        self.dim = dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_edges = max_num_edges
        self.self_loops = self_loops
        self.diag = torch.eye(self.num_nodes).bool().to(device)

    def forward(self):
        #covariance structure
        nodevec1 = self.static_feat
        nodevec2 = self.static_feat
        #lagged covariance structure
        nodevec11 = self.static_feat
        nodevec21 = self.static_feat1

        adj = (nodevec21 @ nodevec11.T)@(torch.linalg.pinv(nodevec1 @ nodevec1.T))

        return adj

    def NoiseQ(self):
        #covariance structure
        nodevec1 = self.static_feat
        nodevec2 = self.static_feat
        #lagged covariance structure
        nodevec11 = self.static_feat
        nodevec21 = self.static_feat1
        G_1 = (nodevec21 @ nodevec11.T)@(torch.linalg.pinv(nodevec1 @ nodevec1.T))
        G_1=G_1.cpu().detach().numpy()
        x0_c = self.static_feat.cpu().detach().numpy().T

        C0 = x0_c.T @ x0_c / (x0_c.shape[0] - 1)

        G_eval, G_evects = np.linalg.eig(G_1)
        L_evals = (1/tau) * np.log(G_eval)
        L = G_evects @ np.diag(L_evals) @ np.linalg.pinv(G_evects)
        L = np.matrix(L)
        Q = -(L @ C0 + C0 @ L.H)  # Noise covariance
        Q=torch.from_numpy(Q)
        return Q,L,G_eval
