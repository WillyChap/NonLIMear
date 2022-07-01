import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(name, model, **kwargs):
    name = name.lower().strip()
    parameters = get_trainable_params(model)
    if name == 'adam':
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        wd = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        print('Using Adam optimizer: Lr=', lr, 'Wd=', wd)
        return optim.Adam(parameters, lr=lr, weight_decay=wd)
    elif name == 'sgd':
        print('Using SGD optimizer')
        lr = kwargs['lr'] if 'lr' in kwargs else 0.01
        momentum = 0.9
        wd = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        nesterov = kwargs['nesterov'] if 'nesterov' in kwargs else True
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    elif name == 'rmsprop':
        lr = kwargs['lr'] if 'lr' in kwargs else 0.005
        return optim.RMSprop(parameters, lr=lr, momentum=0.0, eps=1e-10)
    else:
        raise ValueError("Unknown optimizer", name)


def get_loss(name, reduction='mean'):
    # Specify loss function
    name = name.lower().strip()
    if name in ['l1', 'mae']:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ['l2', 'mse']:
        loss = nn.MSELoss(reduction=reduction)
    elif name in ['bingbop','crps']:
        loss = nn.MSELoss(reduction=reduction)
    else:
        raise ValueError()  # default
    return loss


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params


def crps_cost_function(y_pred,y_true):
    """
    compute the CRPS cost function of a normal distribution defined by the
    mean and std. 
    
    Args: 
        y_true: true values
        y_pred: tensor containing preds [mean,std]
    
    Returns: 
        mean_crps: Scalar with mean CRPS over the batch
    """
    mu = y_pred[:,0]
    sigma = y_pred[:,1]
    var=torch.abs(sigma)
    #the following three variabsles are for convenience 
    loc =(y_true-mu)/var
    phi =1.0/torch.sqrt(2.0*3.141592653589793)*torch.exp(-torch.square(loc)/2.0)
    Phi = 0.5*(1.0+torch.erf(loc/1.4142135623730951)) #loc/sqrt(2.0)
    #crps for the target pair. 
    crps = torch.sqrt(var)*(loc*(2.0*Phi-1.) + 2.0 * phi - 1.0 / 1.7724538509055159)
    return torch.mean(crps)