import torch
import torch.nn as nn
import torch.optim as optim
import math
from evml.regression_losses import EvidentialMarginalLikelihood, EvidenceRegularizer, modified_mse as mmse_loss 

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
    elif name in ['linex']:
        loss = Custom_LINEX()
    elif name in ['gauss']:
        print('getttt probable babbby')
        loss = nn.GaussianNLLLoss()
    elif name in ['laplace']:
        print('getttt probable babbby')
        loss = Custom_Laplace()
    elif name in ['cauchy']:
        print('getttt probable babbby')
        loss = Custom_Cauchy()
    elif name in ['crps']:
        print('getttt probable babbby')
        loss = Custom_CRPS()
    elif name in ["evloss"]:
        loss = EvidentialLoss()
    else:
        raise ValueError('Available Losses: MAE, L1, L2, MSE, Gauss, Laplace, Cauchy, CRPS ... ')  # default
    return loss


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params

class EvidentialLoss(nn.Module):
    def __init__(self):
        super(EvidentialLoss, self).__init__()
        self.nll_loss = EvidentialMarginalLikelihood() ## original loss, NLL loss
        self.reg = EvidenceRegularizer() ## evidential regularizer
        self.mmse_loss = mmse_loss ## lipschitz MSE loss
        
    def forward(self, input, target, scale=None, eps=1e-06, reduction='mean'):
        gamma, nu, alpha, beta = input
        loss = self.nll_loss(gamma, nu, alpha, beta, target)
        loss += self.reg(gamma, nu, alpha, beta, target)
        loss += self.mmse_loss(gamma, nu, alpha, beta, target)
        return loss
    
class Custom_CRPS(nn.Module):
    """
    compute the CRPS cost function of a normal distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Gaussian CRPS: Scalar with CRPS over the batch
    
    """
    def __init__(self):
        super(Custom_CRPS,self).__init__();
        
    def forward(self,input, target, scale, eps=1e-06, reduction='mean'):
        # Inputs and targets much have same shape
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        if input.size() != target.size():
              raise ValueError("input and target must have same size")

        # Second dim of scale must match that of input or be equal to 1
        scale = scale.view(input.size(0), -1)
        if scale.size(1) != input.size(1) and scale.size(1) != 1:
            raise ValueError("scale is of incorrect size")

        # Check validity of reduction mode
        if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
            raise ValueError(reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(scale < 0):
            raise ValueError("scale has negative entry/entries")

        # Clamp for stability
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=eps)

        # Calculate loss (without constant)
        loc =(target-input)/scale
        pie = torch.as_tensor(math.pi) #yummmm
        phi =1.0 / torch.sqrt((2.0*pie))*torch.exp(-torch.square(loc)/2.0)
        Phi = 0.5*(1.0+torch.erf((loc/torch.sqrt(torch.as_tensor(2.0)))))
        loss = scale * (loc * (2.0 * Phi - 1.) + 2.0 * phi - 1.0 / torch.sqrt(pie))

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
def CRPSloss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    compute the CRPS cost function of a normal distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Gaussian CRPS: Scalar with CRPS over the batch
    
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
          raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loc =(target-input)/scale
    pie = torch.as_tensor(math.pi) #yummmm
    phi =1.0 / torch.sqrt((2.0*pie))*torch.exp(-torch.square(loc)/2.0)
    Phi = 0.5*(1.0+torch.erf((loc/torch.sqrt(torch.as_tensor(2.0)))))
    loss = scale * (loc * (2.0 * Phi - 1.) + 2.0 * phi - 1.0 / torch.sqrt(pie))

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    

class Custom_Laplace(nn.Module):
    """
    compute the Negative Log Liklihood cost function of a laplace distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        laplace NLL loss
    
    """
    def __init__(self):
        super(Custom_Laplace,self).__init__();
        
    def forward(self,input, target, scale, eps=1e-06, reduction='mean'):
        loss = torch.log(2*scale) + torch.abs(input - target)/scale

        # Inputs and targets much have same shape
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        if input.size() != target.size():
              raise ValueError("input and target must have same size")

        # Second dim of scale must match that of input or be equal to 1
        scale = scale.view(input.size(0), -1)
        if scale.size(1) != input.size(1) and scale.size(1) != 1:
            raise ValueError("scale is of incorrect size")

        # Check validity of reduction mode
        if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
            raise ValueError(reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(scale < 0):
            raise ValueError("scale has negative entry/entries")

        # Clamp for stability
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=eps)

        # Calculate loss (without constant)
        loss = (torch.log(2*scale) + torch.abs(input - target) / scale).view(input.size(0), -1).sum(dim=1)

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
    
def LaplaceNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    compute the Negative Log Liklihood cost function of a laplace distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        laplace NLL loss
    
    """
    loss = torch.log(2*scale) + torch.abs(input - target)/scale

    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
          raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = (torch.log(2*scale) + torch.abs(input - target) / scale).view(input.size(0), -1).sum(dim=1)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    
    
class Custom_Cauchy(nn.Module):
    """
    compute the Negative Log Liklihood cost function of a Cauchy distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Cauchy NLL loss
    """
    
    def __init__(self):
        super(Custom_Cauchy,self).__init__();
        
    def forward(self,input, target, scale, eps=1e-06, reduction='mean'):
        # Inputs and targets much have same shape
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        if input.size() != target.size():
            raise ValueError("input and target must have same size")

        # Second dim of scale must match that of input or be equal to 1
        scale = scale.view(input.size(0), -1)
        if scale.size(1) != input.size(1) and scale.size(1) != 1:
            raise ValueError("scale is of incorrect size")

        # Check validity of reduction mode
        if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
            raise ValueError(reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(scale < 0):
            raise ValueError("scale has negative entry/entries")

        # Clamp for stability
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=eps)

        # Calculate loss (without constant)
        loss = (torch.log(3.14159265*scale) + torch.log(1 + ((input - target)**2)/scale**2)) .view(input.size(0), -1).sum(dim=1)

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
   
    
def CauchyNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    compute the Negative Log Liklihood cost function of a Cauchy distribution defined by the
    mean and std. 
    
    Args: 
        input: median value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Cauchy NLL loss
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = (torch.log(3.14159265*scale) + torch.log(1 + ((input - target)**2)/scale**2)) .view(input.size(0), -1).sum(dim=1)


    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    
    
    
    
class Custom_LINEX(nn.Module):
    """
    compute the Linear exponential loss cost function of a normal distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        LINEX Loss: Scalar with LINEX over the batch
    
    """
    def __init__(self):
        super(Custom_LINEX,self).__init__();
        
    def forward(self,input, target, eps=1e-06, reduction='mean'):
        # Inputs and targets much have same shape
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        if input.size() != target.size():
              raise ValueError("input and target must have same size")

        # Check validity of reduction mode
        if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
            raise ValueError(reduction + " is not valid")


        # Calculate loss (without constant)
        # Calculate loss (without constant)
        errz = (torch.abs(target)-torch.abs(input))
        a=torch.as_tensor(2)
        loss = (2/torch.square(a)) * (torch.exp(a*errz) - a*(errz)-1)
        
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    

    
def LINEXloss(input, target, eps=1e-06, reduction='mean'):
    """
    compute the Linear exponential loss cost function of a normal distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Gaussian CRPS: Scalar with CRPS over the batch
    
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
          raise ValueError("input and target must have same size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Calculate loss (without constant)
    errz = (torch.abs(target)-torch.abs(input))
    a=.5
    loss = (2/torch.square(a)) * (torch.exp(a*errz) - a*(errz)-1)
    
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss