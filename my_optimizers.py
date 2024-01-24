import math
import torch
import copy
from torch.optim.optimizer import Optimizer


class OptimizerOpt(Optimizer):
    '''
    The class allows for a series of operations, including buffer manipulation, buffer initialization
    and gradient norm computing
    '''
    def __init__(self, params, defaults):
        super(OptimizerOpt, self).__init__(params, defaults)
    def set_model(self, indicator):
        '''
        set value for the neural model
        '''
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) !=0:
                    if indicator == 'current': 
                        # get value from buffer 'current_model'
                        p.data = state['current_model'].clone() 
                    elif indicator == 'previous':
                        # get value from buffer 'previous_model'
                        p.data = state['previous_model'].clone() 
                    elif indicator == 'current_diffusion':
                        # get value from buffer 'current_diffusion_model' 
                        p.data = state['current_diffusion_model'].clone() 
                    elif indicator == 'previous_diffusion':
                        # get value from buffer 'previous_diffusion_model'
                        p.data = state['previous_diffusion_model'].clone() 
                    else:
                        raise ValueError('Invalid indicator')
                else:
                    raise ValueError('Optimizer is not initialized properly')
                
    def set_buff(self, indicator):
        '''
        set value for the buffer
        '''
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) !=0:
                    if indicator == 'current_diffusion': 
                        # load neural model into the buffer 'current_diffusion_model'
                        state['current_diffusion_model'] = p.data.clone()
                    elif indicator == 'shift':
                        # shift buffer from buffer'current_diffusion_model' to buffer 'previous_diffusion_model'
                        state['previous_diffusion_model'] = state['current_diffusion_model'].clone() 
                    else:
                        raise ValueError('Invalid indicator')
                else:
                    raise ValueError('Optimizer is not initialized properly')
                    
    def init_buff(self): # Initialize buffer data
        '''
            state['current_model']: current model within iteration
            state[['current_diffusion_model']: current combined model over iterations
        '''
        for group in self.param_groups: 
            for p in group['params']:     
                state = self.state[p]
                if len(state) == 0:
                    state['forward_gradient'] = torch.zeros_like(p.data)
                    state['backward_gradient'] = torch.zeros_like(p.data)
                    state['current_model'] = p.data.clone() 
                    state['previous_model'] = p.data.clone()  
                    state['current_diffusion_model'] = p.data.clone() 
                    state['previous_diffusion_model'] = p.data.clone()  
                    state['step'] = 1
                
    def get_gradient_norm(self):
        norm = 0.0   
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) ==0:
                    raise ValueError('Optimizer is not initialized properly')
                grad = state['forward_gradient']
                norm+= torch.norm(grad.clone())**2
        return norm**0.5
    
    
class SS_OG(OptimizerOpt):
    '''
        stochastic same-sample optimistic gradient method
    '''
    def __init__(self, params, lr=1e-4, gamma = 1.0, weight_decay=0.0000):
        if not 0.0<=lr<1.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        defaults = dict(lr=lr, gamma = gamma, weight_decay=weight_decay)
        super(SS_OG, self).__init__(params, defaults) 
        
    def forward_step(self): # store the forward gradient and load the previous model
        '''
        Query the gradient oracle relative to the current model and save it in the buffer state['forward_gradient']
        '''
        norm = 0.0   
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    raise ValueError('Optimizer is not initialized properly')
                 # get the forward gradient
                state['forward_gradient'] = grad.clone() 
                # set to previous model in order to computing backward gradient during the next step
                p.data= state['previous_diffusion_model'].clone() 
                norm+= torch.norm(grad.clone())**2 
        return norm**0.5
    
    def step(self, closure = None):
        '''
        Examples:
            >>> optimizer.zero_gradients()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.forward_step()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()      
        '''
        a = 0.0
        loss = None 
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # get backward gradient
                grad  = p.grad.data 
                if len(state) == 0:
                    raise ValueError('Optimizer is not initialized properly')
                # shift the model buffer  
                state['previous_model'] = state['current_model'].clone()        
                p.data = state['current_diffusion_model'].clone()
                if state['step']==1: 
                    p.data.add_(state['forward_gradient'], alpha=-2.0*group['lr'])
                else:
                    # forward gradient
                    p.data.add_(state['forward_gradient'], alpha=-(1.0+group['gamma'])*group['lr'])
                    # backward gradient
                    p.data.add_(grad, alpha=group['lr'])  
                state['current_model'] = p.data.clone()
                state['step'] +=1
        return loss


class GD(OptimizerOpt):
    def __init__(self, params, lr): 
        defaults = dict(lr=lr)
        super(GD, self).__init__(params, defaults)
                
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups: #the list of all parameters, groups are layers
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['forward_gradient'] = torch.zeros_like(p.grad.data)
                d_p = p.grad.data
                state['forward_gradient'] = d_p.clone()
                p.data.add_(d_p, alpha = -group['lr'])
        return loss
    
class Cata(OptimizerOpt):
    def __init__(self, params, lr, beta, P): 
        defaults = dict(lr=lr, beta=beta, P=P)
        super(Cata, self).__init__(params, defaults)
        #self.param_groups['params_z'] = copy.deepcopy(self.param_groups['params'])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            beta = group['beta']
            P = group['P']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                state = self.state[p]                              
                if len(state) == 0:
                    state['forward_gradient'] = torch.zeros_like(p.data.clone())
                    state['backward_gradient'] = torch.zeros_like(p.data.clone())
                    state['step'] = 1
                state['forward_gradient'] = d_p.clone()
                if 'z' not in param_state:
                    param_state['z'] = torch.clone(p.data).detach()
                else:
                    zz = param_state['z']
                    zz.add_(p.data-zz, alpha = group['beta'])
                    d_p.add_(p.data-zz, alpha = P)
                p.data.add_(d_p, alpha = -group['lr'])
        return loss
    

class Adam(OptimizerOpt):
    def __init__(self, params, lr, betas, eps):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['forward_gradient'] = torch.zeros_like(p.grad.data)                  
                state['forward_gradient'] = grad.clone()
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
    
    
    
