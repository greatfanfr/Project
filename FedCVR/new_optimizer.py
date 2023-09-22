import math

import torch
from torch.optim.optimizer import Optimizer, required
from differential_privacy import differential_privacy

class SGD_Avg(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_Avg, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_Avg, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss


class SGD_Prox(SGD_Avg):
    def __init__(self, params, global_params, penalty,
                 lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(SGD_Prox, self).__init__(params=params, lr=lr, momentum=momentum, dampening=dampening,
                                       weight_decay=weight_decay, nesterov=nesterov)

        self.global_params = global_params
        self.penalty = penalty

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, (g_name, g_param) in zip(group['params'],
                                            self.global_params.items()):
                if p.grad is None:
                    continue

                d_p = p.grad
                print(d_p.shape)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                d_p.add_(p - self.global_params[g_name], alpha=self.penalty)
                p.add_(d_p, alpha=-group['lr'])

        return loss


class SGD_SCAFFOLD(SGD_Avg):
    def __init__(self, params, global_ctl, local_ctl,
                 lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(SGD_SCAFFOLD, self).__init__(params=params, lr=lr, momentum=momentum, dampening=dampening,
                                           weight_decay=weight_decay, nesterov=nesterov)

        self.global_ctl = global_ctl
        self.local_ctl = local_ctl

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, (g_name, g_param), (l_name, l_param) in zip(group['params'],
                                                               self.global_ctl.items(),
                                                               self.local_ctl.items()):
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                # d_p=d_p.add()
                d_p=d_p.add(self.global_ctl[g_name] - self.local_ctl[l_name], alpha=1.0)
                p.add_(d_p, alpha=-group['lr'])

        return loss


class SGD_VRA(SGD_Avg):
    def __init__(self, params, dual_params, global_params, penalty,
                 lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(SGD_VRA, self).__init__(params=params, lr=lr, momentum=momentum, dampening=dampening,
                                      weight_decay=weight_decay, nesterov=nesterov)

        self.dual_params = dual_params
        self.global_params = global_params
        self.penalty = penalty

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, (d_name, d_param), (g_name, g_param) in zip(group['params'],
                                                               self.dual_params.items(),
                                                               self.global_params.items()):
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                d_p.sub_(self.dual_params[d_name], alpha=1.0)
                d_p.add_(p - self.global_params[g_name], alpha=self.penalty)
                p.add_(d_p, alpha=-group['lr'])
                # print('a', d_p.size())
                # print('b', self.dual_params[d_name].size())

        return loss


class SGD_CVR(SGD_Avg):
    def __init__(self, params, lctr_params, global_params, penalty,
                 lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(SGD_CVR, self).__init__(params=params, lr=lr, momentum=momentum, dampening=dampening,
                                      weight_decay=weight_decay, nesterov=nesterov)

        self.lctr_params = lctr_params
        self.global_params = global_params
        self.penalty = penalty
        self.comb_factor = 1.0 / (1.0 + self.penalty * lr)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, (d_name, d_param), (g_name, g_param) in zip(group['params'],
                                                               self.lctr_params.items(),
                                                               self.global_params.items()):
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                d_p.sub_(self.lctr_params[d_name], alpha=1.0)
                # d_p.add_(p - self.global_params[g_name], alpha=self.penalty)
                p.add_(d_p, alpha=-group['lr'])
                p.mul_(self.comb_factor)
                p.add_(self.global_params[g_name], alpha=1.0 - self.comb_factor)

        return loss
