import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        device = x.device
        if self.rand_init:
            delta = torch.rand_like(x, requires_grad=True) * 2 * self.eps - self.eps
            x_adv = x + delta
            x_adv = x_adv.clamp(0, 1)
            x_adv =x_adv.clone().detach().requires_grad_(True).to(x.device)
        else:
            x_adv = x.clone().detach().requires_grad_(True)

        for i in range(self.n):
            self.model.zero_grad()
            outputs = self.model(x_adv)
            loss = self.loss_func(outputs, y).sum()
            loss.backward()
            # with torch.no_grad():
            grad = self.alpha * x_adv.grad.sign()
            if targeted:
                x_adv = x_adv - grad
            else:
                x_adv = x_adv + grad
            x_adv = x_adv.clamp(0, 1)#
            x_adv = x_adv.clamp(x - self.eps, x + self.eps).clone().detach().requires_grad_(True).to(x.device)
            if self.early_stop:
                with torch.no_grad():
                    outputs = self.model(x_adv)
                    if targeted :
                        successful_attack = (outputs.argmax(1) == y)
                    else:
                        successful_attack = (outputs.argmax(1) != y)
                if successful_attack.all():
                    return x_adv

        return x_adv


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma=sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """
    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        device = x.device
        if self.rand_init:
            delta = torch.rand_like(x, requires_grad=True) * 2 * self.eps - self.eps
            x_adv = x + delta
            x_adv = x_adv.clamp(0, 1)
            x_adv =x_adv.clone().detach().requires_grad_(True).to(x.device)
        else:
            x_adv = x.clone().detach().requires_grad_(True)

        for i in range(self.n):
            for model in self.models:
                model.zero_grad()
            outputs = [model(x_adv) for model in self.models]
            loss = torch.stack([self.loss_func(output, y) for output in outputs]).sum()
            loss.backward()
            if targeted:
                x_adv = x_adv - self.alpha * x_adv.grad.sign()
            else:
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
            x_adv = x_adv.clamp(0, 1)
            x_adv = x_adv.clamp(x - self.eps, x + self.eps).clone().detach().requires_grad_(True).to(x.device)
            if self.early_stop:
                with torch.no_grad():
                    outputs = torch.stack([model(x_adv) for model in self.models])
                    if targeted :
                        successful_attack = (outputs.argmax(1) == y)
                    else:
                        successful_attack = (outputs.argmax(1) != y)
                if successful_attack.all():
                    return x_adv
        return x_adv


if __name__ == '__main__':
    # Creating a demo model optimization
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    x = torch.rand(10,requires_grad=True)
    y = torch.rand(10)
    loss = nn.MSELoss()
    y_pred = model(x)
    l = loss(y_pred, y)
    l.backward()
    print(model[0].weight.grad)
    print(model[2].weight.grad)
    print(x.grad)
