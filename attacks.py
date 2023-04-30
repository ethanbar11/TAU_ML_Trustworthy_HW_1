import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255.,
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
            x_adv = x_adv.clamp(x - self.eps, x + self.eps)
            x_adv = x_adv.clone().detach().requires_grad_(True).to(x.device)
        else:
            x_adv = x.clone().detach().requires_grad_(True)
        for i in range(self.n):
            self.model.zero_grad()
            assert (x_adv - x).abs().max() <= self.eps + 1e-5
            outputs = self.model(x_adv)
            loss = self.loss_func(outputs, y).sum()
            loss.backward()
            # with torch.no_grad():
            grad = self.alpha * x_adv.grad.sign()
            if targeted:
                x_adv = x_adv - grad
            else:
                x_adv = x_adv + grad
            x_adv = x_adv.clamp(0, 1)  #
            x_adv = x_adv.clamp(x - self.eps, x + self.eps).clone().detach().requires_grad_(True).to(x.device)
            if self.early_stop:
                with torch.no_grad():
                    outputs = self.model(x_adv)
                    if targeted:
                        successful_attack = (outputs.argmax(1) == y)
                    else:
                        successful_attack = (outputs.argmax(1) != y)
                if successful_attack.all():
                    return x_adv

        return x_adv


def clip_perturbation(x, perturbation, eps):
    x_adv = torch.clamp(x + perturbation, x - eps, x + eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def project_perturbation(perturbation, eps):
    return torch.clamp(perturbation, -eps, eps)


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255., momentum=0.,
                 k=200, sigma=1 / 255., rand_init=True, early_stop=True):
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
        self.sigma = sigma
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

        batch_size = x.shape[0]
        perturbation = torch.zeros_like(x)
        if self.rand_init:
            perturbation.uniform_(-self.eps, self.eps)
        x_adv = clip_perturbation(x, perturbation, self.eps)
        x_adv = x_adv.clone().detach().requires_grad_(True).to(x.device)
        historical_gradient = torch.zeros_like(x)
        queries = torch.zeros(batch_size)

        for iteration in range(self.n):
            self.model.zero_grad()
            perturbation.requires_grad_()
            logits, queries = self.get_grad(batch_size, queries, x, x_adv)
            # logits = logits.view(batch_size, 2 * self.k, -1)
            loss = self.loss_func(logits, y.unsqueeze(1).expand(-1, 2 * self.k).reshape(-1)).sum()
            loss.backward()
            grad = self.alpha * x_adv.grad.sign()
            historical_gradient = self.momentum * historical_gradient + (1 - self.momentum) * grad
            if targeted:
                x_adv = x_adv - historical_gradient
            else:
                x_adv = x_adv + historical_gradient
            x_adv = x_adv.clamp(0, 1)  #
            x_adv = x_adv.clamp(x - self.eps, x + self.eps).clone().detach().requires_grad_(True).to(x.device)

            if self.early_stop:
                with torch.no_grad():
                    logits_adv = self.model(x_adv)
                    if targeted:
                        success = (logits_adv.argmax(dim=1) == y).float()
                    else:
                        success = (logits_adv.argmax(dim=1) != y).float()
                    if success.sum() == batch_size:
                        break

        return x_adv, queries

    def get_grad(self, batch_size, queries, x, x_adv):
        z = self.sigma * torch.randn(batch_size, self.k, *x.shape[1:]).to(x.device)
        perturbed_z_plus = clip_perturbation(x_adv.unsqueeze(1), z, self.eps)
        perturbed_z_minus = clip_perturbation(x_adv.unsqueeze(1), -z, self.eps)
        perturbed_z = torch.cat((perturbed_z_plus, perturbed_z_minus), dim=1)
        perturbed_z = perturbed_z.view(-1, *x.shape[1:])
        queries_per_step = 2 * self.k
        queries += queries_per_step
        logits = self.model(perturbed_z)
        return logits, queries


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8 / 255., n=50, alpha=1 / 255.,
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
            x_adv = x_adv.clone().detach().requires_grad_(True).to(x.device)
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
                    if targeted:
                        successful_attack = (outputs.argmax(1) == y)
                    else:
                        successful_attack = (outputs.argmax(1) != y)
                if successful_attack.all():
                    return x_adv
        return x_adv


if __name__ == '__main__':
    # Creating a demo model optimization
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    x = torch.rand(10, requires_grad=True)
    y = torch.rand(10)
    loss = nn.MSELoss()
    y_pred = model(x)
    l = loss(y_pred, y)
    l.backward()
    print(model[0].weight.grad)
    print(model[2].weight.grad)
    print(x.grad)
