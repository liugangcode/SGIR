'''
This file contains a collection of regression loss functions.
'''

import torch
import torch.nn.functional as F

__all__ = ['weighted_l1_loss']

#### adapted from https://github.com/YyzHarry/imbalanced-regression

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss

def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss


def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.view(loss.size())
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1., weights=None):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss


def qunatile_loss(inputs, targets, weights=None, tau=None):
    if tau is not None:
        diff = inputs - targets
        mask = diff.ge(0).float() - tau
        loss = mask * diff
    else:
        loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss

#### adapted from https://github.com/jiawei-ren/BalancedMSE

# from torch.nn.modules.loss import _Loss
# import joblib

# class GAILoss(_Loss):
#     def __init__(self, init_noise_sigma, gmm):
#         super(GAILoss, self).__init__()
#         self.gmm = joblib.load(gmm)
#         self.gmm = {k: torch.tensor(self.gmm[k]).cuda() for k in self.gmm}
#         self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

#     def forward(self, pred, target):
#         noise_var = self.noise_sigma ** 2
#         loss = gai_loss(pred, target, self.gmm, noise_var)
#         return loss


# def gai_loss(pred, target, gmm, noise_var):
#     gmm = {k: gmm[k].reshape(1, -1).expand(pred.shape[0], -1) for k in gmm}
#     mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var + 0.5 * noise_var.log()
#     sum_var = gmm['variances'] + noise_var
#     balancing_term = - 0.5 * sum_var.log() - 0.5 * (pred - gmm['means']).pow(2) / sum_var + gmm['weights'].log()
#     balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
#     loss = mse_term + balancing_term
#     loss = loss * (2 * noise_var).detach()

#     return loss.mean()


# class BMCLoss(_Loss):
#     def __init__(self, init_noise_sigma):
#         super(BMCLoss, self).__init__()
#         self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

#     def forward(self, pred, target):
#         noise_var = self.noise_sigma ** 2
#         loss, loss_all = bmc_loss(pred, target, noise_var)
#         return loss, loss_all


# def bmc_loss(pred, target, noise_var):
#     target = target.view(-1,1)
#     pred = pred.view(-1,1)
#     logits = - 0.5 * (pred - target.T).pow(2) / noise_var
#     loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda(), reduction='none')

#     return loss.mean() * (2 * noise_var).detach(), loss


# class BNILoss(_Loss):
#     def __init__(self, init_noise_sigma, bucket_centers, bucket_weights):
#         super(BNILoss, self).__init__()
#         self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))
#         self.bucket_centers = torch.tensor(bucket_centers).cuda()
#         self.bucket_weights = torch.tensor(bucket_weights).cuda()

#     def forward(self, pred, target):
#         noise_var = self.noise_sigma ** 2
#         loss = bni_loss(pred, target, noise_var, self.bucket_centers, self.bucket_weights)
#         return loss


# def bni_loss(pred, target, noise_var, bucket_centers, bucket_weights):
#     mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

#     num_bucket = bucket_centers.shape[0]
#     bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
#     bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

#     balancing_term = - 0.5 * (pred.expand(-1, num_bucket) - bucket_center).pow(2) / noise_var + bucket_weights.log()
#     balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
#     loss = mse_term + balancing_term
#     loss = loss * (2 * noise_var).detach()
#     return loss.mean()