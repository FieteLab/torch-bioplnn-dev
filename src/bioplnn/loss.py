import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty.
Advances in Neural Information Processing Systems,
31. https://papers.nips.cc/paper_files/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html

"""


""" USAGE
batch_size = 8
target = torch.randint(0, 2, (batch_size, ))
logits = torch.randn((batch_size, 2))
criterion = EDLLoss(num_classes=2)
loss = criterion(logits, target, 1, 1)
"""


def relu_evidence(logits):
    return F.relu(logits)


def exp_evidence(logits):
    return torch.exp(torch.clip(logits / 10, -10, 10))


def softplus_evidence(logits):
    return F.softplus(logits)


def get_edl_diagnostics(predictions, targets, evidences, uncertainties):
    acc = np.equal(predictions, targets)
    evidence_total = np.sum(evidences, axis=1)
    ev_succ = evidence_total[acc == 1]
    ev_fail = evidence_total[acc == 0]
    u = uncertainties
    u_succ = u[acc == 1]
    u_fail = u[acc == 0]

    return ev_succ, ev_fail, u_succ, u_fail


def get_edl_vars(logits, num_classes, evidence_fn=exp_evidence):
    evidence = evidence_fn(logits)
    alpha = evidence + 1
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    return evidence, u, prob, alpha


class EDLLoss(nn.Module):
    def __init__(self, num_classes=2, evidence_fn=exp_evidence):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.evidence_fn = evidence_fn
        self.step_count = 0

    def KL(self, alpha):
        beta = torch.ones(
            (1, self.num_classes), dtype=torch.float, device=alpha.device
        )
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdims=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(
            torch.lgamma(alpha), axis=1, keepdim=True
        )
        lnB_uni = torch.sum(
            torch.lgamma(beta), axis=1, keepdim=True
        ) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = (
            torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True)
            + lnB
            + lnB_uni
        )
        return kl

    def forward(self, logits, target, global_step=None, annealing_step=65535):
        target = F.one_hot(target, num_classes=self.num_classes)

        evidence, u, prob, alpha = get_edl_vars(
            logits, num_classes=self.num_classes, evidence_fn=self.evidence_fn
        )

        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S

        A = torch.sum((target - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )

        if global_step is not None:
            self.step_count = global_step
        annealing_coef = (
            float(self.step_count) / annealing_step
        )  # rho in the paper
        self.step_count += 1
        annealing_coef = min(1.0, annealing_coef)
        # print(annealing_coef)

        alp = E * (1 - target) + 1
        C = annealing_coef * self.KL(alp)

        loss = (A + B) + C
        loss = loss.squeeze(dim=1)
        return loss, {"evidence": evidence, "uncertainty": u, "prob": prob}
