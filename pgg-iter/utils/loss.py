import torch
from torch.nn import functional as F


def advantage(A):
    """
    Нормализация advantage.
    """
    # std = 1e-4 + A.std() if len(A) > 0 else 1
    # adv = (A - A.mean()) / std
    # adv = adv.detach()
    # # adv[adv != adv] = 0
    # # UPD: зануление nan'ов, т.к. видимо float('nan') != float('nan')
    # #      довольно тонко
    return A


def valueLoss(v, returns):
    """
    MSE для Value-функции.
    """
    return 0.5 * F.mse_loss(v, returns)
    # return 0.5 * ((v - returns) ** 2).mean()


def entropyLoss(prob, logProb):
    return (prob * logProb).sum(1).mean()


def ppo_loss(A, rho, eps=0.2):
    """
    Пока что кстати непонятно это действительно один в один
    соответсвует clipping PPO, или просто короткая форма записать баг.
    ПРОВЕРЬ!
    """
    return -torch.min(rho * A, rho.clamp(1 - eps, 1 + eps) * A).mean()


def PG(pi, rho, A):
    """
    Возвращает policy gradients и энтропию по:
     - логитам `pi`
     - отношению вероятностей действий `rho`
     - значениям advantage `A`
    """
    polLoss = ppo_loss(A, rho)  # PPO лосс

    prob = F.softmax(pi, dim=1)  # получаем распределения
    logProb = F.log_softmax(pi, dim=1)  # логарифм вероятностей распределений
    entLoss = entropyLoss(prob, logProb)  # умножаем и складываем для энтропии

    return polLoss, entLoss
