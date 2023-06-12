from env.pgg_iter import IterativePGG
from controller.controller import HarvestController

import hydra
import wandb
from omegaconf import OmegaConf
import torch
import numpy as np


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    cfg_wb = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project='pgg-iterative', group='n3k1', config=cfg_wb, job_type=cfg.name, mode='disabled')

    env = IterativePGG(cfg.env)
    print(cfg)
    eog = HarvestController(cfg)
    rewards, mediator_perc_old, med_c_arr = eog.train(env)

    return rewards, mediator_perc_old, med_c_arr


if __name__ == '__main__':
    main()
