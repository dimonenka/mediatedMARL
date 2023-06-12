from main import main
import hydra
import numpy as np
import torch


@hydra.main(config_path='conf', config_name='config')
def collect_and_print(cfg):
    rew, perc, commit = [], [], []
    seeds = 10

    # for seed in range(seeds):
    seed = 3

    print(f'seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rewards, mediator_perc_old, med_c_arr = main(cfg)
    rew.append(rewards)
    perc.append(mediator_perc_old)
    commit.append(med_c_arr)

    print('FINISHED')
    print(f'REWARD: {np.mean(rew):.4f}\t% MEDIATOR: {np.mean(perc):.4f}\t PI(C)_med: {np.mean(commit):.4f}')


if __name__ == '__main__':
    collect_and_print()

# 5904.9  n=10
# 57.660504  # n=3

