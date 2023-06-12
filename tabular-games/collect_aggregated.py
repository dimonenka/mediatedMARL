from run import train
import numpy as np
import torch
import hydra
np.set_printoptions(suppress=True)


def print_vanilla(mean_step_reward, pick_mediator, policy_agents, policy_mediator, value_agents, value_mediator, it):
    print()
    print()
    print('FINAL RESULT')
    print(f'ITERATION: {it}\n')
    print(f'{mean_step_reward=:.3f}\n{pick_mediator=:.3f}')
    print('------ POLICY AGENTS ------')
    print(format('D', ' >12'), end='')
    print(format('C', ' >7'), end='')
    print(format('MED', ' >9'))
    for i in range(2):
        print(f'AGENT {i}', end=' ')
        print(policy_agents[i])
    print()

    # print('------ V APPROXIMATION AGENTS ------')
    # for i in range(2):
    #     print(f'AGENT {i}: ', end=' ')
    #     print(value_agents[i], end='\n')
    # print('-------------------------------')
    # print()

    print('------ POLICY MEDIATOR ------')
    coalitions = np.array([[0, 1], [1, 0], [1, 1]])
    print(format('AGENT 0', ' >28'), end='')
    print(format('AGENT 1', ' >23'))
    print(format('C     D', ' >28'), end='')
    print(format('C     D', ' >23'))

    i = 0
    for coalition in coalitions:
        print(f'MEDIATOR {coalition}', end='   ')
        print(f'{policy_mediator[i]}   |   {policy_mediator[i+1]}')
        i += 2
    print()

    # print('------ V APPROXIMATION MEDIATOR ------')
    # for i, coalition in enumerate(coalitions):
    #     print(f'MEDIATOR {coalition}: ', end=' ')
    #     print(value_mediator[i], end='\n')
    #     print('-------------------------------')
    #     print()
    # print('###############################')

# def print_nstep(mean_step_reward, pick_mediator, policy_agents, policy_mediator, value_agents, value_mediator, it):


@hydra.main(config_path='conf', config_name='config')
def go(cfg):
    seeds = np.arange(5)
    policy_agents = []
    policy_mediator = []
    value_agents = []
    value_mediator = []
    mean_rew = []
    med = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # mean_step_reward, pick_mediator, policy_agents, policy_mediator, value_agents, value_mediator
        info = train(cfg)
        mean_rew.append(info[0])
        med.append(info[1])
        policy_agents.append(info[2])
        policy_mediator.append(info[3])
        value_agents.append(info[4])
        value_mediator.append(info[5])

    mean_rew = np.mean(mean_rew)
    med = np.mean(med)
    policy_agents = np.array(policy_agents).mean(0)
    policy_mediator = np.array(policy_mediator).mean(0)
    value_agents = np.array(value_agents).mean(0)
    value_mediator = np.array(value_mediator).mean(0)

    if cfg.type.name == 'vanilla':
        print_vanilla(mean_rew, med, policy_agents, policy_mediator, value_agents, value_mediator, cfg.type.env.iterations)
    elif cfg.type.name == 'n_step':
        print_vanilla(mean_rew, med, policy_agents, policy_mediator, value_agents, value_mediator,
                      cfg.type.env.iterations)

    a = 0


if __name__ == '__main__':
    go()
