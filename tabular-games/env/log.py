import torch
import numpy as np
import wandb

np.set_printoptions(precision=4, floatmode='fixed')


def pd_log(controller, reward, pick_mediator):
    agent_is = torch.tensor([[1, 0], [0, 1]]).long()
    dummy_state = [1, 0]

    policy_agents = []
    policy_mediator = []
    value_agents = []
    value_mediator = []
    # state = torch.tensor([[1, 0]], device=controller.device, dtype=controller.dtype)

    mean_step_reward = np.mean(np.concatenate(reward))
    std_step_reward = np.std(np.concatenate(reward))
    pick_mediator = np.mean(np.concatenate(pick_mediator))

    print(f'{mean_step_reward=:.3f}\n{std_step_reward=:.3f}\n{pick_mediator=:.3f}')
    print(f'ENTROPY_AGENT: {controller.agents[0].entropy_coef:.3f}, ENTROPY_MEDIATOR: {controller.mediator.entropy_coef:.3f}',
          end='\n\n')

    wandb.log({
        'step_reward_mean': mean_step_reward,
        'step_reward_std': std_step_reward,
        'pick_mediator': pick_mediator,
        'entropy_agent': controller.agents[0].entropy_coef,
        'entropy_mediator': controller.mediator.entropy_coef,
    })

    print('------ POLICY AGENTS ------')
    print(format('D', ' >12'), end='')
    print(format('C', ' >7'), end='')
    print(format('MED', ' >9'))
    for i in range(controller.n_agents):
        policy = controller.agents[i].get_policy(controller._tensorify(dummy_state))
        probs = policy.probs.detach().numpy()
        print(f'AGENT {i}', end=' ')
        print(probs[0])
        policy_agents.append(probs[0])
    print()

    # print('------ V APPROXIMATION AGENTS ------')
    # for i in range(controller.n_agents):
    #     with torch.no_grad():
    #         v = controller.agents[i].critic(controller._tensorify([dummy_state]))
    #     print(f'AGENT {i}: ', end=' ')
    #     print(v.detach().cpu().numpy()[0], end='\n')
    #     value_agents.append(v.detach().cpu().numpy()[0])

    print('-------------------------------')
    print()

    print('------ POLICY MEDIATOR ------')
    coalitions = torch.tensor([[0, 1], [1, 0], [1, 1]]).long()
    print(format('AGENT 0', ' >28'), end='')
    print(format('AGENT 1', ' >23'))
    print(format('C     D', ' >28'), end='')
    print(format('C     D', ' >23'))
    for coalition in coalitions:
        probs = []
        for agent_i in agent_is:
            policy = controller.mediator.get_policy(controller._tensorify([dummy_state, coalition, agent_i]))
            probs.append(policy.probs.detach().numpy())

        print(f'MEDIATOR {coalition.numpy()}', end='   ')
        print(f'{probs[0][0]}   |   {probs[1][0]}')
        policy_mediator.append(np.array(probs))


    # print('------ V APPROXIMATION MEDIATOR ------')
    # for coalition in coalitions:
    #     with torch.no_grad():
    #         v = controller.mediator.critic(controller._tensorify([dummy_state, coalition]))
    #     print(f'MEDIATOR {coalition.numpy()}: ', end=' ')
    #     print(v.detach().cpu().numpy()[0], end='\n')
    #     value_mediator.append(v.detach().cpu().numpy()[0])
    #     print('-------------------------------')
    #     print()
    # print('###############################')

    return mean_step_reward, pick_mediator, np.array(policy_agents), np.concatenate(policy_mediator).squeeze(1), \
           np.array(value_agents), np.array(value_mediator)


def rpd_log(controller, reward, pick_mediator):
    agent_is = torch.tensor([[1, 0], [0, 1]]).long()
    dummy_state = [0, 0]
    state = torch.tensor([[1, 0], [0, 1]], device=controller.device, dtype=controller.dtype)

    mean_step_reward = np.mean(np.concatenate(reward))
    std_step_reward = np.std(np.concatenate(reward))
    pick_mediator = np.mean(np.concatenate(pick_mediator))

    print(f'{mean_step_reward=:.3f}\n{std_step_reward=:.3f}\n{pick_mediator=:.3f}')
    print(f'ENTROPY_AGENT: {controller.agents[0].entropy_coef:.3f}, ENTROPY_MEDIATOR: {controller.mediator.entropy_coef:.3f}',
          end='\n\n')

    wandb.log({
        'step_reward_mean': mean_step_reward,
        'step_reward_std': std_step_reward,
        'pick_mediator': pick_mediator,
        'entropy_agent': controller.agents[0].entropy_coef,
        'entropy_mediator': controller.mediator.entropy_coef,
    })

    print('------ POLICY AGENTS ------')
    print(format('D', ' >12'), end='')
    print(format('C', ' >7'), end='')
    print(format('MED', ' >9'))
    for i in range(controller.n_agents):
        policy = controller.agents[i].get_policy(controller._tensorify(dummy_state))
        probs = policy.probs.detach().numpy()
        print(f'AGENT {i}', end=' ')
        print(probs[0])
    print()

    for s in range(2):
        print(f'\t\tSTATE: {s}')
        print('------ POLICY MEDIATOR ------')
        coalitions = torch.tensor([[0, 1], [1, 0], [1, 1]]).long()
        print(format('AGENT 0', ' >28'), end='')
        print(format('AGENT 1', ' >23'))
        print(format('C     D', ' >28'), end='')
        print(format('C     D', ' >23'))
        for coalition in coalitions:
            probs = []
            for agent_i in agent_is:
                policy = controller.mediator.get_policy(controller._tensorify([state[s], coalition, agent_i]))
                probs.append(policy.probs.detach().numpy())

            print(f'MEDIATOR {coalition.numpy()}', end='   ')
            print(f'{probs[0][0]}   |   {probs[1][0]}')

        print('------ V APPROXIMATION MEDIATOR ------')
        for coalition in coalitions:
            with torch.no_grad():
                v = controller.mediator.critic(controller._tensorify([state[s], coalition]))
            print(f'MEDIATOR {coalition.numpy()}: ', end=' ')
            print(v.detach().cpu().numpy()[0], end='\n')

        print('-------------------------------')
        print()
    print('###############################')


def iter_log(controller, reward, pick_mediator):
    agent_is = torch.tensor([[1, 0], [0, 1]]).long()
    states = torch.tensor([[1, 0, 0], [0, 1, 1]], device=controller.device, dtype=controller.dtype)
    coalition_agent = torch.tensor([[2, 2]]).long()
    coalition_mediator = torch.tensor([[0, 1], [1, 0], [1, 1]]).long()

    policy_agents = []
    policy_mediator = []
    value_agents = [1]
    value_mediator = []

    mean_step_reward = np.mean(np.concatenate(reward))
    std_step_reward = np.std(np.concatenate(reward))
    pick_mediator = np.mean(np.concatenate(pick_mediator))
    
    print(f'{mean_step_reward=:.3f}\n{std_step_reward=:.3f}\n{pick_mediator=:.3f}')
    print(f'ENTROPY_AGENT: {controller.agents[0].entropy_coef:.3f}, ENTROPY_MEDIATOR: {controller.mediator.entropy_coef:.3f}',
          end='\n\n')

    wandb.log({
        'step_reward_mean': mean_step_reward,
        'step_reward_std': std_step_reward,
        'pick_mediator': pick_mediator,
        'entropy_agent': controller.agents[0].entropy_coef,
        'entropy_mediator': controller.mediator.entropy_coef,
    })

    print('------ POLICY ------')

    for i, state in enumerate(states):
        print(f'STATE {i}')
        for med_avail in coalition_agent:
            print(f'\tCOALITION: {med_avail.detach().numpy()}')
            print(format('D', ' >12'), end='')
            print(format('C', ' >7'), end='')
            print(format('MED', ' >9'))
            for i, agent in enumerate(controller.agents):
                obs = controller._tensorify([state, med_avail[i]])
                policy = agent.get_policy(obs)
                probs = policy.probs.detach().numpy()
                print(f'AGENT {i}', end=' ')
                print(probs[0])
                policy_agents.append(probs[0])
        print()

    for i, state in enumerate(states):
        print(f'STATE {i}')
        for med_avail in coalition_mediator:
            print(f'\tCOALITION: {med_avail.detach().numpy()}')
            who = torch.where(med_avail == 1)[0]
            for i, agent_i in enumerate(agent_is):
                obs = controller._tensorify([state, med_avail, agent_i])
                policy = controller.mediator.get_policy(obs)
                probs = policy.probs.detach().numpy()
                policy_mediator.append(np.array(probs))
                print(f'MEDIATOR FOR AGENT: {np.argmax(agent_i.detach().numpy())}', end=' ')
                if i in who:
                    print(probs[0])
                else:
                    print()
        print()

    # print('------ V APPROXIMATION ------')
    # for i, agent in enumerate(controller.agents):
    #     ggg_ = controller._tensorify(state)
    #     v_ag = agent.critic(ggg_).item()
    #     print(f'AGENT {i}: ', end=' ')
    #     print(v_ag, end='\n')

    # print()
    # for coal in coalition_mediator:
    #     print(f'\tCOALITION: {coal.detach().numpy()}')
    #     with torch.no_grad():
    #         ggg = controller._tensorify((state, coal))
    #         v_med = controller.mediator.critic(ggg)
    #     print(f'MEDIATOR: ', end=' ')
    #     print(v_med.detach().cpu().numpy()[0], end='\n')
    #     value_mediator.append(v_med.detach().cpu().numpy()[0])

    print('\/\/\/\/\/\/\/\/\/', end='\n\n')
    # np.concatenate(policy_mediator).squeeze(1)
    return mean_step_reward, pick_mediator, np.array(policy_agents), policy_mediator, \
           np.array(value_agents), np.array(value_mediator)


def pgg_log(controller, reward, pick_mediator, actions):
    agent_is = torch.tensor([[1, 0], [0, 1]]).long()
    dummy_state = [0, 0, 0]
    state = torch.tensor([[0, 0, 0]], device=controller.device, dtype=controller.dtype)

    reward = np.concatenate(reward)
    mean_step_reward = np.mean(reward)
    mean_reward_per_agent = np.mean(reward / controller.n_agents)
    std_step_reward = np.std(reward)
    pick_mediator = np.mean(np.concatenate(pick_mediator))
    committed_rate = np.sum(actions) / len(actions)

    print(f'{mean_step_reward=:.3f}'
          f'\n{std_step_reward=:.3f}'
          f'\n{mean_reward_per_agent=:.3f}'
          f'\n{pick_mediator=:.3f}'
          f'\n{committed_rate=:.3f}')
    print(f'ENTROPY_AGENT: {controller.agents[0].entropy_coef:.3f},'
          f' ENTROPY_MEDIATOR: {controller.mediator.entropy_coef:.3f}',
        end='\n\n')

    wandb.log({
        'step_reward_mean': mean_step_reward,
        'step_reward_std': std_step_reward,
        'step_reward_per_agent': mean_reward_per_agent,
        'pick_mediator': pick_mediator,
        'committed_rate': committed_rate,
        'entropy_agent': controller.agents[0].entropy_coef,
        'entropy_mediator': controller.mediator.entropy_coef,
    })

    #
    print('------ POLICY AGENTS ------')
    for i in range(controller.n_agents):
        policy = controller.agents[i].get_policy(controller._tensorify(dummy_state))
        probs = policy.probs.detach().numpy()
        print(f'AGENT {i}', end=' ')
        print(probs[0])
    print()

    print('------ POLICY MEDIATOR ------')
    # coalitions = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 0, 0]]).long()
    coalitions = torch.tensor([[0, 1], [1, 0], [1, 1]]).long()
    for coalition in coalitions:
        probs = []
        for agent_i in agent_is:
            policy = controller.mediator.get_policy(controller._tensorify([dummy_state, coalition, agent_i]))
            probs.append(policy.probs.detach().numpy())

        print(f'MEDIATOR {coalition.numpy()}', end='   ')
        print(f'{probs}')

    print()
    # # print('------ V APPROXIMATION ------')
    # # for i, agent in enumerate(controller.agents):
    # #     ggg_ = controller._tensorify((state, [2]))
    # #     v_ag = agent.critic(ggg_).item()
    # #     print(f'AGENT {i}: ', end=' ')
    # #     print(v_ag, end='\n')
    # #
    # # print()
    # # for coal in coalition_mediator:
    # #     print(f'\tCOALITION: {coal.detach().numpy()}')
    # #     with torch.no_grad():
    # #         ggg = controller._tensorify((state, coal))
    # #         v_med = controller.mediator.critic(ggg)
    # #     print(f'MEDIATOR: ', end=' ')
    # #     print(v_med.detach().cpu().numpy()[0], end='\n')
    #
    # print('\/\/\/\/\/\/\/\/\/', end='\n\n')