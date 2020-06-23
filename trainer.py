import os
import math

import numpy as np
from gym import wrappers
from config import Config
from marlgrid.utils.logger import TensorBoardLogger
from marlgrid.utils import get_output_folder, time_seq
from marlgrid.utils import beta_scheduler

from collections import deque

import wandb


def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


def normalize(states):
    for state_no in range(len(states)):
        states[state_no] = states[state_no] / 255
        return states


def stack_states(old_stacks, states):
    new_stacks = []
    for agent_no, state in enumerate(states):
        old_stacks[agent_no] = np.concatenate((old_stacks[agent_no], state), axis=2)
        new_stacks.append(np.delete(old_stacks[agent_no], [0, 1, 2], 2))

    return new_stacks

class Trainer:
    def __init__(self, agents, env, config: Config, args, record=False):
        wandb.init(project="marlgrid_27mar", name=config.env + config.task + str(config.seed) + 'DDQN_RB', reinit=True)

        self.agents = agents
        for agent in agents:
            self.agent = agent
            self.agent.epsilon = config.epsilon

            self.agent.state_deque = deque(maxlen=config.multi_step)
            self.agent.reward_deque = deque(maxlen=config.multi_step)
            self.agent.action_deque = deque(maxlen=config.multi_step)

        self.config = config
        self.args = args
        self.env = env
        self.env.seed(config.seed)

        self.agent.is_training = True


        if record:
            os.makedirs('video', exist_ok=True)
            filepath = self.outputdir + '/video/' + config.env + '-' + time_seq()
            env = wrappers.Monitor(env, filepath,
                                   video_callable=lambda episode_id: episode_id % self.config.record_ep_interval == 0)

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)
        self.beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agents.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)

    def train(self, pre_fr=0):
        losses = []
        loss_a = 0
        all_rewards = []
        episode_rewards = 0
        ep_num = 0
        is_win = False
        stacked_states = [np.zeros((self.config.state_shape[1], self.config.state_shape[2], self.config.state_shape[0]))] * len(self.agents)
        stacked_next_states = [np.zeros(np.shape(stacked_states[0]))] * len(self.agents)

        states = normalize(self.env.reset())

        last_fr = 0
        buffer = [None] * len(self.config.agents)
        model = [None] * len(self.config.agents)
        target_model = [None] * len(self.config.agents)

        if self.config.randAgents > 0:
            agents = self.agents[:-self.config.randAgents]
        else:
            agents = self.agents

        for fr in range(pre_fr + 1, self.config.frames + 1):
            epsilon = self.epsilon_by_frame(fr)
            stacked_states = stack_states(stacked_states, states)

            actions = self.agents.action_step(stacked_states, epsilon)
            self.env.render()
            next_states, rewards, dones, _ = self.env.step(actions)
            next_states = normalize(next_states)
            stacked_next_states = stack_states(stacked_next_states, next_states)
            team_learnt = [False] * len(self.config.agents)
            for team, action, state, next_state, reward, done in zip(self.config.teams, actions, stacked_states,
                                                                     stacked_next_states, rewards, dones):
                if self.config.env == 'pursuit-evasion' and self.config.agents[team-1] == 'Random':
                    break

                if team_learnt[team - 1] is False:

                    for agent_id, agent in enumerate(agents):
                        self.agent = agent
                        if self.config.teams[agent_id] == team:
                            # self.agent.state_deque.append(state)
                            # self.agent.reward_deque.append(sum(rewards))
                            # self.agent.action_deque.append(action)
                            #
                            # if len(self.agent.state_deque) == self.config.multi_step or done:
                            #     n_reward = multi_step_reward(self.agent.reward_deque, self.config.gamma)
                            #     n_state = self.agent.state_deque[0]
                            #     n_action = self.agent.action_deque[0]
                            self.agent.buffer.add(state, action, reward, next_state, done)

                            loss_a = 0
                            if self.agent.buffer.size() > self.config.batch_size:
                                loss_a, _ = self.agent.learning(fr)
                                losses.append(loss_a)
                                self.board_logger.scalar_summary('Loss per frame', fr, loss_a)

                    buffer[team - 1] = self.agent.buffer
                    model[team - 1] = self.agent.model
                    target_model[team - 1] = self.agent.target_model
                else:
                    self.agent.buffer = buffer[team - 1]
                    self.agent.model = model[team - 1]
                    self.agent.target_model = target_model[team - 1]

                if fr % self.config.log_interval == 0:
                    self.board_logger.scalar_summary('Reward per episode', ep_num, episode_rewards)

                states = next_states
                team_learnt[team-1] = True
            if sum(rewards) > 0:
                print(sum(rewards))
            episode_rewards += sum(rewards)

            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d, epsilon: %4f" % (
                    fr, np.mean(all_rewards[-20:]), loss_a, ep_num, epsilon))

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agents.save_checkpoint(fr, self.outputdir)

            if all(dones):
                print(episode_rewards)
                states = normalize(self.env.reset())

                all_rewards.append(episode_rewards)

                episode_rewards = 0
                ep_num += 1
                if ep_num == self.config.episodes:
                    avg_reward = 0
                    wandb.join()
                    break
                avg_reward = float(np.mean(all_rewards[-5:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                for agent in self.agents:
                    agent.state_deque.clear()
                    agent.reward_deque.clear()
                    agent.action_deque.clear()

                ######wandb
                x = fr - last_fr
                avg_reward_name = "avg_reward_"+self.config.env + '8_'+ str(self.config.randAgents)+'Rand'+self.config.task +\
                                  'NumAgents:'+ str(len(self.config.teams))
                ep_len_name = "ep_length_"+self.config.env+'8_'+ str(self.config.randAgents) +'Rand'+self.config.task +\
                              'NumAgents:' + str(len(self.config.teams))
                fr_name = "fr_"+self.config.env+'8_'+ str(self.config.randAgents) +'Rand'+self.config.task +\
                          'NumAgents:' + str(len(self.config.teams))

                wandb.log({avg_reward_name: avg_reward, ep_len_name: x, fr_name: fr})
                last_fr = fr

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')