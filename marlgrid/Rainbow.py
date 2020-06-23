import gym
import numpy as np
from enum import IntEnum
from contextlib import contextmanager, ExitStack
from abc import ABC, abstractmethod


from .objects import Agent
from marlgrid.utils.random_process import OUNoise
import random
import torch
from torch.optim import Adam
from replay_buffer import ReplayBuffer2, PrioritizedReplayBuffer
from config import Config
from marlgrid.utils import get_class_attr_val
from model import *


class InteractiveAgent(Agent):
    class DefaultActions(IntEnum):
        left = 0  # Rotate left
        right = 1  # Rotate right
        forward = 2  # Move forward
        # pickup = 3  # Pick up an object
        # drop = 4  # Drop an object
        # toggle = 4  # Toggle/activate an object
        done = 3  # Done completing task

    def __init__(self, view_size, view_tile_size=7, actions=None, **kwargs):
        super().__init__(**{"color": kwargs['color'], **kwargs})
        if actions is None:
            actions = self.DefaultActions

        self.actions = actions
        self.view_size = view_size
        self.view_tile_size = view_tile_size

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(view_tile_size * view_size, view_tile_size * view_size, 3),
            dtype="uint8",
        )

        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.reset()

    def reset(self):
        self.done = False
        self.pos = None
        self.carrying = None
        self.mission = ""

    def render(self, img):
        if not self.done:
            super().render(img)

    @property
    def active(self):
        return not self.done

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        # print(f"DIR IS {self.dir}")
        assert self.dir >= 0 and self.dir < 4
        return np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        return np.add(self.pos, self.dir_vec)

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        dir = self.dir
        # Facing right
        if dir == 0:  # 1
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif dir == 1:  # 0
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif dir == 2:  # 3
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif dir == 3:  # 2
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def sees(self, x, y):
        raise NotImplementedError


class RainbowAgent(InteractiveAgent):
    def __init__(self, config: Config, seed, **kwargs):
        # InteractiveAgent init notably sets self.action_space, self.observation_space
        super().__init__(kwargs['view_size'], color=kwargs['color'])
        self.config = config
        self.is_training = True
        self.seed = seed
        if self.config.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(self.config.max_buff, self.config.alpha, self.config.memory, self.seed)
        else:
            self.buffer = ReplayBuffer2(self.config.max_buff, self.config.memory, self.seed)


        torch.manual_seed(self.config.seed)
        self.model = DQN(self.config.state_shape, self.config.action_dim, config)
        self.target_model = DQN(self.config.state_shape, self.config.action_dim, config)
        if self.config.noisy:
            self.model.update_noisy_modules()
            self.target_model.update_noisy_modules()
        self.target_model.load_state_dict(self.model.state_dict(), strict=False)
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.randomer = OUNoise(self.config.action_dim)
        self.last_obs = None

        self.replay = []
        self.n_episodes = 0

        if self.config.use_cuda:
            self.cuda()
            self.model = DQN(self.config.state_dim, self.config.action_dim, config) .cuda()
            self.target_model = DQN(self.config.state_dim, self.config.action_dim, config).cuda()

    def action_step(self, state, epsilon=None):
        if self.config.noisy:
            self.model.sample_noise()
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state.view(1, self.config.state_shape[0], self.config.state_shape[1], -1)) # self.model.forward(torch.flatten(state))

            if self.config.c51:
                q_value = (q_value * self.model.support).sum(2)

            action = torch.argmax(q_value)
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        if self.config.prioritized_replay:
            s0, a, r, s1, done, weights, indices = self.buffer.sample(self.config.batch_size, self.config.alpha)
        else:
            s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size)

        s0 = torch.FloatTensor(np.float32(s0)).to(self.config.device)
        s1 = torch.FloatTensor(np.float32(s1)).to(self.config.device)
        a = torch.LongTensor(a).to(self.config.device)
        r = torch.FloatTensor(r).to(self.config.device)
        done = torch.FloatTensor(done).to(self.config.device)
        weights = torch.FloatTensor(weights).to(self.config.device)

        if not self.config.c51:
            q_values = self.model(s0)
            target_next_q_values = self.target_model(s1)

            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            if self.config.double:
                next_q_values = self.model(s1)
                next_actions = next_q_values.max(1)[1].unsqueeze(1)
                next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
            else:
                next_q_value = target_next_q_values.max(1)[0]

            expected_q_value = r + (self.config.gamma ** self.config.multi_step) * next_q_value * (1 - done)

            loss = (q_value - expected_q_value.detach()).pow(2)
            # loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
            if self.config.prioritized_replay:
                prios = torch.abs(loss) + 1e-5
            loss = (loss * weights).mean()

        else:
            q_dist = self.model(s0)
            a = a.unsqueeze(1).unsqueeze(1).expand(self.config.batch_size, 1, self.config.num_atoms)
            q_dist = q_dist.gather(1, a).squeeze(1)
            q_dist.data.clamp_(0.01, 0.99)
            target_dist = projection_distribution(current_model=self.model, target_model=self.target_model,
                                                  next_state=s1, reward=r, done=done, support=self.target_model.support,
                                                  offset=self.target_model.offset, args=self.config)

            loss = - (target_dist * q_dist.log()).sum(1)
            if self.config.prioritized_replay:
                prios = torch.abs(loss) + 1e-6
            loss = (loss * weights).mean()

        self.model_optim.zero_grad()
        loss.backward()
        if self.config.prioritized_replay:
            self.buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print('Target Model is updated')

        return loss, None


    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def reset_itself(self):
        self.randomer.reset()

    def decay_epsilon(self):
        self.config.epsilon -= self.config.eps_decay

    def load_weights(self, model_path):
        if model_path is None: return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")


def projection_distribution(current_model, target_model, next_state, reward, done, support, offset, args):
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)

    target_next_q_dist = target_model(next_state)

    if args.double:
        next_q_dist = current_model(next_state)
        next_action = (next_q_dist * support).sum(2).max(1)[1]
    else:
        next_action = (target_next_q_dist * support).sum(2).max(1)[1]

    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(target_next_q_dist.size(0), 1,
                                                               target_next_q_dist.size(2))
    target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
    done = done.unsqueeze(1).expand_as(target_next_q_dist)
    support = support.unsqueeze(0).expand_as(target_next_q_dist)

    Tz = reward + args.gamma * support * (1 - done)
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    target_dist = target_next_q_dist.clone().zero_()
    target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))

    return target_dist
