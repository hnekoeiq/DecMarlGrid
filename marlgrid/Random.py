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
from replay_buffer import ReplayBuffer
from config import Config
from marlgrid.utils import get_class_attr_val
from model import DQN, CnnDQN


class InteractiveAgent(Agent):
    class DefaultActions(IntEnum):
        left = 0  # Rotate left
        right = 1  # Rotate right
        forward = 2  # Move forward
        # pickup = 3  # Pick up an object
        # drop = 4  # Drop an object
        # toggle = 4  # Toggle/activate an object
        # done = 3  # Done completing task

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


class RandomAgent(InteractiveAgent):
    def __init__(self, **kwargs):
        # InteractiveAgent init notably sets self.action_space, self.observation_space
        super().__init__(kwargs['view_size'], color=kwargs['color'])
        self.last_obs = None

        self.replay = []
        self.n_episodes = 0

    def action_step(self, obs):
        return self.action_space.sample()
        # raise NotImplementedError

    def save_step(self, *values):
        self.replay.append(values)
        # raise NotImplementedError

    def start_episode(self):
        return self.action_space.sample()
        # raise NotImplementedError

    def end_episode(self):
        return self.action_space.sample()
        # raise NotImplementedError

    @contextmanager
    def episode(self):
        self.start_episode()
        yield self
        self.end_episode()
        self.n_episodes += 1

