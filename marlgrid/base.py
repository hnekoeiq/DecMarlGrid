# Multi-agent gridworld.
# Based on MiniGrid: https://github.com/maximecb/gym-minigrid.


import gym
import numpy as np
import gym_minigrid
from enum import IntEnum
import math

from .objects import Wall, Goal, Lava, EmptySpace, Agent
from .agents import InteractiveAgent
from gym_minigrid.rendering import fill_coords, point_in_rect, downsample, highlight_img

TILE_PIXELS = 32


class ObjectRegistry:
    def __init__(self, objs=[], max_num_objects=1000):
        self.key_to_obj_map = {}
        self.obj_to_key_map = {}
        self.max_num_objects = max_num_objects
        for obj in objs:
            self.add_object(obj)

    def get_next_key(self):
        for k in range(self.max_num_objects):
            if k not in self.key_to_obj_map:
                break
        else:
            raise ValueError("Object registry full.")
        return k

    def __len__(self):
        return len(self.id_to_obj_map)

    def add_object(self, obj):
        new_key = self.get_next_key()
        self.key_to_obj_map[new_key] = obj
        self.obj_to_key_map[obj] = new_key
        return new_key

    def contains_object(self, obj):
        return obj in self.obj_to_key_map

    def contains_key(self, key):
        return key in self.key_to_obj_map

    def get_key(self, obj):
        if obj in self.obj_to_key_map:
            return self.obj_to_key_map[obj]
        else:
            return self.add_object(obj)

    def obj_of_key(self, key):
        return self.key_to_obj_map[key]


class MultiGrid:

    tile_cache = {}

    def __init__(self, shape, obj_reg=None, orientation=0):
        self.orientation = orientation
        if isinstance(shape, tuple):
            self.width, self.height = shape
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)  # w,h
        elif isinstance(shape, np.ndarray):
            self.width, self.height = shape.shape
            self.grid = shape
        else:
            # print(shape)
            raise ValueError("Must create grid from shape tuple or array.")

        if self.width < 3 or self.height < 3:
            raise ValueError("Grid needs width, height >= 3")

        self.obj_reg = ObjectRegistry(objs=[None]) if obj_reg is None else obj_reg

    def __getitem__(self, *args, **kwargs):
        return self.__class__(
            np.ndarray.__getitem__(self.grid, *args, **kwargs),
            obj_reg=self.obj_reg,
            orientation=self.orientation,
        )

    def rotate_left(self, k=1):
        return self.__class__(
            np.rot90(self.grid, k=k),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - k) % 4,
        )

    def slice(self, topX, topY, width, height, rot_k=0):
        """
        Get a subset of the grid
        """
        sub_grid = self.__class__(
            (width, height),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - rot_k) % 4,
        )
        x_min = max(0, topX)
        x_max = min(topX + width, self.width)
        y_min = max(0, topY)
        y_max = min(topY + height, self.height)

        x_offset = x_min - topX
        y_offset = y_min - topY
        sub_grid.grid[
            x_offset : x_max - x_min + x_offset, y_offset : y_max - y_min + y_offset
        ] = self.grid[x_min:x_max, y_min:y_max]
        sub_grid.grid = np.rot90(sub_grid.grid, k=-rot_k)
        sub_grid.width, sub_grid.height = sub_grid.grid.shape

        return sub_grid

    def set(self, i, j, obj):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[i, j] = self.obj_reg.get_key(obj)

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height

        return self.obj_reg.obj_of_key(self.grid[i, j])

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h, obj_type=Wall):
        self.horz_wall(x, y, w, obj_type=obj_type)
        self.horz_wall(x, y + h - 1, w, obj_type=obj_type)
        self.vert_wall(x, y, h, obj_type=obj_type)
        self.vert_wall(x + w - 1, y, h, obj_type=obj_type)

    def __str__(self):
        render = (
            lambda x: "  "
            if x is None or not hasattr(x, "str_render")
            else x.str_render(dir=self.orientation)
        )
        hstars = "*" * (2 * self.width + 2)
        return (
            hstars
            + "\n"
            + "\n".join(
                "*" + "".join(render(self.get(i, j)) for i in range(self.width)) + "*"
                for j in range(self.height)
            )
            + "\n"
            + hstars
        )

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        array[i, j, :] = 0
                    else:
                        array[i, j, :] = v.encode()
        return array

    @classmethod
    def decode(cls, array):
        raise NotImplementedError
        width, height, channels = array.shape
        assert channels == 3
        # objects = {k: WorldObj.decode(k) for k in np.unique(array[:,:,0])}
        # print(objects)
        vis_mask[i, j] = np.ones(shape=(width, height), dtype=np.bool)
        grid = cls((width, height))

    def process_vis(grid, agent_pos):
        mask = np.zeros_like(grid.grid, dtype=np.bool)
        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width - 1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask

    @classmethod
    def render_tile(cls, obj, highlight=False, tile_size=TILE_PIXELS, subdivs=3):
        if obj is None:
            key = (
                tile_size,
                highlight,
            )
        else:
            key = (tile_size, highlight, *obj.encode())

        if key in cls.tile_cache:
            img = cls.tile_cache[key]
        else:
            img = np.zeros(
                shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
            )

            # Draw the grid lines (top and left edges)
            fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
            fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

            if obj != None:
                obj.render(img)

            if highlight:
                highlight_img(img)

            img = downsample(img, subdivs)

            cls.tile_cache[key] = img

        return img

    def render(self, tile_size, highlight_mask=None):

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        for j in range(0, self.height):
            for i in range(0, self.width):
                obj = self.get(i, j)

                tile_img = MultiGrid.render_tile(
                    obj, highlight=highlight_mask[i, j], tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = np.rot90(tile_img, -self.orientation)

        return img


class MultiGridEnv(gym.Env):
    def __init__(
        self,
        agents,
        grid_size=None,
        width=None,
        height=None,
        max_steps=1000,
        see_through_walls=False,
        done_condition=None,
        seed=1337,
    ):

        if grid_size is not None:
            assert width == None and height == None
            width, height = grid_size, grid_size

        if done_condition is not None and done_condition not in ("any", "all"):
            raise ValueError("done_condition must be one of ['any', 'all', None].")
        self.done_condition = done_condition

        self.num_agents = len(agents)
        self.agents = agents

        self.action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(len(agent.actions)) for agent in self.agents)
        )
        self.observation_space = gym.spaces.Tuple(
            tuple(
                gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(agent.view_size, agent.view_size, 3),
                    dtype="uint8",
                )
                for agent in self.agents
            )
        )
        self.reward_range = [(0, 1) for _ in range(len(self.agents))]

        self.window = None

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        self.seed(seed=seed)

        self.reset()

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return self.np_random.randint(0, 2) == 0

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def reset(self):
        for agent in self.agents:
            agent.reset()

        self._gen_grid(self.width, self.height)

        for agent in self.agents:
            # Make sure _gen_grid initialized agent positions
            assert (agent.pos is not None) and (agent.dir is not None)
            # Make sure the agent doesn't overlap with an object
            start_cell = self.grid.get(*agent.pos)
            # assert start_cell is None or start_cell.can_overlap()
            assert start_cell is agent

        self.step_count = 0

        obs = self.gen_obs()
        return obs

    def gen_obs_grid(self, agent):
        topX, topY, botX, botY = agent.get_view_exts()

        grid = self.grid.slice(
            topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
        )

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent.view_size // 2, agent.view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        return grid, vis_mask

    def gen_agent_obs(self, agent):
        grid, vis_mask = self.gen_obs_grid(agent)
        return grid.render(tile_size=agent.view_tile_size)  # ,highlight_mask=~vis_mask)

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        # obs_list = []
        # for agent in self.agents:
        #     grid, vis_mask = self.gen_obs_grid(agent)

        #     obs_list.append({
        #         'image': grid.encode(vis_mask),
        #         'direction': agent.dir,
        #         'mission': agent.mission
        #     })

        return [self.gen_agent_obs(agent) for agent in self.agents]
        # return obs_list

    # def get_obs_render(self, obs, agent, tile_size=TILE_PIXELS//2):
    #     grid, vis_mask = MultiGrid.decode(obs)

    def __str__(self):
        return self.grid.__str__()

    def step(self, actions):
        assert len(actions) == len(self.agents)
        rewards = np.zeros((len(self.agents,)), dtype=np.float)

        self.step_count += 1

        wasteds = []
        done = np.array([agent.done for agent in self.agents], dtype=np.bool)

        for agent_no, (agent, action) in enumerate(zip(self.agents, actions)):
            wasted = False
            if agent.active:

                cur_pos = agent.pos
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)

                if self.agents.config.env == 'pursuit-evasion':

                    bot_pos = agent.pos + np.array([0, -1])
                    bot_cell = self.grid.get(*bot_pos)
                    abov_pos = agent.pos + np.array([0, +1])
                    abov_cell = self.grid.get(*abov_pos)
                    left_pos = agent.pos + np.array([-1, 0])
                    left_cell = self.grid.get(*left_pos)
                    right_pos = agent.pos + np.array([+1, 0])
                    right_cell = self.grid.get(*right_pos)

                    # Checking for surroundig
                    if agent_no > (len(self.agents) - self.agents.config.randAgents - 1):
                        if (isinstance(bot_cell, Agent) or isinstance(bot_cell, Wall)) \
                                and (isinstance(abov_cell, Agent) or isinstance(abov_cell, Wall)) \
                                and (isinstance(left_cell, Agent) or isinstance(left_cell, Wall)) \
                                and (isinstance(right_cell, Agent) or isinstance(right_cell, Wall)):
                            rewards[:len(self.agents) - self.agents.config.randAgents] += \
                                np.array([10] * (len(self.agents) - self.agents.config.randAgents))
                            done[:] = True
                    else:
                        for i in range(-self.agents.config.randAgents, 0):
                            if bot_cell == self.agents[i] or abov_cell == self.agents[i] or \
                                    left_cell == self.agents[i] or right_cell == self.agents[i]:
                                rewards[agent_no] += 0.05

                # Rotate left
                if action == agent.actions.left:
                    agent.dir = (agent.dir - 1) % 4

                # Rotate right
                elif action == agent.actions.right:
                    agent.dir = (agent.dir + 1) % 4

                # Move forward
                elif action == agent.actions.forward:
                    # Under these conditions, the agent can move forward.
                    if (fwd_cell is None) or fwd_cell.can_overlap():

                        # Move the agent to the forward cell
                        agent.pos = fwd_pos

                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent)
                        elif fwd_cell.can_overlap():
                            fwd_cell.agent = agent

                        if cur_cell == agent:
                            self.grid.set(*cur_pos, None)
                        else:
                            cur_cell.agent = None
                    else:
                        wasted = True

                    if isinstance(fwd_cell, Goal):  # No extra wasting logic
                        if self.agents.config.env == 'checkers':
                            if agent_no == 0:
                                rewards[agent_no] += fwd_cell.reward / 10
                            else:
                                rewards[agent_no] += fwd_cell.reward
                            self.grid.set(*fwd_pos, agent)
                        else:
                            rewards[agent_no] += fwd_cell.reward
                            agent.done = True
                            fwd_cell.agent = None
                            fwd_cell.reward = 50

                    if isinstance(fwd_cell, Lava):
                        agent.done = True

                # # Pick up an object
                # elif action == agent.actions.pickup:
                #     if fwd_cell and fwd_cell.can_pickup():
                #         if agent.carrying is None:
                #             agent.carrying = fwd_cell
                #             agent.carrying.cur_pos = np.array([-1, -1])
                #             self.grid.set(*fwd_pos, None)
                #     else:
                #         wasted = True

                # # Drop an object
                # elif action == agent.actions.drop:
                #     if not fwd_cell and agent.carrying:
                #         self.grid.set(*fwd_pos, agent.carrying)
                #         agent.carrying.cur_pos = fwd_pos
                #         agent.carrying = None
                #     else:
                #         wasted = True

                # # Toggle/activate an object
                # elif action == agent.actions.toggle:
                #     if fwd_cell:
                #         wasted = bool(fwd_cell.toggle(agent, fwd_pos))
                #     else:
                #         wasted = True

                # Done action (not used by default)
                # elif action == agent.actions.done:
                #     # dones[agent_no] = True
                #     wasted = True

                else:
                    raise ValueError(f"Environment can't handle action {action}.")
            wasteds.append(wasted)

        if self.step_count >= self.max_steps:
            done[:] = True

        if self.done_condition is None:
            pass
        elif self.done_condition == "any":
            done = any(done)
        elif self.done_condition == "all":
            done = all(done)

        obs = [self.gen_agent_obs(agent) for agent in self.agents]

        wasteds = np.array(wasteds, dtype=np.bool)

        return obs, rewards - [0.01]*len(self.agents), done, wasteds # - [0.001]*len(self.agents)

    @property
    def agent_positions(self):
        return [
            tuple(agent.pos) if agent.pos is not None else None for agent in self.agents
        ]

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        max_tries = int(max(1, min(max_tries, 1e5)))
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)

        agent_positions = self.agent_positions
        for try_no in range(max_tries):
            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            if (
                (self.grid.get(*pos) is None)
                and (pos not in agent_positions)
                and (reject_fn is None or (not reject_fn(pos)))
            ):
                break
        else:
            raise RecursionError("Rejection sampling failed in place_obj.")

        self.grid.set(*pos, obj)
        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, agent, top=None, size=None, rand_dir=True, max_tries=100):
        agent.pos = self.place_obj(agent, top=top, size=size, max_tries=max_tries)
        if rand_dir:
            agent.dir = self._rand_int(0, 4)
        return agent

    def place_agents(self, top=None, size=None, rand_dir=True, max_tries=100):
        for agent in self.agents:
            self.place_agent(
                agent, top=top, size=size, rand_dir=rand_dir, max_tries=max_tries
            )
            if hasattr(self, "mission"):
                agent.mission = self.mission

    def render(
        self,
        mode="human",
        close=False,
        highlight=True,
        tile_size=TILE_PIXELS,
        show_agent_views=True,
        max_agents_per_col=3,
    ):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == "human" and not self.window:
            from gym.envs.classic_control.rendering import SimpleImageViewer

            self.window = SimpleImageViewer()
            # self.window.show(block=False)

        # Compute which cells are visible to the agent
        highlight_mask = np.full((self.width, self.height), False, dtype=np.bool)
        for agent in self.agents:
            xlow, ylow, xhigh, yhigh = agent.get_view_exts()
            if agent.active:
                highlight_mask[
                    max(0, xlow) : min(self.grid.width, xhigh),
                    max(0, ylow) : min(self.grid.height, yhigh),
                ] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size, highlight_mask=highlight_mask if highlight else None
        )
        rescale = lambda X, rescale_factor=2: np.kron(
            X, np.ones((rescale_factor, rescale_factor, 1))
        )

        if show_agent_views:
            agent_no = 0
            cols = []
            rescale_factor = None

            for col_no in range(len(self.agents) // (max_agents_per_col + 1) + 1):
                col_count = min(max_agents_per_col, len(self.agents) - agent_no)
                views = []
                for row_no in range(col_count):
                    tmp = self.gen_agent_obs(self.agents[agent_no])
                    if rescale_factor is None:
                        rescale_factor = img.shape[0] // (
                            min(3, col_count) * tmp.shape[1]
                        )
                    views.append(rescale(tmp, rescale_factor))
                    agent_no += 1

                col_width = max([v.shape[1] for v in views])
                img_col = np.zeros((img.shape[0], col_width, 3), dtype=np.uint8)
                for k, view in enumerate(views):
                    start_x = (k * img.shape[0]) // len(views)
                    start_y = 0  # (k*img.shape[1])//len(views)
                    dx, dy = view.shape[:2]
                    img_col[start_x : start_x + dx, start_y : start_y + dy, :] = view
                cols.append(img_col)
            img = np.concatenate((img, *cols), axis=1)

        if mode == "human":
            self.window.imshow(img)

        return img
