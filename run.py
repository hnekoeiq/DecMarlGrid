import argparse
from tester import Tester
from config import Config
from marlgrid.DDQN import DDQNAgent
from marlgrid.Rainbow import RainbowAgent
from marlgrid.Random import RandomAgent
from trainer import Trainer
import marlgrid.agents as Agents
import marlgrid.envs.cluttered as cluttered
import marlgrid.envs.doorkey as doorkey
import marlgrid.envs.empty as empty
from marlgrid.envs import checkers, pursuit_evasion
from arguments import get_args
import random


args = get_args()
config = Config()
config.env = 'checkers' # 'pursuit-evasion' #
config.task = 'Dec'
if config.env is 'checkers':
    config.max_steps = 150
    if config.task == 'SP':
        config.agents = ['Rainbow']
        config.n_agents_per_teams = [2]
        config.teams = [1, 1]
        config.randAgents = 0
        config.episodes = 800
    elif config.task == 'Dec':
        config.agents = ['Rainbow', 'Rainbow']
        config.n_agents_per_teams = [1, 1]
        config.teams = [1, 2]
        config.randAgents = 0
        config.episodes = 800
elif config.env is 'pursuit-evasion':
    config.max_steps = 150
    if config.task == 'SP':
        config.agents = ['Rainbow', 'Random']
        config.n_agents_per_teams = [4, 2]
        config.teams = [1, 1, 1, 1, 2, 2]
        config.randAgents = 2
        config.episodes = 100
    elif config.task == 'Dec':
        config.agents = ['Rainbow', 'Rainbow', 'Rainbow', 'Rainbow', 'Random']
        config.n_agents_per_teams = [1, 1, 1, 1, 2] # [2, 1] #
        config.teams = [1, 2, 3, 4, 5, 5] # [1, 1, 2] #
        config.randAgents = 2
        config.episodes = 200

config.memory = False

config.gamma = args.gamma
config.epsilon = args.epsilon
config.epsilon_min = 0.01
config.eps_decay = args.eps_decay
config.tau = 0.001
config.frames = args.frames
config.use_cuda = args.cuda
config.learning_rate = 1e-3
config.learning_rate_actor = 1e-4
config.max_buff = 10000
config.update_tar_interval = 100
config.batch_size = 64
config.print_interval = 500
config.log_interval = 200
config.win_reward = 810
config.win_break = False
config.seed = random.randint(1, 10000)
config.sims = 5

# Algorithm parameters
config.noisy = False # args.noisy
config.dueling = False # args.dueling
config.c51 = False # args.c51
config.prioritized_replay = True # args.prioritized_replay
config.double = True # args.double
config.multi_step = 1 # args.multi_step
config.multi_fr = 3

config.Vmin = args.Vmin
config.Vmax = args.Vmax
config.num_atoms = args.num_atoms
config.alpha = args.alpha

config.beta_start = args.beta_start
config.beta_start = args.beta_frames
config.sigma_init = args.sigma_init
# Environment parameters
config.action_dim = 3
config.state_shape = (3*config.multi_fr, 35, 35)
config.state_dim = config.state_shape[1]
config.view_size = 5

if __name__ == '__main__':

    if args.cuda:
        config.device = 'cuda'
    else:
        config.device = 'cpu'
    TestRLAgents = []
    TestRLAgents_Teams = []
    agent_colors = ['blue', 'red', 'yellow', 'green', 'worst', 'grey', 'purple', 'olive', 'orange']
    seed = config.seed
    agent_count = 0
    for team_no, agent in enumerate(config.agents):
        seed += 1
        for agent_no in range(config.n_agents_per_teams[team_no]):
            if agent is "DDQN":
                TestRLAgents.append(DDQNAgent(config, view_size=config.view_size,
                                              color=agent_colors[team_no], seed=seed))

            elif agent is "Rainbow":
                TestRLAgents.append(RainbowAgent(config, view_size=config.view_size,
                                                 color=agent_colors[agent_count], seed=seed))

            elif agent is 'Random':
                TestRLAgents.append(RandomAgent(view_size=config.view_size, color=agent_colors[agent_count]))
            agent_count += 1

    agents = Agents.IndependentLearners(
        TestRLAgents,
        config=config
    )
    env = []
    if config.env == 'empty':
        env = empty.EmptyMultiGrid(agents, grid_size=7, max_steps=config.max_steps)
    elif config.env == 'cluttered':
        env = cluttered.ClutteredMultiGrid(agents, grid_size=9, n_clutter=7, randomize_goal=True)
    elif config.env == 'doorkey':
        env = doorkey.DoorKeyEnv(agents, grid_size=7)
    elif config.env == 'checkers':
        env = checkers.CheckersMultiGrid(agents, grid_size=8, max_steps=config.max_steps)
    elif config.env == 'pursuit-evasion':
        env = pursuit_evasion.PursuitEvasionMultiGrid(agents, grid_size=8, max_steps=config.max_steps)

    if args.train:
        for sim in range(config.sims):
            config.seed = config.seed + sim
            trainer = Trainer(agents, env, config, args=args)
            trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agents, env, args.model_path)
        tester.test()