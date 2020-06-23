import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train',
                        action='store_true', help='train model')
    parser.add_argument('--test', dest='test',
                        action='store_true', help='test model')

    parser.add_argument('--env', default='Cluttered',
                        type=str, help='gym environment')
    parser.add_argument('--agents', default=["Rainbow"], # ["Rainbow", "Rainbow"]
                        type=str, help='RL agents')
    parser.add_argument('--n_agents_per_teams', default=[2], # [1, 1]
                        type=list, help='number of agents in each team')
    parser.add_argument('--teams', default=[1, 1],
                        type=list, help='agents team')
    # parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path',
                        type=str, help='if test, import the model')
    parser.add_argument('--gamma', default=0.9,
                        type=float, help='discount')
    parser.add_argument('--episodes', default=200, type=int)
    parser.add_argument('--frames', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epsilon', default=1.0,
                        type=float, help='noise epsilon')
    parser.add_argument('--epsilon_min', default=0.01,
                        type=float, help='minimum noise epsilon')
    parser.add_argument('--eps_decay', default=1000,
                        type=float, help='epsilon decay')
    parser.add_argument('--max_buff', default=1000,
                        type=int, help='replay buff size')
    parser.add_argument('--output', default='out',
                        type=str, help='result output dir')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='use cuda')
    parser.add_argument('--load_config', type=str,
                        help='load the config from obj file')

    # Algorithm Arguments
    parser.add_argument('--double', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--noisy', action='store_true',
                        help='Enable Noisy Network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='enable prioritized experience replay')
    parser.add_argument('--c51', action='store_true',
                        help='enable categorical dqn')
    parser.add_argument('--multi-step', type=int, default=1,
                        help='N-Step Learning')
    parser.add_argument('--Vmin', type=int, default=-10,
                        help='Minimum value of support for c51')
    parser.add_argument('--Vmax', type=int, default=10,
                        help='Maximum value of support for c51')
    parser.add_argument('--num_atoms', type=int, default=51,
                        help='Number of atom for c51')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--beta_start', type=float, default=0.4,
                        help='Start value of beta for prioritized replay')
    parser.add_argument('--beta_frames', type=int, default=100000,
                        help='End frame of beta schedule for prioritized replay')
    parser.add_argument('--sigma_init', type=float, default=0.4,
                        help='Sigma initialization value for NoisyNet')

    step_group = parser.add_argument_group('step')
    step_group.add_argument('--customize_step', dest='customize_step', action='store_true', help='customize max step per episode')
    step_group.add_argument('--max_steps', default=50, type=int, help='max steps per episode')

    record_group = parser.add_argument_group('record')
    record_group.add_argument('--record', dest='record', action='store_true', help='record the video')
    record_group.add_argument('--record_ep_interval', default=20, type=int, help='record episodes interval')

    checkpoint_group = parser.add_argument_group('checkpoint')
    checkpoint_group.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='use model checkpoint')
    checkpoint_group.add_argument('--checkpoint_interval', default=500, type=int, help='checkpoint interval')

    retrain_group = parser.add_argument_group('retrain')
    retrain_group.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    retrain_group.add_argument('--retrain_model', type=str, help='retrain model path')
    args = parser.parse_args()

    return args