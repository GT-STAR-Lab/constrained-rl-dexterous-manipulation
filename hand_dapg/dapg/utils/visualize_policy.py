import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
import garage

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--render', is_flag=True, help='render the policy rollouts')
def main(env_name, policy, mode, render):
    init_state = {
        'qpos': np.zeros((36,), dtype=np.float64),
        'qvel': np.zeros((36,), dtype=np.float64),
        'hand_qpos': np.zeros((30,), dtype=np.float64),
        'palm_pos': np.array([-0.00692036, -0.19996033, 0.15038709]),
        # 'obj_pos': np.array([-0.33460906, 0.14691826, 0.035]),
        'obj_pos': np.array([0.1, 0.15, 0.035]),
        # 'target_pos': np.array([ -0.08026819, -0.03606687, 0.25210981])
        'target_pos': np.array([ -0.2, -0.2, 0.25210981])
        # 'target_pos': np.array([ -0.13026819, -0.03606687, 0.25210981])
    }
    e = GymEnv(env_name)
    pi = pickle.load(open(policy, 'rb'))
    # render policy
    e.visualize_policy(pi, num_episodes=500, horizon=e.horizon, mode=mode, render=render)

if __name__ == '__main__':
    main()
