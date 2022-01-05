import gym
# from baselines import deepq, A2C
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
import DBA_env

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 10
    return is_solved


def main():
    # env = gym.make("CartPole-v0")
    env = DBA_env.env_dba(None,None)
    model = A2C('MlpPolicy', env, verbose=1).learn(1000)
    print(env)
    # act = A2C.learn(
    #     env,
    #     network='mlp',
    #     lr=1e-3,
    #     total_timesteps=100000,
    #     buffer_size=50000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     print_freq=10,
    #     callback=callback
    # )
    # print("Saving model to cartpole_model.pkl")
    model.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()