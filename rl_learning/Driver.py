"""Driver.py
"""


import simpy
import glob
import Globals
import Simulator
import ProgressBar
import TraceUtils
import Odn
import DBA_env
import DBA_env_continuos
import DBA_env_continuos2
import DBA_env_master
import DBA_env_sub_agent
import DBA_env_sub_agent2
import DBA_env_sub_agent_prediction
import DBA_env_sub_agent_prediction2
import DBA_env_dqn
import my_clock
import pandas as pd
from stable_baselines import DQN, PPO2, A2C, ACKTR
import packet
from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines3 import SAC, PPO


def run_sim(M, t, output_dir=None, bar=False, verbose=Globals.VERB_NO):
    """
    Runs network simulation based on the network model 'M'.
    """

    print(" [+] Initialising the simulation...")
    print(" [+] Total simulation time is {:.2f} time units".format(t))

    # initialise simpy environment
    env = simpy.Environment()

    odn = Odn.ODN(env, M)
    myClock = my_clock.MyClock(Globals.XGSPON_CYCLE)
    M.myClock = myClock
    M.odn = odn
    M.simTime = t

    # if (M.oltType == 'rl'):
    #     M.rlModel = PPO2.load("rl_model.pkl")
    #     # M.rlModel = SAC.load("rl_model.pkl")
       
    # else:
    #     M.rlModel = None

    M.rlModel = None
    traceUtils = TraceUtils.TU(None, M)
    M.traceUtils = traceUtils

    # bind the network model 'M' to the SimPy simulation environment
    network = Simulator.setup_network(None, M, verbose=verbose)
    # rlAgent = DBA_env.env_dba(env, network, M)
    # M.rlAgent = rlAgent
    
    print(" [+] Simulations started...")

    # show progress bar
    # if bar:
    #     print("\n [ Running progress bar ]")
    #     progress_bar = ProgressBar.setup(env, t)
    #     progress_bar.run()

    # run the simulation
    # env.run(until=t)

    # rlEnv = DBA_env_continuos2.env_dba(network, M)
    # rlEnv = DBA_env_master.env_dba(network, M)
    # rlEnv = DBA_env_sub_agent_prediction.env_dba(network, M)
    rlEnv = DBA_env_sub_agent_prediction2.env_dba(network, M)
    # rlEnv = DBA_env_sub_agent.env_dba(network, M)
    # rlEnv = DBA_env_sub_agent2.env_dba(network, M)
    # rlEnv = car.Continuous_MountainCarEnv(network, M)
    
    model = A2C(  'MlpPolicy', rlEnv, gamma = 0.4 ,n_steps=32, ent_coef=0, vf_coef=0.25, learning_rate=0.000125,verbose=1).learn(int(t/0.000125), log_interval=100)
    # model = SAC('MlpPolicy', rlEnv, verbose=1).learn(int(t/0.000125), log_interval=1)
    # model = DDPG(  'MlpPolicy', rlEnv, verbose=1).learn(int(t/0.000125), log_interval=10)
    # model = PPO2( 'MlpLstmPolicy', rlEnv, nminibatches = 1, gamma=0.4, n_steps=128, ent_coef=0.01, vf_coef=0.5, learning_rate=0.00025,verbose=1).learn(int(t/0.000125), log_interval=10)
    # model = PPO2( 'MlpPolicy', rlEnv, n_steps=128, ent_coef=0.01, vf_coef=0.5, learning_rate=0.00025,verbose=1).learn(int(t/0.000125), log_interval=10)
    # model = PPO2(  'MlpPolicy', rlEnv, lam=0.8, cliprange= 0.2, gamma=0.98, n_steps=32, noptepochs=20, nminibatches=1, ent_coef=0.0, vf_coef=0.5, learning_rate=0.00001,verbose=1).learn(int(t/0.000125), log_interval=10)
    # model = DQN(MlpPolicy, rlEnv, verbose=1).learn(total_timesteps=int(t/0.000125))
    # model = PPO('MlpPolicy', rlEnv, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1, target_kl=None, tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=1, seed=None, device='auto', _init_setup_model=True).learn(int(t/0.000125), log_interval=10)

    model.save("rl_model.pkl")
    model.save("urllc.pkl")
    model.save("ip.pkl")
    model.save("embb.pkl")
    model.save("video.pkl")
    M.rlEnv = rlEnv
    M.model = model
    if bar:
        print("\n")

    print(" [+] Simulation completed")

    # print summary statistics
    traceUtils.print_stats(network, verbose=verbose)

    # if 'output_dir' is defined, save node names, queue monitor,
    # and traces
    if output_dir is not None:
        # traceUtils.save_node_names(network, output_dir)
        # traceUtils.save_queue_mon(output_dir, network)
        # save generated, forwarded, received, discarded packets
        for trace in ['g', 'f', 'r', 'd', 'u','e','v','i', 'rU', 'rE', 'rV', 'rI', 'rM1', 'gM', 'dU', 'dE', 'dV', 'dI', 'qU', 'qE','qV', 'qI', 'dS']:
            traceUtils.save_node_pkts(network, output_dir, trace)
        output_dir = './output/*total.csv'
        results = []
        glob.glob(output_dir)
        for name in (glob.glob(output_dir)):
            results.append(name)
        print(results)

        #combine all files in the list
        if (len(results) != 0):
            combined_csv = pd.concat([pd.read_csv(f) for f in results ], axis=1, verify_integrity=False)
            combined_csv.to_csv( './output/combined.csv')

    

