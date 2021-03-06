"""Driver.py
"""
import threading
import simpy
from rl_learning import PPBP, Globals
import Simulator
import ProgressBar
import TraceUtils
import Odn
import pandas as pd
import os
import glob
from stable_baselines import A2C, PPO2, DQN
import DBA_env
from stable_baselines3 import SAC, PPO
from rl_learning import DBA_env_dqn
from rl_learning import DBA_env_master
from rl_learning import DBA_env_sub_agent2
from rl_learning import DBA_env_sub_agent_prediction2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def loadCsvAsArray(path):
    data = []
    f = open(path, "r")
    for x in f:
        data.append(float(x.replace('\n',''))/4)
    return data

def importModel(path, dict, onuId, tconId):
    dict[onuId][tconId] = PPO2.load(path)

def run_sim(M, t, output_dir=None, bar=False, verbose=Globals.VERB_NO):
    """
    Runs network simulation based on the network model 'M'.
    """

    print(" [+] Initialising the simulation...")
    print(" [+] Total simulation time is {:.2f} time units".format(t))

    # initialise simpy environment
    env = simpy.Environment()

    odn = Odn.ODN(env, M)

    M.odn = odn
    M.simTime = t
    M.traceUtils = None
    M.masterModel = None
    M.myClock = None
    M.subModel = None
    M.urllcModel = None
    M.embbModel = None
    M.videoModel = None
    M.ipModel = None
    # bind the network model 'M' to the SimPy simulation environment
    counterDataFiles = 0
    counterDataCount = 8
    dataFilesName = "data"
    M.B = {}
    for x in M.G.nodes():
        if x not in M.B.keys():
            M.B[x] = {Globals.TCON1_ID: None, Globals.TCON2_ID: None}
        if M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
            M.B[x][Globals.TCON1_ID] = loadCsvAsArray(dataFilesName + str(counterDataFiles) + ".csv")
            counterDataFiles = (counterDataFiles + 1) % counterDataCount
            M.B[x][Globals.TCON2_ID] = loadCsvAsArray(dataFilesName + str(counterDataFiles) + ".csv")
            counterDataFiles = (counterDataFiles + 1) % counterDataCount


    network = Simulator.setup_network(env, M, verbose=verbose)

    if ('rl' in M.oltType):
        # M.rlModel = A2C.load("./rl_learning/rl_model.pkl")
        # M.rlModel = SAC.load("rl_model.pkl")
        # M.rlModel = PPO.load("./rl_learning/rl_model.pkl")
        # masterEnv = DBA_env_master.env_dba(network, M)
        # M.masterEnv = masterEnv
        # subEnv =  DBA_env_sub_agent2.env_dba(network, M)
        # M.subEnv = subEnv
        # M.masterModel = PPO.load("./rl_learning/master.pkl")
        # M.subModel = PPO.load("./rl_learning/sub.pkl")
        rlEnv = DBA_env_sub_agent_prediction2.env_dba(network, M)
        rlEnv = DummyVecEnv([lambda: rlEnv])
        M.rlEnv = rlEnv
        M.rl_models = {}
        threads = []
        
        M.urllcModel = PPO2.load("./rl_learning/urllc.pkl")
        M.embbModel = PPO2.load("./rl_learning/embb.pkl")
        M.videoModel = PPO2.load("./rl_learning/video.pkl")
        M.ipModel = PPO2.load("./rl_learning/ip.pkl")



        for x in M.G.nodes():
            if x not in M.rl_models.keys():
                M.rl_models[x] = {Globals.TCON1_ID: None, Globals.TCON2_ID: None}
            if M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                # th = threading.Thread(target=importModel, args=("./rl_learning/urllc.pkl", M.rl_models, x, Globals.TCON1_ID))
                # threads.append(th)
                # th.start()
                
                # th = threading.Thread(target=importModel, args=("./rl_learning/embb.pkl", M.rl_models, x, Globals.TCON2_ID))
                # threads.append(th)
                # th.start()

                M.rl_models[x][Globals.TCON1_ID] = M.urllcModel
                M.rl_models[x][Globals.TCON2_ID] = M.embbModel
            if M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                # th = threading.Thread(target=importModel, args=("./rl_learning/video.pkl", M.rl_models, x, Globals.TCON1_ID))
                # threads.append(th)
                # th.start()
                
                # th = threading.Thread(target=importModel, args=("./rl_learning/ip.pkl", M.rl_models, x, Globals.TCON2_ID))
                # threads.append(th)
                # th.start()
                M.rl_models[x][Globals.TCON1_ID] = M.videoModel
                M.rl_models[x][Globals.TCON2_ID] = M.ipModel
        for th in threads:
            th.join()

                # M.rl_models[x][Globals.TCON1_ID] = PPO2.load("./rl_learning/video.pkl")
                # M.rl_models[x][Globals.TCON2_ID] = PPO2.load("./rl_learning/ip.pkl")
       

    else:
        M.rlModel = None
         


    traceUtils = TraceUtils.TU(env, M)
    M.traceUtils = traceUtils


    # rlAgent = DBA_env.env_dba(env, network, M)
    # M.rlAgent = rlAgent
    
    print(" [+] Simulations started...")

    # show progress bar
    if bar:
        print("\n [ Running progress bar ]")
        progress_bar = ProgressBar.setup(env, t)
        progress_bar.run()

    # end_event = env.event()
    # end_event.succeed

    env.run(until=t)
    # node_names = list(network.keys())
    # node_names.sort()
    # for node_name in node_names:
    #     node = network[node_name]
    #     type = node.type
    #     if ((type == Globals.ONU_TYPE) or (type == Globals.ONU_TYPE2)):
    #         node.emptyBuffer(t)

    if bar:
        print("\n")

    print(" [+] Simulation completed")

    # print summary statistics
    traceUtils.print_stats(network, verbose=verbose)

    # if 'output_dir' is defined, save node names, queue monitor,
    # and traces
    if output_dir is not None:
        traceUtils.save_node_names(network, output_dir)
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
        combined_csv = pd.concat([pd.read_csv(f) for f in results], axis=1, verify_integrity=False)
        combined_csv.to_csv( "./output/combined.csv")


    

