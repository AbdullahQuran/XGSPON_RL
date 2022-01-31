"""Driver.py
"""
import os
import simpy
from stable_baselines import A2C, PPO2, DQN
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import DBA_env
import numpy
from pickle import GLOBAL
import time
from typing import Counter
import gym
from gym import spaces
from numpy.lib.function_base import average
import Globals
import random
import numpy
from numpy.core.fromnumeric import mean
from gym.utils import seeding
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR, deepq
from stable_baselines.common.cmd_util import make_vec_env
import PPBP
import DBA_env_sub_agent_prediction2
import ModelUtils
import Simulator
# from stable_baselines3 import SAC
# from stable_baselines3 import SAC
import sys
import TraceUtils
import Globals
import Odn
import my_clock
import IO

def loadCsvAsArray(path):
    data = []
    f = open(path, "r")
    for x in f:
        data.append(float(x.replace('\n','')))
    print(path)
    print("avg:" + str(numpy.mean(data)))
    print("max:" + str(numpy.max(data)))
    print("min:" + str(numpy.min(data)))
    print("**************")
    return data


def main():
    onu1Count = 1
    onu2Count = 0
    totalOnus = onu1Count + onu2Count
    rlModel = PPO2.load("./rl_model_master1onu.pkl")
    n_actions = ()
    for x in range(1 * 2):
        n_actions = n_actions + (8,)
    discreteToMultiDiscMap = tuple(numpy.ndindex(n_actions))
    print("enter sizes")
    # for i in range(10000):
    #     s = [[7/Globals.OBSERVATION_MAX],[11/Globals.OBSERVATION_MAX]]
    #     action = rlModel.predict(s, deterministic=True)
    
    for line in sys.stdin:
        temp = line.split(",")
        if (line == "q"):
            break
        if (len(temp) != 2):
            print("wrong input")
            continue
        print("######")
        s = [[int(temp[0])/ Globals.OBSERVATION_MAX], [int(temp[1])/ Globals.OBSERVATION_MAX]]
        action = rlModel.predict(s, deterministic=True)
        # print(action)
        # action = discreteToMultiDiscMap[action[0]]
        # prop = rlModel.action_probability(s)
        print(s)
        print(action)
        # print(prop)





    # for s in states:
    #     action = rlModel.predict(s, deterministic=True)  
    #     # action = rlModel.action_probability(s)
          
    #     print("######")
    #     print(s)
    #     print(action)


def master_main():
    onu1Count = 1
    onu2Count = 1
    totalOnus = onu1Count + onu2Count
    rlModel = PPO2.load("./rl_model_master0..pkl")
    maxObservationURLLC = onu1Count * Globals.URLLC_AB_MIN * 2
    maxObservationEMBB = onu1Count * Globals.EMBB_AB_MIN * 2
    maxObservationVideo = onu2Count * Globals.VIDEO_AB_MIN * 2
    maxObservationIP = onu2Count * Globals.IP_AB_MIN * 2
    state = numpy.empty((1,4), dtype=numpy.float32)
    print("enter sizes")
    # for i in range(10000):
    #     s = [[7/Globals.OBSERVATION_MAX],[11/Globals.OBSERVATION_MAX]]
    #     action = rlModel.predict(s, deterministic=True)
    
    for line in sys.stdin:
        temp = line.split(",")
        if (line == "q"):
            break
        if (len(temp) != 4):
            print("wrong input")
            continue
        print("######")
        # state[0][0] = int(temp[0])/ maxObservationURLLC
        # state[0][1] = int(temp[1])/ maxObservationEMBB
        # state[0][2] = int(temp[2])/ maxObservationVideo
        # state[0][3] = int(temp[3])/ maxObservationIP
        state[0][0] = temp[0]
        state[0][1] = temp[1]        
        state[0][2] = temp[2]
        state[0][3] = temp[3]      
        action = rlModel.predict(state, deterministic=True)
        allocated = action[0] * [maxObservationURLLC * 2, maxObservationEMBB * 2, maxObservationVideo * 2, maxObservationIP * 2] 
        allocated2 = action[0] * [maxObservationURLLC, maxObservationEMBB, maxObservationVideo, maxObservationIP] 
        # print(action)
        # action = discreteToMultiDiscMap[action[0]]
        # prop = rlModel.action_probability(s)
        print(state)
        print(action)
        print(allocated)
        print(allocated2)
        # print(prop)



def predict_main():
    onu1Count = 1
    onu2Count = 1
    totalOnus = onu1Count + onu2Count
    rlModel = PPO2.load("./rl_model2.pkl")
    maxObservationURLLC = onu1Count * Globals.URLLC_AB_MIN * 2
    maxObservationEMBB = onu1Count * Globals.EMBB_AB_MIN * 2
    maxObservationVideo = onu2Count * Globals.VIDEO_AB_MIN * 2
    maxObservationIP = onu2Count * Globals.IP_AB_MIN * 2
    state = numpy.empty((1,2), dtype=numpy.float32)
    print("enter sizes")
    # for i in range(10000):
    #     s = [[7/Globals.OBSERVATION_MAX],[11/Globals.OBSERVATION_MAX]]
    #     action = rlModel.predict(s, deterministic=True)
    
    for line in sys.stdin:
        temp = line.split(",")
        if (line == "q"):
            break
        if (len(temp) != 2):
            print("wrong input")
            continue
        print("######")
        # state[0][0] = int(temp[0])/ maxObservationURLLC
        # state[0][1] = int(temp[1])/ maxObservationEMBB
        # state[0][2] = int(temp[2])/ maxObservationVideo
        # state[0][3] = int(temp[3])/ maxObservationIP
        m = 31
        state[0][0] = int(temp[0])/(m * 5)
        state[0][1] = int(temp[1])/m        
        # state[0][2] = temp[2]
        # state[0][3] = temp[3]      
        action = rlModel.predict(state, deterministic=True)
        allocated = action[0] * [maxObservationURLLC * 2, maxObservationEMBB * 2, maxObservationVideo * 2, maxObservationIP * 2] 
        allocated2 = action[0] * [maxObservationURLLC, maxObservationEMBB, maxObservationVideo, maxObservationIP] 
        # print(action)
        # action = discreteToMultiDiscMap[action[0]]
        # prop = rlModel.action_probability(s)
        print(state)
        print(action[0] * 2 * m)
        print(allocated)
        print(allocated2)
        # print(prop)


def makeModel():
    links = ModelUtils.load_links("three_node")
    print("************************")
    print(links)
    print("************************")
    node_types = ModelUtils.load_node_types("three_node")
    print("************************")
    print(node_types)
    print("************************")
    param = ModelUtils.load_param("three_node")
    print("************************")
    print(param)
    print("************************")
    oltType = param['OLT']['type']
    M = ModelUtils.make_model(links, node_types, param, "three_node")
    M.oltType = oltType
    env = simpy.Environment()
    odn = Odn.ODN(env, M)
    myClock = my_clock.MyClock(Globals.XGSPON_CYCLE)
    M.myClock = myClock
    M.odn = odn
    M.simTime = 5

    # if (M.oltType == 'rl'):
    #     M.rlModel = PPO2.load("rl_model.pkl")
    #     # M.rlModel = SAC.load("rl_model.pkl")
       
    # else:
    #     M.rlModel = None

    M.rlModel = None
    traceUtils = TraceUtils.TU(None, M)
    M.traceUtils = traceUtils
    network = Simulator.setup_network(None, M)
 

    return M, network

def writeToFile(lines):
    file_name = './output/zzz.csv'
    node_file = os.path.join("./", file_name)
    fp1 = IO.open_for_writing(file_name)
    for i in lines:
        fp1.write(i)
    IO.close_for_writing(fp1)

def predict_main2():
    M, network = makeModel()
    env = DBA_env_sub_agent_prediction2.env_dba(network, M)
    env = DummyVecEnv([lambda: env])
    rlModel = PPO2.load("../urllc.pkl")
    obs = env.reset()
    state = None
    # When using VecEnv, done is a vector
    done = [False for _ in range(1)]
    # state = numpy.empty((1,2), dtype=numpy.float32)
    # state[0][0] = 0
    # state[0][1] = 0
    time = 2
    size = 40
    ww = 350
    # bursts, t = PPBP.createPPBPTrafficGen(time/0.01, 5, 0.8, size/12.5, 1, 0.000125 * 10)
    bursts = loadCsvAsArray("../data0.csv")
    idx = random.randint(0, len(bursts))
    # for i in range(10000):
    #     s = [[7/Globals.OBSERVATION_MAX],[11/Globals.OBSERVATION_MAX]]
    #     action = rlModel.predict(s, deterministic=True)
    episodeLength = 5
    maxBurst = numpy.max(bursts)
    queueLengthMax = ww * episodeLength
    lines = []
    for i in range(int(time/0.000125)):
        idx = (idx + 10) % len(bursts)
        b = bursts[idx]
        b = int(b * 630/2/1.9/271) * 10
        line = ""
        line = line + str(obs[0][0][0]) + "," + str(obs[0][0][1]) + ","
        # print("state before: " + str(obs[0][0]))
        obs[0][0][0] = obs[0][0][0] / queueLengthMax
        obs[0][0][1] = obs[0][0][1] / ww
        action, state = rlModel.predict(obs, state=state)
        action = int(action[0] * ww)
        line = line + "," + str(action) + ","
        obs[0][0][0] = obs[0][0][0] * queueLengthMax
        obs[0][0][1] = obs[0][0][1] * ww
        
        obs[0][0][0] = max(obs[0][0][0] + b - action, 0)
        obs[0][0][1] = b
        line = line + "," + str(obs[0][0][0]) + "," + str(obs[0][0][1]) + "\n"
        lines.append(line)
        # print("state after: " + str(obs[0][0]))
        # print("action:" + str(action))
        print("************************")

    writeToFile(lines)



def test2():
    n_actions = (10, 20, 30)
    # n_actions = ()
    # for x in range(3):
    #     n_actions = n_actions + (8,)
    action_space = spaces.MultiDiscrete(n_actions)
    action_space = spaces.Discrete(numpy.prod(n_actions))
    mapping = tuple(numpy.ndindex(n_actions))
    multidiscrete_action = mapping[0]


# master_main()
predict_main2()
# test2()