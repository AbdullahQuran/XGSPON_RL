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
from stable_baselines3 import SAC, PPO
import copy
import PPBP

class env_dba(gym.Env):
    "environment to simulate the DBA in XGSPON with mobile fronhaul ONUs and FTTx onus"

    def __init__(self, network, M):
    
        self.myClock = M.myClock
        self.network = network
        self.M = M
        self.onu_type1_count = 0
        self.onu_type2_count = 0
        self.onus_count = 0
        self.onu1_array = []
        self.onu2_array = []
        self.reward_stat = []
        self.reward_mat = []
        self.episode = []
        self.save_reward = []
        self.save_eps = []
        self.loadDict = {} 
        node_names = list(network.keys())
        self.prevAdjustTime = -1
        self.prevStep = -1
        self.maxEpisodeLength = 100
        for node_name in node_names:
            node = network[node_name]
            if node.type == Globals.ONU_TYPE:
                self.onu1_array.append(node)
                self.onu_type1_count = self.onu_type1_count + 1
            elif node.type == Globals.ONU_TYPE2:
                self.onu2_array.append(node)
                self.onu_type2_count = self.onu_type2_count + 1
            elif node.type == Globals.OLT_TYPE:
                self.olt = node
        

        self.Bursts = []
        self.pkt_rate = M.G.nodes[node_name][Globals.NODE_PKT_RATE_KWD]

        self.createWorkloadGenerators(40, Globals.SIM_TIME, self.pkt_rate[1])

        self.onu_type1_count = 3
        self.onus_count = self.onu_type2_count + self.onu_type1_count
        self.serviceCount = Globals.SERVICE_COUNT
        self.scaleFactor = 1 
        self.maxObservationURLLC = Globals.OBSERVATION_MAX
        self.maxObservationEMBB = Globals.OBSERVATION_MAX
        self.maxObservationVideo = Globals.OBSERVATION_MAX
        self.maxObservationIP =  Globals.OBSERVATION_MAX
        self.maxObservationURLLC = self.maxObservationURLLC * self.scaleFactor
        self.maxObservationEMBB = self.maxObservationEMBB * self.scaleFactor
        self.maxObservationVideo = self.maxObservationVideo * self.scaleFactor
        self.maxObservationIP = self.maxObservationIP * self.scaleFactor
        self.accuracy = 0.125 * Globals.URLLC_AB_MIN  
        self.maxObservationQueueLength = self.maxEpisodeLength * self.maxBurstSize
        self.maxObservationReport = self.maxBurstSize 
        self.maxObservation = self.maxObservationURLLC
        self.minObservation = 0
        self.allocationStep = 1
        # self.action_space = spaces.MultiDiscrete([8]* self.serviceCount)
        # numpy.array([0.0,0.0,0.0,0.0]), numpy.array([+1.0,+1.0,+1.0,1.0])
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1, ),  dtype=numpy.float32)

        # observation_space = [URRLC_SUM, EMBB_SUM, VIDEO_SUM, IP_SUM]
        self.observation_space = spaces.Box(low=self.minObservation, high=self.maxObservationQueueLength, shape=(1, 3), dtype=numpy.float32)
        # self.state = numpy.array([numpy.repeat(0, self.serviceCount)], numpy.float32)
        self.state = numpy.empty(3, dtype=numpy.float32)
        self.allocationArray = numpy.empty(3, dtype=numpy.float32)
        # self.allocationArray = numpy.array([numpy.repeat(0, self.serviceCount)], numpy.float32)
        self.counter = 0
        self.burstCounter = 0
        self.selectedAction = None
        self.URLLC_reward_violation_counter = 0
        self.EMBB_reward_violation_counter = 0
        self.IP_reward_violation_counter = 0
        self.Video_reward_violation_counter = 0
        self.reward_violation_counter = 0
        self.episodeCounter = 0
        self.eps_counter = 0
        self.reward_sum = 0
        self.prediction_violation_counter_urllc = 0
        self.prediction_violation_counter_embb = 0
        self.prediction_violation_counter_video = 0
        self.prediction_violation_counter_ip =0
        self.trainingSizes = [20, 80]
        self.trainingStat = []
        self.predictionAccuracyViolation = 0
        self.appendHeader()
        self.seed()

        
    def createWorkloadGenerators(self, average, time, step):
        self.Bursts = []
        B, t = PPBP.createPPBPTrafficGen(time/0.01, 5, 0.8, average/12.5, 1, step * 10)
        self.Bursts.append(B)
        self.Bursts.append(t)
        self.maxBurstSize = numpy.max(self.Bursts[0])
        self.minBurstSize = numpy.min(self.Bursts[0])
        self.meanBurstSize = numpy.mean(self.Bursts[0])
        self.burstLen = len(self.Bursts[0])
        print ("max:" + str(numpy.max(self.Bursts[0])))
        print ("min:" + str(numpy.min(self.Bursts[0])))
        print ("mean:" + str(numpy.mean(self.Bursts[0])))
        print ("len:" + str(len(self.Bursts[0])))
        


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):

        currentSize = self.Bursts[0][self.counter % self.burstLen]
        self.state = self.deNormalizeState(self.state)
        requested = copy.deepcopy(self.state)
        self.multiDiscreteToAllocation(self.prevAction)
        
        
        done1 = False
        if (currentSize  + 10 < self.allocationArray[0]) or (currentSize - 10  > self.allocationArray[0]):
            self.predictionAccuracyViolation = self.predictionAccuracyViolation + 1
        self.state[0] = max(currentSize + self.state[0] - self.allocationArray[0], 0)
        if (self.counter == 0):
            self.state[1] = 0  
            self.state[2] = 0  
        else:    
            self.state[1] = 0  
            self.state[2] = 0  
        # self.updateStateRandom(self.state)
        self.state = self.normalizeState(self.state)
        # state seen by agent: depends on 1- agent action in prevoius state(grants) 2- generation in current state(requests) so the aget can calculate the reward
        # curent state seen by agent: 1- generation at time t, 2- grant at time t-1  

        done2 = False
        reward = 0
        self.counter = self.counter + 1
        self.reward_sum = self.reward_sum + reward
        
        # if (self.counter >= self.maxEpisodeLength):
        #     done2 = True
        #     if (requested[0] > 1 * self.meanBurstSize):
        #         reward = 0
                
        #     else:
        #         reward = 1
        
        if ((abs(currentSize  - self.allocationArray[0])  >  10)):
            reward = 0  
        else:
            reward = 1

        done = done1 or done2
        self.appendStat(requested, self.allocationArray, action, reward, done, [], self.reward_sum)
        Globals.appendList(self.reward_mat, reward)
        Globals.appendList(self.episode, self.myClock.now/0.000125)
        self.reward_stat = [self.save_eps, self.save_reward]
        
        if done == True:
            Globals.appendList(self.save_reward, self.reward_sum)
            self.eps_counter =  self.eps_counter + 1
            Globals.appendList(self.save_eps, self.eps_counter)

        self.myClock.inc()
        self.state = numpy.array(self.state, dtype=numpy.float32)
        self.prevState = copy.deepcopy(self.state)
        self.prevAction = action 
        return self.state, reward, done, {}


    def reset(self):
        # print("reset is called")
        # if (self.episodeCounter % 100 == 0):
        #     print (self.URLLC_reward_violation_counter)
        #     print (self.EMBB_reward_violation_counter)
        #     print (self.IP_reward_violation_counter)
        #     print (self.Video_reward_violation_counter)
        
        self.episodeCounter = self.episodeCounter + 1
        self.state = self.resetState(self.state)
        self.prevAction = 0
        # self.updateStateRandom(self.state)
        self.state = self.normalizeState(self.state)
        self.counter = 0
        self.reward_sum = 0
        self.reward_violation_counter = 0
        self.prediction_violation_counter_urllc = 0
        self.prediction_violation_counter_embb = 0
        self.prediction_violation_counter_video = 0
        self.prediction_violation_counter_ip =0
        self.state = numpy.array(self.state, dtype=numpy.float32)
        return self.state


    def actionToVal(self, action, maxValue):
        action = numpy.round(action * 2, 2)
        return int(action * maxValue)
        # if action == 0:
        #     return int(0.25 * maxValue)
        # if action == 1:
        #     return int(0.5 * maxValue)
        # if action == 2:
        #     return int(0.75 * maxValue)
        # if action == 3:
        #     return int(maxValue)
        # if action == 4:
        #     return int(1.25 * maxValue)
        # if action == 5:
        #     return int(1.5 * maxValue)
        # if action == 6:
        #     return int(1.75 * maxValue)
        # if action == 7:
        #     return int(2 * maxValue)
            

    def multiDiscreteToAllocation (self, rlAction):
        for i in range(3):
            self.allocationArray[i] = self.actionToVal(rlAction, Globals.URLLC_AB_MIN)
        self.allocationArray = numpy.array(self.allocationArray, dtype=numpy.float32)


    def deNormalizeState(self,state):
        max = self.maxBurstSize * self.maxEpisodeLength
        for i in range(self.onu_type1_count):
            state[i] = state[i] * max
        return state

    def normalizeState(self, state, maintainBounds = True):
        max = self.maxBurstSize * self.maxEpisodeLength
        for i in range(self.onu_type1_count):
            if (maintainBounds):
                if (state[i] < self.minObservation):
                    state[i] = self.minObservation
                if (state[i] > max):
                    state[i] = max
            state[i] = state[i]/max
            
        return state

    def updateStateRandom(self,state):
        for i in range(self.onu_type1_count):
            state[i] = int(self.np_random.uniform(low=0, high=self.maxObservationURLLC)) 

        return state


    def resetState(self, state):
        # state = numpy.array([numpy.repeat(0, self.serviceCount)], numpy.float32)
        state = numpy.empty(self.onu_type1_count, dtype=numpy.float32)
        for i in range(self.onu_type1_count):
            state[i] = 0
       
        return state

    # Reset the state of the environment to an initial state
    
    def render(self, mode='human', close=False):
    # Render the environment to the screen
        pass


    def getTotal(self, matrix):
        totalURLLC = 0
        totalEMBB = 0
        totalVideo = 0
        totalIP = 0
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                totalURLLC = totalURLLC + matrix[x][Globals.TCON1_ID]
                totalEMBB = totalEMBB + matrix[x][Globals.TCON2_ID]
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                totalVideo = totalVideo + matrix[x][Globals.TCON1_ID]
                totalIP = totalIP + matrix[x][Globals.TCON2_ID]
        return totalURLLC+totalEMBB+totalVideo+totalIP

    def computeExponReward(self, x):
        y = 1.05**(-2* abs(x))
        return y


    def computeReward(self):
        state = copy.deepcopy(self.state)
        allocated = copy.deepcopy(self.allocationArray)
        requestedSum = numpy.sum(state)
        allocatedSum = numpy.sum(allocated)
        difference = self.state - self.allocationArray
        done = False
        violation = False
        # case 1 
        penalty = 0
        
        for row in range(len(state)):
            #or (difference[row] > 0 and difference[row] <= self.accuracy)
            if (difference[row] < 0 and abs(difference[row]) <= self.accuracy):
                difference[row] = 0
            elif (difference[row] > 0 and abs(difference[row]) <= Globals.LENGTH_TRESHOLD_URLLC):
                difference[row] = 0
            elif allocated[row] == 2 * Globals.URLLC_AB_MIN and state[row] >= 2 * Globals.URLLC_AB_MIN:
                # the queue size is so large. Should grant the highest possible val
                difference[row] = 0   
            else:
                violation = True

        penalty = sum(numpy.abs(difference))
        reward = self.computeExponReward(penalty)
        if not violation:
            reward = 1

        Globals.appendList(self.reward_mat, reward)
        Globals.appendList(self.episode, self.myClock.now/0.000125)

        self.reward_stat = [self.save_eps, self.save_reward]

        return reward, done
                    
       

    def get_reward(self):
        return self.reward_stat

    def appendHeader(self):
        line = ""
        for onu in self.onu1_array: 
            line = line + "," + onu.name + "URLLC" + "," + onu.name + "EMBB"
        for onu in self.onu2_array: 
            line = line + "," + onu.name + "Video" + "," + onu.name + "IP"
        line = line + ","
        
        for onu in self.onu1_array: 
            line = line + "," + onu.name + "URLLC_g" + "," + onu.name + "EMBB_g"
        for onu in self.onu2_array: 
            line = line + "," + onu.name + "Video_g" + "," + onu.name + "IP_g"
        line = line + ","
        
        for onu in self.onu1_array: 
            line = line + "," + onu.name + "URLLC_a" + "," + onu.name + "EMBB_a"
        for onu in self.onu2_array: 
            line = line + "," + onu.name + "Video_a" + "," + onu.name + "IP_a"
        line = line + ","
        
        line = line + ",reward" + "\n"
        Globals.appendList(self.trainingStat, line)
        # self.trainingStat.append(line)

    def appendStat(self, requested, granted, action, reward, done, state = [], episodeReward = 0):
        line = ""
        if (len(state) != 0):
            for x in state:
                line = line + "," + str(x)
            line = line + ","

        for x in requested:
            line = line + "," + str(x)
        line = line + ","

        
        for x in granted:
            line = line + "," + str(x)
        line = line + ","

        for x in action:
            line = line + "," + str(x)
        line = line + ","

        line = line + "," + str(reward)
        if (done):
            line = line + ",EOP," + str(episodeReward) +"\n" 
        else:
            line = line + "\n" 
        Globals.appendList(self.trainingStat, line)