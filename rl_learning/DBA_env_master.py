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
# from stable_baselines3 import SAC
import copy


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
        self.maxEpisodeLength = 1
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
        
        self.onus_count = self.onu_type2_count + self.onu_type1_count
        self.serviceCount = Globals.SERVICE_COUNT
        self.scaleFactor = 1 
        self.maxObservationURLLC = self.onu_type1_count * Globals.URLLC_AB_MIN * 2
        self.maxObservationEMBB = self.onu_type1_count * Globals.EMBB_AB_MIN * 2
        self.maxObservationVideo = self.onu_type1_count * Globals.VIDEO_AB_MIN * 2
        self.maxObservationIP =  self.onu_type1_count * Globals.IP_AB_MIN * 2
        self.maxObservationURLLC = self.maxObservationURLLC * self.scaleFactor
        self.maxObservationEMBB = self.maxObservationEMBB * self.scaleFactor
        self.maxObservationVideo = self.maxObservationVideo * self.scaleFactor
        self.maxObservationIP = self.maxObservationIP * self.scaleFactor
        self.accuracy = 0.125 * 0.5 * Globals.URLLC_AB_MIN  
        self.maxObservation = max(self.maxObservationURLLC, self.maxObservationEMBB, self.maxObservationVideo, self.maxObservationIP) 
        self.minObservation = 0
        self.allocationStep = 1 
        # self.action_space = spaces.MultiDiscrete([8]* self.serviceCount)
        # numpy.array([0.0,0.0,0.0,0.0]), numpy.array([+1.0,+1.0,+1.0,1.0])
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.serviceCount, ),  dtype=numpy.float32)

        # observation_space = [URRLC_SUM, EMBB_SUM, VIDEO_SUM, IP_SUM]
        self.observation_space = spaces.Box(low=self.minObservation, high=self.maxObservation, shape=(1, self.serviceCount), dtype=numpy.float32)
        # self.state = numpy.array([numpy.repeat(0, self.serviceCount)], numpy.float32)
        self.state = numpy.empty(self.serviceCount, dtype=numpy.float32)
        self.allocationArray = numpy.empty(self.serviceCount, dtype=numpy.float32)
        # self.allocationArray = numpy.array([numpy.repeat(0, self.serviceCount)], numpy.float32)
        self.counter = 0
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
        self.appendHeader()
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        # action = self.discreteToMultiDiscMap[action]
        self.state = self.deNormalizeState(self.state)
        requested = copy.deepcopy(self.state)
        self.multiDiscreteToAllocation(action)
        reward, done1 = self.computeReward()
        self.updateStateRandom(self.state)
        self.state = self.normalizeState(self.state)
        # state seen by agent: depends on 1- agent action in prevoius state(grants) 2- generation in current state(requests) so the aget can calculate the reward
        # curent state seen by agent: 1- generation at time t, 2- grant at time t-1  

        done2 = False
        
        self.counter = self.counter + 1
        self.reward_sum = self.reward_sum + reward
        if (self.counter >= self.maxEpisodeLength):
            done2 = True
            if (self.reward_sum >= self.maxEpisodeLength * 1):
                reward = self.reward_sum
                self.reward_sum = self.reward_sum + reward


        done = done1 or done2
        self.appendStat(requested, self.allocationArray, action, reward, done, [], self.reward_sum)
        
        
        if done == True:
            Globals.appendList(self.save_reward, self.reward_sum)
            self.eps_counter =  self.eps_counter + 1
            Globals.appendList(self.save_eps, self.eps_counter)

        self.myClock.inc()
        self.state = numpy.array(self.state, dtype=numpy.float32)
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
        self.updateStateRandom(self.state)
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
        action = round(action, 2)
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
        self.allocationArray[0] = self.actionToVal(rlAction[0], self.maxObservationURLLC)
        self.allocationArray[1] = self.actionToVal(rlAction[1], self.maxObservationEMBB)
        self.allocationArray[2] = self.actionToVal(rlAction[2], self.maxObservationVideo)
        self.allocationArray[3] = self.actionToVal(rlAction[3], self.maxObservationIP)
        self.allocationArray = numpy.array(self.allocationArray, dtype=numpy.float32)
        return self.allocationArray

    def deNormalizeState(self,state):
        state[0] = state[0] * self.maxObservationURLLC
        state[1] = state[1] * self.maxObservationEMBB
        state[2] = state[2] * self.maxObservationVideo
        state[3] = state[3] * self.maxObservationIP
        return state

    def normalizeState(self, state, maintainBounds = True):
        
        if (maintainBounds):
            if (state[0] < self.minObservation):
                state[0] = self.minObservation
            if (state[0] > self.maxObservationURLLC):
                state[0] = self.maxObservationURLLC
            if (state[1] < self.minObservation):
                state[1] = self.minObservation
            if (state[1] > self.maxObservationEMBB):
                state[1] = self.maxObservationEMBB
            if (state[2] < self.minObservation):
                state[2] = self.minObservation
            if (state[2] > self.maxObservationVideo):
                state[2] = self.maxObservationVideo
            if (state[3] < self.minObservation):
                state[3] = self.minObservation
            if (state[3] > self.maxObservationIP):
                state[3] = self.maxObservationIP
    

        state[0] = state[0]/self.maxObservationURLLC
        state[1] = state[1]/self.maxObservationEMBB
        state[2] = state[2]/self.maxObservationVideo
        state[3] = state[3]/self.maxObservationIP
        
        return state

    def updateStateRandom(self,state):
        min = 0
        state[0] = int(self.np_random.uniform(low=0, high=self.maxObservationURLLC)) 
        state[1] = int(self.np_random.uniform(low=0, high=self.maxObservationEMBB)) 
        state[2] = int(self.np_random.uniform(low=0, high=self.maxObservationVideo)) 
        state[3] = int(self.np_random.uniform(low=0, high=self.maxObservationIP)) 
        return state


    def resetState(self, state):
        # state = numpy.array([numpy.repeat(0, self.serviceCount)], numpy.float32)
        state = numpy.empty(self.serviceCount, dtype=numpy.float32)
        state[0] = 0
        state[1] = 0
        state[2] = 0
        state[3] = 0
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
        
        if requestedSum < Globals.FB:
            for row in range(len(state)):
                #or (difference[row] > 0 and difference[row] <= self.accuracy)
                if (difference[row] < 0 and abs(difference[row]) <= self.accuracy):
                    difference[row] = 0   
                elif allocated[row] == self.maxObservationURLLC and state[row] >= self.maxObservationURLLC:
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
                    
        else:
            minURLLC = min(state[0], Globals.URLLC_AB_MIN * self.onu_type1_count)
            minEMBB = min(state[1], Globals.EMBB_AB_MIN * self.onu_type1_count)
            minVideo = min(state[2], Globals.VIDEO_AB_MIN * self.onu_type2_count)
            minIP = min(state[3], Globals.IP_AB_MIN * self.onu_type2_count)
            totalMin =  minURLLC + minEMBB + minVideo + minIP

            remainingURLLC = state[0] - minURLLC 
            remainingEMBB = state[1] - minEMBB 
            remainingVideo = state[2] - minVideo 
            remainingIP = state[3] - minIP 
            
            remainingFB = Globals.FB - totalMin


            if not(allocated[0] > minURLLC and allocated[1] > minEMBB and allocated[2] > minVideo and allocated[3] > minIP):
                penalty = penalty + totalMin
                
            if remainingFB >= remainingURLLC and not(allocated[0] >= state[0] and allocated[0] <= state[0] + self.accuracy):
                penalty = penalty + abs(allocated[0] - state[0])
            elif remainingFB < remainingURLLC:
                penalty = penalty + abs(allocated[0] - (remainingFB + minURLLC))
            elif remainingFB >= remainingURLLC and (allocated[0] >= state[0] and allocated[0] <= state[0] + self.accuracy):
                remainingFB = remainingFB - remainingURLLC
            
            if remainingFB >= remainingEMBB and not(allocated[1] >= state[1] and allocated[1] <= state[1] + self.accuracy):
                penalty = penalty + abs(allocated[1] - state[1])
            elif remainingFB < remainingEMBB:
                penalty = penalty + abs(allocated[1] - (remainingFB + minEMBB))
            elif remainingFB >= remainingEMBB and (allocated[1] >= state[1] and allocated[1] <= state[1] + self.accuracy):
                remainingFB = remainingFB - remainingEMBB
            
            if remainingFB >= remainingVideo and not(allocated[2] >= state[2] and allocated[2] <= state[2] + self.accuracy):
                penalty = penalty + abs(allocated[2] - state[2])
            elif remainingFB < remainingVideo:
                penalty = penalty + abs(allocated[2] - (remainingFB + minVideo))
            elif remainingFB >= remainingVideo and (allocated[2] >= state[2] and allocated[2] <= state[2] + self.accuracy):
                remainingFB = remainingFB - remainingVideo
            
            if remainingFB >= remainingIP and not(allocated[3] >= state[3] and allocated[3] <= state[3] + self.accuracy):
                penalty = penalty + abs(allocated[3] - state[3])
            elif remainingFB < remainingIP:
                penalty = penalty + abs(allocated[3] - (remainingFB + minIP))
            elif remainingFB >= remainingIP and (allocated[3] >= state[3] and allocated[3] <= state[3] + self.accuracy):
                remainingFB = remainingFB - remainingIP
           

            penalty = min(100, penalty)
        reward = self.computeExponReward(penalty)
        # if not violation:
        #     reward = 1

        Globals.appendList(self.reward_mat, reward)
        Globals.appendList(self.episode, self.myClock.now/0.000125)

        self.reward_stat = [self.save_eps, self.save_reward]

        return reward, done

    def reward_function_queue_length(self, state):
        # compute the reward of 5G onu and delay of FTTx Onu in the last step
        # get the delay in the last step from the deuque
        # if delay > treshold (5G ONU)  --> -ve reward 
        # if delay < treshold (FTTx ONU)--> +ve reward 
        # multiply the reward by weighted factor eg. (10 for 5G ONU, 1 for FTTx ONU)
        done = False
        reward = 0
        onuMaxReward = 1 / (self.onu_type1_count + self.onu_type2_count) / 2
        violation = False
        violationCount = 0
        for onu in self.onu1_array:
            id = int(onu.name)-1
            if ((state[0][id] <= 0 and abs(state[0][id]) <= 0.25 * Globals.URLLC_AB_MIN) or (state[0][id] >= 0 and state[0][id] <= Globals.LENGTH_TRESHOLD_URLLC2)):
                partialReward = onuMaxReward
                self.prediction_violation_counter_urllc = 0
            elif state[0][id] > Globals.LENGTH_TRESHOLD_URLLC2:
                violation = True
                partialReward = 0
            else:
                partialReward = self.computeExponReward(state[0][id])
                partialReward = partialReward * onuMaxReward
                violationCount = violationCount + 1

            reward = reward + partialReward
            # if (state[0][id] > 4 * Globals.LENGTH_TRESHOLD_URLLC2):
            #     self.reward_violation_counter = self.reward_violation_counter + 1 


            id = int(onu.name)-1
            if ((state[1][id] <= 0 and abs(state[1][id]) <= 0.25 * Globals.EMBB_AB_MIN) or (state[1][id] >= 0 and state[1][id] <= Globals.LENGTH_TRESHOLD_EMBB2)):
                partialReward = onuMaxReward 
                self.prediction_violation_counter_embb = 0
            elif state[1][id] > Globals.LENGTH_TRESHOLD_EMBB2:
                violation = True
                partialReward = 0
            else:
                partialReward = self.computeExponReward(state[1][id])
                partialReward = partialReward * onuMaxReward 
                violationCount = violationCount + 1
                # self.prediction_violation_counter_embb = self.prediction_violation_counter_embb + 1

            reward = reward + partialReward
            # if (abs(state[1][id]) > 4 * Globals.LENGTH_TRESHOLD_EMBB2):
            #     self.reward_violation_counter = self.reward_violation_counter + 1 

        for onu in self.onu2_array:
            id = int(onu.name)-1
            if ((state[0][id] <= 0 and abs(state[0][id]) <= 0.25 * Globals.VIDEO_AB_MIN) or (state[0][id] >= 0 and state[0][id] <= Globals.LENGTH_TRESHOLD_VIDEO2)):
                partialReward = onuMaxReward 
                self.prediction_violation_counter_video = 0
            elif state[0][id] > Globals.LENGTH_TRESHOLD_VIDEO2:
                violation = True
                partialReward = 0
            else:
                partialReward = self.computeExponReward(state[0][id])
                partialReward = partialReward * onuMaxReward
                violationCount = violationCount + 1
                # self.prediction_violation_counter_video = self.prediction_violation_counter_video + 1

            reward = reward + partialReward
            # if (abs(state[0][id]) > 4 * Globals.LENGTH_TRESHOLD_VIDEO2):
            #     self.reward_violation_counter = self.reward_violation_counter + 1 

            id = int(onu.name)-1
            if ((state[1][id] <= 0 and abs(state[1][id]) <= 0.25 * Globals.IP_AB_MIN) or  (state[1][id] >= 0 and state[1][id] <= Globals.LENGTH_TRESHOLD_IP2)):
                partialReward = onuMaxReward
                self.prediction_violation_counter_ip = 0
            elif state[1][id] > Globals.LENGTH_TRESHOLD_IP2:
                violation = True
                partialReward = 0
            else:
                partialReward = self.computeExponReward(state[1][id])
                partialReward = partialReward * onuMaxReward 
                violationCount = violationCount + 1
                # self.prediction_violation_counter_ip = self.prediction_violation_counter_ip + 1

            reward = reward + partialReward
            # if (abs(state[1][id]) > 4 * Globals.LENGTH_TRESHOLD_IP2):
            #     self.reward_violation_counter = self.reward_violation_counter + 1 

        if violation == True:
            self.reward_violation_counter = self.reward_violation_counter + 1
            reward = 0
        else:
            self.reward_violation_counter = 0
        
        if reward == 1:
            reward = 10

        if (self.reward_violation_counter > 5 or self.prediction_violation_counter_urllc > 10 or self.prediction_violation_counter_embb > 10 or self.prediction_violation_counter_video > 10 or self.prediction_violation_counter_ip > 10):
            done = True
            reward = 0
            self.prediction_violation_counter_urllc = 0
            self.prediction_violation_counter_embb = 0
            self.prediction_violation_counter_video = 0
            self.prediction_violation_counter_ip = 0



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