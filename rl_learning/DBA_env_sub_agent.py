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
        self.serviceCount = 2  
        self.action_space = spaces.Box(low=0.001, high=0.99, shape=(self.onu_type1_count, ),  dtype=numpy.float32)
        self.observation_space = spaces.Box(low=Globals.OBSERVATION_MIN, high=Globals.OBSERVATION_MAX, shape=(1, self.onu_type1_count), dtype=numpy.float32)
        # self.state = numpy.array([numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count)], numpy.integer)
        # self.state = numpy.array([numpy.repeat(0, self.onu_type1_count)], numpy.float32)
        self.state = numpy.empty(self.onu_type1_count, dtype=numpy.float32)
        # self.state = numpy.array([numpy.repeat(0, self.onus_count * self.serviceCount)], numpy.integer)
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
      

    def generateOnuWorkloads(self):
        for onu in self.onu1_array:
            id = int(onu.name) - 1
            onu.URLLC_ONU1_T1_pkt_gen_process()
            onu.eMBB_ONU1_T2_pkt_gen_process()

        
        for onu in self.onu2_array:
            id = int(onu.name) - 1
            onu.Video_ONU2_T2_pkt_gen_process()
            onu.IP_ONU2_T3_pkt_gen_process()
    

    def sendOnuTraffic(self):
        for onu in self.onu1_array:
            onu.forward_process()
        for onu in self.onu2_array:
            onu.forward_process()
        
    def onuSendReportPkt(self):
        self.loadDict = {}
        for onu in self.onu1_array:
            pkt = onu.gen_report_packet()
            self.olt.if_recv(onu.name, pkt)
            self.loadDict[pkt['ONU_ID']] = copy.deepcopy(pkt['QUEUE_LENGTH']) 
        for onu in self.onu2_array:
            pkt = onu.gen_report_packet()
            self.olt.if_recv(onu.name, pkt)
            self.loadDict[pkt['ONU_ID']] = copy.deepcopy(pkt['QUEUE_LENGTH'])

    def oltSendGrantPkt(self, action, maxAllowedAllocation):
        # print (action)
        pkt = self.olt.generate_grant_msg(action, maxAllowedAllocation)
        for onu in self.onu1_array:
            onu.if_recv(self.olt.name, pkt)
        for onu in self.onu2_array:
            onu.if_recv(self.olt.name, pkt)
        return pkt


    def resetOnuGrantedSize(self):
        for onu in self.onu1_array:
            onu.tcon_allocated_size[Globals.TCON1_ID] = 0
            onu.tcon_allocated_size[Globals.TCON2_ID] = 0
        for onu in self.onu2_array:
            onu.tcon_allocated_size[Globals.TCON1_ID] = 0
            onu.tcon_allocated_size[Globals.TCON2_ID] = 0


    def adjustOnuWorkloadGenerators(self):
        stepsLength = len(self.trainingSizes)
        currentStep = int((self.myClock.now / Globals.SIM_TIME) * stepsLength) 
        if (currentStep >= stepsLength - 1):
            currentStep = stepsLength - 1
        if (self.prevStep == currentStep):
            return
        self.prevStep = currentStep
        average = self.trainingSizes[currentStep]
        for onu in self.onu1_array:
            onu.createWorkloadGenerators(average, 2*Globals.SIM_TIME / stepsLength, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
        for onu in self.onu2_array:
            onu.createWorkloadGenerators(average, 2*Globals.SIM_TIME / stepsLength, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
    

    # the OLT then applies the action to all ONUs
    def step(self, action):
        # action = self.discreteToMultiDiscMap[action]
        self.state = self.deNormalizeState(self.state)
        requested = copy.deepcopy(self.state)
        grantPkt = self.oltSendGrantPkt(action, None)
        self.sendOnuTraffic()
        self.resetOnuGrantedSize()
        tmpState = self.updateStateFromGrant(grantPkt[Globals.GRANT_PACKET_BWM_FIELD])
        reward, done1 = self.reward_function_queue_length2(copy.deepcopy(tmpState), requested, grantPkt[Globals.GRANT_PACKET_BWM_FIELD])
        # self.adjustOnuWorkloadGenerators()
        self.generateOnuWorkloads()
        self.onuSendReportPkt()
        # self.updateStateFromOlt(tmpState)
        self.updateStateRandom()
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
        self.appendStat(requested, grantPkt[Globals.GRANT_PACKET_BWM_FIELD], action, reward, done, tmpState, self.reward_sum)
        
        
        if done == True:
            Globals.appendList(self.save_reward, self.reward_sum)
            self.eps_counter =  self.eps_counter + 1
            Globals.appendList(self.save_eps, self.eps_counter)

        self.myClock.inc()
        self.state = numpy.array(self.state, dtype=numpy.float32)
        return self.state, reward, done, {}

    def compressState(self, state):
        w=[]
        shape = (2,2)
        index_min =numpy.zeros(shape, dtype=int)
        newState = numpy.zeros(shape, dtype=int)
        number = [12, 25, 37, 50, 62, 75, 87, 100]
        for i in range(len(state)):
            for j in range(len(state[i])):
                w[:]= [abs(item-state[i][j]) for item in number]
                index_min[i][j] =  numpy.argmin(w)
                newState[i][j] = number[index_min[i][j]]
        return newState

    def deNormalizeState(self,state):
        tmpState = copy.deepcopy(state)
        for i in range(self.onu_type1_count):
            tmpState[0][i] = tmpState[0][i] * Globals.OBSERVATION_MAX
        
        # tmpState[0][-1] = tmpState[0][-1] * Globals.OBSERVATION_MAX
        return tmpState

    def normalizeState(self,state):
        tmpState = copy.deepcopy(self.state)
        for i in range(self.onu_type1_count):
            if (tmpState[0][i] < Globals.OBSERVATION_MIN):
                tmpState[0][i] = Globals.OBSERVATION_MIN
            if (state[0][i] > Globals.OBSERVATION_MAX):
                state[0][i] = Globals.OBSERVATION_MAX
            tmpState[0][i] = self.state[0][i] / Globals.OBSERVATION_MAX
        
        return tmpState

    def updateStateFromGrant(self, grantPkt):
        tmpState = copy.deepcopy(self.state)
        for onu in self.onu1_array:
            id = int(onu.name)
            tmpState[0][id-1] = self.state[0][id-1] - grantPkt[onu.name][Globals.TCON1_ID]
        return tmpState
    
    def updateStateFromOlt(self, prevState):
        for onu in self.onu1_array:
            id = int(onu.name)
            self.state[0][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON1_ID]

    def updateStateRandom(self):
        min = 0
        max = Globals.OBSERVATION_MAX
        for onu in self.onu1_array:
            id = int(onu.name)

            self.state[0][id-1] = int(self.np_random.uniform(low=min, high=max))

    def resetState(self, state):
        state = numpy.empty(self.onu_type1_count, dtype=numpy.float32)
        for i in range(len(state[0])):
            state[0][i] = 0
        return state

    def reset(self):
        # print("reset is called")
        # if (self.episodeCounter % 100 == 0):
        #     print (self.URLLC_reward_violation_counter)
        #     print (self.EMBB_reward_violation_counter)
        #     print (self.IP_reward_violation_counter)
        #     print (self.Video_reward_violation_counter)
        
        self.episodeCounter = self.episodeCounter + 1
        for onu in self.onu1_array:
            onu.reset()
        
        for onu in self.onu2_array:
            onu.reset()
            
        self.olt.reset()
        self.state = self.resetState(self.state)
        self.adjustOnuWorkloadGenerators()
        self.generateOnuWorkloads()
        self.onuSendReportPkt()
        self.updateStateFromOlt(self.state)
        self.updateStateRandom()
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

    # Reset the state of the environment to an initial state
    
    def render(self, mode='human', close=False):
    # Render the environment to the screen
        pass

    def no_assinment_allocation(self):
        grant_dict ={}
        return grant_dict

    def normal_assinment_allocation(self):
        grant_dict ={}
        return grant_dict

    def Bonus_assinment_allocation(self):
        grant_dict ={}
        return grant_dict
    

    def reward_function_delay(self):
        # compute the reward of 5G onu and delay of FTTx Onu in the last step
        # get the delay in the last step from the deuque
        # if delay > treshold (5G ONU)  --> -ve reward 
        # if delay < treshold (FTTx ONU)--> +ve reward 
        # multiply the reward by weighted factor eg. (10 for 5G ONU, 1 for FTTx ONU)
        URLLC_delay = []
        eMBB_delay = []
        Video_delay = []
        IP_delay = []
        node_names = list(self.network.keys())
        node_names.sort()
        
        for onu in self.onu1_array:
            tmp = 0
            w = []
            if len(onu.reported_ONU1_T1_URLLC) != 0:
                x = list(zip(*onu.reported_ONU1_T1_URLLC))
                w = x[0][-1:]
                # stime, pkt = onu.reported_ONU1_T1_URLLC[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            URLLC_delay.append(mean(w))

            tmp = 0
            w = []
            if len(onu.reported_ONU1_T2_eMBB) != 0:
                x = list(zip(*onu.reported_ONU1_T2_eMBB))
                w = x[0][-1:]
                # stime, pkt = onu.reported_ONU1_T2_eMBB[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            eMBB_delay.append(mean(w))

        for onu in self.onu2_array:
            tmp = 0
            w = []
            if len(onu.reported_ONU2_T2_Video) != 0:
                x = list(zip(*onu.reported_ONU2_T2_Video))
                w = x[0][-1:]
                # stime, pkt = onu.reported_ONU2_T2_Video[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            Video_delay.append(mean(w))
            
            tmp = 0
            w = []
            if len(onu.reported_ONU2_T3_IP) != 0:
                x = list(zip(*onu.reported_ONU2_T3_IP))
                w = x[0][-1:]
                # stime, pkt = onu.reported_ONU2_T3_IP[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            IP_delay.append(mean(w))

        URLLC_delay_avg = numpy.mean(URLLC_delay) 
        eMBB_delay_avg = numpy.mean(eMBB_delay)
        Video_delay_avg = numpy.mean(Video_delay)
        IP_delay_avg = numpy.mean(IP_delay)
        
        if numpy.isnan(URLLC_delay_avg) == True:
            URLLC_delay_avg = Globals.DELAY_TRESHOLD_MAX_URLLC
        if numpy.isnan(eMBB_delay_avg) == True:
            eMBB_delay_avg = Globals.DELAY_TRESHOLD_MAX_EMBB
        if numpy.isnan(Video_delay_avg) == True:
            Video_delay_avg = Globals.DELAY_TRESHOLD_MAX_VIDEO
        if numpy.isnan(IP_delay_avg) == True:
            IP_delay_avg = Globals.DELAY_TRESHOLD_MAX_IP


        URLLC_reward_ = 20**((Globals.DELAY_TRESHOLD_MIN_URLLC - URLLC_delay_avg)*1000) if URLLC_delay_avg < Globals.DELAY_TRESHOLD_MIN_URLLC else -20 **((Globals.DELAY_TRESHOLD_MIN_URLLC - URLLC_delay_avg)*1000) 
        eMBB_reward = 7**((Globals.DELAY_TRESHOLD_MIN_EMBB - eMBB_delay_avg)*1000) if eMBB_delay_avg < Globals.DELAY_TRESHOLD_MIN_EMBB else -7**((Globals.DELAY_TRESHOLD_MIN_EMBB - eMBB_delay_avg)*1000)        
        video_reward = 5**((Globals.DELAY_TRESHOLD_MIN_VIDEO - Video_delay_avg)*1000) if Video_delay_avg < Globals.DELAY_TRESHOLD_MIN_VIDEO else -5**((Globals.DELAY_TRESHOLD_MIN_VIDEO - Video_delay_avg)*1000)         
        IP_reward = 1**((Globals.DELAY_TRESHOLD_MIN_IP - IP_delay_avg)*1000) if IP_delay_avg < Globals.DELAY_TRESHOLD_MIN_IP else -1**((Globals.DELAY_TRESHOLD_MIN_IP - IP_delay_avg)*1000)

        # URLLC_reward_ =  (URLLC_delay_avg - Globals.DELAY_TRESHOLD_MIN_URLLC)*-1  
        # eMBB_reward =  (eMBB_delay_avg - Globals.DELAY_TRESHOLD_MIN_EMBB)*-1         
        # video_reward =  (Video_delay_avg - Globals.DELAY_TRESHOLD_MIN_VIDEO)*-1          
        # IP_reward =  (IP_delay_avg - Globals.DELAY_TRESHOLD_MIN_IP)*-1 

        URLLC_reward_ = 20**((Globals.DELAY_TRESHOLD_MIN_URLLC - URLLC_delay_avg)*1000) if URLLC_delay_avg < Globals.DELAY_TRESHOLD_MIN_URLLC else -20 **((Globals.DELAY_TRESHOLD_MIN_URLLC - URLLC_delay_avg)*1000) 
        eMBB_reward = 7**((Globals.DELAY_TRESHOLD_MIN_EMBB - eMBB_delay_avg)*1000) if eMBB_delay_avg < Globals.DELAY_TRESHOLD_MIN_EMBB else -7**((Globals.DELAY_TRESHOLD_MIN_EMBB - eMBB_delay_avg)*1000)        
        video_reward = 5**((Globals.DELAY_TRESHOLD_MIN_VIDEO - Video_delay_avg)*1000) if Video_delay_avg < Globals.DELAY_TRESHOLD_MIN_VIDEO else -5**((Globals.DELAY_TRESHOLD_MIN_VIDEO - Video_delay_avg)*1000)         
        IP_reward = 1**((Globals.DELAY_TRESHOLD_MIN_IP - IP_delay_avg)*1000) if IP_delay_avg < Globals.DELAY_TRESHOLD_MIN_IP else -1**((Globals.DELAY_TRESHOLD_MIN_IP - IP_delay_avg)*1000)


        reward = Globals.WEIGHT_URLLC * URLLC_reward_ + Globals.WEIGHT_EMBB * eMBB_reward + Globals.WEIGHT_VIDEO * video_reward +Globals.WEIGHT_IP * IP_reward
        # self.reward_mat.append(reward)
        Globals.appendList(self.reward_mat, reward)

        # self.episode.append(self.myClock.now/0.000125)
        Globals.appendList(self.episode, self.myClock.now/0.000125)

        self.reward_stat = [self.reward_mat, self.episode]
             
        return reward

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
        y = 1.1**(-2* abs(x))
        return y


    def reward_function_queue_length2(self, state, requested, granted):
        done = False
        violation = False
        allocatedSum = 0
        totalRequested = 0
        
        for onu in self.onu1_array:
            allocatedSum = allocatedSum + granted[onu.name][1]
            totalRequested = totalRequested + requested[0][int(onu.name)-1]
        

        for row in range(len(state)):
            for col in range(len(state[row]) - 1):
                if (state[row][col] <= 0 and abs(state[row][col]) <= numpy.ceil(0.25 * Globals.URLLC_AB_MIN)) or (state[row][col] >= 0 and state[row][col] <= Globals.LENGTH_TRESHOLD_URLLC2):
                    state[row][col] = 0   
                elif granted[str(col+1)][row + 1] == 2 * Globals.URLLC_AB_MIN and requested[row][col] >= 2 * Globals.URLLC_AB_MIN:
                    # the queue size is so large. Should grant the highest possible val
                    state[row][col] = 0   
                else:
                    violation = True

        absState = numpy.abs(state)
        sum = numpy.sum(absState)
        reward = self.computeExponReward(sum)
        if not violation:
            reward = 1
        #     if (allocatedSum > maxAllocation):
        #         reward = self.computeExponReward(state[0][-1])
        # elif totalRequested > maxAllocation and allocatedSum >= 0.95 * maxAllocation and allocatedSum <= 1.02 * maxAllocation:
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
            for onu in self.onu1_array:
                id = int(onu.name) - 1 
                line = line + "," + str(state[0][id])
            for onu in self.onu2_array: 
                id = int(onu.name) - 1 
                line = line + "," + str(state[0][id]) + "," + str(state[1][id])
            line = line + ","

        for onu in self.onu1_array:
            id = int(onu.name) - 1 
            line = line + "," + str(requested[0][id])
        for onu in self.onu2_array: 
            id = int(onu.name) - 1 
            line = line + "," + str(requested[0][id]) + "," + str(requested[1][id])

        # line = line + "," + str(requested[0][-1])
        line = line + ","
        
        for onu in self.onu1_array: 
            line = line + "," + str(granted[onu.name][Globals.TCON1_ID])
        for onu in self.onu2_array: 
            line = line + "," + str(granted[onu.name][Globals.TCON1_ID])
        line = line + ","
        
        for onu in self.onu1_array: 
            Tcon1 = int(onu.name) - 1
            line = line + "," + str(action[Tcon1])
        
        line = line + ","
        
        line = line + "," + str(reward)
        if (done):
            line = line + ",EOP," + str(episodeReward) +"\n" 
        else:
            line = line + "\n" 
        Globals.appendList(self.trainingStat, line)