import time

import gym
from gym import spaces
from rl_learning import Globals
import random
import numpy
from gym.utils import seeding
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR, deepq
from stable_baselines.common.cmd_util import make_vec_env


class env_dba(gym.Env):
    "environment to simulate the DBA in XGSPON with mobile fronhaul ONUs and FTTx onus"

    def __init__(self,simpyEnv, network, M):
    
        # self.network = network
        self.simpyEnv = simpyEnv
        # self.M = M
        # self.onu_type1_count = 0
        # self.onu_type2_count = 0
        # self.onu1_array = []
        # self.onu2_array = []
        # node_names = list(network.keys())
        
        # for node_name in node_names:
        #     node = network[node_name]
        #     if node.type == Globals.ONU_TYPE:
        #         self.onu1_array.append(node)
        #         self.onu_type1_count = self.onu_type1_count + 1
        #     elif node.type == Globals.ONU_TYPE2:
        #         self.onu2_array.append(node)
        #         self.onu_type2_count = self.onu_type2_count + 1
        #     elif node.type == Globals.OLT_TYPE:
        #         self.olt = node
        self.onu_type1_count = 10
        self.onu_type2_count = 10
        self.onu1_array = numpy.arange(1,11,1)
        self.onu2_array = numpy.arange(1,11,1)
        self.action_space = spaces.MultiBinary(80)
        self.observation_space = spaces.Box(low=Globals.OBSERVATION_MIN, high=Globals.OBSERVATION_MAX, shape=(4, self.onu_type1_count + self.onu_type2_count), dtype=numpy.integer)
        self.state = numpy.array([numpy.repeat(0, self.onu_type1_count + self.onu_type2_count),numpy.repeat(0, self.onu_type1_count + self.onu_type2_count),numpy.repeat(0, self.onu_type1_count + self.onu_type2_count),numpy.repeat(0, self.onu_type1_count + self.onu_type2_count)], numpy.int32)
        self.counter = 0
        self.selectedAction = None

    # action and observation space 
      

    def generateOnuWorkloads(self):
        for onu in self.onu1_array:
            id = int(onu.node_name) - 1
            payload = onu.generatePayload(Globals.TCON1_ID)
            payload2 = onu.generatePayload(Globals.TCON2_ID)
            self.state[0][id-1] = self.state[0][id-1] + len(payload)
            self.state[1][id-1] = self.state[1][id-1] + len(payload2)
            self.state[2][id-1] = 0
            self.state[3][id-1] = 0
        
        for onu in self.onu2_array:
            id = int(onu.node_name) - 1
            payload = onu.generatePayload(Globals.TCON1_ID)
            payload2 = onu.generatePayload(Globals.TCON2_ID)
            self.state[2][id-1] = self.state[2][id-1] + len(payload)
            self.state[3][id-1] = self.state[3][id-1] + len(payload2)
            self.state[0][id-1] = 0
            self.state[1][id-1] = 0
    

    def convertStateToOltState(self):
        for j in range(len(self.state[0])):
            if self.network[str(j+1)].type == Globals.ONU_TYPE:
                self.olt.onu_queue_status[str(j+1)][Globals.TCON1_ID] = self.state[0][j]
                self.olt.onu_queue_status[str(j+1)][Globals.TCON2_ID] = self.state[1][j]
            else:
                self.olt.onu_queue_status[str(j+1)][Globals.TCON1_ID] = self.state[2][j]
                self.olt.onu_queue_status[str(j+1)][Globals.TCON2_ID] = self.state[3][j]

    def allocateBandwidth(self, action):
        self.olt.generateGrantMsg(self.selectedAction)

        
    # the OLT then applies the action to all ONUs
    def step(self, action):
        # take BW allocation action in this step
        # make necessary checks befre allcation
        # check if allocation is fair per TCON
        # action is an array with length of onus_count*services: [0 1 1 1 0 1 1 ...]
        # ordered as following: URLLC, eMBB, Video, IP
        # self.generateOnuWorkloads()
        self.selectedAction = action
        totalOnus = self.onu_type1_count + self.onu_type2_count
        # self.allocateBandwidth(action)
        for onu in range(self.onu_type1_count):
            id = int(onu) - 1 
            if (action[id] == 1):
                self.state[0][id] = 0
            else:
                self.state[0][id] = -1 * self.state[0][id]
            
            if (action[id + totalOnus] == 1):
                self.state[1][id] = 0
            else:
                self.state[1][id] = -1 * self.state[1][id]
            
            if (action[totalOnus + totalOnus + id] == 1):
                self.state[2][id] = -1000

            if (action[id + totalOnus + totalOnus + totalOnus] == 1):
                self.state[3][id] = -1000
        
        
        for onu in range(self.onu_type2_count):
            id = int(onu) - 1 
            if (action[totalOnus + totalOnus + id] == 1):
                self.state[2][id] = 0
            else: 
                self.state[2][id] = -1 * self.state[2][id]
            
            if (action[id + totalOnus + totalOnus + totalOnus] == 1):
                self.state[3][id] = 0
            else:
                self.state[3][id] = -1 * self.state[3][id]      

            if (action[id] == 1):
                self.state[0][id] = -1000
            
            if (action[id + totalOnus] == 1):
                self.state[1][id] = -1000
        
        
        # calculate reward from previous step
        reward = self.reward_function2()
        # get the state before applying the action 
        # state = report_dict + grant_dict 
        # update the state
        done = False
        self.counter = self.counter + 1
        if (self.counter > 200):
            done = True


        return self.state, reward, done, {}


    def updateStateFromOlt(self):
        for onu in self.onu1_array:
            id = int(onu.node_name)
            self.state[0][id-1] = self.olt.onu_queue_status[onu.node_name][Globals.TCON1_ID]
            self.state[1][id-1] = self.olt.onu_queue_status[onu.node_name][Globals.TCON2_ID]
            self.state[2][id-1] = 0
            self.state[3][id-1] = 0
        
        for onu in self.onu2_array:
            id = int(onu.node_name)
            self.state[2][id-1] = self.olt.onu_queue_status[onu.node_name][Globals.TCON1_ID]
            self.state[3][id-1] = self.olt.onu_queue_status[onu.node_name][Globals.TCON2_ID]
            self.state[0][id-1] = 0
            self.state[1][id-1] = 0


    def updateStateFromOlt2(self):
        for onu in self.onu1_array:
            id = int(onu)
            self.state[0][id-1] = random.randint(0, 1000)
            self.state[1][id-1] = random.randint(0, 1000)
            self.state[2][id-1] = 0
            self.state[3][id-1] = 0
        
        for onu in self.onu2_array:
            id = int(onu)
            self.state[2][id-1] = random.randint(0, 1000)
            self.state[3][id-1] = random.randint(0, 1000)
            self.state[0][id-1] = 0
            self.state[1][id-1] = 0

    def reset(self):
        self.updateStateFromOlt2()
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
    

    def reward_function2(self):
        return float(numpy.sum(self.state))

    # def reward_function(self):
    #     # compute the reward of 5G onu and delay of FTTx Onu in the last step
    #     # get the delay in the last step from the deuque
    #     # if delay > treshold (5G ONU)  --> -ve reward 
    #     # if delay < treshold (FTTx ONU)--> +ve reward 
    #     # multiply the reward by weighted factor eg. (10 for 5G ONU, 1 for FTTx ONU)
    #     URLLC_delay = []
    #     eMBB_delay = []
    #     Video_delay = []
    #     IP_delay = []
    #     node_names = list(self.network.keys())
    #     node_names.sort()
        
    #     for onu in self.onu1_array:
    #         tmp = onu.reported_ONU1_T1_URLLC[-1][Globals.REPORT_TIME] - onu.reported_ONU1_T1_URLLC[-1][Globals.TIME_STAMP]
    #         URLLC_delay.append(tmp)
    #         tmp = onu.reported_ONU1_T2_eMBB[-1][Globals.REPORT_TIME] - onu.reported_ONU1_T2_eMBB[-1][Globals.TIME_STAMP]
    #         eMBB_delay.append(tmp)

    #     for onu in self.onu2_array:
    #         tmp = onu.reported_ONU2_T2_Video[-1][Globals.REPORT_TIME] - onu.reported_ONU2_T2_Video[-1][Globals.TIME_STAMP]
    #         Video_delay.append(tmp)
    #         tmp = onu.reported_ONU2_T3_IP[-1][Globals.REPORT_TIME] - onu.reported_ONU2_T3_IP[-1][Globals.TIME_STAMP]
    #         IP_delay.append(tmp)

    #     URLLC_delay_avg = numpy.mean(URLLC_delay)
    #     eMBB_delay_avg = numpy.mean(eMBB_delay)
    #     Video_delay_avg = numpy.mean(Video_delay)
    #     IP_delay_avg = numpy.mean(IP_delay)
        

    #     URLLC_reward_ = 1 if URLLC_delay_avg < Globals.DELAY_TRESHOLD_MIN_URLLC else -1 
    #     eMBB_reward = 1 if eMBB_delay_avg < Globals.DELAY_TRESHOLD_MIN_EMBB else -1        
    #     video_reward = 1 if Video_delay_avg < Globals.DELAY_TRESHOLD_MIN_VIDEO else -1         
    #     IP_reward = 1 if IP_delay_avg < Globals.DELAY_TRESHOLD_MIN_IP else -1 
    #     reward = Globals.WEIGHT_URLLC * URLLC_reward_ + Globals.WEIGHT_EMBB * eMBB_reward + Globals.WEIGHT_VIDEO * video_reward +Globals.WEIGHT_IP * IP_reward       
    #     return reward
 
