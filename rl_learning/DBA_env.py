import time
import gym
from gym import spaces
import Globals
import random
import numpy
from gym.utils import seeding
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR, deepq
from stable_baselines.common.cmd_util import make_vec_env
# from stable_baselines3 import SAC


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
        node_names = list(network.keys())
        
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
        self.serviceCount = 4  
        # self.onu_type1_count = 10
        # self.onu_type2_count = 10
        # self.onu1_array = numpy.arange(1,11,1)
        # self.onu2_array = numpy.arange(1,11,1)
        self.action_space = spaces.MultiBinary(self.onus_count * self.serviceCount)
        self.observation_space = spaces.Box(low=Globals.OBSERVATION_MIN, high=Globals.OBSERVATION_MAX, shape=(self.serviceCount, self.onus_count), dtype=numpy.integer)
        self.state = numpy.array([numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count)], numpy.int32)
        self.counter = 0
        self.selectedAction = None
        spaces.Discrete

    # action and observation space 
      

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
        for onu in self.onu1_array:
            pkt = onu.gen_report_packet()
            self.olt.if_recv(onu.name, pkt)
        for onu in self.onu2_array:
            pkt = onu.gen_report_packet()
            self.olt.if_recv(onu.name, pkt)


    def convertStateToOltState(self):
        for j in range(len(self.state[0])):
            if self.network[str(j+1)].type == Globals.ONU_TYPE:
                self.olt.onu_queue_status[str(j+1)][Globals.TCON1_ID] = self.state[0][j]
                self.olt.onu_queue_status[str(j+1)][Globals.TCON2_ID] = self.state[1][j]
            else:
                self.olt.onu_queue_status[str(j+1)][Globals.TCON1_ID] = self.state[2][j]
                self.olt.onu_queue_status[str(j+1)][Globals.TCON2_ID] = self.state[3][j]

    def oltSendGrantPkt(self, action):
        print (action)
        action = [0, 1, 0, 0, 0, 0, 1, 1]
        pkt = self.olt.generate_grant_msg(action)
        for onu in self.onu1_array:
            onu.if_recv(self.olt.name, pkt)
        for onu in self.onu2_array:
            onu.if_recv(self.olt.name, pkt)
        
    # the OLT then applies the action to all ONUs
    def step(self, action):
        # take BW allocation action in this step
        # make necessary checks befre allcation
        # check if allocation is fair per TCON
        # action is an array with length of onus_count*services: [0 1 1 1 0 1 1 ...]
        # ordered as following: URLLC, eMBB, Video, IP
        # self.generateOnuWorkloads()
        
        # pseudo code
            # generate Messages at ONUs
            # send messages at ONUs based on grant size
            # generate Report at ONUs
            # send Report to OLT 
            # OLT generates Grant
            # OLT sends Grant to ONUs
        self.generateOnuWorkloads()
        self.sendOnuTraffic()
        self.onuSendReportPkt()
        self.oltSendGrantPkt(action)


        self.updateStateFromOlt()
   
        
        
        # calculate reward from previous step
        reward = self.reward_function()
        # get the state before applying the action 
        # state = report_dict + grant_dict 
        # update the state
        done = False
        self.counter = self.counter + 1
        # if (self.counter > 200):
        #     done = True

        self.myClock.inc()
        return self.state, reward, done, {}


    def updateStateFromOlt(self):
        for onu in self.onu1_array:
            id = int(onu.name)
            self.state[0][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON1_ID]
            self.state[1][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON2_ID]
            self.state[2][id-1] = 0
            self.state[3][id-1] = 0
        
        for onu in self.onu2_array:
            id = int(onu.name)
            self.state[2][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON1_ID]
            self.state[3][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON2_ID]
            self.state[0][id-1] = 0
            self.state[1][id-1] = 0


    def updateStateFromOlt2(self):
        for onu in self.onu1_array:
            id = int(onu.name)
            self.state[0][id-1] = random.randint(0, 1000)
            self.state[1][id-1] = random.randint(0, 1000)
            self.state[2][id-1] = 0
            self.state[3][id-1] = 0
        
        for onu in self.onu2_array:
            id = int(onu.name)
            self.state[2][id-1] = random.randint(0, 1000)
            self.state[3][id-1] = random.randint(0, 1000)
            self.state[0][id-1] = 0
            self.state[1][id-1] = 0

    def reset(self):
        self.updateStateFromOlt()
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

    def reward_function(self):
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
            if len(onu.reported_ONU1_T1_URLLC) != 0:
                stime, pkt = onu.reported_ONU1_T1_URLLC[-1]
                tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            URLLC_delay.append(tmp)

            tmp = 0
            if len(onu.reported_ONU1_T2_eMBB) != 0:
                stime, pkt = onu.reported_ONU1_T2_eMBB[-1]
                tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            eMBB_delay.append(tmp)

        for onu in self.onu2_array:
            tmp = 0
            if len(onu.reported_ONU2_T2_Video) != 0:
                stime, pkt = onu.reported_ONU2_T2_Video[-1]
                tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            Video_delay.append(tmp)
            
            tmp = 0
            if len(onu.reported_ONU2_T3_IP) != 0:
                stime, pkt = onu.reported_ONU2_T3_IP[-1]
                tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            IP_delay.append(tmp)

        URLLC_delay_avg = numpy.mean(URLLC_delay)
        eMBB_delay_avg = numpy.mean(eMBB_delay)
        Video_delay_avg = numpy.mean(Video_delay)
        IP_delay_avg = numpy.mean(IP_delay)
        

        URLLC_reward_ = 1 if URLLC_delay_avg < Globals.DELAY_TRESHOLD_MIN_URLLC else -1 
        eMBB_reward = 1 if eMBB_delay_avg < Globals.DELAY_TRESHOLD_MIN_EMBB else -1        
        video_reward = 1 if Video_delay_avg < Globals.DELAY_TRESHOLD_MIN_VIDEO else -1         
        IP_reward = 1 if IP_delay_avg < Globals.DELAY_TRESHOLD_MIN_IP else -1 
        reward = Globals.WEIGHT_URLLC * URLLC_reward_ + Globals.WEIGHT_EMBB * eMBB_reward + Globals.WEIGHT_VIDEO * video_reward +Globals.WEIGHT_IP * IP_reward       
        return reward
    
 
