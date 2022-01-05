from pickle import GLOBAL
import time
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
        # self.onu_type1_count = 10
        # self.onu_type2_count = 10
        # self.onu1_array = numpy.arange(1,11,1)
        # self.onu2_array = numpy.arange(1,11,1)
        min_values = [0]*self.serviceCount
        max_vlaue = [1]*self.serviceCount
        # self.action_space = spaces.Box(low=numpy.array([0.0]*(4)), high=numpy.array([0.000002]*(4)), dtype=numpy.float16)
        # self.action_space = spaces.MultiDiscrete([3,3,3,3])
        # self.action_space = spaces.MultiDiscrete([8]* 4)
        self.action_space = spaces.MultiDiscrete([8]* self.onus_count*2)
        # self.observation_space = spaces.MultiDiscrete([Globals.OBSERVATION_MAX] * (self.serviceCount * self.onus_count))
        self.observation_space = spaces.Box(low=Globals.OBSERVATION_MIN, high=Globals.OBSERVATION_MAX, shape=(self.serviceCount, self.onus_count), dtype=numpy.integer)
        # self.state = numpy.array([numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count)], numpy.integer)
        self.state = numpy.array([numpy.repeat(0, self.onus_count),numpy.repeat(0, self.onus_count)], numpy.integer)
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
        self.trainingSizes = [20]
        self.trainingStat = []
        self.appendHeader()
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
        self.loadDict = {}
        for onu in self.onu1_array:
            pkt = onu.gen_report_packet()
            self.olt.if_recv(onu.name, pkt)
            self.loadDict[pkt['ONU_ID']] = copy.deepcopy(pkt['QUEUE_LENGTH']) 
        for onu in self.onu2_array:
            pkt = onu.gen_report_packet()
            self.olt.if_recv(onu.name, pkt)
            self.loadDict[pkt['ONU_ID']] = copy.deepcopy(pkt['QUEUE_LENGTH'])

    def oltSendGrantPkt(self, action):
        # print (action)
        pkt = self.olt.generate_grant_msg(action)
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
        if (self.prevStep == currentStep):
            return
        self.prevStep = currentStep
        average = self.trainingSizes[currentStep]
        for onu in self.onu1_array:
            onu.createWorkloadGenerators(average, 2*Globals.SIM_TIME / stepsLength, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
        for onu in self.onu2_array:
            onu.createWorkloadGenerators(average, 2*Globals.SIM_TIME / stepsLength, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
        
        
        # if (self.prevAdjustTime == -1):
        #     self.prevAdjustTime = self.myClock.now 
        #     for onu in self.onu1_array:
        #         onu.createWorkloadGenerators(60, Globals.SIM_TIME / 2, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
        #     for onu in self.onu2_array:
        #         onu.createWorkloadGenerators(60, Globals.SIM_TIME / 2, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
        # if (self.myClock.now - self.prevAdjustTime > Globals.SIM_TIME / 2):
        #     self.prevAdjustTime = Globals.SIM_TIME / 2
        #     for onu in self.onu1_array:
        #         onu.createWorkloadGenerators(60, Globals.SIM_TIME / 2, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
        #     for onu in self.onu2_array:
        #         onu.createWorkloadGenerators(60, Globals.SIM_TIME / 2, self.M.G.nodes[onu.name][Globals.NODE_PKT_RATE_KWD][1])
        # else: 
        #     return


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

    def appendStat(self, requested, granted, action, reward, done):
        line = ""
        for onu in self.onu1_array: 
            line = line + "," + str(requested[onu.name][Globals.TCON1_ID]) + "," + str(requested[onu.name][Globals.TCON2_ID])
        for onu in self.onu2_array: 
            line = line + "," + str(requested[onu.name][Globals.TCON1_ID]) + "," + str(requested[onu.name][Globals.TCON2_ID])
        line = line + ","
        
        for onu in self.onu1_array: 
            line = line + "," + str(granted[onu.name][Globals.TCON1_ID]) + "," + str(granted[onu.name][Globals.TCON2_ID])
        for onu in self.onu2_array: 
            line = line + "," + str(granted[onu.name][Globals.TCON1_ID]) + "," + str(granted[onu.name][Globals.TCON2_ID])
        line = line + ","
        
        for onu in self.onu1_array: 
            Tcon1 = 2*int(onu.name) - 2
            Tcon2 = 2*int(onu.name) - 1
            line = line + "," + str(action[Tcon1]) + "," + str(action[Tcon2])
        for onu in self.onu2_array: 
            Tcon1 = 2*int(onu.name) - 2
            Tcon2 = 2*int(onu.name) - 1
            line = line + "," + str(action[Tcon1]) + "," + str(action[Tcon2])
        line = line + ","
        
        line = line + "," + str(reward)
        if (done):
            line = line + ",EOP\n" 
        else:
            line = line + "\n" 
        Globals.appendList(self.trainingStat, line)
        
        # self.trainingStat.append(line)

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
        
        requested = copy.deepcopy(self.olt.onu_queue_status)
        grantPkt = self.oltSendGrantPkt(action)
        self.sendOnuTraffic()
        self.resetOnuGrantedSize()
        reward, done1 = self.reward_function_queue_length(grantPkt)
        self.adjustOnuWorkloadGenerators()
        self.generateOnuWorkloads()
        self.onuSendReportPkt()
        self.updateStateFromOlt()
      
        # calculate reward from previous step
        # optimalAction = self.getOptimalAction(requested)
        # get the state before applying the action 
        # state = report_dict + grant_dict 
        # update the state
        # done = False
        done2 = False
        
        self.counter = self.counter + 1
        if (self.counter > 100):
            done2 = True
            # if (self.reward_sum < 70):
            #     reward = -1 * self.reward_sum

        done = done1 or done2
        self.appendStat(requested, grantPkt[Globals.GRANT_PACKET_BWM_FIELD], action, reward, done)
        
        self.reward_sum = self.reward_sum + reward
        
        if done == True:
            Globals.appendList(self.save_reward, self.reward_sum)
            self.eps_counter =  self.eps_counter + 1
            Globals.appendList(self.save_eps, self.eps_counter)

        self.myClock.inc()
        return self.state, reward, done, {}


    def updateStateFromOlt(self):
        for onu in self.onu1_array:
            id = int(onu.name)
            self.state[0][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON1_ID]
            self.state[1][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON2_ID]
        
        for onu in self.onu2_array:
            id = int(onu.name)
            self.state[0][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON1_ID]
            self.state[1][id-1] = self.olt.onu_queue_status[onu.name][Globals.TCON2_ID]



    def reset(self):
        # print("reset is called")
        # if (self.episodeCounter % 25 == 0):
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
        self.adjustOnuWorkloadGenerators()
        self.generateOnuWorkloads()
        self.onuSendReportPkt()
        self.updateStateFromOlt()
        self.counter = 0
        self.reward_sum = 0
        self.reward_violation_counter = 0
        self.prediction_violation_counter_urllc = 0
        self.prediction_violation_counter_embb = 0
        self.prediction_violation_counter_video = 0
        self.prediction_violation_counter_ip =0
    
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


    def reward_function_queue_length(self, grantPkt):
        # compute the reward of 5G onu and delay of FTTx Onu in the last step
        # get the delay in the last step from the deuque
        # if delay > treshold (5G ONU)  --> -ve reward 
        # if delay < treshold (FTTx ONU)--> +ve reward 
        # multiply the reward by weighted factor eg. (10 for 5G ONU, 1 for FTTx ONU)
        URLLC_queue_lenght = []
        eMBB_queue_length = []
        Video_queue_length = []
        IP_queue_length = []
        node_names = list(self.network.keys())
        node_names.sort()
        requested = self.olt.onu_queue_status
        for onu in self.onu1_array:
            tmp = 0
            w = []
            if len(onu.queue_lenght_URLLC) != 0:
                x = list(zip(*onu.queue_lenght_URLLC))
                w = x[1][-1:]
                # stime, pkt = onu.reported_ONU1_T1_URLLC[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            URLLC_queue_lenght.append(requested[onu.name][Globals.TCON1_ID])

            tmp = 0
            w = []
            if len(onu.queue_lenght_eMBB) != 0:
                x = list(zip(*onu.queue_lenght_eMBB))
                w = x[1][-1:]
                # stime, pkt = onu.reported_ONU1_T2_eMBB[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            eMBB_queue_length.append(requested[onu.name][Globals.TCON2_ID])

        for onu in self.onu2_array:
            tmp = 0
            w = []
            if len(onu.queue_lenght_Video) != 0:
                x = list(zip(*onu.queue_lenght_Video))
                w = x[1][-1:]
                # stime, pkt = onu.reported_ONU2_T2_Video[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            Video_queue_length.append(requested[onu.name][Globals.TCON1_ID])
            
            tmp = 0
            w = []
            if len(onu.queue_lenght_IP) != 0:
                x = list(zip(*onu.queue_lenght_IP))
                w = x[1][-1:]
                # stime, pkt = onu.reported_ONU2_T3_IP[-1]
                # tmp = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            IP_queue_length.append(requested[onu.name][Globals.TCON2_ID])

        URLLC_length_avg = numpy.mean(URLLC_queue_lenght) 
        eMBB_length_avg = numpy.mean(eMBB_queue_length)
        Video_length_avg = numpy.mean(Video_queue_length)
        IP_length_avg = numpy.mean(IP_queue_length)
        URLLC_length_max = numpy.max(URLLC_queue_lenght) 
        eMBB_length_max = numpy.max(eMBB_queue_length)
        Video_length_max = numpy.max(Video_queue_length)
        IP_length_max = numpy.max(IP_queue_length)

        
        
        if numpy.isnan(URLLC_length_avg) == True:
            URLLC_length_avg = Globals.LENGTH_TRESHOLD_URLLC
            URLLC_length_max = Globals.LENGTH_TRESHOLD_URLLC
            return 0
        if numpy.isnan(eMBB_length_avg) == True:
            eMBB_length_avg = Globals.LENGTH_TRESHOLD_EMBB
            eMBB_length_max = Globals.LENGTH_TRESHOLD_EMBB
            return 0
        if numpy.isnan(Video_length_avg) == True:
            Video_length_avg = Globals.LENGTH_TRESHOLD_VIDEO
            Video_length_max = Globals.LENGTH_TRESHOLD_VIDEO
            return 0
        if numpy.isnan(IP_length_avg) == True:
            IP_length_avg = Globals.LENGTH_TRESHOLD_IP
            IP_length_max = Globals.LENGTH_TRESHOLD_IP
            return 0

        ONU1_reward_ = 0
        # if URLLC_length_avg < Globals.LENGTH_TRESHOLD_URLLC and eMBB_length_avg < Globals.LENGTH_TRESHOLD_EMBB:
        #     ONU1_reward_ = (Globals.LENGTH_TRESHOLD_URLLC - URLLC_length_avg)**3  
        # elif URLLC_length_avg > Globals.LENGTH_TRESHOLD_URLLC: 
        #     ONU1_reward_ = ((Globals.LENGTH_TRESHOLD_URLLC - URLLC_length_avg)**3)
        # elif eMBB_length_avg >  Globals.LENGTH_TRESHOLD_EMBB: 
        #     ONU1_reward_ = -1*(Globals.LENGTH_TRESHOLD_URLLC - URLLC_length_avg)**3
        # elif URLLC_length_avg > Globals.LENGTH_TRESHOLD_URLLC and eMBB_length_avg > Globals.LENGTH_TRESHOLD_EMBB:
        #     ONU1_reward_ = 2*((Globals.LENGTH_TRESHOLD_URLLC - URLLC_length_avg)**3)
        # elif (URLLC_length_avg < Globals.LENGTH_TRESHOLD_URLLC and eMBB_length_avg > Globals.LENGTH_TRESHOLD_EMBB) or (URLLC_length_avg > Globals.LENGTH_TRESHOLD_URLLC and eMBB_length_avg < Globals.LENGTH_TRESHOLD_EMBB):
        #     ONU1_reward_ = -1*abs(((Globals.LENGTH_TRESHOLD_URLLC - URLLC_length_avg)**3))
        
        # if Video_length_avg < Globals.LENGTH_TRESHOLD_VIDEO: 
        #     video_reward = (40 - Video_length_avg)**0.3 
        # else: 
        #     video_reward = (35 - Video_length_avg)**0.3
        
        # if IP_length_avg < Globals.LENGTH_TRESHOLD_IP:
        #     IP_reward = (55 - IP_length_avg)**0.1
        # else: 
        #     IP_reward = (50 - IP_length_avg)**0.1

        done = False
        bwm = grantPkt[Globals.GRANT_PACKET_BWM_FIELD]
        totalAllocated = self.getTotal(bwm)
 
       
        if (URLLC_length_max <= Globals.LENGTH_TRESHOLD_URLLC and eMBB_length_max <= Globals.LENGTH_TRESHOLD_EMBB and IP_length_max <= Globals.LENGTH_TRESHOLD_IP and  Video_length_max <= Globals.LENGTH_TRESHOLD_VIDEO) or totalAllocated > 0.95 * Globals.FB: 
            reward = 0
            postivePartialReward = 1/(self.onus_count * 2)
            negativePartialReward = 0
            self.reward_violation_counter = 0
            for onu in self.onu1_array:
                # if 1.25 *self.loadDict[onu.name][1] >= bwm[onu.name][1] and  self.loadDict[onu.name][1] <= bwm[onu.name][1]:
                if (bwm[onu.name][1] - self.loadDict[onu.name][1] > 0 and bwm[onu.name][1] - self.loadDict[onu.name][1] <= .25 * Globals.URLLC_AB_MIN) or (self.loadDict[onu.name][1] - bwm[onu.name][1] <= Globals.LENGTH_TRESHOLD_URLLC and self.loadDict[onu.name][1] - bwm[onu.name][1] > 0):
                    reward = reward + postivePartialReward
                    self.prediction_violation_counter_urllc = 0
                else:
                    reward = reward - negativePartialReward
                    self.prediction_violation_counter_urllc = self.prediction_violation_counter_urllc + 1

                # if 1.25 *self.loadDict[onu.name][2] >= bwm[onu.name][2] and  self.loadDict[onu.name][2] <= bwm[onu.name][2]:
                if (bwm[onu.name][2] - self.loadDict[onu.name][2] > 0 and bwm[onu.name][2] - self.loadDict[onu.name][2] <= .25 * Globals.EMBB_AB_MIN) or (self.loadDict[onu.name][2] - bwm[onu.name][2] <= Globals.LENGTH_TRESHOLD_EMBB and self.loadDict[onu.name][2] - bwm[onu.name][2] > 0):
                    reward = reward + postivePartialReward
                    self.prediction_violation_counter_embb = 0
                else: 
                    reward = reward - negativePartialReward
                    self.prediction_violation_counter_embb = self.prediction_violation_counter_embb + 1
            
            for onu in self.onu2_array:
                # if 1.25 *self.loadDict[onu.name][1] >= bwm[onu.name][1] and  self.loadDict[onu.name][1] <= bwm[onu.name][1]:
                if (bwm[onu.name][1] - self.loadDict[onu.name][1] > 0 and bwm[onu.name][1] - self.loadDict[onu.name][1] <= .25 * Globals.VIDEO_AB_MIN) or (self.loadDict[onu.name][1] - bwm[onu.name][1] <= Globals.LENGTH_TRESHOLD_VIDEO and self.loadDict[onu.name][1] - bwm[onu.name][1] > 0):
                    reward = reward + postivePartialReward
                    self.prediction_violation_counter_video = 0 
                else: 
                    reward = reward - negativePartialReward
                    self.prediction_violation_counter_video = self.prediction_violation_counter_urllc + 1

                # if 1.25 *self.loadDict[onu.name][2] >= bwm[onu.name][2] and  self.loadDict[onu.name][2] <= bwm[onu.name][2]:
                if (bwm[onu.name][2] - self.loadDict[onu.name][2] > 0 and bwm[onu.name][2] - self.loadDict[onu.name][2] <= .25 * Globals.IP_AB_MIN) or (self.loadDict[onu.name][2] - bwm[onu.name][2] <= Globals.LENGTH_TRESHOLD_IP and self.loadDict[onu.name][2] - bwm[onu.name][2] > 0):
                    reward = reward + postivePartialReward
                    self.prediction_violation_counter_ip = 0 
                else: 
                    reward = reward - negativePartialReward
                    self.prediction_violation_counter_ip = self.prediction_violation_counter_ip + 1
        
        # elif URLLC_length_avg > Globals.LENGTH_TRESHOLD_URLLC and eMBB_length_avg > Globals.LENGTH_TRESHOLD_EMBB and IP_length_avg > Globals.LENGTH_TRESHOLD_IP and  Video_length_avg > Globals.LENGTH_TRESHOLD_VIDEO:
        else:   
            # reward = -10*(URLLC_length_avg + eMBB_length_avg + eMBB_length_avg + Video_length_avg)
            # if totalAllocated > 0.95 * Globals.FB:
            #     reward = 0
            # if totalAllocated < 0.95 * Globals.FB:
            #     reward = -10
            reward = 0
            # self.reward_violation_counter = self.reward_violation_counter + 1
        
        if (reward != 1):
            self.reward_violation_counter = self.reward_violation_counter + 1
            reward = 0
        # if (URLLC_length_avg > 5 * Globals.LENGTH_TRESHOLD_URLLC or eMBB_length_avg > 5 * Globals.LENGTH_TRESHOLD_EMBB or  IP_length_avg > 5* Globals.LENGTH_TRESHOLD_IP or Video_length_avg > 5* Globals.LENGTH_TRESHOLD_VIDEO) and totalAllocated < 0.95 * Globals.FB:
        #     self.reward_violation_counter = self.reward_violation_counter + 1
        
        #############################################################################
        # if (URLLC_length_max > 5 * Globals.LENGTH_TRESHOLD_URLLC or eMBB_length_max >  5 * Globals.LENGTH_TRESHOLD_EMBB or Video_length_max > 5* Globals.LENGTH_TRESHOLD_VIDEO or IP_length_max > 5 * Globals.LENGTH_TRESHOLD_IP or totalAllocated > Globals.FB) and totalAllocated < 0.95 * Globals.FB:
        #     self.reward_violation_counter = self.reward_violation_counter + 1

        ##################################################################################   
        
                  
        if (self.reward_violation_counter > 5 or self.prediction_violation_counter_urllc > 10 or self.prediction_violation_counter_embb > 10 or self.prediction_violation_counter_video > 10 or self.prediction_violation_counter_ip > 10):
            done = True

        # URLLC_reward_ =  (URLLC_delay_avg - Globals.DELAY_TRESHOLD_MIN_URLLC)*-1  
        # eMBB_reward =  (eMBB_delay_avg - Globals.DELAY_TRESHOLD_MIN_EMBB)*-1         
        # video_reward =  (Video_delay_avg - Globals.DELAY_TRESHOLD_MIN_VIDEO)*-1          
        # IP_reward =  (IP_delay_avg - Globals.DELAY_TRESHOLD_MIN_IP)*-1 

        # URLLC_reward_ = 20**((Globals.DELAY_TRESHOLD_MIN_URLLC - URLLC_delay_avg)*1000) if URLLC_delay_avg < Globals.DELAY_TRESHOLD_MIN_URLLC else -20 **((Globals.DELAY_TRESHOLD_MIN_URLLC - URLLC_delay_avg)*1000) 
        # eMBB_reward = 7**((Globals.DELAY_TRESHOLD_MIN_EMBB - eMBB_delay_avg)*1000) if eMBB_delay_avg < Globals.DELAY_TRESHOLD_MIN_EMBB else -7**((Globals.DELAY_TRESHOLD_MIN_EMBB - eMBB_delay_avg)*1000)        
        # video_reward = 5**((Globals.DELAY_TRESHOLD_MIN_VIDEO - Video_delay_avg)*1000) if Video_delay_avg < Globals.DELAY_TRESHOLD_MIN_VIDEO else -5**((Globals.DELAY_TRESHOLD_MIN_VIDEO - Video_delay_avg)*1000)         
        # IP_reward = 1**((Globals.DELAY_TRESHOLD_MIN_IP - IP_delay_avg)*1000) if IP_delay_avg < Globals.DELAY_TRESHOLD_MIN_IP else -1**((Globals.DELAY_TRESHOLD_MIN_IP - IP_delay_avg)*1000)


        # reward = Globals.WEIGHT_URLLC * URLLC_reward_ + Globals.WEIGHT_EMBB * eMBB_reward + Globals.WEIGHT_VIDEO * video_reward +Globals.WEIGHT_IP * IP_reward
        # reward = Globals.WEIGHT_ONU1 * ONU1_reward_ 

        # self.reward_mat.append(reward)
        # self.episode.append(self.myClock.now/0.000125)
        Globals.appendList(self.reward_mat, reward)
        Globals.appendList(self.episode, self.myClock.now/0.000125)

        self.reward_stat = [self.save_eps, self.save_reward]
             
        return reward, done
 
    def get_reward(self):
        return self.reward_stat