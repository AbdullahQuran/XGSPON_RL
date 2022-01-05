import random
import collections  # provides 'deque'
import inspect
from numpy.core.numeric import count_nonzero
import pandas as pd
import networkx as nx
from rl_learning import Globals
import Utils
import PacketGenerator
from enum import Enum
import packet
import json
import numpy
import copy 

class OLT(object):
    """
    Models an optical network unit
    """

    def __init__(self, env, M, node_name, verbose):
        self.M = M
        self.rlModel = M.rlModel
        self.odn = M.odn
        self.env = env  # SimPy environment
        self.name = node_name  # must be unique
        self.type = M.G.nodes[node_name][Globals.NODE_TYPE_KWD]
        self.pkt_rate = M.G.nodes[node_name][Globals.NODE_PKT_RATE_KWD]
        self.proc_delay = M.G.nodes[node_name][Globals.NODE_PROC_DELAY_KWD]
        self.queue_check = M.G.nodes[node_name][Globals.NODE_QUEUE_CHECK_KWD]
        self.queue_cutoff = M.G.nodes[node_name][Globals.NODE_QUEUE_CUTOFF_KWD]
        # precomputed table of shortest paths
        self.path_G = None
        self.nodes = nx.nodes(M.G)
        self.cycle = 0
        self.onu1Count = 0
        self.onu2Count = 0
        self.start_time = -1
        self.end_time = -1
        self.prop_delay = 0.001

        self.conns = {}
        # self.customConnections = {}
        self.verbose = verbose

        # processing queue and queue length monitor
        self.proc_queue = collections.deque()
        self.queue_mon = collections.deque()

        # packets persistent storage
        self.generated = collections.deque()
        self.received = collections.deque()
        self.forwarded = collections.deque()
        self.discarded = collections.deque()
        self.Grant_message_OLT = collections.deque()
        self.generated_ONU1_T1_URLLC = collections.deque()
        self.generated_ONU1_T2_eMBB = collections.deque()
        self.generated_ONU2_T2_Video = collections.deque()
        self.generated_ONU2_T3_IP = collections.deque()
        self.reported_ONU1_T1_URLLC = collections.deque()
        self.reported_ONU1_T2_eMBB = collections.deque()
        self.reported_ONU2_T2_Video = collections.deque()
        self.reported_ONU2_T3_IP = collections.deque()
        self.report_message_ONU = collections.deque()
        self.meanDealy_all = collections.deque()
        

        # counters for sent/received packets (key=node name)
        self.pkt_sent = {}
        self.pkt_recv = {}
        self.FB_remaining = Globals.FB
        self.onu_queue_status = {}  # {1: {1: 1400, 2: 500}, onu_id:{tcon_id: reported_size}}
        self.tcon_alloc_matrix = {}  # {1: {1: 1400, 2: 500}, onu_id:{tcon_id: reported_size}}
        self.tcon_unsatisfied_matrix = {}  # {1: {1: 1400, 2: 500}, onu_id:{tcon_id: reported_size}}
        self.vb_min_matrix = {}  # {1: {1: 1400, 2: 500}, onu_id:{tcon_id: reported_size}}
        self.vb_sur_matrix = {}  # {1: {1: 1400, 2: 500}, onu_id:{tcon_id: reported_size}}
        self.vb_max_matrix = {}
        self.onuCount = 0
        for x in M.G.nodes():
            if M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                self.onu_queue_status[x] = {Globals.TCON1_ID: 0, Globals.TCON2_ID: 0}
                self.tcon_alloc_matrix[x] = {Globals.TCON1_ID: 0, Globals.TCON2_ID: 0}
                self.tcon_unsatisfied_matrix[x] = {Globals.TCON1_ID: 0, Globals.TCON2_ID: 0}
                self.onuCount = self.onuCount + 1
            if M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                self.vb_min_matrix[x] = {Globals.TCON1_ID: Globals.URLLC_AB_MIN, Globals.TCON2_ID: Globals.EMBB_AB_MIN}
                self.vb_sur_matrix[x] = {Globals.TCON1_ID: Globals.URLLC_AB_SUR, Globals.TCON2_ID: Globals.EMBB_AB_SUR}
                self.vb_max_matrix[x] = {Globals.TCON1_ID: Globals.URLLC_RM, Globals.TCON2_ID: Globals.EMBB_RM}
                self.onu1Count = self.onu1Count + 1 

            if M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                self.vb_min_matrix[x] = {Globals.TCON1_ID: Globals.VIDEO_AB_MIN, Globals.TCON2_ID: Globals.IP_AB_MIN}
                self.vb_sur_matrix[x] = {Globals.TCON1_ID: Globals.VIDEO_AB_SUR, Globals.TCON2_ID: Globals.IP_AB_SUR}
                self.vb_max_matrix[x] = {Globals.TCON1_ID: Globals.VIDEO_RM, Globals.TCON2_ID: Globals.IP_RM}
                self.onu2Count = self.onu2Count + 1 
    
        # dt interval for incoming queue monitoring
        self.queue_monitor_deltat = Globals.QUEUE_MONITOR_DELTAT

    

    def test(self):
        self.FB_remaining = 10000
        self.onu_queue_status['1'] = {Globals.TCON1_ID: 100, Globals.TCON2_ID: 500}     
        self.onu_queue_status['2'] = {Globals.TCON1_ID: 200, Globals.TCON2_ID: 300}     
        self.onu_queue_status['3'] = {Globals.TCON1_ID: 250, Globals.TCON2_ID: 100}     
        self.onu_queue_status['4'] = {Globals.TCON1_ID: 150, Globals.TCON2_ID: 50}     
        
        self.vb_min_matrix['1'] = {Globals.TCON1_ID: 100, Globals.TCON2_ID: 600}
        self.vb_min_matrix['2'] = {Globals.TCON1_ID: 100, Globals.TCON2_ID: 600}
        self.vb_min_matrix['3'] = {Globals.TCON1_ID: 300, Globals.TCON2_ID: 150}
        self.vb_min_matrix['4'] = {Globals.TCON1_ID: 300, Globals.TCON2_ID: 150}
        x = self.generateGrantMsg()
        print (x)

    def add_conn(self, c, conn):
        """Adds a connection from this node to the node 'c'"""
        self.conns[c] = conn
        # self.customConnections[c] = customConnection
        self.pkt_sent[c] = 0
        self.pkt_recv[c] = 0

    def if_up(self):
        """Activates interfaces- this sets up SimPy processes"""

        # start recv processes on all receiving connections
        for c in self.conns:
            self.env.process(self.if_recv(c))

        # activate packet generator, packet forwarding, queue monitoring
        self.env.process(self.generate_grant_msg())
        # self.env.process(self.queue_monitor())

    def fullCycle(self):
        while True:
            for c in self.conns:
                self.if_recv(c)
            self.generate_grant_msg()
            yield self.env.timeout(Globals.SERVICE_INTERVAL)

    def if_recv(self, c):
        """Node receive interface from node 'c'"""

        # the connection from 'self.name' to 'c'
        conn = self.conns[c]

        while True:
            # pick up any incoming packets
            pkt = yield conn.get()
            if(pkt[Globals.ONU_PACKET_TYPE_FIELD] == Globals.ONU_REPORT_PACKET_TYPE):
                self.process_report_msg(pkt)
                # increment the counter for this sending node
                self.pkt_recv[c] += 1

                # put the packet in the processing queue

                # report as per verbose level
                if self.verbose >= Globals.VERB_LO:
                    Utils.report(
                        self.env.now,
                        self.name,
                        pkt,
                        self.proc_queue,
                        inspect.currentframe().f_code.co_name,
                        self.verbose,
                    )


    def process_report_msg(self, pkt):
        onuId = pkt[Globals.REPORT_PACKET_ONU_ID_FIELD]
        tconRequestedSizes = pkt[Globals.REPORT_PACKET_QUEUE_LENGTH_FIELD]
        # print("tconRequestedSizes: " + str(tconRequestedSizes))
        # print("report packet: " + str(tconRequestedSizes))
        self.onu_queue_status[onuId] = tconRequestedSizes
        # if(tconRequestedSizes[Globals.TCON1_ID] > 0):
            # print ("requested: " + str(tconRequestedSizes[Globals.TCON1_ID]))
        # self.generate_grant_msg2()
        if self.verbose > Globals.VERB_NO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose,
            )

    def computeTransTime(self, pkt):
        return 0

    def send_to_odn(self, pkt, dest_node):
        """Sends packet to the destination node"""

        # get the connection to the destination node
        # conn = self.conns[self.name]
        # conn.put(pkt)  # put the packet onto the connection
        self.env.process(self.odn.put_grant(pkt, self.prop_delay))
        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.OLT_TYPE:
            
            # self.DICT = pkt.get(Globals.GRANT_PACKET_BWM_FIELD) 
            # self.result = json.dumps(self.DICT)
            # self.Grant_message_OLT[Globals.GRANT_PACKET_BWM_FIELD] = self.result
            # self.Grant_message_OLT.append([self.env.now, pkt])
            pass
            # print(json.dumps(pkt[Globals.GRANT_PACKET_BWM_FIELD]))
            # print (self.result)
        # report as per verbose level
        if self.verbose > Globals.VERB_NO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose,
            )

        

    def initAllocationVars(self, cycle): # updating the values of Assured and Allocated bytes
        self.FB_remaining = Globals.FB
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if cycle % Globals.URLLC_SI_MIN == 0: # recharge URLLC AB min 
                    self.vb_min_matrix[x][Globals.TCON1_ID] = max(0, Globals.URLLC_AB_MIN)
                    self.vb_max_matrix[x] [Globals.TCON1_ID] = Globals.URLLC_RM

                if cycle % Globals.EMBB_SI_MIN == 0:  # recharge EMBB ABmin
                    self.vb_min_matrix[x][Globals.TCON2_ID] = max(0, Globals.EMBB_AB_MIN)
                    self.vb_max_matrix[x] [Globals.TCON2_ID] = Globals.EMBB_RM

                if cycle % Globals.URLLC_SI_MAX == 0:  # recharge EMBB ABmin
                    self.vb_sur_matrix[x][Globals.TCON1_ID] = max(0, Globals.URLLC_AB_SUR)
                if cycle % Globals.EMBB_SI_MAX == 0:  # recharge EMBB ABmin
                    self.vb_sur_matrix[x][Globals.TCON2_ID] = max(0, Globals.EMBB_AB_SUR)
                    
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if cycle % Globals.VIDEO_SI_MIN == 0: # recharge URLLC AB min 
                    self.vb_min_matrix[x][Globals.TCON1_ID] = max(0, Globals.VIDEO_AB_MIN)
                    self.vb_max_matrix[x] [Globals.TCON1_ID] = Globals.VIDEO_RM
                if cycle % Globals.IP_SI_MIN == 0: # recharge URLLC AB min 
                    self.vb_min_matrix[x][Globals.TCON2_ID] = max(0, Globals.IP_AB_MIN)
                    self.vb_max_matrix[x] [Globals.TCON2_ID] = Globals.IP_RM

                if cycle % Globals.VIDEO_SI_MAX == 0: # recharge URLLC AB min 
                    self.vb_sur_matrix[x][Globals.TCON1_ID] = max(0, Globals.VIDEO_AB_SUR)
                if cycle % Globals.VIDEO_SI_MIN == 0: # recharge URLLC AB min 
                    self.vb_sur_matrix[x][Globals.TCON2_ID] = max(0, Globals.IP_AB_SUR)
   

    def generate_grant_msg(self):
        """Process that generates networks packets"""
        # yield self.env.timeout(Globals.SERVICE_INTERVAL)
        while True:
            self.cycle = self.cycle + 1 
            self.initAllocationVars(self.cycle)
            # choose the destination node
            dest_node = Globals.BROADCAST_GRANT_DEST_ID
            self.report_message_ONU.append([self.env.now, copy.deepcopy(self.onu_queue_status)])
            # create the packet
            # print ("requested: " + str(self.onu_queue_status) +"   "+ str(self.env.now))
            bwm = self.generateGrantMsg()
            bwm_print = list(bwm.values())
            # print ("allocated: " + str(bwm) +"   " +  str(self.env.now))
            # if (bwm['1'][Globagrant message:ls.TCON1_ID] > 0 or bwm['1'][Globals.TCON2_ID]) > 0:
                # print ("bwm: " + str(bwm)) 
            pkt = packet.make_grant_packet(self.env, dest_node, self.name, bwm, payload='')
            pkt2 = packet.make_grant_packet(self.env, dest_node, self.name, bwm_print, payload='')
            self.Grant_message_OLT.append([self.env.now, pkt2])
            self.send_to_odn(pkt, 0)
            # add to generated packets monitor
            self.generated.append([self.env.now, pkt])
            # report as per verbose level
            if self.verbose >= Globals.VERB_LO:
                Utils.report(
                    self.env.now,
                    self.name,
                    pkt,
                    self.proc_queue,
                    inspect.currentframe().f_code.co_name,
                    self.verbose,
                )
            yield self.env.timeout(Globals.SERVICE_INTERVAL)
        

    def generate_grant_msg2(self):
        """Process that generates networks packets"""
        self.FB_remaining = Globals.FB
        # choose the destination node
        dest_node = Globals.BROADCAST_GRANT_DEST_ID
        # create the packet
        bwm = self.generateGrantMsg()
        # print ("bwm: " + str(bwm))
        pkt = packet.make_grant_packet(self.env, dest_node, self.name, bwm)
        self.send_to_odn(pkt, 0)
        # add to generated packets monitor
        self.generated.append([self.env.now, pkt])
        # report as per verbose level
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose,
            )

    def shouldAllocateOnu(self, rlAction, x, tconId):
        return True
        if (len(rlAction) == 0):
            return True
        id = int(x) - 1
        
        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            if (rlAction[id + self.onuCount * (tconId - 1)]) == 1:
                return True
            else:
                return False
        else:
            id = int(x) - 1
            if (rlAction[id + self.onuCount * (tconId) + self.onuCount]) == 1:
                return True
            else:
                return False

    def getRlState(self):
        # state = numpy.array([numpy.repeat(0, self.onuCount),numpy.repeat(0, self.onuCount),numpy.repeat(0, self.onuCount),numpy.repeat(0, self.onuCount)], numpy.int32)
        state = numpy.array([numpy.repeat(0, self.onuCount),numpy.repeat(0, self.onuCount)], numpy.integer)
        for x in self.M.G.nodes():
            id = int(x) - 1
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                state[0][id] = self.onu_queue_status[x][Globals.TCON1_ID]
                state[1][id] = self.onu_queue_status[x][Globals.TCON2_ID]
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                state[0][id] = self.onu_queue_status[x][Globals.TCON1_ID]
                state[1][id] = self.onu_queue_status[x][Globals.TCON2_ID]
        return state

    def get_pririty (self, priority_Counter):
        if priority_Counter == 0:
            flags = [True, False, False, False]
        elif priority_Counter == 1:
            flags = [False, True, False, False]
        elif priority_Counter == 2:
            flags = [False, False, True, False]
        elif priority_Counter == 3:
            flags = [False, False, False, True]
        return flags 

    def apply_action(self, rlAction):
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if rlAction[0] == 0:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_AB_MIN
                elif rlAction[0] == 1 :
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   Globals.URLLC_AB_MIN + int((Globals.URLLC_RM - Globals.URLLC_AB_MIN)*rlAction[0])

            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if rlAction[1] == 0:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_AB_MIN
                elif rlAction[1] == 1 :
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   Globals.EMBB_AB_MIN + int((Globals.EMBB_RM - Globals.EMBB_AB_MIN)*rlAction[1])

            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if rlAction[2] == 0:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_AB_MIN
                elif rlAction[2] == 1 :
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   Globals.VIDEO_AB_MIN + int((Globals.VIDEO_RM - Globals.VIDEO_AB_MIN)*rlAction[2])
            
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if rlAction[3] == 0:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_AB_MIN
                elif rlAction[3] == 1 :
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   Globals.IP_AB_MIN + int((Globals.IP_RM - Globals.IP_AB_MIN)*rlAction[3])
    
    def shouldAllocateOnuContinousPerOnu (self, rlAction):
        rlAction = [1000000*i for i in rlAction]
        for x in self.M.G.nodes():
            Tcon1 = 2*int(x) - 2
            Tcon2 = 2*int(x) - 1
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if rlAction[0] > 1 and rlAction[0] < 1.5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_AB_MIN
                elif rlAction[0] > 1.5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   0

                if rlAction[1] > 1 and rlAction[1] < 1.5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_AB_MIN
                elif rlAction[1] > 1.5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   0

                # self.vb_min_matrix[x][Globals.TCON1_ID] =  int((Globals.URLLC_AB_MIN + (Globals.URLLC_RM - Globals.URLLC_AB_MIN)*(max(0, rlAction[Tcon1] - 1)))*(rlAction[Tcon1] - int(rlAction[Tcon1])))
                # self.vb_min_matrix[x][Globals.TCON2_ID] =  int((Globals.EMBB_AB_MIN + (Globals.EMBB_RM - Globals.EMBB_AB_MIN)*(max(0, rlAction[Tcon2] - 1)))*(rlAction[Tcon2] - int(rlAction[Tcon2])))

            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if rlAction[2] > 1 and rlAction[2] < 1.5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_AB_MIN
                elif rlAction[2] > 1.5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   0
                
                if rlAction[3] > 1 and rlAction[3] < 1.5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_AB_MIN
                elif rlAction[3] < 1.5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   0

    def shouldAllocateOnuMultiDiscreteA2C (self, rlAction):
        
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if rlAction[0] == 1:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_AB_MIN
                elif rlAction[0] == 2:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   0

                if rlAction[1] ==1:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_AB_MIN
                elif rlAction[1] == 2:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   0

                # self.vb_min_matrix[x][Globals.TCON1_ID] =  int((Globals.URLLC_AB_MIN + (Globals.URLLC_RM - Globals.URLLC_AB_MIN)*(max(0, rlAction[Tcon1] - 1)))*(rlAction[Tcon1] - int(rlAction[Tcon1])))
                # self.vb_min_matrix[x][Globals.TCON2_ID] =  int((Globals.EMBB_AB_MIN + (Globals.EMBB_RM - Globals.EMBB_AB_MIN)*(max(0, rlAction[Tcon2] - 1)))*(rlAction[Tcon2] - int(rlAction[Tcon2])))

            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if rlAction[2] == 1:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_AB_MIN
                elif rlAction[2] == 2:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   0
                
                if rlAction[3] == 1:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_AB_MIN
                elif rlAction[3] == 2:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_RM                        
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   0

    def shouldAllocateOnuMultiDiscreteA2CService3Acrions (self, rlAction):
        for x in self.M.G.nodes():
            
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if rlAction[0] == 1:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_AB_MIN
                elif rlAction[0] == 5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.5 *Globals.URLLC_RM)
                elif rlAction[0] == 3:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(2/3 *Globals.URLLC_RM)
                elif rlAction[0] == 4:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(5/6 * Globals.URLLC_RM)
                elif rlAction[0] == 2:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_RM                       
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   0

                if rlAction[1] ==1:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_AB_MIN
                elif rlAction[1] == 5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.5*Globals.EMBB_RM)                        
                elif rlAction[1] == 3:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(2/3*Globals.EMBB_RM)
                elif rlAction[1] == 4:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(5/6 *Globals.EMBB_RM)
                elif rlAction[1] == 2:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_RM
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   0

            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if rlAction[2] == 1:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_AB_MIN
                elif rlAction[2] == 5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.5 *Globals.VIDEO_RM)
                elif rlAction[2] == 3:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(2/3 *Globals.VIDEO_RM)
                elif rlAction[2] == 4:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(5/6 * Globals.VIDEO_RM)
                elif rlAction[2] == 2:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_RM                       
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   0
                
                if rlAction[3] ==1:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_AB_MIN
                elif rlAction[3] == 5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.5*Globals.IP_RM)                        
                elif rlAction[3] == 3:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(2/3*Globals.IP_RM)
                elif rlAction[3] == 4:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(5/6 *Globals.IP_RM)
                elif rlAction[3] == 2:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_RM
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   0

                # self.vb_min_matrix[x][Globals.TCON1_ID] =  int((Globals.URLLC_AB_MIN + (Globals.URLLC_RM - Globals.URLLC_AB_MIN)*(max(0, rlAction[Tcon1] - 1)))*(rlAction[Tcon1] - int(rlAction[Tcon1])))
                # self.vb_min_matrix[x][Globals.TCON2_ID] =  int((Globals.EMBB_AB_MIN + (Globals.EMBB_RM - Globals.EMBB_AB_MIN)*(max(0, rlAction[Tcon2] - 1)))*(rlAction[Tcon2] - int(rlAction[Tcon2])))
                # self.vb_min_matrix[x][Globals.TCON1_ID] =  int((Globals.VIDEO_AB_MIN + (Globals.VIDEO_RM - Globals.VIDEO_AB_MIN)*int(max(0, rlAction[Tcon1] - 1)))*(rlAction[Tcon1] - int(rlAction[Tcon1])))
                # self.vb_min_matrix[x][Globals.TCON2_ID] =  int((Globals.IP_AB_MIN + (Globals.IP_RM - Globals.IP_AB_MIN)*(max(0, rlAction[Tcon2] - 1)))*(rlAction[Tcon2] - int(rlAction[Tcon2])))
                # print([self.vb_min_matrix['1'][Globals.TCON1_ID], self.vb_min_matrix['1'][Globals.TCON2_ID], self.vb_min_matrix['2'][Globals.TCON1_ID], self.vb_min_matrix['2'][Globals.TCON2_ID],])

    def shouldAllocateOnuMultiDiscreteA2C6ACtions (self, rlAction):
        for x in self.M.G.nodes():
            Tcon1 = 2*int(x) - 2
            Tcon2 = 2*int(x) - 1
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if rlAction[Tcon1] == 1:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.17 *Globals.URLLC_AB_MIN)
                elif rlAction[Tcon1] == 2:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.33 *Globals.URLLC_AB_MIN)
                elif rlAction[Tcon1] == 3:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_AB_MIN
                elif rlAction[Tcon1] == 4:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.5 *Globals.URLLC_RM)
                elif rlAction[Tcon1] == 5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(2/3 *Globals.URLLC_RM)
                elif rlAction[Tcon1] == 6:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(5/6 * Globals.URLLC_RM)
                elif rlAction[Tcon1] == 7:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_RM                       
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   0

                if rlAction[Tcon2] == 1:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.17 *Globals.EMBB_AB_MIN)
                elif rlAction[Tcon2] == 2:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.33*Globals.EMBB_AB_MIN)                        
                elif rlAction[Tcon2] == 3:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_AB_MIN
                elif rlAction[Tcon2] == 4:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.5*Globals.EMBB_RM)                        
                elif rlAction[Tcon2] == 5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(2/3*Globals.EMBB_RM)
                elif rlAction[Tcon2] == 6:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(5/6 *Globals.EMBB_RM)
                elif rlAction[Tcon2] == 7:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_RM
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   0

    def shouldAllocateOnuMultiDiscreteA2C6ACtions (self, rlAction):
        for x in self.M.G.nodes():
            Tcon1 = 2*int(x) - 2
            Tcon2 = 2*int(x) - 1
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                if rlAction[Tcon1] == 1:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.5 *Globals.URLLC_AB_MIN)
                elif rlAction[Tcon1] == 2:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.75 *Globals.URLLC_AB_MIN)
                elif rlAction[Tcon1] == 3:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.URLLC_AB_MIN
                elif rlAction[Tcon1] == 4:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(1.25 *Globals.URLLC_AB_MIN) 
                elif rlAction[Tcon1] == 5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(1.5 *Globals.URLLC_AB_MIN)
                elif rlAction[Tcon1] == 6:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(1.75 * Globals.URLLC_AB_MIN)
                elif rlAction[Tcon1] == 7:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(2 * Globals.URLLC_AB_MIN)                       
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   int(0.25 *Globals.URLLC_AB_MIN)

                if rlAction[Tcon2] == 1:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.5 *Globals.EMBB_AB_MIN)
                elif rlAction[Tcon2] == 2:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.75*Globals.EMBB_AB_MIN)                        
                elif rlAction[Tcon2] == 3:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.EMBB_AB_MIN
                elif rlAction[Tcon2] == 4:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(1.25*Globals.EMBB_AB_MIN)                        
                elif rlAction[Tcon2] == 5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(1.5*Globals.EMBB_AB_MIN)
                elif rlAction[Tcon2] == 6:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(1.75 *Globals.EMBB_AB_MIN)
                elif rlAction[Tcon2] == 7:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(2 *Globals.EMBB_AB_MIN)
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.25 *Globals.EMBB_AB_MIN)

            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if rlAction[Tcon1] == 1:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.5*Globals.VIDEO_AB_MIN)
                elif rlAction[Tcon1] == 2:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(0.75 *Globals.VIDEO_AB_MIN)
                elif rlAction[Tcon1] == 3:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  Globals.VIDEO_AB_MIN
                elif rlAction[Tcon1] == 4:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(1.25 *Globals.VIDEO_AB_MIN)
                elif rlAction[Tcon1] == 5:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(1.5 *Globals.VIDEO_AB_MIN)
                elif rlAction[Tcon1] == 6:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(1.75 * Globals.VIDEO_AB_MIN)
                elif rlAction[Tcon1] == 7:
                    self.vb_min_matrix[x][Globals.TCON1_ID] =  int(2 * Globals.VIDEO_AB_MIN)                      
                else: 
                    self.vb_min_matrix[x][Globals.TCON1_ID] =   int(0.25 * Globals.VIDEO_AB_MIN)
                
                if rlAction[Tcon2] == 1:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(0.5 *Globals.IP_AB_MIN)
                elif rlAction[Tcon2] == 2:
                    self.vb_min_matrix[x][Globals.TCON2_ID] = int(0.75 *Globals.IP_AB_MIN)                       
                elif rlAction[Tcon2] == 3:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  Globals.IP_AB_MIN
                elif rlAction[Tcon2] == 4:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(1.25*Globals.IP_AB_MIN)                        
                elif rlAction[Tcon2] == 5:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(1.5*Globals.IP_AB_MIN)
                elif rlAction[Tcon2] == 6:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(1.75 *Globals.IP_AB_MIN)
                elif rlAction[Tcon2] == 7:
                    self.vb_min_matrix[x][Globals.TCON2_ID] =  int(2 *Globals.IP_AB_MIN)
                else: 
                    self.vb_min_matrix[x][Globals.TCON2_ID] =   int(0.25 *Globals.IP_AB_MIN)

    def getTotalVbMin(self):
        totalURLLC = 0
        totalEMBB = 0
        totalVideo = 0
        totalIP = 0
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                totalURLLC = totalURLLC + self.vb_min_matrix[x][Globals.TCON1_ID]
                totalEMBB = totalEMBB + self.vb_min_matrix[x][Globals.TCON2_ID]
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                totalVideo = totalVideo + self.vb_min_matrix[x][Globals.TCON1_ID]
                totalIP = totalIP + self.vb_min_matrix[x][Globals.TCON2_ID]
        return totalURLLC,totalEMBB, totalVideo,totalIP


    def generateGrantMsg(self):
        # x = float("{0:.3f}".format(self.env.now))
        # if x / 0.25 == 0:
        print(self.env.now)
        action = []
        priority_Counter = 0
        
        if (self.rlModel != None):
            state = self.getRlState()
            action, _ = self.rlModel.predict(state)
            # self.apply_action(action)
            # self.shouldAllocateOnuContinousPerOnu(action)
            # self.shouldAllocateOnuMultiDiscreteA2CService3Acrions (action)
            # self.shouldAllocateOnuMultiDiscreteA2CService6Actions(action)
            self.shouldAllocateOnuMultiDiscreteA2C6ACtions(action)



        action = 0
        allocationResult= self.initAllocationDict()
        tmp = self.initTmp()


        totalURLLC,totalEMBB, totalVideo, totalIP = self.getTotalVbMin()
        total = totalURLLC + totalEMBB + totalVideo + totalIP
        
        # for x in self.M.G.nodes():
        #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE: 
        #             allocationResult[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID]
        #             allocationResult[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID]
                
        #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
        #             allocationResult[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID]
        #             allocationResult[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID]
        # return allocationResult
        if (total <= self.FB_remaining):
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE: 
                    allocationResult[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID]
                    allocationResult[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID]
                
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                    allocationResult[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID]
                    allocationResult[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID]

        else:
            minIP = Globals.IP_AB_MIN * self.onu2Count
            minVideo = Globals.VIDEO_AB_MIN * self.onu2Count
            minURLLC = Globals.URLLC_AB_MIN * self.onu1Count
            minEMBB = Globals.EMBB_AB_MIN * self.onu1Count
            total2 = minIP + minVideo + minURLLC + minEMBB
            fb_tmp = self.FB_remaining - total2 
            
            if (fb_tmp > totalURLLC - minURLLC):
                tmp = totalURLLC - minURLLC
                total2 = total2 - minURLLC + totalURLLC
                minURLLC = totalURLLC
                fb_tmp = fb_tmp - tmp
            elif fb_tmp > 0:
                total2 = total2 + fb_tmp
                minURLLC = minURLLC + fb_tmp
                fb_tmp = 0
            
            if (fb_tmp > totalEMBB - minEMBB):
                tmp = totalEMBB - minEMBB
                total2 = total2 - minEMBB + totalEMBB
                minEMBB = totalEMBB
                fb_tmp = fb_tmp - tmp
            
            elif fb_tmp > 0:
                total2 = total2 + fb_tmp
                minEMBB = minEMBB + fb_tmp
                fb_tmp = 0

            if (fb_tmp > totalVideo - minVideo):
                tmp = totalVideo - minVideo
                total2 = total2 - minVideo + totalVideo
                minVideo = totalVideo
                fb_tmp = fb_tmp - tmp
            
            elif fb_tmp > 0:
                total2 = total2 + fb_tmp
                minVideo = minVideo + fb_tmp
                fb_tmp = 0           

            if (fb_tmp > totalIP - minIP):
                tmp = totalIP - minIP
                total2 = total2 - minIP + totalIP
                minIP = totalIP
                fb_tmp = fb_tmp - tmp
            
            elif fb_tmp > 0:
                total2 = total2 + fb_tmp
                minIP = minIP + fb_tmp
                fb_tmp = 0 

            minIP = minIP / self.onu2Count
            minVideo = minVideo / self.onu2Count
            minURLLC = minURLLC / self.onu1Count
            minEMBB = minEMBB / self.onu1Count

            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE: 
                    allocationResult[x][Globals.TCON1_ID] = int(minURLLC)
                    allocationResult[x][Globals.TCON2_ID] = int(minEMBB)
                
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                    allocationResult[x][Globals.TCON1_ID] = int(minVideo)
                    allocationResult[x][Globals.TCON2_ID] = int(minIP)

        # tmp, flag = self.allocate_max_all(action)

        # if flag == True:
        #     tmp1 = self.allocateRemaining(action)
        #     allocationResult = self.aggregateAllocation(tmp1, tmp)
        #     return allocationResult
        
        # tmp = self.allocateAssuredUrllc(action)
                    
        # allocationResult = self.aggregateAllocation(allocationResult, tmp)

        # tmp = self.allocateAssuredEmbb(action)
              
        # allocationResult = self.aggregateAllocation(allocationResult, tmp)
     
        # tmp = self.allocateAssuredVideo(action)     

        # allocationResult = self.aggregateAllocation(allocationResult, tmp)

        # tmp = self.allocateAssuredIP(action)
                  
        # allocationResult = self.aggregateAllocation(allocationResult, tmp)

        # tmp = self.allocateSurplusUrllc(action)
                    
        # allocationResult = self.aggregateAllocation(allocationResult, tmp)

        # tmp = self.allocateSurplusEmbb(action)
                
        # allocationResult = self.aggregateAllocation(allocationResult, tmp)

        # tmp = self.allocateSurplusVideo(action)
                
        # allocationResult = self.aggregateAllocation(allocationResult, tmp)
        
        # tmp = self.allocateSurplusIP(action)

        # allocationResult = self.aggregateAllocation(allocationResult, tmp)
        
        # tmp = self.allocateRemaining(action)
        
        # allocationResult = self.aggregateAllocation(allocationResult, tmp)
        return allocationResult

    def allocateRemaining(self, rlAction):
        allocationResult = self.initAllocationDict()
        if self.FB_remaining == 0:
            return allocationResult
        else: 
            while (self.FB_remaining > 0 and (self.cycle % Globals.IP_SI_MIN == 0 or self.cycle % Globals.VIDEO_SI_MIN == 0 or self.cycle % Globals.URLLC_SI_MIN == 0 or self.cycle % Globals.EMBB_SI_MIN==0 )):
                urllc_onu_count = 0
                embb_onu_count = 0
                video_onu_count = 0
                ip_onu_count = 0
                onu_type1_count = 0
                onu_type2_count = 0
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                            continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                        urllc_onu_count = int(urllc_onu_count + 1)
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                        embb_onu_count = int(embb_onu_count + 1)
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                        video_onu_count = int(video_onu_count + 1)
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                        ip_onu_count = int(ip_onu_count + 1)
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                        onu_type2_count = int(onu_type2_count + 1)
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        onu_type1_count = int(onu_type1_count + 1)
                
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                            continue

                y = int(self.FB_remaining*0.3)
                while self.FB_remaining - 4*y < 0:
                    y = int(y - 1)
                # print(y, onu_type1_count, self.env.now)

                for x in self.M.G.nodes():
                    if  (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.FB_remaining > 0 and self.cycle % Globals.URLLC_SI_MIN == 0):
                        newValue = int(y / urllc_onu_count) if urllc_onu_count > 0 else 0
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue -1
                        if y <  urllc_onu_count:
                            newValue = 1 
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        # self.vb_max_matrix[x][Globals.TCON1_ID] = self.vb_max_matrix[x][Globals.TCON1_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue if self.onu_queue_status[x][Globals.TCON1_ID] > 0 else 0
                    
                    if (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] == 0 and self.FB_remaining > 0 and self.cycle % Globals.URLLC_SI_MIN == 0):    
                        newValue = int(y / onu_type1_count)
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue -1
                        if y <  onu_type1_count:
                            newValue = 1
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                    
                    if  (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.FB_remaining > 0 and  self.cycle % Globals.EMBB_SI_MIN == 0):
                        newValue = int(y / embb_onu_count) if embb_onu_count > 0 else 0
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue -1
                        if y <  embb_onu_count:
                            newValue = 1
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        # self.vb_max_matrix[x][Globals.TCON2_ID] = self.vb_max_matrix[x][Globals.TCON2_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue if self.onu_queue_status[x][Globals.TCON2_ID] > 0 else 0
                    
                    if (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] == 0 and self.FB_remaining > 0 and self.cycle % Globals.EMBB_SI_MIN == 0):
                        newValue = int(y/onu_type1_count)
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue -1
                        if y <  onu_type1_count:
                            newValue = 1
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                    
                    if  (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.FB_remaining > 0 and self.cycle % Globals.VIDEO_SI_MIN == 0):
                        newValue = int(y / video_onu_count) if video_onu_count > 0 else 0
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue -1
                        if y == 0 and self.FB_remaining > 0:
                            newValue = 1
                        if y <  video_onu_count:
                            newValue = 1 
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        # self.vb_max_matrix[x][Globals.TCON1_ID] = self.vb_max_matrix[x][Globals.TCON1_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue if self.onu_queue_status[x][Globals.TCON1_ID] > 0 else 0
                    
                    if (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] == 0 and self.FB_remaining > 0 and self.cycle % Globals.VIDEO_SI_MIN == 0):
                        newValue = int(y / onu_type2_count)
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue -1
                        if y <  onu_type2_count:
                            newValue = 1 
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                    
                    if  (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.FB_remaining > 0 and self.cycle % Globals.IP_SI_MIN == 0):
                        newValue = int(y / ip_onu_count) if ip_onu_count > 0 else 0
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue -1
                        if y == 0 and self.FB_remaining > 0:
                            newValue = 1
                        if y <  ip_onu_count:
                            newValue = 1 
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue

                    if (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] == 0 and self.FB_remaining > 0 and self.cycle % Globals.IP_SI_MIN == 0):
                        newValue = int(y / onu_type2_count)
                        while self.FB_remaining - newValue < 0:
                            newValue = newValue - 1
                        if y <  onu_type1_count:
                            newValue = 1
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        # self.vb_max_matrix[x][Globals.TCON2_ID] = self.vb_max_matrix[x][Globals.TCON2_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue if self.onu_queue_status[x][Globals.TCON2_ID] > 0 else 0
                while (self.FB_remaining > 0 and self.cycle % Globals.IP_SI_MIN == 0):
                    for x in self.M.G.nodes():
                        if self.FB_remaining < 0:
                            print('error: fb_remaining is negative')
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.FB_remaining > 0 :
                            newValue = 1
                            while self.FB_remaining - newValue < 0:
                                newValue = newValue -1
                            allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue 
                        if self.FB_remaining==0:
                            break 
                    # print (self.FB_remaining)
        return allocationResult

    
    
    def allocate_max_all(self, rlAction):
        allocationResult = self.initAllocationDict()
        urllc_requested_sum = 0
        embb_requested_sum = 0
        video_requested_sum = 0
        ip_requested_sum = 0
        for x in self.M.G.nodes():
            if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                    continue
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                urllc_requested_sum = urllc_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                embb_requested_sum = embb_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                video_requested_sum = video_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                ip_requested_sum = ip_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
        
        requested_all = urllc_requested_sum + embb_requested_sum + video_requested_sum + ip_requested_sum
        flag = False
        if requested_all < self.FB_remaining:
            flag = True
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                    continue
                if  (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2) and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                    newValue = min(self.vb_max_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
                    allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                    self.FB_remaining = self.FB_remaining - newValue
                    self.vb_max_matrix[x][Globals.TCON1_ID] = self.vb_max_matrix[x][Globals.TCON1_ID] - newValue
                    self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue 
                
                if  (self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2) and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                    newValue = min(self.vb_max_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
                    allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                    self.FB_remaining = self.FB_remaining - newValue
                    self.vb_max_matrix[x][Globals.TCON2_ID] = self.vb_max_matrix[x][Globals.TCON2_ID] - newValue
                    self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
        return allocationResult, flag
            
   
    def aggregateAllocation(self, allocation1, allocation2):
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                allocation1[x][Globals.TCON1_ID] = allocation1[x][Globals.TCON1_ID] + allocation2[x][Globals.TCON1_ID]
                allocation1[x][Globals.TCON2_ID] = allocation1[x][Globals.TCON2_ID] + allocation2[x][Globals.TCON2_ID]
        return allocation1

   
    
    def allocateAssuredUrllc(self, rlAction):
            allocationResult = self.initAllocationDict()
            if (self.FB_remaining <= 0):
                return allocationResult
            # FB > 0
            urllc_AB_min_sum = 0 # sum of remaining grants in the AB min 
            urllc_onu_count = 0 # number of active ONUs requesting for EMBB traffic
            urllc_requested_sum = 0 # totoal request of of embb traffic from all onus
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                    urllc_onu_count = urllc_onu_count + 1
                    urllc_requested_sum = urllc_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
            
            if (urllc_onu_count == 0 or urllc_requested_sum == 0):
                return allocationResult

            # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    if (self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        urllc_AB_min_sum = urllc_AB_min_sum + self.vb_min_matrix[x][Globals.TCON1_ID]
            
            # URLLC allocation
            if (self.FB_remaining >= urllc_AB_min_sum):
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        newValue = min(self.vb_min_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue 
            else:
                urllc_onu_count =0
                urllc_requested_sum = 0
                counter = 0
                while (self.FB_remaining > 0):
                    urllc_onu_count =0
                    urllc_requested_sum = 0
                    counter= counter + 1
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                                continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.vb_min_matrix[x][Globals.TCON1_ID] > 0:
                            urllc_onu_count = urllc_onu_count + 1
                            urllc_requested_sum = urllc_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
                    if (urllc_onu_count == 0 or urllc_requested_sum == 0):
                        break
                    tmp = int(self.FB_remaining / urllc_onu_count) # distribute FB equally over URLLC onus
                    if(tmp == 0 or tmp < urllc_onu_count):
                        break
                    # print ("allocateAssuredUrllc:" + str(counter))
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                            continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.vb_min_matrix[x][Globals.TCON1_ID] > 0:
                            newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp, self.vb_min_matrix[x][Globals.TCON1_ID])
                            allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue
                            self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
                            self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
            return allocationResult

    def allocateAssuredEmbb(self, rlAction):
            allocationResult = self.initAllocationDict()
            if (self.FB_remaining <= 0):
                return allocationResult
            # FB > 0
            embb_AB_min_sum = 0 # sum of remaining grants in the AB min 
            embb_onu_count = 0 # number of active ONUs requesting for EMBB traffic
            embb_requested_sum = 0 # totoal request of of embb traffic from all onus
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                    embb_onu_count = embb_onu_count + 1
                    embb_requested_sum = embb_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
            if (embb_onu_count == 0 or embb_requested_sum == 0):
                return allocationResult 

            # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    if (self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        embb_AB_min_sum = embb_AB_min_sum + self.vb_min_matrix[x][Globals.TCON2_ID]
            
            # URLLC allocation
            if (self.FB_remaining >= embb_AB_min_sum ):
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        newValue = min(self.vb_min_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue 
            else:
                counter = 0
                while (self.FB_remaining > 0 ):
                    embb_onu_count = 0
                    embb_requested_sum = 0
                    counter= counter + 1 
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                                continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.vb_min_matrix[x][Globals.TCON2_ID]>0:
                            embb_onu_count = embb_onu_count + 1
                            embb_requested_sum = embb_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
                    if (embb_onu_count == 0 or embb_requested_sum == 0):
                        break
                    tmp = int(self.FB_remaining / embb_onu_count) # distribute FB equally over URLLC onus
                    if(tmp == 0 or tmp < embb_onu_count):
                        break
                    # print ("allocateAssuredEmbb:" + str(counter))
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                            continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.vb_min_matrix[x][Globals.TCON2_ID] > 0:
                            newValue = min(self.onu_queue_status[x][Globals.TCON2_ID], tmp, self.vb_min_matrix[x][Globals.TCON2_ID])
                            allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue
                            self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
                            embb_requested_sum = embb_requested_sum - newValue
                            self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
                            
            return allocationResult

    def allocateAssuredVideo(self, rlAction):
            allocationResult = self.initAllocationDict()
            if (self.FB_remaining <= 0):
                return allocationResult
            # FB > 0
            video_AB_min_sum = 0 # sum of remaining grants in the AB min 
            video_onu_count = 0 # number of active ONUs requesting for EMBB traffic
            video_requested_sum = 0 # totoal request of of embb traffic from all onus
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                    video_onu_count = video_onu_count + 1
                    video_requested_sum = video_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
            if (video_onu_count == 0 or video_requested_sum == 0):
                return allocationResult 


            # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                    if (self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        video_AB_min_sum = video_AB_min_sum + self.vb_min_matrix[x][Globals.TCON1_ID]
            
            # URLLC allocation
            if (self.FB_remaining >= video_AB_min_sum):
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                        newValue = min(self.vb_min_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue 
            else:
                video_onu_count = 0
                video_requested_sum = 0
                counter = 0
                while (self.FB_remaining > 0):
                    video_onu_count = 0
                    video_requested_sum = 0
                    counter= counter + 1
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                                continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.vb_min_matrix[x][Globals.TCON1_ID] > 0:
                            video_onu_count = video_onu_count + 1
                            video_requested_sum = video_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
                    if (video_onu_count == 0 or video_requested_sum == 0):
                        break

                    tmp = int(self.FB_remaining / video_onu_count) # distribute FB equally over URLLC onus
                    if(tmp == 0 or tmp < video_onu_count):
                        break
                    # print ("allocateAssuredVideo:" + str(counter))
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                            continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.vb_min_matrix[x][Globals.TCON1_ID] > 0:
                            newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp, self.vb_min_matrix[x][Globals.TCON1_ID])
                            allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue
                            self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
                            self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
            return allocationResult

    

    def allocateSurplusUrllc(self, rlAction):
            allocationResult = self.initAllocationDict()
            if (self.FB_remaining <= 0):
                return allocationResult
            # FB > 0
            urllc_AB_sur_sum = 0 # sum of remaining grants in the AB min 
            urllc_onu_count = 0 # number of active ONUs requesting for EMBB traffic
            urllc_requested_sum = 0 # totoal request of of embb traffic from all onus
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                    urllc_onu_count = urllc_onu_count + 1
                    urllc_requested_sum = urllc_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
            
            if (urllc_onu_count == 0 or urllc_requested_sum == 0):
                return allocationResult

            # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    if (self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        urllc_AB_sur_sum = urllc_AB_sur_sum + self.vb_sur_matrix[x][Globals.TCON1_ID]
            
            # URLLC allocation
            if (self.FB_remaining >= urllc_AB_sur_sum ):
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        newValue = min(self.vb_sur_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue 
            else:
                counter = 0
                while (self.FB_remaining > 0 ):
                    urllc_onu_count = 0
                    urllc_requested_sum = 0
                    counter= counter + 1
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                                continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.vb_sur_matrix[x][Globals.TCON1_ID]>0:
                            urllc_onu_count = urllc_onu_count + 1
                            urllc_requested_sum = urllc_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
                    if (urllc_onu_count == 0 or urllc_requested_sum == 0):
                        break
                    tmp = int(self.FB_remaining / urllc_onu_count) # distribute FB equally over URLLC onus
                    if(tmp == 0 or tmp < urllc_onu_count):
                        break
                    # print ("allocateSurplusUrllc:" + str(counter))
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                            continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] and self.vb_sur_matrix[x][Globals.TCON1_ID]> 0:
                            newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp, self.vb_sur_matrix[x][Globals.TCON1_ID])
                            allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue
                            self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
                            self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
                            urllc_requested_sum = urllc_requested_sum - newValue

            return allocationResult
    


    def allocateSurplusEmbb(self, rlAction):
            allocationResult = self.initAllocationDict()
            if (self.FB_remaining <= 0):
                return allocationResult
            # FB > 0
            embb_AB_sur_sum = 0 # sum of remaining grants in the AB min 
            embb_onu_count = 0 # number of active ONUs requesting for EMBB traffic
            embb_requested_sum = 0 # totoal request of of embb traffic from all onus
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                    embb_onu_count = embb_onu_count + 1
                    embb_requested_sum = embb_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
            
            if (embb_onu_count == 0 or embb_requested_sum == 0 or self.FB_remaining < embb_onu_count):
                return allocationResult

            # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    if (self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        embb_AB_sur_sum = embb_AB_sur_sum + self.vb_sur_matrix[x][Globals.TCON2_ID]
            
            # URLLC allocation
            if (self.FB_remaining >= embb_AB_sur_sum  ):
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        newValue = min(self.vb_sur_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_sur_matrix[x][Globals.TCON2_ID] = self.vb_sur_matrix[x][Globals.TCON2_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue 
            else:
                counter = 0
                while (self.FB_remaining > 0 ):
                    embb_requested_sum = 0
                    embb_onu_count = 0
                    counter= counter + 1
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                                continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.vb_sur_matrix[x][Globals.TCON2_ID] > 0:
                            embb_onu_count = embb_onu_count + 1
                            embb_requested_sum = embb_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
                    if (embb_onu_count == 0 or embb_requested_sum == 0):
                        break
                    tmp = int(self.FB_remaining / embb_onu_count) # distribute FB equally over URLLC onus
                    if(tmp == 0 or tmp < embb_onu_count):
                        break
                    # print ("allocateSurplusEmbb:" + str(counter))
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                            continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.vb_sur_matrix[x][Globals.TCON2_ID]>0:
                            newValue = min(self.onu_queue_status[x][Globals.TCON2_ID], tmp, self.vb_sur_matrix[x][Globals.TCON2_ID])
                            allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue
                            self.vb_sur_matrix[x][Globals.TCON2_ID] = self.vb_sur_matrix[x][Globals.TCON2_ID] - newValue
                            self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
                        
            return allocationResult

    def allocateSurplusVideo(self, rlAction):
            allocationResult = self.initAllocationDict()
            if (self.FB_remaining <= 0):
                return allocationResult
            # FB > 0
            video_AB_sur_sum = 0 # sum of remaining grants in the AB min 
            video_onu_count = 0 # number of active ONUs requesting for EMBB traffic
            video_requested_sum = 0 # totoal request of of embb traffic from all onus
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
                    video_onu_count = video_onu_count + 1
                    video_requested_sum = video_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
            
            if (video_onu_count == 0 or video_requested_sum == 0):
                return allocationResult

            # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                    if (self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        video_AB_sur_sum = video_AB_sur_sum + self.vb_sur_matrix[x][Globals.TCON1_ID]
            
            # URLLC allocation
            if (self.FB_remaining >= video_AB_sur_sum ):
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                        newValue = min(self.vb_sur_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
                        allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue 
            else:
                counter = 0
                while (self.FB_remaining > 0 ):
                    video_requested_sum = 0
                    video_onu_count = 0
                    counter= counter + 1
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                                continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.vb_sur_matrix[x][Globals.TCON1_ID] > 0:
                            video_onu_count = video_onu_count + 1
                            video_requested_sum = video_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]
                    if (video_onu_count == 0 or video_requested_sum == 0):
                        break
                    tmp = int(self.FB_remaining / video_onu_count) # distribute FB equally over URLLC onus
                    if(tmp == 0 or tmp < video_onu_count):
                        break
                    # print ("allocateSurplusVideo:" + str(counter))
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
                            continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0 and self.vb_sur_matrix[x][Globals.TCON1_ID] > 0:
                            newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp, self.vb_sur_matrix[x][Globals.TCON1_ID])
                            allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue
                            self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
                            self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
            return allocationResult

    

    def allocateAssuredIP(self, rlAction):
            allocationResult = self.initAllocationDict()
            if (self.FB_remaining <= 0):
                return allocationResult
            # FB > 0
            IP_AB_min_sum = 0 # sum of remaining grants in the AB min 
            IP_onu_count = 0 # number of active ONUs requesting for EMBB traffic
            IP_requested_sum = 0 # totoal request of of embb traffic from all onus
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                    IP_onu_count = IP_onu_count + 1
                    IP_requested_sum = IP_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
            if (IP_onu_count == 0 or IP_requested_sum == 0):
                return allocationResult 

            # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                    if (self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        IP_AB_min_sum = IP_AB_min_sum + self.vb_min_matrix[x][Globals.TCON2_ID]
            
            # URLLC allocation
            if (self.FB_remaining >= IP_AB_min_sum ):
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                        newValue = min(self.vb_min_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue 
            else:
                counter = 0
                
                while (self.FB_remaining > 0 ):
                    IP_requested_sum = 0
                    IP_onu_count = 0
                    counter= counter + 1
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                                continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.vb_min_matrix[x][Globals.TCON2_ID]>0:
                            IP_onu_count = IP_onu_count + 1
                            IP_requested_sum = IP_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
                    if (IP_onu_count == 0 or IP_requested_sum == 0):
                        break
                    tmp = int(self.FB_remaining / IP_onu_count) # distribute FB equally over URLLC onus
                    if(tmp == 0 or tmp < IP_onu_count):
                        break
                    # print ("allocateAssuredIP:" + str(counter))
                    for x in self.M.G.nodes():
                        if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                            continue
                        if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] and self.vb_min_matrix[x][Globals.TCON2_ID] >0:
                            newValue = min(self.onu_queue_status[x][Globals.TCON2_ID], tmp, self.vb_min_matrix[x][Globals.TCON2_ID])
                            allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                            self.FB_remaining = self.FB_remaining - newValue
                            self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
                            self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
            return allocationResult

    def allocateSurplusIP(self, rlAction):
        allocationResult = self.initAllocationDict()
        if (self.FB_remaining <= 0):
            return allocationResult
        # FB > 0
        IP_AB_sur_sum = 0 # sum of remaining grants in the AB min 
        IP_onu_count = 0 # number of active ONUs requesting for EMBB traffic
        IP_requested_sum = 0 # totoal request of of embb traffic from all onus
        for x in self.M.G.nodes():
            if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                    continue
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
                IP_onu_count = IP_onu_count + 1
                IP_requested_sum = IP_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
        if (IP_onu_count == 0 or IP_requested_sum == 0):
            return allocationResult 

        # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                if (self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                    IP_AB_sur_sum = IP_AB_sur_sum + self.vb_sur_matrix[x][Globals.TCON2_ID]
        
        # URLLC allocation
        if (self.FB_remaining >= IP_AB_sur_sum ):
            for x in self.M.G.nodes():
                if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                    continue
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                    newValue = min(self.vb_sur_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
                    allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                    self.FB_remaining = self.FB_remaining - newValue
                    self.vb_sur_matrix[x][Globals.TCON2_ID] = self.vb_sur_matrix[x][Globals.TCON2_ID] - newValue
                    self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue 
        else:
            counter = 0
            
            while (self.FB_remaining > 0 ):
                IP_requested_sum = 0
                IP_onu_count = 0
                counter= counter + 1
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                            continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0 and self.vb_sur_matrix[x][Globals.TCON2_ID]>0:
                        IP_onu_count = IP_onu_count + 1
                        IP_requested_sum = IP_requested_sum  + self.onu_queue_status[x][Globals.TCON2_ID]
                if (IP_onu_count == 0 or IP_requested_sum == 0):
                    break
                tmp = int(self.FB_remaining / IP_onu_count) # distribute FB equally over URLLC onus
                if(tmp == 0 or tmp < IP_onu_count):
                    break
                # print ("allocateAssuredIP:" + str(counter))
                for x in self.M.G.nodes():
                    if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
                        continue
                    if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] and self.vb_sur_matrix[x][Globals.TCON2_ID] >0:
                        newValue = min(self.onu_queue_status[x][Globals.TCON2_ID], tmp, self.vb_sur_matrix[x][Globals.TCON2_ID])
                        allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
                        self.FB_remaining = self.FB_remaining - newValue
                        self.vb_sur_matrix[x][Globals.TCON2_ID] = self.vb_sur_matrix[x][Globals.TCON2_ID] - newValue
                        self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
        return allocationResult


    def initAllocationDict(self):
        allocationResult = {}
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                allocationResult[x] = {Globals.TCON1_ID: 0, Globals.TCON2_ID: 0}
        return allocationResult
    
    def initTmp(self):
        tmp = {}
        for x in self.M.G.nodes():
            if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                tmp[x] = {Globals.TCON1_ID: 0, Globals.TCON2_ID: 0}
        return tmp

    def discard_packet(self, pkt):
        """Discards the packet (puts the packet into the node packet sink)"""

        # place this packet in the node sink
        self.discarded.append([self.env.now, pkt])

    def queue_monitor(self):
        """Queue monitor process"""

        while True:

            # add to queue monitor time now and queue_length
            self.queue_mon.append([self.env.now, len(self.proc_queue)])

            # incur monitor queue delay
            yield self.env.timeout(self.queue_monitor_deltat)

    def num_onu(self):
            for x in self.M.G.nodes():
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    onu_type1_count = onu_type1_count + 1
                if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
                    onu_type2_count = onu_type2_count + 1
            return onu_type1_count + onu_type2_count

 # allocate resources for URLLC
    # def allocateAssuredUrllc(self, rlAction):
    #     allocationResult = self.initAllocationDict()
    #     if (self.FB_remaining <= 0):
    #         return allocationResult
    #     # FB > 0
    #     tcon1_2_AB_min_sum = 0
    #     onu_type1_count = 0
    #     tcon1_2_requested_sum = 0
    #     # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
    #     for x in self.M.G.nodes():
    #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #             if (self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                 tcon1_2_AB_min_sum = tcon1_2_AB_min_sum + self.vb_min_matrix[x][Globals.TCON1_ID]
        
    #     # URLLC allocation
    #     if (self.FB_remaining >= tcon1_2_AB_min_sum):
    #         for x in self.M.G.nodes():
    #             if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                 continue
    #             if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                 newValue = min(self.vb_min_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
    #                 allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
    #                 self.FB_remaining = self.FB_remaining - newValue
    #                 self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
    #                 self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue 
    #     else:
    #         while (self.FB_remaining > 0):
    #             onu_type1_count = 0
    #             tcon1_2_requested_sum = 0
    #             for x in self.M.G.nodes():
    #                 if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                         continue
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
    #                     onu_type1_count = onu_type1_count + 1
    #                     tcon1_2_requested_sum = tcon1_2_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]

    #             if (onu_type1_count == 0 or tcon1_2_requested_sum == 0):
    #                 break
    #             tmp = int(self.FB_remaining / onu_type1_count) # distribute FB equally over URLLC onus
    #             if(tmp == 0):
    #                 break
    #             for x in self.M.G.nodes():
    #                 if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                     continue
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                     newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp)
    #                     allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
    #                     self.FB_remaining = self.FB_remaining - newValue
    #                     self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
    #                     self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
    #     return allocationResult

    # def allocateAssuredTcon3(self, rlAction):
    #     allocationResult = self.initAllocationDict()
    #     if (self.FB_remaining <= 0):
    #         return allocationResult
    #     # FB > 0
    #     tcon1_3_AB_min_sum = 0
    #     tcon2_3_AB_min_sum = 0
    #     onu_type1_count = 0
    #     onu_type2_count = 0   
    #     tcon1_3_requested_sum = 0
    #     tcon2_3_requested_sum = 0

    #     # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
    #     for x in self.M.G.nodes():
    #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #             if (self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
    #                 tcon1_3_AB_min_sum = tcon1_3_AB_min_sum + self.vb_min_matrix[x][Globals.TCON2_ID]
    #                 tcon1_3_requested_sum = tcon1_3_requested_sum + self.onu_queue_status[x][Globals.TCON2_ID]
    #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #             if (self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                 tcon2_3_AB_min_sum = tcon2_3_AB_min_sum + self.vb_min_matrix[x][Globals.TCON1_ID]
    #                 tcon2_3_requested_sum = tcon2_3_requested_sum + self.onu_queue_status[x][Globals.TCON1_ID]

    #     # eMBB and allocation
    #     if (self.FB_remaining >= (tcon1_3_AB_min_sum + tcon2_3_AB_min_sum)):
    #         for x in self.M.G.nodes():
    #             if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                 if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
    #                     continue
    #                 newValue = min(self.vb_min_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
    #                 allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
    #                 self.FB_remaining = self.FB_remaining - newValue
    #                 self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
    #                 self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
                    
    #             if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #                 if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                     continue
    #                 newValue = min(self.vb_min_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
    #                 allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
    #                 self.FB_remaining = self.FB_remaining - newValue
    #                 self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
    #                 self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
    #     else:
    #         while (self.FB_remaining > 0):
    #             onu_type1_count = 0
    #             onu_type2_count = 0
    #             tcon1_3_requested_sum = 0
    #             tcon2_3_requested_sum = 0
    #             for x in self.M.G.nodes():
    #                 if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)
    #                     and not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
    #                         continue
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
    #                     onu_type1_count = onu_type1_count + 1
    #                     tcon1_3_requested_sum = tcon1_3_requested_sum + self.onu_queue_status[x][Globals.TCON2_ID]
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
    #                     onu_type2_count = onu_type2_count + 1
    #                     tcon2_3_requested_sum = tcon2_3_requested_sum + self.onu_queue_status[x][Globals.TCON1_ID]

    #             if ((onu_type1_count == 0 or tcon1_3_requested_sum == 0) and (onu_type2_count == 0 or tcon2_3_requested_sum == 0)):
    #                 break

    #             tmp = int(self.FB_remaining / (onu_type1_count + onu_type2_count)) # distribute FB equally over URLLC onus
    #             if(tmp == 0):
    #                 break
    #             for x in self.M.G.nodes():
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                     if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON2_ID)):
    #                         continue
    #                     newValue = min(self.onu_queue_status[x][Globals.TCON2_ID], tmp)
    #                     allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
    #                     self.FB_remaining = self.FB_remaining - newValue
    #                     self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
    #                     self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
    
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #                     if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                         continue
    #                     newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp)
    #                     allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue 
    #                     self.FB_remaining = self.FB_remaining - newValue
    #                     self.vb_min_matrix[x][Globals.TCON1_ID] = self.vb_min_matrix[x][Globals.TCON1_ID] - newValue
    #                     self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue

    #     return allocationResult
        # def allocateSurplusUrllc(self, rlAction):
    #     allocationResult = self.initAllocationDict()
    #     if (self.FB_remaining <= 0):
    #         return allocationResult
    #     # FB > 0
    #     urllc_AB_sur_sum = 0
    #     onu_type1_count = 0
    #     urllc_requested_sum = 0
    #     # get the summation of AB min for tcon2 (URLLC) and 3(EMBB and Video)
    #     for x in self.M.G.nodes():
    #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #             if (self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                 urllc_AB_sur_sum = urllc_AB_sur_sum + self.vb_sur_matrix[x][Globals.TCON1_ID]
        
    #     # URLLC allocation
    #     if (self.FB_remaining >= urllc_AB_sur_sum):
    #         for x in self.M.G.nodes():
    #             if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                 continue
    #             if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                 newValue = min(self.vb_sur_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
    #                 allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
    #                 self.FB_remaining = self.FB_remaining - newValue
    #                 self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
    #                 self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue 
    #     else:
    #         while (self.FB_remaining > 0):
    #             onu_type1_count = 0
    #             urllc_requested_sum = 0
    #             for x in self.M.G.nodes():
    #                 if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                         continue
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
    #                     onu_type1_count = onu_type1_count + 1
    #                     urllc_requested_sum = urllc_requested_sum  + self.onu_queue_status[x][Globals.TCON1_ID]

    #             if (onu_type1_count == 0 or urllc_requested_sum == 0):
    #                 break
    #             tmp = int(self.FB_remaining / onu_type1_count) # distribute FB equally over URLLC onus
    #             if(tmp == 0):
    #                 break
    #             for x in self.M.G.nodes():
    #                 if (not self.shouldAllocateOnu(rlAction, x, Globals.TCON1_ID)):
    #                     continue
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                     newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp)
    #                     allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
    #                     self.FB_remaining = self.FB_remaining - newValue
    #                     self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
    #                     self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
    #     return allocationResult
    # def allocateSurplusTcon3(self, rlAction):
    #     allocationResult = self.initAllocationDict() #{onu1 {TCON1: 0, TCON2: 0}}
    #     if (self.FB_remaining <= 0):
    #         return allocationResult
    #     # FB > 0
        
    #     tcon1_3_AB_sur_sum = 0
    #     tcon2_3_AB_sur_sum = 0
    #     onu_type1_count = 0
    #     onu_type2_count = 0
    #     tcon1_3_requested_sum = 0
    #     tcon2_3_requested_sum = 0

    #     # get the summation of AB min for tcon2 (URLLC) and TCON3(eMBB and Video)
    #     for x in self.M.G.nodes():
    #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #             tcon1_3_AB_sur_sum = tcon1_3_AB_sur_sum + self.vb_sur_matrix[x][Globals.TCON2_ID]
    #             tcon1_3_requested_sum = tcon1_3_requested_sum + self.onu_queue_status[x][Globals.TCON2_ID]
    #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #             tcon2_3_AB_sur_sum = tcon2_3_AB_sur_sum + self.vb_sur_matrix[x][Globals.TCON1_ID]
    #             tcon2_3_requested_sum = tcon2_3_requested_sum + self.onu_queue_status[x][Globals.TCON1_ID]
    #     # eMBB and video surplus allocation
    #     if (self.FB_remaining >= (tcon1_3_AB_sur_sum + tcon2_3_AB_sur_sum)):
    #         for x in self.M.G.nodes():
    #             if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                 newValue = min(self.vb_sur_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
    #                 allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
    #                 self.FB_remaining = self.FB_remaining - newValue
    #                 self.vb_sur_matrix[x][Globals.TCON2_ID] = self.vb_sur_matrix[x][Globals.TCON2_ID] - newValue
    #                 self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
                    
    #             if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #                 newValue = min(self.vb_sur_matrix[x][Globals.TCON1_ID], self.onu_queue_status[x][Globals.TCON1_ID])
    #                 allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
    #                 self.FB_remaining = self.FB_remaining - newValue
    #                 self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
    #                 self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue
    #     else:
    #         while (self.FB_remaining > 0):
    #             onu_type1_count = 0
    #             onu_type2_count = 0
    #             tcon1_3_requested_sum = 0
    #             tcon2_3_requested_sum = 0
    #             for x in self.M.G.nodes():
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE and self.onu_queue_status[x][Globals.TCON2_ID] > 0:
    #                     onu_type1_count = onu_type1_count + 1
    #                     tcon1_3_requested_sum = tcon1_3_requested_sum + self.onu_queue_status[x][Globals.TCON2_ID]
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON1_ID] > 0:
    #                     onu_type2_count = onu_type2_count + 1
    #                     tcon2_3_requested_sum = tcon2_3_requested_sum + self.onu_queue_status[x][Globals.TCON1_ID]
                
    #             if ((onu_type1_count == 0 or tcon1_3_requested_sum == 0) and (onu_type2_count == 0 or tcon2_3_requested_sum == 0)):
    #                 break
    #             tmp = int(self.FB_remaining / (onu_type1_count + onu_type2_count)) # distribute FB equally over TCON3 onus
    #             if(tmp == 0):
    #                 break
    #             for x in self.M.G.nodes():
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
    #                     newValue = min(self.onu_queue_status[x][Globals.TCON2_ID], tmp)
    #                     allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
    #                     self.FB_remaining = self.FB_remaining - newValue
    #                     self.vb_sur_matrix[x][Globals.TCON2_ID] = self.vb_sur_matrix[x][Globals.TCON2_ID] - newValue
    #                     self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
    
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #                     newValue = min(self.onu_queue_status[x][Globals.TCON1_ID], tmp)
    #                     allocationResult[x][Globals.TCON1_ID] = allocationResult[x][Globals.TCON1_ID] + newValue
    #                     self.FB_remaining = self.FB_remaining - newValue
    #                     self.vb_sur_matrix[x][Globals.TCON1_ID] = self.vb_sur_matrix[x][Globals.TCON1_ID] - newValue
    #                     self.onu_queue_status[x][Globals.TCON1_ID] = self.onu_queue_status[x][Globals.TCON1_ID] - newValue

    #     return allocationResult

       
    # def allocateAssuredIP(self, rlAction):
    #     allocationResult = self.initAllocationDict() #{onu1 {TCON1: 0, TCON2: 0}}
    #     if (self.FB_remaining <= 0):
    #         return allocationResult
    #     # FB > 0
    #     #                 
    #     tcon2_4_AB_sum = 0
    #     tcon2_4_requested_sum = 0
    #     onu_type2_count = 0
        

    #     for x in self.M.G.nodes():
    #         if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #             tcon2_4_AB_sum = tcon2_4_AB_sum + self.vb_min_matrix[x][Globals.TCON2_ID]
    #             tcon2_4_requested_sum = tcon2_4_requested_sum + self.onu_queue_status[x][Globals.TCON2_ID]

    #     # eMBB and video surplus allocation
    #     if (self.FB_remaining >= tcon2_4_AB_sum):
    #         for x in self.M.G.nodes():                
    #             if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #                 newValue = min(self.vb_min_matrix[x][Globals.TCON2_ID], self.onu_queue_status[x][Globals.TCON2_ID])
    #                 allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue
    #                 self.FB_remaining = self.FB_remaining - newValue
    #                 self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
    #                 self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
    #     else:
    #         while (self.FB_remaining > 0):
    #             onu_type2_count = 0
    #             tcon2_4_requested_sum = 0
    #             for x in self.M.G.nodes():
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2 and self.onu_queue_status[x][Globals.TCON2_ID] > 0 :
    #                     onu_type2_count = onu_type2_count + 1
    #                     tcon2_4_requested_sum = tcon2_4_requested_sum + self.onu_queue_status[x][Globals.TCON2_ID]
                
    #             if ((onu_type2_count == 0 or tcon2_4_requested_sum == 0)):
    #                 break
    #             tmp = int(self.FB_remaining / (onu_type2_count)) # distribute FB equally over TCON3 onus
    #             if(tmp == 0):
    #                 break
    #             for x in self.M.G.nodes():    
    #                 if self.M.G.nodes[x][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
    #                     newValue = min(self.onu_queue_status[x][Globals.TCON2_ID], tmp)
    #                     allocationResult[x][Globals.TCON2_ID] = allocationResult[x][Globals.TCON2_ID] + newValue 
    #                     self.FB_remaining = self.FB_remaining - newValue
    #                     self.vb_min_matrix[x][Globals.TCON2_ID] = self.vb_min_matrix[x][Globals.TCON2_ID] - newValue
    #                     self.onu_queue_status[x][Globals.TCON2_ID] = self.onu_queue_status[x][Globals.TCON2_ID] - newValue
    #     return allocationResult

