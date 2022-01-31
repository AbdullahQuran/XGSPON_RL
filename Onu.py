from os import name
import random
import collections  # provides 'deque'
import inspect
from  math import ceil
import networkx as nx

from rl_learning import Globals
import Utils
import PacketGenerator
from enum import Enum
import packet
from rl_learning import PPBP 
import simpy

class ONU(object):
    """
    Models an optical network unit
    """
    array = []
    tm = []
    B = []
    sum  = []
    size = 1

    def __init__(self, env, M, node_name, verbose):
        self.counter = 0
        self.M = M
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
        self.burstsSourcesCountONU = {Globals.TCON1_ID: 1, Globals.TCON2_ID: 1}
        self.burstsSourcesCountONU2 = {Globals.TCON1_ID: 1, Globals.TCON2_ID: 1}
        self.start_time = -1
        self.end_time = -1
        self.prop_delay = 0.001

        self.conns = {}
        # self.customConnections = {}
        self.verbose = verbose

        # processing queue and queue length monitor
        self.proc_queue = collections.deque()
        # queues to store packets of each TCON
        self.tcon_queue = {
            Globals.TCON1_ID: collections.deque(),
            Globals.TCON2_ID: collections.deque(),
        }

       
        # length of each TCON queue. Initially zeroes
        self.tcon_length = {
            Globals.TCON1_ID: 0,
            Globals.TCON2_ID: 0,
        }  # {1: 500. tcon_id: size}
        
        # the allocated size in bytes (chars) for each TCON by the OLT. Initially zeroes
        self.tcon_allocated_size = {
            Globals.TCON1_ID: 0,
            Globals.TCON2_ID: 0,
        }  # {1: 500. tcon_id: size}
        
        if (self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE):
            self.amplificationFactor = {
                # Globals.TCON1_ID: 630/2/1.9,
                # Globals.TCON2_ID: 350/2/1.9,
                # Globals.TCON1_ID: 630/2/1.9/16/10/2,
                # Globals.TCON2_ID: 630/2/1.9/16/1.8/10/2
                Globals.TCON1_ID: 480/2/1.9/446.4,
                Globals.TCON2_ID: 480/2/1.9/446.4,
                 }
        else:
            # self.amplificationFactor = {
            #     Globals.TCON1_ID: 0,
            #     Globals.TCON2_ID: 460/2/1.9,
            # }
            self.amplificationFactor = {
                Globals.TCON1_ID: 0,
                Globals.TCON2_ID: 480/2/1.9/446.4,
            }
            # self.amplificationFactor = {
            #     Globals.TCON1_ID: 0,
            #     Globals.TCON2_ID: 630/2/1.9/16/1.36/10/2,
            # }

        self.queue_mon = collections.deque()

        # packets persistent storage
        self.generated = collections.deque()
        self.generated_ONU1_T1_URLLC = collections.deque()
        self.generated_ONU1_T2_eMBB = collections.deque()
        self.generated_ONU2_T2_Video = collections.deque()
        self.generated_ONU2_T3_IP = collections.deque()
        self.reported_ONU1_T1_URLLC = collections.deque()
        self.reported_ONU1_T2_eMBB = collections.deque()
        self.reported_ONU2_T2_Video = collections.deque()
        self.reported_ONU2_T3_IP = collections.deque()
        self.report_message_ONU = collections.deque()
        self.discarded_URLLC = collections.deque()
        self.discarded_eMBB = collections.deque()
        self.discarded_Video = collections.deque()
        self.discarded_IP = collections.deque()
        self.queue_lenght_URLLC = collections.deque()
        self.queue_lenght_eMBB = collections.deque()
        self.queue_lenght_Video = collections.deque()
        self.queue_lenght_IP = collections.deque()
        self.received = collections.deque()
        self.forwarded = collections.deque()
        self.discarded = collections.deque()
        self.Grant_message_OLT = collections.deque()
        self.meanDealy_all = collections.deque()

        self.lastReportSize = {
            Globals.TCON1_ID: 0,
            Globals.TCON2_ID: 0,
        }

        # counters for sent/received packets (key=node name)
        self.pkt_sent = {}
        self.pkt_recv = {}

        # dt interval for incoming queue monitoring
        self.queue_monitor_deltat = Globals.QUEUE_MONITOR_DELTAT
        self.Bursts = {Globals.TCON1_ID:[], Globals.TCON2_ID:[]}
        averageSize = 40
        if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            # B, t = PPBP.createPPBPTrafficGen(self.M.simTime/0.1, 5, 0.8, averageSize/12.5, self.burstsSourcesCountONU[Globals.TCON1_ID], self.pkt_rate[1] * 10)
            self.Bursts[Globals.TCON1_ID].append(M.B[self.name][Globals.TCON1_ID])
            # self.Bursts[Globals.TCON1_ID].append(t)
            # B, t = PPBP.createPPBPTrafficGen(self.M.simTime/0.1, 5, 0.8, averageSize/12.5, self.burstsSourcesCountONU[Globals.TCON2_ID], self.pkt_rate[1] * 10)
            self.Bursts[Globals.TCON2_ID].append(M.B[self.name][Globals.TCON2_ID])
            # self.Bursts[Globals.TCON2_ID].append(t)
        else:
            # B, t = PPBP.createPPBPTrafficGen(self.M.simTime/0.1, 5, 0.8, averageSize/12.5, self.burstsSourcesCountONU2[Globals.TCON1_ID], self.pkt_rate[1] * 10)
            self.Bursts[Globals.TCON1_ID].append(M.B[self.name][Globals.TCON1_ID])
            # self.Bursts[Globals.TCON1_ID].append(t)
            # B, t = PPBP.createPPBPTrafficGen(self.M.simTime/0.1, 5, 0.8, averageSize/12.5, self.burstsSourcesCountONU2[Globals.TCON2_ID], self.pkt_rate[1] * 10)
            self.Bursts[Globals.TCON2_ID].append(M.B[self.name][Globals.TCON2_ID])
            # self.Bursts[Globals.TCON2_ID].append(t)

        self.genrationCounter = {
            Globals.TCON1_ID: random.randint(0, len(M.B)-1),
            Globals.TCON2_ID: random.randint(0, len(M.B)-1),
        }



    def add_conn(self, c, conn):
        """Adds a connection from this node to the node 'c'"""
        self.conns[c] = conn
        # self.customConnections[c] = customConnection
        self.pkt_sent[c] = 0
        self.pkt_recv[c] = 0

    def if_up(self):
        """Activates interfaces- this sets up SimPy processes"""

        # start recv processes on all receiving connections
        # for c in self.conns:
        #     self.env.process(self.if_recv(c))

        # # activate packet generator, packet forwarding, queue monitoring
        # if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
        #     self.env.process(self.URLLC_ONU1_T1_pkt_gen_process())
        #     self.env.process(self.eMBB_ONU1_T2_pkt_gen_process())
        # else:
        #     self.env.process(self.Video_ONU2_T2_pkt_gen_process())
        #     self.env.process(self.IP_ONU2_T3_pkt_gen_process())
    
        # self.env.process(self.send_report_packet())
        # self.env.process(self.forward_process())
        # self.env.process(self.queue_monitor())
        self.env.process(self.fullCycle())

    def emptyBuffer(self, reportTime):
        for pkt in self.tcon_queue[Globals.TCON1_ID]:
            pkt[Globals.REPORT_TIME] = self.env.now
            pktDelay = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                self.reported_ONU1_T1_URLLC.append([pktDelay, pkt])
            else:
                self.reported_ONU2_T2_Video.append([pktDelay, pkt])

        for pkt in self.tcon_queue[Globals.TCON2_ID]:
            pkt[Globals.REPORT_TIME] = self.env.now
            pktDelay = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
            if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                self.reported_ONU1_T2_eMBB.append([pktDelay, pkt])
            else:
                self.reported_ONU2_T3_IP.append([pktDelay, pkt])



    def fullCycle(self):
        self.xgponCounter = 0 
        while(True):
            if (self.M.oltType == 'g'):
                report_cycles = [0] 
                grant_cycles = [Globals.SERVICE_INTERVAL - 1]
                grant_cycles_r = [(x + Globals.PROPAGATION_TIME) % Globals.SERVICE_INTERVAL for x in grant_cycles]

                if self.env.now < Globals.GEN_TIME:
                    if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        self.URLLC_ONU1_T1_pkt_gen_process()
                        self.eMBB_ONU1_T2_pkt_gen_process()
                    else:
                        # self.Video_ONU2_T2_pkt_gen_process()
                        self.IP_ONU2_T3_pkt_gen_process()
                self.forward_process()
            
                if (self.xgponCounter % Globals.SERVICE_INTERVAL == 0):
                    self.send_report_packet()
                if (self.xgponCounter % Globals.SERVICE_INTERVAL in grant_cycles_r and self.xgponCounter >= Globals.SERVICE_INTERVAL):
                    self.tcon_allocated_size[Globals.TCON1_ID] = 0
                    self.tcon_allocated_size[Globals.TCON2_ID] = 0
                    for c in self.conns:
                        yield self.env.process(self.if_recv(c))
                
                yield self.env.timeout(Globals.XGSPON_CYCLE)
            
            elif (self.M.oltType == 'ibu'):
                report_cycles = [0, 2, 4] 
                grant_cycles = [1, 3, 5]   
                grant_cycles_r = [x + Globals.PROPAGATION_TIME for x in grant_cycles]
                if self.env.now < Globals.GEN_TIME:
                    if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        self.URLLC_ONU1_T1_pkt_gen_process()
                        self.eMBB_ONU1_T2_pkt_gen_process()
                    else:
                        # self.Video_ONU2_T2_pkt_gen_process()
                        self.IP_ONU2_T3_pkt_gen_process()
                self.forward_process()
            
                if (self.xgponCounter % Globals.SERVICE_INTERVAL in report_cycles):
                    self.send_report_packet()
                
                if (self.xgponCounter % Globals.SERVICE_INTERVAL in grant_cycles_r and self.xgponCounter >= Globals.SERVICE_INTERVAL):
                    self.tcon_allocated_size[Globals.TCON1_ID] = 0
                    self.tcon_allocated_size[Globals.TCON2_ID] = 0
                    for c in self.conns:
                        yield self.env.process(self.if_recv(c))
                
                yield self.env.timeout(Globals.XGSPON_CYCLE)


            else:
                # report_cycles = [0,1,2,3,4,5,6,7,8,9] 
                # grant_cycles = [0,1,2,3,4,5,6,7,8,9]  
                report_cycles = [0] 
                grant_cycles = [1]
                grant_cycles_r = [(x + Globals.PROPAGATION_TIME) % Globals.SERVICE_INTERVAL for x in grant_cycles]
     
                if self.env.now < Globals.GEN_TIME:
                    if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                        self.URLLC_ONU1_T1_pkt_gen_process()
                        self.eMBB_ONU1_T2_pkt_gen_process()
                    else:
                        # self.Video_ONU2_T2_pkt_gen_process()
                        self.IP_ONU2_T3_pkt_gen_process()
                self.forward_process()
                if (self.xgponCounter % Globals.SERVICE_INTERVAL in grant_cycles_r):
                    self.tcon_allocated_size[Globals.TCON1_ID] = 0
                    self.tcon_allocated_size[Globals.TCON2_ID] = 0
                    for c in self.conns:
                        yield self.env.process(self.if_recv(c))
                
                if (self.xgponCounter % Globals.SERVICE_INTERVAL in report_cycles):
                    self.send_report_packet()
                
                yield self.env.timeout(Globals.XGSPON_CYCLE)
            
            
            self.xgponCounter =  self.xgponCounter + 1

    
    def if_recv(self, c):
        """Node receive interface from node 'c'"""
        # the connection from 'self.name' to 'c'
        conn = self.conns[c]
        # pick up any incoming packets
        pkt = yield conn.get()
        # increment the counter for this sending node
        # put the packet in the processing queue
        if(pkt[Globals.OLT_PACKET_TYPE_FIELD] == Globals.GRANT_PACKET_TYPE):
            self.pkt_recv[c] += 1
            self.process_grant_msg(pkt)


        # self.proc_queue.append(pkt)
        # report as per verbose level
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose, )

    def process_grant_msg(self, pkt):
        # yield self.env.timeout(self.proc_delay)
        pkt[Globals.DEST_TIME_STAMP] = self.env.now
        # packet terminates here- put it in the receive queue
        self.received.append([self.env.now, pkt])
        bwm = pkt[Globals.GRANT_PACKET_BWM_FIELD]
        # print ("grant message: " + str(bwm))
        allocatedSizes = bwm[self.name]
        if (allocatedSizes[Globals.TCON1_ID] > 0):
            self.tcon_allocated_size[Globals.TCON1_ID] = allocatedSizes[Globals.TCON1_ID]
        else:
            self.tcon_allocated_size[Globals.TCON1_ID] = 0
        if (allocatedSizes[Globals.TCON2_ID] > 0):
            self.tcon_allocated_size[Globals.TCON2_ID] = allocatedSizes[Globals.TCON2_ID]
        else:
            self.tcon_allocated_size[Globals.TCON2_ID] = 0
        
        
        # self.send_report_packet2()
        
        # for onu in bwm:
        #     if onu[Globals.GRANT_PACKET_ONU_ID_FIELD] == self.name:
        #         for tconId in bwm[self.name]
        #         self.start_time = onu[Globals.GRANT_PACKET_ONU_START_FIELD]
        #         self.end_time = onu[Globals.GRANT_PACKET_ONU_END_FIELD]


    def forward_process(self):
        """Node packet forwarding process"""

        payload1 = ""
        sizeToSend = self.tcon_allocated_size[Globals.TCON1_ID] 
        while(sizeToSend > 0 and len(self.tcon_queue[Globals.TCON1_ID]) > 0):
            pkt = self.tcon_queue[Globals.TCON1_ID].popleft()
            
            if (pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] <= sizeToSend):
                pkt[Globals.REPORT_TIME] = self.env.now
                transTime = self.computeTransTime(pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD])
                pktDelay = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP] + transTime
                if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    
                    self.reported_ONU1_T1_URLLC.append([pktDelay, pkt])
                else:

                    self.reported_ONU2_T2_Video.append([pktDelay, pkt])
                
                tmp5 = ""
                tmp5 = tmp5.ljust(int(pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD]), "1")
                payload1 = payload1 + tmp5
                sizeToSend = sizeToSend - pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD]
                self.tcon_allocated_size[Globals.TCON1_ID] = self.tcon_allocated_size[Globals.TCON1_ID] - pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD]
            
            ## fragmentation ##
            else: # packet size > grant size --> 1- fragement the packet and send in portions, 2- report time is the time of the last portion 
                
                tmp = pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] - sizeToSend # tmp is the size remained unsent 
                pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = tmp
                tmp2 = ""
                tmp3 = "" 
                tmp2= tmp2.ljust(int(tmp), "1") # tmp2 is a string of the unsent size
                tmp3 = tmp3.ljust(int(sizeToSend), "1") # tmp3 is a string of the sent size 
                # pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD] = tmp2
                self.tcon_queue[Globals.TCON1_ID].appendleft(pkt)
                payload1 = payload1 + tmp3
                self.tcon_allocated_size[Globals.TCON1_ID] = self.tcon_allocated_size[Globals.TCON1_ID] - sizeToSend
                sizeToSend = 0
                break
            # else:
            #     payload1 = payload1 + pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD]
            #     # pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] - sizeToSend 
            #     sizeToSend = 0
            #     # self.tcon_queue[Globals.TCON1_ID].appendleft(pkt)
        
        
        ### updaing size of tcon_length
        self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] - len(payload1)
        if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            self.queue_lenght_URLLC.append([self.env.now, self.tcon_length[Globals.TCON1_ID]])            
        else:
            self.queue_lenght_Video.append([self.env.now, self.tcon_length[Globals.TCON1_ID]])            

        
        payload2 = ""
        sizeToSend = self.tcon_allocated_size[Globals.TCON2_ID]
        while(sizeToSend > 0 and len(self.tcon_queue[Globals.TCON2_ID]) > 0): # if we have allocation bytes and there are packets in tcon_queue
            pkt = self.tcon_queue[Globals.TCON2_ID].popleft() # pop the fisrts pkt from left
            if (pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] <= sizeToSend): # check its size agains AB
                pkt[Globals.REPORT_TIME] = self.env.now # report it
                pktDelay = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
                transTime = self.computeTransTime(pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD])
                pktDelay = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP] + transTime
                # yield self.env.timeout(transTime)
                if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:

                    self.reported_ONU1_T2_eMBB.append([pktDelay, pkt])      
                else:

                    self.reported_ONU2_T3_IP.append([pktDelay, pkt])
                tmp4 = ""
                tmp4 = tmp4.ljust(int(pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD]), "1")
                payload2 = payload2 + tmp4
                sizeToSend = sizeToSend - pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] # decrease the AB with the size of the packets
                self.tcon_allocated_size[Globals.TCON2_ID] = self.tcon_allocated_size[Globals.TCON2_ID] - pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD]

            else:
                tmp = pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] - sizeToSend # tmp is the size remained unsent 
                pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = tmp
                tmp2 = ""
                tmp3 = "" 
                tmp2 = tmp2.ljust(int(tmp), "1") # tmp2 is a string with leanght of the unsent size
                tmp3 = tmp3.ljust(int(sizeToSend), "1") # tmp3 is a string with leanght of the sent size 
                # pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD] = tmp2
                self.tcon_queue[Globals.TCON2_ID].appendleft(pkt)
                payload2 = payload2 + tmp3
                self.tcon_allocated_size[Globals.TCON2_ID] = self.tcon_allocated_size[Globals.TCON2_ID] - sizeToSend
                sizeToSend = 0
                break
        
        self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] - len(payload2) # decrease the size of tcon with pkt size        
        if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            self.queue_lenght_eMBB.append([self.env.now, self.tcon_length[Globals.TCON2_ID]])            
        else: 
            self.queue_lenght_IP.append([self.env.now, self.tcon_length[Globals.TCON2_ID]])            

        # if (pkt):
        #     trans_time = self.computeTransTime(pkt) + self.prop_delay
        #     yield self.env.timeout(trans_time)
        # else:
        #     yield self.env.timeout(0.001)


    def computeTransTime(self, payloadSize):
        transTime = len(payloadSize) / Globals.MAX_LINK_CAPACITY_Bps
        # print(str(transTime) +'    ' +   str(len(payloadSize)))
        return transTime

    def send_to_odn(self, pkt, dest_node):
        """Sends packet to the destination node"""

        # get the connection to the destination node
        # conn = self.conns[self.name]
        # conn.put(pkt)  # put the packet onto the connection

        # self.env.process(self.odn.put_request(pkt, self.prop_delay, self.name))
        self.odn.put_request(pkt, self.prop_delay, self.name)
        # if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
        self.report_message_ONU.append([self.env.now, pkt])
        # elif self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
        # self.env.process(self.odn.put_request(pkt, self.prop_delay, self.name))
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
    

    def URLLC_ONU1_T1_pkt_gen_process(self):
        """Process that generates networks packets"""   
        counter = 0
        # pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        # print ('urllc gen time =  ' + str(self.env.now))
        # choose the destination node
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.urllc_generatePayload(Globals.TCON1_ID)
        # print(self.queue_size)
        pkt = packet.make_payload_packet(self.env,counter, dest_node, self.name, self.name, payload, Globals.TCON1_ID)       
        counter = counter + 1
        self.generated.append([self.env.now, pkt])
        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            self.generated_ONU1_T1_URLLC.append([self.env.now, pkt])        
        self.lastReportSize[Globals.TCON1_ID] = len(payload)
        # self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] + len(payload)
        # self.tcon_queue[Globals.TCON1_ID].append(pkt)
        if self.tcon_length[Globals.TCON1_ID] >= Globals.queue_cutoff_bytes_urllc:
            # pkt = self.tcon_queue[Globals.TCON1_ID].popleft()
            # self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] - len(payload)
            self.discarded_URLLC.append([self.env.now, pkt])
        else:
            self.tcon_queue[Globals.TCON1_ID].append(pkt)
            self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] + len(payload)


        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # add to generated packets monitor            
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose)

    def eMBB_ONU1_T2_pkt_gen_process(self):
        """Process that generates networks packets"""   
        counter = 0
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        # print ('embb gen time =  ' + str(self.env.now))
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.embb_generatePayload(Globals.TCON2_ID)
        self.lastReportSize[Globals.TCON2_ID] = len(payload)
        
        pkt = packet.make_payload_packet(
            self.env, counter, dest_node, self.name, self.name, payload, Globals.TCON2_ID)           
        self.generated.append([self.env.now, pkt])
        counter = counter + 1
        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            self.generated_ONU1_T2_eMBB.append([self.env.now, pkt])
        # if the forward queue is full, discard the last received
        # packet to open space in the queue
        # self.tcon_queue[Globals.TCON2_ID].append(pkt)
        if self.tcon_length[Globals.TCON2_ID] >= Globals.queue_cutoff_bytes_embb:
            # pkt = self.tcon_queue[Globals.TCON2_ID].popleft()
            # self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] - len(payload)
            self.discarded_eMBB.append([self.env.now, pkt])
        else:
            self.tcon_queue[Globals.TCON2_ID].append(pkt)
            self.tcon_length[Globals.TCON2_ID] = self.tcon_length[
            Globals.TCON2_ID] + len(payload)
        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # report as per verbose level
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose)
        
    def Video_ONU2_T2_pkt_gen_process(self):
        """Process that generates networks packets"""   
        counter = 0
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        # print ('video gen time =  ' + str(self.env.now))
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.video_generatePayload(Globals.TCON1_ID)
        self.lastReportSize[Globals.TCON1_ID] = len(payload)
        pkt = packet.make_payload_packet(self.env, counter,  dest_node, self.name, self.name, payload, Globals.TCON1_ID)           
        self.generated.append([self.env.now, pkt])
        counter = counter +1 
        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
            self.generated_ONU2_T2_Video.append([self.env.now, pkt])
        # if the forward queue is full, discard the last received
        # packet to open space in the queue
        # self.tcon_queue[Globals.TCON1_ID].append(pkt)
        # self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] + len(payload)
        if self.tcon_length[Globals.TCON1_ID] >= Globals.queue_cutoff_bytes_video:
            # pkt = self.tcon_queue[Globals.TCON1_ID].popleft()
            # self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] - len(payload)
            self.discarded_Video.append([self.env.now, pkt])
        else:
            self.tcon_queue[Globals.TCON1_ID].append(pkt)
            self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] + len(payload)

        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # report as per verbose level
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose)

            
    def IP_ONU2_T3_pkt_gen_process(self):
        """Process that generates networks packets"""
        counter = 0
            
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        # print ('IP gen time =  ' + str(self.env.now))
        # choose the destination node
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.ip_generatePayload(Globals.TCON2_ID)
        pkt = packet.make_payload_packet(self.env, counter, dest_node, self.name, self.name, payload, Globals.TCON2_ID)
        counter = counter +1 
        self.generated.append([self.env.now, pkt])
        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
            self.generated_ONU2_T3_IP.append([self.env.now, pkt])
        
        self.lastReportSize[Globals.TCON2_ID] = len(payload)
        # self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] + len(payload)
        # if the forward queue is full, discard the last received
        # packet to open space in the queue
        # self.tcon_queue[Globals.TCON2_ID].append(pkt)
        if self.tcon_length[Globals.TCON2_ID] >= Globals.queue_cutoff_bytes_ip:
            # pkt = self.tcon_queue[Globals.TCON2_ID].popleft()
            # self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] - len(payload)
            self.discarded_IP.append([self.env.now, pkt])
        else:
            self.tcon_queue[Globals.TCON2_ID].append(pkt)
            self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] + len(payload)


        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # add to generated packets monitor
        
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.env.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose,
            )

    def send_report_packet(self):

        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        neededSizes = self.substractAllocation(self.tcon_length, self.tcon_allocated_size)
        # neededSizes = self.tcon_length
        # print ("report: " + str(neededSizes))
        # create the packet
        pkt = packet.make_report_packet(
            self.env, dest_node, self.name, neededSizes, self.name, payload='', reportSize=self.lastReportSize)

        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # yield self.env.timeout(self.prop_delay)
        self.send_to_odn(pkt, "none")

        # self.proc_queue.appendleft(pkt)

        # add to generated packets monitor

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



    def send_report_packet2(self):
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        neededSizes = self.substractAllocation(self.tcon_length, self.tcon_allocated_size)
        # print ("report: " + str(neededSizes))
        # create the packet
        pkt = packet.make_report_packet(self.env, dest_node, self.name, neededSizes, self.name)
        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # yield self.env.timeout(self.prop_delay)
        self.send_to_odn(pkt, "none")
        # self.proc_queue.appendleft(pkt)
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
    # def make_pkt(self, dest_node):
    #     """Creates a network packet"""
    #     pkt = {}
    #     pkt[Globals.TIME_STAMP] = self.env.now
    #     pkt[Globals.ID] = Utils.gen_id()
    #     pkt[Globals.SOURCE] = self.name
    #     pkt[Globals.DEST_NODE] = dest_node
    #     pkt[Globals.HOP_NODE] = Globals.NONE  # the initial value
    #     pkt[Globals.DEST_TIME_STAMP] = -1.0  # the initial value
    #     pkt[Globals.NO_HOPS] = 0  # the initial value
    #     return pkt

    def substractAllocation(self, allocation1, allocation2):
        tmp = self.initAllocationDict()
        tmp[Globals.TCON1_ID] = max((allocation1[Globals.TCON1_ID] - allocation2[Globals.TCON1_ID]), 0)
        tmp[Globals.TCON2_ID] = max((allocation1[Globals.TCON2_ID] - allocation2[Globals.TCON2_ID]), 0)
        return tmp

    def initAllocationDict(self):
        allocationResult = {Globals.TCON1_ID: 0, Globals.TCON2_ID: 0}
        return allocationResult

    def generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        payloadSize = self.Bursts[trconId][0].pop()
        data = [1] * int(payloadSize)
        # data = [1] * 50
        data = ' '.join(map(str, data))
        return data 
    
    def urllc_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        
        # payloadSize = self.Bursts[trconId][0].pop()
        payloadSize = self.Bursts[trconId][0][self.genrationCounter[trconId]] * self.amplificationFactor[Globals.TCON1_ID]
        self.genrationCounter[trconId] = (self.genrationCounter[trconId] + 1) % len(self.Bursts[trconId][0])
        if (payloadSize > Globals.URLLC_AB_MIN + Globals.URLLC_AB_SUR):
            payloadSize = Globals.URLLC_AB_MIN + Globals.URLLC_AB_SUR
        # payloadSize = 25
        # data = [1] * ceil(ceil(payloadSize)*Globals.URLLC_SCALE*0.125)
        # data = [1] * 50
        data = [1] * int(payloadSize)
        data = ' '.join(map(str, data))
        return data

    def embb_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        # payloadSize = self.Bursts[trconId][0].pop()
        payloadSize = self.Bursts[trconId][0][self.genrationCounter[trconId]] * self.amplificationFactor[Globals.TCON2_ID]
        self.genrationCounter[trconId] = (self.genrationCounter[trconId] + 1) % len(self.Bursts[trconId][0])       
        if (payloadSize > Globals.EMBB_AB_MIN + Globals.EMBB_AB_SUR):
            payloadSize = Globals.EMBB_AB_MIN + Globals.EMBB_AB_SUR
        # payloadSize = 25
        data = [1] * int(payloadSize)
        # data = [1] * 50
        data = ' '.join(map(str, data))
        return data

    def video_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        # payloadSize = self.Bursts[trconId][0].pop()
        payloadSize = self.Bursts[trconId][0][self.genrationCounter[trconId]]
        self.genrationCounter[trconId] = (self.genrationCounter[trconId] + 1) % len(self.Bursts[trconId][0])
        if (payloadSize > Globals.VIDEO_AB_MIN + Globals.VIDEO_AB_SUR):
            payloadSize = Globals.VIDEO_AB_MIN + Globals.VIDEO_AB_SUR
        data = [1] * int(payloadSize)

        # data = [1] * 50
        data = ' '.join(map(str, data))
        return data
    
    def ip_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        # payloadSize = self.Bursts[trconId][0].pop()
        payloadSize = self.Bursts[trconId][0][self.genrationCounter[trconId]] * self.amplificationFactor[Globals.TCON2_ID]
        self.genrationCounter[trconId] = (self.genrationCounter[trconId] + 1) % len(self.Bursts[trconId][0])
        if (payloadSize > Globals.IP_AB_MIN + Globals.IP_AB_SUR):
            payloadSize = Globals.IP_AB_MIN + Globals.IP_AB_SUR
        data = [1] * int(payloadSize)

        # data = [1] * 50
        data = ' '.join(map(str, data))
        return data

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
    