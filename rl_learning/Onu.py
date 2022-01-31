from os import name
import random
import collections  # provides 'deque'
import inspect
from  math import ceil
import networkx as nx
from networkx.algorithms import coloring
import Globals
import Utils
import PacketGenerator
from enum import Enum
import packet
import PPBP
import numpy as np

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
        self.myClock = M.myClock
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
        self.counterU = 0
        self.counterE = 0
        self.counterV = 0
        self.counterI = 0
        self.counterRep = 0

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
        self.queue_lenght_URLLC1 = collections.deque()
        self.queue_lenght_eMBB1 = collections.deque()
        self.queue_lenght_Video1 = collections.deque()
        self.queue_lenght_IP1 = collections.deque()
        self.received = collections.deque()
        self.forwarded = collections.deque()
        self.discarded = collections.deque()
        self.Grant_message_OLT = collections.deque()
        self.meanDealy_all = collections.deque()
        self.action_all = collections.deque()



        # counters for sent/received packets (key=node name)
        self.pkt_sent = {}
        self.pkt_recv = {}

        # dt interval for incoming queue monitoring
        self.queue_monitor_deltat = Globals.QUEUE_MONITOR_DELTAT
        self.Bursts = {Globals.TCON1_ID:[], Globals.TCON2_ID:[]}

        self.createWorkloadGenerators(40, Globals.SIM_TIME, self.pkt_rate[1])

    def createWorkloadGenerators(self, average, time, step):
        self.Bursts = {Globals.TCON1_ID:[], Globals.TCON2_ID:[]}
        if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            B, t = PPBP.createPPBPTrafficGen(1/0.1, 5, 0.8, average/12.5, self.burstsSourcesCountONU[Globals.TCON1_ID], step * 10)
            self.Bursts[Globals.TCON1_ID].append(B)
            self.Bursts[Globals.TCON1_ID].append(t)
            B, t = PPBP.createPPBPTrafficGen(1/0.1, 5, 0.8, average/12.5, self.burstsSourcesCountONU[Globals.TCON2_ID], step * 10)
            self.Bursts[Globals.TCON2_ID].append(B)
            self.Bursts[Globals.TCON2_ID].append(t)
        else:
            B, t = PPBP.createPPBPTrafficGen(1/0.1, 5, 0.8, average/12.5, self.burstsSourcesCountONU2[Globals.TCON1_ID], step * 10)
            self.Bursts[Globals.TCON1_ID].append(B)
            self.Bursts[Globals.TCON1_ID].append(t)
            B, t = PPBP.createPPBPTrafficGen(1/0.1, 5, 0.8, average/12.5, self.burstsSourcesCountONU2[Globals.TCON2_ID], step * 10)
            self.Bursts[Globals.TCON2_ID].append(B)
            self.Bursts[Globals.TCON2_ID].append(t)

        print ("max:" + str(np.max(self.Bursts[Globals.TCON1_ID][0])))
        print ("min:" + str(np.min(self.Bursts[Globals.TCON1_ID][0])))
        print ("mean:" + str(np.mean(self.Bursts[Globals.TCON1_ID][0])))
        print ("len:" + str(len(self.Bursts[Globals.TCON1_ID][0])))
        print ("max:" + str(np.max(self.Bursts[Globals.TCON2_ID][0])))
        print ("min:" + str(np.min(self.Bursts[Globals.TCON2_ID][0])))
        print ("mean:" + str(np.mean(self.Bursts[Globals.TCON2_ID][0])))
        print ("len:" + str(len(self.Bursts[Globals.TCON2_ID][0])))

    def reset(self):
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
        }

        self.queue_lenght_URLLC = collections.deque()
        self.queue_lenght_eMBB = collections.deque()
        self.queue_lenght_Video = collections.deque()
        self.queue_lenght_IP = collections.deque()
        
        self.counterU = 0
        self.counterE = 0
        self.counterV = 0
        self.counterI = 0
        self.counterRep = 0


    def add_conn(self, c, conn):
        """Adds a connection from this node to the node 'c'"""
        self.conns[c] = conn
        # self.customConnections[c] = customConnection
        self.pkt_sent[c] = 0
        self.pkt_recv[c] = 0


    def if_recv(self, src, pkt):
        """Node receive interface from node 'c'"""
        # the connection from 'self.name' to 'c'
            # increment the counter for this sending node
            # put the packet in the processing queue
        if(pkt[Globals.OLT_PACKET_TYPE_FIELD] == Globals.GRANT_PACKET_TYPE):
            # self.pkt_recv[src] += 1
            self.process_grant_msg(pkt)


        # self.proc_queue.append(pkt)
        # report as per verbose level
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.myClock,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose, )

    def process_grant_msg(self, pkt):
        pkt[Globals.DEST_TIME_STAMP] = self.myClock.now
        # packet terminates here- put it in the receive queue
        # self.received.append([self.myClock.now, pkt])
        Globals.appendDequeue(  self.received, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

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
                pkt[Globals.REPORT_TIME] = self.myClock.now
                pktDelay = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
                transTime = self.computeTransTime(pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD])
                if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    # self.reported_ONU1_T1_URLLC.append([pktDelay, pkt])
                    Globals.appendDequeue(self.reported_ONU1_T1_URLLC, [pktDelay, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)
                else:
                    # self.reported_ONU2_T2_Video.append([pktDelay, pkt])
                    Globals.appendDequeue(self.reported_ONU2_T2_Video, [pktDelay, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)
                
                tmp5 = ""
                tmp5 = tmp5.ljust(pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD], "1")
                payload1 = payload1 + tmp5
                sizeToSend = sizeToSend - pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD]
            else:
                tmp = pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] - sizeToSend # tmp is the size remained unsent 
                pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = tmp
                tmp2 = ""
                tmp3 = "" 
                tmp2= tmp2.ljust(tmp, "1") # tmp2 is a string of the unsent size
                tmp3 = tmp3.ljust(sizeToSend, "1") # tmp3 is a string of the sent size 
                # pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD] = tmp2
                self.tcon_queue[Globals.TCON1_ID].appendleft(pkt)
                payload1 = payload1 + tmp3
                sizeToSend = 0
                break
                # self.tcon_queue[Globals.TCON1_ID].appendleft(pkt)
                # break
            # else:
            #     payload1 = payload1 + pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD]
            #     # pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] - sizeToSend 
            #     sizeToSend = 0
            #     # self.tcon_queue[Globals.TCON1_ID].appendleft(pkt)
        self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] - len(payload1)
        if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            # self.queue_lenght_URLLC.append([self.myClock.now, self.tcon_length[Globals.TCON1_ID]])
            # self.queue_lenght_URLLC1.append([self.myClock.now, self.tcon_length[Globals.TCON1_ID]])
            Globals.appendDequeue(self.queue_lenght_URLLC, [self.myClock.now, self.tcon_length[Globals.TCON1_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
            Globals.appendDequeue(self.queue_lenght_URLLC1, [self.myClock.now, self.tcon_length[Globals.TCON1_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
    
        else:
            # self.queue_lenght_Video.append([self.myClock.now, self.tcon_length[Globals.TCON1_ID]])
            # self.queue_lenght_Video1.append([self.myClock.now, self.tcon_length[Globals.TCON1_ID]])
            Globals.appendDequeue(self.queue_lenght_Video, [self.myClock.now, self.tcon_length[Globals.TCON1_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
            Globals.appendDequeue(self.queue_lenght_Video1, [self.myClock.now, self.tcon_length[Globals.TCON1_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
        
        payload2 = ""
        sizeToSend = self.tcon_allocated_size[Globals.TCON2_ID]
        while(sizeToSend > 0 and len(self.tcon_queue[Globals.TCON2_ID]) > 0):
            pkt = self.tcon_queue[Globals.TCON2_ID].popleft()
            if (pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] <= sizeToSend):
                pkt[Globals.REPORT_TIME] = self.myClock.now
                pktDelay = pkt[Globals.REPORT_TIME] - pkt[Globals.TIME_STAMP]
                transTime = self.computeTransTime(pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD])
                if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
                    # self.reported_ONU1_T2_eMBB.append([pktDelay, pkt])      
                    Globals.appendDequeue(self.reported_ONU1_T2_eMBB, [pktDelay, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)
                else:
                    # self.reported_ONU2_T3_IP.append([pktDelay, pkt])
                    Globals.appendDequeue(self.reported_ONU2_T3_IP, [pktDelay, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)
                
                tmp4 = ""
                tmp4 = tmp4.ljust(pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD], "1")
                payload2 = payload2 + tmp4
                sizeToSend = sizeToSend - pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD]
            else:
                tmp = pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] - sizeToSend # tmp is the size remained unsent 
                pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = tmp
                tmp2 = ""
                tmp3 = "" 
                tmp2 = tmp2.ljust(tmp, "1") # tmp2 is a string with leanght of the unsent size
                tmp3 = tmp3.ljust(sizeToSend, "1") # tmp3 is a string with leanght of the sent size 
                # pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD] = tmp2
                self.tcon_queue[Globals.TCON2_ID].appendleft(pkt)
                payload2 = payload2 + tmp3
                sizeToSend = 0
                break
                # self.tcon_queue[Globals.TCON2_ID].appendleft(pkt)
                # break
            # else:
            #     payload2 = payload2 + pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD]
            #     # pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] - sizeToSend 
            #     sizeToSend = 0
            #     # self.tcon_queue[Globals.TCON2_ID].appendleft(pkt)
        self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] - len(payload2) # decrease the size of tcon with pkt size        
        if self.M.G.nodes[self.name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            # self.queue_lenght_eMBB.append([self.myClock.now, self.tcon_length[Globals.TCON2_ID]])
            # self.queue_lenght_eMBB1.append([self.myClock.now, self.tcon_length[Globals.TCON2_ID]])            
            Globals.appendDequeue(self.queue_lenght_eMBB, [self.myClock.now, self.tcon_length[Globals.TCON2_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
            Globals.appendDequeue(self.queue_lenght_eMBB1, [self.myClock.now, self.tcon_length[Globals.TCON2_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
        else: 
            # self.queue_lenght_IP.append([self.myClock.now, self.tcon_length[Globals.TCON2_ID]])
            # self.queue_lenght_IP1.append([self.myClock.now, self.tcon_length[Globals.TCON2_ID]])            
            Globals.appendDequeue(self.queue_lenght_IP, [self.myClock.now, self.tcon_length[Globals.TCON2_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
            Globals.appendDequeue(self.queue_lenght_IP1, [self.myClock.now, self.tcon_length[Globals.TCON2_ID]], Globals.QUEUE_SIZE_LIMIT, True, False)
            



    def computeTransTime(self, payloadSize):
        transTime = len(payloadSize) / Globals.MAX_LINK_CAPACITY_Bps
        # print(str(transTime) +'    ' +   str(len(payloadSize)))
        return transTime

    def send_to_odn(self, pkt, dest_node):
        """Sends packet to the destination node"""

        # get the connection to the destination node
        # conn = self.conns[self.name]
        # conn.put(pkt)  # put the packet onto the connection
        self.env.process(self.odn.put_request(pkt, self.prop_delay, self.name))
        # if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
        # self.report_message_ONU.append([self.myClock.now, pkt])
        Globals.appendDequeue(self.report_message_ONU, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        # elif self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
        # report as per verbose level
        if self.verbose > Globals.VERB_NO:
            Utils.report(
                self.myClock.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose,
            )
    

    def URLLC_ONU1_T1_pkt_gen_process(self):
        """Process that generates networks packets"""   
        
        # pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        # choose the destination node
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.urllc_generatePayload(Globals.TCON1_ID)
        # payload = self.generatePayload(Globals.TCON1_ID)
        tcon_id = Globals.TCON1_ID
        pkt = packet.make_payload_packet(self.myClock.now, self.counterU, dest_node, self.name, self.name, payload, Globals.TCON1_ID)       
        self.counterU = self.counterU + 1       
        # self.generated.append([self.myClock.now, pkt])
        Globals.appendDequeue(self.generated, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)
        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            # self.generated_ONU1_T1_URLLC.append([self.myClock.now, pkt])        
            Globals.appendDequeue(self.generated_ONU1_T1_URLLC, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)
        self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] + len(payload)
        # self.tcon_queue[Globals.TCON1_ID].append(pkt)
        Globals.appendDequeue(self.tcon_queue[Globals.TCON1_ID], pkt, Globals.QUEUE_SIZE_LIMIT, True, False)
        if self.tcon_length[Globals.TCON1_ID] >= Globals.queue_cutoff_bytes_urllc:
                pkt = self.tcon_queue[Globals.TCON1_ID].popleft()
                self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] - len(payload)
                # self.discarded_URLLC.append([self.myClock.now, pkt])
                Globals.appendDequeue(self.discarded_URLLC, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # add to generated packets monitor            
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.myClock.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose)

    def eMBB_ONU1_T2_pkt_gen_process(self):
        """Process that generates networks packets"""   
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.embb_generatePayload(Globals.TCON2_ID)
        # payload = self.generatePayload(Globals.TCON2_ID)
        self.tcon_length[Globals.TCON2_ID] = self.tcon_length[
            Globals.TCON2_ID] + len(payload)
        tcon_id = Globals.TCON2_ID
        pkt = packet.make_payload_packet(
            self.myClock.now, self.counterE, dest_node, self.name, self.name, payload, Globals.TCON2_ID)
        self.counterE = self.counterU + 1           
        # self.generated.append([self.myClock.now, pkt])
        Globals.appendDequeue(self.generated, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE:
            # self.generated_ONU1_T2_eMBB.append([self.myClock.now, pkt])
            Globals.appendDequeue(self.generated_ONU1_T2_eMBB, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        # if the forward queue is full, discard the last received
        # packet to open space in the queue
        # self.tcon_queue[Globals.TCON2_ID].append(pkt)
        Globals.appendDequeue(self.tcon_queue[Globals.TCON2_ID], pkt, Globals.QUEUE_SIZE_LIMIT, True, False)
        if self.tcon_length[Globals.TCON2_ID] >= Globals.queue_cutoff_bytes_embb:
            pkt = self.tcon_queue[Globals.TCON2_ID].popleft()
            self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] - len(payload)
            # self.discarded_eMBB.append([self.myClock.now, pkt])
            Globals.appendDequeue(self.discarded_eMBB, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # report as per verbose level
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.myClock.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose)

    def Video_ONU2_T2_pkt_gen_process(self):
        """Process that generates networks packets"""   
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.video_generatePayload(Globals.TCON1_ID)
        # payload = self.generatePayload(Globals.TCON1_ID)
        self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] + len(payload)
        tcon_id = Globals.TCON1_ID
        pkt = packet.make_payload_packet(self.myClock.now, self.counterV, dest_node, self.name, self.name, payload, Globals.TCON2_ID)
        self.counterV = self.counterV + 1           
        # self.generated.append([self.myClock.now, pkt])
        Globals.appendDequeue(self.generated, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
            # self.generated_ONU2_T2_Video.append([self.myClock.now, pkt])
            Globals.appendDequeue(self.generated_ONU2_T2_Video, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        # if the forward queue is full, discard the last received
        # packet to open space in the queue
        # self.tcon_queue[Globals.TCON1_ID].append(pkt)
        Globals.appendDequeue(self.tcon_queue[Globals.TCON1_ID], pkt, Globals.QUEUE_SIZE_LIMIT, True, False)

        if self.tcon_length[Globals.TCON1_ID] >= Globals.queue_cutoff_bytes_video:
            pkt = self.tcon_queue[Globals.TCON1_ID].popleft()
            self.tcon_length[Globals.TCON1_ID] = self.tcon_length[Globals.TCON1_ID] - len(payload)
            # self.discarded_Video.append([self.myClock.now, pkt])
            Globals.appendDequeue(self.discarded_Video, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # report as per verbose level
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.myClock.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose)
    
    def IP_ONU2_T3_pkt_gen_process(self):
        """Process that generates networks packets"""
  
        pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)/10
        # choose the destination node
        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        # create the packet
        payload = self.ip_generatePayload(Globals.TCON2_ID)
        # payload = self.generatePayload(Globals.TCON2_ID)
        tcon_id = Globals.TCON2_ID
        pkt = packet.make_payload_packet(self.myClock.now, self.counterI, dest_node, self.name, self.name, payload, Globals.TCON2_ID)
        self.counterI = self.counterI + 1
        # self.generated.append([self.myClock.now, pkt])
        Globals.appendDequeue(self.generated, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        if  self.M.G.nodes[pkt.get(Globals.SOURCE)][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2:
            # self.generated_ONU2_T3_IP.append([self.myClock.now, pkt])
            Globals.appendDequeue(self.generated_ONU2_T3_IP, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] + len(payload)
        # if the forward queue is full, discard the last received
        # packet to open space in the queue
        # self.tcon_queue[Globals.TCON2_ID].append(pkt)
        Globals.appendDequeue(self.tcon_queue[Globals.TCON2_ID], pkt, Globals.QUEUE_SIZE_LIMIT, True, False)

        if self.tcon_length[Globals.TCON2_ID] >= Globals.queue_cutoff_bytes_ip:
                pkt = self.tcon_queue[Globals.TCON2_ID].popleft()
                self.tcon_length[Globals.TCON2_ID] = self.tcon_length[Globals.TCON2_ID] - len(payload)
                # self.discarded_IP.append([self.myClock.now, pkt])
                Globals.appendDequeue(self.discarded_IP, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        # add the node generated packet to the head of the queue
        # self.proc_queue.appendleft(pkt)
        # add to generated packets monitor
        
        if self.verbose >= Globals.VERB_LO:
            Utils.report(
                self.myClock.now,
                self.name,
                pkt,
                self.proc_queue,
                inspect.currentframe().f_code.co_name,
                self.verbose,
            )

    def gen_report_packet(self):

        dest_node = Globals.BROADCAST_REPORT_DEST_ID
        neededSizes = self.substractAllocation(self.tcon_length, self.tcon_allocated_size)
        # print ("report: " + str(neededSizes))
        # create the packet
        pkt = packet.make_report_packet(
            self.myClock.now, self.counterRep, dest_node, self.name, neededSizes, self.name, payload='')
        self.counterRep = self.counterRep + 1
        # self.report_message_ONU.append([self.myClock.now, pkt])
        Globals.appendDequeue(self.report_message_ONU, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)

        return pkt

        # # add the node generated packet to the head of the queue
        # # self.proc_queue.appendleft(pkt)
        # self.send_to_odn(pkt, "none")

        # # self.proc_queue.appendleft(pkt)

        # # add to generated packets monitor

        # # report as per verbose level
        # if self.verbose >= Globals.VERB_LO:
        #     Utils.report(
        #         self.myClock.now,
        #         self.name,
        #         pkt,
        #         self.proc_queue,
        #         inspect.currentframe().f_code.co_name,
        #         self.verbose,
        #     )


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
        data = ' '.join(map(str, data))
        return data 

    def discard_packet(self, pkt):
        """Discards the packet (puts the packet into the node packet sink)"""
        # place this packet in the node sink
        # self.discarded.append([self.myClock.now, pkt])
        Globals.appendDequeue(self.discarded, [self.myClock.now, pkt], Globals.QUEUE_SIZE_LIMIT, True, False)
    
    def urllc_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        payloadSize = self.Bursts[trconId][0].pop()
        # data = [1] * ceil(ceil(payloadSize)*Globals.URLLC_SCALE*0.125)
        data = [1] * ceil(payloadSize)
        data = ' '.join(map(str, data))
        return data

    def embb_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        payloadSize = self.Bursts[trconId][0].pop()
        # data = [1] * ceil(int(payloadSize)*Globals.EMBB_SCALE*0.125)
        # data = [1] * 15
        data = [1] * ceil(payloadSize)
        data = ' '.join(map(str, data))
        return data

    def video_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        payloadSize = self.Bursts[trconId][0].pop()
        # data = [1] * ceil(int(payloadSize)*Globals.VIDEO_SCALE*0.125)
        # data = [1] * 15
        data = [1] * ceil(payloadSize)
        data = ' '.join(map(str, data))
        return data
    
    def ip_generatePayload(self, trconId):
        if (len(self.Bursts[trconId][0]) == 0):
            return ""
        payloadSize = self.Bursts[trconId][0].pop()
        # data = [1] * ceil(ceil(payloadSize)*Globals.IP_SCALE*0.125)
        # data = [1] * 15
        data = [1] * ceil(payloadSize)
        data = ' '.join(map(str, data))
        return data

    