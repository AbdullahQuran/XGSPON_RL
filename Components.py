"""Components.py
"""

import random
import collections  # provides 'deque'
import inspect
import networkx as nx
from rl_learning import Globals
import Utils
import PacketGenerator
from enum import Enum




class ChannelType(Enum):
    UP = 1
    DOWN = 2



class CustomChannel():
    def __init__(self, channel, channelType, assignedWaveLength, checkTime):
        self.channel = channel
        self.type = channelType
        self.assignedWaveLength = assignedWaveLength
        self.checkTime = checkTime


# class CustomFiber():
#     def __init__(self, channelUp, channelDown, waveLengthListUp, waveLengthListDown, delay):
#         self.channelUp = channelUp
#         self.channelDown = channelDown
#         self.waveLengthListUp = waveLengthListUp
#         self.waveLengthListDown = waveLengthListDown
#         self.delay = delay


class Channel(object):
    """
    Models a connection between two nodes
    """
    def __init__(self, env, delay, conn_out, conn_in, assignedWaveLength, checkTime):
        self.env = env
        self.delay = delay
        self.conn_out = conn_out
        self.conn_in = conn_in
        self.assignedWaveLength = assignedWaveLength
        self.checkTime = checkTime
        
    def latency(self, pkt):
        """Latency for putting packet onto the wire
        """
        yield self.env.timeout(Globals.CYCLE_TIME)
        self.conn_out.put(pkt)

    def latency_conn_in(self, pkt):
        """Latency for putting packet onto the wire
        """
        yield self.env.timeout(Globals.CYCLE_TIME)
        self.conn_in.put(pkt)

    def put(self, pkt):
        """Puts the packet 'pkt' onto the wire
        """
        self.env.process(self.latency(pkt))

    def put_conn_in(self, pkt):
        self.env.process(self.latency_conn_in(pkt))


    def get(self):
        """Retrieves packet from the connection
        """
        return self.conn_in.get()


class Node(object):
    """
    Models a network node
    """

    def __init__(self, env, M, node_name, verbose):
        self.M = M
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

        # counters for sent/received packets (key=node name)
        self.pkt_sent = {}
        self.pkt_recv = {}

        # dt interval for incoming queue monitoring
        self.queue_monitor_deltat = Globals.QUEUE_MONITOR_DELTAT

    def add_conn(self, c, conn):
        """Adds a connection from this node to the node 'c'
        """
        self.conns[c] = conn
        # self.customConnections[c] = customConnection
        self.pkt_sent[c] = 0
        self.pkt_recv[c] = 0

    def if_up(self):
        """Activates interfaces- this sets up SimPy processes
        """

        # start recv processes on all receiving connections
        for c in self.conns:
            self.env.process(self.if_recv(c))

        # activate packet generator, packet forwarding, queue monitoring
        self.env.process(self.pkt_gen_process())
        self.env.process(self.forward_process())
        self.env.process(self.queue_monitor())

    def if_recv(self, c):
        """Node receive interface from node 'c'
        """

        # the connection from 'self.name' to 'c'
        conn = self.conns[c]

        while True:

            # pick up any incoming packets
            pkt = yield conn.get()

            if len(self.proc_queue) < self.queue_cutoff:

                # increment the counter for this sending node
                # self.pkt_recv[c] += 1

                # put the packet in the processing queue
                self.proc_queue.append(pkt)

                # report as per verbose level
                if self.verbose >= Globals.VERB_LO:
                    Utils.report(self.env.now, self.name, pkt, self.proc_queue,
                                 inspect.currentframe().f_code.co_name,
                                 self.verbose)

            else:  # processing queue full, discard this packet

                self.discard_packet(pkt)

                # report as per verbose level
                if self.verbose >= Globals.VERB_LO:
                    Utils.report(self.env.now, self.name, pkt, self.proc_queue,
                                 inspect.currentframe().f_code.co_name +
                                 '_discard', self.verbose)

    def forward_process(self):
        """Node packet forwarding process
        """

        while True:

            # if there are any packets in the processing queue
            if len(self.proc_queue) > 0:

                # get the first packet from the queue
                pkt = self.proc_queue.popleft()
                source_node = pkt[Globals.SOURCE]  # source node
                dest_node = pkt[Globals.DEST_NODE]  # destination node

                # report as per verbose level
                if self.verbose >= Globals.VERB_LO:
                    Utils.report(self.env.now, self.name, pkt, self.proc_queue,
                                 inspect.currentframe().f_code.co_name +
                                 '_get', self.verbose)

                # the destination is *this* node
                if dest_node == self.name:

                    # incur packet processing time
                    yield self.env.timeout(Globals.CYCLE_TIME)

                    # set the destination arrival time
                    pkt[Globals.DEST_TIME_STAMP] = self.env.now

                    # packet terminates here- put it in the receive queue
                    self.received.append([self.env.now, pkt])

                    # report as per verbose level
                    if self.verbose >= Globals.VERB_LO:
                        Utils.report(self.env.now, self.name, pkt,
                                     self.proc_queue,
                                     inspect.currentframe().f_code.co_name +
                                     '_sink', self.verbose)

                # the destination is some other node, forward packet
                else:

                    # incur packet processing time
                    yield self.env.timeout(Globals.CYCLE_TIME)

                    # increment the packet hop counter
                    pkt[Globals.NO_HOPS] += 1

                    # get next-hop node along the shortest path
                    hop_node = 0

                    # register the next-hop node with the packet
                    pkt[Globals.HOP_NODE] = hop_node

                    # forward packet to the next-hop node
                    self.send_to_node(pkt, hop_node)

                    # count this packet as sent to 'hop_node'
                    self.pkt_sent[self.name] += 1

                    # if the source node is not this node, record that
                    # the packet was forwarded by this node
                    if source_node != self.name:
                        self.forwarded.append([self.env.now, pkt])

                    # report as per verbose level
                    if self.verbose >= Globals.VERB_LO:
                        Utils.report(self.env.now, self.name, pkt,
                                     self.proc_queue,
                                     inspect.currentframe().f_code.co_name +
                                     '_fwd', self.verbose)

            # self.queue is empty
            else:
                #  incur queue check delay
                yield self.env.timeout(self.queue_check)

    def send_to_node(self, pkt, dest_node):
        """Sends packet to the destination node
        """

        # get the connection to the destination node
        conn = self.conns[self.name]
        conn.put(pkt)  # put the packet onto the connection

        # report as per verbose level
        if self.verbose > Globals.VERB_NO:
            Utils.report(self.env.now, self.name, pkt, self.proc_queue,
                         inspect.currentframe().f_code.co_name,
                         self.verbose)

    def pkt_gen_process(self):
        """Process that generates networks packets
        """

        while True:

            pkt_gen_deltat = PacketGenerator.run(self.pkt_rate)

            yield self.env.timeout(pkt_gen_deltat)

            # choose the destination node
            dest_node = random.choice(list(self.nodes))
            while dest_node == self.name:
                dest_node = random.choice(list(self.nodes))


            # create the packet
            pkt = self.make_pkt(dest_node)

            # if the forward queue is full, discard the last received
            # packet to open space in the queue
            if len(self.proc_queue) >= self.queue_cutoff:
                pkt_discard = self.proc_queue.pop()
                self.discard_packet(pkt_discard)

            # add the node generated packet to the head of the queue
            self.proc_queue.appendleft(pkt)

            # add to generated packets monitor
            self.generated.append([self.env.now, pkt])

            # report as per verbose level
            if self.verbose >= Globals.VERB_LO:
                Utils.report(self.env.now, self.name, pkt, self.proc_queue,
                             inspect.currentframe().f_code.co_name,
                             self.verbose)

    def make_pkt(self, dest_node):
        """Creates a network packet
        """

        pkt = {}
        pkt[Globals.TIME_STAMP] = self.env.now
        pkt[Globals.ID] = Utils.gen_id()
        pkt[Globals.SOURCE] = self.name
        pkt[Globals.DEST_NODE] = dest_node
        pkt[Globals.HOP_NODE] = Globals.NONE  # the initial value
        pkt[Globals.DEST_TIME_STAMP] = -1.0  # the initial value
        pkt[Globals.NO_HOPS] = 0  # the initial value

        return pkt

    def discard_packet(self, pkt):
        """Discards the packet (puts the packet into the node packet sink)
        """

        # place this packet in the node sink
        self.discarded.append([self.env.now, pkt])

    def queue_monitor(self):
        """Queue monitor process
        """

        while True:

            # add to queue monitor time now and queue_length
            self.queue_mon.append([self.env.now, len(self.proc_queue)])

            # incur monitor queue delay
            yield self.env.timeout(self.queue_monitor_deltat)
