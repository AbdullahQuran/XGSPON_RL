import simpy
import Globals
class ODN(object):
    """This class represents optical distribution Network."""
    def __init__(self, env, M):
        self.env = env
        self.upstream = {}# upstream chanel
        self.downstream = {} # downstream chanel
        #create downstream splitter
        for node_name in M.G.nodes():
            if (M.G.nodes[node_name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE):
                self.upstream[node_name] = simpy.Store(env)
            elif (M.G.nodes[node_name][Globals.NODE_TYPE_KWD] == Globals.OLT_TYPE):
                self.downstream[node_name] = simpy.Store(env)
            elif (M.G.nodes[node_name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2):
                self.upstream[node_name] = simpy.Store(env)   

        self.env.process(self.forward_To_ONU())
        self.env.process(self.forward_To_OLT())


        # for i in range(NUMBER_OF_OLTs):

    # def up_latency(self, value,ONU):
    #     """Calculates upstream propagation delay."""
    #     yield self.env.timeout(ONU.delay)
    #     self.upstream[ONU.lamb].put(value)

    # def directly_upstream(self,ONU,value):
    #     self.upstream[ONU.lamb].put(value)

    # def down_latency(self,ONU,value):
    #     """Calculates downstream propagation delay."""
    #     self.downstream[ONU.oid].put(value)

    def put_request(self, value, prop_delay, onu_id):
        """ONU Puts the Request message in the upstream """
        yield self.env.timeout(0.000001)
        # print("request: " + str(value["QUEUE_LENGTH"]))
        self.upstream[onu_id].put(value)
        # self.env.process(self.up_latency(value,ONU))

    def get_request(self,onu_id):
        """OLT gets the Request message from upstream  """
        return self.upstream[onu_id].get()

    def put_grant(self,value,prop_delay):
        """OLT Puts the Grant message in the downstream """
        # change to broadcast
        yield self.env.timeout(0.000001)
        # print("reseponse: " + str(value["BWM"]))
        for olt in self.downstream:
            self.downstream[olt].put(value)
            # self.env.process(self.down_latency(onu,value))

    def get_grant(self, id):
        """ONU gets the Grant message from downstream """
        return self.downstream[id].get()

    def forward_To_ONU(self):
        while True:
            for dstream in self.downstream:
                pkt = yield self.get_grant(dstream)
                for node_name in self.network:
                    if (self.network[node_name].type == Globals.ONU_TYPE or self.network[node_name].type == Globals.ONU_TYPE2):
                        # print("sending grant packet to {}".format(node_name))
                        x = self.network[node_name].conns[node_name]
                        x.put_conn_in(pkt)

            yield self.env.timeout(Globals.CYCLE_TIME)


    def forward_To_OLT(self):
        while True:
            for ustream in self.upstream:
                pkt = yield self.get_request(ustream)
                for node_name in self.network:
                    if (self.network[node_name].type == Globals.OLT_TYPE):
                        # print("sending report packet from {} to {}".format(ustream, node_name))
                        self.network[node_name].conns[node_name].put_conn_in(pkt)
            yield self.env.timeout(Globals.CYCLE_TIME)

    
        