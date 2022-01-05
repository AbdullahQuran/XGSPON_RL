"""Simulator.py
"""
import simpy
import Components
from rl_learning import Globals
import Constants
import Olt
import Onu


def setup_network(env, M, verbose=Globals.VERB_HI):
    """Binds the model graph 'M' to the SimPy simulation environment 'env'
    """

    print(" [+] Found {:d} nodes and {:d} links"
          .format(len(M.G.nodes()), len(M.G.edges())))

    # create simulation network model
    network = create_network_model(env, M, verbose)

    # activate node interfaces (this sets SimPy processes)
    M.odn.network = network

    for node_name in network:
        network[node_name].if_up()



    return network


def create_network_model(env, M, verbose=Globals.VERB_HI):
    """Creates network: creates network nodes, binds connections to the nodes
    """

    network = {}

    # create nodes
    for node_name in M.G.nodes():
        if (M.G.nodes[node_name][Globals.NODE_TYPE_KWD] == Globals.OLT_TYPE):
            network[node_name] = Olt.OLT(env, M, node_name, verbose)
            # network[node_name] = Components.Node(env, M, node_name, verbose)
        elif (M.G.nodes[node_name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE):
            network[node_name] = Onu.ONU(env, M, node_name, verbose)
        elif (M.G.nodes[node_name][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2):
            network[node_name] = Onu.ONU(env, M, node_name, verbose)
        

        # network[node_name] = Components.Node(env, M, node_name, verbose)

    # create node links
    conn_dict = init_conn2(env, M)
    # bind connections to the nodes
    for node_name1 in conn_dict:

        # node_name1 = c

        # Note: conn_a and conn_b are the same connection pipe viewed
        # from a-b and b-a perspectives
        conn_1 = conn_dict[node_name1]
        # conn_2 = conn_dict[c][1].channel

        # customConnection1 = conn_dict[c][0]
        # customConnection2 = conn_dict[c][1]

        # bind interfaces to nodes
        network[node_name1].add_conn(node_name1, conn_1)
        # network[node_name2].add_conn(node_name1, conn_2, customConnection2)

    # for node_name in M.G.nodes():
    #     print(node_name)
    #     for x in network[node_name].customConnections:
    #         print(network[node_name].customConnections[x].type)
    #     print("################")
    return network


def init_conn2(env, M):
    """
    Creates connection pipes.

    Keys of 'conn_dict' are tuples that contain the two node names,
    and values are tuples that contain the two connection objects,
    belongings to each node.
    """

    # initialise connections dictionary
    conn_dict = {}
    odn = M.odn
    # loop over all edges in the network graph
    for c in M.G.edges():
        
        # fetch the link attributes for this edge
        link_attr_dict = M.G[c[0]][c[1]]
        # fetch the capacity and transmission delay for this link
        link_capacity = link_attr_dict[Globals.LINK_CAPACITY_KWD]
        transm_delay = link_attr_dict[Globals.LINK_TRANSM_DELAY_KWD]

        pipe = simpy.Store(env, capacity=link_capacity)
        if (M.G.nodes[c[0]][Globals.NODE_TYPE_KWD] == Globals.OLT_TYPE):
            conn_dict[c[0]] = Components.Channel(env, transm_delay, odn.downstream[c[0]], pipe, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME) 
            # network[node_name] = Components.Node(env, M, node_name, verbose)
        elif (M.G.nodes[c[0]][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or M.G.nodes[c[0]][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2):
            conn_dict[c[0]] = Components.Channel(env, transm_delay, odn.upstream[c[0]], pipe, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME) 

        pipe2 = simpy.Store(env, capacity=link_capacity)
        if (M.G.nodes[c[1]][Globals.NODE_TYPE_KWD] == Globals.OLT_TYPE):
            conn_dict[c[1]] = Components.Channel(env, transm_delay, odn.downstream[c[1]], pipe2, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME) 
            # network[node_name] = Components.Node(env, M, node_name, verbose)
        elif (M.G.nodes[c[1]][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE or M.G.nodes[c[1]][Globals.NODE_TYPE_KWD] == Globals.ONU_TYPE2):
            conn_dict[c[1]] = Components.Channel(env, transm_delay, odn.upstream[c[1]], pipe2, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME) 



        # # create two communication pipes, 1->2 and 2->1
        # pipe_12 = simpy.Store(env, capacity=link_capacity)
        # pipe_21 = simpy.Store(env, capacity=link_capacity)

        # # create two connection objects
        # conn_1 = Components.Channel(env, transm_delay, pipe_12, pipe_21)
        # conn_2 = Components.Channel(env, transm_delay, pipe_21, pipe_12)

        # connectionDown = Components.CustomChannel(conn_1, Components.ChannelType.DOWN, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME)
        # connectionUP = Components.CustomChannel(conn_2, Components.ChannelType.UP, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME)

        # # add the connection to the dictionary
        # conn_dict[c] = (connectionDown, connectionUP)

    return conn_dict






def init_conn(env, M):
    """
    Creates connection pipes.

    Keys of 'conn_dict' are tuples that contain the two node names,
    and values are tuples that contain the two connection objects,
    belongings to each node.
    """

    # initialise connections dictionary
    conn_dict = {}

    # loop over all edges in the network graph
    for c in M.G.edges():

        # fetch the link attributes for this edge
        link_attr_dict = M.G[c[0]][c[1]]
        # fetch the capacity and transmission delay for this link
        link_capacity = link_attr_dict[Globals.LINK_CAPACITY_KWD]
        transm_delay = link_attr_dict[Globals.LINK_TRANSM_DELAY_KWD]

        # create two communication pipes, 1->2 and 2->1
        pipe_12 = simpy.Store(env, capacity=link_capacity)
        pipe_21 = simpy.Store(env, capacity=link_capacity)

        # create two connection objects
        conn_1 = Components.Channel(env, transm_delay, pipe_12, pipe_21)
        conn_2 = Components.Channel(env, transm_delay, pipe_21, pipe_12)

        connectionDown = Components.CustomChannel(conn_1, Components.ChannelType.DOWN, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME)
        connectionUP = Components.CustomChannel(conn_2, Components.ChannelType.UP, Constants.UNDEFINED_WAVE_LENGTH, Constants.UNDEFINED_CHECK_TIME)

        # add the connection to the dictionary
        conn_dict[c] = (connectionDown, connectionUP)

    return conn_dict



