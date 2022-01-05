"""Network.py
"""


import copy
import networkx as nx
import Globals
import Utils


class Model():
    """Models a network as a graph with parameters required
    for simulation.
    """

    def __init__(self, model_name, links, node_types, param):

        self.model_name = model_name

        # perform a few sanity checks
        Utils.verify_input(links, node_types, param)

        # create network graph
        self.G = nx.DiGraph()
        self.add_edges(links)  # creates nodes and links

        # add network attributes
        self.assign_node_attr(node_types, param)  # to nodes
        self.assign_link_attr(param)  # to links

        # calculate the table of shortest paths
        # self.shortest_paths = Utils.shortest_path_table(self.G)

    def __str__(self):
        """Returns the string that identifies the model
        """

        return "Network model '{:s}'".format(self.model_name)

    def add_edges(self, links):
        """Adds edges with attributes to the model graph
        """

        # create a local copy of the list of links
        links_local = copy.deepcopy(links)

        # open space for link attributes
        for edge in links_local:
            edge.append({Globals.LINK_CAPACITY_KWD: None,
                         Globals.LINK_TRANSM_DELAY_KWD: None})

        # add edges with links to the model graph
        self.G.add_edges_from(links_local)

    def assign_node_attr(self, node_types, param):
        """Assigns attributes to nodes
        """

        for node_name in self.G.nodes():
            node_type = node_types[node_name]  # type of this node
            # it was previously checked that each node type has associated
            # parameters
            self.G.nodes[node_name].update(param[node_type])
            self.G.nodes[node_name][Globals.NODE_TYPE_KWD] = node_type

    def assign_link_attr(self, param):
        """Assigns attributes to links
        """

        # loop over all edges in the model graph
        for link in self.G.edges():

            n1 = link[0]  # node 1
            n2 = link[1]  # node 2

            # get node types
            n1_type = self.G.nodes[n1][Globals.NODE_TYPE_KWD]
            n2_type = self.G.nodes[n2][Globals.NODE_TYPE_KWD]

            # the link name as X-Y and Y-X. One of these two must be
            # a valid keyword for 'param'
            link_name = str(n1_type) + "-" + str(n2_type)
            link_name_r = str(n2_type) + "-" + str(n1_type)

            if link_name in param:
                link_param = param[link_name]
            elif link_name_r in param:
                link_param = param[link_name_r]
            else:
                Utils.error("no parameters for link {:s}".format(link_name))

            self.G[n1][n2][Globals.LINK_CAPACITY_KWD] =\
                link_param[Globals.LINK_CAPACITY_KWD]
            self.G[n1][n2][Globals.LINK_TRANSM_DELAY_KWD] =\
                link_param[Globals.LINK_TRANSM_DELAY_KWD]

