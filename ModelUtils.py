"""ModelUtils.py
"""

import os
import json

import networkx as nx

from rl_learning import Globals

import Utils
import Network
import IO


def links2graph(links):
    """
    Takes the list of network links and returns a NetworkX graph
    object.
    """

    G = nx.Graph()
    G.add_edges_from(links)

    return G


def load_links(model_name, input_dir="."):
    """
    Loads network links definition file, performs a few sanity
    checks, returns links.

    This requires model TOPO file only.
    """

    file_name = model_name + Globals.TOPO_SUFFIX
    file_path = os.path.join(input_dir, file_name)
    lines = IO.file_lines(file_path)

    links = []
    for line in lines:
        fields = line.split()
        links.append([fields[0], fields[1]])

    print(" [+] Verifying network topology..", end="")

    # if any node is connected to itself trigger error
    for edge in links:
        if edge[0] == edge[1]:
            print("{:s} - {:s}".format(edge[0], edge[1]))
            Utils.error("Link connects node to itself")

    # extract unique links
    unique_links = []
    for edge in links:
        edge_flip = (edge[1], edge[0])
        if (edge not in unique_links) and (edge_flip not
                                           in unique_links):
            unique_links.append(edge)

    print("All good")
    print(" [+] Total number of links: {:d}".format(len(links)))
    print(" [+] Unique links: {:d}".format(len(unique_links)))

    return unique_links


def load_node_types(model_name, input_dir="."):
    """Loads node types declaration
    """

    print(" [+] Loading node types")

    file_name = model_name + Globals.TYPE_SUFFIX
    file_path = os.path.join(input_dir, file_name)
    lines = IO.file_lines(file_path)

    node_types = {}
    for line in lines:
        fields = line.split()
        node_types[fields[0]] = fields[1]

    return node_types


def load_param(model_name, input_dir="."):
    """Loads network parameters
    """

    print(" [+] Loading network parameters")

    file_name = model_name + Globals.PARAM_SUFFIX
    file_path = os.path.join(input_dir, file_name)
    model_param = json.loads(open(file_path).read())

    return model_param


def make_model(links, node_types, param, model_name):
    """Create a simulation model
    """

    print(" [+] Preparing model")

    M = Network.Model(model_name, links, node_types, param)

    return M


def pickle_model(G, model_name, output_dir="."):
    """Stores model to a file with 'pickle'
    """

    model_file = model_name + Globals.PICKLE_SUFFIX
    model_path = os.path.join(output_dir, model_file)
    nx.write_gpickle(G, model_path)

    print(" [+] Model '{:s}' saved".format(model_name))


def unpickle_model(model_name, input_dir="."):
    """Loads previously pickled model
    """

    model_file = model_name + Globals.PICKLE_SUFFIX
    model_path = os.path.join(input_dir, model_file)
    G = nx.read_gpickle(model_path)

    print(" [+] Model '{:s}' loaded".format(model_name))

    return G
