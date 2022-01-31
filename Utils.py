"""Utils.py
"""
import sys
import os
import subprocess
import random
import string
import networkx as nx
from rl_learning import Globals
import IO


def benchmark2anx(input_file, output_file):
    """Loads the network produced by the program *benchmark* ('input_file'),
    does some basic sanity checks, removes duplicate edges, and saves the
    unique edges to the file 'output_file'.
    """

    # load file lines
    lines = IO.file_lines(input_file)

    # get the edges
    edges = []
    for line in lines:
        fields = line.split()
        edges.append((int(fields[0]), int(fields[1])))

    # sanity check: a node must not be connected to itself
    cntr = 0
    for edge in edges:
        if edge[0] == edge[1]:
            cntr += 1
            if cntr == 1:
                error("node connected to itself")

    # extract unique edges
    unique_edges = []
    for edge in edges:
        edge_flip = (edge[1], edge[0])
        if (edge not in unique_edges) and (edge_flip not in unique_edges):
            unique_edges.append(edge)

    # write unique edges to a file
    fp = IO.open_for_writing(output_file)
    for edge in unique_edges:
        fp.write("{:d} {:d}\n".format(edge[0], edge[1]))
    fp.close()

    print("\t Total number of edges read: {:d}".format(len(edges)))
    print("\t {:d} unique edges saved as '{:s}'\n".format(len(unique_edges),
                                                          output_file))


def verify_input(model_edges, node_types, model_param):
    """Verifies model parameters
    """

    print(" [+] Checking model input..", end="")

    # 1. Extract nodes
    nodes_all = []
    for edge in model_edges:
        nodes_all.append(edge[0])
        nodes_all.append(edge[1])
    nodes = set(nodes_all)

    # 2. Check that every has a type
    for n in nodes:
        if n not in node_types:
            error("Node '{:s}' does not have type".format(n))

    # 3. Check that every node type has defined node parameters
    for n in node_types:
        n_type = node_types[n]
        if n_type not in model_param:
            error("Node type '{:s}' does not have parameters".format(n_type))

    # 4. Check that every link has defined link parameters
    for edge in model_edges:

        n1 = edge[0]
        n2 = edge[1]

        n1_t = node_types[n1]
        n2_t = node_types[n2]

        link_name = str(n1_t) + "-" + str(n2_t)
        link_name_r = str(n2_t) + "-" + str(n1_t)

        if (link_name not in model_param) and (link_name_r not
                                               in model_param):
            error("Link type '{:s}' does not have parameters"
                  .format(link_name))

    print("All good")


def shortest_path_table(G):
    """
    Calculates the table of shortest paths between all nodes.
    The 'table' is actually a dictionary path_G{} whose keys
    are tuples (n1, n2). The value of path_G[(n1,n2)] is the
    next hop node on the shortest path from n1 -> n2.
    """

    print(" [*] Calculating shortest paths")

    path_G = {}

    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 != n2:
                path_s = nx.shortest_path(G, source=n1, target=n2)
                if (path_s[0] != n1) or (path_s[-1] != n2):
                    error("This should never happen")
                path_G[(n1, n2)] = path_s[1]

    return path_G

pkt_counter = 0
def gen_id(n=16):
    global pkt_counter
    """Generates a unique packet ID as a string of 'n' characters
    """
    pkt_counter = pkt_counter + 1
    return pkt_counter
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    alphabet = alphabet + string.digits
    pkt_id = ""

    for _ in range(n):
        pkt_id = pkt_id + random.choice(alphabet)

    return pkt_id


def report(self_env_now, self_name, pkt, proc_queue, fname, verbose):
    """Prints packet report on stdout
    """

    print("\n  {:s}(): @{:.4f} NODE '{:s}'. Packet ID: '{:s}'"
          .format(fname, self_env_now, self_name, pkt[Globals.ID]))

    print_pkt(pkt)

    if verbose == Globals.VERB_EX:
        report_queue(proc_queue)


def report_queue(proc_queue):
    """Prints all the packets currently in the queue
    """

    print("\n\t **queue now has {:d} packet(s):".format(len(proc_queue)))
    for pkt in proc_queue:
        print_pkt(pkt)


def print_pkt(pkt):
    """Formats and prints packet on the stdout
    """

    time_str = "t={:.4f}".format(pkt[Globals.TIME_STAMP])
    id_str = "id='{:s}'".format(pkt[Globals.ID])
    source_str = "source={:s}".format(pkt[Globals.SOURCE])
    dest_str = "dest={:s}".format(pkt[Globals.DEST_NODE])
    # hop_str = "hop_node={:s}".format(pkt[Globals.HOP_NODE])
    # no_hops = "hops={:d}".format(pkt[Globals.NO_HOPS])

    print("\t [ {:s} {:s} {:s} {:s} {:s} {:s} ]"
          .format(time_str, id_str, source_str, dest_str, "hop_str", "no_hops"))


def strip_wsnl_list(items_in):
    """Takes a list of strings and strips whitespaces from each string.
    Returns a list of stripped strings.
    """

    if not is_list(items_in):
        error("argument is not a list")

    for item in items_in:
        if not is_str(item):
            error("argument is not a list of strings")

    items = []
    for item in items_in:
        items.append(item.strip())

    return items


def is_list(arg):
    """True if list
    """

    return bool(isinstance(arg, list))


def is_str(arg):
    """True if string
    """

    return bool(isinstance(arg, str))


def clean_up(rm_files):
    """Removes files supplied in the list rm_files.
    """

    RM = "/bin/rm"

    for f in rm_files:

        print(" [ Removing '{:s}' ]".format(f))

        if os.path.exists(f):
            try:
                subprocess.call([RM, "-rf", f])
            except subprocess.CalledProcessError:
                error("file does not exist")
            except OSError:
                error("failed to call 'rm'")


def version():
    """Prints 'anx' version
    """

    print("\n Abstract Network Simulator", end="")
    print(" v{:d}.{:d}.{:d}\n".format(Globals.VERSION, Globals.REVISION, Globals.SUBREV))


def error(message=None):
    """Generic error function
    """

    sys.stdout = sys.__stderr__

    if message is None:
        message = "(no message)"
    else:
        message.rstrip('\n')

    # Retrieve details of the caller function from the caller function's stack
    # frame (i.e. the second most recent stack frame, denoted by "1").
    funame = sys._getframe(1).f_code.co_name
    fmodule = sys._getframe(1).f_code.co_filename
    fmodule = fmodule.split('/')
    fmodule = fmodule[-1]
    fmodule = fmodule.split('.')[0]

    if funame == '?':
        fmessage = " [ main in module '{:s}' ]\n".format(fmodule)
    else:
        fmessage = " [ function '{:s}()' in module '{:s}' ]\n".format(funame,
                                                                      fmodule)

    fmessage = '\n ERROR: {:s}\n'.format(message) + fmessage

    message_list = fmessage.split("\n")

    n = 0
    for line in message_list:
        if len(line) > n:
            n = len(line)

    cstr = ""
    for _ in range(n-1):
        cstr = cstr + "="

    print('\n {:s} {:s}\n'.format(fmessage, cstr))

    sys.exit(1)

