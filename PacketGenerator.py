"""PacketGenerator.py
"""


import random
import Utils


def run(pkt_rate):
    """Defines method for packet generation
    Simple hack for now, needs further development.
    """

    gen_method = pkt_rate[0]

    if gen_method == "poisson":
        pkt_rate_deltat = random.expovariate(pkt_rate[1])
    elif gen_method == "periodic":
        pkt_rate_deltat = pkt_rate[1]
    elif gen_method == "PPBP":
        pkt_rate_deltat = pkt_rate[1]



    else:
        Utils.error("Packet generator not implemented")

    return pkt_rate_deltat
