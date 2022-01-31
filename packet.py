from rl_learning import Globals
import Utils

def make_grant_packet(env, dest_node, src_node, bwm, payload):
    """Creates a network packet
    """
    pkt = {}
    pkt[Globals.TIME_STAMP] = env.now
    pkt[Globals.ID] = Utils.gen_id()
    pkt[Globals.SOURCE] = src_node
    pkt[Globals.DEST_NODE] = dest_node
    pkt[Globals.HOP_NODE] = Globals.NONE  # the initial value
    pkt[Globals.DEST_TIME_STAMP] = -1.0  # the initial value
    pkt[Globals.NO_HOPS] = 0  # the initial value
    pkt[Globals.GRANT_PACKET_BWM_FIELD] = bwm  # the initial value {"ONU": 1, "grant_period_start": 3, "grant_period_end": 5}
    pkt[Globals.OLT_PACKET_TYPE_FIELD] = Globals.GRANT_PACKET_TYPE
    pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = len(payload)
    pkt[Globals.REPORT_TIME] = 0  # the initial value
    pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD] = ""
    return pkt

def make_report_packet(env, dest_node, src_node, tcon_length, onuId, payload, reportSize = {}):
    """Creates a network packet
    """
    pkt = {}
    pkt[Globals.TIME_STAMP] = env.now
    pkt[Globals.ID] = Utils.gen_id()
    pkt[Globals.SOURCE] = src_node
    pkt[Globals.DEST_NODE] = dest_node
    pkt[Globals.HOP_NODE] = 0  # the initial value
    pkt[Globals.DEST_TIME_STAMP] = -1.0  # the initial value
    pkt[Globals.NO_HOPS] = 0  # the initial value
    pkt[Globals.REPORT_PACKET_QUEUE_LENGTH_FIELD] = tcon_length #{1: 500, tcon_id: size}
    pkt[Globals.REPORT_PACKET_ONU_ID_FIELD] = onuId 
    pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD] = ""
    pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = len(payload)
    pkt[Globals.REPORT_PACKET_LAST_REPORT_SIZE] = reportSize
    pkt[Globals.ONU_PACKET_TYPE_FIELD] = Globals.ONU_REPORT_PACKET_TYPE
    pkt[Globals.REPORT_TIME] = 0  # the initial value

    return pkt

def make_payload_packet(env,id, dest_node, src_node, onuId, payload,tcon_id):
    """Creates a network packet
    """
    pkt = {}
    pkt[Globals.TIME_STAMP] = env.now
    pkt[Globals.ID] = id
    pkt[Globals.SOURCE] = src_node
    pkt[Globals.DEST_NODE] = dest_node
    pkt[Globals.HOP_NODE] = Globals.NONE  # the initial value
    pkt[Globals.DEST_TIME_STAMP] = -1.0  # the initial value
    pkt[Globals.NO_HOPS] = 0  # the initial value
    pkt[Globals.REPORT_PACKET_ONU_ID_FIELD] = onuId 
    pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD] = payload
    pkt[Globals.REPORT_PACKET_PAYLOAD_LENGTH_FIELD] = max(len(payload), 1)
    pkt[Globals.ONU_PACKET_TYPE_FIELD] = Globals.ONU_PAYLOAD_PACKET_TYPE
    pkt[Globals.ONU_PACKET_TCON_ID_FIELD] = tcon_id
    pkt[Globals.REPORT_TIME] = 0  # the initial value

    return pkt

def make_grant_object(onuId, tcon1Size, tcon2Size):
    obj = {}
    obj[Globals.GRANT_OBJ_ONU_ID] = onuId  
    obj[Globals.GRANT_OBJ_ALLOACTED_SIZE] = {Globals.TCON1_ID: tcon1Size, Globals.TCON2_ID:tcon2Size} # {TCON_id: size, tcon2: size2}  
    return obj




