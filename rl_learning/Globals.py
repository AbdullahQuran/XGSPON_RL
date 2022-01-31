"""Globals.py
"""
import params
# from scripts import params

# version
VERSION = 0
REVISION = 3
SUBREV = 3

# queue monitoring interval
QUEUE_MONITOR_DELTAT = 0.000125

#
# Network model keywords- model graph definition keywords
#

MODEL_NAME_KWD = 'model_name'
# node attribute keywords
NODE_TYPE_KWD = 'type'
NODE_PKT_RATE_KWD = 'node_pkt_rate'
# packet processing delay
NODE_PROC_DELAY_KWD = 'node_proc_delay'
# node queue cutoff
NODE_QUEUE_CUTOFF_KWD = 'node_queue_cutoff'
# the interval for checking the queue when
# not locked in processing
NODE_QUEUE_CHECK_KWD = 'node_queue_check'

# link attribute keywords
LINK_CAPACITY_KWD = 'link_capacity'  # link capacity
# delay in packet transmission through the link
LINK_TRANSM_DELAY_KWD = 'link_transm_delay'

# the table of shortest paths
PATH_G_KWD = 'path_G'

#
# packet structure keywords
#
ID = 'id'
TIME_STAMP = 'time_stamp'
SOURCE = 'source'
DEST_NODE = 'dest_node'
HOP_NODE = 'hop_node'
NO_HOPS = 'no_hops'
DEST_TIME_STAMP = 'dest_time_stamp'
REPORT_TIME = 'report_time'
GRANT_DICT = {}
NONE = 'none'

#
# suffixes for input files
#
TOPO_SUFFIX = ".topo"
TYPE_SUFFIX = ".type"
PARAM_SUFFIX = ".param"

# suffix for pickling the model
PICKLE_SUFFIX = ".pickle"

# suffixes for output files
NODES_LIST_FILE = 'nodes.dat'
GEN_SUFFIX = '_gen.csv'
FWD_SUFFIX = '_fwd.csv'
DISCARD_SUFFIX = '_discard.csv'
RECV_SUFFIX = '_recv.csv'
QUEUE_SUFFIX = '_queue.csv'
URLLC_SUFFIX = "_urllc.csv"
EMBB_SUFFIX = "_eMBB.csv"
VIDEO_SUFFIX = "_video.csv"
IP_SUFFIX = "_ip.csv"
REPROT_URLLC_SUFFIX = "_report_URLLC.csv"
REPORT_EMBB_SUFFIX = '_report_eMBB.csv'
REPORT_VIDEO_SUFFIX = '_report_Video.csv'
REPORT_IP_SUFFIX = '_report_IP.csv'
REPORT_MSG_SUFFIX = '_report_msg.csv'
REPORT_MSG_SUFFIX = '_report_msg.csv'
GRANT_MSG_SUFFIX = '_grant_msg.csv'
GRANTS = '_grants.csv'

DISCARDED_URLLC_SUFFIX = '_discarded_URLLC.csv'
DISCARDED_EMBB_SUFFIX = '_discarded_eMBB.csv'
DISCARDED_VIDEO_SUFFIX = '_discarded_Video.csv'
DISCARDED_IP_SUFFIX = '_discarded_IP.csv'

LENGTH_URLLC_SUFFIX = '_queue_lenght_URLLC.csv'
LENGTH_EMBB_SUFFIX = '_queue_lenght_eMBB.csv'
LENGTH_VIDEO_SUFFIX = '_queue_lenght_Video.csv'
LENGTH_IP_SUFFIX = '_queue_lenght_IP.csv'

DELAY_STATS_SUFFIX = '_delay_stats.csv'

#
# define verbose levels
#
VERB_NO = 0
VERB_LO = 1
VERB_HI = 2
VERB_EX = 3

#
# simulation progress bar
#
BAR_LENGTH = 40



###############

OLT_TYPE = 'OLT'
ONU_TYPE = 'ONU' # 5g ONU
ONU_TYPE2 = 'ONU2' # FTTx onu
BROADCAST_PKT_ID = -1
GRANT_PACKET_TYPE = "GRANT_PACKET"

###########
# OLT configs
OLT_CYCLE_TIME_KWD = 'cycle_time'


PKT_TYPE_KWD = 'pkt_type'
PKT_TYPE_REPORT = 1
PKT_TYPE_BEACON = 2
DOWNSTREAM_DALAY = 1

GRANT_PACKET_BWM_FIELD = "BWM"
GRANT_PACKET_ONU_ID_FIELD = "ONU_ID"
GRANT_PACKET_ONU_START_FIELD = "START_TIME"
GRANT_PACKET_ONU_END_FIELD = "END_TIME"
REPORT_PACKET_QUEUE_LENGTH_FIELD = "QUEUE_LENGTH"
REPORT_PACKET_ONU_ID_FIELD = "ONU_ID"
REPORT_PACKET_PAYLOAD_FIELD = "PAYLOAD"
REPORT_PACKET_PAYLOAD_LENGTH_FIELD = 'PAYLOAD_LENGTH'
REPORT_PACKET_LAST_REPORT_SIZE = 'LAST_REPORT_SIZE'
ONU_PACKET_TYPE_FIELD = "PACKET_TYPE"
OLT_PACKET_TYPE_FIELD = ONU_PACKET_TYPE_FIELD
ONU_PACKET_TCON_ID_FIELD = "TCON_ID"

ONU_PAYLOAD_PACKET_TYPE = 1
ONU_REPORT_PACKET_TYPE = 2

GRANT_OBJ_ONU_ID = "ONU_ID"
GRANT_OBJ_START_TIME = "START_TIME"
GRANT_OBJ_END_TIME = "END_TIME"
GRANT_OBJ_ALLOACTED_SIZE = "TCON_ALLOCATED_SIZE_ARRAY"


CYCLE_TIME = 0

BROADCAST_GRANT_DEST_ID = "-1"
BROADCAST_REPORT_DEST_ID = "-2"

######### DBA variables

XGSPON_CYCLE = 0.000125
SERVICE_INTERVAL =  XGSPON_CYCLE   

TCON1_ID = 1 # 5g: URLLC, FFTx: video
TCON2_ID = 2 # 5g: embb, FFTx: IP


MAX_LINK_CAPACITY_bps = 10 * 1000000000
MAX_LINK_CAPACITY_Bps = MAX_LINK_CAPACITY_bps / 8
MAX_LINK_CAPACITY_Bpxgpon = MAX_LINK_CAPACITY_Bps / (1000000/125)
FB = params.FB

# sizes in bytes
URLLC_RF = 100
URLLC_RM = 110
URLLC_PACKET_LATENCY_THRESHOLD = 0.001 # 1 milliseconds
URLLC_FLOW_JITTER_THRESHOLD = 0.005 # 5 milliseconds
URLLC_SI_MIN = 1
URLLC_AB_MIN = params.URLLC_AB_MIN

URLLC_SI_MAX = 1
URLLC_AB_SUR = params.URLLC_AB_SUR

EMBB_RF = 100
EMBB_RM = 110
EMBB_PACKET_LATENCY_THRESHOLD = 0.001 # 1 milliseconds
EMBB_FLOW_JITTER_THRESHOLD = 0.005 # 5 milliseconds
EMBB_SI_MIN = 1
EMBB_AB_MIN = params.EMBB_AB_MIN

EMBB_SI_MAX = 1
EMBB_AB_SUR = params.EMBB_AB_SUR

VIDEO_RF = 100
VIDEO_RM = 110
VIDEO_PACKET_LATENCY_THRESHOLD = 0.001 # 1 milliseconds
VIDEO_FLOW_JITTER_THRESHOLD = 0.005 # 5 milliseconds
VIDEO_SI_MIN = 1
VIDEO_AB_MIN = params.VIDEO_AB_MIN

VIDEO_SI_MAX = 1
VIDEO_AB_SUR = params.VIDEO_AB_SUR

IP_RF = 100
IP_RM = 110
IP_PACKET_LATENCY_THRESHOLD = 0.001 # 1 milliseconds
IP_FLOW_JITTER_THRESHOLD = 0.005 # 5 milliseconds
IP_SI_MIN = 1
IP_AB_MIN = params.IP_AB_MIN

IP_SI_MAX = 1
IP_AB_SUR = params.IP_AB_SUR

N_DISCRETE_ACTIONS = 3
OBSERVATION_MIN = 0.0
OBSERVATION_MAX = 110.0

DELAY_TRESHOLD_MIN_URLLC = 0.000500
DELAY_TRESHOLD_MIN_EMBB = 0.000200
DELAY_TRESHOLD_MIN_VIDEO = 0.000500
DELAY_TRESHOLD_MIN_IP = 0.000500


DELAY_TRESHOLD_MAX_URLLC = 0.000350
DELAY_TRESHOLD_MAX_EMBB = 0.000350
DELAY_TRESHOLD_MAX_VIDEO = 0.050000
DELAY_TRESHOLD_MAX_IP = 0.200000

LENGTH_TRESHOLD_URLLC = 2
LENGTH_TRESHOLD_EMBB = 5
LENGTH_TRESHOLD_VIDEO = 5
LENGTH_TRESHOLD_IP = 5

LENGTH_TRESHOLD_URLLC2 = 5
LENGTH_TRESHOLD_EMBB2 = 5
LENGTH_TRESHOLD_VIDEO2 = 5
LENGTH_TRESHOLD_IP2 = 5


WEIGHT_ONU1 = 10
WEIGHT_VIDEO = 1
WEIGHT_IP = 1

URLLC_SCALE = 10
EMBB_SCALE = 10
VIDEO_SCALE = 10
IP_SCALE = 10

queue_cutoff_bytes_urllc = 10000
queue_cutoff_bytes_embb = 10000
queue_cutoff_bytes_video = 10000
queue_cutoff_bytes_ip = 10000

SIM_TIME = params.SIM_TIME
GEN_TIME = params.GEN_TIME

QUEUE_SIZE_LIMIT = 35.0/0.000125

SERVICE_COUNT = 4

def appendDequeue(queue, element, limit = QUEUE_SIZE_LIMIT, isPopLeft = True, isAppendLeft = True):
    if (len(queue) >= limit):
        if (isPopLeft):
            queue.popleft()
        else:
            queue.pop()
    if isAppendLeft:
        queue.appendleft(element)
    else:
        queue.append(element)


def appendList(queue, element, limit = QUEUE_SIZE_LIMIT):
    if (len(queue) >= limit):
            queue.pop(0)
    queue.append(element)