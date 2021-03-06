import os
import constants


def writeGlobals(URLLC_AB_MIN, URLLC_AB_SUR, EMBB_AB_MIN, EMBB_AB_SUR, VIDEO_AB_MIN, VIDEO_AB_SUR, IP_AB_MIN, IP_AB_SUR, FB, SIM_TIME, GEN_TIME):
    text = "URLLC_AB_MIN = {}\n".format(URLLC_AB_MIN)
    text = text + "URLLC_AB_SUR = {}\n".format(URLLC_AB_SUR)
    text = text + "EMBB_AB_MIN = {}\n".format(EMBB_AB_MIN)
    text = text + "EMBB_AB_SUR = {}\n".format(EMBB_AB_SUR)
    text = text + "VIDEO_AB_MIN = {}\n".format(VIDEO_AB_MIN)
    text = text + "VIDEO_AB_SUR = {}\n".format(VIDEO_AB_SUR)
    text = text + "IP_AB_MIN = {}\n".format(IP_AB_MIN)
    text = text + "IP_AB_SUR = {}\n".format(IP_AB_SUR)
    text = text + "FB = {}\n".format(FB)
    text = text + "SIM_TIME = {}\n".format(SIM_TIME)
    text = text + "GEN_TIME = {}\n".format(GEN_TIME)
    constants.writeFile(constants.globalsFilePath, text)



def writeTopoFile(totalOnuCount):
    text = ""
    for i in range(totalOnuCount):
        text = text + str(i+1) + " " + "0\n"
    constants.writeFile(constants.topoFilePath, text)

def writeNodeTypeFile(onu1Count, onu2Count):
    counter = 1
    text = "0 OLT\n"
    for i in range(onu1Count):
        text = text + str(counter) + " " + "ONU\n"
        counter = counter + 1
    for i in range(onu2Count):
        text = text + str(counter) + " " + "ONU2\n"
        counter = counter + 1
    constants.writeFile(constants.nodeTypeFilePath, text)

def writeParamFile(runningMode):
    text = """
        {
        "OLT" : {
      "node_pkt_rate" :    [ "periodic", 0.2 ],
      "node_proc_delay" :  0.0,
      "node_queue_check" : 0.0,
      "node_queue_cutoff": 128,
      "wave_length_check": 0.005,
      "type": \"""" + runningMode + """\"
      },

        
    "ONU" : {
        "node_pkt_rate" :    [ "periodic", 0.000125],
        "node_proc_delay" :  0.0,
        "node_queue_check" : 0.0,
        "node_queue_cutoff": 128,
        "wave_length_check": 0.005
        },

    "ONU2" : {
        "node_pkt_rate" :    [ "periodic", 0.000125],
        "node_proc_delay" :  0.0,
        "node_queue_check" : 0.0,
        "node_queue_cutoff": 128,
        "wave_length_check": 0.005
        },
        

    "OLT-ONU" : {
        "link_transm_delay" : 0.0,
        "link_capacity" : 1024
        },

    "OLT-ONU2" : {
        "link_transm_delay" : 0.0,
        "link_capacity" : 1024
        }
    }
    """
    constants.writeFile(constants.paramFilePath, text)



# olt_types = ["basic"]
olt_types = ["rl_predict"]
# olt_types = ["basic", "rl_predict"]

trials = 1
# onu1count, onu2count
# expArray = [
#     [2,2],
#     [4,4],
#     [6,6],
#     [8,8],
#     [10,10],
#     [12,12],
#     [14,14],
#     [16,16]
#     ] 

expArray = [
    # [2,2, 150],
    # [2,2, 140],
    # [2,2, 130],
    # [2,2, 120],
    # [2,2, 112],
    # [2,2, 100],
    # [2,2, 90],
    # [2,2, 80],
    # [2,2, 70],
    # [2,2, 60],
    [2,2, 58],
    [2,2, 56],
    [2,2, 54],
    [2,2, 52],
    [2,2, 50],
    [2,2, 48],
    [2,2, 46],
    [2,2, 44],
    [2,2, 42],
    [2,2, 40],
    # [2,2, 40],
    # [2,2, 30],
    # [2,2, 20],
    # [2,2, 10],
    # [4,4, 750],
    # [4,4, 500],
    # [4,4, 280],
    # [4,4, 375],
    # [4,4, 250],
    # [4,4, 210],
    # [4,4, 160],
    # [4,4, 105],
    # [4,4, 75],
    ] 


URLLC_AB_MIN = 40
URLLC_AB_SUR = 30
EMBB_AB_MIN = 35
EMBB_AB_SUR = 30
VIDEO_AB_MIN = 32
VIDEO_AB_SUR = 30
IP_AB_MIN = 30
IP_AB_SUR = 30
SIM_TIME = 4
GEN_TIME = 2

for oltType in olt_types:
    resultsDir1 = constants.resultsDir + oltType
    command = "rd /s /q " + resultsDir1
    os.system(command)
    resultsDir1 = constants.resultsDir + oltType + "\\"
    for exp in expArray:
        expNam = '-'.join([str(elem) for elem in exp])
        expDir = resultsDir1 + expNam
        command = "rd /s /q " + expDir
        os.system(command)
        for t in range(trials):
            resultsDir = expDir + "\\" + str(t)
            command = "rd /s /q " + resultsDir
            os.system(command)
            command = "mkdir " + resultsDir
            os.system(command)
            writeTopoFile(exp[0] + exp[1])
            writeNodeTypeFile(exp[0], exp[1])
            writeParamFile(oltType)
            writeGlobals(URLLC_AB_MIN,URLLC_AB_SUR,EMBB_AB_MIN,EMBB_AB_SUR,VIDEO_AB_MIN,VIDEO_AB_SUR,IP_AB_MIN,IP_AB_SUR,exp[2],SIM_TIME, GEN_TIME)
            # start experiment
            command = "conda init & " + constants.RUN_COMMAND

            os.system(constants.RUN_COMMAND)
            # copy results to the 
            copyCommand = "Xcopy /E /I {} {}".format(constants.OUTPUT_PATH, resultsDir) 
            os.system(copyCommand)
            copyCommand = "copy {} {}".format(constants.paramFilePath, resultsDir) 
            os.system(copyCommand)
            copyCommand = "copy {} {}".format(constants.nodeTypeFilePath, resultsDir) 
            os.system(copyCommand)
            copyCommand = "copy {} {}".format(constants.topoFilePath, resultsDir) 
            os.system(copyCommand)

''
