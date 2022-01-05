"""TraceUtils.py
"""
from numpy.core.fromnumeric import mean
from Onu import ONU
import os
import Globals
import IO
import Utils
from Components import Node
from Network import Model
import os
import pandas as pd
import numpy as np
import scipy.stats
from stable_baselines import A2C, a2c
from DBA_env_continuos import env_dba



class TU(object):
 
    def __init__(self, env, M):
        self.env = env
        self.M = M
        self.x_urllc = []
        self.meanDealy_URLLC = []
        self.jitter_URLLC = []
        self.x_embb = []
        self.meanDealy_embb = []
        self.jitter_embb = []
        self.x_video = []
        self.meanDealy_video = []
        self.jitter_video = []
        self.x_ip = []
        self.meanDealy_ip = []
        self.jitter_ip = []
        self.x_all = []
        self.meanDealy_all = []
        self.jitter_all = []
        self.urllc_sum =[]
        self.embb_sum =[]
        self.video_sum =[]
        self.ip_sum =[]
        self.gen_sum = []
        self.z =[]
        self.x = []
        self.urllc_dis= []
        self.embb_dis = []
        self.video_dis = []
        self.ip_dis = []
        self.mean_cf_urllc = []
        self.lower_cf_urllc = []
        self.upper_cf_urllc = []
        self.mean_cf_embb = []
        self.lower_cf_embb = []
        self.upper_cf_embb = []
        self.mean_cf_video = []
        self.lower_cf_video = []
        self.upper_cf_video = []
        self.mean_cf_ip = []
        self.lower_cf_ip = []
        self.upper_cf_ip = []
        self.upper_cf_all = []
        self.lower_cf_all = []
        self.mean_cf_all = []
        self.entropy = []
        self.loss = []
        self.reward = []


        



    def print_stats(self, network, verbose=Globals.VERB_NO):
        """Prints statistics collected during the simulation run
        """

        print(" [+] Printing summary statistics:")

        grand_tot_sent = 0
        grand_tot_recv = 0

        for n in network:

            node = network[n]

            if verbose > Globals.VERB_NO:
                print("\n Node {:s} [{:s}]:".format(n, node.type))

            tot_pkt_sent = 0
            tot_pkt_recv = 0

            if verbose > Globals.VERB_NO:
                print("\t Total packets generated: {:,}"
                    .format(len(node.generated)))
                print("\t Total packets forwarded: {:,}"
                    .format(len(node.forwarded)))

            for c in node.conns:

                if verbose > Globals.VERB_NO:
                    print("\t -> connection to {:s}".format(c))
                    print("\t    sent {:,}".format(node.pkt_sent[c]))
                    print("\t    recv {:,}".format(node.pkt_recv[c]))

                tot_pkt_sent = tot_pkt_sent + node.pkt_sent[c]
                tot_pkt_recv = tot_pkt_recv + node.pkt_recv[c]

            if verbose > Globals.VERB_NO:
                print("\t [ total sent: {:,} ]".format(tot_pkt_sent))
                print("\t [ total recv: {:,} ]".format(tot_pkt_recv))

            grand_tot_sent = grand_tot_sent + tot_pkt_sent
            grand_tot_recv = grand_tot_recv + tot_pkt_recv

        print("\n\tTotal packets sent: {:,}".format(grand_tot_sent))
        print("\tTotal packets recv: {:,}\n".format(grand_tot_recv))


    def save_node_names(self, network, output_dir):
        """Saves the list of node names
        """

        file_name = Globals.NODES_LIST_FILE
        file_path = os.path.join(output_dir, file_name)

        fp = IO.open_for_writing(file_path)

        node_names = list(network.keys())
        node_names.sort()

        for node_name in node_names:
            fp.write("{:s}\n".format(node_name))

        IO.close_for_writing(fp)


    def save_queue_mon(self, output_dir, network):
        """For each node, saves node queue length as a function of simulation
        time.
        """

        print(" [+] Saving queue monitors..")

        node_names = list(network.keys())
        node_names.sort()

        for node_name in node_names:

            node = network[node_name]
            node_name = node.name

            file_name = node_name + Globals.QUEUE_SUFFIX
            output_file = os.path.join(output_dir, file_name)

            fp = IO.open_for_writing(output_file)

            # each element of node.queue_mon is a list [time_now, queue_length]
            fp.write("stime,queue_length\n")
            for item in node.queue_mon:
                fp.write("{:f},{:d}\n".format(item[0], item[1]))

            IO.close_for_writing(fp)



    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        w = [m, m-h, m+h]
        return w


    def save_node_pkts(self,network, output_dir, queue_type):
        """For given queue type: for all nodes saves all packets to a file
        """
        counter_print = 0

        if queue_type == 'g':
            print(" [+] Saving generated packets..")
            suffix = Globals.GEN_SUFFIX
        elif queue_type == 'f':
            print(" [+] Saving forwarded packets..")
            suffix = Globals.FWD_SUFFIX
        elif queue_type == 'd':
            print(" [+] Saving discarded packets..")
            suffix = Globals.DISCARD_SUFFIX
        elif queue_type == 'r':
            print(" [+] Saving received packets..")
            suffix = Globals.RECV_SUFFIX
        elif queue_type == 'u':
            print(" [+] Saving URLLC packets..")
            suffix = Globals.URLLC_SUFFIX
        elif queue_type == 'v':
            print(" [+] Saving Video packets..")
            suffix = Globals.VIDEO_SUFFIX
        elif queue_type == 'e':
            print(" [+] Saving eMBB packets..")
            suffix = Globals.EMBB_SUFFIX
        elif queue_type == 'i':
            print(" [+] Saving IP packets..")
            suffix = Globals.IP_SUFFIX
        elif queue_type == 'rU':
            print(" [+] Saving Granted URLLC packets..")
            suffix = Globals.REPROT_URLLC_SUFFIX
        elif queue_type == 'rE':
            print(" [+] Saving Granted eMBB packets..")
            suffix = Globals.REPORT_EMBB_SUFFIX
        elif queue_type == 'rV':
            print(" [+] Saving Granted Video packets..")
            suffix = Globals.REPORT_VIDEO_SUFFIX
        elif queue_type == 'rI':
            print(" [+] Saving Granted IP packets..")
            suffix = Globals.REPORT_IP_SUFFIX
        elif queue_type == 'rM1':
            print(' [+] Saving Report packets..')
            suffix = Globals.REPORT_MSG_SUFFIX
        elif queue_type == 'gM':
            print(' [+] Saving Grant packets..')
            suffix = Globals.GRANT_MSG_SUFFIX
        elif queue_type == 'dU':
            print(" [+] Saving discarded URLLC packets..")
            suffix = Globals.DISCARDED_URLLC_SUFFIX
        elif queue_type == 'dE':
            print(" [+] Saving discarded eMBB packets..")
            suffix = Globals.DISCARDED_EMBB_SUFFIX
        elif queue_type == 'dV':
            print(' [+] Saving discarded video packets..')
            suffix = Globals.DISCARDED_VIDEO_SUFFIX
        elif queue_type == 'dI':
            print(' [+] Saving discarded IP packets..')
            suffix = Globals.DISCARDED_IP_SUFFIX 
        elif queue_type == 'qU':
            print(" [+] Saving length of URLLC buffer..")
            suffix = Globals.LENGTH_URLLC_SUFFIX
        elif queue_type == 'qE':
            print(" [+] Saving length of eMBB buffer..")
            suffix = Globals.LENGTH_EMBB_SUFFIX
        elif queue_type == 'qV':
            print(' [+] Saving length of video buffer..')
            suffix = Globals.LENGTH_VIDEO_SUFFIX
        elif queue_type == 'qI':
            print(' [+] Saving length of IP buffer..')
            suffix = Globals.LENGTH_IP_SUFFIX 
        elif queue_type == 'dS':
            print(' [+] Saving delay stats..')
            suffix = Globals.DELAY_STATS_SUFFIX         
        else:
            Utils.error("Unknown queue '{:s}'".format(queue_type))

        node_names = list(network.keys())
        node_names.sort()
        for node_name in node_names:
            node = network[node_name]
            type = node.type

            if (type == Globals.ONU_TYPE and (queue_type not in ['u','rU', 'e','rE','g', 'dU', 'dE', 'qU', 'qE'])):
                continue
            
            if (type == Globals.ONU_TYPE2 and (queue_type not in ['v', 'i', 'rM1','g', 'rV','rI', 'dV', 'dI', 'qV', 'qI'])):
                continue

            if (type == Globals.OLT_TYPE and (queue_type not in ['g', 'd', 'r', 'gM', 'dS'])):
                continue

            if queue_type == 'g':
                queue = node.generated
            elif queue_type == 'f':
                queue = node.forwarded
            elif queue_type == 'd':
                queue = node.discarded
            elif queue_type == 'r':
                queue = node.received
            elif queue_type == 'u':
                queue = node.generated_ONU1_T1_URLLC
            elif queue_type == 'e':
                queue = node.generated_ONU1_T2_eMBB
            elif queue_type == 'v':
                queue = node.generated_ONU2_T2_Video
            elif queue_type == 'i':
                queue = node.generated_ONU2_T3_IP
            elif queue_type == 'rU':
                queue = node.reported_ONU1_T1_URLLC
            elif queue_type == 'rE':
                queue = node.reported_ONU1_T2_eMBB
            elif queue_type == 'rV':
                queue = node.reported_ONU2_T2_Video
            elif queue_type == 'rI':
                queue = node.reported_ONU2_T3_IP
            elif queue_type == 'rM1':
                queue = node.report_message_ONU
            elif queue_type == 'gM':
                queue = node.Grant_message_OLT   
            elif queue_type == 'dU':
                queue = node.discarded_URLLC
            elif queue_type == 'dE':
                queue = node.discarded_eMBB
            elif queue_type == 'dV':
                queue = node.discarded_Video
            elif queue_type == 'dI':
                queue = node.discarded_IP  
            elif queue_type == 'qU':
                queue = node.queue_lenght_URLLC1
            elif queue_type == 'qE':
                queue = node.queue_lenght_eMBB1
            elif queue_type == 'qV':
                queue = node.queue_lenght_Video1
            elif queue_type == 'qI':
                queue = node.queue_lenght_IP1
            elif queue_type == 'dS': 
                queue = node.meanDealy_all
                queue1 = node.action_all   


            else:
                Utils.error("Unknown queue 2 '{:s}'".format(queue_type))


            file_name = node.name + suffix
            node_file = os.path.join(output_dir, file_name)
            fp = IO.open_for_writing(node_file)
            
            if queue in ['rU', 'rE', 'rV', 'rI']:
                continue
            
            elif queue not in []:
                fp.write("id,timestamp,report_time,packetsize\n")
            
            if (type == Globals.OLT_TYPE and  queue_type == "gM"):
                queue2 = node.report_message_ONU
            
            
            if queue_type in ['rU']:
                if type == Globals.ONU_TYPE:
                    if (len(queue) == 0):
                        continue 
                    self.z = list(zip(*queue))
                    if len(self.z) > 0:
                        w = self.mean_confidence_interval(self.z[0], 0.95)
                        self.x_urllc.append(self.z[0])
                        self.meanDealy_URLLC.append(mean(self.z[0])) # meanDealy is a list containing the delay of each ONU
                        self.mean_cf_urllc.append(w[0]), self.lower_cf_urllc.append(w[1]), self.upper_cf_urllc.append(w[2])
                        x = []
                        for i in range(len(self.z[0])-1):
                            x.append(abs(self.z[0][i]- self.z[0][i+1]))
                        self.jitter_URLLC.append(mean(x)) # jitter is a list containing the jitter of each ONU
                        
                    # print(jitter_URLLC)
            
            if queue_type in ['rE']:
                # for node_name in node_names:
                #     node = network[node_name]
                #     type = node.type
                if type == Globals.ONU_TYPE:
                    if (len(queue) == 0):
                        continue 
                    self.z = list(zip(*queue))
                    if len(self.z) > 0:
                        w = self.mean_confidence_interval(self.z[0], 0.95)
                        self.x_embb.append(self.z[0])
                        self.meanDealy_embb.append(mean(self.z[0])) # meanDealy is a list containing the delay of each ONU
                        self.mean_cf_embb.append(w[0]), self.lower_cf_embb.append(w[1]), self.upper_cf_embb.append(w[2])
                        x = []
                        for i in range(len(self.z[0])-1):
                            x.append(abs(self.z[0][i]- self.z[0][i+1]))
                        self.jitter_embb.append(mean(x))

                        # print(jitter_embb)
            
            if queue_type in ['rV']:
                # for node_name in node_names:
                #     node = network[node_name]
                #     type = node.type
                if type == Globals.ONU_TYPE2:
                    if (len(queue) == 0):
                        continue
                    self.z = list(zip(*queue))
                    if len(self.z) > 0:
                        w = self.mean_confidence_interval(self.z[0], 0.95)
                        self.x_video.append(self.z[0])
                        self.meanDealy_video.append(mean(self.z[0])) # meanDealy is a list containing the delay of each ONU
                        self.mean_cf_video.append(w[0]), self.lower_cf_video.append(w[1]), self.upper_cf_video.append(w[2])
                        x = []
                        for i in range(len(self.z[0])-1):
                            x.append(abs(self.z[0][i]- self.z[0][i+1]))
                        self.jitter_video.append(mean(x)) # jitter is a list containing the jitter of each ONU


                        # print(jitter_video)

            if queue_type in ['rI']:
                # for node_name in node_names:
                #     node = network[node_name]
                #     type = node.type
                if type == Globals.ONU_TYPE2:
                    if (len(queue) == 0):
                        continue
                    self.z = list(zip(*queue))
                    if len(self.z) > 0:
                        w = self.mean_confidence_interval(self.z[0], 0.95)
                        self.x_ip.append(self.z[0])
                        self.meanDealy_ip.append(mean(self.z[0])) # meanDealy is a list containing the delay of each ONU
                        self.mean_cf_ip.append(w[0]), self.lower_cf_ip.append(w[1]), self.upper_cf_ip.append(w[2])
                        x = []
                        for i in range(len(self.z[0])-1):
                            x.append(abs(self.z[0][i]- self.z[0][i+1]))
                        self.jitter_ip.append(mean(x)) # jitter is a list containing the jitter of each ONU
                        

                        # print(jitter_ip)
            
            if queue_type in ['u', 'e', 'v', 'i']:
                if type == Globals.ONU_TYPE and queue_type =='u':
                    if (len(queue) == 0):
                        continue
                    stime, self.z = list(zip(*queue)) 
                    if len(self.z) > 0:
                        self.x_urllc = []
                        for items in self.z:
                            self.x_urllc.append(len(items['PAYLOAD']))
                        self.urllc_sum.append(sum(self.x_urllc)) # meanDealy is a list containing the delay of each ONU
            
                if type == Globals.ONU_TYPE and queue_type in ['e']:
                    if (len(queue) == 0):
                        continue
                    stime, self.z = list(zip(*queue))
                    if len(self.z) > 0:
                        self.x_embb =[]
                        for items in self.z:
                            self.x_embb.append(len(items['PAYLOAD']))
                        self.embb_sum.append(sum(self.x_embb)) # meanDealy is a list containing the delay of each ONU
            
                if type == Globals.ONU_TYPE2 and queue_type in ['v']:
                    if (len(queue) == 0):
                        continue
                    stime, self.z = list(zip(*queue))
                    if len(self.z) > 0:
                        self.x_video = []
                        for items in self.z:
                            self.x_video.append(len(items['PAYLOAD']))
                        self.video_sum.append(sum(self.x_video)) # meanDealy is a list containing the delay of each ONU

                if type == Globals.ONU_TYPE2 and queue_type in ['i']:
                    if (len(queue) == 0):
                        continue
                    stime, self.z = list(zip(*queue))
                    if len(self.z) > 0:
                        self.x_ip = []
                        for items in self.z:
                            self.x_ip.append(len(items['PAYLOAD']))
                        self.ip_sum.append(sum(self.x_ip)) # meanDealy is a list containing the delay of each ONU

            if queue_type in ['dU', 'dE', 'dV', 'dI']:
                if type == Globals.ONU_TYPE and queue_type =='dU':
                    if (len(queue) == 0):
                        continue
                    stime = list(zip(*queue)) if len(list(zip(*queue))) > 0 else [[0], [0]]
                    if len(stime) > 0:
                        self.urllc_dis.append(len(stime[0])) # meanDealy is a list containing the delay of each ONU
                        print (self.urllc_dis)
            
                if type == Globals.ONU_TYPE and queue_type in ['dE']:
                    if (len(queue) == 0):
                        continue
                    stime= list(zip(*queue)) if len(list(zip(*queue))) > 0 else [[0], [0]]
                    if len(stime) > 0:
                        self.embb_dis.append(len(stime[0]))
                        print (self.embb_dis)

                if type == Globals.ONU_TYPE2 and queue_type in ['dV']:
                    if (len(queue) == 0):
                        continue
                    stime= list(zip(*queue)) if len(list(zip(*queue))) > 0 else [[0], [0]]
                    if len(stime) > 0:
                        self.video_dis.append(len(stime[0]))
                        print (self.video_dis)

                    
                if type == Globals.ONU_TYPE2 and queue_type in ['dI']:
                    if (len(queue) == 0):
                        continue
                    stime= list(zip(*queue)) if len(list(zip(*queue))) > 0 else [[0], [0]]
                    if len(stime) > 0:
                        self.ip_dis.append(len(stime[0])) if len(stime[0]) > 0  else 0
                        print (self.ip_dis)

          
          
            for item in queue:
                stime, pkt = item
                
                
                #### printing grants with request ####
                if ( type == Globals.OLT_TYPE and  queue_type == "gM"):
                    str1 = ""
                    for x in pkt[Globals.GRANT_PACKET_BWM_FIELD]:
                        for y in x:
                            str1 = str1 + str(x[y]) + ","
                    
                    str1 = str1 + ",,"
                    queueLengthEl = queue2.popleft()
                    for onu in queueLengthEl[1]:
                        for tcon in queueLengthEl[1][onu]: 
                            str1 = str1 + str(queueLengthEl[1][onu][tcon]) + ","
                    fp.write("{:d},{:f},{:s}\n"
                            .format(pkt[Globals.ID], pkt[Globals.TIME_STAMP], (str1)))
               
               
                #### printing queue lenght ####
                elif (queue_type in ['qU', 'qE', 'qV', 'qI']):    
                     fp.write("{:f},{:d}\n".format(item[0], item[1]))
                
                ##### printing delay stats


                
                ##### printing reported packets #####

                else:   
                    fp.write("{:d},{:f},{:f},{:f},{:f},{:s},{:s}\n"
                                    .format(pkt[Globals.ID], pkt[Globals.TIME_STAMP],  pkt[Globals.REPORT_TIME], len(pkt[Globals.REPORT_PACKET_PAYLOAD_FIELD]), stime,  "" , ""))                    
            IO.close_for_writing(fp)

    
        if queue_type in ['dS']:
            self.meanDealy_all = [self.meanDealy_URLLC, self.meanDealy_embb, self.meanDealy_video, self.meanDealy_ip]
            self.jitter_all = [self.jitter_URLLC, self.jitter_embb, self.jitter_video, self.jitter_ip]
            self.discard_all = [self.urllc_dis, self.embb_dis, self.video_dis, self.ip_dis]
            self.upper_cf_all = [self.upper_cf_urllc,self.upper_cf_embb,self.upper_cf_video,self.upper_cf_ip]
            self.lower_cf_all = [self.lower_cf_urllc,self.lower_cf_embb,self.lower_cf_video,self.lower_cf_ip]
            self.mean_cf_all = [self.mean_cf_urllc,self.mean_cf_embb,self.mean_cf_video,self.mean_cf_ip]
           
            file_name = 'delay_total.csv'
            node_file = os.path.join(output_dir, file_name)
            fp1 = IO.open_for_writing(node_file)
            fp1.write("Avg delay,")
            for i in range(max(len(self.meanDealy_URLLC), len(self.meanDealy_video))):
                fp1.write("onu" + str(i+1) + ',')
            fp1.write('\n')

            ########################## printing delay #################################
            if (queue_type == 'dS'):
                # l1 = [1, 2 , 3 , 4]
                # meanDealy_all =  [l1, l1, l1, l1, l1]
                if len(self.meanDealy_all[0]) > 0:
                    fp1.write("{},".format(mean(self.meanDealy_URLLC)))
                    for i in self.meanDealy_URLLC:
                        # print(len(self.meanDealy_all))
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))

                fp1.write("\n")

                if len(self.meanDealy_all[1]) > 0:
                    fp1.write("{},".format(mean(self.meanDealy_embb)))
                    for i in self.meanDealy_embb:
                    #     print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.meanDealy_all[2]) > 0:
                    fp1.write("{},".format(mean(self.meanDealy_video)))
                    for i in self.meanDealy_video:
                         # print(len(self.meanDealy_video))
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.meanDealy_all[3]) > 0:
                    fp1.write("{},".format(mean(self.meanDealy_ip)))
                    for i in self.meanDealy_ip:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))


            ################################## printing jitter #################################

                fp1.write("\n\n")
                fp1.write(",")
                # for i in range(max(len(self.meanDealy_URLLC), len(self.meanDealy_video))):
                #     fp1.write("onu" + str(i+1) + ',')
                fp1.write('\n')
                # for i in range(max(len(self.meanDealy_URLLC), len(self.meanDealy_video))):
                #     fp1.write("onu" + str(i+1) + ',')
                
                if len(self.jitter_all[0]) > 0:
                    fp1.write("{},".format(mean(self.jitter_URLLC)))
                    for i in self.jitter_URLLC:
                        # print(len(self.meanDealy_all))
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.jitter_all[1]) > 0:
                    fp1.write("{},".format(mean(self.jitter_embb)))
                    for i in self.jitter_embb:
                    #     print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.jitter_all[2]) > 0:
                    fp1.write("{},".format(mean(self.jitter_video)))
                    for i in self.jitter_video:
                         # print(len(self.meanDealy_video))
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.jitter_all[3]) > 0:
                    fp1.write("{},".format(mean(self.jitter_ip)))
                    for i in self.jitter_ip:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))

                
            
            ############################## printing confidence delay  #############################
                fp1.write("\n\n")
                fp1.write(",")
                # for i in range(max(len(self.meanDealy_URLLC), len(self.meanDealy_video))):
                #     fp1.write("onu" + str(i+1) + ',')
                fp1.write('\n')

                if len(self.mean_cf_all[0]) > 0:
                    fp1.write("{},".format(mean(self.mean_cf_urllc)))
                    for i in self.mean_cf_urllc:
                        # print(len(self.meanDealy_all))
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.mean_cf_all[1]) > 0:
                    fp1.write("{},".format(mean(self.mean_cf_embb)))
                    for i in self.mean_cf_embb:
                    #     print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.mean_cf_all[2]) > 0:
                    fp1.write("{},".format(mean(self.mean_cf_video)))
                    for i in self.mean_cf_video:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                
                if len(self.mean_cf_all[3]) > 0:
                    fp1.write("{},".format(mean(self.mean_cf_ip)))
                    for i in self.mean_cf_ip:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))

                ############################## mean confidence ###############################
                fp1.write("\n\n")
                fp1.write(",")
                # for i in range(max(len(self.meanDealy_URLLC), len(self.meanDealy_video))):
                #     fp1.write("onu" + str(i+1) + ',')
                fp1.write('\n')

                if len(self.upper_cf_all[0]) > 0:
                    fp1.write("{},".format(mean(self.upper_cf_urllc)))
                    for i in self.upper_cf_urllc:
                        # print(len(self.meanDealy_all))
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.upper_cf_all[1]) > 0:
                    fp1.write("{},".format(mean(self.upper_cf_embb)))
                    for i in self.upper_cf_embb:
                    #     print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.upper_cf_all[2]) > 0:
                    fp1.write("{},".format(mean(self.upper_cf_video)))
                    for i in self.upper_cf_video:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                
                if len(self.upper_cf_all[3]) > 0:
                    fp1.write("{},".format(mean(self.upper_cf_ip)))
                    for i in self.upper_cf_ip:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))

                ######################################### lower confidence #######################

                fp1.write("\n\n\n")
                fp1.write(",")
                # for i in range(max(len(self.meanDealy_URLLC), len(self.meanDealy_video))):
                #     fp1.write("onu" + str(i+1) + ',')
                fp1.write('\n')

                if len(self.lower_cf_all[0]) > 0:
                    fp1.write("{},".format(mean(self.lower_cf_urllc)))
                    for i in self.lower_cf_urllc:
                        # print(len(self.meanDealy_all))
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.lower_cf_all[1]) > 0:
                    fp1.write("{},".format(mean(self.lower_cf_embb)))
                    for i in self.lower_cf_embb:
                    #     print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                if len(self.lower_cf_all[2]) > 0:
                    fp1.write("{},".format(mean(self.lower_cf_video)))
                    for i in self.lower_cf_video:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))
                fp1.write("\n")
                
                if len(self.lower_cf_all[3]) > 0:
                    fp1.write("{},".format(mean(self.lower_cf_ip)))
                    for i in self.lower_cf_ip:
                        # print(i)
                        fp1.write("{},".format(i))
                else: 
                    fp1.write("{},".format(0))

            


            ###################### printing discard ######################
            fp1.write("\n\n\n")
            total_pkts = int(Globals.SIM_TIME/0.000125)
            fp1.write(',')
            # for i in range(max(len(self.urllc_dis), len(self.video_dis)) -2):
            #     fp1.write("onu" + str(i+1) + ',')
            fp1.write(str(total_pkts)+ ",")
            fp1.write('\n')
            if len(self.discard_all[0]) > 0:
                fp1.write("{},".format(mean(self.urllc_dis)/total_pkts))
                for i in self.urllc_dis:
                    # print(len(self.meanDealy_all))
                    fp1.write("{},".format(i))
            else: 
                fp1.write("{},".format(0))
            fp1.write("\n")
            if len(self.discard_all[1]) > 0:
                fp1.write("{},".format(mean(self.embb_dis)/total_pkts))
                for i in self.embb_dis:
                #     print(i)
                    fp1.write("{},".format(i))
            else: 
                fp1.write("{},".format(0))
            fp1.write("\n")
            if len(self.discard_all[2]) > 0:
                fp1.write("{},".format(mean(self.video_dis)/total_pkts))
                for i in self.video_dis:
                        # print(len(self.meanDealy_video))
                    # print(i)
                    fp1.write("{},".format(i))
            else: 
                fp1.write("{},".format(0))
            fp1.write("\n")
            if len(self.discard_all[3]) > 0:
                fp1.write("{},".format(mean(self.ip_dis)/total_pkts))
                for i in self.ip_dis:
                    # print(i)
                    fp1.write("{},".format(i))
            else: 
                fp1.write("{},".format(0))
            # IO.close_for_writing(fp2)
            IO.close_for_writing(fp1)

######################################### ENTROPY and REWARD ###################################
        if queue_type == 'dS':
            
            episode, reward = self.M.rlEnv.reward_stat
            # print(len)
            # entropy, loss  = self.M.model.get_entropy()
            # print(len(entropy), len(reward), len(loss), len(episode))
            file_name = 'reward.csv'
            node_file = os.path.join(output_dir, file_name)
            fp1 = IO.open_for_writing(node_file)
            for i in range(len(reward)):
                fp1.write("{:f},{:f},{:f},{:f}\n"
                                    .format( episode[i],reward[i], 0, 0 ))
            IO.close_for_writing(fp1)
        
        if queue_type == 'dS': 
            trainingStat = self.M.rlEnv.trainingStat
            # print(len)
            # entropy, loss  = self.M.model.get_entropy()
            # print(len(entropy), len(reward), len(loss), len(episode))
            file_name = 'training_stats.csv'
            node_file = os.path.join(output_dir, file_name)
            fp1 = IO.open_for_writing(node_file)
            for i in range(len(trainingStat)):
                fp1.write(trainingStat[i])
            IO.close_for_writing(fp1)
        
        if queue_type == 'dS':
            list1 = list(queue1)
            file_name = 'actions.csv'
            node_file = os.path.join(output_dir, file_name)
            fp1 = IO.open_for_writing(node_file)
            for i in list1:
                for items in i:
                    fp1.write("{:f},".format(items))
                fp1.write("\n")
            IO.close_for_writing(fp1)

  
        if queue_type == 'dS':
            self.gen_sum = [self.urllc_sum, self.embb_sum, self.video_sum, self.ip_sum]
            file_name = 'gen_total.csv'
            node_file = os.path.join(output_dir, file_name)
            fp1 = IO.open_for_writing(node_file)
            total_sum = sum([sum(i) for i in self.gen_sum])
            steps = (Globals.SIM_TIME/0.000125)
            fp1.write("total capacity, total requested, load/capcity\n")
            fp1.write("{},{},{}\n".format(steps*Globals.FB, total_sum, total_sum/(steps*Globals.FB)))
            x0 = (steps*Globals.URLLC_AB_MIN/Globals.URLLC_SI_MIN + steps*Globals.URLLC_AB_SUR/Globals.URLLC_SI_MAX)*len(self.urllc_sum)
            x1 = sum(self.urllc_sum)
            x2 = sum(self.urllc_sum)/x0 if (x0 > 0) else 0
            x3 = (steps*Globals.EMBB_AB_MIN/Globals.EMBB_SI_MIN + steps*Globals.EMBB_AB_SUR/Globals.EMBB_SI_MAX)*len(self.embb_sum)
            x4 = sum(self.embb_sum)
            x5 = sum(self.embb_sum)/x3 if (x3 > 0) else 0
            x6 = (steps*Globals.VIDEO_AB_MIN/Globals.VIDEO_SI_MIN + steps*Globals.VIDEO_AB_SUR/Globals.VIDEO_SI_MAX)*len(self.video_sum)
            x7 = sum(self.video_sum)
            x8 = sum(self.video_sum)/x6 if (x6 > 0) else 0
            x9 = (steps*Globals.IP_AB_MIN/Globals.IP_SI_MAX +  steps*Globals.IP_AB_SUR/Globals.IP_SI_MAX)*len(self.ip_sum)
            x10 = sum(self.ip_sum)
            x11 = sum(self.ip_sum)/x9 if (x9 > 0) else 0
            fp1.write("{},{},{}\n{},{},{}\n{},{},{}\n{},{},{}\n".format(x0, x1, x2 ,x3, x4, x5, x6, x7, x8, x9, x10, x11))

        # if queue_type == 'dS':
        #     self.gen_sum = [self.urllc_sum, self.embb_sum, self.video_sum, self.ip_sum]
        #     file_name = 'gen.csv'
        #     node_file = os.path.join(output_dir, file_name)
        #     fp1 = IO.open_for_writing(node_file)
        #     total_sum = sum([sum(i) for i in self.gen_sum])
        #     steps = (Globals.SIM_TIME/0.000125)
        #     fp1.write("total capacity, total requested, load/capcity\n")
        #     fp1.write("{},{},{}\n".format(steps*Globals.FB, total_sum, total_sum/(steps*Globals.FB)))
        #     x0 = (steps*Globals.URLLC_AB_MIN/Globals.URLLC_SI_MIN + steps*Globals.URLLC_AB_SUR/Globals.URLLC_SI_MAX)*len(self.urllc_sum)
        #     x1 = sum(self.urllc_sum)
        #     x2 = sum(self.urllc_sum)/x0 if (x0 > 0) else 0
        #     x3 = (steps*Globals.EMBB_AB_MIN/Globals.EMBB_SI_MIN + steps*Globals.EMBB_AB_SUR/Globals.EMBB_SI_MAX)*len(self.embb_sum)
        #     x4 = sum(self.embb_sum)
        #     x5 = sum(self.embb_sum)/x3 if (x3 > 0) else 0
        #     x6 = (steps*Globals.VIDEO_AB_MIN/Globals.VIDEO_SI_MIN + steps*Globals.VIDEO_AB_SUR/Globals.VIDEO_SI_MAX)*len(self.video_sum)
        #     x7 = sum(self.video_sum)
        #     x8 = sum(self.video_sum)/x6 if (x6 > 0) else 0
        #     x9 = (steps*Globals.IP_AB_MIN/Globals.IP_SI_MAX +  steps*Globals.IP_AB_SUR/Globals.IP_SI_MAX)*len(self.ip_sum)
        #     x10 = sum(self.ip_sum)
        #     x11 = sum(self.ip_sum)/x9 if (x9 > 0) else 0
        #     fp1.write("{},{},{}\n{},{},{}\n{},{},{}\n{},{},{}\n".format(x0, x1, x2 ,x3, x4, x5, x6, x7, x8, x9, x10, x11))


         


            
            # fp1.write("{}\n{}\n{}\n{}".format(sum(self.urllc_sum), sum(self.embb_sum), sum(self.video_sum), sum(self.ip_sum)))

            # fp1.write("{}\n{}\n{}\n{}".format((sum(self.urllc_sum))/(steps*Globals.URLLC_AB_MIN/Globals.URLLC_SI_MIN + steps*Globals.URLLC_AB_SUR/Globals.URLLC_SI_MAX)*len(self.urllc_sum),
            #  sum(self.embb_sum)/(steps*Globals.EMBB_AB_MIN/Globals.EMBB_SI_MIN + steps*Globals.EMBB_AB_SUR/Globals.EMBB_SI_MAX)*len(self.embb_sum), 
            #  sum(self.video_sum)/(steps*Globals.VIDEO_AB_MIN/Globals.VIDEO_SI_MIN + steps*Globals.VIDEO_AB_SUR/Globals.VIDEO_SI_MAX)*len(self.video_sum), 
            #  sum(self.ip_sum)/(steps*Globals.IP_AB_MIN/Globals.IP_SI_MAX +  steps*Globals.IP_AB_SUR/Globals.IP_SI_MAX)*len(self.ip_sum)))
            
            # fp1.write('\n\n\n\n\n\n')
            # for i in range(len(self.urllc_sum)):
            #     fp1.write("urllc" + str(i+1) + ',')
            # for i in range(len(self.embb_sum)):
            #     fp1.write("embb" + str(i+1) + ',')
            # for i in range(len(self.video_sum)):
            #     fp1.write("video" + str(i+1) + ',')
            # for i in range(len(self.ip_sum)):
            #     fp1.write("ip" + str(i+1) + ',')
            # fp1.write('\n')
            #     # l1 = [1, 2 , 3 , 4]
            #     # meanDealy_all =  [l1, l1, l1, l1, l1]
            # if len(self.gen_sum[0]) > 0:
            #     # fp1.write("{},".format(sum(self.urllc_sum)))
            #     for i in self.urllc_sum:
            #         fp1.write("{},".format(i))
            #         # fp1.write("\n")
            # if len(self.gen_sum[1]) > 0:
            #     # fp1.write("{}".format(sum(self.embb_sum)))
            #     for i in self.embb_sum:
            #         fp1.write("{},".format(i))
            #         # fp1.write("\n")
            # if len(self.gen_sum[2]) > 0:
            #     # fp1.write("{}".format(sum(self.video_sum)))
            #     for i in self.video_sum:
            #         fp1.write("{},".format(i))
            #         # fp1.write("\n")    
            # if len(self.gen_sum[3]) > 0:
            #     # fp1.write("{},".format(sum(self.ip_sum)))
            #     for i in self.ip_sum:
            #         # print(i)
            #         fp1.write("{},".format(i))
            #         # fp1.wriSte("\n")

            IO.close_for_writing(fp1)
            print (self.gen_sum, '\n' ,self.urllc_sum)


    
            
           

            


        

    