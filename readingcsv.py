import pandas as pd
import IO
import os
import numpy as np 


root = 'D:\\anxCopy\\results\\'
delayDict ={}
loadDict = {}
expDict= {} # {exp: data} 
i = 0
header = ['Avg delay',  ' load/capcity']
type = ['rl_predict', 'ibu', 'g']
for typ in type:
    rootdir = os.path.join(root, typ)
    dirs = os.listdir(rootdir)
    delayDict[typ] = {}
    loadDict[typ]= {}
    for x in dirs:
        delayDict[typ][x] = []
        loadDict[typ][x] = []
        current_dir = os.path.join(rootdir, x)
        subdir = os.listdir(current_dir)
        for y in subdir:
            dir = os.path.join(current_dir, y)
            if os.path.isdir(dir) == False:
                continue
            files = os.listdir(dir)
            for file in files:
                if file == "combined.csv":
                    combined = os.path.join(dir, file)
                    data = pd.read_csv(combined)
                    delay =  data[header[0]].values.tolist()
                    load = data[header[1]].values.tolist()
                    delayDict[typ][x].append(delay)
                    loadDict[typ][x].append(load)
                    # print(delayDict, loadDict)
                    i = i + 1
######### dealy for RL + basic ##################

delay_avg_urllc = []
delay_avg_embb = []
delay_avg_video = []
delay_avg_ip = []



load_avg_urllc = []
load_avg_embb = []
load_avg_video = []
load_avg_ip = []

############## jitter for RL + basic ###############
jitter_avg_urllc = []
jitter_avg_embb = []
jitter_avg_video = []
jitter_avg_ip = []

################## drop for RL + basic ###############
drop_avg_urllc = []
drop_avg_embb = []
drop_avg_video = []
drop_avg_ip = []

trial_avg_rl = []
trial_std_rl = []
trial_avg = []
trial_std = []
load_avg_rl = []
load_std_rl = []
load_avg = []
load_std = []

delayMat = [delay_avg_urllc, delay_avg_embb, delay_avg_video, delay_avg_ip]


for type in delayDict:
    urllc_delay_avg_exp = []
    embb_delay_avg_exp = []
    video_delay_avg_exp = []
    ip_delay_avg_exp = []

    urllc_jitter_avg_exp = []
    embb_jitter_avg_exp = []
    video_jitter_avg_exp = []
    ip_jitter_avg_exp = []

    urllc_drop_avg_exp = []
    embb_drop_avg_exp = []
    video_drop_avg_exp = []
    ip_drop_avg_exp = []

    urllc_load_avg_exp = []
    embb_load_avg_exp = []
    video_load_avg_exp = []
    ip_load_avg_exp = []
    total_load_avg_exp = []

    for experimnet in delayDict[type]:
        delay_avg_urllc = []
        delay_avg_embb = []
        delay_avg_video = []
        delay_avg_ip = []
        load_avg_total = []
        load_avg_urllc = []
        load_avg_embb = []
        load_avg_video = []
        load_avg_ip = []
        jitter_avg_urllc = []
        jitter_avg_embb = []
        jitter_avg_video = []
        jitter_avg_ip = []
        drop_avg_urllc = []
        drop_avg_embb = []
        drop_avg_video = []
        drop_avg_ip = []
       

        for delay in delayDict[type][experimnet]:
            delay_avg_urllc.append((delay[0]))
            delay_avg_embb.append((delay[1]))
            delay_avg_video.append((delay[2]))
            delay_avg_ip.append((delay[3]))
            jitter_avg_urllc.append((delay[5]))
            jitter_avg_embb.append((delay[6]))
            jitter_avg_video.append((delay[7]))
            jitter_avg_ip.append((delay[8]))
            drop_avg_urllc.append((delay[25]))
            drop_avg_embb.append((delay[26]))
            drop_avg_video.append((delay[27]))
            drop_avg_ip.append((delay[28]))
        
        for load in loadDict[type][experimnet]:
            load_avg_total.append(load[0]) 
            load_avg_urllc.append(load[1])
            load_avg_embb.append(load[2]) 
            load_avg_video.append(load[3]) 
            load_avg_ip.append(load[4])

        
        delay_avg = [np.mean(delay_avg_urllc), np.mean(delay_avg_embb), np.mean(delay_avg_video), np.mean(delay_avg_ip)]
        delay_std = [np.std(delay_avg_urllc), np.std(delay_avg_embb), np.std(delay_avg_video), np.std(delay_avg_ip)]
        file_name =  type + "_" + experimnet + '.csv'
        node_file = os.path.join(root, file_name)
        fp1 = IO.open_for_writing(node_file)
        delayMat = [delay_avg_urllc, delay_avg_embb, delay_avg_video, delay_avg_ip]
        jitterMat = [jitter_avg_urllc, jitter_avg_embb , jitter_avg_video , jitter_avg_ip]
        dropMat = [drop_avg_urllc, drop_avg_embb, drop_avg_video, drop_avg_ip]
        loadMat = [load_avg_total, load_avg_urllc, load_avg_embb, load_avg_video ,load_avg_ip]
        
        fp1.write("Delay\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:s},".format("trial" + str(i)))
        fp1.write("{:s},{:s}".format("avg" , "std"))
        fp1.write("\n")

        

        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(delay_avg_urllc[i]))
        fp1.write("{:f},".format(np.mean(delay_avg_urllc)))
        fp1.write("{:f},".format(np.std(delay_avg_urllc)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(delay_avg_embb[i]))
        fp1.write("{:f},".format(np.mean(delay_avg_embb)))
        fp1.write("{:f},".format(np.std(delay_avg_embb)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(delay_avg_video[i]))
        fp1.write("{:f},".format(np.mean(delay_avg_video)))
        fp1.write("{:f},".format(np.std(delay_avg_video)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(delay_avg_ip[i]))
        fp1.write("{:f},".format(np.mean(delay_avg_ip)))
        fp1.write("{:f},".format(np.std(delay_avg_ip)))
        fp1.write("\n")
        fp1.write("\n")
        fp1.write("\n")


        fp1.write("Jitter\n")

        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(jitter_avg_urllc[i]))
        fp1.write("{:f},".format(np.mean(jitter_avg_urllc)))
        fp1.write("{:f},".format(np.std(jitter_avg_urllc)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(jitter_avg_embb[i]))
        fp1.write("{:f},".format(np.mean(jitter_avg_embb)))
        fp1.write("{:f},".format(np.std(jitter_avg_embb)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(jitter_avg_video[i]))
        fp1.write("{:f},".format(np.mean(jitter_avg_video)))
        fp1.write("{:f},".format(np.std(jitter_avg_video)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(jitter_avg_ip[i]))
        fp1.write("{:f},".format(np.mean(jitter_avg_ip)))
        fp1.write("{:f},".format(np.std(jitter_avg_ip)))

        fp1.write("\n")
        fp1.write("\n")
        fp1.write("\n")


        fp1.write("Drop\n")

        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(drop_avg_urllc[i]))
        fp1.write("{:f},".format(np.mean(drop_avg_urllc)))
        fp1.write("{:f},".format(np.std(drop_avg_urllc)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(drop_avg_embb[i]))
        fp1.write("{:f},".format(np.mean(drop_avg_embb)))
        fp1.write("{:f},".format(np.std(drop_avg_embb)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(drop_avg_video[i]))
        fp1.write("{:f},".format(np.mean(drop_avg_video)))
        fp1.write("{:f},".format(np.std(drop_avg_video)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(drop_avg_ip[i]))
        fp1.write("{:f},".format(np.mean(drop_avg_ip)))
        fp1.write("{:f},".format(np.std(drop_avg_ip)))

        fp1.write("\n")
        fp1.write("\n")
        fp1.write("\n")

        fp1.write("load\n")

        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(load_avg_urllc[i]))
        fp1.write("{:f},".format(np.mean(load_avg_urllc)))
        fp1.write("{:f},".format(np.std(load_avg_urllc)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(load_avg_embb[i]))
        fp1.write("{:f},".format(np.mean(load_avg_embb)))
        fp1.write("{:f},".format(np.std(load_avg_embb)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(load_avg_video[i]))
        fp1.write("{:f},".format(np.mean(load_avg_video)))
        fp1.write("{:f},".format(np.std(load_avg_video)))
        fp1.write("\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(load_avg_ip[i]))
        fp1.write("{:f},".format(np.mean(load_avg_ip)))
        fp1.write("{:f},".format(np.std(load_avg_ip)))

        fp1.write("\n")
        fp1.write("\n")
        fp1.write("\n")
        fp1.write("total load\n")
        for i in range(len(delay_avg_urllc)):
            fp1.write("{:f},".format(load_avg_total[i]))
        fp1.write("{:f},".format(np.mean(load_avg_total)))
        fp1.write("{:f},".format(np.std(load_avg_total)))
        IO.close_for_writing(fp1)

    
        urllc_delay_avg_exp.append(np.mean(delay_avg_urllc))
        urllc_delay_avg_exp.append(np.std(delay_avg_urllc))
        embb_delay_avg_exp.append(np.mean(delay_avg_embb))
        embb_delay_avg_exp.append(np.std(delay_avg_embb))
        video_delay_avg_exp.append(np.mean(delay_avg_video))
        video_delay_avg_exp.append(np.std(delay_avg_video))
        ip_delay_avg_exp.append(np.mean(delay_avg_ip))
        ip_delay_avg_exp.append(np.std(delay_avg_ip))

        urllc_jitter_avg_exp.append(np.mean(jitter_avg_urllc))
        urllc_jitter_avg_exp.append(np.std(jitter_avg_urllc))
        embb_jitter_avg_exp.append(np.mean(jitter_avg_embb))
        embb_jitter_avg_exp.append(np.std(jitter_avg_embb))
        video_jitter_avg_exp.append(np.mean(jitter_avg_video))
        video_jitter_avg_exp.append(np.std(jitter_avg_video))
        ip_jitter_avg_exp.append(np.mean(jitter_avg_ip))
        ip_jitter_avg_exp.append(np.std(jitter_avg_ip))

        urllc_drop_avg_exp.append(np.mean(drop_avg_urllc))
        urllc_drop_avg_exp.append(np.std(drop_avg_urllc))
        embb_drop_avg_exp.append(np.mean(drop_avg_embb))
        embb_drop_avg_exp.append(np.std(drop_avg_embb))
        video_drop_avg_exp.append(np.mean(drop_avg_video))
        video_drop_avg_exp.append(np.std(drop_avg_video))
        ip_drop_avg_exp.append(np.mean(drop_avg_ip))
        ip_drop_avg_exp.append(np.std(drop_avg_ip))

        urllc_load_avg_exp.append(np.mean(load_avg_urllc))
        urllc_load_avg_exp.append(np.std(load_avg_urllc))
        embb_load_avg_exp.append(np.mean(load_avg_embb))
        embb_load_avg_exp.append(np.std(load_avg_embb))
        video_load_avg_exp.append(np.mean(load_avg_video))
        video_load_avg_exp.append(np.std(load_avg_video))
        ip_load_avg_exp.append(np.mean(load_avg_ip))
        ip_load_avg_exp.append(np.std(load_avg_ip))
        total_load_avg_exp.append(np.mean(load_avg_total))
        total_load_avg_exp.append(np.std(load_avg_total))

    
    file_name = type + '_summary' + '.csv'
    node_file = os.path.join(root, file_name)
    fp1 = IO.open_for_writing(node_file)

    counter = 0
    line = 'delay\n'
    fp1.write(line)
    line = 'exp,load,urllc,std,embb,std,video,std,ip,std\n'
    fp1.write(line)
    exps =  list(delayDict[type].keys())
    for i in range(len(exps)):
        try:
            fp1.write(exps[i].split("-")[0] + ',')
            fp1.write(str(total_load_avg_exp[counter]) + ',')
            line1 = str(urllc_delay_avg_exp[counter]) + ','+ str(urllc_delay_avg_exp[counter + 1]) + ',' + str(embb_delay_avg_exp[counter]) + ',' + str(embb_delay_avg_exp[counter + 1]) + ',' + str(video_delay_avg_exp[counter]) + ',' + str(video_delay_avg_exp[counter+ 1]) + ',' + str(ip_delay_avg_exp[counter]) + ',' + str(ip_delay_avg_exp[counter + 1])
            fp1.write(line1)
            fp1.write("\n")
            counter = counter + 2
        except:
            counter = counter + 2
            fp1.write("\n")
            continue

    fp1.write("\n")
    fp1.write("\n")
    line = 'jitter\n'
    fp1.write(line)
    

    counter = 0
    
    for i in range(len(exps)):
        try:
            fp1.write(exps[i].split("-")[0] + ',')
            fp1.write(str(total_load_avg_exp[counter]) + ',')
            line1 = str(urllc_jitter_avg_exp[counter]) + ','+ str(urllc_jitter_avg_exp[counter + 1]) + ',' + str(embb_jitter_avg_exp[counter]) + ',' + str(embb_jitter_avg_exp[counter + 1]) + ',' + str(video_jitter_avg_exp[counter]) + ',' + str(video_jitter_avg_exp[counter+ 1]) + ',' + str(ip_jitter_avg_exp[counter]) + ',' + str(ip_jitter_avg_exp[counter + 1])
            fp1.write(line1)
            fp1.write("\n")
            counter = counter + 2
        except:
            counter = counter + 2
            fp1.write("\n")
            continue
    
    fp1.write("\n")
    fp1.write("\n")
    line = 'drop\n'
    fp1.write(line)
    

    counter = 0
    for i in range(len(exps)):
        try:
            fp1.write(exps[i].split("-")[0] + ',')
            fp1.write(str(total_load_avg_exp[counter]) + ',')
            line1 = str(urllc_drop_avg_exp[counter]) + ','+ str(urllc_drop_avg_exp[counter + 1]) + ',' + str(embb_drop_avg_exp[counter]) + ',' + str(embb_drop_avg_exp[counter + 1]) + ',' + str(video_drop_avg_exp[counter]) + ',' + str(video_drop_avg_exp[counter+ 1]) + ',' + str(ip_drop_avg_exp[counter]) + ',' + str(ip_drop_avg_exp[counter + 1])
            fp1.write(line1)
            fp1.write("\n")
            counter = counter + 2
        except:
            counter = counter + 2
            fp1.write("\n")
            continue

    fp1.write("\n")
    fp1.write("\n")
    line = 'load\n'
    fp1.write(line)
    
    counter = 0
    for i in range(len(exps)):
        try:
            fp1.write(exps[i].split("-")[0] + ',')
            fp1.write(str(total_load_avg_exp[counter]) + ',')
            line1 = str(urllc_load_avg_exp[counter]) + ','+ str(urllc_load_avg_exp[counter + 1]) + ',' + str(embb_load_avg_exp[counter]) + ',' + str(embb_load_avg_exp[counter + 1]) + ',' + str(video_load_avg_exp[counter]) + ',' + str(video_load_avg_exp[counter+ 1]) + ',' + str(ip_load_avg_exp[counter]) + ',' + str(ip_load_avg_exp[counter + 1])
            fp1.write(line1)
            fp1.write("\n")
            counter = counter + 2
        except:
            counter = counter + 2
            fp1.write("\n")
            continue
    
    
    IO.close_for_writing(fp1)


                
# for type in delayDict:
#     for experimnet in type:
#         for trial in experimnet:
#             if type == "rl_predict":
#                 load_avg.append(np.mean(trial))
#                 load_std_rl.append(np.std(trial))
#             else:
#                 load_avg.append(np.mean(trial))
#                 load_std.append(np.std(trial))

# file_name = 'summary.csv'
# node_file = os.path.join(root, file_name)
# fp1 = IO.open_for_writing(node_file)
# for type in delayDict:
#     for experimnet in type:
#         for trail in experimnet:
#             fp1.write("{:f},{:f},".format(np.mean(trail), np.std(trail)))
#     fp1.write("\n")
# IO.close_for_writing(fp1)



