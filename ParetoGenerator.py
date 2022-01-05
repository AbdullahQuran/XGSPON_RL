from __future__ import division
from operator import itemgetter
import sys, pdb, csv
# import matplotlib.pyplot as plt
import simpy
import random
import math
import numpy as np

"""
Author: Sourav Maji
Software simulates a Poisson Pareto Burst Process
1. A Poisson process is defined where the inter-arrival times of events (i.e, bursts)
are generated from an exponential distribution, say mean burst arrivals are m_burstArrivals /sec
then inter_burst_arrivals = 1.0/m_burstArrivals
inter_arrival_x = exp(inter_burst_arrivals)
#param1: m_burstArrivals
2. The burst lengths are generated from a pareto distribution
A pareto distribution parameters are: m_shape, m_scale
for 1 < m_shape < 2, the distribution has infinite variance, and 
the PPBP demonstrates a long range dependence (LRD) process, which is characterized
by the Hurst parameter m_h. 
If 0 < m_h < 0.5, it is a negatively correlated process
If m_h = 0.5, the subsequent events are not correlated to present or prior events (no-dependency)
If 0.5 < m_h < 1, it is a positively correlated LRD process
m_shape = 3 -2*m_h
Pareto mean = m_burstLength
and 
mean = m_shape * m_scale/(m_shape -1)
or m_scale = (m_shape -1) * mean / m_shape
or m_scale = (m_shape -1) * m_burstLength / m_shape
m_shape = f(m_h)
m_scale = f(m_burstLength, m_shape)
#param2: m_h
#param3: m_burstLength
3. Each individual burst is generated at a constant bit rate (CBR) i.e., r (e.g. 1 Mbps)
#param4: r
4. The end time of the last event generation of this process, which is also the duration, assuming that it starts at 0 sec
#param5: time_end
"""

"""
Implementation:
bursts: (bs1, be1), (bs2, be2), (bs3, be3) ... (bsn, ben) 
       of format (burst1_start_time, burst1_end_time)
            bs2 - bs1, bs3 - bs2 ... from an exponential distribution
            be1 - bs1, be2 - bs2, ... from a pareto distribution
agg_bursts: Sum of the bursts that are in burst_list, depending at times these bursts
                overlap, the rate in that overlapping time is going to be a multiple of r.
       of format (burst_agg_s1, burst_agg_e1, rate)
Finally play out the agg_burst_list
"""



bursts = []
sort_bursts = []
agg_bursts = []
agg_bg = []
T = []
rate = []
pktsLi = []
MTU = 1500
burstiness = []

def PPBPgen(m_burstArrivals, m_h, m_burstLength, r, time_end):
  ts = 0
  inter_burst_arrivals = 1.0/m_burstArrivals
  m_shape = (3 - 2*m_h)
  m_scale = ((m_shape - 1) * m_burstLength)/m_shape
  print ("m_scale: " + str(m_scale) + ",m_shape: " + str(m_shape))
  pps = r/(MTU*8)
  delta = MTU*8/r
  if(m_scale < delta):
    print ("Cannot proceed as burst length is too low than link rate packet length ...")
    return;

  while True:
    tLen = np.random.pareto(m_shape) + m_scale
    te = ts + tLen
    bursts.append([ts, "s"])
    bursts.append([te, "e"])
    tInt = np.random.exponential(inter_burst_arrivals)
    ts = ts + tInt
    if(ts > time_end):
      break;
    #op1 = str(tLen) + "," + str(tInt) + "\n"
    #opFileDisPar.write(op1)

  #Now form the aggregate burst list
  #1. Sort the list based on the timestamps
  global sort_bursts
  sort_bursts = sorted(bursts, key = itemgetter(0))
  
  curPkts = 0
  m=0
  for i in range(len(sort_bursts)-1):
    if(sort_bursts[i][1] == "s"):
      m = m + 1
    curPkts = m*pps*(sort_bursts[i+1][0] - sort_bursts[i][0])
    if(sort_bursts[i][0] < time_end):
      #agg_bursts.append([sort_bursts[i][0], sort_bursts[i][1], sort_bursts[i+1][0], sort_bursts[i+1][1], m, int(math.ceil(curPkts))])
      agg_bursts.append([sort_bursts[i][0], sort_bursts[i][1], sort_bursts[i+1][0], sort_bursts[i+1][1], m, curPkts])
    if(sort_bursts[i+1][1] == "e"):
      m = m - 1

  for j in range(len(agg_bursts)):
    T.append(agg_bursts[j][2])
    pkts = agg_bursts[j][5]
    pktsLi.append(pkts)
    wtTm = 0.0
    if (pkts > 0):
      wtTm = (agg_bursts[j][2] - agg_bursts[j][0])/pkts
    agg_bg.append([agg_bursts[j][0], agg_bursts[j][2], pkts, wtTm])

np.random.seed(42)
PPBPgen(5, 0.6, 0.02, 10**7, 10)
 
m_burstArrivals = int(sys.argv[1])
m_h = float(sys.argv[2])
m_burstLength = float(sys.argv[3])
r = float(sys.argv[4])
time_end = int(sys.argv[5])
time_tr = int(sys.argv[6])
#opFile = open(str(sys.argv[7]), 'w')
#opFileDisPar = open(str(sys.argv[8]), 'w')
#opFileDisPar.write("burstLen,burstIntArr\n")

print ("Usage:python PPBP.py m_burstArrivals, m_h, m_burstLength, r, time_end")
print ("Example: python PPBP.py 5 0.6 0.02 10000000 10")

#set this value whenever you need the same set of random numbers

# PPBPgen(m_burstArrivals, m_h, m_burstLength, r, time_end)

print ("Bursts:")
print (bursts)

print ("Sorted bursts:")
print (sort_bursts)
 
print ("Agg. bursts:")
print (agg_bursts)

print ("Agg. background:")
print (agg_bg)

print ("PktsLi: ")
print (pktsLi)

print ("Mean PktsLi: ")
print (np.mean(pktsLi))


delTm = 0.1
tObs = T[0] + delTm
burstWin = 0
for i in range(len(T)):
  if(T[i] < tObs):
    burstWin = burstWin + pktsLi[i]*MTU
  else:
    burstiness.append(burstWin)
    burstWin = 0
    tObs = tObs + delTm

b = np.std(burstiness)/(np.mean(burstiness))

print ("Burstiness: " + str(b))

print ("Plotting the aggregate bursts:...")
data = rt.pltTimeData(T, pktsLi, 0.1)
X = [row[0] for row in data]
Y = [row[1] for row in data]

X_trunc = []
Y_trunc = []

#opFile.write("Tm,Rt\n");
for i in range(len(X)):
  if(X[i] >= time_tr): 
    #op = str(X[i]) + "," + str(Y[i]) + "\n"
    #opFile.write(op)
    X_trunc.append(X[i])
    Y_trunc.append(Y[i])

print ("Mean traffic rate: " + str(np.mean(Y_trunc)))
print ("Another mean rate: " + str(sum(pktsLi)*MTU*8/(time_end - time_tr)))

"""
plt.plot(X_trunc, Y_trunc)
plt.xlabel("time (s)")
plt.ylabel("traffic rate (Mbps)")
plt.grid(True)
plt.savefig("bgRateNotFromRun.png")
opFile.close()
opFileDisPar.close()
plt.show()
plt.plot(T, rate)
plt.xlabel("time (s)")
plt.ylabel("traffic rate (Mbps)")
plt.grid(True)
plt.savefig("bgRateNotFromRun.png")
"""