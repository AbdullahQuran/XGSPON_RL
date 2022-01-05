# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import threading
import time

class PPBP(threading.Thread):

   def __init__(self, T, lam=10.0, H=0.8, meanD=4, step = 0.01):
      threading.Thread.__init__(self)
      self.alpha=1.7
      self.beta=2
      self.a=1.0/self.alpha
      self.tm = None
      self.B = None
      self.T = T
      self.lam = lam
      self.H = H
      self.meanD = meanD
      self.step = step
      # self.x=np.linspace(0.02, self.beta,50)
      # self.y=1-(self.alpha-1)*self.a*(1.0-self.x/self.beta)-self.a
      # self.xx=np.linspace(self.beta,8, 100)
      # self.yy=1-self.a*np.power(self.xx/self.beta, 1.0-self.alpha)
      # # plt.plot(x,y, 'g')
      # # plt.plot(xx, yy, 'r')

   def rvsG(self, alpha, beta, n):
      vals=[]
      a=1-1.0/alpha
      for i in range(n):
         u=np.random.random()
         if (u<a):
            vals.append( beta*(alpha*a-1.0+u)/(a*alpha-1))
         else:
            vals.append(beta*np.power(alpha*(1-u), 1.0/(1-alpha)))
      return vals 

   # vals=rvsG(alpha,beta, 500)
   # histo=plt.hist(vals, bins=50)
   # # plt.show()

   def timesBurstInterval(self,ts, tf, h):
      j=(int)(ts/h)+1 if np.fmod(ts, h) else (int)(ts/h)
      k=(int)(tf/h) if np.fmod(tf, h) else (int)(tf/h)-1
      return j,k

   def internetTraffic(self):#lam, meanD, H, 
      alpha=3.0-2*self.H
      beta=(alpha-1)*self.meanD/alpha
      h=0.8
      # print (beta, h)
      tm=np.arange(0, self.T, self.step)
      B=np.zeros(tm.shape[0], dtype=float)
      theta=1.0/self.lam
      S=st.expon(scale=theta)
      D=st.pareto(alpha)
      
      n=st.poisson.rvs(self.lam*self.meanD)
      initD= self.rvsG(alpha, beta, n)
      for i in range(n):
         j,k=self.timesBurstInterval(0, min(initD[i],self.T), self.step)
         B[j:k+1]+=1.0   
      ts=0
      tf=0
      while(ts<self.T):
         x=S.rvs(size=1)# length of interarrivals intervals
         ts=ts+x# burst start time
         if ts>=self.T:
            break
         d=beta*D.rvs(size=1)     
         tf=ts+d# burst stop time
         j,k=self.timesBurstInterval(ts, min(tf,self.T),self.step) 
         B[j:k+1]+=1.0 #update the number of active bursts in [ts, tf]
      return tm, B

   #T, lam=10.0, H=0.8, meanD=4
   def run(self):
      tm, B = self.internetTraffic()
      self.tm = tm
      self.B = B
   


   
   
   # tm, B = internetTraffic(time)
   # plt.rcParams['figure.figsize'] = (15.0, 6.0)
   # plt.title('Internet traffic as a Poisson Pareto Burst Process. Hurst exponent 0.75')
   # plt.xlabel('time t')
   # plt.ylabel('Number of active bursts')
   # plt.plot (tm, B)
   # plt.show()

def createPPBPTrafficGen(T, lam=10.0, H=0.8, meanD=4, size=1, step = 0.01):
   threads = []
   tm = []
   B = []
   sum  = []
   for i in range(size):
      threads.append(PPBP(T, lam, H, meanD, step))
   
   for i in range(size):
      threads[i].start()
   
   for t in threads:
      t.join()
   
   for t in threads:
      tm.append(t.tm)
      B.append(t.B)

   for i in range(len(B[0])):
      w = 0
      for j in range(size):
         w = w + B[j][i]      
      sum.append(w)
   return sum, tm[0]


# createPPBPTrafficGen(100, lam=10.0, H=0.8, meanD=4, size=50, step = 0.01)
# array = []
# tm = []
# B = []
# sum  = []
# size = 1

# for i in range(size):
#    array.append(PPBP(step))

# for i in range(size):
#    x,y = array[i].run(5000)
#    tm.append(x)
#    B.append(y)

# for i in range(len(B[0])):
#    w = 0
#    for j in range(size):
#       w = w + B[j][i]      
#    sum.append(w)

# plt.rcParams['figure.figsize'] = (25.0, 6.0)
# plt.title('Internet traffic as a Poisson Pareto Burst Process. Hurst exponent 0.8')
# plt.xlabel('time t')
# plt.ylabel('Number of active bursts')
# plt.plot (tm[0], sum)
# plt.show()
# print (tm, B)