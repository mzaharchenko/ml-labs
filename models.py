import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
plt.style.use(['seaborn-whitegrid'])
plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['font.family'] = 'DejaVu Sans'

class PatternClassifier:
    def __init__(self,s,pw):
        self.s = s
        self.pw = pw
        self.num_features = s.shape[0]
        self.num_classes = s.shape[1]
        self.ns = sum(abs(s[:,0]-s[:,1]))
    
    def classify(self,steps = 20,K = 1000):
        pi,p12th,p21th,p12ex,p21ex = np.zeros(steps),np.zeros(steps),np.zeros(steps),\
                                     np.zeros(steps),np.zeros(steps)
        Pc_ = np.zeros([2,2,steps])
        s_ = 1-self.s
        plots = []
        for i in range(steps):
            pi[i] = (1/steps)*i
            pI = pi[i]

            if pI == 0: 
                pI=0.0001
            if pI == 0.5: 
                pI=0.4999

            pI_ = 1-pI
            G1 = np.zeros(self.num_features)
            G2 = np.zeros(self.num_features) 

            for a in range(self.num_features):
                G1[a]=np.log((self.s[a,0]*pI_+s_[a,0]*pI)/(self.s[a,1]*pI_+s_[a,1]*pI))
                G2[a]=np.log((self.s[a,0]*pI+s_[a,0]*pI_)/(self.s[a,1]*pI+s_[a,1]*pI_))
            l0_ = np.log(self.pw[1]/self.pw[0])
            L0 = l0_/(2*np.log(pI_)-2*np.log(pI)) + self.ns/2
            L0r = np.floor(L0)
            if pI<0.5:
                p12th[i] = binom.cdf(L0r,self.ns,pI_)
                p21th[i] = 1-binom.cdf(L0r,self.ns,pI)
            else:
                p12th[i] = 1-binom.cdf(L0r,self.ns,pI_)
                p21th[i] = binom.cdf(L0r,self.ns,pI)

            for k in range(K):
                for j in range(self.num_classes):
                    x = self.s[:,j].copy()
                    r = np.random.rand(self.num_features)
                    ir = np.where(r<pI)[0]
                    x[ir] = 1 - x[ir]
                    x_ = 1 - x
                    u = np.dot(G1,x) + np.dot(G2,x_) - l0_
                    if u > 0:
                        iai = 0
                    else:
                        iai = 1

                    Pc_[iai,j,i] = Pc_[iai,j,i]+1

                    if (k == 0) and (i == 1):
                        IAx = np.reshape(x_,(7,5))
                        plots.append(IAx)
            Pc_/= K 
            p12ex[i] = Pc_[1,0,i] 
            p21ex[i] = Pc_[0,1,i]
        self.p12ex = p12ex
        self.p12th = p12th
        self.p21ex = p21ex
        self.p21th = p21th
        self.pi = pi
        self.plots = plots
        
    def plot(self,num,ax):
        plt.subplot(num)
        ms= 1
        axes = ax
        axes.set_xlim(min(self.pi), max(self.pi))
        axes.set_ylim(-0.01, ms)
        plt.plot(self.pi,self.p12th,'xkcd:lightblue',label = 'p12th',linewidth=2.0)
        plt.plot(self.pi,self.p21th,'xkcd:pink',label = 'p21th',linewidth=2.0)
        plt.plot(self.pi,self.p12ex,'xkcd:blue',linestyle = '--',label = 'p12ex',linewidth=2.0)
        plt.plot(self.pi,self.p21ex,'xkcd:plum',linestyle = '--',label = 'p21ex',linewidth=2.0)
        plt.title('Теоретические вероятности ошибок и их оценки')
        plt.xlabel('pi')
        plt.ylabel('P')
        ann = 'pw = {}'.format(self.pw)
        plt.annotate(ann,(0.1,0.75*ms)) 
        plt.legend();