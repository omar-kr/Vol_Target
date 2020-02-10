# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:48:30 2020

@author: karkaro
"""

#-----------------
#  Packages 
#-----------------

import numpy as np
import matplotlib.pyplot as plt



#-----------------
#  Normal Vol excess return  
#-----------------


class normalVol:
    
    #----------------
    # Constructor
    #----------------
    def __init__(self,mu,vol,r,K,seed,npaths,ndays,w):
        self.mu      = mu/260         # daily mean of the excess return 
        self.vol     = vol/260        # daily vol of the excess return 
        self.r       = r/260          # constant risk free rate | the mean excess return of the VT portfolio will not depend on r0
        self.K       = K/260          # Target vol
        self.seed    = seed           # numpy random seed used when generating the trajectories of excess return 
        self.npaths  = npaths         # number of paths in the simulation , e.g: 5000
        self.ndays   = ndays          # number of days considered in the experiment, typically: 5200 (20 years)
        self.w       = w              # time window used to estimate the volatility of the Index
        self.excessReturnBH  = self.excessReturnPaths() # dataset of the excess returns  
        self.returnIndex     = self.r + self.excessReturnBH  # dataset of the Index returns 
        self.volIndex        = self.volIndexPaths()     # vol of the Index based on window= w
        self.wIndex          = self.K/self.volIndex     # weight of the Index
        self.returnPortfolio = (self.wIndex*self.returnIndex[:,w:]) + self.r*(1-self.wIndex)
        self.excessReturnVT  = np.mean(self.returnPortfolio,axis=1).reshape((self.npaths,1))-self.r
        self.volPortfolio    = np.std(self.returnPortfolio,axis=1).reshape((self.npaths,1))
        self.volHB           = np.std(self.excessReturnBH,axis=1).reshape((self.npaths,1))
     
    
        
    #----------------
    # Methods
    #----------------
    
    # This function computes the excess return ri-r0 =  N(mu,vol)
    def excessReturnPaths(self):
            np.random.seed(self.seed)
            return np.random.normal(self.mu,self.vol,self.npaths*self.ndays).reshape((self.npaths,self.ndays))
    
    # This function plot the evolution of the liquid index through time 
    def plotIndex(self,ipath,s=100):
        spot = list(s*np.cumprod(1+self.returnIndex[ipath,:].reshape((1,self.ndays)),axis=1)[0])
        plt.plot([s]+spot,label="Index")
        plt.ylabel  = "Index spot value"
        plt.xlabel  = "Days"
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    # This function return the empirical volatility of the index when using a time window w 
    def volIndexPaths(self):
        # The number of columns outputed is ndays - w
        volRisky = np.std(self.returnIndex[:,0:self.w],axis=1).reshape((self.npaths,1))
        for j in range(1,self.ndays-self.w):
            volRisky = np.c_[volRisky,np.std(self.returnIndex[:,j:(j+self.w)],axis=1).reshape((self.npaths,1))]
        return volRisky
    
    # This function returns someuseful statistics for our strategies 
    def summary(self,output=False):
        d = {}
        d["AAER_VT"]   = 260 * np.mean(self.excessReturnVT)
        d["AAER_BH"]   = 260 * np.mean(self.excessReturnBH)
        d["AAV_VT"]    = 260 * np.mean(self.volPortfolio)
        d["AAV_BH"]    = 260 * np.mean(self.volHB)
        d["Sharpe_VT"] = np.mean(self.excessReturnVT/self.volPortfolio)
        d["Sharpe_BH"] = np.mean(self.excessReturnVT/self.volHB)
        d["wmean_VT"]  = np.mean(self.wIndex)
        d["wmean_BH"]  = 1
        if output:
            return d
        else:
            print("-----Hold Buy-----")
            print("Average excess return: " + str(round(d["AAER_BH"]*100,2)) +"%")
            print("Average volatility: "+str(round(d["AAV_BH"] *100,2))+"%")
            print("Sharpe ratio: "+str(round(d["Sharpe_BH"],2)))
            print("Average exposure: "+str(round(d["wmean_BH"]*100,2))+"%"+"\n")
            print("-----Vol Target-----")
            print("Average excess return: " + str(round(d["AAER_VT"]*100,2)) +"%")
            print("Average volatility: "+str(round(d["AAV_VT"] *100,2))+"%")
            print("Sharpe ratio: "+str(round(d["Sharpe_VT"],2)))
            print("Average exposure: "+str(round(d["wmean_VT"]*100,2))+"%")


#-----------------
#  Garch excess return  
#-----------------
            
            
class garch(normalVol):
    
    def __init__(self,wgarch,alpha,beta,mu,vol,r,K,seed,npaths,ndays,w):
        self.wgarch = wgarch
        self.alpha  = alpha
        self.beta   = beta
        normalVol.__init__(self,mu,vol,r,K,seed,npaths,ndays,w)

    def excessReturnPaths(self):
        cash = np.zeros((self.npaths,1))
        rtn  = np.zeros((self.npaths,1))
        vol  = np.zeros((self.npaths,1))
        for iday in range(self.ndays):
            vol  = np.sqrt(self.wgarch + self.alpha*np.power(rtn-self.mu,2) + self.beta*np.power(vol,2))
            z    = np.random.normal(0,1,self.npaths).reshape((self.npaths,1))
            rtn  = self.mu + vol*z
            cash = np.c_[cash,rtn]
        print("Excess return computed for Garch Target Vol")
        return cash[:,1:].reshape((self.npaths,self.ndays))

    def plotSimVol(self):
        cash = np.zeros((1,1))
        rtn  = np.zeros((1,1))
        vol  = np.zeros((1,1))
        for iday in range(self.ndays):
            vol  = np.sqrt(self.wgarch + self.alpha*np.power(rtn-self.mu,2) + self.beta*np.power(vol,2))
            z    = np.random.normal(0,1,1).reshape((1,1))
            rtn  = self.mu + vol*z
            cash = np.c_[cash,vol]
        plt.plot(cash.T,label="b="+str(self.beta) + "-a="+str(self.alpha))
        plt.legend(loc="best")
        plt.grid()
        plt.show()
        

        