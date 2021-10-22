#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:05:58 2020

@author: xaviersiebert
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin,cos  #, floor
from sklearn.model_selection import train_test_split
from sys import exit
from scipy.stats import multivariate_normal
from sklearn import datasets # mnist et al.
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm

_xymin=-1
_xymax=1

# def flip(p):
#     return 0 if np.random.random() >= p else 1 # watch out sign!!

#XS TODO : remove this, make it class method
def eta2(x, y, dt='line'):
    if (dt=='line'):
        return 0.5*(1+sin(0.5*pi*y))
    elif (dt=='squares'):
        return 0.5*(1+sin(0.5*pi*x)*sin(0.5*pi*y))
    elif(dt=='triangles'):
        return 0.5*(1+sin(0.5*pi*(y-sin(x))))
    else:
        print ('eta2 : unknown data type, aborting')
        exit()

# def g1(x):
#     return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)  

def bayes(p):
     return 0 if p < 0.5 else 1

class dataSynth():
    def __init__(self, n = 1000, d=2,dt='line',nb_classes=2):
        self.n = n # number of points generated in total
        self.d = d # dimension of the data
        #np.random.seed(19760105) # to reproduce results
#        self.xs = [np.random.uniform(-1, 1, d) for _ in range (n)]
        self.X = None
        self.Xtest = None
        self.Xtrain = None
        self.XcoldStart = None
        self.Xunlabeled = None

        self.Y=None
        self.Ytest = None
        self.Ytrain = None
        self.YcoldStart = None
        self.Yunlabeled = None
        
        self.dataType=dt
        #self.bayess=np.zeros(n) # pour comparer
        self.n_coldStart=0
        self.n_unlabeled=0
        self.n_train=0
        self.n_test=0
        
        self.nb_classes=nb_classes

    def getSize(self):
        return self.n

    def get_n_coldStart(self):
        return self.n_coldStart
    
    def flip(self,p):
        #rng = np.random.RandomState(2021)
        try:
            a=np.random.binomial(1,p)
        except ValueError:
            print('<<dataSynth::flip>> eta should be in [0,1], got this :')
            print(p)
            exit()
        return a
 #        return 0 if np.random.random() >= p else 1 # watch out sign!! 

    def makeX(self):
        if (self.dataType=='squares_non_uniform'):
            self.X = np.random.uniform(low=_xymin, high=_xymax, size=(self.n,self.d)) # ré-écrire la composante y
            n1=int(self.n *0.99)
            n2=self.n-n1
            a=np.random.uniform(-1.0, 0.0, n1)
            b=np.random.uniform(0.0, 1.0, n2)
            for i in range(n1):
                self.X[i][0]=a[i]
            for i in range(n2):
                self.X[n1+i][0]=b[i]
        elif (self.dataType[0:8]=='dasgupta'):
            self.makeX_Dasgupta()
        elif (self.dataType[0:8]=='gaussian'):
            self.makeX_Gaussian() # XS hack : fabrique le Y aussi 
        elif (self.dataType[0:5]=='mnist'):
            self.makeX_Mnist() # XS hack : fabrique le Y aussi 
        else:
            self.X = np.random.uniform(low=_xymin, high=_xymax, size=(self.n,self.d))
    
    def makeY(self):
        self.Y=self.flip(self.getEta()) 
    
    def makeData(self):
        self.makeX()
        if (self.dataType[0:8]!='gaussian' and self.dataType[0:5]!='mnist'):
            self.makeY()

        #     #XS TODO : plot facultatif    
        #     #self.plotEta_y()

    def getType(self):
        return self.dataType
    
    def getEta(self,XX=None):
        if XX is None:
            XX=self.X        
        nn=len(XX)
        if (self.dataType=='line'):
            y = 0.5*(1+np.sin(0.5*pi*XX[:,1]))
        elif (self.dataType=='squares'):
            y = 0.5*(1+np.sin(0.5*pi*XX[:,0])*np.sin(0.5*pi*XX[:,1]))
        elif (self.dataType=='squares_non_uniform'):
            y = 0.5*(1+np.sin(0.5*pi*XX[:,0])*np.sin(0.5*pi*XX[:,1]))
        elif (self.dataType=='triangles'):
            y = 0.5*(1+np.abs(XX[:,0])-np.abs(XX[:,1])) # 0.5*(1+sin(0.5*pi*(x2[i]-sin(0.5*pi*(x1[i])-0.5*pi)))))             
        elif (self.dataType=='dasgupta'):
            y=np.zeros((nn), float)
            for i in range(nn):
                x=XX[i,0]
                if x <= -0.7:
                    y[i] = 1
                elif x<= -0.3:
                    y[i] = 0
                elif x<= 0:
                    y[i] = 1
                else:
                    y[i] = 1
        elif (self.dataType=='dasgupta_noise'):
              # XS TODO : transformer dasgupta en fonction
            y=np.zeros((nn), float)
            for i in range(nn):
                x=XX[i,0]
                if x <= -0.4:
                    y[i] = -2.0/3.0 - 5.0/3.0 * x
                elif x<= 0.4:
                    y[i] = 0.5 + 1.25 * x
                else:
                    y[i] = 5.0/3.0 - 5.0/3.0 * x
        else:
            print('eta not available for your dataset %s'%self.dataType)            
        return y

    def makeX_Mnist(self):
        digits = datasets.load_digits()
        self.X = digits.data
        self.n = digits.data.shape[0] # number of points generated in total
        self.d = digits.data.shape[1] # dimension of the data
        self.Y = digits.target

    def makeX_Gaussian(self):
        if self.nb_classes==3:
            ccc = [(-0.5, 0.5), (0, 0.5), (0.5, -0.5)]
        elif self.nb_classes==5:
            ccc = [(0.5, 0), (-0.5, 0), (0, 0.5), (0.25, -0.5), (-0.25, -0.5)]
        elif self.nb_classes==10:
            ccc=[]
            for a in range(10):
                theta=2*a*pi/10
                ccc.append((cos(theta),sin(theta)))
        else:    
            ccc=self.nb_classes # random choice
        X, self.Y= make_blobs(n_samples=self.n, n_features=self.d, cluster_std=0.1, centers=ccc,
                                   center_box=(_xymin,_xymax), shuffle=True, random_state=None) 
        max_abs_scaler = MinMaxScaler(feature_range=[_xymin,_xymax])
        self.X = max_abs_scaler.fit_transform(X)

    def makeX_Gaussian_Boris(self):
        # Parameters of the mixture components
        mu1=[0,-1]
        sig1=[[0.5,0],[0,0.5]]
        mu2=[np.sqrt(3)/2,0]
        sig2=[[0.5,0],[0,0.5]]
        mu3=[-np.sqrt(3)/2,1]
        sig3=[[0.5,0],[0,0.5]]
        norm_params1 = np.array([mu1,mu2,mu3])
        norm_params2 = np.array([sig1,sig2,sig3])
        n_components = norm_params1.shape[0]
        var1 = multivariate_normal(mean=mu1, cov=sig1)
        var2 = multivariate_normal(mean=mu2, cov=sig2)
        var3 = multivariate_normal(mean=mu3, cov=sig3)
        # Weight of each component
        weights = np.ones(n_components, dtype=np.float64) / float(n_components)     
        # A stream of indices from which to choose the component
        mixture_idx = np.random.choice(n_components, size=self.n, replace=True, p=weights)
        # y is the mixture sample
        self.X=np.random.multivariate_normal(norm_params1[0],norm_params2[0])
        self.X=np.reshape(self.X,(1,self.d))
        vector_label=[weights[0]*var1.pdf(self.X),weights[1]*var2.pdf(self.X),weights[2]*var3.pdf(self.X)]
        self.Y=np.argmax(vector_label) 
                     
        for i in mixture_idx[1:]:
            z=np.random.multivariate_normal(norm_params1[i],norm_params2[i])/4
            vector_label=[weights[0]*var1.pdf(z),weights[1]*var2.pdf(z),weights[2]*var3.pdf(z)]
            self.Y=np.append(self.Y,np.argmax(vector_label))
            self.X = np.concatenate([self.X,np.reshape(z,(1,self.d))])

        print(self.X)
        print(self.Y)
        
    def makeX_Dasgupta(self):
        #this is the "continuous" version with some uniform background
        n0=int(0.005 * self.n)
        n=self.n-n0
        n1=int(0.45 * n)
        n2=int(0.05 * n)
        n3=int(0.025 * n)
        n4=n3
        n5=n1
        self.X = np.random.uniform(low=_xymin, high=_xymax, size=(self.n,self.d)) # ré-écrire la composante y
        
        # uniform background -- XS TODO : inutile, vu la ligne précédente...
        o=np.random.uniform(-1.0, 1.0, n0)
        for i in range(n0):
            self.X[i][0]=o[i]
        
        a=np.random.uniform(-1.0, -0.8, n1)
        for i in range(n1):
            self.X[i+n0][0]=a[i]
        
        b=np.random.uniform(-0.6, -0.4, n2)
        for i in range(n2):
            self.X[i+n0+n1][0]=b[i]
        
        c=np.random.uniform(-0.2, 0, n3)
        for i in range(n3):
            self.X[i+n0+n1+n2][0]=c[i]
   
        d=np.random.uniform(0, 0.2, n4)
        for i in range(n4):
            self.X[i+n0+n1+n2+n3][0]=d[i]

        e=np.random.uniform(0.8, 1.0, n5)
        for i in range(n5):
            self.X[i+n0+n1+n2+n3+n4][0]=e[i]

        self.shuffleX()
         
    # def approxDasgupta(self):
    #     #XS TODO: this is the discontinuous version
    #     n1=int(0.45 * self.n)
    #     n2=int(0.05 * self.n)
    #     n3=int(0.025 * self.n)
    #     n4=n3
    #     n5=n1
    #     self.X = np.random.uniform(low=_xymin, high=_xymax, size=(self.n,self.d))
        
    #     a=np.random.uniform(-1.0, -0.8, n1)
    #     for i in range(n1):
    #         self.X[i][0]=a[i]
    #         self.Y[i]=0 # change using eta
        
    #     b=np.random.uniform(-0.6, -0.4, n2)
    #     for i in range(n2):
    #         self.X[i+n1][0]=b[i]
    #         self.Y[i+n1]=1 # change using eta
        
    #     c=np.random.uniform(-0.2, 0, n3)
    #     for i in range(n3):
    #         self.X[i+n1+n2][0]=c[i]
    #         self.Y[i+n1+n2]=0 # change using eta
   
    #     d=np.random.uniform(0, 0.2, n4)
    #     for i in range(n4):
    #         self.X[i+n1+n2+n3][0]=d[i]
    #         self.Y[i+n1+n2+n3]=1 # change using eta

    #     e=np.random.uniform(0.8, 1.0, n5)
    #     for i in range(n5):
    #         self.X[i+n1+n2+n3+n4][0]=e[i]
    #         self.Y[i+n1+n2+n3+n4]=1 # change using eta

    #     self.shuffle()

    def shuffleX(self):
        l = [i for i in range(self.n)]
        np.random.shuffle(l)
        Xtrain=np.zeros((self.n,2), float)
        for i in range (self.n):
            Xtrain[i,0]=self.X[l[i]][0]
            Xtrain[i,1]=self.X[l[i]][1]
        self.X=Xtrain

    def shuffle(self):
        l = [i for i in range(self.n)]
        np.random.shuffle(l)
        Xtrain=np.zeros((self.n,2), float)
        Ytrain=np.zeros((self.n), int)
        for i in range (self.n):
            Xtrain[i,0]=self.X[l[i]][0]
            Xtrain[i,1]=self.X[l[i]][1]
            Ytrain[i]=self.Y[l[i]]
        self.X=Xtrain
        self.Y=Ytrain
        
    def getTestData(self):
        return self.Xtest,self.Ytest
    
    def getTrainData(self):
        return self.Xtrain,self.Ytrain
    
    def split(self,n_train,n_test,n_unlabeled,n_coldStart):
        # XS TODO : rajouter tests pour voir si on peut séparer de la sorte
        if (self.n==0):
            print('<dataSynth::split> : no data')
            exit()
        if (n_train+n_test+n_unlabeled+n_coldStart != self.n):
            print('<dataSynth::split> : wrong data count : n_train+n_test+n_unlabeled+n_coldStart=%d != self.n=%d'
                  %(n_train+n_test+n_unlabeled+n_coldStart,self.n))
            exit()
        self.n_coldStart=n_coldStart
        self.n_unlabeled=n_unlabeled
        self.n_train=n_train
        self.n_test=n_test
        
        # répartition aléatoire, sans tenir compte de l'éventuel 
        # déséquilibre entre classes...
        #np.random.seed(19760105)
        l = [i for i in range(self.n)]
        np.random.shuffle(l)
        
        n1=self.n_coldStart
        n2=n1+self.n_unlabeled
        n3=n2+self.n_train
        n4=n3+self.n_test
        self.coldStart_idx = l[0:n1]
        self.unlabeled_idx = l[n1:n2]
        self.train_idx = l[n2:n3]
        self.test_idx = l[n3:n4]
        self.XcoldStart=self.X[self.coldStart_idx]
        self.Xunlabeled=self.X[self.unlabeled_idx]
        self.Xtrain=self.X[self.train_idx]
        self.Xtest=self.X[self.test_idx]
        self.YcoldStart=self.Y[self.coldStart_idx]
        self.Yunlabeled=self.Y[self.unlabeled_idx]
        self.Ytrain=self.Y[self.train_idx]
        self.Ytest=self.Y[self.test_idx]
    
    def splitStratified(self,n_train,n_test,n_unlabeled,n_coldStart):
        if (n_train+n_test+n_unlabeled+n_coldStart != self.n):
            print('<splitStratified> : wrong data count')
            exit()
        if (self.n==0):
            print('<splitStratified> : no data (n=0)')
            exit()
            
        p_test=n_test*1.0/self.n #300/1000
        p_coldStart=n_coldStart*1.0/(self.n-n_test) # 50/700
        p_unlabeled=n_unlabeled*1.0/(self.n-n_test-n_coldStart)# 100/650
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(self.X, self.Y, test_size=p_test)#, stratify=self.Y)
        self.Xtrain, self.XcoldStart, self.Ytrain, self.YcoldStart = train_test_split(self.Xtrain, self.Ytrain, test_size=p_coldStart)#, stratify=self.Ytrain)
        self.Xtrain, self.Xunlabeled, self.Ytrain, self.Yunlabeled = train_test_split(self.Xtrain, self.Ytrain, test_size=p_unlabeled)#, stratify=self.Ytrain)

        self.n_coldStart=n_coldStart
        self.n_unlabeled=n_unlabeled
        self.n_train=n_train
        self.n_test=n_test

    def plot(self, showme=True):       
        # XS TODO : this is for binary only
        for i in range(self.n): 
            if self.Y[i]==1:
                color=".g"
            else:
                color=".b"
            plt.plot(self.X[i][0],self.X[i][1],color)
        plt.ylabel('x_2')
        plt.xlabel('x_1')
        plt.title('blue : y=0; green : y=1')
        if showme:
            plt.show()
        #plt.pause(3)
        #plt.close()
        else: 
            return plt
 
    def scatterPlot(self,showme=True):
        #if (self.dataType[0:8]=='gaussian'):
        plt.scatter(self.X[:,0],self.X[:,1], marker='.',c=self.Y,cmap='tab10')
        #else:
            # XS TODO : limité à 3 classes max...
            #colormap = np.array(['b', 'g','r'])
            #plt.scatter(self.X[:, 0],self.X[:,1], marker='.',c=colormap[self.Y])
        plt.ylabel('x_2')
        plt.xlabel('x_1')
        if (self.dataType[0:8]!='gaussian'):
            plt.title('blue : y=0; cyan : y=1')
        if showme:
            plt.show()
        #plt.pause(3)
        #plt.close()
        else: 
            return plt
    
 
    def plotEta_y(self):      
        # XS TODO : this is a hack, too specific !
        n=100
        x = np.zeros(n)
        y = np.linspace(-1,1,n)
        z = np.zeros(n)
        for i in range(n):
            z[i] = eta2(x[i],y[i])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(y,z, 'r')
        plt.show()

    # def plotBayes(self):            
    #     for i in range(self.n):
    #         if self.bayess[i]==1:
    #             color=".g"
    #         else:
    #             color=".b"
    #         plt.plot(self.x1s[i],self.x2s[i],color)
    #     plt.ylabel('y')
    #     plt.xlabel('x')
    #     plt.show()
    #     #plt.pause(3)
    #     #plt.close()

    def contourPlot(self):            
        x = np.arange(-1, 1, 2.0/self.n)
        y = np.arange(-1, 1, 2.0/self.n)
        z = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                z[j][i]=eta2(x[i],y[j])
#                print (x[i],y[j],z[i][j])
        plt.contourf(x, y, z, 10,origin='lower') # 10 niveaux
        plt.colorbar()
        for i in range(self.n): 
            if self.Y[i]==1:
                color=".g"
            else:
                color=".b"
            plt.plot(self.x1s[i],self.x2s[i],color)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()
