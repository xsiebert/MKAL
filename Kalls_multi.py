#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:55:48 2021
KALLS MULTICLASS = MKAL
@author: xaviersiebert
"""
from math import pi, sqrt, log, exp, cos, sin, acos, asin
import dataSynth
from sys import exit
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pyplot as plt
#import time
    
#--------
def distance(Xi,Xj,d=2):
    dist=0
    for l in range(d):
        dist+= (Xi[l]-Xj[l])**2
    return sqrt(dist)    

def compute_p_hat_EZ(Xs,Xprime):
    d = distance(Xs,Xprime)
    cnt=0
    for X in myData.Xtrain:
        dd = distance(X,Xs)
        if (dd<=d):
            cnt+=1
    cnt=cnt*1.0/n_train
    return cnt
    
def compute_p_hat(Xs,Xprime):
    rho=distance(Xs,Xprime)
    rho2=rho**2
#    rho_2=rho/sqrt(2)
    if (rho==0):
        print ('compute_p_hat : zero distance !?')
        return 10 # cXS TODO : heck this
    aire_tot = 4 #(_xymax-_xymin)**2 to be more general
    aire_disk_4 = pi*rho2/4

    UL=[_xymin,_xymax]
    UR=[_xymax,_xymax]
    LL=[_xymin,_xymin]
    LR=[_xymax,_xymin]
    
    pUL=0
    pUR=0
    pLL=0
    pLR=0
    
    # UL
    #if (UL[0]>=Xs[0]-rho_2) and (UL[1]<=Xs[1]+rho_2):
    if distance(UL,Xs)<rho :
        pUL = (Xs[0]-UL[0])*(UL[1]-Xs[1])
    else:
        pUL=aire_disk_4
        if UL[1] < Xs[1]+rho :
            alpha = acos((UL[1]-Xs[1])/rho)
            tarte=(rho2*alpha/2)
            triangle=(UL[1]-Xs[1])*rho*sin(alpha)/2
            pUL = pUL - tarte + triangle
        if UL[0] > Xs[0]-rho:                       
            beta = acos((Xs[0]-UL[0])/rho)
            tarte=(rho2*beta/2)
            triangle=(Xs[0]-UL[0])*rho*sin(beta)/2
            pUL = pUL - tarte + triangle    
    if (pUL<=0):
        print ('weird : pUL = %8.3f'%pUL)            

    # UR
    #if (UR[0]<=Xs[0]+rho_2) and (UR[1]<=Xs[1]+rho_2):
    if distance(UR,Xs)<rho :
        pUR = (UR[0]-Xs[0])*(UR[1]-Xs[1])
    else:
        pUR=aire_disk_4
        if UR[1]<=Xs[1]+rho:
            alpha = acos((UR[1]-Xs[1])/rho)
            tarte=(rho2*alpha/2)
            triangle=(UR[1]-Xs[1])*rho*sin(alpha)/2
            pUR = pUR - tarte + triangle
        if UR[0]<=Xs[0]+rho:
            beta = acos((UR[0]-Xs[0])/rho)
            tarte=(rho2*beta/2)
            triangle=(UR[0]-Xs[0])*rho*sin(beta)/2
            pUR = pUR - tarte + triangle  
    if (pUR<=0):
        print ('weird : pUR = %8.3f'%pUR)            
       
    # LL
    #if (LL[0]>=Xs[0]-rho_2) and (LL[1]>=Xs[1]-rho_2):
    if distance(LL,Xs)<rho :
        pLL = (Xs[0]-LL[0])*(Xs[1]-LL[1])
    else:
        pLL=aire_disk_4
        if LL[1]>=Xs[1]-rho:
            alpha = acos((Xs[1]-LL[1])/rho)
            tarte=(rho2*alpha/2)
            triangle=(Xs[1]-LL[1])*rho*sin(alpha)/2
            pLL = pLL - tarte + triangle
        if LL[0]>=Xs[0]-rho:
            beta = acos((Xs[0]-LL[0])/rho)
            tarte=(rho2*beta/2)
            triangle=(Xs[0]-LL[0])*rho*sin(beta)/2
            pLL = pLL - tarte + triangle  
    if (pLL<=0):
        print ('weird : pLL = %8.3f'%pLL)            
   
    # LR
#    if (LR[0]<=Xs[0]+rho_2) and (LR[1]>=Xs[1]-rho_2):
    if distance(LR,Xs)<rho :
        pLR = (LR[0]-Xs[0])*(Xs[1]-LR[1])
    else:
        pLR=aire_disk_4
        if LR[1]>Xs[1]-rho:
            alpha = acos((Xs[1]-LR[1])/rho)
            tarte=(rho2*alpha/2)
            triangle=(Xs[1]-LR[1])*rho*sin(alpha)/2
            pLR = pLR - tarte + triangle
        if LR[0]<=Xs[0]+rho:
#            print(LR[0]-Xs[0])
            beta = acos((LR[0]-Xs[0])/rho)
            tarte=(rho2*beta/2)
            triangle=(LR[0]-Xs[0])*rho*sin(beta)/2
            pLR = pLR - tarte + triangle  
    if (pLL<=0):
        print ('weird : pLR = %8.3f'%pLR)            
  
    
    # total
    p=(pUL+pUR+pLL+pLR)/aire_tot
    if (p>1):
        print (pUL, pUR, pLL, pLR)
    return p

def reliable(X_s,delta_s,alpha,L,S_hat):
    # 75/94 * (c/64 L)^(d/alpha))
    # ---> cc * c^d_alpha
    d=2 # XS TODO : make this a parameter
    d_alpha=d/alpha
#    cc=75.0/94 * (1.0/(64*L)**d_alpha) # version originale (cf. article)
    # XS TODO : replaced 1.0 by 0.3
    cc=75.0/94 * (0.15/L)**d_alpha # version modifiée (sinon aucun point n'est True)
    # XS TODO : pas trop élevé sinon aucun point n'est informatif
    for X_prime,Y_prime, LB in S_hat: # LB = c = lower bound guarantee on X'
#        print (X_prime,Y_prime, LB)
        p_hat=compute_p_hat(X_s,X_prime)
        p_hat_prime=compute_p_hat(X_prime,X_s)

        # XS TEST
        # p_hat_EZ=compute_p_hat_EZ(X_s,X_prime)
        # if(p_hat-p_hat_EZ) > 0: #0.1:
        #     print('p_hat = %8.3f'%p_hat)
        #     print('p_hat_EZ = %8.3f'%p_hat_EZ)
        # XS END TEST
        
        p_up=cc * LB**d_alpha
#        print('check if reliable: p_hat = %f, p_hat_prime = %f, p_up = %f'%(p_hat,p_hat_prime,p_up))
        if (p_hat <= p_up) or (p_hat_prime <= p_up):
            return True
    return False

def find_knn(x_s,k=2):
    # returns the indices of the k nearest neighbors of x_s *in the whole dataset*
    kk=int(k)
    knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=kk, p=2) # p=2 : Euclidean (L2) distance
    knn.fit(myData.Xtrain)
    a=knn.kneighbors(x_s.reshape(1, -1), return_distance=False) 
    return a[0,1:] # remove index 0 == x_s itself    
    

def marge(y):
    # computes the zeta function from the paper (called z here)
    # returns the label ?
    (unique, counts) = np.unique(y, return_counts=True)
    frequencies = np.asarray((unique, counts))
    z0=frequencies[0]
    z1=frequencies[1]
    if len(z0)==1:
        z=1
        lab=z0[0]
    else:    
        z2=np.sort(z1)
        z=(1.0/len(y))*(z2[-1]-z2[-2])
        lab=z0[np.argmax(z1)]
    return z,lab

def confidentLabel(X_s,k_prime,t,delta):   
    # output : Y_s_hat,LB_s_hat,nQ_s # nQ_s = |Q_s|
    
#    log_1_delta = -log(delta) # = log (1/delta)
    n_cut=1000
    n_neighbors_max = int(k_prime)+1 # I added +1 because find_knn will remove the point itself
    is_informative_point = False
    if (n_neighbors_max > t):
        n_neighbors_max=int(t)
    print ('confidentLabel : considering at most %d neighbors'%n_neighbors_max)
    if (n_neighbors_max > 10*n_cut):
        print ('confidentLabel : warning : n_neighbors_max is very big : %d > %d'%(n_neighbors_max,n_cut))
    KNN=find_knn(X_s,n_neighbors_max) # compute here the (n_neighbors_max) nearest neighbors of X
    Y=[]
    b_dk=0
    kk=0    
    for k in range (n_neighbors_max-1): # starts with k=0 ---> kk=k+1 in formulas
        kk=k+1
#        b_dk= sqrt( (2/kk)*(log_1_delta+log(log_1_delta)+log(log(exp(1)*kk))))
        b_dk=sqrt(2.0/kk)
        Y.append(myData.Ytrain[KNN[k]])# Y_k = label de X_k, kth nearest neighbor of X_s
        (eta_hat,label_y)=marge(Y)
        if eta_hat > 4*b_dk :
            print ('found confident Label after %d neighbors'%kk)
            Y.pop() # remove last Y_k from the list # XS TODO : useful ?? 
            break
        elif (k>n_cut): # XS TODO : this is a hard cutoff to avoid using too mny points
            # XS TODO : chack what happens with LB_hat afterwards
            print ('could not find confident Label after %d neighbors -- force quit '%kk)
            break
        if (k==33):
            print('33 voisins...')

    if(kk==len(Y)):
        print ('warning : could not find confident Label after %d neighbors'%kk)
    LB_hat = eta_hat-2*b_dk
    if (LB_hat >= 0.1 * b_dk):
        is_informative_point = True
    # else:
    #     toto=0.1 * b_dk
    #     print ('LB_hat=%f,0.1 * b_dk=%f'%(LB_hat,toto))
    Y_s=label_y
    return Y_s,LB_hat,kk,is_informative_point  # k+1 = |Q_s|
    
def compute_k(epsilon,delta,bigDelta,C=15.0,beta=1.0):
    petitc=1
    log_1_delta = -log (delta) # = log (1/delta)
#    a=log(log_1_delta) 
#    b=log(log (512*sqrt(exp(1))/bigDelta) )
    k = petitc / bigDelta**2 * (log_1_delta + log(log_1_delta) + log(log (512*sqrt(exp(1))/bigDelta) ))
    return k

if __name__ == '__main__':
#    tic = time.perf_counter()
    for repet in range(10):
        dataType='gaussian5_01' 
        _xymin=-1
        _xymax=1
    
        Q=[] # set of labeled nearest neighbors 
        I=[] # informative set # XS TODO : sert à rien
        S=[] # active set
        S_hat=[] # estimate of active set : list of triplets (X,Y,LB)
        
        epsilon=0.1 # accuracy parameter
        delta=0.1 # Confidence parameter
        beta=1.0 # margin noise parameter
        C=15.0 # margin noise parameter
        L=1.0 #4*pi # smoothness parameter
        alpha=1.0 # smoothness parameter
        tau = 1.0 # lower bound on alpha
        
        log_1_delta = -log (delta) # = log (1/delta)
        bigDelta = (epsilon/(2*C))**(1.0/(beta+1))
        if bigDelta < epsilon/2:
            bigDelta=epsilon/2
        
        M=4 # nombre de classes - 1 (M=1 : binaire)
        
        n_test=30000
        n_train=100000
        n_unlabeled=0
        n_coldStart=0
        nb_reliab=0
    
        n=n_test+n_train+n_coldStart  # n=4*10**5 # label budget
        w=n_train+n_coldStart  # XS TODO : reminder about number of points
        budget=w # could automatically set to n/2 or so
        # attention, le budget s'appelle n dans l'article !
        
        t=budget
        nb_points_used=[]
        
        global myData
        myData=dataSynth.dataSynth(n,2,dataType,M+1)
        myData.makeData()
        myData.split(n_train,n_test,n_unlabeled,n_coldStart)
    
        # computes statistics on the data labels
        # figures out the number of classes
        Y=myData.Ytrain
        (unique, counts) = np.unique(Y, return_counts=True)
        frequencies = np.asarray((unique, counts))
        nb_class=len(frequencies[0])
    
        if (nb_class!=M+1):
            print('problem generating data : check the number of classes (M=%d,nb_class=%d%(M,nb_class))')
            exit()
            
        viz=False # put True to show plots
        if viz:
            plt_check=myData.scatterPlot(True)
            plt.figure()
            plt=myData.scatterPlot(False)  
            plt.ion()
                
        # note: first point will always be considered informative
    #    s=1 # index of point currently examined (watch out : s-1 in python &!!)
    
        n_hat=0
        for s in range (1,w):
            if t<=nb_class-1 : # always non-informative
                break
    #    while(t>0 and s<w): # where w=total number of points in the pool
            delta_s = delta/(32*M*s**2)  
            try: # hack pour assurer de ne pas dépasser la longueur de X
                X_s = myData.Xtrain[s-1]
            except IndexError:
                break
    #        print ('point courant : X_s = ', X_s)
            
            if not reliable(X_s,delta_s,alpha,L,S_hat):
                # run confidentLabel to get an estimate of the label Y_s and the bound LB_s
                # confidentLabel also returns the number (k) of neighbors used, to account for in the budget
                print ("%d (x=%8.3f,y=%8.3f) : reliable=FALSE : potentially informative point"%(s,X_s[0],X_s[1]))
                if viz:
                    plt.scatter(X_s[0],X_s[1],color='yellow', marker='x')
                    plt.draw()
                k_prime = compute_k(epsilon,delta_s,bigDelta)
                Y_s,LB_s,k,is_informative_point = confidentLabel(X_s,k_prime,t,delta)
    #            print ('confidentLabel - examen du point numero %d :'%(s-1), X_s)
    #            print('le vrai label est %d, confidentLabel retourne %d'%(myData.Ytrain[s-1],Y_s))
    
                t=t-k # account for neighbors used in the budget
                I.append(s-1) # ou s ?
                if is_informative_point:
                    S_hat.append((X_s,Y_s,LB_s)) # append a tuple
                    n_hat+=1
                    nb_points_used.append(budget-t) # take into account the number of points used (oracle requests)
                print('----- n_hat=%d ----- '%n_hat)
            else:
                # do not go into confidentLabel when reliable = TRUE
                print ("%d (x=%8.3f,y=%8.3f) : reliable=TRUE : not an informative point"%(s,X_s[0],X_s[1]))          
                nb_reliab +=1
                if viz:
                    plt.scatter(X_s[0],X_s[1],color='black', marker='.')
                    plt.draw()
    
        if viz:
            plt.savefig('./Figures/multi/%s.png'%dataType,format='png',dpi=200)
            plt.show()
        exit()
        knn_act = KNeighborsClassifier(n_neighbors=1)
        knn_pas = KNeighborsClassifier(n_neighbors=1)
        knn_pas5 = KNeighborsClassifier(n_neighbors=5)
        n_info=len(S_hat)
        print(n_info)
        print(n_hat)
        err_act=[]
        err_pas=[]
        err_pas5=[]
        n_pas=[]
        n_pas5=[]
        n_act=nb_points_used
         
        # print('------- S_hat ------')
        # print(S_hat)
        # print('------- S_hat [0] ------')
        # print(S_hat[0])
        # print('------- X [chipo pour 10] ------')
    
        # X = [a_tuple[0] for a_tuple in S_hat[0:10]]
        # Y = [a_tuple[1] for a_tuple in S_hat[0:10]]
        # print(X)
        # print(Y)
        print('-------- active 1NN running ------------------')
        for i in range (1, n_info+1):
            X=[a_tuple[0] for a_tuple in S_hat[0:i]]
            Y=[a_tuple[1] for a_tuple in S_hat[0:i]]
    #        print (X)
    #        print (Y)
            knn_act.fit(X,Y) # fit informative points
            err=1-knn_act.score(myData.Xtest,myData.Ytest)
    #        print('%d:%f'%(i, err))
            err_act.append(err)
    
        # toc = time.perf_counter()
        # print("Execution time : {toc - tic:0.4f} seconds")
    
    #    print (nb_points_used)
    #    print (err_act)
    
        print('-------- passive 1NN running ------------------')
    
        step1=10
        step2=100
    
        for j in range(1,step2,step1): # step saves time
            knn_pas.fit(myData.Xtrain[0:j],myData.Ytrain[0:j]) # fit random points
            err=1-knn_pas.score(myData.Xtest,myData.Ytest)
    # #        print('%d:%f'%(j, err))       
            err_pas.append(err)
            n_pas.append(j)
            
        for j in range(step2,n_train,step2): # step saves time
            knn_pas.fit(myData.Xtrain[0:j],myData.Ytrain[0:j]) # fit random points
            err=1-knn_pas.score(myData.Xtest,myData.Ytest)
    # #        print('%d:%f'%(j, err))       
            err_pas.append(err)
            n_pas.append(j)
           
        print('-------- passive 5NN running ------------------')
    
        step1=10
        step2=100
    
        for j in range(1,step2,step1): # step saves time
            knn_pas5.fit(myData.Xtrain[0:j],myData.Ytrain[0:j]) # fit random points
            try:
                err=1-knn_pas5.score(myData.Xtest,myData.Ytest)
                err_pas5.append(err)
                n_pas5.append(j)
            except ValueError: # Expected n_neighbors <= n_samples,  but n_samples = 1, n_neighbors = 5
                 pass           
    
        for j in range(step2,n_train,step2): # step saves time
            knn_pas5.fit(myData.Xtrain[0:j],myData.Ytrain[0:j]) # fit random points
            try:
                err=1-knn_pas5.score(myData.Xtest,myData.Ytest)
                err_pas5.append(err)
                n_pas5.append(j)
            except ValueError: # Expected n_neighbors <= n_samples,  but n_samples = 1, n_neighbors = 5
                 pass           
    # #        print('%d:%f'%(j, err))       
    
        if viz: 
            plt.figure()
            plt.plot(nb_points_used,err_act, color='r', label='active MKAL 1NN')
            plt.plot(n_pas,err_pas, color='b', label='passive 1NN')
            plt.plot(n_pas5,err_pas5, color='g', label='passive 5NN')
            plt.xlabel("number of labels used")
            plt.ylabel("error on test set")
            plt.ylim((0,0.5))
            plt.title("multiclass")
            plt.legend(loc='upper right')
            plt.savefig('./Figures/multi/active_vs_passive5_%s_%d_classes_r%d.png'%(dataType,M+1,repet),format='png',dpi=200)
            plt.show()
        
        # tac = time.perf_counter()
        # print("Execution time : {tac - toc:0.4f} seconds")
    
        fact=open('./results/multi/%s_results_active_r%d.txt'%(dataType,repet),'w')
        for i in range(len(n_act)):
            fact.write('%d  %8.3f \n'%(n_act[i],err_act[i]))
        fact.close()
    
        fpas=open('./results/multi/%s_results_passive_r%d.txt'%(dataType,repet),'w')
        for i in range(len(n_pas)):
            fpas.write('%d  %8.3f \n'%(n_pas[i],err_pas[i]))
        fpas.close()
    
        fpas5=open('./results/multi/%s_results_passive5_r%d.txt'%(dataType,repet),'w')
        for i in range(len(n_pas5)):
            fpas5.write('%d  %8.3f \n'%(n_pas5[i],err_pas5[i]))
        fpas5.close()

    
# to make gifs:
#    https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30