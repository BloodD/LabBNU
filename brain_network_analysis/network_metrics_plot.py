import os
import sys
import numpy as np
from matplotlib import cm 
import matplotlib.pyplot as plt
import csv 


def load_parameter(datadir,idlist):
    
    gE_l = ["Efficiency_%04d/gE.txt"%i for i in idlist]
    lE_l = ["Efficiency_%04d/locE.txt"%i for i in idlist]
    Cp_l = ["SmallWorld_%04d/Cp.txt"%i for i in idlist]
    Lp_l = ["SmallWorld_%04d/Lp.txt"%i for i in idlist]
    Gamma_l = ["SmallWorld_%04d/Gamma.txt"%i for i in idlist]
    Lambda_l = ["SmallWorld_%04d/Lambda.txt"%i for i in idlist]
    
    print gE_l
    print Gamma_l
    
    gE_list = []
    lE_list = []
    Cp_list = []
    Lp_list = []
    Gamma_list = []
    Lambda_list = []
    
    for i in gE_l:        
        p = os.path.join(datadir,i)
        print p
        sub = np.loadtxt(p, dtype=float)   
        gE_list.append(sub)
    gE_list = np.array(gE_list)  
    
    for i in lE_l:        
        p = os.path.join(datadir,i)
        print p
        sub = np.loadtxt(p, dtype=float)   
        lE_list.append(sub)
    lE_list = np.array(lE_list)  
    
    for i in Cp_l:        
        p = os.path.join(datadir,i)
        print p
        sub = np.loadtxt(p, dtype=float)   
        Cp_list.append(sub)
    Cp_list = np.array(Cp_list)  
    
    for i in Lp_l:        
        p = os.path.join(datadir,i)
        print p
        sub = np.loadtxt(p, dtype=float)   
        Lp_list.append(sub)
    Lp_list = np.array(Lp_list) 
    
    for i in Gamma_l:        
        p = os.path.join(datadir,i)
        print p
        sub = np.loadtxt(p, dtype=float)   
        Gamma_list.append(sub)
    Gamma_list = np.array(Gamma_list)
    
    for i in Lambda_l:        
        p = os.path.join(datadir,i)
        print p
        sub = np.loadtxt(p, dtype=float)   
        Lambda_list.append(sub)
    Lambda_list = np.array(Lambda_list) 
    
    para_list =np.array([gE_list,
                 lE_list,
                 Cp_list,
                 Lp_list,
                 Gamma_list,
                 Lambda_list])
    
    return para_list

def main():
    
    data_dir = "G:/DTI_networks/sorder/net_m"   
    data_dir_fa = "G:/DTI_netwoks_wm_fa_method/sorder/net_m"
    data_dir_reg = "G:/DTI_networks_wm_reg_method/sorder/net_m"
   
    dir_list = [data_dir,data_dir_fa,data_dir_reg]
    sub_num = 50
    idlist = range(1,sub_num+1)
    para_list_all = []
    for iter in dir_list:
        para_list = load_parameter(iter,idlist)
        para_list_all.append(para_list)
    para_list_all = np.array(para_list_all)
    print para_list_all.shape
  
    print np.mean(para_list_all[0][0],0)
    
    t = np.arange(0.0,0.105,0.005)
    print t.shape
    
    plt.figure(1)
    plt.plot(t,np.mean(para_list_all[0][0],0),'ro-', label='WGM')
    plt.plot(t,np.mean(para_list_all[1][0],0),'b<-', label='FA')
    plt.plot(t,np.mean(para_list_all[2][0],0),'g>-', label='REG')
    plt.ylabel("global efficiency")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.figure(2)
    plt.plot(t,np.mean(para_list_all[0][1],0),'ro-',label='WGM')
    plt.plot(t,np.mean(para_list_all[1][1],0),'b<-',label='FA')
    plt.plot(t,np.mean(para_list_all[2][1],0),'g>-',label='REG')
    plt.ylabel("local efficiency")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper right")
    plt.grid(True)

    
    plt.figure(3)
    plt.plot(t,np.mean(para_list_all[0][2],0),'ro-',label='WGM')
    plt.plot(t,np.mean(para_list_all[1][2],0),'b<-',label='FA')
    plt.plot(t,np.mean(para_list_all[2][2],0),'g>-',label='REG')
    plt.ylabel("Cp")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper right")
    plt.grid(True)
    
    
        
    plt.figure(4)
    plt.plot(t,np.mean(para_list_all[0][3],0),'ro-',label='WGM')
    plt.plot(t,np.mean(para_list_all[1][3],0),'b<-',label='FA')
    plt.plot(t,np.mean(para_list_all[2][3],0),'g>-',label='REG')
    plt.ylabel("Lp")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper left")
    plt.grid(True)
   
    
    plt.figure(5)
    print para_list_all[0][4]
    plt.plot(t,np.mean(para_list_all[0][4],0),'ro-',label='WGM')
    plt.plot(t,np.mean(para_list_all[1][4],0),'b<-',label='FA')
    plt.plot(t,np.mean(para_list_all[2][4],0),'g>-',label='REG')
    plt.ylabel("Gamma")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper left")
    plt.grid(True)
  
    
    plt.figure(6)
    plt.plot(t,np.mean(para_list_all[0][5],0),'ro-',label='WGM')
    plt.plot(t,np.mean(para_list_all[1][5],0),'b<-',label='FA')
    plt.plot(t,np.mean(para_list_all[2][5],0),'g>-',label='REG')
    plt.ylabel("Lambda")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    
  
   
    
if __name__=="__main__":
    main()
    
    