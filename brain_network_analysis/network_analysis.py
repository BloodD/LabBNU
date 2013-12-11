import os
import sys
import scipy as sp
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import csv 
from scipy import signal
from matplotlib import cm 
from compiler.ast import flatten
from matplotlib.dates import num2date

def read_sess_list(sess):
    """Read the subject id"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess

def read_networks(datadir,idlist):
    "Load the networks"
    txtlist = ["%05d_ProbabilisticMatrix_OPD_90.txt"%i for i in idlist]
       
    mat_net = []
    row_net = []
    
    for i in range(0,len(txtlist)):        
        net_file = os.path.join(datadir,txtlist[i])
        sub = np.loadtxt(net_file, dtype=float)   
        mat_net.append(sub)
        flatten_sub = sub.flatten()
        row_net.append(flatten_sub)
    mat_net = np.array(mat_net)  
    row_net = np.array(row_net)
    print mat_net.shape, row_net.shape
    return mat_net,row_net

def net_reorder(net):

    """
    Reorder the network by odd partition and even partition for left and right brain
    """
    m = net.shape[0]
    tmp_net = np.zeros_like(net)
    result_net = np.zeros_like(net)
    index_odd = []
    index_even = []
   
    for i in range(0,m/2):
        index_odd.append(i*2)
        index_even.append(i*2+1)
    
    for i,odd in enumerate(index_odd):   
        tmp_net[i] = net[odd]
    for i,even in enumerate(index_even):
        tmp_net[i+m/2] = net[even]
    
    tmp_net = tmp_net.T
    
    for i,odd in enumerate(index_odd):    
        result_net[i] = tmp_net[odd]
    for i,even in enumerate(index_even):
        result_net[i+m/2] = tmp_net[even]
    
    result_net = result_net.T
    return result_net  

def nets_reorder_all(nets,num):
    tmp_nets = nets
    for i in range(0,num):
        tmp_nets[i] = net_reorder(tmp_nets[i])
    return tmp_nets
    
def net_diagonal_average(nets,num):
    """
    Diagonal average the networks
    """
    net_tmp = nets
    net_tmp_r = []
    for i in range(0,num):
        net_tmp[i] = (net_tmp[i]+net_tmp[i].T)/2
        flatten_net = net_tmp[i].flatten()
        net_tmp_r.append(flatten_net) 
        net_r = np.array(net_tmp_r)
    return net_tmp,net_r
    
def net_average(nets,num):
    """
    Average the networks for group analysis
    """    
    net_a = np.zeros_like(nets[0])
    for i in range(0,num):
        net_a=net_a+nets[i]
    net_a=net_a/num
    net_a_r = net_a.flatten()
    return net_a,net_a_r

def net_standard_deviation(nets,net_a,num):
    """
    Standard deviation of group networks.
    """
    sd = np.zeros_like(net_a)
    
    for i in range(0,num):
        sd =sd+(nets[i]-net_a)**2
    sd = sd/(num-1)
    sd_r = sd.flatten() 
    return sd,sd_r

def net_filter(net_a,net_std,thr):
    """
    Define the unconnected networks by mean < (thr - 2*std)
    """
    net_mask = net_a > thr-2*net_std      
    return net_mask

def symm_std(nets,num):   
    aymm = []
    for i in range(0,num):
        diff_level = np.abs(nets[i]-nets[i].T).sum()/2.0
        aymm.append(diff_level)
    print aymm
    return aymm

def network_diag_correlation(net):
    shape = net.shape[0]
    tmp = net.T
    indexs_l = np.transpose(np.tril_indices(shape))
    #print indexs_l
    l = []
    r = []
    for i in indexs_l:
        l.append(net[i[0]][i[1]])
        r.append(tmp[i[0]][i[1]])
    #print l,r
    corr = stats.pearsonr(l,r)
    return corr 

def net_diag_corr(nets,num):
    corr_list = []
    for i in range(0,num):
        corr = network_diag_correlation(nets[i])
        corr_list.append(corr)
    return np.array(corr_list)

def save_net_diag_aver(nets,num,target_dir):
    
    for i in range(0,num):
        target_file = os.path.join(target_dir,"Subject%05d.txt"%(i+1))
        np.savetxt(target_file,nets[i],fmt='%1.4e',delimiter='    ',newline='\n')
    return True
    
def plot_net_save(nets,num,target_dir):
    """
    Plot the connectivity map.
    """
    shape = nets[0].shape[0]
    for i in range(0,num):
        plt.imshow(nets[i],interpolation='nearest',cmap=cm.jet)
        plt.colorbar()
        plt.title("Subject %05d"%(i+1))
        target_file = os.path.join(target_dir,"Subject%05d.png"%(i+1))
        plt.savefig(target_file)
        plt.close()
        print "save ok!"
    return True
     
def plot_net_edages_save_a(net_row,net_stds,subid):
    
    N = len(net_row)
    #N = 100
    ind = np.arange(N)  
    width = 0.3
    c1 = plt.bar(ind, net_row[:N],width, color='r',yerr=net_stds[:N])
    plt.xlabel('edges(1-100)')
    plt.ylabel('probability connectivity')
    plt.title('Network edges of subject %s'%subid)
    #plt.xticks(np.arange(0,N,3))
    #plt.yticks(np.arange(0,0.16,0.01))
    plt.grid(True)
    plt.legend( (c1[0]), ('W+GM'),'upper left')
    plt.show()
    
def plot_net_edages_save(net_row,net_stds,subid):
    
    N = len(net_row[0])
    N=50
    ind = np.arange(N) 
    width = 0.20
    print net_row.shape,net_stds.shape
    
    c1 = plt.bar(ind, net_row[0,:N],width, color='r',yerr=net_stds[0,:N],ecolor='k')
    c2 = plt.bar(ind+width, net_row[1,:N],width, color='g',yerr=net_stds[1,:N],ecolor='k')
    c3 = plt.bar(ind+width*2, net_row[2,:N],width, color='y',yerr=net_stds[2,:N],ecolor='k')
    
    plt.xlabel('edges(1-100)')
    plt.ylabel('probability connectivity')
    plt.title('Network edges strength contrast')
    #plt.xticks(np.arange(0,(N+1),1))
    #plt.yticks(np.arange(0,1,0.0001))
    #plt.grid(True)
    plt.legend( (c1[0],c2[0],c3[0]), ('WGM', 'WM_fa','WM_reg'),'upper left')
    plt.show()
    
def net_row_save(net_row,filename):
    net_out = net_row.T
    np.savetxt(filename,net_out,fmt='%1.4e',delimiter=',',newline='\n')
    print "Networks have been saved in the %s"%filename    



def main():
    """
    DTI connectivity analysis.
    """
    
    t = np.arange(0.0,0.1,0.005)
    data_test = "G:/00002_ProbabilisticMatrix_OPD_90.txt"
    ## fc1000 test data
    net = np.loadtxt(data_test, dtype=float)   
    print net
    n = net_reorder(net)
    ne = (n+n.T)/2 
    sparsity_l = []
    sp_l = []
    np.savetxt('G:/av.txt',ne,fmt='%1.4e',delimiter='    ',newline='\n')
    for thr in t:
        thr_mask = ne>thr
        net_new =  ne*thr_mask
        sparsity = np.sum(net_new>0)/(90*90.0)
        sparsity_l.append(sparsity)
        sp = np.sum(net_new)/(90)
        sp_l.append(sp)
    plt.figure(1)
    plt.plot(t,sparsity_l,'ro-')
    plt.grid(True)
    plt.figure(2)
    plt.plot(t,sp_l,'b^-')
    plt.grid(True)
    plt.figure(3)
    plt.imshow(net_new,interpolation='nearest',cmap=cm.jet)
    plt.colorbar()
    #plt.xlabel("AAL Node 0-89")
    #plt.ylabel("AAL Node 0-89")
    #plt.xticks(np.arange(0,90,5))
    #plt.yticks(np.arange(0,90,5))
    #plt.title("Mean connectivity WGM")
    plt.grid(True)
    plt.show()
    
    ### data_source
    
    data_dir = "G:/DTI_networks/"
    mat_save_dir = "G:/DTI_networks/sorder"
    data_dir_fa = "G:/DTI_netwoks_wm_fa_method"
    mat_fa_save_dir = "G:/DTI_netwoks_wm_fa_method/sorder"
    data_dir_reg = "G:/DTI_networks_wm_reg_method"
    mat_reg_save_dir = "G:/DTI_networks_wm_reg_method/sorder"
    
    sub_num = 50
    idlist = range(1,sub_num+1)
    print idlist
    t = np.arange(0.0,0.1,0.005)

    #WGM AAL mask 
    #1. read networks
    nets1,row_nets1 = read_networks(data_dir,idlist)
    #2. diag average and save as txt 
    nets_ro1 = nets_reorder_all(nets1,sub_num)
    lr_corr1 = net_diag_corr(nets_ro1,sub_num)
    print "min, max, mean, std:",lr_corr1[:,0].min(),lr_corr1[:,0].max(), np.mean(lr_corr1[:,0]),np.std(lr_corr1[:,0])
    asymm1 = symm_std(nets1,sub_num)
    
    net_da1,net_r1 = net_diagonal_average(nets1,sub_num)
    #3.save the preprocessed networks
    save_net_diag_aver(net_da1,sub_num,mat_save_dir)
    #plot_net_save(net_da1,sub_num,mat_save_dir)
    #3. group average and standard_deviation
    net_aver1,net_a_r1=net_average(net_da1,sub_num) 
    net_sd1,net_sd_r1 = net_standard_deviation(net_da1,net_aver1,sub_num)
    sparsity_l1 = []
    sp_l1 = []
    for thr in t:
        thr_mask = net_filter(net_aver1,net_sd1,thr)
        net_new_average_1 =  net_aver1*thr_mask
        #compute the sparsity and Network strength
        sparsity = np.sum(net_new_average_1>0)/(90*90.0)
        sparsity_l1.append(sparsity)
        sp = np.sum(net_new_average_1)/(90)
        sp_l1.append(sp)
    '''
    plt.figure(1)
    plt.plot(t,sparsity_l1,'ro-')
    plt.ylabel("sparsity")
    plt.xlabel("Threshold")
    plt.xticks(t)
    plt.grid(True)
    plt.figure(2)
    plt.plot(t,sp_l1,'bo-')
    plt.ylabel("Sp")
    plt.xlabel("Threshold")
    plt.xticks(t)
    plt.grid(True)
    '''
    plt.figure(3)
    plt.imshow(net_aver1,interpolation='nearest',cmap=cm.jet)
    plt.colorbar()
    plt.title("Group Mean Map of WGM")
    plt.grid(True)
    target_file = os.path.join(data_dir,"Group_Mean_Map_of_WGM.png")
    plt.savefig(target_file)
    plt.show()
    
    #WM-fa
    #1. read networks
    nets2,row_nets2 = read_networks(data_dir_fa,idlist)
    #2. diag average and save as txt 
    nets_ro2 = nets_reorder_all(nets2,sub_num)
    lr_corr2 = net_diag_corr(nets_ro2,sub_num)
    print "min, max, mean, std:",lr_corr2[:,0].min(),lr_corr2[:,0].max(), np.mean(lr_corr2[:,0]),np.std(lr_corr2[:,0])
    asymm2 = symm_std(nets2,sub_num)
    
    net_da2,net_r2 = net_diagonal_average(nets2,sub_num)
    #3.save the preprocessed networks
    save_net_diag_aver(net_da2,sub_num,mat_fa_save_dir)
    #plot_net_save(net_da2,sub_num,mat_fa_save_dir)
    #3. group average and standard_deviation
    net_aver2,net_a_r2=net_average(net_da2,sub_num) 
    net_sd2,net_sd_r2 = net_standard_deviation(net_da2,net_aver2,sub_num)
    sparsity_l2 = []
    sp_l2 = []
    for thr in t:
        thr_mask = net_filter(net_aver2,net_sd2,thr)
        net_new_average_2 =  net_aver2*thr_mask
        #compute the sparsity and Network strength
        sparsity = np.sum(net_new_average_2>0)/(90*90.0)
        sparsity_l2.append(sparsity)
        sp = np.sum(net_new_average_2)/(90)
        sp_l2.append(sp)
    '''
    plt.figure(1)
    plt.plot(t,sparsity_l2,'ro-')
    plt.ylabel("sparsity")
    plt.xlabel("Threshold")
    plt.xticks(t)
    plt.grid(True)
    plt.figure(2)
    plt.plot(t,sp_l2,'bo-')
    plt.ylabel("Sp")
    plt.xlabel("Threshold")
    plt.xticks(t)
    plt.grid(True)
    '''
    plt.figure(3)
    plt.imshow(net_aver2,interpolation='nearest',cmap=cm.jet)
    plt.colorbar()
    plt.title("Group Mean Map of FA_method")
    plt.grid(True)
    target_file = os.path.join(data_dir_fa,"Group_Mean_Map_of_FA_method.png")
    plt.savefig(target_file)
    plt.show()
    
    #WM-reg
    #1. read networks
    nets3,row_nets3 = read_networks(data_dir_reg,idlist)
    #2. diag average and save as txt 
    nets_ro3 = nets_reorder_all(nets3,sub_num)
    lr_corr3 = net_diag_corr(nets_ro3,sub_num)
    print "min, max, mean, std:",lr_corr3[:,0].min(),lr_corr3[:,0].max(), np.mean(lr_corr3[:,0]),np.std(lr_corr3[:,0])
    asymm3 = symm_std(nets3,sub_num)
    
    net_da3,net_r3 = net_diagonal_average(nets3,sub_num)
    #3.save the preprocessed networks
    save_net_diag_aver(net_da3,sub_num,mat_reg_save_dir)
    #plot_net_save(net_da3,sub_num,mat_reg_save_dir)
    #3. group average and standard_deviation
    net_aver3,net_a_r3 = net_average(net_da3,sub_num) 
    net_sd3,net_sd_r3 = net_standard_deviation(net_da3,net_aver3,sub_num)
    sparsity_l3 = []
    sp_l3 = []
    for thr in t:
        thr_mask = net_filter(net_aver3,net_sd3,thr)
        net_new_average_3 =  net_aver3*thr_mask
        #compute the sparsity and Network strength
        sparsity = np.sum(net_new_average_3>0)/(90*90.0)
        sparsity_l3.append(sparsity)
        sp = np.sum(net_new_average_3)/(90)
        sp_l3.append(sp)
    '''
    plt.figure(1)
    plt.plot(t,sparsity_l3,'ro-')
    plt.ylabel("sparsity")
    plt.xlabel("Threshold")
    plt.xticks(t)
    plt.grid(True)
    plt.figure(2)
    plt.plot(t,sp_l3,'bo-')
    plt.ylabel("sparsity")
    plt.xlabel("Threshold")
    plt.xticks(t)
    plt.grid(True)
    '''
    plt.figure(3)
    plt.imshow(net_aver3,interpolation='nearest',cmap=cm.jet)
    plt.colorbar()
    plt.title("Group Mean Map of reg_method")
    plt.grid(True)
    target_file = os.path.join(data_dir_reg,"Group_Mean_Map_of_Reg_method.png")
    plt.savefig(target_file)
    plt.show()
    
    
    plt.figure(1)
    plt.plot(t,sparsity_l1,'ro-',label='WGM')
    plt.plot(t,sparsity_l2,'b<-',label='FA')
    plt.plot(t,sparsity_l3,'g>-',label='REG')
    plt.ylabel("sparsity")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.figure(2)
    plt.plot(t,sp_l1,'ro-',label='WGM')
    plt.plot(t,sp_l2,'b<-',label='FA')
    plt.plot(t,sp_l3,'g>-',label='REG')
    plt.ylabel("Sp")
    plt.xlabel("Threshold")
    plt.xticks(np.arange(0.0,0.105,0.01))
    plt.legend(loc="upper right")
    plt.grid(True)    
    plt.show()
    
    ### group analysis############
    #stat0. network pattern
    corr1 = stats.pearsonr(net_a_r1,net_a_r2) 
    corr2 = stats.pearsonr(net_a_r1,net_a_r3) 
    corr3 = stats.pearsonr(net_a_r2,net_a_r3) 
    print "network pattern correlate coefficient:"
    print  corr1, corr2, corr3
    
    #stat1. mean differents map
    mean_diff = net_aver2 - net_aver3
    strong = mean_diff.flatten() > 0
    print strong.sum()
    weak = mean_diff.flatten() < 0
    print weak.sum()
    plt.imshow(mean_diff,interpolation='nearest',cmap=cm.jet)
    plt.colorbar()
    plt.title("Mean connectivity difference:WM_fa - WM_reg")
    plt.grid(True)
    plt.show()
    
    #stat2.stability map
    stable = net_sd2 - net_sd3 
    strong = stable.flatten() > 0
    print strong.sum()
    weak = stable.flatten() < 0
    print weak.sum()
    
    plt.imshow(stable,interpolation='nearest',cmap=cm.jet)
    plt.colorbar()
    plt.title("Stability difference: WM_fa - WM_reg")
    plt.grid(True)
    plt.show()
    
    #3.symmetry_1 
    m1 = np.mean(lr_corr1[:,0])
    std1 = np.std(lr_corr1[:,0])
    m2 = np.mean(lr_corr2[:,0])
    std2 = np.std(lr_corr2[:,0])
    m3 = np.mean(lr_corr3[:,0])
    std3 = np.std(lr_corr3[:,0])
    
    print m1,std1,m2,std2,m3,std3
    
    t_test1 = stats.ttest_ind(lr_corr1[:,0],lr_corr2[:,0])
    t_test2 = stats.ttest_ind(lr_corr1[:,0],lr_corr3[:,0])
    t_test3 = stats.ttest_ind(lr_corr2[:,0],lr_corr3[:,0])
    
    print t_test1
    print t_test2
    print t_test3
       
    N = 1
    ind = np.arange(N)  
    width = 0.1

    c1 = plt.bar(ind, [m1],width, color='r',yerr=[std1],ecolor='k')
    c2 = plt.bar(ind+0.3, [m2],width, color='b',yerr=[std2],ecolor='k')
    c3 = plt.bar(ind+0.6, [m3],width, color='g',yerr=[std3],ecolor='k')
    plt.ylabel('Patern Symmetry')
    plt.yticks(np.arange(0,1,0.3))
    #plt.xticks(ind+width/2., ('WM_fa','WM_reg') )
    #plt.xticks(ind+0.5+width/2., ('WM_reg') )
    plt.grid(True)
    plt.legend( (c1[0],c2[0],c3[0]), ('WGM','WM_FA','WM_reg'),'upper right')
    plt.title("Symmetry")
    #plt.text(ind+width/2., 8*m1, '*',ha='center', va='bottom')   

    plt.show()
    
    #3.symmetry_2 
    m1 = np.mean(asymm1)
    std1 = np.std(asymm1)
    m2 = np.mean(asymm2)
    std2 = np.std(asymm2)
    m3 = np.mean(asymm3)
    std3 = np.std(asymm3)
    
    t_test1 = stats.ttest_ind(asymm1,asymm2)
    t_test2 = stats.ttest_ind(asymm1,asymm3)
    t_test3 = stats.ttest_ind(asymm2,asymm3)
    
    print t_test1
    print t_test2
    print t_test3
    
  
    N = 1
    ind = np.arange(N)  
    width = 0.1

    c1 = plt.bar(ind, [m1],width, color='r',yerr=[std1],ecolor='k')
    c2 = plt.bar(ind+0.3, [m2],width, color='b',yerr=[std2],ecolor='k')
    c3 = plt.bar(ind+0.6, [m3],width, color='g',yerr=[std3],ecolor='k')
    plt.ylabel('Strength Asymmetry')
    
    #plt.xticks(ind+width/2., ('WM_fa','WM_reg') )
    #plt.xticks(ind+0.5+width/2., ('WM_reg') )
    plt.grid(True)
    plt.legend( (c1[0],c2[0],c3[0]), ('WGM','WM_FA','WM_reg'),'upper right')
    plt.title("Asymmetry")
    #plt.text(ind+width/2., 8*m1, '*',ha='center', va='bottom')   
    plt.show()
    
    '''
    data_test = "G:/00002_ProbabilisticMatrix_OPD_90.txt"
    ## fc1000 test data
    net = np.loadtxt(data_test, dtype=float)   
    print net
    n = net_reorder(net,90)
    ne = (n+n.T)/2 
    sparsity_l = []
    sp_l = []
    np.savetxt('G:/av.txt',ne,fmt='%1.4e',delimiter='    ',newline='\n')
    for thr in t:
        thr_mask = ne>thr
        net_new =  ne*thr_mask
        sparsity = np.sum(net_new>0)/(90*90.0)
        sparsity_l.append(sparsity)
        sp = np.sum(net_new)/(90)
        sp_l.append(sp)
    plt.figure(1)
    plt.plot(t,sparsity_l,'ro-')
    plt.grid(True)
    plt.figure(2)
    plt.plot(t,sp_l,'b^-')
    plt.grid(True)
    plt.figure(3)
    plt.imshow(net_new,interpolation='nearest',cmap=cm.jet)
    plt.colorbar()
    #plt.xlabel("AAL Node 0-89")
    #plt.ylabel("AAL Node 0-89")
    plt.xticks(np.arange(0,90,5))
    plt.yticks(np.arange(0,90,5))
    #plt.title("Mean connectivity WGM")
    plt.grid(True)
    plt.show()
    '''

if __name__=="__main__":
    main()

        