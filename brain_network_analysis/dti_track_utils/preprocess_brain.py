#!/usr/bin/env python
#Make subject ROI

"""
prepare files and fast single subject. 
"""

import os 
import sys
import argparse
import subprocess as subp
import time 
import shutil

def main():
    """
    Prepare files and fast subject brain.
    """

    parser = argparse.ArgumentParser(description = 'prepare files and fast subject brain')
    parser.add_argument('-s',
                        dest = 'sess',
                        required = True,
                        help = 'Input sess list file.'
                        )
    parser.add_argument('-d',
                        dest = 'datadir',
                        required = True,
                        help = 'Data directory.'
                        )
    parser.add_argument('-dr',
                        dest = 'resultdir',
                        required = True,
                        help = 'Data directory.'
                        )

    args = parser.parse_args()

    sessfile = args.sess
    sess = read_sess_list(sessfile)
    data_dir = args.datadir
    data_res = args.resultdir
    
    mni_brain = os.path.join(data_dir,'MNI152_T1_2mm_brain.nii.gz')
    xfm_s2b = 'standard2brain.mat'
    xfm_b2d = 'brain2diffusion.mat'
    #xfm_ = 'brain2standard.mat'
    for index ,sid in enumerate(sess):
        index = index+1
        print index,sid
        
        sess_dir = os.path.join(data_dir,sid+'_vol')
        s2b_xfm = os.path.join(sess_dir,xfm_s2b)
        b2d_xfm = os.path.join(sess_dir,xfm_b2d)
        nodif   = os.path.join(sess_dir,'nodif_brain.nii.gz')

        res_dir = os.path.join(data_res,'%05d'%index)
        nodif_brain = os.path.join(res_dir,'native_space/nodif_brain.nii.gz')
        print nodif_brain
        
        brain_s = os.path.join(sess_dir,'brain.nii.gz')
        
        aal_diffusion = os.path.join(sess_dir,sid+'_aal_diff.nii.gz')
        aal_dest = os.path.join(data_res,'%05d/native_space/%05d_AAL_WM_R.nii.gz'%(index,index))

        cmd = "cp %s %s"%(aal_diffusion,aal_dest)
        print cmd
        os.system(cmd)
        
        '''
        cmd1 = "cp %s %s"%(nodif_brain,nodif)
        print cmd1
        os.system(cmd1)
        
        cmd2 = 'flirt -dof 6 -in %s -ref %s -omat %s'%(brain_s,nodif,b2d_xfm)
        print cmd2
        #os.system(cmd2)
        subp.Popen(cmd2,shell=True)
        
        cmd3 = 'flirt -in %s -ref %s -omat %s'%(mni_brain,brain_s,s2b_xfm)
        print cmd3
        #os.system(cmd3) 
        subp.Popen(cmd3,shell=True)
        
        if os.path.exists(sess_dir):
            if os.path.exists(brain_s):
                cmd = "fast -g -o %s %s"%(os.path.join(sess_dir,sid),brain_s)
                print cmd
                subp.Popen(cmd,shell=True)
        '''
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

def read_sess_list(sess):
    """read session list file,the file content subject IDs"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess

if __name__=='__main__':
    main()
