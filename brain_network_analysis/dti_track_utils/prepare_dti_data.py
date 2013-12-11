#!/usr/bin/env python
import os
import sys
from tran_bvals import trans_bval 

def main():
    

    data_dir = '/nfs/s2/dticenter/nspworking/subj2008/nii'
    sess_f   = raw_input('input sess file>>>')
    sess_list = read_sess_list(sess_f)

    dest_dir = './'
    pg = open('prepare.log','w')
    for subject in sess_list:
        pg.write('Perpare %s data \n'%subject)
        sess_dir = os.path.join(data_dir,subject)
        dti_f = os.path.join(sess_dir,'dti/004/d.nii.gz')
        bvals = os.path.join(sess_dir,'dti/004/fsl.bval')
        bvecs = "./bvecs"
        anat = os.path.join(sess_dir,'3danat/reg/brain.nii.gz')
        print dti_f
        print bvals
        print anat
        sess = os.path.join(dest_dir,subject+'/DTI')
        os.system('mkdir -p %s'%sess)
        
        dti = os.path.join(sess,'sub.nii.gz')
        cmd1 = 'cp %s %s '%(dti_f,dti)
        os.system(cmd1)
        pg.write(cmd1+'\n')
        
        bvals_d = os.path.join(sess,'bvals')
        cmd2 = 'cp %s %s '%(bvals,bvals_d)
        os.system(cmd2)
        pg.write(cmd2+'\n')

        trans_bval(bvals_d)
        bvecs_d = os.path.join(sess,'bvecs')
        cmd3 = 'cp %s %s '%(bvecs,bvecs_d)
        os.system(cmd3)
        pg.write(cmd3+'\n')

        anat_dir = os.path.join(dest_dir,subject+'_vol')
        os.system('mkdir -p %s'%anat_dir)
        anat_d = os.path.join(anat_dir,'brain.nii.gz')
        cmd4 = 'cp %s %s '%(anat,anat_d)
        os.system(cmd4)
        pg.write(cmd4+'\n')
    
    pg.close()
    print 'work done...'

def read_sess_list(sess):
    """docstring for fname"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess

if __name__=='__main__':
    main()
