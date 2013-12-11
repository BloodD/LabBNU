#!/usr/bin/env python
#iMake subject ROI
"""
Brain region segmentation pipeline.
"""
import os 
import sys
import argparse
import subprocess as subp
import time 
import shutil
import numpy as np
import scipy as sp

from spectral_seg import spectral_seg
from scipy.sparse import csc_matrix
from mvpa2.datasets.mri import map2nifti,fmri_dataset

def main():
    parser = argparse.ArgumentParser(description = 'Make ROIs mask')
    parser.add_argument('-s',
                        dest = 'sess',
                        required = True,
                        help = 'Input sess list file.'
                        )
    parser.add_argument('-l',
                        dest = 'label',
                        required = True,
                        help = 'Input label file.'
                        )
    parser.add_argument('-n',
                        dest = 'sample',
                        required = True,
                        help = 'sample number default=5000.'
                        )

    args = parser.parse_args()
    label = args.label
    sample = args.sample

    sessfile = args.sess
    sess = read_sess_list(sessfile)
    origin = '/nfs/t1'
    data_dir = '../probtrack'
    xfm1 = 'xfm/brain2diff.mat'
    xfm2 = 'xfm/brain2standard.mat'
    xfm3 = 'xfm/standard2brain.mat'
    
    fdt_dir = os.path.join(os.getcwd(),'fdt_result')
    if not os.path.exists(fdt_dir):
        os.mkdir(fdt_dir)

    for sid in sess:
        sid = sid.rstrip()
        sess_dir = os.path.join(data_dir, sid)
        xfm_file1 = os.path.join(sess_dir, xfm1)
        xfm_file2 = os.path.join(sess_dir, xfm2)
        xfm_file3 = os.path.join(sess_dir, xfm3)

        mask_dir = os.path.join(sess_dir, 'mask')
        merged = os.path.join(sess_dir,'bedpostX/merged')
        mask   = os.path.join(sess_dir,'diffmask/b0_brain_mask.nii.gz')
        vox    = os.path.join(sess_dir,'vox')
        brain_s  = os.path.join(sess_dir,'anat/brain.nii.gz')
        brain_wm = os.path.join(sess_dir,'anat/brain_pve_2.nii.gz')
        brain_seg = os.path.join(sess_dir,'anat/brain_seg_2.nii.gz')
        outbase  = os.path.join(sess_dir,'anat/brain')
        fdt_res  = os.path.join(sess_dir,'fdt_rofa*')
        fdt_sub  = os.path.join(fdt_dir,'fdt_rofa_%s.nii.gz'%sid)
        st_brain = os.path.join(os.getcwd(),'MNI152_T1_2mm_brain.nii.gz')
        roi_f = os.path.join(sess_dir,'mask/'+label)
        dot_f = os.path.join(sess_dir,'fdt_matrix2.dot')
        hdf   = os.path.join(sess_dir,'fdt_matrix2.hdf5')
        parcel_f = os.path.join(sess_dir,'mask/'+label.replace('.nii.gz',
                                  '_parcel_3.nii.gz'))
        #print roi_f
        #print dot_f
        #print hdf 
        print parcel_f

        if not os.path.exists(sess_dir):
            print 'can not find: %s'%sess_dir
            continue
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        if not os.path.exists(brain_wm):
            cmd = "fast -g -o %s %s"%(outbase,brain_s)
            print cmd
            subp.Popen(cmd,shell=True)
        outmask = os.path.join(mask_dir,'fg.nii.gz')
        
        ###transform the roi into structure space.
        cmdf = "flirt -in %s  -ref %s -interp nearestneighbour\
                 -applyxfm -init %s -out %s"%(label,brain_s,xfm_file3,outmask)
        print cmdf
        subp.Popen(cmdf,shell=True)
        
        ###track the fibers
        cmdprob = "probtrackx2 --samples=%s --mask=%s --seed=%s --targetmasks=%s\
                   --omatrix2 --target2=%s --nsamples=%s --xfm=%s\
                   --opd --forcedir --dir=%s --pd"%(merged,mask,roi_f,brain_seg,\
                   brain_seg,sample,xfm_file1,sess_dir)
        print cmdprob
        subp.Popen(cmdprob,shell=True)
        
        ###transform the dot matrix to hdf5 format.
        dot2hdf5(dot_f,roi_f,hdf)
        
        ###segmentation 
        spectral_seg(hdf,parcel_f)

        cmdmerg = "cp %s %s"%(fdt_res,fdt_sub)
        print cmdmerg
        os.system(cmdmerg)

        ###transform the segmentation result into standard space 
        cmdfr = "flirt -in %s  -ref %s -interp nearestneighbour\
                -applyxfm -init %s -out %s"%(fdt_sub,st_brain,xfm_file2,fdt_sub.replace('fdt_result','fdt_stand'))
        print cmdfr
        subp.Popen(cmdfr,shell=True)

        print "Processing end..."

def read_sess_list(sess):
    """docstring for fname"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess

def dot2hdf5(filename,maskfile,outputf):
    """
    load dot file
    """
    print "load data:"    
    data = np.loadtxt(filename)
    
    print "load mask:"
    seed_set = fmri_dataset(samples=maskfile,mask=maskfile)
    seed = seed_set.copy(sa=[])

    print seed

    sparse_set = csc_matrix((data[:,2],(data[:,0]-1,data[:,1]-1)))
    seed.samples = sparse_set.T.todense()
    print seed.samples.shape
    seed.save(outputf)


if __name__=='__main__':
    main()
