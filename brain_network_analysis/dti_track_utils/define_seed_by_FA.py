#!/usr/bin/env python
# coding: utf-8
#filename:mkroi.py
"""
Make facenet whiter matter ROIs for single subject.
"""
import os
import sys
import csv
import nibabel as nib
import numpy as np
import argparse
import subprocess as subp
import time
import scipy.ndimage.morphology  as morph
import operator


def main():
    """
    Make ROI pipeline.
    """
    parser = argparse.ArgumentParser(description = 'Define AAL Seed ROIs for single subject.')
    #parser.add_argument('-s',
    #                    dest = 'sess',
    #                    required = True,
    #                    help = 'Input session list file.'
    #                    )
    #parser.add_argument('-t',
    #                    dest = 'template',
    #                    required = True,
    #                    help = 'Input template file.'
    #                    )
    #parser.add_argument('-f',
    #                    dest = 'fa',
    #                    required = True,
    #                    help = 'Input FA map file.'
    #                    )
    parser.add_argument('-d',
                        dest = 'datadir',
                        required = True,
                        help = 'Data working directory.'
                        )
    #parser.add_argument('-w',
    #                    dest = 'wm',
    #                    required = True,
    #                    help = 'Out put white matter seed file.'
    #                    )
    #parser.add_argument('-g',
    #                    dest = 'gm',
    #                    required = True,
    #                    help = 'Output grey matter seed file.'
    #                    )
    parser.add_argument('-thr',
                        dest = 'thresh',
                        type = float,
                        default = 0.2,
                        help = 'Threshold for white matter and grey matter.eg:0.2'
                        )
    parser.add_argument('-v',
                        dest = 'verb',
                        default = False,
                        help = 'Output print information.(Fasle,True)'
                        )
    
    args = parser.parse_args()
    
    verb = args.verb
    #sessfile = args.sess
   
    #sess = read_sess_list(sessfile,verb)
    #template = args.template
    #FA_map = args.fa
    threshold = args.thresh
    #wm_o    = args.wm
    #gm_o    = args.gm
    data_dir = args.datadir 
    
    st = time.time()
    subnum = 20
    sess = range(1,subnum+1)
    sess = ["%05d"%i for i in sess]
    print sess
  
    for sid in sess:
        print "++++++++S%s+++++++++++++++++++++++"%sid
        #/nfs/j3/userhome/dangxiaobin/workingdir/network/result/00001/native_space
        sess_dir = os.path.join(data_dir, sid)
        fa_file = "bdt_%s_FA.nii.gz"%sid
        fa_aal_file = "bdt_%s_FA_Parcellated_AAL_Contract_90_2MM.nii.gz"%sid
        fa_aal_w = "bdt_%s_FA_Parcellated_AAL_Contract_90_2MM_WM.nii.gz"%sid
        fa_aal_g = "bdt_%s_FA_Parcellated_AAL_Contract_90_2MM_GM.nii.gz"%sid
        native_dir = os.path.join(sess_dir,"native_space")
        fa  = os.path.join(native_dir,fa_file)
        fa_aal = os.path.join(native_dir,fa_aal_file)
        aal_wm_o = os.path.join(native_dir,fa_aal_w)
        aal_gm_o = os.path.join(native_dir,fa_aal_g)
        log_file = os.path.join(data_dir,'S%s_roi_size.txt'%sid)
        print fa
        print fa_aal
        #load images
        fa_img = nib.load(fa)
        fa_aal_img = nib.load(fa_aal)

        roi_log = open(log_file,'w')
        fline = "%10s    %10s    %10s    %10s    %10s\n"%('AAL_ID','RawSize','WM Size','GM Size','Other')
        roi_log.write(fline)
                
        roi_generator(fa_img,fa_aal_img,0.2,0.01,aal_wm_o,\
                      aal_gm_o,1,1,roi_log,verb)
        roi_log.close()
        print "Save subject roi size file to %s"%log_file
        
    print 'Total process time: %s'%(time.time()-st)
    return True

def roi_generator(fa_img,fa_aal_img,threshold,threshold_l,fa_aal_w,fa_aal_g,wm_o,gm_o,roi_log,v):
    """
    Using whiter matter and Harvard mask to generat roi whiter matter mask.
    """
    fa_data = fa_img.get_data()
    data = fa_aal_img.get_data()
    thr = threshold
    thr_l = threshold_l
    #print data.shape
    #print maskdata.shape
    #print wmdata.shape

    roi_ids = np.unique(data)
    id = roi_ids[1:]
    id = [int(i) for i in id]
    if v:
        print "\nRoi Id: %s"%id
    tmproi = np.zeros(data.shape)
    gmbuff = np.zeros(data.shape)
    wmbuff = np.zeros(data.shape)

    #make single roi
    for i in id:

        tmproi = data== i
        indexs = np.transpose(tmproi.nonzero())
        
        #print indexs
        size_w = 0
        size_g = 0
        for coor in indexs:
            x = coor[0]
            y = coor[1]
            z = coor[2]
            
            if fa_data[x,y,z]>=thr:
                wmbuff[x,y,z] = i
                size_w += 1
            elif fa_data[x,y,z]<thr and fa_data[x,y,z]>thr_l:
                gmbuff[x,y,z] = i
                size_g += 1 
            else:
                pass

        
        rsize = len(indexs)
        if v:   
            print 'ID: %s  RS: %s  WMS: %s  GMS: %s W/G: %s'%(i,rsize,size_w,size_g,(rsize-size_w-size_g))
            
        log_line = "%10s    %10s    %10s    %10s    %10s\n"%(i,rsize,size_w,size_g,(rsize-size_w-size_g))
        roi_log.write(log_line)
        
    if wm_o:
        resultimg = wmbuff    
        result = fa_img
        result._data = resultimg
        nib.save(result, fa_aal_w)
        print "Save white matter file to %s"%fa_aal_w
    if gm_o:
        resultimg = gmbuff    
        result = fa_img
        result._data = resultimg
        nib.save(result, fa_aal_g)
        print "Save grey matter file to %s"%fa_aal_g
            
    return True

def read_sess_list(sess,v):
    """read session list."""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    if v:
        print 'Session id : %s'%sess
    return sess

if __name__=='__main__':
    main()
