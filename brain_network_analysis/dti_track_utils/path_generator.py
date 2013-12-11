#!/usr/bin/env python
import os
import sys

def read_sess_list(sess):
    """docstring for fname"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess

def main():
    print """This script can create file paths for Panda.
             You can copy and paste the string into panda Add path windows.
             
             Auther: dangxiaobin@gmail.com
             Data  : 2013.11.15
           -----------------tags table----------------------------------------------------------------------------
           Tag: 1. raw session            S1001
                2. T1 file                S1001_vol/brain.nii.gz
                3. native space folder    00001/native_space
                31.native space bedpostx  00001/native_space.bedpostX
                4. FA.nii.gz map          00001/native_space/bdt_00001_FA.nii.gz
                5. parcelated AAL(WGM)    00001/native_space/bdt_00001_FA_Parcellated_AAL_Contract_90_2MM.nii.gz
                6. parcelated AAL(GM)     00001/native_space/bdt_00001_FA_Parcellated_AAL_Contract_90_2MM_GM.nii.gz
                7. parcelated AAL(WM)     00001/native_space/bdt_00001_FA_Parcellated_AAL_Contract_90_2MM_WM.nii.gz
                8. registration AAL       00001/native_space/S1009_ALL_WM_R.nii.gz
          -------------------------------------------------------------------------------------------------------\n
          """
    
    f = raw_input("sess file:>>>")
    sess =read_sess_list(f)
    slen = len(sess)
    offset = raw_input("Id offset:>>>")
    i = int(offset)
    j = i+slen
    tag = raw_input("tag:>>>")
    tag = int(tag)
    print "tag:", tag
    
    #tag: 1. Diffusion data  .../S1001
    #     2. T1              .../S1001_vol/brain.nii.gz
    #     3. native          .../00001/native_space
    #     4. parcelled labels  .../00001/native_space/....nii.gz

    if tag == 1:
        for sid in sess:
            ddata = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Data_30/%s"%sid
            print ddata
    elif tag == 2:
        for sid in sess:
            brainT="/nfs/j3/userhome/dangxiaobin/workingdir/network/Data_30/%s_vol/brain.nii.gz"%sid
            print brainT
    elif tag == 3:
        for sid in range(i,j):
            native = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Result_20/%05d/native_space"%sid
            print native
    elif tag == 31:
        for sid in range(i,j):
            native = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Result_20/%05d/native_space.bedpostX"%sid
            print native
    elif tag == 4:
        for sid in range(i,j):
            pac_label = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Result_20/%05d/native_space/bdt_%05d_FA.nii.gz"%(sid,sid)
            print pac_label
    elif tag == 5:
        for sid in range(i,j):
            pac_label = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Data_30/result/%05d/native_space/bdt_%05d_FA_Parcellated_AAL_Contract_90_2MM.nii.gz"%(sid,sid)
            print pac_label
    elif tag == 6:
        for sid in range(i,j):
            pac_label = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Data_30/result/%05d/native_space/bdt_%05d_FA_Parcellated_AAL_Contract_90_2MM_GM"%(sid,sid)
            print pac_label
    elif tag == 7:
        for sid in range(i,j):
            pac_label = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Result_20/%05d/native_space/bdt_%05d_FA_Parcellated_AAL_Contract_90_2MM_WM.nii.gz"%(sid,sid)
            print pac_label
    elif tag == 8:
        for sid in range(i,j):
            pac_label = "/nfs/j3/userhome/dangxiaobin/workingdir/network/Result_20/%05d/native_space/%05d_AAL_WM_R.nii.gz"%(sid,sid)
            print pac_label



if __name__=="__main__":
    main()
