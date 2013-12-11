# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os 
import sys
import time as time
import numpy as np
import scipy as sp
import pylab as pl
import pdb 
#sys.path.remove('/usr/local/neurosoft/epd-7.2.1/lib/python2.7/site-packages/nibabel-1.3.0-py2.7.egg')
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from mvpa2.base.hdf5 import *
from mvpa2.datasets.base import Dataset
from mvpa2.datasets.mri import map2nifti

from sklearn.feature_extraction import image
from sklearn.cluster.spectral import SpectralClustering
from pylab import imread, imshow, gray, mean
from scipy.spatial import distance as ds
import nibabel as ni

print ni.__version__

def create_mask(conn,thn):
    mat = conn 
    shape = mat.shape
    n = shape[0]
    m = shape[1]
    mask = np.zeros(n)
    mask = np.array([False for x in mask])
    #print mask.shape,n,m,th,thn
    
    i = 0
    while i < n:
        tmp = np.array(mat[i])
        #print mat[i]
        nzero = tmp.nonzero()
        non = np.array(nzero)
        num = non.size
        if  num >= thn:
            mask[i]= True
        i+= 1
    return  mask

def mask_feature(mat,mask):
    map = mat 
    mak = mask
    map = map[mak]
    return map

def is_inside(v, shape):
 
    return ((v[0] >= 0) & (v[0] < shape[0]) &
            (v[1] >= 0) & (v[1] < shape[1]) &
            (v[2] >= 0) & (v[2] < shape[2]))
    
def get_neighbors(coor,radius,shape):
    
    neighbors = []
    offsets = []
   
    for x in np.arange(-radius, radius + 1):
        for y in np.arange(-radius, radius + 1):
            for z in np.arange(-radius, radius + 1):
                offsets.append([x,y,z])
    
    offsets = np.array(offsets)
    #print offsets
    
    for offset in offsets:
        
        if offset.tolist() == [0,0,0]:
            #print offset.tolist()
            continue
           
        tmp_neigh = coor + offset
        #print tmp_neigh
        inside = is_inside(tmp_neigh,shape)
        if inside :
            neighbors.append([tmp_neigh[0],tmp_neigh[1],tmp_neigh[2]])
    
    #neighbors = np.array(neighbors)

    #print neighbors.shape
    return neighbors   

# Apply spectral clustering (this step goes much faster if you have pyamg installed)

def spectral_seg(hfilename,outf):
    '''
    Spectral clustering...
    '''
    tmpset = Dataset([])
    #pdb.set_trace()
    print "hdf name:",hfilename
    st =  time.time()
    ###1.load connectivity profile of seed mask voxels
    conn = h5load(hfilename)
    tmpset.a = conn.a
    print "connection matrix shape:"
    print conn.shape
    ###2.features select
    mask = create_mask(conn.samples,5)
    conn_m = conn.samples[mask]
    map = conn_m.T
    print "masked conn matrix:"
    print map.shape,map.max(),map.min()
    
    ###3.average the connection profile.
    temp = np.zeros(map.shape)
    voxel = np.array(conn.fa.values())
    v = voxel[0]
    v = v.tolist()
    
    shape = [256,256,256]
    
    i = 0
    for coor in v:
        mean_f = map[i]
        #print mean_f.shape
        #plt.plot(mean_f)
        #plt.show()
        
        neigh =get_neighbors(coor,2,shape)
        #print "neigh:",neigh

        count = 1
        for n in neigh:
            if n in v:
               mean_f = (mean_f*count + map[v.index(n)])/(count+1)
               count+=1

        temp[i] = mean_f
        i+=1
    #sys.exit(0)
    map = temp
    print "average connection matrix"
    
    ###4.spacial distance
    spacedist = ds.cdist(v,v,'euclidean') 
    #print spacedist
    
    ###5.correlation matrix
    corr = np.corrcoef(map)
    corr = np.abs(corr)
    
    ###6.mix similariry matrix.
    corr = 0.7*corr + 0.3/(spacedist+1)
    #plt.imshow(corr,interpolation='nearest',cmap=cm.jet)
    #cb = plt.colorbar() 
    #pl.xticks(())
    #pl.yticks(())
    #pl.show()
    print "mix up the corr and spacial matrix"
    
    #sys.exit(0)
    ###7.spectral segmentation    
    print "do segmentation"
    cnum = 3
    near = 100
    sc = SpectralClustering(cnum,'arpack',None,100,1,'precomputed',near,None,True)
    sc.fit_predict(corr)
    
    tmpset.samples = sc.labels_+1
    print "Number of voxels: ", sc.labels_.size
    print "Number  of clusters: ", np.unique(sc.labels_).size
    print "Elapsed time: ", time.time() - st
    
    ###8.save the segmentation result.
    print "save the result to xxx_parcel.nii.gz"
    result = map2nifti(tmpset)
    result.to_filename(outf)
    print ".....Segment end........"
    
    return True
