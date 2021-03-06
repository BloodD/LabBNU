#!usr/bin/env python 
#-*- coding: utf-8 -*-
##############################################################################
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
##############################################################################

"""
register roi to WGMI
"""
import os
import sys
import numpy as np
import nibabel as nib
import time

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
    
    neighbors = np.array(neighbors)
    #print neighbors.shape
    return neighbors   

def get_neighbors_surface(coor,radius,shape):
    neigh = []
    if radius <=0:
        return neigh
    elif radius ==1:
        return get_neighbors(coor,radius,shape)
    else:
        neighbors_all = get_neighbors(coor,radius,shape)
        neighbors_in = get_neighbors(coor,radius-1,shape)
        neigh = [i for i in neighbors_all.tolist() if i not in neighbors_in.tolist()]
       # print neighbors_all
       # print neighbors_in
        #print np.array(neigh).shape
        return neigh

def roi_to_wm(img,brain_wm,nth):
    """
    Transform the functional roi to wm.
    Algorithm: find the nearest wm voxel.
    """
    
    data = img.get_data()
    wmdata = brain_wm.get_data()
    shape = data.shape

    roi_ids = np.unique(data)
    roi = roi_ids[1:]
    roi = [int(i) for i in roi]
    print roi
    
    wmdata = wmdata!=0
    result_mask = np.zeros(data.shape)
    #print wmdata   
    
    #First, get the nonzero voxel index in image data.
    #Here image data is a label volume.
    #ROIs is in it
    for roi_id in roi:
        #print roi_id
        tmp_mask = data==roi_id
        #print tmp_mask
        indexs = np.transpose(tmp_mask.nonzero())
        #print indexs
        
        #Second, find the nearest wm voxel for each indexs.
        print indexs.shape
        for coor in indexs:
            #print coor
            x = coor[0]
            y = coor[1]
            z = coor[2]
    
            if wmdata[x,y,z]==1:
                result_mask[x,y,z] = roi_id
            else:
                #find the nearest neighbor.
                flag = False
                radius = 1
                mindist_voxel = []
                mindist = 1000     
                while radius<100:      
                    neigh_list = get_neighbors(coor,radius,shape)
                    radius += 1
                    #find the nearest white matter voxel.
                    for n in neigh_list:
                        #print n
                        if wmdata[n[0],n[1],n[2]]==1:
                            flag = True
                            dist = np.sqrt((n[0]-x)**2+(n[1]-y)**2+(n[2]-z)**2)
                            # if the distance is smaller than tag, choose it to be nearest.
                            
                            if dist < mindist:
                                mindist = dist
                                mindist_voxel = n
                            
                    if flag:
                        break
                #print mindist_voxel
                if mindist_voxel!=[]:
                    result_mask[mindist_voxel[0],mindist_voxel[1],mindist_voxel[2]] = roi_id 
    for roi_id in roi:
        tmp_mask = result_mask==roi_id
        roi_size = tmp_mask.sum()    
        print roi_id, roi_size
    result = img
    result._data = result_mask
    #roi_name = os.path.join(mkdir,'roi_%s.nii.gz'%i)
    nib.save(result,"test_regroi.nii.gz")
    
    return True

def main():
    """
    Test 
    """
    img = nib.load("facenet.nii.gz")
    wm = nib.load("brain_seg_2.nii.gz")
    st = time.time()
    roi_to_wm(img,wm,10)
    print 'Total process time: %s'%(time.time()-st)
    #coor = np.array([2,2,2])
    #shape = [100,100,100]
    #radius = 2
    #get_neighbors_surface(coor,radius,shape)
    #get_neighbors(coor,radius,shape)
    print "End!"
    return True

if __name__=="__main__":
    main()
