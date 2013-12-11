#!usr/bin/env python 
#-*- coding: utf-8 -*-
##############################################################################
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
##############################################################################

import os
import sys
import numpy as np
import nibabel as nib


def main():
    image = raw_input(">>>")
    coorf = raw_input(">>>")
    img = nib.load(image)
    imdata = img.get_data()
    coors = read_coor_list(coorf)
    print coors
    num = 1000
    for coor in coors:
        print coor
        region_img, size = region_growing(imdata,coor,0.5,num,6)
        print "size:",size
        region_img += region_img
    result = img
    result._data = region_img
    nib.save(result,image.replace('.nii.gz','_rg.nii.gz'))

def read_coor_list(file):
    coorf  = open(file,'r')
    coors = coorf.readlines()
    clist = [line.rstrip('\n').split() for line in coors]
    clist = [[int(i) for i in line] for line in clist]
    return clist

def region_growing(image,coordinate,threshold=0.2,number=1000,neighbors=6):
    """
    Give a coordinate, return a logical region in 3D space.

    The region is iteratively grown by comparing all unallocated
    neighbor voxels to the region mean signal value. The voxel with
    the smallest difference is allocated to the respective region.

    This process stops when the intensity difference become larger
    than a certain distance threshold.

    Parameters
    ----------
    image : input 3D image.
    coordinate : the position of the seedpoint.
    threshold : maximum intensity distance threshold(default=0.2).
                distance between the voxel signal and current region mean signal.
    number : voxel number threshold(default=200).
    neighbors : neighbors exploration rule(default=6,six neighbors).

    Returns
    -------
    region : the result logical region in 3D space.
    region_size   : the voxel number the region contains.
    
    Reference
    ---------
    Thank for Author: D. Kroon, University of Twente's matlab code implementation.
    This code is rewriten with python.
    """
    #init the parameters
    t = threshold
    num = number
    neigh = neighbors

    tmp_image = np.zeros_like(image)
    image_shape = image.shape

    x = coordinate[0]
    y = coordinate[1]
    z = coordinate[2]

    if num > (image_shape[0]*image_shape[1]*image_shape[2]):
        print "The voxel number contrain is too big."
        return False

    #if x y z is not inside of the image, return False.
    inside = (x>=0)and(x<image_shape[0])and(y>=0)and\
             (y<image_shape[1])and(z>=0)and(z<image_shape[2])
    if inside!=True:
        print "The coordinate is out of the image range."
        return False

    region_mean = image[x,y,z]
    region_size = 0
    voxel_distance = 0.0

    neighbor_free = 10000
    neighbor_pos = -1
    neighbor_list = np.zeros((neighbor_free,4))

    neighbors  = [[1,0,0],\
                 [-1,0,0],\
                 [0,1,0],\
                 [0,-1,0],\
                 [0,0,-1],\
                 [0,0,1],\
                 [1,1,0],\
                 [1,1,1],\
                 [1,1,-1],\
                 [0,1,1],\
                 [-1,1,1],\
                 [1,0,1],\
                 [1,-1,1],\
                 [-1,-1,0],\
                 [-1,-1,-1],\
                 [-1,-1,1],\
                 [0,-1,-1],\
                 [1,-1,-1],\
                 [-1,0,-1],\
                 [-1,1,-1],\
                 [0,1,-1],\
                 [0,-1,1],\
                 [1,0,-1],\
                 [1,-1,0],\
                 [-1,0,1],\
                 [-1,1,0]]

    #region growing code block.
    while (voxel_distance < t) and (region_size < num):
        #mark the x y z voxel as result region voxel under the condition.
        tmp_image[x,y,z] = 2
        region_size += 1
        
        #add neighbor voxel to neighbor_list, if it is not checked before.
        for i in range(neigh):
            xn = x + neighbors[i][0]
            yn = y + neighbors[i][1]
            zn = z + neighbors[i][2]
            
            #check coordinate is inside the image.
            inside = (xn>=0)and(xn<image_shape[0])and(yn>=0)and\
                    (yn<image_shape[1])and(zn>=0)and(zn<image_shape[2])

            #print "xn,yn,zn:",xn,yn,zn,"inside:",inside
        
            if inside and tmp_image[xn,yn,zn]==0:
                neighbor_pos = neighbor_pos+1
                neighbor_list[neighbor_pos] = [xn,yn,zn,image[xn,yn,zn]]
                tmp_image[xn,yn,zn] = 1
        
        #print neighbor_list[:neighbor_pos+1]
        
        #add new list space for free neighbor voxel.
        if (neighbor_pos+100 > neighbor_free):
            neighbor_free +=10000
            new_list = np.zeros((10000,4))
            #print "new_list",new_list,new_list.shape
            neighbor_list = np.vstack((neighbor_list,new_list))
            #print "new neigh list:",neighbor_list,neighbor_list.shape
        
        #choose the min distance voxel
        distance = np.abs(neighbor_list[:neighbor_pos+1,3] - np.tile(region_mean,neighbor_pos+1))
        #print "distance list:",distance,distance.shape
        voxel_distance = distance.min()
        index = distance.argmin()
        #print "distance , index :",voxel_distance,index 
        
        #get the new voxel coordinate.
        x = neighbor_list[index][0]
        y = neighbor_list[index][1]
        z = neighbor_list[index][2]
        
        #re-calculate the region mean signal.
        region_mean = (region_mean*region_size+neighbor_list[index,3])/(region_size+1)
        #print "new x,y,z:",x,y,z,region_mean
        
        #delete the voxel in the unallocated neighbor list.
        neighbor_list[index] = neighbor_list[neighbor_pos]
        neighbor_pos -= 1
    
    #get the logical result region.Choose the voxels that value=2.
    region = tmp_image>1

    return region,region_size

if __name__=="__main__":
    main()
