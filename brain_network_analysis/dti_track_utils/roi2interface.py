#!/usr/bin/env python

"""
register roi to wm
"""
import os
import sys
import numpy as np
import nibabel as nib
import time
import argparse
import subprocess as subp

def main():
    """
    """
    parser = argparse.ArgumentParser(description = 'Transform Node Mask to white and grey matter interface.')
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
    '''
    parser.add_argument('-dr',
                        dest = 'resultdir',
                        required = True,
                        help = 'Data directory.'
                        )
    '''
    args = parser.parse_args()
    
    sessf = args.sess
    sess  = read_sess_list(sessf)
    data_dir = args.datadir
    AAL = os.path.join(data_dir,'AAL_Contract_90_2MM.nii.gz')
    #two transform matrix.
    xfm_s2b = 'standard2brain.mat'
    xfm_b2d = 'brain2diffusion.mat'
    
    for sid in sess:
        print '+++++++++%s+++++++++++++++++++++\n'%sid
        st = time.time()
        #1.flirt the AAL template to subject brain.
        sess_dir = os.path.join(data_dir,sid+'_vol')
        brain_s = os.path.join(sess_dir,'brain.nii.gz')
        aal = os.path.join(sess_dir,sid+'_AAL.nii.gz')
        s2b = os.path.join(sess_dir,xfm_s2b)
        b2d = os.path.join(sess_dir,xfm_b2d)

        cmd1 = 'flirt -in %s -ref %s -applyxfm -interp nearestneighbour -init %s -out %s'%(AAL,brain_s,s2b,aal)
        print cmd1
        os.system(cmd1)
        #2.registration 
        ### method : find the nearest white matter for every grey matter voxel.
        
        img = nib.load(aal)
        wm_mask = os.path.join(sess_dir,sid+'_seg_2.nii.gz')
        wm  = nib.load(wm_mask)
        out_img = os.path.join(sess_dir,sid+'_aal_wgb.nii.gz')

        roi_f = os.path.join(data_dir,sid+'.txt')
        roi_log = open(roi_f,'w')
        fline = "%10s    %10s\n"%('AAL_ID','WGI Size')
        roi_log.write(fline)
        roi_to_wm(img,wm,out_img,roi_log)
        roi_log.close()
        print "Write the log to %s"%roi_f

        #3.flirt the AAL WGB mask to diffusion space.
        nodif = os.path.join(sess_dir,'nodif_brain.nii.gz')
        aal_dif = os.path.join(sess_dir,sid+'_aal_diff.nii.gz')
        cmd2 = 'flirt -dof 6 -in %s -ref %s -applyxfm -interp nearestneighbour -init %s -out %s'%(out_img,nodif,b2d,aal_dif)
        print cmd2
        #os.system(cmd2)
        subp.Popen(cmd2,shell=True)
        print '%s process time: %s'%(sid,time.time()-st)
    #coor = np.array([2,2,2])
    #shape = [100,100,100]
    #radius 
    #get_neighbors_surface(coor,radius,shape)
    #get_neighbors(coor,radius,shape)
    print "End!"
    return True

def read_sess_list(sess):
    """read session list file,the file content subject IDs"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess

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



def roi_to_wm(img,brain_wm,outf,log):
    """
    Transform the functional roi to wm.
    Algorithm: find the nearest wm voxel.
    """
        
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
    neighbors = np.array(neighbors)

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
        print roi_id,indexs.shape[0]
        for coor in indexs:
           # print coor
            x = coor[0]
            y = coor[1]
            z = coor[2]
    
            if wmdata[x,y,z]==1:
                pass
                #result_mask[x,y,z] = roi_id
            else:
                #find the nearest neighbor.
                flag = False
                radius = 1
                mindist_voxel = []
                mindist = 1000
                while radius<2:
                    #neigh_list = get_neighbors(coor,radius,shape)
                    neigh_list = [(ioff+coor).tolist() for ioff in neighbors]
                    #print neigh_list
                    radius += 1
                    #find the nearest white matter voxel.
                    for n in neigh_list:
                        if wmdata[n[0],n[1],n[2]]==1:
                            flag = True
                        #    dist = np.sqrt((n[0]-x)**2+(n[1]-y)**2+(n[2]-z)**2)
                            # if the distance is smaller than tag, choose it to be nearest.
                        #    if dist < mindist:
                         #       mindist = dist
                         #       mindist_voxel = n
                    
                            result_mask[n[0],n[1],n[2]] = roi_id
                    if flag:
                        break
                #print mindist_voxel
                #if mindist_voxel!=[]:
                #    result_mask[mindist_voxel[0],mindist_voxel[1],mindist_voxel[2]] = roi_id 
    
    for roi_id in roi:
        tmp_mask = result_mask==roi_id
        roi_size = tmp_mask.sum()    
        #print roi_id, roi_size
        line = "%10s    %10s\n"%(roi_id,roi_size)
        log.write(line)
    result = img
    result._data = result_mask
    nib.save(result,outf)
    print "Save the wgm all to %s"%outf
    return True


if __name__=="__main__":
    main()
