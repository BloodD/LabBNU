#!/usr/bin/env python
import os
import sys
import numpy as np


def main():
    print """This script can create session list DTI tracking.
             1.session_all: contains all 2008 subjects DTI folders
             2.sess_cur: contains subjects that have been done.
             3.sess_new : contains subjects will be process.
             4.sess_nn .....

          """
    sfa = raw_input("all sess file:>>>")
    sfc = raw_input("current sess file:>>>")
    sess_all = read_sess_list(sfa)
    sess_cur = read_sess_list(sfc)
    N = raw_input("How many to create:???")
    
    sess_out = open('./sess_new','w')
    sess_new = []
    i = 0
    while i < int(N):
        new = np.random.randint(1,389)
        new = "S1%03d"%new
        print new
        if (new in sess_all) and (new not in sess_cur):
            sess_new.append(new)
            i = i +1

    print len(sess_new)

    for s in sess_new:
        sess_out.write(s+'\n')

    print "create sess list has done..."

                
def read_sess_list(sess):
    """docstring for fname"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess
if __name__=='__main__':
    main()
