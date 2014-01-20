#!/usr/bin/env python
"""
=======================================================================
Random forest trainning and test module.

Input : train raw samples within trainning list.
Output : a forest classifier.

TODO:
    1. Cross-splite the train sess list for combining the sample parts
       trainning part/validation part
    2. feature scaling?
    3. for trainning k clf and scoring
    4. average the scores (Dice)
       clf evaluation. write the parameter log and scores
    5. save the clf.

    manual change the parameter for grid search.
=======================================================================
"""

import numpy as np
import pylab as pl
import nibabel as nib
import time 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split,KFold
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model, datasets
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.metrics import roc_curve, auc, roc_auc_score
import  matplotlib.pyplot as plt

def read_sess_list(sess):
    """
    Load subject Id list.
    Tested
    """
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\r\n') for line in sess]
    sess = np.array(sess)
    print sess
    return sess

def combin_sample(sess,f):
    sam = np.array([])
    flag = 0
    for i in sess:
        if f ==0:
            se = np.loadtxt('repo/train/sample_%s'%i)
        else:
            se = np.loadtxt('repo/test/sample_%s'%i)
        
        if flag==0:
            sam = se
            flag = 1
        else:
            sam= np.vstack((sam,se))
    return sam

def main():

    st = time.time()
    #read train part sess list
    train = read_sess_list('./train_split.sess')
    #read test part sess list
    test = read_sess_list('./test_split.sess')
    #read mask indexs 
    coor = np.loadtxt('./coordinates')
    print coor,coor.shape
    #rean img template
    img = nib.load('MNI_brain.nii.gz')
    data = img.get_data()
    
    TR_sample = combin_sample(train,0)
    #TE_sample = combin_sample(test,1)
    print TR_sample.shape
    #print TE_sample.shape
    
    #print TR_sample[0]
    #print TR_sample[4000]
    mask_t = TR_sample[:,3]>=2.3
    #mask_c = TE_sample[:,3]>=2.3
    TR = TR_sample[mask_t]
    #TE = TE_sample
    print TR.shape
    #print TE.shape

    X = TR[:,:50]
    y = TR[:,50]
    #X_c = TE[:,:23]
    #y_c = TE[:,23]
    #print X.shape,y.shape,X_c.shape,y_c.shape
    
    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(10)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_train = (X - mean)/std
    #X_test = (X_c - mean)/std
    y_train = y
    #y_test = y_c
    
    #training model and save model.
    clf = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth=36, 
                                 max_features='sqrt',min_samples_split=8,
                                 n_estimators=200, n_jobs=50, 
                                 oob_score=True)
    clf.fit(X_train,y_train)
    print "fi:",clf.feature_importances_
    print "oob:",clf.oob_score_
    #print "mean accuracy:",clf.score(X_test, y_test)
    
    #y_p  = clf.predict(X_test)
    #y_pp = clf.predict_proba(X_test)
    for i in range(41):
        print i
        sub = np.loadtxt("repo/test/sample_%s"%test[i])
        print sub.shape
        mask = sub[:,3] >= 2.3
        sub_md = sub[mask]
        X_t = sub_md[:,:50]
        y_t = sub_md[:,50]
        co = coor[mask]

        tmp = np.zeros_like(data)
        y_p = clf.predict(X_t)

        di = dice(y_t,y_p)
        print di
        print np.sum(y_p)

        for j,c in enumerate(co):
            tmp[tuple(c)] = y_p[j]
        img._data = tmp
        nib.save(img,"predicted_%s.nii.gz"%test[i])
        
    '''
    ra = roc_auc_score(y_test,y_p)
    print "auc score:",ra
    di = dice(y_test,y_p)
    print di
    fpr, tpr, thresholds = roc_curve(y_test, y_pp[:, 1],pos_label=1)
    plt.plot(fpr, tpr, 'b-', label='rf')
    '''
    '''
    #LR
    clf = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.00001, C=1, fit_intercept=True, intercept_scaling=1, class_weight='auto')
    clf.fit(X_train,y_train)
    #print "fi:",clf.feature_importances_
    #print "oob:",clf.oob_score_
    #print "param:",clf.get_params(True)

    print "mean accuracy:",clf.score(X_test, y_test)
    y_p  = clf.predict(X_test)
    y_pp = clf.predict_proba(X_test)
    print y_pp
    ra = roc_auc_score(y_test, y_p)
    print "auc score:",ra
    di = dice(y_test,y_p)
    print di

    fpr, tpr, thresholds = roc_curve(y_test, y_pp[:, 1],pos_label=1)
    print "time used:%s"%(time.time()-st)
    plt.plot(fpr, tpr, 'r-', label='lr')
    
    
    #SVC
    clf = svm.SVC(kernel='linear',C=1,class_weight='auto',tol=0.0001,probability=True,degree=3,max_iter=-1)
    clf.fit(X_train,y_train)
    #print "fi:",clf.feature_importances_
    #print "oob:",clf.oob_score_
    #print "param:",clf.get_params(True)
    print "mean accuracy:",clf.score(X_test, y_test)
    y_p  = clf.predict(X_test)
    y_pp = clf.predict_proba(X_test)
    print y_pp.shape
    fpr, tpr, thresholds = roc_curve(y_test, y_pp[:, 1],pos_label=1)
    print "time used:%s"%(time.time()-st)
    plt.plot(fpr, tpr, 'y-', label='svc')
    '''
    plt.legend()
    plt.show()
    '''
    
    s1 = 'precision' 
    s2 = 'recall'
    s3 = 'f1'

    s1 = 'precision' 
    s2 = 'recall'
    s3 = 'f1'
    s4 = 'accuracy'
    s5 = 'roc_auc'
    
    param = [{'C':[0.001,0.01,0.1,1,20,100]}]
    print("# parameters for %s" %s1)
    model = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None)
    clf = GridSearchCV(model, param, cv=3, n_jobs=1, scoring=s1)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s2)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1, scoring=s2)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s3)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1 ,scoring=s3)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s4)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1,scoring=s4)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("# parameters for %s" %s5)
    clf = GridSearchCV(model, param, cv=10,n_jobs=1, scoring=s5)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")      
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print("Detailed classification report:")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, clf.predict(X_test)

    print "time used:%s"%(time.time()-st)
    '''
    return 
def dice(r,p):
    return 2*np.sum(r*p)/(np.sum(r)+np.sum(p))

if __name__=="__main__":
    main()
