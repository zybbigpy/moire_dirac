#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:10:58 2023

@author: wmiao
"""

import numpy as np

# start cal berry information
def _wf_dpr(wf1,wf2):

    return np.dot(wf1.flatten().conjugate(),wf2.flatten())


def _one_berry_loop(wf, berry_evals=False):
    
    # wf shape should be [nk, band_component, band_index]
    # number of occupied state
    nocc=wf.shape[2]
    # temporary matrices
    prd=np.identity(nocc,dtype=complex)
    ovr=np.zeros([nocc,nocc],dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    for i in range(wf.shape[0]-1):
        # generate overlap matrix, go over all bands
        for j in range(nocc):
            for k in range(nocc):
                ovr[j,k]=_wf_dpr(wf[i,:,j],wf[i+1,:,k])
        # only find Berry phase
        if berry_evals==False:
            # multiply overlap matrices
            prd=np.dot(prd,ovr)
        # also find phases of individual eigenvalues
        else:
            # cleanup matrices with SVD then take product
            #matU,sing,matV=np.linalg.svd(ovr)
            #prd=np.dot(prd,np.dot(matU,matV))
            prd=np.dot(prd,ovr)
    # calculate Berry phase
    if berry_evals==False:
        det=np.linalg.det(prd)
        pha=(-1.0)*np.angle(det)
        return pha
    # calculate phases of all eigenvalues
    else:
        evals=np.linalg.eigvals(prd)
        eval_pha=(-1.0)*np.angle(evals)
        eval_pha=np.sort(eval_pha)
        #print(eval_pha)
        return eval_pha


def no_2pi(x,clos):
    """Make x as close to clos by adding or removing 2pi"""
    while abs(clos-x)>np.pi:
        if clos-x>np.pi:
            x+=2.0*np.pi
        elif clos-x<-1.0*np.pi:
            x-=2.0*np.pi
    return x


def _one_phase_cont(pha, clos):

    ret=np.copy(pha)

    for i in range(len(ret)):
        # which number to compare to
        if i==0: 
            cmpr=clos
        else: 
            cmpr=ret[i-1]
        # make sure there are no 2pi jumps
        ret[i]=no_2pi(ret[i],cmpr)
    return ret


def cal_wilson_loop(wfs, berry_evals=False):

    # wfs shape [nk, nk, band_component, band_index]
    # wfs should be imposed pbc
    nk = wfs.shape[0]
    ret=[]
    
    for i in range(nk):
        wf_use=wfs[i,:,:,:]
        ret.append(_one_berry_loop(wf_use,berry_evals))
    
    #ret=_one_phase_cont(ret,ret[0])

    return np.array(ret)


def one_flux_plane(wfs2d):
    """
    Compute berry curv of 2d system.
    """

    # wfs2d shape [nk, nk, band_component, band_index]
    # wfs2d should be imposed pbc
    nk0=wfs2d.shape[0]
    nk1=wfs2d.shape[1]
    # number of bands (will compute flux of all bands taken together)
    nbnd=wfs2d.shape[3]

    # store berry curv on the mesh
    all_phases=np.zeros((nk0-1, nk1-1, nbnd), dtype=float)

    # go over all plaquettes
    for i in range(nk0-1):
        for j in range(nk1-1):
            # generate a small loop made out of four pieces
            wf_use=[]
            wf_use.append(wfs2d[i,j])
            wf_use.append(wfs2d[i+1,j])
            wf_use.append(wfs2d[i+1,j+1])
            wf_use.append(wfs2d[i,j+1])
            wf_use.append(wfs2d[i,j])
            wf_use=np.array(wf_use, dtype=complex)

            all_phases[i,j,:]=_one_berry_loop(wf_use, berry_evals=True)
    
    #print("check", all_phases.shape)
    return all_phases