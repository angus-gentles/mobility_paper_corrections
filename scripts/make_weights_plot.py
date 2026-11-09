#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:54:44 2026

@author: A.Gentles
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
#import cmocean

column_width2=7.1413
column_width=3.48761

figW=column_width
figH=3/4.*column_width

mpl.rc('font', **{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)
font={'size':9}
plt.rc('font', **font)
mpl.rcParams['text.usetex']=True
mpl.rcParams['lines.linewidth']=1.5
mpl.rcParams['axes.linewidth']=1.0
mpl.rcParams['lines.markersize']=5
mpl.rcParams['lines.markeredgewidth']=0.2
mpl.rcParams['axes.labelsize']=8
mpl.rcParams['axes.labelweight']='normal'
mpl.rcParams['xtick.major.size']=3
mpl.rcParams['xtick.major.width']=1
mpl.rcParams['ytick.major.size']=3
mpl.rcParams['ytick.major.width']=1
#~ mpl.rcParams['font.family']='serif'
mpl.rcParams['font.weight']='medium'
mpl.rcParams['font.size']=9
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['savefig.bbox']='tight'
mpl.rcParams['ytick.labelsize']=8
mpl.rcParams['xtick.labelsize']=8
mpl.rcParams['figure.figsize'] = figW, figH
nullfmt = NullFormatter()


hbar=1.0545718e-34
e= 1.602176634e-19
a_GaAs=5.6533
a_GaSb=6.0959
a_InAs=6.0584
a_InSb=6.4794

def interpolate_exp_meff(x, y):
    # Binary endpoint electron effective masses (m_e units)
    m_GaAs = 0.067
    m_InAs = 0.026
    m_GaSb = 0.039
    m_InSb = 0.0135

    # Ternary bowing parameters (Donati et al.)
    b_GaAsSb = -0.0
    b_InAsSb = -0.035
    b_InGaSb = -0.0092
    b_InGaAs = -0.0091

    # Linear quaternary interpolation
    m_lin = (
        m_GaAs * (1 - x) * (1 - y)
        + m_InAs * x * (1 - y)
        + m_GaSb * (1 - x) * y
        + m_InSb * x * y
    )

    # Ternary bowing corrections
    m_bow = (
        b_InGaAs * x * (1 - x) * (1 - y)
        + b_InGaSb * x * (1 - x) * y
        + b_GaAsSb * (1 - x) * y * (1 - y)
        + b_InAsSb * x * y * (1 - y)
    )

    return m_lin + m_bow

def make_gaussian_mask(Es, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((Es - mu) / sigma)**2)

def make_main_mask(UAs,UBs,Egs,meffs,weights,meff_exp,meff_threshold=0.0005):
    # Create a mask for non-zero bandgap
    valid_mask = (Egs > 0) & (np.abs(meffs - meff_exp) < meff_threshold) & (UAs<0)

    # If no valid entries exist, handle it explicitly
    if not np.any(valid_mask):
        return None  # or raise an exception

    # Apply mask to weights
    valid_weights = weights[valid_mask]

    # Find best index within the filtered array
    best_valid_idx = np.argmax(valid_weights)

    # Map back to original indices
    original_indices = np.where(valid_mask)[0]
    best_idx = original_indices[best_valid_idx]
    print(UAs[best_idx],UBs[best_idx])

    return valid_mask

def calculate_spectral_weights(weightsfile,VCAfile,meff_exp):
    these_weights=[]
    all_masks=[]
    with h5py.File(weightsfile, 'r') as fweight:
        weights=fweight['zs'][:]
        ks=fweight['xs'][:]
        Es=fweight['ys'][:]
        with h5py.File(VCAfile,'r') as fcur:
            keys=list(fcur.keys())
            UAs=[]
            UBs=[]
            Egs=[]
            meffs=[]
            for i,key in enumerate(keys):
                eigs=fcur[key]['eigenvalues'][:]
                UAs.append(fcur[key].attrs['UA'])
                UBs.append(fcur[key].attrs['UB'])
                Egs.append(fcur[key].attrs['Eg'])
                meffs.append(fcur[key].attrs['meff'])
                cbm=eigs[:,14].min()
                eigs-=cbm
                mask=np.zeros((len(ks),len(Es)),dtype=bool)
                for j, eig in enumerate(eigs):
                    inds = np.digitize(eig, Es) - 1  # subtract 1 to make 0-based
                    inds = inds[(inds >= 0) & (inds < mask.shape[1])]
                    mask[j, inds] = True
                mask_int = mask.astype(int)
                B_masked = np.ma.masked_where(mask_int.T == 0, weights.T)
                this_mask=make_gaussian_mask(Es,mu=0.0,sigma=1.0)
                #print(B_masked.shape)
                B_masked*=this_mask[:,None]
                ymin,ymax=-3,2
                y_mask = (Es >= ymin) & (Es <= ymax)
                sum_in_bounds = np.sum(B_masked[y_mask, :])
                all_masks.append(mask_int[:, y_mask])
                these_weights.append(sum_in_bounds)
    UAs=np.array(UAs)
    UBs=np.array(UBs)
    Egs=np.array(Egs)
    meffs=np.array(meffs)
    these_weights=np.array(these_weights)
    main_mask=make_main_mask(UAs,UBs,Egs,meffs,these_weights,meff_exp)
    return all_masks,y_mask,UAs,UBs,these_weights,main_mask


Us_start=[]
x, y = (1.0,0.5)
base=f"In{x}Ga{1-x}As{1-y}Sb{y}"
h5file=f"../data/test_VCA/{base}.test.hdf5"
weightsfile=f"../data/total_EBS/{base}_cube_2x2x2.total_EBS.hdf5"
meff_exp=interpolate_exp_meff(x,y)



all_masks,y_mask,UAs,UBs,these_weights,main_mask=calculate_spectral_weights(weightsfile,h5file,meff_exp)
sc=plt.scatter(UAs[main_mask],UBs[main_mask],c=these_weights[main_mask],cmap='plasma',s=2)
plt.xlabel(r"$U^\mathrm{A}_p$ (eV)")
plt.ylabel(r"$U^\mathrm{B}_p$ (eV)")
plt.colorbar(sc, label=r'$\zeta$')
plt.savefig("../Figures/weights_vs_Us.pdf",format='pdf')
plt.close()

