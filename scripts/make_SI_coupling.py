#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:12:55 2026

@author: A.Gentles
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.ticker import NullFormatter

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
mpl.rcParams['figure.figsize'] = figW, 2*figH+1.0
nullfmt = NullFormatter()

plasma=plt.get_cmap("plasma")

def linear_func(x, m, b):
    return m * x + b

def function(ks, M_inf, dM, k1, p, A, C, k0, n, ks_turn, m):
    k1_vec = ks[:, 0, :]
    k2_vec = ks[:, 1, :]

    dots = np.sum(k1_vec * k2_vec, axis=1)
    norm1 = np.linalg.norm(k1_vec, axis=1)
    norm2 = np.linalg.norm(k2_vec, axis=1)

    cos_theta = np.divide(dots, norm1 * norm2,
                          out=np.ones_like(dots),
                          where=(norm1 * norm2) > 1e-14)

    k_mag = norm1

    # isotropic baseline: rises from (M_inf - dM) at k=0 to asymptote M_inf
    M0 = M_inf - dM / (1.0 + (k_mag / k1)**p)

    # radial strength of angular term
    f = A / (1.0 + (k_mag / k0)**n) + C

    # angular turn-on
    turn_on = 1.0 - np.exp(-(k_mag / ks_turn)**m)

    return M0 + f * turn_on * (1.0 - cos_theta)

# -------------------------------------------------
# Compute cos(theta)
# -------------------------------------------------
def compute_cos_theta(k1, k2):
    return np.sum(k1 * k2, axis=1) / (
        np.linalg.norm(k1, axis=1) * np.linalg.norm(k2, axis=1)
    )


def plot(ax,k1,k2,cp,colour):
    #dots=np.dot(k1, k2)
    dots=[]
    for i in range(k1.shape[0]):
        dot=np.dot(k1[i], k2[i])
        dots.append(dot)
    dots=np.array(dots)
    params, covar = curve_fit(linear_func, dots/(np.linalg.norm(k1, axis=1)*np.linalg.norm(k2, axis=1)), cp)
    #angles=np.arccos(dots/(np.linalg.norm(k1, axis=1)*np.linalg.norm(k2, axis=1)))
    ax.scatter(dots/(np.linalg.norm(k1, axis=1)*np.linalg.norm(k2, axis=1)), cp, s=1, color=colour)
    ax.text(0.05, 0.95, f"m={params[0]:.3e}, b={params[1]:.3e}", transform=ax.transAxes, fontsize=8, verticalalignment='top')
    

def plot_ax(base,title,ax):
    with h5py.File(f"../data/{base}.coupling_elements.SI.hdf5", "r") as f:
        k1 = f["k1"][:]
        k2 = f["k2"][:]
        el_np = f["coupling_elements_np"][:]
    
    ks=np.zeros((len(k1),2,3))
    ks[:,0,:]=k1
    ks[:,1,:]=k2
    kmag = np.linalg.norm(k1, axis=1)
    cos_all = compute_cos_theta(k1, k2)
    fwd = cos_all > 0.99
    if fwd.sum() < 4:                      # fallback if few forward points
        fwd = cos_all > np.percentile(cos_all, 90)
    
    hi = fwd & (kmag >= np.percentile(kmag[fwd], 70))
    lo = fwd & (kmag <= np.percentile(kmag[fwd], 30))
    M_inf0 = np.mean(el_np[hi])
    dM0 = M_inf0 - np.mean(el_np[lo])      # drop the max(..., 1e-3); negative for a falling baseline
    kmed = np.median(kmag)
    
    p0 = [M_inf0, dM0, 0.5*kmed, 2.0,          # M_inf, dM, k1, p
          np.max(el_np), np.min(el_np), kmed, 2.0,   # A, C, k0, n
          0.5*kmed, 2.0]                        # ks_turn, m
    
    #lower = [-np.inf, -np.inf, 1e-6, 0.3, -np.inf, -np.inf, 1e-6, 0.3, 1e-6, 0.3]
    #upper = [ np.inf,  0.0,    np.inf, 8.0,  np.inf,  np.inf, np.inf, 8.0, np.inf, 8.0]

    #lower = [-np.inf, -np.inf, 1e-6, 0.3, -np.inf, -np.inf, 1e-6, 0.3, 1e-6, 0.3]
    #upper = [ np.inf,  0.0,    np.inf, 8.0,  np.inf,  np.inf, np.inf, 8.0, np.inf, 8.0]
    params, covar = curve_fit(function, ks, el_np, p0=p0)#, max_nfev=20000)
    print(params)
    colours=plasma(0.5*(np.sum(k1*k2,axis=1)/np.linalg.norm(k1, axis=1)/np.linalg.norm(k2, axis=1))+0.5)
    ax.scatter(np.linalg.norm(k1, axis=1), el_np, s=1, c=colours)
    
    k2_test0=np.zeros((200,2,3))
    k2_test0[:,0,0]=np.linspace(0.00001,0.1,200)
    k2_test0[:,1,0]=np.linspace(0.00001,0.1,200)
    
    k2_test1=np.zeros((200,2,3))
    k2_test1[:,0,0]=np.linspace(0.00001,0.1,200)
    k2_test1[:,1,1]=np.linspace(0.00001,0.1,200)
    
    k2_test2=np.zeros((200,2,3))
    k2_test2[:,0,0]=np.linspace(0.00001,0.1,200)
    k2_test2[:,1,0]=-np.linspace(0.00001,0.1,200)
    
    ax.plot(np.linspace(0.00001,0.1,200), function(k2_test0, *params), color="black", linestyle="-", label=r"fit, $\theta =0$")
    ax.plot(np.linspace(0.00001,0.1,200), function(k2_test1, *params), color="black", linestyle=":", label=r"fit, $\theta = \pi/2$")
    ax.plot(np.linspace(0.00001,0.1,200), function(k2_test2, *params), color="black", linestyle="--", label=r"fit, $\theta = \pi$")
    ax.legend()
    ax.set_ylabel(r"$M(k,\theta )$ (eV)")
    if base=="In1.0Ga0.0As0.5Sb0.5":
        ax.set_ylim(0.3,5.0)
    else:
        ax.set_xlabel("$k$ ($2\pi/a_0$)")
    ax.text(
            0.05, 0.1, title,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1
            )
        )
    
bases=["In1.0Ga0.0As0.5Sb0.5","In0.5Ga0.5As1.0Sb0.0"]
titles=[r"(a) In(As$_{0.5}$Sb$_{0.5})$",r"(b) (In$_{0.5}$Ga$_{0.5}$)As"]
fig,axes=plt.subplots(2)
for i,ax in enumerate(axes):
    base,title=bases[i],titles[i]
    plot_ax(base,title,ax)
fig.savefig("../Figures/coupling_elements.pdf",format='pdf')
#plt.close(fig)



