#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:42:34 2026

@author: A.Gentles
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.ticker import NullFormatter
import pandas as pd

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
mpl.rcParams['figure.figsize'] = figW, 2*figH+0.5 #2*figH+0.5
nullfmt = NullFormatter()

plasma=plt.get_cmap("plasma")

bases_noalloy = [
    ["GaAs","In0.25Ga0.75As1.0Sb0.0","In0.375Ga0.625As1.0Sb0.0","In0.5Ga0.5As1.0Sb0.0","In0.625Ga0.375As1.0Sb0.0","In0.75Ga0.25As1.0Sb0.0","InAs"],
    ["InAs","In1.0Ga0.0As0.75Sb0.25","In1.0Ga0.0As0.625Sb0.375","In1.0Ga0.0As0.5Sb0.5","In1.0Ga0.0As0.375Sb0.625","In1.0Ga0.0As0.25Sb0.75","InSb"]
    ]

bases_alloy = [
    ["In0.25Ga0.75As1.0Sb0.0","In0.375Ga0.625As1.0Sb0.0","In0.5Ga0.5As1.0Sb0.0","In0.625Ga0.375As1.0Sb0.0","In0.75Ga0.25As1.0Sb0.0"],
    ["In1.0Ga0.0As0.75Sb0.25","In1.0Ga0.0As0.625Sb0.375","In1.0Ga0.0As0.5Sb0.5","In1.0Ga0.0As0.375Sb0.625","In1.0Ga0.0As0.25Sb0.75"]
    ]
factor=1/1000
factor=1
names = [r"(a) (In$_{x}$Ga$_{1-x}$)As, T = 300 K", r"(b) In(As$_{1-x}$Sb$_{x}$), T = 300 K"]
sheets = ['InGaAs','InAsSb']
markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']
fig,axes=plt.subplots(2,1)
for i in range(2):
    df = pd.read_excel('../data/mobility_exp.xlsx', sheet_name=sheets[i])#, engine='xlsx')
    marker_dict = {name: markers[i % len(markers)] for i, name in enumerate(df['Name'].unique())}
    groups = df.groupby(['Year', 'Name'])
    for (year, name), group in groups:
        axes[i].scatter(group['x'], group['mu'],
                   label=f'{name}, {year}',
                   marker=marker_dict.get(name, 'o'),
                   s=20, alpha=0.8)
    mu_alloy_data,mu_noalloy_data = [],[]
    xs_alloy, xs_noalloy = np.array([0.25,0.375,0.5,0.625,0.75]) , np.array([0.0, 0.25,0.375,0.5,0.625,0.75,1.0])
    for base in bases_noalloy[i]:
        data=np.loadtxt(f"../data/total_mobs/{base}.noalloy.mob.dat")
        print(data[np.logical_and(data[:,0]<311 , data[:,0]>290 )])
        mu_noalloy_data.append(data[np.logical_and(data[:,0]<311 , data[:,0]>290 )][0,3])
    for base in bases_alloy[i]:
        data=np.loadtxt(f"../data/total_mobs/{base}.alloy.mob.dat")
        mu_alloy_data.append(data[np.logical_and(data[:,0]<311 , data[:,0]>290 )][0,3])
    mu_alloy_data=np.array(mu_alloy_data)
    mu_noalloy_data=np.array(mu_noalloy_data)
    axes[i].scatter(xs_noalloy,mu_noalloy_data*factor,color = 'black',label='eph')
    axes[i].scatter(xs_alloy,mu_alloy_data*factor,color = 'red',label='eph + Alloy')
    axes[i].text(0.05,0.95,names[i],transform=axes[i].transAxes,va='top')
    axes[i].set_ylabel(r"$\mu$ (10$^3$ cm$^2$/Vs)")
    #axes[i].legend(loc='lower right')
    axes[i].set_ylim(0, axes[i].get_ylim()[1])
    #axes[i].set_yscale('log')
axes[1].set_xlabel( '$x$' )
fig.savefig('../Figures/all_mobs_300K.pdf',format='pdf')
plt.close(fig)
lims=[0,0]
#lims=[-10000,-10000]
fig,axes=plt.subplots(2,1)
for i in range(2):
    df = pd.read_excel('../data/mobility_exp.xlsx', sheet_name=sheets[i])#, engine='xlsx')
    marker_dict = {name: markers[i % len(markers)] for i, name in enumerate(df['Name'].unique())}
    groups = df.groupby(['Year', 'Name'])
    for (year, name), group in groups:
        axes[i].scatter(group['x'], group['mu'],
                   label=f'{name}, {year}',
                   marker=marker_dict.get(name, 'o'),
                   s=20, alpha=0.8)
    mu_alloy_data,mu_noalloy_data = [],[]
    xs_alloy, xs_noalloy = np.array([0.25,0.375,0.5,0.625,0.75]) , np.array([0.0, 0.25,0.375,0.5,0.625,0.75,1.0])
    for base in bases_noalloy[i]:
        data=np.loadtxt(f"../data/total_mobs_offset/{base}.noalloy.mob.dat")
        print(data[np.logical_and(data[:,0]<311 , data[:,0]>290 )])
        mu_noalloy_data.append(data[np.logical_and(data[:,0]<311 , data[:,0]>290 )][0,3])
    for base in bases_alloy[i]:
        data=np.loadtxt(f"../data/total_mobs_offset/{base}.alloy.mob.dat")
        mu_alloy_data.append(data[np.logical_and(data[:,0]<311 , data[:,0]>290 )][0,3])
    mu_alloy_data=np.array(mu_alloy_data)
    mu_noalloy_data=np.array(mu_noalloy_data)
    axes[i].scatter(xs_noalloy,mu_noalloy_data*factor,color = 'black',label='eph')
    axes[i].scatter(xs_alloy,mu_alloy_data*factor,color = 'red',label='eph + Alloy')
    axes[i].text(0.05,0.10,names[i],transform=axes[i].transAxes,va='top')
    axes[i].set_ylabel(r"$\mu$ (cm$^2$/Vs)")
    axes[i].legend(loc='upper left')
    axes[i].set_ylim(lims[i], axes[i].get_ylim()[1])
    #axes[i].set_yscale('log')
axes[1].set_xlabel( '$x$' )
#fig.savefig('../Figures/all_mobs_300K_offset.pdf',format='pdf')
