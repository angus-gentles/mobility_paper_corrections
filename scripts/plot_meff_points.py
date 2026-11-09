#!/usr/bin/env python3

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

file1='../data/test_VCA/In1.0Ga0.0As0.5Sb0.5.test.hdf5'
meff_exp=0.011
UA,UB,meffs=[],[],[]
with h5py.File(file1,'r') as f:
    for key in f.keys():
        if f[key].attrs['Eg']>0:
            UA.append(f[key].attrs['UA'])
            UB.append(f[key].attrs['UB'])
            meffs.append(f[key].attrs['meff'])

tol = 0.002
UA = np.array(UA)
UB = np.array(UB)
meffs = np.array(meffs)

mask = np.abs(meffs - meff_exp) <= tol

# Plot all other points with colormap
sc = plt.scatter(
    UA[~mask], UB[~mask],
    c=meffs[~mask],
    cmap='plasma',
    s=2
)

# Highlight matching points in red
plt.scatter(
    UA[mask], UB[mask],
    color='red',
    s=2,
    label=rf'$|m_e^* - {meff_exp:.3f}| \leq {tol:.3f}$'
)

plt.colorbar(sc, label=r'$m_e^*$')
plt.xlabel(r'$U_p^\mathrm{A}$ (eV)')
plt.ylabel(r'$U_p^\mathrm{B}$ (eV)')
plt.legend()

plt.savefig(
    "../Figures/In1.0Ga0.0As0.5Sb0.5_meff_spline.pdf",
    format='pdf'
)
