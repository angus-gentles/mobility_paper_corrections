#!/usr/bin/env python3

import numpy as np
import h5py
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
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
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
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
#~ mpl.rcParams['figure.autolayout']= True
nullfmt = NullFormatter()

def make_mask(ks,bins, weights, best_eigs):
    distances_2=ks
    mask=np.zeros((len(distances_2),len(bins)),dtype=bool)
    print(mask.shape,weights.shape)
    for i, eigs in enumerate(best_eigs[i] for i in range(len(distances_2))):
        # Digitize eigenvalues into bin indices
        inds = np.digitize(eigs, bins) - 1  # subtract 1 to make 0-based
        # Keep only valid indices
        inds = inds[(inds >= 0) & (inds < mask.shape[1])]
        # Mark those bins as True for this k-point
        mask[i, inds] = True
    mask_int = mask.astype(int)
    #B_masked = np.ma.masked_where(mask_int.T == 0, B[:, :, 2].T)
    #y_vals = bins
    #ymin, ymax = -0.5, 13.0
    #y_mask = (y_vals >= ymin) & (y_vals <= ymax)

    # Apply along axis=0 since shape is (601, 191)
    #sum_in_bounds = np.sum(B_masked[y_mask, :])
    return mask_int

x,y=1.0,0.5

r=(x,y)
base=f"In{x}Ga{1-x}As{1-y}Sb{y}_cube_2x2x2"
base0=f"In{x}Ga{1-x}As{1-y}Sb{y}"
results_file=f"{base0}.best_VCA_U.hdf5"
filename=f"../data/VCA_data/{base0}.VCA_data.hdf5"
print(base0)
weightsfile=f"../data/total_EBS/{base0}_cube_2x2x2.total_EBS.hdf5"

fig,ax=plt.subplots(figsize=(figW,figH))

with h5py.File(weightsfile,'r') as fweights:
    weights=fweights['zs'][:]
    ks=fweights['xs'][:]
    Es=fweights['ys'][:]
#ax.pcolormesh(ks,Es,weights.T*this_mask[:,None],cmap='viridis',shading='auto')
with h5py.File(filename,'r') as f:
    group=f['UA_-1.9499999999999997_UB_3.25']
    ks1=group['k_points'][:]
    Es1=group['eigenvalues'][:]
    Es1-=Es1[:,14].min()
mask_int=make_mask(ks,Es,weights,Es1)
print(mask_int.shape,weights.shape)
masked_vals = np.ma.masked_where(mask_int.T == 0, mask_int.T)
print(masked_vals.shape)
#exit(0)
ax.pcolormesh(ks,Es,weights.T,cmap='plasma',shading='auto')
ax.pcolormesh(ks,Es,masked_vals,cmap='Greys',shading='auto')
ax.set_xticks([ks[0],ks[90],ks[190]], labels=[r"$L$", r"$\Gamma$",r"$X$"])
ax.set_ylim(-3,3)
ax.set_xlabel(r"$\bm{k}$")
ax.set_ylabel(r"$E-E_{\mathrm{cbm}}$ (eV)")
fig.tight_layout()
fig.savefig(f"../Figures/{base0}_EBS.pdf")
