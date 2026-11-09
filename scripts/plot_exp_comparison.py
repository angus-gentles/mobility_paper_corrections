#!/usr/bin/env python3

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
mpl.rcParams['figure.figsize'] = figW, 2*figH+0.5 #2*figH+0.5
nullfmt = NullFormatter()

plasma=plt.get_cmap("plasma")

bases=['In0.5Ga0.5As1.0Sb0.0','In1.0Ga0.0As0.5625Sb0.4375']
names=[r'(a) (In$_{0.5}$Ga$_{0.5}$)As',r'(b) In(As$_{0.56}$Sb$_{0.44}$)']
ref_names={'In0.5Ga0.5As1.0Sb0.0':['Kim, 2025', 'Oliver, 1981'],'In1.0Ga0.0As0.5625Sb0.4375': ['Svennson, 2019']}
exp_filenames=["InGaAs_x-0.53.mobility.csv","InAsSb_x-0.44.mobility.csv"]
exp_filenames={'In0.5Ga0.5As1.0Sb0.0':["InGaAs_x-0.53.mobility.csv","In0.53Ga0.47As_mob_oliver1981.csv"],'In1.0Ga0.0As0.5625Sb0.4375': ["InAsSb_x-0.44.mobility.csv"]}
#fig2,ax2 = plt.subplots(1,1)
factors=[2.25,1.38]
fig1,ax = plt.subplots(2,1)
for i in (0,1):
    base=bases[i]
    exp_data=[]
    exp_colours=[]
    for j,exp_filename in enumerate(exp_filenames[base]):
        exp_data.append(np.loadtxt(f"../data/mobility_files/{exp_filename}", delimiter=",", skiprows=1))
        if i==0:
            exp_data[j][:,1]*=10**3
        exp_colour=0.2 + 0.1*j
        exp_colours.append(exp_colour)
    alloy_data=np.loadtxt(f"../data/mobility_files/{bases[i]}.perturbo_exp_alloy.mob.dat",skiprows=1)
    noalloy_data=np.loadtxt(f"../data/mobility_files/{bases[i]}.perturbo_exp_noalloy.mob.dat",skiprows=1)
    
    percentage=alloy_data[:,1]/noalloy_data[:,1]   * 100
    #ax2.scatter(alloy_data[:,0],percentage,label=names[i])
    for j,exp_datum in enumerate(exp_data):
        ax[i].scatter(exp_datum[:,0], exp_datum[:,1], marker='o', label=ref_names[base][j],color=plasma(exp_colours[j]))
    ax[i].scatter(noalloy_data[:,0], noalloy_data[:,3], marker='^', label='eph',color=plasma(0.5))
    #if i==0:
    ax[i].scatter(noalloy_data[:,0], noalloy_data[:,3]*factors[i], marker='v', label='eph corrected',color=plasma(0.4))
    ax[i].scatter(alloy_data[:,0], alloy_data[:,3], marker='s', label='eph + Alloy',color=plasma(0.8))
    ax[i].text(0.05,0.1,names[i],transform=ax[i].transAxes,va='top')
    ax[i].set_xlim(20,350)
    ax[i].set_xlabel(r'T (K)')
    ax[i].set_ylabel(r'$\mu$ (cm$^2$/Vs)')
    ax[i].set_yscale('log')
    ax[i].legend()
#ax2.legend()

#fig1.savefig("../Figures/mobility_plot.1.pdf",format='pdf')

