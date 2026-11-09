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
mpl.rcParams['figure.figsize'] = figW, 2*figH+0.5
nullfmt = NullFormatter()

plasma=plt.get_cmap("plasma")

conv1= 0.0299792458

bases0=[['InAs',
       'In1.0Ga0.0As0.75Sb0.25',
       'In1.0Ga0.0As0.625Sb0.375',
       'In1.0Ga0.0As0.5Sb0.5',
       'In1.0Ga0.0As0.375Sb0.625',
       'In1.0Ga0.0As0.25Sb0.75',
       'InSb'],
       ['GaAs',
       'In0.25Ga0.75As1.0Sb0.0',
       'In0.375Ga0.625As1.0Sb0.0',
       'In0.5Ga0.5As1.0Sb0.0',
       'In0.625Ga0.375As1.0Sb0.0',
       'In0.75Ga0.25As1.0Sb0.0',
       'InAs']]
bases1=[['InAs','In1.0Ga0.0As0.5Sb0.5','InSb'],
       ['GaAs','In0.5Ga0.5As1.0Sb0.0','InAs']]
names1=[['InAs',r'InAs$_{0.5}$Sb$_{0.5}$','InSb'],
        ['GaAs',r'In$_{0.5}$Ga$_{0.5}$As','InAs']]
names1=['(a) In(As$_{1-x}$Sb$_x$)','(b) Ga(As$_{1-x}$Sb$_x$)']
points=np.array([0,20,40,60,80])
xlabels=['L',r'$\Gamma$','X','K',r'$\Gamma$']
base='In1.0Ga0.0As0.5Sb0.5'
plasma=plt.get_cmap("plasma")
conv1=0.0299792458   # cm^-1 -> THz

bases1=[['InAs','In1.0Ga0.0As0.5Sb0.5','InSb'],
        ['GaAs','In0.5Ga0.5As1.0Sb0.0','InAs']]
labels1=[['InAs',r'InAs$_{0.5}$Sb$_{0.5}$','InSb'],
         ['GaAs',r'In$_{0.5}$Ga$_{0.5}$As','InAs']]
panels1=[r'(a) InAs$_{1-x}$Sb$_x$', r'(b) In$_x$Ga$_{1-x}$As']

points=np.array([0,20,40,60,80])
xlabels=['L',r'$\Gamma$','X','K',r'$\Gamma$']

fig,ax = plt.subplots(2,1,sharex=True)
ymax=0
for i,bases in enumerate(bases1):
    for j,base in enumerate(bases):
        data=np.loadtxt(f'../data/freq_files_offset/{base}.freq.gp')
        c=plasma(0.85*j/(len(bases)-1))       # avoid the pale-yellow end of plasma
        for k in range(1,data.shape[1]):
            ax[i].plot(data[:,0],data[:,k]*conv1,c=c,lw=1.0,
                       label=labels1[i][j] if k==1 else None)
        ymax=max(ymax,(data[:,1:]*conv1).max())

for i,a in enumerate(ax):
    a.set_xlim(data[points[0],0],data[points[-1],0])
    a.set_ylim(0,ymax*1.25)                   # headroom for legend/label, same on both panels
    a.set_ylabel('Frequency (THz)')
    for p in points[1:-1]:
        a.axvline(data[p,0],c='k',ls='--',lw=0.5,zorder=0)
    a.text(0.02,0.95,panels1[i],transform=a.transAxes,ha='left',va='top')
    a.legend(loc='upper right',ncol=3,fontsize=7,frameon=False,
             handlelength=1.2,columnspacing=1.0)

ax[1].set_xticks(data[points,0])
ax[1].set_xticklabels(xlabels)
ax[1].set_xlabel('Wave vector')
fig.subplots_adjust(hspace=0.08)
fig.savefig('../Figures/all.freq.offset.pdf')
plt.close('all')

xs=[0.0,0.25,0.375,0.5,0.625,0.75,1.0]
fig,ax = plt.subplots(2,1)
LOsb,TOsb=[],[]
TO_minus_LOsb=[]
for i,bases in enumerate(bases0):
    LOs,TOs,TO_minus_LOs=[],[],[]
    for j,base in enumerate(bases):
        file=f'../data/freq_files_offset/{base}.freq.gp'
        data=np.loadtxt(file)
        LO=data[20,4]*conv1
        TO=data[20,6]*conv1
        LOs.append(LO)
        TOs.append(TO)
        TO_minus_LOs.append(TO-LO)
    LOsb.append(LOs)
    TOsb.append(TOs)
    TO_minus_LOsb.append(TO_minus_LOs)

for i in range(2):
    ax[i].scatter(xs,LOsb[i],c='r',label='LO')
    ax[i].scatter(xs,TOsb[i],c='b',label='TO')
    ax[i].set_xlabel('Sb fraction')
    ax[i].set_ylabel('Frequency (THz)')
    ax[i].set_xlim(0,1)
    ax[i].legend()
    ax[i].text(0.05,0.9,names1[i],transform=ax[i].transAxes)
fig.savefig(f'../Figures/all.LO-TO.offset.pdf')
plt.close('all')

fig,ax = plt.subplots(2,1)
ax[0].scatter(xs,TO_minus_LOsb[0],c='k',label='TO-LO')
ax[1].scatter(xs,TO_minus_LOsb[1],c='k',label='TO-LO')
ax[1].set_xlabel('x')
for i in range(2):
    
    ax[i].set_ylabel(r'f$_{\mathrm{TO}}$-f$_{\mathrm{LO}}$ (THz)')
    ax[i].set_xlim(0,1)
    ax[i].text(0.05,0.075,names1[i],transform=ax[i].transAxes)
fig.savefig(f'../Figures/all.TO-LO.diff.offset.pdf')
plt.close(fig)
LOsb,TOsb,diffsb=[],[],[]
for bases in bases0:
    LOs,TOs,diffs=[],[],[]
    for base in bases:
        data=np.loadtxt(f'../data/freq_files_offset/{base}.freq.gp')
        TO=data[20,4]*conv1   # branches sorted ascending: 4-5 TO, 6 LO
        LO=data[20,6]*conv1
        LOs.append(LO); TOs.append(TO); diffs.append(LO-TO)
    LOsb.append(LOs); TOsb.append(TOs); diffsb.append(diffs)
from matplotlib.lines import Line2D

alloys=[r'In(As$_{1-x}$Sb$_x$)', r'(In$_x$Ga$_{1-x}$)As']
markers=['o','^']

fig,ax = plt.subplots(2,1,sharex=True)

for i in range(2):
    m=markers[i]
    # top: absolute values
    ax[0].plot(xs,LOsb[i],c='r',marker=m,ls=':',lw=0.8)
    ax[0].plot(xs,TOsb[i],c='b',marker=m,ls=':',lw=0.8)
    # bottom: differences
    ax[1].plot(xs,diffsb[i],c='k',marker=m,ls=':',lw=0.8)

ax[0].set_ylabel('Frequency (THz)')
ax[1].set_ylabel(r'$f_{\mathrm{LO}}-f_{\mathrm{TO}}$ (THz)')
ax[1].set_xlabel(r'$x$')
ax[1].set_xlim(0,1)

ax[0].text(0.02,0.9,'(a)',transform=ax[0].transAxes)
ax[1].text(0.02,0.9,'(b)',transform=ax[1].transAxes)

# top legend: quantity (colour) + alloy (marker)
h_top=[Line2D([],[],c='r',ls='',marker='s',label='LO'),
       Line2D([],[],c='b',ls='',marker='s',label='TO'),
       Line2D([],[],c='gray',ls='',marker='o',label=alloys[0]),
       Line2D([],[],c='gray',ls='',marker='^',label=alloys[1])]
ax[0].legend(handles=h_top,fontsize=6,loc='upper right')

# bottom legend: alloy only
h_bot=[Line2D([],[],c='k',ls='',marker='o',label=alloys[0]),
       Line2D([],[],c='k',ls='',marker='^',label=alloys[1])]
ax[1].legend(handles=h_bot,fontsize=6,loc='upper right')

fig.subplots_adjust(hspace=0.08)
fig.savefig('../Figures/all.LO-TO.combined.offset.pdf')
plt.close('all')