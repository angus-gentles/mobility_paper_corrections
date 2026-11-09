#!/usr/bin/env python3
import pandas as pd
import numpy as np



import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
#import cmocean

import h5py
from scipy.optimize import curve_fit
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "cm"  # Use Computer Modern (LaTeX-style)
#os.environ["PATH"] = "/home/gentles/tools/textlive/2025/bin/x86_64-linux:" + os.environ["PATH"]
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
#~ mpl.rcParams['figure.autolayout']= True
nullfmt = NullFormatter()

def bowing_curve(y, A, B, c):
    return A * y + B * (1 - y) + c * y * (1 - y)
ternaries = ['Ga_AsSb', 'In_AsSb', 'InGa_As', 'InGa_Sb']
# Define Donati bowing parameters
donati_bowing = {
    'GaAsSb': np.array([0.039, 0.067, -0.0]),
    'InAsSb': np.array([0.0135, 0.026, -0.035]),
    'InGaSb': np.array([0.0135, 0.039, -0.0092]),
    'InGaAs': np.array([0.026, 0.067, -0.0091])
}

names = {'GaAsSb': r"(a) Ga(As$_{1-x}$Sb$_{x}$)",
    'InAsSb': r"(b) In(As$_{1-x}$Sb$_x$)",
    'InGaSb': r"(c) (In$_x$Ga$_{1-x}$)Sb",
    'InGaAs': r"(d) (In$_x$Ga$_{1-x}$)As"}

fig, axarr = plt.subplots(2, 2, figsize=(column_width2, column_width2*4/5))
fig.subplots_adjust(hspace=0.15, wspace=0.15)

ternaries = ['GaAsSb', 'InAsSb', 'InGaAs', 'InGaSb']

for idx, ternary in enumerate(ternaries):
    df = pd.read_excel('../meff_bowing_exp.ods', sheet_name=ternary, engine='odf')
    
    row, col = divmod(idx, 2)
    ax = axarr[row, col]
    ax.grid(True, ls=':')

    groups = df.groupby(['Year', 'Name'])
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']
    marker_dict = {name: markers[i % len(markers)] for i, name in enumerate(df['Name'].unique())}

    for (year, name), group in groups:
        ax.scatter(group['x'], group['m_eff'],
                   label=f'{name}, {year}',
                   marker=marker_dict.get(name, 'o'),
                   s=20, alpha=0.8)

    # Plot Donati curve
    x = np.linspace(0, 1, 100)
    params = donati_bowing[ternary]
    y = bowing_curve(x, *params)
    ax.plot(x, y, label='Vurgaftman', color='black', linewidth=2)

    ax.text(0.04, 0.1, names[ternary], transform=ax.transAxes,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='black'))
    '''
    if ternary in ['InGaAs', 'InGaSb']:
        ax.set_xlabel('x', fontsize=10)
    else:
        ax.set_xlabel('y', fontsize=10)
    '''
    if col==0:
        ax.set_ylabel(r'$m^*_e$ ($m_e$)', fontsize=10)
    if row==1:
        ax.set_xlabel('x', fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(loc='best', fontsize=7)
    #if ternary=='InAsSb':
    #    print('here')
    #    yticks = np.arange(0.1, 0.51, 0.1)
    #    ax.set_yticks(yticks)
        
        # Optional: Add some visual padding around the y-axis range
    #    ax.set_ylim(0.05, 0.55)
    

# Save full figure
fig.savefig('../Figures/meff_exp.pdf', dpi=300, format='pdf', bbox_inches='tight')
#plt.close('all')

