#!/usr/bin/env python3
"""Plot phonon DOS from <base>.<i>.total_dos.dat files and their mean, per base.

Run in the folder holding the files:  python plot_dos.py
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
mpl.rcParams['figure.figsize'] = figW, figH #2*figH+0.5
nullfmt = NullFormatter()

plasma=plt.get_cmap("plasma")

import numpy as np
import matplotlib.pyplot as plt
import os

base = "In1.0Ga0.0As0.5Sb0.5"
ddir = "../data/sc_dos"

area = lambda x, y: np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))  # trapezoid rule


def find(*names):
    """Return the first of `names` that exists in ddir."""
    for n in names:
        p = os.path.join(ddir, n)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"none of {names} found in {ddir}")


# ---------------------------------------------------------------- panel (a)
data = [np.loadtxt(f"{ddir}/{base}.{i}.total_dos.dat") for i in range(6)]

# phonopy gives each run its own frequency grid, so interpolate onto a shared one
freq = np.linspace(min(d[0, 0] for d in data), max(d[-1, 0] for d in data), 300)
dos = [np.interp(freq, d[:, 0], d[:, 1], left=0, right=0) for d in data]

# 64-atom SC vs 2-atom VCA cell: normalise every curve to unit area to compare shapes
dos = [y / area(freq, y) for y in dos]
sc_mean = np.mean(dos, axis=0)

# matdyn.x: column 0 = frequency in cm^-1, column 1 = total DOS
vca = np.loadtxt(f"{ddir}/{base}.VCA.dos")
vca_freq = vca[:, 0] / 33.35641          # cm^-1 -> THz
vca_dos = vca[:, 1] / area(vca_freq, vca[:, 1])

# ---------------------------------------------------------------- panel (b)
# config-averaged total and species projections from phonon_projection.py.
# Normalise all four by the area of the total so the projections keep their
# relative weights and the total matches panel (a).
ptot = np.loadtxt(find(f"{base}.total_dos.dat", "total_dos.dat"))
pf, pt = ptot[:, 0], ptot[:, 1]
norm = area(pf, pt)

species = ["In", "As", "Sb"]
proj = {}
for s in species:
    d = np.loadtxt(find(f"{base}.pdos_{s}.dat", f"pdos_{s}.dat"))
    proj[s] = np.interp(pf, d[:, 0], d[:, 1], left=0, right=0) / norm
pt = pt / norm

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.8), sharex=True)

ax[0].plot(freq, sc_mean, color="k", lw=2, label="SC mean")
ax[0].plot(vca_freq, vca_dos, color="r", lw=2, label="VCA")
ax[0].set_ylabel("DOS (normalised)")
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_ylim(bottom=0)
ax[0].text(0.03, 0.15, "(a)", transform=ax[0].transAxes, va="top", fontsize=10)

ax[1].plot(pf, pt, color="k", lw=1.6, label="SC mean")
for s, c in zip(species, [plasma(0.8), plasma(0.5), plasma(0.2)]):
    ax[1].plot(pf, proj[s], color=c, lw=1.2, label=f"{s}")
ax[1].set_xlabel("Frequency (THz)")
ax[1].set_ylabel("DOS (normalised)")
ax[1].legend(frameon=False, fontsize=8, ncol=2)
ax[1].set_ylim(bottom=0)
ax[1].text(0.03, 0.15, "(b)", transform=ax[1].transAxes, va="top", fontsize=10)

ax[1].set_xlim(0, 7)
fig.tight_layout()
fig.savefig(f"../Figures/{base}_dos.both.pdf", format="pdf")
plt.show()