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
mpl.rcParams['figure.figsize'] = figW, figH
nullfmt = NullFormatter()



plasma=plt.get_cmap("plasma")


base="In1.0Ga0.0As0.5Sb0.5"

def linear_func(x, m, b):
    return m * x + b
"""
def function(ks, A, C, k0, n, alpha):
    k1 = ks[:, 0, :]   # (N, 3)
    k2 = ks[:, 1, :]   # (N, 3)

    # element-wise dot product
    dots = np.sum(k1 * k2, axis=1)

    # norms per vector
    norm1 = np.linalg.norm(k1, axis=1)
    norm2 = np.linalg.norm(k2, axis=1)

    cos_theta = dots / (norm1 * norm2)

    # |k| (same sphere → use k1)
    k_mag = norm1

    # radial part
    func1 = A / (1.0 + (k_mag / k0)**n) + C

    return func1 * (1.0 + alpha * cos_theta)
"""
def function1(ks, M0, A, C, k0, n):
    k1 = ks[:, 0, :]
    k2 = ks[:, 1, :]

    dots = np.sum(k1 * k2, axis=1)

    norm1 = np.linalg.norm(k1, axis=1)
    norm2 = np.linalg.norm(k2, axis=1)

    cos_theta = np.divide(
        dots,
        norm1 * norm2,
        out=np.ones_like(dots),
        where=(norm1 * norm2) > 1e-14
    )

    k_mag = norm1

    f = A / (1.0 + (k_mag / k0)**n) + C

    return M0 + f * (1.0 - cos_theta)


def function2(ks, M00, M1, k1, G, k0, n):
    k1_vec = ks[:, 0, :]
    k2_vec = ks[:, 1, :]

    # dot and norms
    dots = np.sum(k1_vec * k2_vec, axis=1)
    norm1 = np.linalg.norm(k1_vec, axis=1)
    norm2 = np.linalg.norm(k2_vec, axis=1)

    cos_theta = np.divide(
        dots,
        norm1 * norm2,
        out=np.ones_like(dots),
        where=(norm1 * norm2) > 1e-14
    )

    k_mag = norm1

    # --- isotropic baseline (saturating increase)
    M0 = M00 + M1 * (1.0 - np.exp(-k_mag / k1))

    # --- angular strength (turn-on with k)
    g = G * (1.0 - np.exp(-(k_mag / k0)**n))

    return M0 + g * (1.0 - cos_theta)

def function(ks, M00, M1, k1, A, C, k0, n, ks_turn, m):
    k1_vec = ks[:, 0, :]
    k2_vec = ks[:, 1, :]

    dots = np.sum(k1_vec * k2_vec, axis=1)

    norm1 = np.linalg.norm(k1_vec, axis=1)
    norm2 = np.linalg.norm(k2_vec, axis=1)

    cos_theta = np.divide(
        dots,
        norm1 * norm2,
        out=np.ones_like(dots),
        where=(norm1 * norm2) > 1e-14
    )

    k_mag = norm1

    # ✅ improved baseline (your preferred exponential)
    M0 = M00 + M1 * (1.0 - np.exp(-k_mag / k1))

    # ✅ your original good radial function
    f = A / (1.0 + (k_mag / k0)**n) + C

    # ✅ NEW: angular turn-on
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
    

with h5py.File(f"../data/{base}.coupling_elements.SI.hdf5", "r") as f:
    k1 = f["k1"][:]
    k2 = f["k2"][:]
    qs = f["qs"][:]
    q_bigs = f["q_bigs"][:]
    el_np = f["coupling_elements_np"][:]

sections = []
print(len(k1)//351)
print(len(k1)/28)
#exit(0)
max_k_mag=np.max(np.linalg.norm(k1, axis=1))
print(max_k_mag)
fig, ax = plt.subplots(1, 1)
for i in range(len(k1)//351):
    fig1,ax1=plt.subplots(1,1)
    sections.append((k1[i*351:(i+1)*351], k2[i*351:(i+1)*351], qs[i*351:(i+1)*351], q_bigs[i*351:(i+1)*351], el_np[i*351:(i+1)*351]))
    #print(sections[i][0].shape)

    k_mag=np.linalg.norm(sections[i][0][0])
    

    print(k_mag/max_k_mag)
    colour=plasma(k_mag/max_k_mag)
    plot(ax, sections[-1][0], sections[-1][1], sections[-1][4], colour)
    plot(ax1, sections[-1][0], sections[-1][1], sections[-1][4], colour)
    fig1.savefig(f"../Figures/kmag/{base}.angle.k_{k_mag:.3f}.png", dpi=300)
    plt.close(fig1)
ks=np.zeros((len(k1),2,3))
ks[:,0,:]=k1
ks[:,1,:]=k2
#function(ks, M0, A, C, k0, n)
p0=[3, 1.45, 1.82, 0.016, 1.7]
p0 = [
    np.min(el_np),        # M00
    np.max(el_np)*0.5,    # M1
    np.median(np.linalg.norm(k1, axis=1)),  # k1
    np.max(el_np)*0.5,    # G
    np.median(np.linalg.norm(k1, axis=1)),  # k0
    2.0                   # n
]

p0 = [
    np.min(el_np),           # M00
    np.max(el_np) * 0.3,     # M1
    np.median(np.linalg.norm(k1, axis=1)),  # k1

    np.max(el_np),           # A
    np.min(el_np),           # C
    np.median(np.linalg.norm(k1, axis=1)),  # k0
    2.0,                     # n

    np.median(np.linalg.norm(k1, axis=1)) * 0.5,  # ks_turn
    2.0                      # m
]


params,covar=curve_fit(function, ks, el_np, p0=p0)
print(params)
#print(covar)
#fig.savefig(f"../Figures/{base}.coupling_elements.angle.png", dpi=300)
plt.close(fig)
    #exit(0)
colours=plasma(0.5*(np.sum(k1*k2,axis=1)/np.linalg.norm(k1, axis=1)/np.linalg.norm(k2, axis=1))+0.5)
fig2,ax2=plt.subplots(1,1)
ax2.scatter(np.linalg.norm(k1, axis=1), el_np, s=1, c=colours)
#k2_test=np.zeros((200,2,3))

k2_test0=np.zeros((200,2,3))
k2_test0[:,0,0]=np.linspace(0.00001,0.1,200)
k2_test0[:,1,0]=np.linspace(0.00001,0.1,200)

k2_test1=np.zeros((200,2,3))
k2_test1[:,0,0]=np.linspace(0.00001,0.1,200)
k2_test1[:,1,1]=np.linspace(0.00001,0.1,200)

k2_test2=np.zeros((200,2,3))
k2_test2[:,0,0]=np.linspace(0.00001,0.1,200)
k2_test2[:,1,0]=-np.linspace(0.00001,0.1,200)

ax2.plot(np.linspace(0.00001,0.1,200), function(k2_test0, *params), color="black", linestyle="-", label=r"fit, $\theta =0$")
ax2.plot(np.linspace(0.00001,0.1,200), function(k2_test1, *params), color="black", linestyle=":", label=r"fit, $\theta =\pi/2$")
ax2.plot(np.linspace(0.00001,0.1,200), function(k2_test2, *params), color="black", linestyle="--", label=r"fit, $\theta =\pi$")
ax2.legend()
ax2.set_xlabel("$k$ ($2\pi/a_0$)")
ax2.set_ylabel(r"$M(k,\theta )$ (eV)")
fig2.savefig(f"../Figures/{base}.M-vs-k.png")
plt.close(fig2)

with h5py.File("In1.0Ga0.0As0.5Sb0.5.alloy_scattering_metadata.angular.h5", 'w') as f:
    #model: constant_elastic_alloy
    f.attrs['model'] = "constant_elastic_alloy"
    f.attrs['material'] = "InAs0.5Sb0.5"
    f.attrs['alloy_sublattice'] = 'anion'
    f['composition_x'] = 0.5
    f.attrs['carrier'] = 'electron'
    f.attrs['temperature'] = 300
    f.attrs['bands'] = 'Conduction band only'
    f.attrs['SOC'] = False
    f.attrs['elastic']=True
    f['concentration_prefector_included'] = False
    group=f.create_group('Gamma-Gamma')
    group.attrs['fit_function'] = "M00 + M1 * (1.0 - np.exp(-k_mag / k1)) + (A/(1+(|k|/k0)^n) + C)*(1.0 - np.exp(-(k_mag / ks_turn)**m)) * (1.0 - cos_theta)"
    #(ks, M00, M1, k1, A, C, k0, n, ks_turn, m)
    group['M00'] = params[0]
    group['M1'] = params[1]
    group['k1'] = params[2]
    group['A'] = params[3]
    group['C'] = params[4]
    group['k0'] = params[5]
    group['n'] = params[6]
    group['ks_turn'] = params[7]
    group['m'] = params[8]
    #group['A'] = A
    #group['k0'] = k0
    #group['n'] = n
    #group['C'] = C
    f['delta_function_included'] = False
    f['final_rate_included'] = False


