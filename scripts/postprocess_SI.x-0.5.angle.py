#!/usr/bin/env python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plasma=plt.get_cmap("plasma")

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

def function(ks, M00, B, alpha, A, C, k0, n, ks_turn, m):
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

    # new forward-scattering baseline
    M0 = M00 + B * (k_mag + 1e-12)**alpha

    # angular part
    f = A / (1.0 + (k_mag / k0)**n) + C

    turn_on = 1.0 - np.exp(-(k_mag / ks_turn)**m)

    return M0 + f * turn_on * (1.0 - cos_theta)

def function(ks, M00, M1, ks_turn, m, A, C, k0, n):
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

    # shared switch-on
    switch_on = 1.0 - np.exp(-(k_mag / ks_turn)**m)

    # angular strength
    f = A / (1.0 + (k_mag / k0)**n) + C

    return M00 + switch_on * (
        M1 + f * (1.0 - cos_theta)
    )

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
    
base="In0.5Ga0.5As1.0Sb0.0"
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
#fig, ax = plt.subplots(1, 1)
"""

for i in range(len(k1)//351):
    fig1,ax1=plt.subplots(1,1)
    sections.append((k1[i*351:(i+1)*351], k2[i*351:(i+1)*351], qs[i*351:(i+1)*351], q_bigs[i*351:(i+1)*351], el_np[i*351:(i+1)*351]))
    #print(sections[i][0].shape)

    k_mag=np.linalg.norm(sections[i][0][0])
    

    print(k_mag/max_k_mag)
    colour=plasma(k_mag/max_k_mag)
    plot(ax, sections[-1][0], sections[-1][1], sections[-1][4], colour)
    plot(ax1, sections[-1][0], sections[-1][1], sections[-1][4], colour)
    fig1.savefig(f"figs/coupling_elements.angle.k{k_mag:.3f}.png", dpi=300)
    plt.close(fig1)
"""
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

kmag = np.linalg.norm(k1, axis=1)

p0 = [
    np.min(el_np),          # M00
    np.ptp(el_np),          # M1
    0.5*np.median(kmag),    # ks_turn
    2.0,                    # m
    np.max(el_np)*0.5,      # A
    np.min(el_np)*0.2,      # C
    np.median(kmag),        # k0
    2.0                     # n
]
lower = [
    -np.inf,   # M00
    -np.inf,   # M1
    1e-5,      # ks_turn
    0.1,       # m
    -np.inf,   # A
    -np.inf,   # C
    1e-5,      # k0
    0.1        # n
]

upper = [
    np.inf,
    np.inf,
    np.inf,
    10.0,
    np.inf,
    np.inf,
    np.inf,
    10.0
]

params, covar = curve_fit(
    function,
    ks,
    el_np,
    p0=p0,
    bounds=(lower, upper),
    maxfev=50000
)

#params,covar=curve_fit(function, ks, el_np, p0=p0)
#print(params)
#print(covar)
#fig.savefig(f"../Figures/{base}.coupling_elements.angle.png", dpi=300)
#plt.close(fig)
    #exit(0)
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

ax2.plot(np.linspace(0.00001,0.1,200), function(k2_test0, *params), color="black", linestyle="-", label="fit, k1.k2/norm=1")
ax2.plot(np.linspace(0.00001,0.1,200), function(k2_test1, *params), color="black", linestyle=":", label="fit, k1.k2/norm=0")
ax2.plot(np.linspace(0.00001,0.1,200), function(k2_test2, *params), color="black", linestyle="--", label="fit, k1.k2/norm=-1")
ax2.legend()
ax2.set_xlabel("|k| (1/nm)")
ax2.set_ylabel("Coupling element (eV)")
fig2.savefig(f"../Figures/{base}test.png")
plt.close(fig2)
#exit(0)
with h5py.File(f"{base}.alloy_scattering_metadata.angular.h5", 'w') as f:
    #model: constant_elastic_alloy
    f.attrs['model'] = "constant_elastic_alloy"
    f.attrs['material'] = base
    f.attrs['alloy_sublattice'] = 'anion'
    f['composition_x'] = 0.5
    f.attrs['carrier'] = 'electron'
    f.attrs['temperature'] = 300
    f.attrs['bands'] = 'Conduction band only'
    f.attrs['SOC'] = False
    f.attrs['elastic']=True
    f['concentration_prefector_included'] = False
    group=f.create_group('Gamma-Gamma')
    group.attrs['fit_function'] = (
        "M00 + (1-exp(-(k_mag/ks_turn)^m)) * "
        "( M1 + (A/(1+(k_mag/k0)^n)+C)*(1-cos_theta) )"
        )
    #(ks, M00, M1, k1, A, C, k0, n, ks_turn, m)
    group['M00'] = params[0]
    group['M1'] = params[1]
    group['ks_turn'] = params[2]
    group['m'] = params[3]
    group['A'] = params[4]
    group['C'] = params[5]
    group['k0'] = params[6]
    group['n'] = params[7]
    #group['A'] = A
    #group['k0'] = k0
    #group['n'] = n
    #group['C'] = C
    f['delta_function_included'] = False
    f['final_rate_included'] = False


