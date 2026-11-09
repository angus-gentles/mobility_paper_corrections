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


for i in range(len(k1)//351):
    fig1,ax1=plt.subplots(1,1)
    sections.append((k1[i*351:(i+1)*351], k2[i*351:(i+1)*351], qs[i*351:(i+1)*351], q_bigs[i*351:(i+1)*351], el_np[i*351:(i+1)*351]))
    #print(sections[i][0].shape)

    k_mag=np.linalg.norm(sections[i][0][0])
    

    print(k_mag/max_k_mag)
    colour=plasma(k_mag/max_k_mag)
    #plot(ax, sections[-1][0], sections[-1][1], sections[-1][4], colour)
    plot(ax1, sections[-1][0], sections[-1][1], sections[-1][4], colour)
    fig1.savefig(f"../Figures/kmag/{base}.angle.k_{k_mag:.3f}.png", dpi=300)
    plt.close(fig1)

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

lower = [-np.inf, -np.inf, 1e-6, 0.3, -np.inf, -np.inf, 1e-6, 0.3, 1e-6, 0.3]
upper = [ np.inf,  0.0,    np.inf, 8.0,  np.inf,  np.inf, np.inf, 8.0, np.inf, 8.0]
params, covar = curve_fit(function, ks, el_np, p0=p0,
                          bounds=(lower, upper), max_nfev=20000)

#params, covar = curve_fit(
#    function,
#    ks,
#    el_np,
#    p0=p0,
#    bounds=(lower, upper),
#    maxfev=50000
#)

#params,covar=curve_fit(function, ks, el_np, p0=p0)
print(params)
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
ax2.set_xlabel("$k$ ($2\pi/a_0$)")
ax2.set_ylabel(r"$M(k,\theta )$ (eV)")
fig2.savefig(f"../Figures/{base}.M-vs-k.pdf",format='pdf')
plt.close(fig2)

cos_all = compute_cos_theta(k1, k2)
kmag = np.linalg.norm(k1, axis=1)
k_line = np.linspace(1e-5, kmag.max(), 300)

cases = [(0.0,       r"0",       "theta0"),
         (np.pi/2,   r"\pi/2",   "theta_pi2"),
         (np.pi,     r"\pi",     "theta_pi")]
tol = 0.05   # tolerance in cos(theta)

for theta, label, tag in cases:
    # data within tolerance of this angle
    mask = np.abs(cos_all - np.cos(theta)) < tol

    # fit evaluated along a line at exactly this angle
    ks_line = np.zeros((len(k_line), 2, 3))
    ks_line[:, 0, 0] = k_line
    ks_line[:, 1, 0] = k_line * np.cos(theta)
    ks_line[:, 1, 1] = k_line * np.sin(theta)

    fig3, ax3 = plt.subplots(1, 1)
    ax3.scatter(kmag[mask], el_np[mask], s=3, color=plasma(0.3),
                label=rf"data, $|\cos\theta - \cos({label})| < {tol}$")
    ax3.plot(k_line, function(ks_line, *params), color="black",
             label=rf"fit, $\theta = {label}$")
    ax3.set_xlabel(r"$k$ ($2\pi/a_0$)")
    ax3.set_ylabel(r"$M(k,\theta)$ (eV)")
    ax3.legend()
    fig3.savefig(f"../Figures/{base}.M-vs-k.{tag}.png")
    plt.close(fig3)
    print(f"theta={label}: {mask.sum()} points")


#exit(0)
with h5py.File(f"{base}.alloy_scattering_metadata.angular.h5", 'w') as f:
    #model: constant_elastic_alloy
    f.attrs['model'] = "constant_elastic_alloy"
    f.attrs['material'] = "In0.5Ga0.5As"
    f.attrs['alloy_sublattice'] = 'cation'
    f['composition_x'] = 0.5
    f.attrs['carrier'] = 'electron'
    f.attrs['temperature'] = 300
    f.attrs['bands'] = 'Conduction band only'
    f.attrs['SOC'] = False
    f.attrs['elastic']=True
    f['concentration_prefector_included'] = False
    group=f.create_group('Gamma-Gamma')
    group.attrs['fit_function'] = ("M_inf - dM/(1+(k_mag/k1)^p) + (A/(1+(k_mag/k0)^n) + C)"
                                   "*(1 - exp(-(k_mag/ks_turn)^m))*(1 - cos_theta)")
    for name, val in zip(['M_inf','dM','k1','p','A','C','k0','n','ks_turn','m'], params):
        group[name] = val



