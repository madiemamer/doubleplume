import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean.cm as cm
import os
import sys
import string
cwd = os.getcwd()

if "mmamer3" in cwd:
    moduleA_path = "/Users/mmamer3/Documents/bubblePlume"
    moduleB_path = "/Users/mmamer3/Documents/bubblePlume/model"
    moduleC_path = "/Users/mmamer3/Documents/bubblePlume/utils"
else:
    moduleA_path = '/Users/madelinemamer/Documents/bubblePlume'
    moduleB_path = '/Users/madelinemamer/Documents/bubblePlume/model'
    moduleC_path = '/Users/madelinemamer/Documents/bubblePlume/utils'

for module in [moduleA_path, moduleB_path, moduleC_path]:
    if module not in sys.path:
        sys.path.append(module)

import model.const as c
import model.params as p
from model.doublePlume import doublePlume as dP

plt.rcParams.update({
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16
})

salt_file = p.Sa_filePath
temp_file = p.Ta_filePath

dz = np.arange(-250,0,5)

Nx = 50
Ny = 11
Nz = len(dz)

salt = np.fromfile(salt_file, dtype = "<f8")
salt = salt.reshape((Nz, Ny, Nx))
temp = np.fromfile(temp_file, dtype = "<f8")
temp = temp.reshape((Nz, Ny, Nx))

mp = int(Ny/2)
saltProf = salt[:,mp,1]
tempProf = temp[:,mp,1]
zProf = np.linspace(0,250,Nz)
PProf = np.zeros(Nz)
OProf = np.zeros(Nz)

## Adjusting Temp to only have two layers
T_max = np.nanmax(tempProf)
T_max_i = np.argmax(tempProf)
i_add = Nz - T_max_i

temp_ini_1d_two = np.zeros(len(tempProf))
temp_ini_1d_two[:i_add] = tempProf[1] -3
temp_ini_1d_two[i_add:] = tempProf[2:T_max_i+2]-3

P0 = 101000
# ambient = np.array([[0, 50, 100, 150, 200, 250, 300, 350, 400], #z [m], 0 is at diffuser
#                     [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2], # temp [C]
#                    [33, 33, 33, 32, 31, 30, 29, 28, 27], # salinity [psu]
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0],#1082000, 836750, 591500, 346250, 101000], # [Pa] pressure
#                    [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001], # [?] O2
#                    ])
ambient = np.array([zProf, temp_ini_1d_two[::-1], saltProf[::-1], PProf, OProf])
ambient[3, :] = ambient[0,:] * p.rho_r * c.g + P0
ambient[3, :] = ambient[3, :][::-1]

Qsg0 = 0.075
p.kappa_peel = 1.5
p.R = 2 # [m]
p.C0 = 0.005 # [kg/m^3]
p.S0 = 0.001 # [psu]
p.Qsg = 0.9 * Qsg0 #0.01 #1e-2 
Qg = 0.1 * Qsg0
print(f"Qg: {Qg}")
rho_g = ambient[3,0] * c.M_gas[0][0][0] / (c.Ru * c.T0) # [kg/m^3]
mb0 = Qg * rho_g # [kg/s]
p.mb0 = mb0

p.Usg = Qsg0 * 2 / (np.pi * p.R**2)
print(f"Usg: {p.Usg}")

## kappa test
values = [0.5, 0.75, 0.95]
cmap = cm.phase(np.linspace(0,1,len(values)+1))
fig, ax = plt.subplots(2,3,figsize=(12,14))
i = 0
for value in values:
    p.fp = value 
    model = dP(ambient, p)
    yi, zi, yo, zo, jp, i_peel = dP.returnResults(model)

    # Inner plume conditions
    ui = yi[1]/yi[0]                                                 # inner velocity, m/s
    bi = (2 * yi[0]/(ui * c.pi))**(1/2)                             # inner radius, m
    Ti = yi[2] / (yi[0] * p.rho_r * c.cp) - c.T0                     # inner temperature, C
    Si = yi[4] / yi[0]                                               # inner salinity, psu
    mb = yi[5]                                                       # inner gas mass flux, kg/s
    ci = yi[6] / yi[0]                                               # inner O2 concentration, kg/m^3
    mdot = model.melting(yi, zi, p)

    ax[0,0].plot(yi[0], zi, color = cmap[i], label = value)
    ax[0,1].plot(ui, zi, color = cmap[i], label = value)
    ax[0,2].plot(bi, zi, color = cmap[i], label = value)

    ax[1,0].plot(Ti, zi, color = cmap[i], label = value)
    ax[1,1].plot(Si, zi, color = cmap[i], label = value)
    ax[1,2].plot(mdot*3600*24, zi, color = cmap[i], label = value)

    i = i + 1

for i in range(2):
    for j in range(3):
        ax[i,j].set_ylim(np.min(zi), np.max(zi))
        ax[i,j].grid(alpha=0.1)
        ax[i,j].set_ylabel("Depth [m]")

ax[0,0].set_xlabel("Vol. Flux [m$^3$ s$^{-1}$]")
ax[0,1].set_xlabel("Inner Velocity [m s$^{-1}$]")
ax[0,2].set_xlabel("Inner Width [m]")

ax[1,0].set_xlabel("Inner Temp [C]")
ax[1,1].set_xlabel("Inner Salinity [ppt]")
ax[1,2].set_xlabel("Inner Melt Rate [m d$^{-1}$]")

ax[0,0].legend(title = "$f_p$", title_fontsize=16)
axes = ax
for ax, label in zip(axes.flat, string.ascii_lowercase):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        fontsize=16,
        va='top',
        ha='left'
    )
    
plt.tight_layout()
plt.savefig("Figures/fp_sensitivity.png", dpi = 300)
