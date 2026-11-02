import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean.cm as cm
import os
import sys
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

salt_file = p.Sa_filePath
temp_file = p.Ta_filePath



dz = np.arange(-250, 0, 5)

Nx = 50
Ny = 11
Nz = len(dz)

salt = np.fromfile(salt_file, dtype="<f8")
salt = salt.reshape((Nz, Ny, Nx))
temp = np.fromfile(temp_file, dtype="<f8")
temp = temp.reshape((Nz, Ny, Nx))

mp = int(Ny/2)
saltProf = salt[:, mp, 1]
tempProf = temp[:, mp, 1]
zProf = np.linspace(0, 250, Nz)
PProf = np.zeros(Nz)
OProf = np.zeros(Nz)

## Adjusting Temp to only have two layers
T_max = np.nanmax(tempProf)
T_max_i = np.argmax(tempProf)
i_add = Nz - T_max_i

temp_ini_1d_two = np.zeros(len(tempProf))
temp_ini_1d_two[:i_add] = tempProf[1] - 3
temp_ini_1d_two[i_add:] = tempProf[2:T_max_i+2] - 3

P0 = 101000
ambient = np.array([zProf, temp_ini_1d_two[::-1], saltProf[::-1], PProf, OProf])
ambient[3, :] = ambient[0, :] * p.rho_r * c.g + P0
ambient[3, :] = ambient[3, :][::-1]

p.C0 = 0.005   # [kg/m^3]
p.S0 = 0.001   # [psu]
p.kappa_peel = 1.5

## Qsg test
Qsgs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 20, 30, 40, 50, 60, 70, 80, 90,
                  100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

p.R = 2   # [m]

os.makedirs("initialTotalFlux", exist_ok=True)

mean_sp_mdots = np.zeros(len(Qsgs))
mean_dp_mdots = np.zeros(len(Qsgs))
max_sp_mdots = np.zeros(len(Qsgs))
max_dp_mdots = np.zeros(len(Qsgs))

mean_sp_u = np.zeros(len(Qsgs))
mean_dp_u = np.zeros(len(Qsgs))
max_sp_u = np.zeros(len(Qsgs))
max_dp_u = np.zeros(len(Qsgs))

jps = np.zeros(len(Qsgs))

for i, value in enumerate(Qsgs):

    Qsg0 = value
    p.Qsg = 0.9 * Qsg0
    Qg = 0.1 * Qsg0
    rho_g = ambient[3, 0] * c.M_gas[0][0][0] / (c.Ru * c.T0)   # [kg/m^3]
    mb0 = Qg * rho_g                                            # [kg/s]
    p.mb0 = mb0

    p.Usg = Qsg0 * 2 / (np.pi * p.R**2)

    p.modelType = 'double_plume'
    model = dP(ambient, p)
    yi, zi, yo, zo, jp, i_peel = dP.returnResults(model)
    mdot_dp = model.melting(yi, zi, p)
    Sb = model.boundary_salinity(yi, mdot_dp, p)
    Tb = model.boundary_temperature(Sb, zi, p)

    df = pd.DataFrame(yi.T, index=None)
    df["zi"] = zi
    df["Sb"] = Sb
    df["Tb"] = Tb
    df["mdot"] = mdot_dp
    df["peel"] = np.zeros(len(zi))
    if jp > 0:
        df["peel"].iloc[i_peel] = 1

    df.to_csv(f"initialTotalFlux/innerPlume_Qsg_{value}.csv")

    df = pd.DataFrame(yo.T, index=None)
    df["zo"] = zo
    df.to_csv(f"initialTotalFlux/outerPlume_Qsg_{value}.csv")

    p.modelType = 'single_plume'
    sp_model = dP(ambient, p)
    y, z, _, _, _, _ = dP.returnResults(sp_model)
    mdot_sp = sp_model.melting(y, z, p)
    Sb = sp_model.boundary_salinity(y, mdot_sp, p)
    Tb = sp_model.boundary_temperature(Sb, z, p)

    df = pd.DataFrame(y.T, index=None)
    df["z"] = z
    df["Sb"] = Sb
    df["Tb"] = Tb
    df["mdot"] = mdot_sp

    df.to_csv(f"initialTotalFlux/singlePlume_Qsg_{value}.csv")

    mean_sp_mdots[i] = np.mean(mdot_sp)
    mean_dp_mdots[i] = np.mean(mdot_dp)
    max_sp_mdots[i]  = np.max(mdot_sp)
    max_dp_mdots[i]  = np.max(mdot_dp)

    mean_sp_u[i] = np.mean(y[1]/y[0])
    mean_dp_u[i] = np.mean(yi[1]/yi[0])
    max_sp_u[i]  = np.max(y[1]/y[0])
    max_dp_u[i]  = np.max(yi[1]/yi[0])

    jps[i] = jp

## ---- Single-plume vs. double-plume ratio plot ----
fig, ax = plt.subplots(1, figsize=(7, 5))
noPeels = np.where(jps == 0)[0]
peels = np.where(jps > 0)[0]
cmap = cm.haline(np.linspace(0, 1, 5))

dp_color = cmap[1]

ax.scatter(Qsgs[noPeels], max_dp_mdots[noPeels]/max_sp_mdots[noPeels], color=dp_color, zorder=2)
ax.scatter(Qsgs[noPeels], max_dp_u[noPeels]/max_sp_u[noPeels], color=dp_color, facecolor='none', zorder=2)

ax.scatter(Qsgs[peels], max_dp_mdots[peels]/max_sp_mdots[peels], color=dp_color, marker="*")
ax.scatter(Qsgs[peels], max_dp_u[peels]/max_sp_u[peels], color=dp_color, facecolor='none', marker='*')

ax.set_xlabel('$Q_{sg}$ [m$^3$ s$^{-1}$]')
ax.set_ylabel('Ratio')
ax.grid(alpha=0.25, zorder=1)
ax.set_xscale('log')
ax.legend(['$\\dot{m}_{DPM}/\\dot{m}_{SPM}$', '$u_{DPM}/u_{SPM}$'])
ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ax.set_yticklabels(['', 2, '', 4, '', 6, '', 8, '', 10, '', 12])

gr = mpatches.Rectangle((10**2, 0), width=2000, height=15, facecolor=cmap[3], alpha=0.1)
ant = mpatches.Rectangle((0, 0), width=10, height=15, facecolor=cmap[2], alpha=0.1)

ax.add_patch(gr)
ax.add_patch(ant)
ax.text(10**(-1), 1, s='Antarctica Runoff', color='darkslategray', fontsize=12)
ax.text(12**2, 3, s='Greenland', color='darkolivegreen', fontsize=12)
ax.text(20**2, 2.45, s='Runoff', color='darkolivegreen', fontsize=12)

ax.set_ylim(0, 6)

plt.savefig('Figures/singlePlume_doublePlume_ratio.png', dpi=300)