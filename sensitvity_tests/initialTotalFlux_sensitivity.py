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


## R x Usg grid test
# Qsg is left as a diagnostic here rather than the swept variable, since holding R
# fixed while sweeping Qsg over several orders of magnitude forces Usg well outside
# physically plausible subglacial conduit exit velocities at the high end.
Rs   = np.array([0.5, 1, 2, 3, 5, 7, 10,20,30,40,50])     # [m] -- set from site-appropriate conduit geometry
Usgs = np.array([0.1, 0.5, 1, 2, 3, 5, 6, 7, 8, 9, 10, 
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])       # [m/s] -- set from site-appropriate exit-velocity literature

os.makedirs("initialTotalFlux_grid", exist_ok=True)

for U_val in Usgs:
    for R_val in Rs:

        p.R = R_val
        p.Usg = U_val
        Q_T = p.R**2 * c.pi * p.Usg * 0.5
        p.Qsg = 0.9 * Q_T
        Qg = 0.1 * Q_T
        rho_g = ambient[3, 0] * c.M_gas[0][0][0] / (c.Ru * c.T0)   # [kg/m^3]
        mb0 = Qg * rho_g                                            # [kg/s]
        p.mb0 = mb0

        p.modelType = 'double_plume'
        model = dP(ambient, p)
        yi, zi, yo, zo, jp, i_peel = dP.returnResults(model)
        mdot = model.melting(yi, zi, p)
        Sb = model.boundary_salinity(yi, mdot, p)
        Tb = model.boundary_temperature(Sb, zi, p)

        df = pd.DataFrame(yi.T, index=None)
        df["zi"] = zi
        df["Sb"] = Sb
        df["Tb"] = Tb
        df["mdot"] = mdot
        df["peel"] = np.zeros(len(zi))
        if jp > 0:
            df["peel"].iloc[i_peel] = 1

        df.to_csv(f"initialTotalFlux_grid/innerPlume_R_{R_val}_Usg_{U_val}.csv")

        df = pd.DataFrame(yo.T, index=None)
        df["zo"] = zo
        df.to_csv(f"initialTotalFlux_grid/outerPlume_R_{R_val}_Usg_{U_val}.csv")

        p.modelType = 'single_plume'
        sp_model = dP(ambient, p)
        y, z, _, _, _, _ = dP.returnResults(sp_model)
        mdot = sp_model.melting(y, z, p)
        Sb = sp_model.boundary_salinity(y, mdot, p)
        Tb = sp_model.boundary_temperature(Sb, z, p)

        df = pd.DataFrame(y.T, index=None)
        df["z"] = z
        df["Sb"] = Sb
        df["Tb"] = Tb
        df["mdot"] = mdot

        df.to_csv(f"initialTotalFlux_grid/singlePlume_R_{R_val}_Usg_{U_val}.csv")