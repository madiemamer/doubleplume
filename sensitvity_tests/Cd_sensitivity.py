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

Qsg0 = 7.5
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
values = np.arange(0.001,0.1+0.01,0.01)
i = 0
for value in values:
    p.Cd = value 

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

    df.to_csv(f"innerPlume_Cd_{value}.csv")

    df = pd.DataFrame(yo.T, index=None)
    df["zo"] = zo
    df.to_csv(f"drag/outerPlume_Cd_{value}.csv")

    p.modelType = 'single_plume'
    sp_model = dP(ambient, p)
    y, z, _, _, _, _ = dP.returnResults(sp_model)
    mdot = sp_model.melting(y,z,p)
    Sb = sp_model.boundary_salinity(y, mdot, p)
    Tb = sp_model.boundary_temperature(Sb, z, p)

    df = pd.DataFrame(y.T, index=None)
    df["z"] = z
    df["Sb"] = Sb
    df["Tb"] = Tb
    df["mdot"] = mdot 

    df.to_csv(f"drag/singlePlume_Cd_{value}.csv")
