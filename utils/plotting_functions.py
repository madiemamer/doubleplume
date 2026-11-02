
import re
import cmocean
import string

import gsw
from scipy.interpolate import interp1d

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
import model.gas as gas
from model.doublePlume import doublePlume as dP

salt_file = p.Sa_filePath
temp_file = p.Ta_filePath


plt.rcParams.update({
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16
})

def getSensitivity(innerFiles, singleFiles, ratio=True, getMax=True, getMean=False, gasTest=False):
    nFiles = len(innerFiles)
    # if nFiles != len(singleFiles):
    #     print("Number of Single Plume Files does not match length of Inner Plume Files.")
    #     return

    ratios_mdot = np.zeros(nFiles)
    ratios_u = np.zeros(nFiles)
    var = np.zeros(nFiles) * np.nan
    if gasTest:
        single = pd.read_csv(singleFiles, index_col=0)
    for i in range(nFiles):
        inner = pd.read_csv(innerFiles[i], index_col=0)
        if not gasTest:
            single = pd.read_csv(singleFiles[i], index_col=0)

        match = re.search(r"(\d+\.\d+)", innerFiles[i])
        if match:
            var[i] = float(match.group(1))

        ui, bi, Ti, Si, mb, ci, mdot_i, zi, _ = getVars(inner, model="double_plume")
        u, b, T, S, mdot, z = getVars(single, model="single_plume")

        if getMax:
            ratio_mdot = np.nanmax(mdot_i) / np.nanmax(mdot)
            ratio_u = np.nanmax(ui) / np.nanmax(u)
        else:
            ratio_mdot = np.nanmean(mdot_i) / np.nanmean(mdot)
            ratio_u = np.nanmean(ui) / np.nanmean(u)

        ratios_mdot[i] = ratio_mdot
        ratios_u[i] = ratio_u

    return ratios_mdot, ratios_u, var
        
def getVars(dataFrame, model="double_plume", c=c, p =p):
    if model == "double_plume":
        Q = dataFrame.iloc[:,0]
        J = dataFrame.iloc[:,1]
        H = dataFrame.iloc[:,2]
        Hg = dataFrame.iloc[:,3]
        B = dataFrame.iloc[:,4]
        Qg = dataFrame.iloc[:,5]
        Cg = dataFrame.iloc[:,6]
        z = dataFrame["zi"]
        j_peels = len(np.where(dataFrame["peel"] != 0.0)[0])

        ui = J/Q                                              # inner velocity, m/s
        bi = (2 * Q/(ui * c.pi))**(1/2)                             # inner radius, m
        Ti = H / (Q * p.rho_r * c.cp) - c.T0                     # inner temperature, C
        Si = B / Q                                               # inner salinity, psu
        mb = Qg                                                       # inner gas mass flux, kg/s
        ci = Cg / Q    

        y = [Q,J,H,Hg,B,Qg,Cg]
        depth = np.nanmax(z) - z
        mdot = melting(y, depth, p, model=model)

        return ui, bi, Ti, Si, mb, ci, mdot, z, j_peels

    elif model == "single_plume":
        Q = dataFrame.iloc[:,0]
        J = dataFrame.iloc[:,1]
        H = dataFrame.iloc[:,2]
        B = dataFrame.iloc[:,3]
        z = dataFrame["z"]

        u = J/Q                                              # inner velocity, m/s
        b = (2 * Q/(u * c.pi))**(1/2)                             # inner radius, m
        T = H / (Q) - c.T0                     # inner temperature, C
        S = B / Q   

        y = [Q,J,H,B]
        depth = np.nanmax(z) - z
        mdot = melting(y, depth, p, model=model)

        return u, b, T, S, mdot, z

    else:
        print("No Model found of that type. Please use \"single_plume\" or \"double_plume\" as the model parameter.")
    return

def getVars_fromModel(y=None, yi=None, yo=None, z=None, zi=None, zo=None, ambient=None, model="double_plume", c=c, p=p):
    
    Pa_interp = interp1d(ambient[0,:][::-1], ambient[-2,:])
    rho_a = gsw.density.rho_t_exact(ambient[2,:], ambient[1,:], ambient[-2,:]/10**4)

    if model == "single_plume":
        z_sp = np.linspace(-ambient[0,-1], z[-1]-ambient[0,-1], len(y[0]))
        u = y[1]/y[0]
        b = (2 * y[0]/(u * np.pi))**(1/2) 
        T = y[2]/y[0]
        S = y[3]/y[0]
        Pa = Pa_interp(z)
        rho = gsw.density.rho_t_exact(S, T, Pa/10**4)

        return u, b, T, S, rho, rho_a
    else:
        zi_dp = np.linspace(-ambient[0,-1], 0, len(yi[0]))
        zo_dp = np.linspace(-ambient[0,-1], 0, len(yo[0]))
        ui = yi[1]/yi[0]
        Ti = yi[2]/(yi[0] * c.cp * p.rho_r) - c.T0
        Si = yi[4]/yi[0]
        bi = (2 * yi[0]/(ui * np.pi))**(1/2)

        uo = yo[1]/yo[0]
        uo = uo[::-1]
        To = yo[2]/(yo[0] * c.cp * p.rho_r) - c.T0
        To = To[::-1]
        So = yo[3]/yo[0]
        So = So[::-1]

        bi_interp = interp1d(zi, bi)
        bi_i = bi_interp(zo)
        bo = (2 * yo[0][::-1]/(np.pi*uo) + bi_i**2)**(1/2)
        zo = zo[::-1]

        Pa_i = Pa_interp(-zi_dp)
        Pa_o = Pa_interp(-zo_dp)
        rho_i = gsw.density.rho_t_exact(Si, Ti, Pa_i/10**4)
        rho_o = gsw.density.rho_t_exact(So, To, Pa_o/10**4)
        

        return ui, Ti, Si, bi, uo, To, So, bo, zo, rho_i, rho_o, rho_a

def melting(y, z, p, model="double_plume"):

    # Evaluate the ice wall melt rate. Model from
    # Holland & Jenkins 1999.
    #--------------------------------------------#

    if model == 'single_plume':
        Q, J, H, B = y
        T = H/Q 
    else:
        Q, J, H, Hb, B, mb, C = y
        T = H/(p.rho_r * c.cp * Q) - c.T0

    beta = c.cp * p.gammaT * p.Cd**(1/2) / (c.L_i)
    a = 1
    b = p.gammaS * p.Cd**(1/2) * J/Q - beta * J/Q * (T - c.lam2 - c.lam3 * z)
    c_loc = beta * p.gammaS * p.Cd**(1/2) * J**2/Q**2 * (B/Q * c.lam1 - T + c.lam2 + c.lam3 * z)
    m = (-b + np.sqrt(b**2 - 4 * a * c_loc)) / (2 * a)
    return m

def make_figure(innerFiles, singleFiles, labels={},  c=c, p=p,
                ratio=True, getMax=True, getMean=False, nProfiles=5, 
                log=False, colormap = cm.matter,
                fig=None, ax_dict=None, save=False, figName=None, legendLoc=5):
    ratios_mdot, ratios_u, vars = getSensitivity(innerFiles, singleFiles, ratio=ratio, getMax=getMax, getMean=getMean)
    if labels["varName"] == "Radius":
            vars = np.arange(1,100,10)

    if fig is None:
        fig, ax_dict = plt.subplot_mosaic([['A', 'B', 'B']], figsize=(20,8))

    colormap = cmocean.tools.crop_by_percent(colormap, 20, which='min')

    ax1 = ax_dict['A']
    ax2 = ax_dict['B']
    nFiles = len(innerFiles)
    skip = int(nFiles / nProfiles)
    cmap = colormap(np.linspace(0,1,nFiles))
    for i in range(0, nFiles, skip):
        ui, bi, Ti, Si, mb, ci, mdot, zi, _ = getVars(pd.read_csv(innerFiles[i], index_col=0), "double_plume",c, p)
        u, b, T, S, mdot, z = getVars(pd.read_csv(singleFiles[i], index_col=0), "single_plume",c, p)

        ax1.plot(ui, zi, color = cmap[i], linestyle='-', linewidth = 2, label= vars[i].round(2))
        ax1.plot(u, z, color = cmap[i], linestyle =':')

    ax1.plot([0,1], [300,350],color = 'gray', linestyle = '-', label="DPM")
    ax1.plot([0,1], [300,350],color = 'gray', linestyle = ':', label="SPM")

    ax1.set_ylim(0,250); 
    ax1.legend(title=f"{labels['varName']} {labels['varUnit']}", \
               loc=legendLoc, title_fontsize=16)#, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax1.grid(alpha=0.1); ax2.grid(alpha =0.1)

    ax1.set_ylabel("Distance Above Inlet [m]")
    ax1.set_xlabel(labels['ax1_xLabel'])
    ax2.set_xlabel(labels['ax2_xLabel'])
    ax2.set_ylabel("Ratio")

    ax2.scatter(vars, ratios_mdot, c=vars.ravel(), cmap=colormap, marker="*", s=150)
    ax2.scatter(vars, ratios_u, c=vars.ravel(), cmap=colormap, s=150)
    if log:
        ax2.set_xscale("log")

    current_ylims = ax2.get_ylim()
    ax2.set_ylim(current_ylims)

    current_xlims = ax2.get_xlim()
    ax2.set_xlim(current_xlims)

    ax2.scatter([0,1], [-60,-50],color = 'gray', marker = '*', s=150, label=r"$\dot{m}_{DPM}$/$\dot{m}_{SPM}$")
    ax2.scatter([0,1], [-60,-50],color = 'gray', marker = 'o', s=150, label=r"$u_{DPM}$/$u_{SPM}$")
    ax2.legend();
    
    plt.tight_layout()

    axes = (ax_dict["A"], ax_dict["B"])

    # 2. Iterate directly over the tuple (no .flat needed)
    for ax, label in zip(axes, string.ascii_lowercase):
        ax.text(
            0.02, 0.98, label, 
            transform=ax.transAxes, 
            fontsize=16, 
            va='top', 
            ha='left'
        )

    if save:
        fig.savefig(figName, dpi = 300)

    return fig, ax_dict

def make_comparison_figure(yi, zi, yo, zo, y, z, mdot_sp, mdot_dp, ambient, c=c, p=p, save=False, figName=None):
    sec_to_day = 60 * 60 * 24

    ui, Ti, Si, bi, uo, To, So, bo, zo, rho_i, rho_o, rho_a  = getVars_fromModel(y=None, yi=yi, yo=yo, z=None, zi=zi, zo=zo, ambient=ambient, model="double_plume", c=c, p=p)
    u, b, T, S, rho, rho_a = getVars_fromModel(y=y, yi=None, yo=None, z=z, zi=None, zo=None, ambient=ambient, model="single_plume", c=c, p=p)

    fig, ax = plt.subplots(2,3,figsize=(15,14))
    cmap = cm.haline(np.linspace(0,1,5))

    dp_color = cmap[1]
    sp_color = cmap[3]
    a_color = 'k'
    lw = 2.5

    ax[0,0].plot(b[:-50], z[:-50]-250-6, color = sp_color, linewidth = lw)
    ax[0,0].plot(bi, zi-250, color = dp_color, linewidth = lw)

    ax[0,1].plot(u, z-250-6, color = sp_color, linewidth = lw)
    ax[0,1].plot(ui, zi-250, color = dp_color, linewidth = lw)
    ax[0,1].plot(uo, zo-250, color = dp_color, linestyle = ':', linewidth = lw)
    ax[0,1].plot(np.zeros(len(ambient[0,:])), ambient[0,:][::-1]-250, color = a_color, alpha = 0.5, linewidth = 1.5)

    ax[0,2].plot(mdot_sp * sec_to_day, z-250-6, color = sp_color, linewidth = lw)
    ax[0,2].plot(mdot_dp * sec_to_day, zi-250, color = dp_color, linewidth = lw)

    ax[1,0].plot(T, z-250-6, color = sp_color, linewidth = lw)
    ax[1,0].plot(Ti, zi-250, color = dp_color, linewidth = lw)
    ax[1,0].plot(To, zo-250, color = dp_color, linestyle = ':', linewidth = lw)
    ax[1,0].plot(ambient[1,:], ambient[0,:]-250, color = a_color, alpha = 0.5, linewidth = 1.5)

    ax[1,1].plot(S, z-250-6, color = sp_color, linewidth = lw)
    ax[1,1].plot(Si, zi-250, color = dp_color, linewidth = lw)
    ax[1,1].plot(So, zo-250, color = dp_color, linestyle = ':', linewidth = lw)
    ax[1,1].plot(ambient[2,:], ambient[0,:]-250, color = a_color, alpha = 0.5, linewidth = 1.5)

    ax[1,2].plot(rho_a, ambient[0,:]-250, color = a_color, alpha = 0.5, linewidth = 1.5, label = "Ambient")
    ax[1,2].plot(rho, z-250-6, color = sp_color, linewidth = lw, label = "SPM")
    ax[1,2].plot(rho_i, zi-250, color = dp_color, linewidth = lw, label = "DPM Inner")
    ax[1,2].plot(rho_o, zo-250, color = dp_color, linestyle = ':', linewidth = lw, label = "DPM Outer")
    ax[1,2].set_xlim(1018,1025)


    for i in range(0,2):
        for j in range(0,3):
            ax[i,j].set_ylim(-ambient[0,-1],0)
            ax[i,j].grid(alpha = 0.25)
            ax[i,j].set_yticks([-250,-200,-150,-100,-50,0])
            ax[i,j].set_yticklabels([-250,-200,-150,-100,-50,0], fontsize = 16)

    ax[0,0].set_ylabel('Depth [m]', fontsize = 16)
    ax[0,0].set_xlabel('Plume Width [m]', fontsize = 16)
    ax[0,1].set_xlabel('Velocity [m s$^{-1}$]', fontsize = 16)
    ax[0,2].set_xlabel('Melt Rate [m d$^{-1}$]', fontsize = 16)

    ax[1,0].set_ylabel('Depth [m]', fontsize = 16)
    ax[1,0].set_xlabel('Temperature [$^\circ$C]', fontsize = 16)
    ax[1,1].set_xlabel('Salinity [ppt]', fontsize = 16)
    ax[1,2].set_xlabel('Density [kg m$^{-3}$]', fontsize = 16)
    ax[1,2].legend(loc = 1, fontsize = 14)


    if save:
        plt.savefig(figName, dpi = 300)



def make_bubble_figure(yi=None, zi=None, yo=None, zo=None, ambient=None, fig=None, ax=None, color='blue', save=False, figName=None):
    ui, Mb, Mb_frac, rho_b, d_eff, Ci, Co, Cs, us, beta, beta_T = getBubbleProperties(yi=yi, zi=zi, yo=yo, zo=zo, ambient=ambient)
    # ---- Combined figure ----
    if fig is None:
        fig, ax = plt.subplots(2, 5, figsize=(28, 11))
    lw = 2.5

    # Row 1: bulk plume / bubble state
    ax[0,0].plot(ui, zi - 250, color=color, linewidth=lw)
    ax[0,0].set_xlabel('Velocity $u_i$ [m s$^{-1}$]', fontsize=14)

    ax[0,1].plot(Mb, zi - 250, color=color, linewidth=lw)
    ax[0,1].set_xlabel('Mass Flux $O_2$ [kg s$^{-1}$]', fontsize=14)

    ax[0,2].plot(Mb_frac, zi - 250, color=color, linewidth=lw)
    ax[0,2].axvline(gas.fdis, color='k', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'$f_{{dis}}$ = {gas.fdis}')
    ax[0,2].legend(fontsize=10)
    ax[0,2].set_xlabel('Bubble Mass Fraction Remaining $m_b/m_{b,0}$ [-]', fontsize=14)

    ax[0,3].plot(rho_b, zi - 250, color=color, linewidth=lw)
    ax[0,3].set_xlabel('Bubble Density [kg m$^{-3}$]', fontsize=14)

    ax[0,4].plot(d_eff, zi - 250, color=color, linewidth=lw)
    ax[0,4].set_xlabel('Effective Diameter $d_e$ [mm]', fontsize=14)

    # Row 2: concentration / solubility / transfer coefficients
    ax[1,0].plot(Ci, zi - 250, color=color, linewidth=lw, label='$C_i$ (in-plume)')
    ax[1,0].plot(Co, zo - 250, color=color, linestyle=':', linewidth=lw, label='$C_o$ (peeled)')
    ax[1,0].legend(fontsize=10)
    ax[1,0].set_xlabel('$O_2$ Concentration [kg m$^{-3}$]', fontsize=14)

    ax[1,1].plot(Cs, zi - 250, color=color, linewidth=lw)
    ax[1,1].set_xlabel('$O_2$ Solubility $C_s$ [kg m$^{-3}$]', fontsize=14)

    ax[1,2].plot(us, zi - 250, color=color, linewidth=lw)
    ax[1,2].ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax[1,2].set_xlabel('Slip Velocity [m s$^{-1}$]', fontsize=14)

    ax[1,3].plot(beta, zi - 250, color=color, linewidth=lw)
    ax[1,3].ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax[1,3].set_xlabel(r'Gas Mass Transfer Coeff. $\beta$ [m s$^{-1}$]', fontsize=14)

    ax[1,4].plot(beta_T, zi - 250, color=color, linewidth=lw)
    ax[1,4].ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax[1,4].set_xlabel(r'Heat Transfer $\beta_T $ [m s$^{-1}$]', fontsize=14)

    ax[0,0].set_ylabel('Depth [m]', fontsize=16)
    ax[1,0].set_ylabel('Depth [m]', fontsize=16)

    for i in range(2):
        for j in range(5):
            ax[i,j].set_ylim(-ambient[0,-1], 0)
            ax[i,j].grid(alpha=0.25)

    plt.tight_layout()
    if save:
        plt.savefig(figName, dpi=300)

    return fig, ax

def getBubbleProperties(yi=None, zi=None, yo=None, zo=None, ambient=None):

    model = dP.__new__(dP)   # lightweight instance, no ODE re-solve

    # ---- Derived plume/bubble variables ----
    ui, Ti, Si, bi, uo, To, So, bo, zo, rho_i, rho_o, rho_a = getVars_fromModel(
        y=None, yi=yi, yo=yo, z=None, zi=zi, zo=zo, ambient=ambient,
        model="double_plume", c=c, p=p)

    Mb = yi[5]
    Mb0 = Mb[0]
    Mb_frac = Mb / Mb0

    ui = yi[1] / yi[0]
    Ci = yi[6] / yi[0]
    Co = yo[4] / yo[0]
    Co = Co[::-1]

    Pa_interp = interp1d(ambient[0, :], ambient[-2, :])
    Pa = Pa_interp(zi)
    rho_b = model.gas_density(Ti, Pa, p)

    # self.nb0 is normally set inside __init__ from the run's source conditions;
    # since we bypass __init__ here, reconstruct it directly from yi/zi/ambient.
    rho_b0 = rho_b[0]
    model.nb0 = 6 * Mb0 / (np.pi * rho_b0 * (gas.de[0]/1000)**3)

    d_eff = np.array([
        model.bubble_diameter(Mb[i], rho_b[i]) for i in range(len(zi))
    ]) * 1000   # m -> mm

    nb = 6 * (Mb / rho_b) / (np.pi * (gas.de[0]/1000)**3)

    rho_a_interp = interp1d(ambient[0, :], rho_a)
    rho_a_l = rho_a_interp(zi)

    Cs = model.gas_SOL(rho_a_l, Ti, Si, Pa, Mb/rho_b, p)

    us = np.zeros(len(zi))
    beta = np.zeros(len(zi))
    beta_T = np.zeros(len(zi))

    for i in range(len(zi)):
        us[i]     = model.u_slip(rho_a_l[i], rho_b[i], Ti[i], gas.de[0]/1000, p)
        beta[i]   = float(np.squeeze(model.gas_MT(Ti[i], rho_a_l[i], Mb[i], us[i], p)))
        beta_T[i] = model.gas_HT(Ti[i], rho_a_l[i], us[i], p)

    return ui, Mb, Mb_frac, rho_b, d_eff, Ci, Co, Cs, us, beta, beta_T

def make_solArray(innerFile, outerFile):
    df = pd.read_csv(innerFile, index_col=0)
    zi = df["zi"]
    Nz = len(zi)
    yi = np.zeros((7,Nz))

    for i in range(0,7):
        yi[i,:] = df.iloc[:,i]

    df = pd.read_csv(outerFile, index_col=0)
    zo = df["zo"]
    Nz = len(zo)
    yo = np.zeros((5,Nz))

    for i in range(0,5):
        yo[i,:] = df.iloc[:,i]

    return yi, zi, yo, zo

def parse_param(filepath, pname):
    match = re.search(rf"{pname}_?(\d+\.?\d*)", filepath)
    return float(match.group(1)) if match else np.nan


def scale_size(values, s_min=100, s_max=200):
    v = np.asarray(values, dtype=float)
    norm = (v - v.min()) / (v.max() - v.min())
    return s_min + norm * (s_max - s_min)




def make_figure_Qg(innerFiles, singleFile, labels={},  c=c, p=p,
                ratio=True, getMax=True, getMean=False, nProfiles=5, 
                log=False, colormap = cm.matter,
                fig=None, ax_dict=None, save=False, figName=None, legendLoc=5):
    ratios_mdot, ratios_u, vars = getSensitivity(innerFiles, singleFile[0], ratio=ratio, getMax=getMax, getMean=getMean, gasTest=True)
    
    if labels["varName"] == "Radius":
            vars = np.arange(1,100,10)

    if fig is None:
        fig, ax_dict = plt.subplot_mosaic([['A', 'B', 'B']], figsize=(20,8))

    colormap = cmocean.tools.crop_by_percent(colormap, 20, which='min')

    ax1 = ax_dict['A']
    ax2 = ax_dict['B']
    nFiles = len(innerFiles)
    skip = int(nFiles / nProfiles) + 1
    cmap = colormap(np.linspace(0,1,nFiles))
    u, b, T, S, mdot, z = getVars(pd.read_csv(singleFile[0], index_col=0), "single_plume",c, p)
    ax1.plot(u, z, color = 'k', linestyle =':')
    for i in range(0, nFiles, skip):
        ui, bi, Ti, Si, mb, ci, mdot, zi, _ = getVars(pd.read_csv(innerFiles[i], index_col=0), "double_plume",c, p)
        ax1.plot(ui, zi, color = cmap[i], linestyle='-', linewidth = 2, label= vars[i].round(2))
        

    ax1.plot([0,1], [300,350],color = 'gray', linestyle = '-', label="DPM")
    ax1.plot([0,1], [300,350],color = 'gray', linestyle = ':', label="SPM")

    ax1.set_ylim(0,250); 
    ax1.legend(title=f"{labels['varName']} {labels['varUnit']}", \
               loc=legendLoc, title_fontsize=16)#, bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax1.grid(alpha=0.1); ax2.grid(alpha =0.1)

    ax1.set_ylabel("Distance Above Inlet [m]")
    ax1.set_xlabel(labels['ax1_xLabel'])
    ax2.set_xlabel(labels['ax2_xLabel'])
    ax2.set_ylabel("Ratio")

    ax2.scatter(vars, ratios_mdot, c=vars.ravel(), cmap=colormap, marker="*", s=150)
    ax2.scatter(vars, ratios_u, c=vars.ravel(), cmap=colormap, s=150)
    if log:
        ax2.set_xscale("log")

    current_ylims = ax2.get_ylim()
    ax2.set_ylim(current_ylims)

    current_xlims = ax2.get_xlim()
    ax2.set_xlim(current_xlims)

    ax2.scatter([0,1], [-60,-50],color = 'gray', marker = '*', s=150, label=r"$\dot{m}_{DPM}$/$\dot{m}_{SPM}$")
    ax2.scatter([0,1], [-60,-50],color = 'gray', marker = 'o', s=150, label=r"$u_{DPM}$/$u_{SPM}$")
    ax2.legend();
    
    plt.tight_layout()
    axes = (ax_dict["A"], ax_dict["B"])
    
    # 2. Iterate directly over the tuple (no .flat needed)
    for ax, label in zip(axes, string.ascii_lowercase):
        ax.text(
            0.02, 0.98, label, 
            transform=ax.transAxes, 
            fontsize=16, 
            va='top', 
            ha='left'
        )

    if save:
        fig.savefig(figName, dpi = 300)

    return fig, ax_dict