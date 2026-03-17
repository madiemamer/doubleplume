import numpy as np

# Continue with the gas data ----------------------------------------------
nbub = 1;                                                            # Number of different bubble SIZE CLASS
de = np.array([10.0]);                                                        # Diameter of bubbles in [mm]
K = np.array([1.0]);                                                           # Solubility of bubbles in [--]
lambda_1 = np.array([0.80]);                                                   # Ratio of spreading widths between bubbles and velocity [--]
fdis = np.array([0.001]);                                                      # Threshold below which the bubbles are considered dissolved [--]
mf = np.array([1.0]);                                                          # Mass fraction of all gases in each bubble size class  
q = np.array([50*(0.0353147)*(1/60)]);                                 # total gas volume at standard STP conditions in units of [SCF]
T = np.array([20]);      

nc = 5;                                                              # Number of components in the bubbles; there are five [O2] IGNORE:, N2, CH4, C2H6, C3H8]
f = np.array([[1.0], [0.0], [0], [0], [0]]);                                             # Mole fractions at the water surface
f0 = np.array([[1.0], [0.0], [0], [0], [0]])                                # Mole fractions present at the pipe exit

on = True
mb = np.zeros_like(f)
rho_b = np.zeros_like(f)
mb0 = np.zeros_like(f)
nb = np.zeros_like(f)
xi = np.zeros_like(f)
us = np.zeros_like(f)
beta = np.zeros_like(f)
Cs = np.zeros_like(f)
beta_T = 0