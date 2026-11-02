import math

rho_r = 1031                            # Reference fluid density in [kg/m^3]
Sa_filePath = '/Users/madelinemamer/Documents/bubblePlume/ambient/saltini.bin'
Ta_filePath = '/Users/madelinemamer/Documents/bubblePlume/ambient/tempini.bin'
## Parameters to change
modelType = 'double_plume'
Qsg = 0                                 # Initial Vol. Flux of subglacial discharge m^3/s
Usg = 0
C0 = 0.005                              # Initial Concentration of O2, kg/m^3
S0 = 0                                  # Initial salinity of subglacial discharge in psu
R = 0.05                                # Radius of the pipe in [m]
mb0 = 0.002

## Melt parameterization parameters
Cd = 0.001
gammaT = 2.20e-2
gammaS = 6.2e-4

## Double Plume Model parameters
u0 = 1.1/10                            # A guess value of the initial plume velocity in [m/s];
alpha_1 = 0.055                        # Entrainment coefficient of the inner plume [--]
alpha_2 = 0.110                        # Entrainment coefficient of the outer plume to the inner plume[--]
alpha_3 = 0.110                        # Entrainment coefficient of the ambient water to the outer plume[--]
alpha_sp = 0.110                       # Entrainment coefficient of the single plume
lambda_s = 1.10                        # Ratio of spreading widths betweem conc. and velocity [--]
epsilon = 0.0175                       # Peeling parameter in the continuous peel model for Ep [--]
c1 = 0                                 # Recirculation model parameter
c2 = 1                                 # Model parameter in the continuous peeling term Ep [--]
gamma_i = 1.10 
gamma_o = 1.10 
kappa_peel = 2          
fe = 0.1                               # Fraction of ambient fluid entrained in outplume surface engulfing
fp = 0.95                              # Fraction of the inner plume volume flux that is detrained.
nwidths = 1                             # [--]