import math

#rho_r = 1025
g = 9.81
pi = math.pi

c1 = -1.0
c4 = 0.1
eps = 0.015
us = 0.1
alpha_a = 0.110
alpha_o = 0.110
alpha_i = 0.055

Cd = 0.001
gammaT = 2.20e-2
gammaS = 6.2e-4

cw = 4000
L_i = 334000
lam1 = -5.73e-2
lam2 = 0.08
lam3 = -7.61e-4

## Momentum Amplification factors and spreading factor 
gamma_i = 1.10
gamma_o = 1.10
lam = 1.0

rho_b = 1.4

## From Chris' in model
H = 100;                                                               # Depth at the pipe exit in [m]
Fr0 = 1.6;                                                             # Target Fr # to initiate the first inner plume [--]
u0 = 1.1/10;                                                           # A guess value of the initial plume velocity in [m/s]; this needs to determined on a case-by-case basis 
                                                                        # (following from the previous line), an inappropriate p.u0 will cause the program to crash when the function
                                                                        # pre_proc.m is executed in the main.m function
rho_r = 1031;                                                          # Reference fluid density in [kg/m^3]
alpha_1 = 0.055;                                                       # Entrainment coefficient of the inner plume [--]
alpha_2 = 0.110;                                                       # Entrainment coefficient of the outer plume to the inner plume[--]
alpha_3 = 0.110;                                                       # Entrainment coefficient of the ambient water to the outer plume[--]
lambda_2 = 1.10;                                                       # Ratio of spreading widths betweem conc. and velocity [--]
epsilon = 0.0175;                                                      # Peeling parameter in the continuous peel model for Ep [--]
c1 = 0;                                                                # Model parameter in the entrainment term Ei [--]
c2 = 1;                                                                # Model parameter in the continuous peeling term Ep [--]
H0 = 0
fe = 0.1;                                                              # Volume flux of mbient water added to the inner plume at the peel height, and
                                                                        # the total is used as the initial volume flux of the outer plume calculation 
                                                                        # This is expressed as a fraction of the inner plume volume flux at the peel height 
                                                                        
fp = 0.95;                                                             # Fraction of the inner plume volume flux that is detrained.
                                                                        # This is only used when the discrete peel model is activated by setting p.c2 = 0 

#gamma_i = 1.10;                                                        # Momentum amplification factor for the inner plume 
#gamma_o = 1.10;                                                        # Momentum amplification factor for the outer plume 
nwidths = 1;  # [--]
naverage = 1;  # [--]
MT_method = 'rigid';                                                   # Options, [wueest, rigid, or fluid], for calculating mass transfer coefficients 

## Parameters to change
modelType = 'single_plume'
Qsg = 0                                                                 # Initial Vol. Flux of subglacial discharge m^3/s
Usg = 0
C0 = 0.005                                                              # Initial Concentration of O2, kg/m^3
S0 = 0                                                                  # Initial salinity of subglacial discharge in psu
R = 0.05                                                                # Radius of the pipe in [m]
mb0 = 0.002
l = 10