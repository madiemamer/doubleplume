import numpy as np
import math

pi = math.pi            # pi
g = 9.81                # gravitational acceleration m/s^2
Patm = 101325           # Atmospheric pressure in Pa
T0 = 273.15             # Temperature at 0degC in K
cp = 4000               # Heat capacity of seawater at 5degC J/kg K
L_i = 334000            # Latent heat of ice J/kg

## Melt parameterization Constants
lam1 = -5.73e-2         # Salinity coef C/ppt
lam2 = 0.08             # intercept C
lam3 = -7.61e-4         # Pressure coef m/ppt

## Gas Constants
Ru = 8.314472           # Universal gas constant in J/(mol K)
cp_g = 0.53 * cp        # Heat capacity of gas in J/kg K

# Gas molecular weights in kg/mol
M_gas = np.array([
                    [[2*15.9994]],                      # Oxygen (O2)
                    # [[2*14.0067]],                    # Nitrogen (N2)
                    # [[12.0110 + 4*1.0079]],           # Methane (CH4)
                    # [[2*12.0110 + 6*1.0079]],         # Ethane (C2H6)
                    # [[3*12.0110 + 8*1.0079]],         # Propane (C3H8)
]) / 1000  

# Pressure at the critical point in Pa 
Pcr_gas = np.array([         
                        5042825.270])
                        #3399804.677, 
                        #4594666.065,
                        #4871145.821,
                        #4247170.312]) 

# Temperature at the critical point in K
Tcr_gas = np.array([          
                154.58])
                #126.20,
                #190.56,
                #305.33,
                #369.85]) 

# Eccentricity factor [--]
omega = np.array([           
                0.0216])
                #0.0372,
                #0.0104,
                #0.0979,
                #0.1522]) 

# Heat of dissolution in J/kg 
dhSol_gas= np.array([        
                428733])
                #385846,
                #906985,
                #649800,
                #509096])

