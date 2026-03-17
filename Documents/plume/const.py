import numpy as np
## Constants for DoublePlume model

Ru = 8.314472 # Universal gas constant in J/(mol K)
Patm = 101325 #Atmospheric pressure in Pa
T0 = 273.15 #Temperature at 0degC in K
Ts = 0. #Temperature at STP in degC
cp = 4185.5 #Heat capacity of water in J/kgK NEED TO UPDATE
cp_o = 0.55 * cp # heat capacity of oil in J/kg K
cp_g = 0.53 * cp # heat capacity of gas in J/kg K
M_drop = np.array([0, # Droplet molecular weights in kg/mol
                        0, 
                        12.0110 + 2 * 15.9994])/1000
dhSol_drop = np.array([0, #Heat of dissolution in J/kg
                            0,
                            568000])
M_gas = np.array([         # Gas molecular weights in kg/mol
                    [2*15.9994]])/1000,                 # Oxygen (O2)                 
                    #[2*14.0067],                 # Nitrogen (N2)
                    #[12.0110 + 4*1.0079],        # Methane (CH4)
                    #[2*12.0110 + 6*1.0079],      # Ethane (C2H6)
                    #[3*12.0110 + 8*1.0079]])/1000

Pcr_gas = np.array([         # Pressure at the critical point in Pa 
                        5042825.270])
                        #3399804.677, 
                        #4594666.065,
                        #4871145.821,
                        #4247170.312]) 
Tcr_gas = np.array([         # Temperature at the critical point in K 
                154.58])
                #126.20,
                #190.56,
                #305.33,
                #369.85]) 
omega = np.array([           # Eccentricity factor [--]
                0.0216])
                #0.0372,
                #0.0104,
                #0.0979,
                #0.1522]) 
dhSol_gas= np.array([       # Heat of dissolution in J/kg  
                428733])
                #385846,
                #906985,
                #649800,
                #509096])
Pconv = 100000;        # Convert pressure in ambient matrix to Pa
g = 9.81;              # Acceleration due to gravity
vCO2 = 7.05e-4;        # Specific volume of CO2 in seawater as m^3/kg
