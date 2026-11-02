import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import gsw
import model.const as c
import model.gas as gas

class doublePlume:

    def __init__(self, ambient, p):
        rho_a = gsw.density.rho_t_exact(ambient[2,0], ambient[1,0], (ambient[3,0] - 101325)/10**4)
        if c.lam3 > 0:
            raise ValueError("c.lam3 must be negative when the liquidus uses depth (positive down), e.g. -7.61e-4 degC/m")
        #ambient[0,:] : z
        #ambient[1,:] : Ta
        #ambient[2,:] : Sa
        #ambient[3,:] : Pa
        #ambient[4,:] : Co O2

        if p.modelType == 'single_plume':

            Q0 = 0.5 * p.Usg * p.R**2                                    # Initial Vol. Flux m^3/s
            M0 = Q0 * p.Usg                                              # Initial Momentum m^4/s^2
            C0 = p.C0                                                    # Initial Concentration kg/m^3
            S0 = p.S0                                                    # Initial Salinity psu                                                     
            T0 = self.boundary_temperature(S0, ambient[0,-1] - ambient[0,0], p)         # Initial Temperature C
            
            # Initialize solution vector
            y0 = np.zeros(5)
            y0[0] = Q0
            y0[1] = M0
            y0[2] = T0 * Q0 
            y0[3] = S0 * Q0
            y0[4] = C0 * Q0

            # Call model
            y, z = self.singlePlume_calc(y0, ambient, p)

            # Save solution space
            self.yi = y
            self.zi = z
            self.yo = 0
            self.zo = 0
            self.jp = 0
            self.i_peel = 0

        else:
            gas.de = np.array(gas.de0, dtype=float)

            # Build empty outer plume array
            neighbor = np.zeros((5,2))
            zo = np.array([ambient[0,0], ambient[0,-1]])

            # Get initial conditions
            Qsg = p.Qsg                                                  # Initial subglacial liquid flux, m^3/s
            mb0 = p.mb0                                                  # Initial bubble flux kg/s
            C0 = p.C0                                                    # Initial concentration kg/m^3
            S0 = p.S0                                                    # Initial Salinity psu                                                     
            T0 = self.boundary_temperature(S0, ambient[0,-1] - ambient[0,0], p)   # Initial Temp, local freezing point [C]        # Initial Temp, local freezing point [C]
            rho_b = self.gas_density(T0, ambient[3,0], p)                # Bubble density kg/m^3
            gas.T = T0                                                   # Set initial gas T to be same as subglacial fluid
            Qg = mb0/rho_b  
            # In __init__, after rho_b and mb0 are computed:
            self.nb0 = 6 * mb0 / (rho_b * c.pi * (gas.de0[0]/1000)**3)   #                                             # Initial gas mass flux m^3/s

            us = self.u_slip(rho_a, rho_b, T0, gas.de/1000, p)                # Slip Velocity m/s
            Q0, J0, z0 = self.main_ic(mb0, Qg, Qsg, p.R, rho_a,          # Initial Vol. Flux [m^3/s], Mom. Flux [m^4/s^2], starting z [m]
                                      rho_b, us, p)   
            Xi0 = self.void_fraction(mb0, rho_b, p.R, us, Q0 / (0.5 * c.pi * p.R**2), p)
            if Xi0 > 0.5:
                print("Warning: source void fraction is unphysical; source is too lazy for this R")                   
            mb0 = rho_b * Qg       
            # nd = 6 * (mb0/rho_b)/(c.pi * (gas.de/1000)**3)              # Number of bubbles per second [#/s]
            print(f"In initial, mb0:{mb0}, Qg:{Qg}")
            # Initialize integration vector
            y0 = np.zeros(7)

            y0[0] = Q0                                                   # Vol. Flux m^3/s
            y0[1] = J0                                                   # Mom. Flux m^4/s^2
            y0[2] = p.rho_r * c.cp * (T0 + c.T0) * Q0                    # Heat Flux J/s
            y0[3] = mb0 * c.cp_g * (gas.T + c.T0)                        # Bubble Heat Flux J/s
            y0[4] = S0 * Q0                                              # Salt Flux m^3 psu /s
            y0[5] = mb0                                                  # Bubble mass Flux kg/s
            y0[6] = C0 * Q0                                              # Concentration flux kg/s
                    
            self.printInitialConditions(y0, p)

            # Set integration conditions
            maxIter = 10                                                       
            err = 1000
            iter = 0
            
            # Evaluate if ambient conditions are stratified
            rho_a_all = gsw.density.rho_t_exact(ambient[2,:], ambient[1,:], 0)
            N = np.sqrt(-c.g/p.rho_r * np.gradient(rho_a_all, ambient[0,:]))

            # If stratification exists in the ambient, run the double plume model
            if np.nanmean(N) > 0:
                while err > 0.01 and iter <= maxIter:
                    
                    # Call inner model
                    yi, zi, jp, i_peel = self.inner_calc(ambient, zo, neighbor, y0, p)

                    # Update neighbor object
                    neighbor = yi

                    # Call outer model
                    yo, zo, = self.outer_calc(ambient, zi, yi, neighbor, jp, i_peel, p)

                    # Update neighbor object
                    neighbor = yo

                                       # Evaluate convergence
                    zp = zi[i_peel] if jp > 0 else np.array([])
                    if iter == 0:
                        Qi_0, zi_0, zp_0 = yi[0], zi, zp
                    else:
                        # Integrated (L1) change in inner volume flux on the common height range.
                        # Robust to small shifts in the discontinuities at peels.
                        z_c = np.linspace(max(zi[0], zi_0[0]), min(zi[-1], zi_0[-1]), 2000)
                        Q_new = interp1d(zi, yi[0])(z_c)
                        Q_old = interp1d(zi_0, Qi_0)(z_c)
                        err_Q = np.trapezoid(np.abs(Q_new - Q_old), z_c) / np.trapezoid(np.abs(Q_old), z_c)

                        # Change in peel heights and plume top, relative to the rise height
                        H_rise = zi[-1] - zi[0]
                        if zp.size == zp_0.size:
                            err_zp = np.max(np.abs(zp - zp_0)) / H_rise if zp.size else 0.0
                        else:
                            err_zp = 1.0                        # number of peels changed: not converged
                        err_top = abs(zi[-1] - zi_0[-1]) / H_rise

                        err = max(err_Q, err_zp, err_top)
                        print(f"Iter {iter}: dQ(L1) = {err_Q:.2%}, peel shift = {err_zp:.2%}, "
                              f"top shift = {err_top:.2%}, peels = {jp}")
                        Qi_0, zi_0, zp_0 = yi[0], zi, zp
                    iter += 1

                if iter > maxIter:
                    print(f"Exceeded max iterations; keeping last iteration")
                    
            # If no ambient stratification, run the single plume.
            else:
                print(f"No ambient stratification, running the inner plume model only.")

                # Call inner plume model with no outer plume
                yi, zi, jp, i_peel = self.inner_calc(ambient, zo, neighbor, y0, p)
                yo = np.zeros((5, len(zi)))
                zo = zi.copy()
            
            # Set global variables 
            self.yi = yi
            self.zi = zi
            self.yo = yo
            self.zo = zo
            self.jp = jp
            self.i_peel = i_peel
        return 
    
    def returnResults(self):
        return self.yi, self.zi, self.yo, self.zo, self.jp, self.i_peel
    
    def printInitialConditions(self, y0, p):
        
        print("=== Plume Initial Conditions ===")
        print(f"Qi={y0[0]:.3f} m^3/s, Ji={y0[1]:.3f} m^4/s^2, Hi={y0[2]:.3f} m^3°C/s, Bi={y0[4]:.3f} m^3/s psu, Ci={y0[6]:.6f} kg/s")
        print(f"ui={y0[1]/y0[0]:.3f} m/s, bi={2**(0.5) * y0[0]/np.sqrt(y0[1] * c.pi):.3f} m, \
              Ti={y0[2]/(p.rho_r * c.cp * y0[0]) - c.T0:.3f} °C, Si={y0[4]/y0[0]:.3f} m^3/s psu, Ci={y0[6]/y0[0]:.6f} kg/m^3")

        return
    
    def melting(self, y, z, p):

        # Evaluate the ice wall melt rate. Model from
        # Holland & Jenkins 1999.
        #--------------------------------------------#

        if p.modelType == 'single_plume':
            Q, J, H, B, C = y
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

    def boundary_salinity(self, y, m_dot, p):

        # Evaluate ice boundary salinity based on
        # salt conservation at the boundary. From
        # Holland & Jenkins 1999.
        #--------------------------------------------#

        if p.modelType == 'single_plume':
            Q, J, H, B, C = y
            T = H/Q 
        else:
            Q, J, H, Hb, B, mb, C = y
            T = H/(p.rho_r * c.cp * Q) - c.T0

        return p.gammaS * p.Cd**(1/2) * J/Q * B/Q / (m_dot + p.gammaS * p.Cd**(1/2) * J/Q)

    def boundary_temperature(self, Sb, z, p):

        # Liquidus condition, from Holland & Jenkins 1999
        #--------------------------------------------#

        return c.lam1 * Sb + c.lam2 + c.lam3 * z
    
    def cp_model(self, ui, us, bi, rho_i, rho_a, p):

        # Peeling flux, from Socolofsky et al. 2008
        # Added 1/2 to account for half cone geometry
        #--------------------------------------------#

        # print(f"c2: {p.c2}, epsilon: {p.epsilon}, us:{us}, ui:{ui}, ")
        # print(f"g:{c.g}, rho_a:{rho_a}, rho_i: {rho_i}")
        Ep = p.c2 * p.epsilon * (us/ui)**2 * c.g * (rho_a - rho_i)/p.rho_r * c.pi * bi**2/(2*ui)   # [m^2/s]
        if Ep > 0:
            return 0
        else:
            return Ep
        
    def gas_MT(self, T, rho_a, mb, us, p):

        # Evaluate the gas mass transfer coefficient
        #--------------------------------------------#
        
        #Use eqns. in Clift et al. (1979) for rigid particles
        mu_w = 1721.2*np.exp(-0.0251*T)/1e6                              # Pa s
        
        #Calcuate the diffusion coefficient for each gas in m^2/s from Wise
        #and Houghton (1966). These are for dilute, binary systems. 
        B = np.array([4.2])*1e-2#, 7.9, 7.0, 2.0, 3.1]).T*1e-2;          # [cm^2/s]
        dE = np.array([4390])/0.23886#, 4690, 4640, 3980, 4350]).T/0.238846; # [J/mol]
        D = B*np.exp(-dE/(c.Ru*(c.T0 + T)))/100**2;                      # [m^2/s]
        
        #Calculate the Schmidt number 
        Sc = mu_w/(rho_a*D); # NOT A vector
        
        #Calculate the Peclet number
        Pe = us*(gas.de/1000)/D; # Is de in right units?
        
        #Calculate the Reynolds number 
        Re = rho_a*us*(gas.de/1000)/mu_w; 
        
        #Calculate the Sherwood number for a rigid particle
        Sh = 0
        if Re < 100:
            Sh = 1 + (1 + 1/Pe)**(1/3)*Re**0.41*Sc**(1/3); 
        elif Re < 2000:
            Sh = 1 + 0.724*Re**0.48*Sc**(1/3); 
        else:
            Sh = 1 + 0.425*Re**0.55*Sc**(1/3); 
            
        beta = gas.K*Sh*D/(gas.de/1000);                                 # [m/s]

        return beta

    def gas_SOL(self, rho_a, T, S, P, mb, p):

        # Evaluate the gas solubility coefficient.
        #--------------------------------------------#
        
        #Convert pressure and Temp to absolute
        P = P
        T = T + c.T0

        #Henry's law coef
        H0 = np.array([1.3]) * 1e-3*1000/101325  * c.M_gas[0][0][0] #,0.63,1.21,1.90,1.50]).T * 1e-3*1000/101325 * c.M_gas #[kg/Pa m^3]

        #Partial molar volume at infinite dilution
        nu_bar = np.array([32])/1e6 #,33,37.3,35,35]).T / 1e6                # m^3/mol              

        #Heat of Solution [K]
        dH_solR = c.dhSol_gas / c.Ru * c.M_gas[0][0][0]

        #Compute the temperature effect
        T_0 = 298.15
        H0 = H0 * np.exp(dH_solR * (1/T - 1/T_0))

        # #Compute the salting-out effect. These numbers are for CO2 in water from Weiss 1974
        # Salting-out effect for O2 (Weiss 1970), T in K, S practical salinity
        H0 = H0 * np.exp(S * (-0.033096 + 0.014259 * T/100 - 0.0017000 * (T/100)**2))
        moles = mb/c.M_gas[0][0][0]                                          # Molar flux [mol/s]
        # Pk = P * moles                                          

        # Compute the pressure effect
        H = H0 * np.exp(-(P - 101325) * nu_bar/(c.Ru * T))                  # [kg/Pa m^3]                              # [kg/Pa m^3]
        
        #Compute Peng-Robinson mixture coeffs.
        b = 0.07780 * c.Ru * c.Tcr_gas/c.Pcr_gas; 
        ac = 0.45724 * c.Ru**2 * c.Tcr_gas**2/c.Pcr_gas; 
        mu = 0.37464 + 1.54226 * c.omega - 0.26992 * c.omega**2; 
        alpha = (1 + mu * (1 - np.sqrt(T/c.Tcr_gas)))**2; 
        a = alpha*ac

        A = a * P / (c.Ru**2 * T**2)
        B = b * P / (c.Ru * T)

        ## Compressibility
        # coefficients for cubic in Z: Z^3 + c2 Z^2 + c1 Z + c0 = 0
        c2 = -(1.0 - B)
        c1 = (A - 3.0*B**2 - 2.0*B)
        c0 = -(A*B - B**2 - B**3)

        coef = [1.0, c2[0], c1[0], c0[0]]
        roots = np.roots(coef)

        real_roots = np.real(roots[np.isclose(roots.imag, 0.0, atol=1e-10)])
        if real_roots.size == 0:
            raise RuntimeError("No real Z roots found")
        Z = np.max(real_roots)
        # print(f"Z = {Z}")

        sq2 = np.sqrt(2.0)
        lnphi = (Z - 1.0) - np.log(Z - B) - (A / (2.0*sq2*B)) * \
                np.log((Z + (1.0 + sq2)*B) / (Z + (1.0 - sq2)*B))

        phi = np.exp(lnphi)                                                  # Fugacity coeff. []
        fug = phi * P                                                        # [Pa]

        # Solubility [kg/m^3]
        Cs = fug * H

        return Cs
    
    def gas_HT(self, T, rho, us, p):

        # Evaluate the gas heat transfer coefficient.
        #--------------------------------------------#

        #Enter the physical constant for seawater 
        mu_w = 1721.2*np.exp(-0.0251*T)/1e6; # Pa s

        #Calculate the thermal conductivity of seawater
        k = 1.46e-7; # m^2/s

        #Calculate the Schmidt number 
        Sc = mu_w/(rho*k); 

        #Calculate the Peclet number
        Pe = us*(gas.de[0]/1000)/k; 

        #Calculate the Reynolds number
        Re = rho*us*(gas.de[0]/1000)/mu_w; 

        if Re < 100:
            Sh = 1 + (1 + 1/Pe)**(1/3)*Re**0.41*Sc**(1/3); 
        elif Re < 2000:
            Sh = 1 + 0.724*Re**0.48*Sc**(1/3); 
        else:
            Sh = 1 + 0.425*Re**0.55*Sc**(1/3); 

        #Use the Sherwood number to obtain the mass transfer coeff. and store the
        #value in beta
        #beta = gas.K_t.*Sh(1).*k./gas.de(1); %m/s
        beta = gas.K[0]*Sh*k/(gas.de[0]/1000); #m/s Is it K or K_t? (9/11/2025)

        return beta

    def gas_density(self, T, P, p):

        # Evaluate gas density using the Peng-Robinson
        # equation of state.
        #--------------------------------------------#

        T = T + c.T0

        # Peng-Robinson EOS parameters
        b = 0.07780 * c.Ru * c.Tcr_gas / c.Pcr_gas
        ac = 0.45724 * c.Ru**2 * c.Tcr_gas**2 / c.Pcr_gas
        mu = 0.37464 + 1.54226*c.omega - 0.26992*c.omega**2
        alpha = (1 + mu * (1 - np.sqrt(T/c.Tcr_gas)))**2
        a = ac * alpha

        # Dimensionless parameters
        A = a * P / (c.Ru**2 * T**2)
        B = b * P / (c.Ru * T)

        # Cubic coefficients for Z-factor: Z^3 + c2 Z^2 + c1 Z + c0 = 0
        c2 = -(1 - B)
        c1 = A - 3*B**2 - 2*B
        c0 = -(A*B - B**2 - B**3)

        coef = np.array([1.0, c2[0], c1[0], c0[0]])

        # Solve cubic for Z (compressibility factor)
        roots = np.roots(coef)
        real_roots = np.real(roots[np.isclose(roots.imag, 0, atol=1e-10)])
        if real_roots.size == 0:
            raise RuntimeError("No real Z roots found for PR EOS.")
        Z = np.max(real_roots)  # choose vapor root

        # Compute molar volume
        Vm = Z * c.Ru * T / P  # m^3/mol

        # Gas density
        rho = c.M_gas[0][0][0] / Vm  # kg/m^3

        return rho

    def void_fraction(self, mb, rho_b, bi, us, ui, p):

        # Evaluate the gas void fraction.
        #--------------------------------------------#

        return mb /(rho_b * (0.5 * c.pi * bi**2 * gas.lambda_r**2 * ( us + 2 * ui/(1 + gas.lambda_r**2))))

    def bubble_diameter(self, mb, rho_b):

        # Bubble diameter [m] from the conserved bubble number flux.
        #--------------------------------------------#
        if mb > 0 and self.nb0 > 0:
            return float(np.squeeze((6 * mb / (c.pi * rho_b * self.nb0))**(1/3)))
        return gas.de0 / 1000      # placeholder; nb = 0 below so bubble terms vanish

    def u_slip(self, rho_a, rho_b, T, de, p):

        # Evaluate the slip velocity of the gas bubble.
        #--------------------------------------------#

        # Viscosity of continuous phase
        mu = 1721.2 * np.exp(-0.0251 * T) / 1e6 # Pa s

        #Viscosity in Briada's expts
        mu_w = 0.0009

        # Calculate surface tension for water
        sigma = (75.7060 - 0.1511 * T) / 1000

        #Calculate non-dimensionals
        M = c.g * mu **4 * (rho_a - rho_b) / (rho_a**2 * sigma**3)
        Eo = c.g * (rho_a - rho_b) * de**2 / sigma

        # Get reynolds number at terminal velocity
        H = 4/3 * Eo * M**(-0.149) * (mu/mu_w)**(-0.14)

        if H>2:
            # Elliptical wobbling regimes
            if H>59.3:
                J = 3.42 * H**(0.441)
            elif H > 2:
                J = 0.94 * H**(0.757)
            Re = M**(-0.149) * (J-0.857)
        else:
            ## Spherical regime (Clift p.113)
            Nd = 4 * rho_a * (rho_a - rho_b) * c.g * de**3 / (3 * mu**2)
            if Nd <= 73:
                Re = Nd/24 - 1.7569e-4 * Nd**2 + 6.9252e-7 * Nd**3 - 2.3027e-10*Nd**4
            elif Nd <= 580:
                Re = 10**(-1.7095 + 1.33438 * np.log(Nd) - 0.11591 * np.log(Nd)**2)
            elif Nd <=1.55e7:
                Re = 10**(-1.81391 + 1.34671*np.log(Nd) - 0.12427*np.log(Nd)**2 + 0.006344*np.log(Nd)**3)
            else:
                Re = 10**(5.33283 - 1.21728*np.log(Nd) + 0.19007*np.log(Nd)**2 - 0.007005*np.log(Nd)**3)

        us = mu / (rho_a * de) * Re
        
        if us < 0:
            us = 0.001

        return us
    
    def lima_neto(self, Qg, Ql, R, p):

        # Evaluate the initial condition for the inner
        # plume at the injection point when there is
        # both a discrete and continuous phase being
        # emitted. From Lima and Neto 2012
        #--------------------------------------------#
    
        A = c.pi * R**2 / 2

        epsilon_o = Qg / (Qg + Ql)
        u = Ql / ((1 - epsilon_o) * A)
        
        return u
    
    def wuest(self, u0, us, mb, bi, rho_a, rho_b, lamb, p):

        # Evaluate the initial condition for the inner
        # plume when there is only a discrete phase.
        # From Wuest et al. 1992
        #--------------------------------------------#

        u = fsolve(self.wuest_residual, u0, args = (us, mb, bi, rho_a, rho_b, lamb, p))[0]

        return u
        
    def wuest_residual(self, u, us, mb, bi, rho_a, rho_b, lamb, p):
        
        # Void fractions
        xi_gas = mb / (
            0.5 * c.pi * bi**2 * rho_b * lamb**2 *
            (us + 2*u/(1+lamb**2))
        )

        # Mixed-fluid plume density
        rho_p = (
            (xi_gas * rho_b) +
            (1 - (xi_gas)) * rho_a
        )

        # Residual: target Froude number
        return 0.3 - u / np.sqrt(2 * lamb * bi * c.g * (rho_a - rho_p) / p.rho_r)

    def main_ic(self, mb, Qg, Qsg, bi, rho_a, rho_b, us, p):

        # Evaluate the initial condition for the inner
        # plume at the injection point using either
        # a west or lima neto condition.
        #--------------------------------------------#

        # Calculate initial velocity
        if Qsg <= 0:
            u = self.wuest(p.u0, us, mb, bi, rho_a, rho_b, gas.lambda_r, p)
        else:
            u = self.lima_neto(Qg, Qsg, bi, p)

        Q = c.pi * bi**2 * u / 2 # Added 2 for half cone
        J = Q * u
        z = 0 # initial condition is valid at the diffuser

        return Q, J, z 
    
    def getVars(self, yi, zi, yo, ambient, p):

        # Return variables for the double plume model.
        #--------------------------------------------#

        # Ambient conditions
        ambient_interp = interp1d(ambient[0, :], ambient[1:, :], axis=1, kind='linear', fill_value="extrapolate")
        Ta, Sa, Pa, ca = ambient_interp(zi)

        # Inner plume conditions
        ui = yi[1]/yi[0]                                                 # inner velocity, m/s
        bi = (2 * yi[0]/(ui * c.pi))**(1/2)                             # inner radius, m
        Ti = yi[2] / (yi[0] * p.rho_r * c.cp) - c.T0                     # inner temperature, C
        Si = yi[4] / yi[0]                                               # inner salinity, psu
        mb = yi[5]                                                       # inner gas mass flux, kg/s
        ci = yi[6] / yi[0]                                               # inner O2 concentration, kg/m^3

        # Melting conditions
        depth = ambient[0,-1] - zi
        m_dot = self.melting(yi, depth, p)                                  # melt rate, m/s
        Sb = self.boundary_salinity(yi, m_dot, p)                        # boundary salinity, psu
        Tb = self.boundary_temperature(Sb, depth, p)                        # boundary temp, C

        # Outer plume conditions
        if yo[0] == 0:
            uo = 0
            bo = 0
            To = Ta 
            So = Sa
            co = ca
        else:
            
            uo = yo[1]/yo[0]                                             # outer velocity, m/s
            bo = (2 * yo[0]/(uo * c.pi) + bi**2)**(1/2)                 # outer radius, m
            To = yo[2]/ (yo[0] * p.rho_r * c.cp) - c.T0                  # outer temperature, C
            So = yo[3] / yo[0]                                           # outer salinity, psu
            co = yo[4] / yo[0]                                           # outer O2 concentration, kg/m^3
        # Reset gas temperature

        if mb > 0:
            gas.T = yi[3] / (mb * c.cp_g) - c.T0     # gas temperature from bubble heat flux
        # Densities, kg/m^3
        p_sea = (Pa - 101325) / 1e4   # sea pressure [dbar] for gsw
        rho_a = gsw.density.rho_t_exact(Sa, Ta, p_sea)
        rho_i = gsw.density.rho_t_exact(Si, Ti, p_sea)
        rho_o = gsw.density.rho_t_exact(So, To, p_sea)
        rho_b = self.gas_density(gas.T, Pa, p)                              # bubble density

                # Bubble count and diameter
        d = self.bubble_diameter(mb, rho_b)                             # [m]
        gas.de = np.atleast_1d(1000 * d)                                # [mm]
        nb = self.nb0 if mb > 0 else 0.0
        ## Bubble slip velocity, m/s
        us = self.u_slip(rho_a, rho_b, Ti, d, p)

        # Peeling, m^2/s
        Ep = self.cp_model(ui, us, bi, rho_i, rho_a, p) 

        ## Bubble heat and mass transfer
        beta = self.gas_MT(Ti, rho_a, mb, us, p)                         # Mass transfer coeff
        beta_T = self.gas_HT(Ti, rho_a, us, p)                           # Heat transfer coeff
        #nb = 6 * (mb/rho_b) / (c.pi * (gas.de[0]/1000)**3)              # Bubble count 
                              

        ## Bubble Solubility
        Cs = self.gas_SOL(rho_a, Ti, Si, Pa, mb, p)                       

        ## Bubble Forces and void fraction
        Xi = self.void_fraction(mb, rho_b, bi, us, ui, p)                # Void Fraction
        Fpd = gas.lambda_r**2 * Xi * (rho_a - rho_b)                     # Bubble force

        out = (ui, bi, Ti, Si, mb, ci, uo, bo, To, So, co, Ta, Sa,
               ca, rho_i, rho_b, rho_o, rho_a, us, Ep, beta, beta_T,
               nb, Cs, Xi, Fpd, m_dot, Tb, Sb)
        return tuple(float(np.squeeze(v)) for v in out)                    

    def outer_state(self, z, zo, neighbor):

        # Outer plume state at elevation z; zeros where no outer plume exists.
        #--------------------------------------------#
        zo = np.atleast_1d(zo)
        if neighbor.shape[1] < 2 or z > np.max(zo) or z < np.min(zo):
            return np.zeros(5)
        zs, idx = np.unique(zo, return_index=True)      # sorted, duplicates removed
        return interp1d(zs, neighbor[:, idx], axis=1, kind='linear')(z)
    
    def inner_ic(self, z, y, ambient, zo, neighbor, p):

        # Evaluate the initial condition for a subsequent
        # plume following a peeling event.
        #--------------------------------------------#

        # Extract state for inner plume
        yi = y.copy()

        # Extract for outer plume
        yo = self.outer_state(z, zo, neighbor)
        
        # Get variables
        ui, bi, Ti, Si, mb, ci, uo, bo, To, So, co, Ta, Sa, \
        ca, rho_i, rho_b, rho_o, rho_a, us, Ep, beta, beta_T, \
        nb, Cs, Xi, Fpd, m_dot, Tb, Sb = self.getVars(yi, z, yo, ambient, p)

        # Get the flux of fluid continuing beyond this peel
        Qi = (1 - p.fp) * yi[0]

        # Use Wuest et al. (1992) type i.c. for velocity
        ui = self.wuest(1e-3, us,  mb, bi, rho_a, rho_b, gas.lambda_r, p)
        
        # Compute related plume width
        bi = (2 * Qi / (c.pi * ui))**(1/2)

        # Assemble state space
        z0 = z + 0.1 * bi
        y0 = np.zeros(7)
        y0[0] = Qi                                                       # Volume flux
        y0[1] = Qi * ui                                                  # Momentum Flux
        y0[2] = p.rho_r * c.cp * (Ti + c.T0) * Qi                        # Heat flux (water)
        y0[3] = mb * c.cp_g * (gas.T + c.T0)                             # Heat flux (gas)
        y0[4] = Si * Qi                                                  # Salt Flux m^3 psu /s
        y0[5] = mb                                                       # Bubble mass Flux kg/s
        y0[6] = ci * Qi                                                  # Gas concentration

        return z0, y0

    def inner_derivs(self, z, y, ambient, zo, neighbor, mb0, p):

        # System of equations for the inner plume.
        #--------------------------------------------#

        # Get inner state space
        yi = y.copy()
        
        # Get outer state space
        yo = self.outer_state(z, zo, neighbor)
        
        # Get variables
        ui, bi, Ti, Si, mb, ci, uo, bo, To, So, co, Ta, Sa, \
        ca, rho_i, rho_b, rho_o, rho_a, us, Ep, beta, beta_T, \
        nb, Cs, Xi, Fpd, m_dot, Tb, Sb = self.getVars(yi, z, yo, ambient, p)
        # print(f"mb: {mb}, ui: {ui}")
#        print(f"Ep: {Ep}, Cg: {Cs}, Xi: {Xi}, ui: {ui}, Fpd: {Fpd}")
        # print(f"beta_M: {beta}, beta_T: {beta_T}")

        # To make it ice relevant:
        # 1. Added melt conservation terms to Q, J, H, B
        # 2. Made the geomtry half cone

        yp = np.zeros(7)
        # --- Mass conservation ---
        yp[0] =  c.pi * bi * (p.alpha_1 * (ui + p.c1*uo) + p.alpha_2*uo) + Ep + 2 * bi * m_dot

        # --- Momentum conservation ---
        yp[1] = (1/p.gamma_i) * (0.5 * c.pi * c.g * bi**2 / p.rho_r * (Fpd + p.lambda_s**2 * (1 - Xi) * (rho_a - rho_i)) +
                                c.pi * bi * (p.alpha_1 * (ui + p.c1 * uo) * uo + p.alpha_2 * uo * ui) +
                                Ep*ui - p.Cd * 2 * bi * ui**2)

        # --- Heat conservation (water) ---
        yp[2] = p.rho_r * c.cp * (c.pi * bi * (p.alpha_1 * (ui + p.c1 * uo) * (To + c.T0) + \
                                            p.alpha_2 * uo * (Ti + c.T0)) + Ep*(Ti + c.T0) - \
                                            2 * p.gammaT * p.Cd**(1/2) * bi * ui * (Ti - Tb)) + \
                                            p.rho_r*c.cp*2*bi*m_dot*(Tb + c.T0)  
        yp[5] = - (c.pi * nb * (gas.de[0]/1000)**2/(ui + us) * beta * (Cs - ci))
        delDiss = yp[5]
        ## -- Heat Conservation (gas) --
        # yp[3] = -c.pi * nb * (gas.de[0]/1000)**2/(ui + us) * rho_b * c.cp_g * beta_T * (gas.T - Ti)
        # yp[2] = yp[2] - yp[3]   
        yp[3] = (-c.pi * nb * (gas.de[0]/1000)**2/(ui + us) * rho_b * c.cp_g * beta_T * (gas.T - Ti)
                        + c.cp_g * (gas.T + c.T0) * yp[5])      # enthalpy carried off by dissolved gas
        yp[2] = yp[2] - yp[3]                             # water receives both
        # --- Salinity ---
        yp[4] = c.pi * bi * (p.alpha_1*(ui + p.c1*uo)*So + p.alpha_2 * uo * Si) + Ep*Si \
                + 2 * bi * m_dot * Sb - 2 * p.gammaS * p.Cd**(1/2) * bi * ui * (Si - Sb)
        ## Gas mass Conservation
        

        ## Gas concentration
        yp[6] = c.pi * bi * (p.alpha_1 * (ui + p.c1*uo) * co + p.alpha_2 * uo *ci) + Ep*ci - delDiss  
        
        return yp
    
    def inner_stop(self, z, y, ambient, zo, neighbor, mb0, p):

        # Stopping criteria for the inner plume.
        #--------------------------------------------#

        
        # Default behavior
        value = 1.0

        # Get inner plume state space
        yi = y.copy()

        # Get outer plume state space
        yo = self.outer_state(z, zo, neighbor)

        # Get variables
        ui, bi, Ti, Si, mb, ci, uo, bo, To, So, co, Ta, Sa, \
        ca, rho_i, rho_b, rho_o, rho_a, us, Ep, beta, beta_T, \
        nb, Cs, Xi, Fpd, m_dot, Tb, Sb = self.getVars(yi, z, yo, ambient, p)

        # Event 1: Did the bubbles dissolve? 
        dissolved = True
        if mb/mb0 < gas.fdis:
            print(f"The bubbles dissolved.") 
            dissolved = True
            value = 0.0

        # # if dissolved:
        #     Ep = cp_model(ui, us, bi, rho_i, rho_a) 
        #     direction = 1
        #     value = 2 * c.pi * bi * (p.alpha_1 * (ui + p.c1 * uo) + p.alpha_2 * uo) + Ep

        # Event 2: Are the bubbles still buoyant?
        if rho_b > rho_i:
            print("The bubbles are no longer buoyant")
            value = 0

        # Event 3: Any state-space variables are imaginary?
        for i in range(7): # 7 is length of yi
            if np.abs(np.imag(y[i])) > 0:
                print(f"Imaginary state space.")
                value = 0.0

        # Event 4: Momentum <= 0 
        # if ui <= 0.05: 
        #     print(f"No momentum. ui = {ui}, mom = {yi[1]}")
        #     value = 0.0

        # Event 5: Plume reaches surface
        if z >= ambient[0, -1]:
            value = 0.0

        if np.any(np.isnan(y)):
            value = 0.0

        return value

    inner_stop.terminal = True
    inner_stop.direction = 0.0

    def inner_peel(self, z, y, ambient, zo, neighbor, mb0, p):

        # Discrete peel: the plume liquid is heavier than ambient AND has
        # slowed enough (u_i < u_s/kappa) that the bubbles separate from it.
        #--------------------------------------------#

        yo = self.outer_state(z, zo, neighbor)
        ui, bi, Ti, Si, mb, ci, uo, bo, To, So, co, Ta, Sa, \
        ca, rho_i, rho_b, rho_o, rho_a, us, Ep, beta, beta_T, \
        nb, Cs, Xi, Fpd, m_dot, Tb, Sb = self.getVars(y, z, yo, ambient, p)

        if rho_i <= rho_a:
            return 1.0                       # buoyant plume liquid cannot peel
        return float(np.squeeze(ui - us / p.kappa_peel))

    inner_peel.terminal = True
    inner_peel.direction = -1

    def inner_stall(self, z, y, ambient, zo, neighbor, mb0, p):

        # Safety stop if the plume stalls without meeting the peel criterion.
        #--------------------------------------------#

        return y[1] / y[0] - 1e-3

    inner_stall.terminal = True
    inner_stall.direction = -1

    def inner_calc(self, ambient, zo, neighbor, y0, p):

        # Call the models to solve the inner plume system.
        #--------------------------------------------#

        z0 = ambient[0,0]
        z_elev = z0;                                                     # Elevation of current calculation 
        jp = 0;                                                          # Total number of internal peels 
        dissolved = False;                                               # Records status of the dispersed phase
        z_peel = np.array([]);                                           # Matrix index to intermediate peel locations 
        iter = 0
        mb0 = y0[5]

        # Integrate upwards in space for the inner plume,
        # until the reservoir surface is reached OR the
        # dispersed phase dissolves OR the plume runs out of momentum 
        
        while z_elev < ambient[0,-1] and not dissolved:

                sol = solve_ivp(
                        fun= self.inner_derivs,
                        t_span=(z0, ambient[0, -1]),
                        y0=y0,
                        method='RK45',
                        vectorized=False,
                        max_step = 0.1,
                        rtol = 1e-8,
                        atol = 1e-10,
                        events=[self.inner_stop, self.inner_peel, self.inner_stall],
                        args=(ambient, zo, neighbor, mb0, p) 
                        )

                z = sol.t
                y = sol.y
                for name, te, ye in zip(['stop', 'peel', 'stall'], sol.t_events, sol.y_events):
                    if te.size:
                            zev, yev = te[0], ye[0]
                            g = self.getVars(yev, zev, self.outer_state(zev, zo, neighbor), ambient, p)
                            ui_e, rho_i_e, rho_a_e, us_e = g[0], g[14], g[17], g[18]
                            print(f"  event '{name}' at z = {zev:.2f} m: u_i = {float(np.squeeze(ui_e)):.3f}, "
                                f"u_s/kappa = {float(np.squeeze(us_e)) / p.kappa_peel:.3f}, "
                                f"rho_i - rho_a = {float(np.squeeze(rho_i_e - rho_a_e)):.3f}")

                if jp == 0:
                        ## First solution to record
                        zi = z
                        Qi = y[0]
                        Ji = y[1]
                        Hi = y[2]
                        Hb = y[3]
                        Bi = y[4]
                        O2_mb = y[5]
                        Ci = y[6]
                else:
                        ## Append solution to array
                        zi = np.append(zi, z)
                        Qi = np.append(Qi, y[0])
                        Ji = np.append(Ji, y[1])
                        Hi = np.append(Hi, y[2])
                        Hb = np.append(Hb, y[3])
                        Bi = np.append(Bi, y[4])
                        O2_mb = np.append(O2_mb, y[5])
                        Ci = np.append(Ci, y[6])

                if sol.status == -1:
                        print(f"Inner integration failed: {sol.message}")
                        break
                iter += 1

                # Update elevation
                z_elev = np.max(zi)

                # Check to see if bubbles have dissolved
                if np.min(O2_mb) / mb0 < gas.fdis:
                        dissolved = True

                # Process intermediate peels
                if z_elev < ambient[0,-1] and not dissolved:
                        print(f"Discrete peel {jp+1}")
                        print(f" Height: {z_elev} m")
                        peel_z = np.concatenate([np.atleast_1d(t) for t in sol.t_events])
                        z_peel = np.append(z_peel, peel_z)
                        jp += 1
                        
                        # Initialize a subsequent plume
                        z0, y0 = self.inner_ic(z_elev, y[:,-1], ambient, zo, neighbor, p)

                        # print(f"Initial conditions for next peel:")
                        # print(f"Qi: {y0[0]}, ui: {y0[1]/y0[0]}")

        yi = np.array([Qi, Ji, Hi, Hb, Bi, O2_mb, Ci])

        # Get indices for peel locations
        i_peel = np.zeros(len(z_peel))
        for i in range(len(z_peel)):
            i_peel[i] = np.argmin(abs(z_peel[i] - zi))
            if i > 0:
                if i_peel[i] - 1 == i_peel[i-1]:
                    i_peel[i] = np.nan
        i_peel = i_peel[~np.isnan(i_peel)].astype(int)
        jp = len(i_peel)
        
        return yi, zi, jp, i_peel
    
    # def outer_cpic(self, z_elev, ambient, zi, yi, neighbor, jp, i_peel, p):

        # # Evaluate the initial condition for an outer plume
        # # segment to form from a peeling event from
        # # the inner plume.
        # #--------------------------------------------#
        
        # # Assume it will be possible to integrate to bottom of the reservoir
        # zf = ambient[0,0]

        # # Set up looping paramaters
        # done = False
        # iter = 1

        # # Get inner plume and ambient conditions
        # neighbor_interp = interp1d(zi, neighbor, axis=1, kind='linear', fill_value="extrapolate")
        # ambient_interp = interp1d(ambient[0, :], ambient[1:, :], axis=1, kind='linear', fill_value="extrapolate")

        # # Compute the outer plume initial condition until the outer plume 
        # # is viable or until the maximum # of widths is integrated
        # while not done and iter <= p.nwidths:
            
        #     # Get inner plume condtions
        #     yi_interp = neighbor_interp(z_elev)                           # Because z_elev may not lie on yi's grid
        #     ui = yi_interp[1]/yi_interp[0]
        #     bi = (2 * yi_interp[0]/(ui * c.pi))**(1/2)

        #     # Determine range to get peeling flux
        #     z_upper = z_elev
        #     z_lower = z_elev - iter * bi                                 # Crounse model to expand the search

        #     # Check for hitting bottom of reservoir
        #     if z_lower < ambient[0,0]:
        #         z_lower = ambient[0,0]

        #     i_upper = np.argmin(abs(z_upper - zi))                       # first index with z >= z_lower
        #     i_lower = np.argmin(abs(z_lower - zi))                       # last index with z < z_upper

        #     # Get upper data
        #     Qi_upper, Ji_upper, Hi_upper, Hb_upper, Bi_upper, mb_upper, c_upper = yi[:,i_upper]
        #     Ta_upper, Sa_upper, Pa_upper, ca_upper = ambient_interp(z_upper)

        #     ui_upper = Ji_upper/Qi_upper                                 # inner velocity
        #     bi_upper = np.sqrt(2) * Qi_upper/np.sqrt(Ji_upper * c.pi)            # inner radius
        #     Ti_upper = Hi_upper / (Qi_upper * p.rho_r * c.cp) - c.T0     # inner temperature
        #     Si_upper = Bi_upper / Qi_upper                               # inner salinity
        #     ci_upper = c_upper / Qi_upper 
        #     p_sea = (Pa_upper - 101325) / 1e4                                  # inner O2 concentration
        #     rho_a_upper = gsw.density.rho_t_exact(Sa_upper,              # ambient density
        #                                 Ta_upper, p_sea)
        #     rho_i_upper = gsw.density.rho_t_exact(Si_upper,              # inner density
        #                                 Ti_upper, p_sea)
        #     rho_b = self.gas_density(Ti_upper, Pa_upper, p)              # bubble density
            
        #     # Get the slip velocity
        #     us_upper = self.u_slip(rho_a_upper, rho_b, Ti_upper,
        #                            gas.de/1000, p)
            
        #     # Get the Peeling flux
        #     Ep_upper = self.cp_model(ui_upper, us_upper, bi_upper, 
        #                              rho_i_upper, rho_a_upper, p)

        #     # Get the local dz for the inner plume
        #     dz_upper = (z_upper - zi[i_upper]) / 2

        #     # Get lower data
        #     Qi_lower, Ji_lower, Hi_lower, Hb_lower, Bi_lower, \
        #                 mb_lower, ci_lower = yi[:,i_lower]
        #     Ta_lower, Sa_lower, Pa_lower, \
        #                 ca_lower = ambient_interp(z_lower)

        #     ui_lower = Ji_lower/Qi_lower                                 # inner velocity
        #     bi_lower = np.sqrt(2) * Qi_lower/np.sqrt(Ji_lower * c.pi)            # inner radius
        #     Ti_lower = Hi_lower / (Qi_lower * p.rho_r * c.cp) - c.T0     # inner temperature
        #     Si_lower = Bi_lower/ Qi_lower                                # inner salinity
        #     ci_lower = ci_lower / Qi_lower  
        #     p_sea = (Pa_lower - 101325) / 1e4                                # inner O2 concentration
        #     rho_a_lower = gsw.density.rho_t_exact(Sa_lower,              # ambient density
        #                                 Ta_lower, p_sea)
        #     rho_i_lower = gsw.density.rho_t_exact(Si_lower,              # inner density
        #                                 Ti_lower, p_sea)
        #     rho_b = self.gas_density(Ti_lower, Pa_lower, p)              # bubble density
            
        #     # Get the slip velocity
        #     us_lower = self.u_slip(rho_a_lower, rho_b, Ti_lower, gas.de/1000, p)
            
        #     # Get the Peeling flux
        #     Ep_lower = self.cp_model(ui_lower, us_lower, bi_lower, rho_i_lower, rho_a_lower, p)

        #     # Get the local dz for the inner plume
        #     dz_lower = (zi[i_lower] - z_lower) / 2

        #     # Get all values inbetween
        #     Ep_list = [Ep_lower]
        #     ui_list = [ui_lower]
        #     bi_list = [bi_lower]
        #     Ti_list = [Ti_lower]
        #     Si_list = [Si_lower]
        #     ci_list = [ci_lower]
        #     dz_list = [dz_lower]

        #     for i in range(i_lower, i_upper+1):
        #         Qi, Ji, Hi, Hb, Bi, mb, Ci = yi[:,i]
        #         Ta, Sa, Pa, ca = ambient_interp(zi[i])

        #         ui = Ji/Qi                                               # inner velocity
        #         bi = (2 * Qi/(ui * c.pi))**(1/2)                        # inner radius
        #         Ti = Hi / (Qi * p.rho_r * c.cp) - c.T0                   # inner temperature
        #         Si = Bi / Qi                                             # inner salinity
        #         ci = Ci / Qi                                             # inner gas concentration
        #         p_sea = (Pa - 101325) / 1e4                                   # sea pressure [dbar] for gsw
        #         rho_a = gsw.density.rho_t_exact(Sa, Ta, p_sea)
        #         rho_i = gsw.density.rho_t_exact(Si, Ti, p_sea)
        #         rho_b = self.gas_density(Ti, Pa, p)                      # bubble density
                
        #         # Get the slip velocity
        #         us = self.u_slip(rho_a, rho_b, Ti, gas.de/1000, p)
                
        #         # Get the Peeling flux
        #         Ep = self.cp_model(ui, us, bi, rho_i, rho_a, p)

        #         # Get the local dz for the inner plume
        #         if i+1 >= len(zi):
        #             dz = zi[i-1]
        #         else:
        #             dz = (zi[i+1] - zi[i-1]) / 2

        #         Ep_list.append(Ep)
        #         ui_list.append(ui)
        #         bi_list.append(bi)
        #         Ti_list.append(Ti)
        #         Si_list.append(Si)
        #         ci_list.append(ci)
        #         dz_list.append(dz)

        #     Ep_list.append(Ep_upper)
        #     ui_list.append(ui_upper)
        #     bi_list.append(bi_upper)
        #     Ti_list.append(Ti_upper)
        #     Si_list.append(Si_upper)
        #     ci_list.append(ci_upper)
        #     dz_list.append(dz_upper)

        #     Ep_arr = np.array(Ep_list)
        #     ui_arr = np.array(ui_list)
        #     bi_arr = np.array(bi_list)
        #     Ti_arr = np.array(Ti_list)
        #     Si_arr = np.array(Si_list)
        #     ci_arr = np.array(ci_list)
        #     dz_arr = np.array(dz_list)

        #     # Integrate
        #     Qo = -np.nansum(Ep_arr * dz_arr)                           
        #     uo = -np.nansum(ui_arr * (Ep_arr * dz_arr))                 
        #     bo = -np.nansum(bi_arr * (Ep_arr * dz_arr)) 
        #     To = -np.nansum(Ti_arr * (Ep_arr * dz_arr)) 
        #     So = -np.nansum(Si_arr * (Ep_arr * dz_arr)) 
        #     co = -np.nansum(ci_arr * (Ep_arr * dz_arr)) 

        #     if jp > 0:
        #         while z_lower < zi[i_peel[jp-1]]:
        #             jp = jp - 1
        #             if jp == 0:
        #                 break
        #         if jp == 0:
        #             zf = ambient[0,0]
        #         else:
        #             zf = zi[i_peel[jp-1]]

        #     if abs(Qo) == 0:
        #         To = Ta
        #         So = Sa
        #         co = ca
        #     else:
        #         To = To/Qo 
        #         So = So/Qo
        #         co = co/Qo

        #     p_sea = (Pa_lower - 101325) / 1e4   
        #     rho_o = gsw.density.rho_t_exact(So, To, p_sea)
        #     rho_o = rho_o + (1-c.vCO2) * co

        #     # Use Fr to find outer plume i.c. velocity
        #     uo = self.calculate_outer_fr(0.0001, Qo, bi_upper, rho_a_upper, rho_o, p)
        #     bo = np.sqrt(2 * Qo / (c.pi * uo) + bi**2)

        #     # Compute state-space variables
        #     if Qo > 0:
        #         Qo = -Qo
        #     Jo = -Qo * uo
        #     Ho = p.rho_r * c.cp * (To + c.T0) * Qo
        #     Bo = So * Qo
        #     Co = co * Qo

        #     # Store result in initial condition vector
        #     z0 = z_lower
        #     y0 = np.array([Qo, Jo[0], Ho, Bo, Co])

        #     # Prepare viability check
        #     Ep = Ep_lower

        #     uo = -uo
        #     ui = ui_lower

        #     # Compute outer plume change in volume flux at 
        #     # start of integration
        #     dQdz = c.pi * bi * (p.alpha_1 * (ui + p.c1 * uo) + p.alpha_2 * uo) + c.pi * bo * p.alpha_3 * uo + Ep

        #     if dQdz > 0:
        #         # This outerplume is not viable
        #         flag = 0

        #         #Record heigh and give zero solution
        #         z0 = z_lower
        #         y0 = np.zeros(5)

        #     else:
        #         # This outerplume is viable
        #         flag = 1
        #         # Stop the iterations
        #         done = True
                
        #         if z0 <= ambient[0,0]:
        #             flag = 0
        #             z0 = z_elev
        #             y0 = np.zeros(5)
            
        #     iter += 1

        # return z0, zf, y0, flag
    
    def outer_cpic(self, z_elev, ambient, zi, yi, neighbor, jp, i_peel, p):

        # Evaluate the initial condition for an outer plume
        # segment formed by continuous peeling from the
        # inner plume.
        #--------------------------------------------#

        z_floor = zi[i_peel[jp-1]] if jp > 0 else ambient[0,0]
        zf = z_floor
        done = False
        iter = 1
        flag = 0
        z0, y0 = z_elev, np.zeros(5)

        inner_interp = interp1d(zi, yi, axis=1, kind='linear', fill_value="extrapolate")
        ambient_interp = interp1d(ambient[0, :], ambient[1:, :], axis=1, kind='linear', fill_value="extrapolate")

        while not done and iter <= p.nwidths:

            # Inner plume width at the current elevation
            y_top = inner_interp(z_elev)
            bi_top = np.sqrt(2 * y_top[0]**2 / (c.pi * y_top[1]))

            # Search window, expanding by one plume width per iteration
            z_upper = z_elev
            z_lower = max(z_elev - iter * bi_top, z_floor)

            # Inner-plume grid points in the window plus both end points
            inside = zi[(zi > z_lower) & (zi < z_upper)]
            z_win = np.concatenate(([z_lower], inside, [z_upper]))
            y_win = inner_interp(z_win)
            n = z_win.size

            Ep_w = np.zeros(n); Ti_w = np.zeros(n); Si_w = np.zeros(n); ci_w = np.zeros(n)
            ui_w = np.zeros(n); bi_w = np.zeros(n); rho_a_w = np.zeros(n)
            Ta_w = np.zeros(n); Sa_w = np.zeros(n); ca_w = np.zeros(n); Pa_w = np.zeros(n)

            for k in range(n):
                Qi, Ji, Hi, Hb, Bi, mb, Ci = y_win[:, k]
                Ta, Sa, Pa, ca = ambient_interp(z_win[k])

                ui = Ji / Qi
                bi = np.sqrt(2 * Qi / (c.pi * ui))
                Ti = Hi / (Qi * p.rho_r * c.cp) - c.T0
                Si = Bi / Qi
                ci = Ci / Qi

                p_sea = (Pa - 101325) / 1e4
                rho_a = gsw.density.rho_t_exact(Sa, Ta, p_sea)
                rho_i = gsw.density.rho_t_exact(Si, Ti, p_sea)
                Tg = Hb / (mb * c.cp_g) - c.T0 if mb > 0 else Ti
                rho_b = self.gas_density(Tg, Pa, p)
                d = self.bubble_diameter(mb, rho_b)
                us = self.u_slip(rho_a, rho_b, Ti, d, p)

                Ep_w[k] = self.cp_model(ui, us, bi, rho_i, rho_a, p)
                Ti_w[k], Si_w[k], ci_w[k] = Ti, Si, ci
                ui_w[k], bi_w[k], rho_a_w[k] = ui, bi, rho_a
                Ta_w[k], Sa_w[k], ca_w[k], Pa_w[k] = Ta, Sa, ca, Pa

            # Integrate the peeling flux over the window (Ep < 0 is peeling)
            Qo = -np.trapezoid(Ep_w, z_win)


            if Qo <= 0:
                # No net peeling in this window
                z0, y0, flag = z_lower, np.zeros(5), 0
                if z_lower <= z_floor:
                    break                   # reached the next peel: outer_dpic takes over
                iter += 1
                continue

            To = -np.trapezoid(Ep_w * Ti_w, z_win) / Qo
            So = -np.trapezoid(Ep_w * Si_w, z_win) / Qo
            co = -np.trapezoid(Ep_w * ci_w, z_win) / Qo

            # Outer plume starts at the bottom of the window
            p_sea = (Pa_w[0] - 101325) / 1e4
            rho_o = gsw.density.rho_t_exact(So, To, p_sea)
            uo = float(self.calculate_outer_fr(1e-4, Qo, bi_w[0], rho_a_w[0], rho_o, p)[0])
            bo = np.sqrt(2 * Qo / (c.pi * uo) + bi_w[0]**2)

            # State-space variables (outer plume flows downward, Q < 0)
            Qo = -Qo
            Jo = -Qo * uo
            Ho = p.rho_r * c.cp * (To + c.T0) * Qo
            Bo = So * Qo
            Co = co * Qo
            z0 = z_lower
            y0 = np.array([Qo, Jo, Ho, Bo, Co])

            # Viability: the outer plume must gain volume flux as it descends
            uo_s = -uo
            dQdz = c.pi * bi_w[0] * (p.alpha_1 * (ui_w[0] + p.c1 * uo_s) + p.alpha_2 * uo_s) \
                   + c.pi * bo * p.alpha_3 * uo_s + Ep_w[0]

            if dQdz > 0:
                flag = 0
                y0 = np.zeros(5)
            else:
                flag = 1
                done = True
                if z0 <= ambient[0,0]:
                    # Reached the bottom: close out without re-using z_elev
                    flag = 0
                    z0 = ambient[0,0]
                    y0 = np.zeros(5)

            iter += 1

        return z0, zf, y0, flag

    def outer_dpic(self, z_elev, ambient, zi, neighbor, jp, peel_index, p):

        # Discrete peeling model. Not used.
        #--------------------------------------------#

        if jp == 0:
            # All outer plumes are complete
            # if jdp > 0:

            #     i_pt[jdp,1] = len(zo)
            
            z_diff = ambient[0,0] # diffuser depth
            if z_elev > z_diff:
                z0 = np.array([z_elev, z_diff]) # Closing out outer plumes
                y0 = np.zeros((5,2))
            else:
                # The last outer plume stopped below the diffuser
                z0 = z_elev
                y0 = np.zeros(5)

            flag = 0 # Do not integrate any more plumes

        else:

            # if np.min(zo) <= zi[peel_index[jp]]: #If minimum of outer z is less than the peel heigh of the next peel, two plumes overlap
            neighbor_interp = interp1d(zi, neighbor, axis=1, kind='linear', fill_value="extrapolate")

            ambient_interp = interp1d(ambient[0, :], ambient[1:, :], axis=1, kind='linear', fill_value="extrapolate")

            yi = neighbor_interp(zi[peel_index[jp-1]])
            
            Ta, Sa, Pa, ca = ambient_interp(zi[peel_index[jp-1]])

            ui = yi[1]/yi[0]                                                     # inner velocity
            bi = (2 * yi[0]/(ui * c.pi))**(1/2)                                      # inner radius
            Ti = yi[2] / (yi[0] * p.rho_r * c.cp) - c.T0                         # inner temperature
            Si = yi[4] / yi[0]                                                   # inner salinity
            mb = yi[5]
            ci = yi[6] / yi[0]                                                   # inner O2 concentration
            # rho_a = gsw.density.rho_t_exact(Sa, Ta, Pa/10**4)
            # rho_i = gsw.density.rho_t_exact(Si, Ti, Pa/10**4)

            # Calculate Vol. Flux Initial
            Qo = p.fp * yi[0]

            #Calculate properties of peeling fluid
            To = ((Ti + c.T0) * p.fp * yi[0])/Qo - c.T0
            So = (Si * p.fp * yi[0])/Qo
            Co = (ci * p.fp * yi[0])/Qo

            p_sea = (Pa - 101325) / 1e4                                   # sea pressure [dbar] for gsw
            rho_a = gsw.density.rho_t_exact(Sa, Ta, p_sea)
            rho_i = gsw.density.rho_t_exact(Si, Ti, p_sea)
            rho_o = gsw.density.rho_t_exact(So, To, p_sea)

            uo = self.calculate_outer_fr(ui, Qo, bi, rho_a, rho_o, p)

            Qo = -Qo
            Jo = -Qo * uo
            Ho = p.rho_r * c.cp * (Ti + c.T0) * Qo
            Bo = Si * Qo
            Co = ci*Qo

            z0 = zi[peel_index[jp-1]]
            y0 = np.array([Qo, Jo[0], Ho, Bo, Co])

            flag = 1

        if jp <=1:
            zf = ambient[0,0]
        else:
            zf = zi[peel_index[jp-2]]
        
        return z0, zf, y0, flag
    
    def outer_fr(self, uo, Qo, bi, rho_a, rho_o, p):

        # Evaluate the initial condition for an outer
        # plume segment using a Wuest 1992 Froude
        # condition.
        #--------------------------------------------#

        Fr = 0.1
        bo = (2 * Qo/(c.pi * uo) + bi**2)**(1/2)

        return uo - Fr*np.sqrt(abs((bo - bi)*c.g*(rho_a - rho_o)/rho_o)) 

    def calculate_outer_fr(self, u0, Qo, bi, rho_a, rho_o, p):

        uo = fsolve(self.outer_fr, u0, args = (Qo, bi, rho_a, rho_o, p))

        return uo
    
    def outer_surf(self, z, ambient, zi, neighbor, jp, peel_index, p):

        # Evaluate the initial condition for the outer
        # plume that forms at the surface of the water
        # column, where the inner plume terminates.
        #--------------------------------------------#

        # Get local plume and ambient properties
        neighbor_interp = interp1d(zi, neighbor, axis = 1, kind = 'linear', fill_value = 'extrapolate')
        Qi, Ji, Hi, Hb, Bi, O2_mb, O2_c = neighbor_interp(z)

        ambient_interp = interp1d(ambient[0, :], ambient[1:, :], axis=1, kind='linear', fill_value="extrapolate")

        Ta, Sa, Pa, ca = ambient_interp(z)

        ui = Ji/Qi                                                     # inner velocity
        bi = (2 * Qi/(ui * c.pi))**(1/2)                                       # inner radius
        Ti = Hi/ (Qi * p.rho_r * c.cp) - c.T0                                                 # inner temperature
        Si = Bi / Qi                                                   # inner salinity
        mb = O2_mb
        ci = O2_c / Qi
            
        if z < ambient[0,-1]:
            # Plume stopped below the surface: outer plume is inner plume fluid
            Qo = Qi
            if p.epsilon > 0:
                Qo = 0.1 * Qo
            To, So, co = Ti, Si, ci
        else:
            # Outer plume is a mixture of inner plume fluid and ambient
            Qo = (1 + p.fe) * Qi
            To = (Ti + p.fe * Ta) / (1 + p.fe)
            So = (Si + p.fe * Sa) / (1 + p.fe)
            co = (ci + p.fe * ca) / (1 + p.fe)

        #Use a Fr to set width and velocity
        p_sea = (Pa - 101325) / 1e4                                   # sea pressure [dbar] for gsw
        rho_a = gsw.density.rho_t_exact(Sa, Ta, p_sea)
        rho_i = gsw.density.rho_t_exact(Si, Ti, p_sea)
        rho_o = gsw.density.rho_t_exact(So, To, p_sea)
        rho_b = self.gas_density(Ti, Pa, p)

        uo = self.calculate_outer_fr(ui, Qo, bi, rho_a, rho_o, p)

        # Calculate outer plume state space variables
        Qo = -Qo
        Jo = -Qo * uo[0]
        Ho = p.rho_r * c.cp * (To + c.T0) * Qo
        Bo = So * Qo
        Co = co * Qo

        z0 = z
        y0 = np.array([Qo, Jo, Ho, Bo, Co]) #len(5)

        # Set limits of integration
        if jp > 0:
            # There are discrete peels in the water column, integrate to that
            zf = zi[peel_index[-1]]                         
        else:
            # Integrating from surface at z = 100m to inlet at z = 0 m.
            zf = ambient[0, 0] 

        # print("Initial Conditions for outer plume")
        # print(f"Qo, uo, To, So, Co: {Qo}, {uo}, {To}, {So}, {co}")
        # print(f"z0: {z0}")

        return z0, y0, zf

    def outer_derivs(self, z, y, ambient, zi, neighbor, p):

        # Outer plume system of equations.
        #--------------------------------------------#

        yo = y.copy()
        
        neighbor_interp = interp1d(zi, neighbor, axis=1, kind='linear', fill_value="extrapolate")
        yi = neighbor_interp(z)

        ui, bi, Ti, Si, mb, ci, uo, bo, To, So, co, Ta, Sa, \
        ca, rho_i, rho_b, rho_o, rho_a, us, Ep, beta, beta_T, \
        nb, Cs, Xi, Fpd, m_dot, Tb, Sb = self.getVars(yi, z, yo, ambient, p)
        # print(f"Ep: {Ep}, ui: {ui}, bi: {bi}")
        if ui == 0:
            alpha_2 = 0
            alpha_1 = 0
            Ep = 0
        else:
            alpha_2 = p.alpha_2
            alpha_1 = p.alpha_1

        yp = np.zeros(5)

        #Conservation of mass
        yp[0] =  c.pi * bi * (alpha_1 * (ui + p.c1*uo) + alpha_2 * uo) + c.pi * bo * p.alpha_3 * uo + Ep; 

        #Conservation of momentum 
        yp[1] = 1/p.gamma_o * (-0.5 * c.pi * c.g * (bo**2 - bi**2) / p.rho_r * (rho_a - rho_o) + c.pi * bi * (alpha_1 * (ui + p.c1*uo)*uo + alpha_2 * uo * ui) + Ep*ui); 

        #Conservation of Heat (water)
        yp[2] = p.rho_r * c.cp * (c.pi * bi * (alpha_1 * (ui + p.c1*uo) * (To + c.T0) + alpha_2 * uo * (Ti + c.T0)) + c.pi * bo * p.alpha_3 * uo * (Ta + c.T0) + Ep*(Ti + c.T0)); 

        #Conservation of salinity 
        yp[3] = c.pi * bi * (alpha_1 * (ui + p.c1*uo) * So + alpha_2 * uo * Si) + c.pi * bo * p.alpha_3 * uo * Sa + Ep*Si; 

        #Conservation of dissolved constituents 
        yp[4] = c.pi * bi * (alpha_1 * (ui + p.c1*uo) * co + alpha_2 * uo * ci) + c.pi * bo * p.alpha_3 * uo * ca + Ep * ci

        # Outer-plume exchanges are the negatives of the inner-plume terms
        # (u_o < 0, Q_o < 0, E_p < 0), so dY_o/dz = -(inner-style RHS).
        # The solver handles the downward integration direction itself.
        yp = -yp; 
        
        return yp
    
    def outer_calc(self, ambient, zi, yi, neighbor, jp, i_peel, p):

        # Routine for numerically solving the outer
        # plume system.
        #--------------------------------------------#

        jdp = 0     # Number of discrete peels
        jo = 0      # Number of outer plumes

        z0, y0, zf = self.outer_surf(zi[-1], ambient, zi, neighbor, jp, i_peel, p)
        flag = 1 #Assume outer plume is viable

        #Integrate the top outer plume if it is viable
        # print(f'      - Top outer plume {z0} - {zf}')
        # print(f' Initial conditions are: ')

        if flag == 1:
            sol = solve_ivp(
                fun= self.outer_derivs,
                t_span=(z0, zf),
                y0=y0,
                method='RK45',  # stiff solver
                vectorized=False,
                max_step = 0.1,
                rtol = 1e-6,
                atol = 1e-9,
                events=self.outer_stop,             # **pass the function itself**
                args=(ambient, zi, neighbor, p) 
                )
            
        zo, yo = sol.t, sol.y
        Qo = yo[0]
        Jo = yo[1]
        Ho = yo[2]
        Bo = yo[3]
        Co = yo[4]

        z_elev = np.min(zo)
        z_diff = np.min(zi)

        epsilon = p.epsilon

        # Integrate down until the diffuser is reached
        while z_elev > (z_diff + 1):

            if jp > 0:
                # Discrete peels remain

                if epsilon == 0 or np.isclose(z_elev, zi[i_peel[jp-1]], atol=0.01):
                    # We are at a discrete peeling location, get discrete peeling I.C.
                    z0, zf, y0, flag = self.outer_dpic(z_elev, ambient, zi, neighbor, jp, i_peel, p)
                    jp = jp - 1
                else:
                    z0, zf, y0, flag = self.outer_cpic(z_elev, ambient, zi, yi, neighbor,jp, i_peel, p)

            else:
                #No discrete peels are remaining
                if p.epsilon == 0:
                    # The continuous peel model is turned OFF, so we have to run the
                    # dpic model to close the plume 
                    z0, zf, y0, flag = self.outer_dpic(z_elev, ambient, zi, neighbor, jp, i_peel, p)
 
                else:
                    # We have continuous peels to still consider ... get their
                    # initial conditions 
                    z0, zf, y0, flag = self.outer_cpic(z_elev, ambient, zi, yi, neighbor,jp, i_peel, p); 
                    
            if flag == 1:
    
                jo = jo + 1

                sol = solve_ivp(
                    fun= self.outer_derivs,
                    t_span=(z0, zf),
                    y0=y0,
                    method='RK45',  # stiff solver
                    vectorized=False,
                    max_step = 0.01,
                    rtol = 1e-6,
                    atol = 1e-9,
                    events=self.outer_stop,             # **pass the function itself**
                    args=(ambient, zi, neighbor, p) 
                )

                z, y = sol.t, sol.y

            else:

                z = z0
                y = y0
            
            zo = np.append(zo, z)
            Qo = np.append(Qo, y[0])
            Jo = np.append(Jo, y[1])
            Ho = np.append(Ho, y[2])
            Bo = np.append(Bo, y[3])
            Co = np.append(Co, y[4])

            # Reset elevation
            z_elev = np.min(zo.round(2))  

        yo = np.array([Qo, Jo, Ho, Bo, Co]) 

        return yo, zo
            
    def outer_stop(self, z, y, ambient, zi, neighbor, p):

        # Stop event criteria for the outer plume.
        #--------------------------------------------#

        ambient_interp = interp1d(ambient[0, :], ambient[1:, :], axis=1, kind='linear', fill_value="extrapolate")
        Ta, Sa, Pa, Ca = ambient_interp(z)

        p_sea = (Pa - 101325) / 1e4                                   # sea pressure [dbar] for gsw
        rho_a = gsw.density.rho_t_exact(Sa, Ta, p_sea)
        rho_o = gsw.density.rho_t_exact(y[3]/y[0], y[2]/(p.rho_r * c.cp * y[0]) - c.T0, p_sea)

        value = 1

        # Stop Event 1: Are any of the state-space variables imaginary? 
        for i in range(5):
            if abs(np.imag(y[i])) > 0:
                value = 0; 

        # Stop Event 2: Did the momentum go to zero or become negative? 
        if np.round(y[1],3) <= 0: 
            # print(f"outer plume lost momentum, y[1]: {y[1]}")
            value = 0; 

        # Stop Event 3: Did the plume reach the bottom
        if z <= ambient[0,0]:
            value = 0; 
        
        if np.any(np.isnan(y)):
            value = 0.0

        return value

    outer_stop.terminal = True
    outer_stop.direction = 0

    def singlePlume_derivs(self, z, y, ambient, p):

        # System of equations for the single plume model.
        #--------------------------------------------#

        u, b, T, S, cO, Ta, Sa, Ca, \
        rho, rho_a, m_dot, Tb, Sb = self.getVars_single(y, z, ambient, p)

        yp = np.zeros(5)
        yp[0] = 2 * p.alpha_sp * b * u + 4/c.pi * b * m_dot
        yp[1] = c.g * b**2 * (rho_a - rho)/p.rho_r - 4 * p.Cd/c.pi * b * u**2
        yp[2] = 2 * p.alpha_sp * b * u * Ta + 4/c.pi * m_dot * b * Tb \
                - 4 * p.gammaT * p.Cd**(1/2)/c.pi * b * u * (T - Tb)
        yp[3] = 2 * p.alpha_sp * b * u * Sa + 4/c.pi * m_dot * b * Sb \
                - 4 * p.gammaS * p.Cd**(1/2)/c.pi * b * u * (S - Sb)
        yp[4] = 2 * p.alpha_sp * b * u * Ca 

        return yp

    def singlePlume_calc(self, y0, ambient, p):

        # Routine for numerically solving the single
        # plume system.
        #--------------------------------------------#

        sol = solve_ivp(fun= self.singlePlume_derivs,
                t_span=(6, ambient[0, -1]),
                y0=y0,
                method='RK45',
                vectorized=False,
                max_step = 0.1,
                events=[self.singlePlume_stop],          
                args=[ambient, p]
                )
        
        return sol.y, sol.t
    
    def singlePlume_stop(self, z, y, ambient, p):

        # Stop criteria for the single plume.
        #--------------------------------------------#

        value = 1.0
        u, b, T, S, c0, Ta, Sa, ca, rho, rho_a, m_dot, Tb, Sb = self.getVars_single(y, z, ambient, p)
        if y[1] < 0.01:
            print(f"Plume has slowed down. u: {u}")
            value = 0.0

        return value

    singlePlume_stop.terminal = True
    singlePlume_stop.direction = -1

    def getVars_single(self, y, z,  ambient, p):

        # Return the primitive variables for the single
        # plume model.
        #--------------------------------------------#

        ambient_interp = interp1d(ambient[0, :], ambient[1:, :], axis=1, kind='linear', fill_value="extrapolate")
        Ta, Sa, Pa, ca = ambient_interp(z)

        u = y[1]/y[0]                                                     # inner velocity
        b = (2 * y[0]/(u * c.pi))**(1/2)                                      # inner radius
        T = y[2] / (y[0])                         # inner temperature
        S = y[3] / y[0]                                                   # inner salinity
        c0 = y[4] / y[0]                                                   # inner O2 concentration

        # Melting conditions
        depth = ambient[0,-1] - z
        m_dot = self.melting(y, depth, p)
        Sb = self.boundary_salinity(y, m_dot, p)
        Tb = self.boundary_temperature(Sb, depth, p)

        p_sea = (Pa - 101325) / 1e4                                   # sea pressure [dbar] for gsw
        rho_a = gsw.density.rho_t_exact(Sa, Ta, p_sea)
        rho = gsw.density.rho_t_exact(S, T, p_sea)


        return u, b, T, S, c0, Ta, Sa, ca, rho, rho_a, m_dot, Tb, Sb
                

