
import numpy as np
from casadi import *

class Crystallization():
    """
    Multi-phase enzymatic reactive crystallization process model.
    Defines governing ODEs, parameters, smooth approximations, and integrator interfaces.
    """
    def __init__(self,t_end,Ts):
        """
        Description: Initializes the crystallization model with state/input dimensions,
                     simulation durations, and free dynamics parameters.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs: 
            - t_end: Final simulation time [h]
            - Ts: Sampling time step  [h]
        Outputs:
            - None (Initializes instance attributes)
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        
        # number of states
        self.n_s = 10  # states [-]
        self.n_u = 3   # control actions [-]
        self.n_p = 3   # parameter (free) [-]
        self.n_y = self.n_s # outputs - all states are measured [-]
        
        # sampling interval
        self.dt = Ts # [h]

        # final simulation time
        self.t_end = t_end # [h]

        # number of samples
        self.N_sample = round(self.t_end/(self.dt)) #[-]

        # time span - from zero to t_end in sampling intervals
        self.tspan = np.squeeze(np.linspace(0,self.t_end*3600,self.N_sample+1)) # [s] 

        # initialize system parameters
        self.parameters()

        # Preset free parameters for dynamics
        self.Kcat_f =  1e-2 # forward reaction rate constant [1/s]
        self.k_G = 1e-5 # product crystal growth rate constant [mm/s]
        self.k_d = 1e-3 # donor crystal growth rate constant [1/s]

        # set plant parameter
        self.para_plant = np.array([self.Kcat_f, self.k_G, self.k_d])

        # initialize smooth function handles as CasADi Function objects
        self.setup_smooth_functions()

    def initialize_noise(self,std_mess):
        """
        Description: Initializes a deterministic noise sequence for measurement simulation
                     with zero mean and the specified standard deviation per state.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - std_mess: 1D array of standard deviations for measurement noise of each state
        Outputs:
            - None (Sets self.noise_sequence)
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 

        # generate global noise sequence
        noise_sequence = np.random.randn(self.n_s,self.N_sample+1)  # Standard normal noise sequence [-]
        noise_sequence -= noise_sequence.mean(axis=1)[:, None]  # Centering to ensure exactly zero mean [-]
        noise_sequence *= std_mess[:,None] / noise_sequence.std(ddof=0,axis = 1)[:,None] # Scaling to ensure exactly desired std
        self.noise_sequence = noise_sequence

    def parameters(self):

        """
        Description: Defines all substance properties, universal constants, stoichiometric matrices,
                     indexes, and feed concentrations used in the process model.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - None
        Outputs:
            - None (Initializes instance attributes for parameters and indices)
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 

        ######################################################################## Feed ######################################################################## 
        # maximal flow rate of feed 
        V_max_in = 1*1e-3 # (from ml/min in L/min) [L/min]
        self.V_max_in = V_max_in/(60) # [L/s]

        # maximal volume of reactor
        self.V_max = 0.5 # [L]

        # controls
        self.idx_U_IPA = 0 # control index for isopropylamine feed [-]
        self.idx_U_3MAP = 1 # control index for 3-methoxyphenylacetone feed [-]
        self.idx_U_3DPPA = 2 # control index for 3,3-diphenylpropionic acid feed [-]

        ################################################################### Substance data ################################################################### 
        # ----------------------------------------------------------- Enzyme --------------------------------------------------------------------------------- 
        
        #  catalytic decay rate of enzyme
        self.k_E = 0 # [1/s]
        # molar mass of enzyme 
        self.M_E =  (103.89*1e3)*(0.99999999965e-3)/1e3 # [g/mmol] 

        # ----------------------------------------------------------- IPA ------------------------------------------------------------------------------------ 

        # molar mass of isopropylamine
        self.M_IPA = 59.112*1e-3 # [g/mmol]
        # density of isopropylamine
        self.rho_IPA = 0.690*1e3 # [g/L]


        # ----------------------------------------------------------- 3MPEA ---------------------------------------------------------------------------------- 

        # molar mass of (S)-1-(3-methoxyphenyl)ethylamine
        self.M_3MPEA = 151.209*1e-3 # [g/mmol]
        # density of (S)-1-(3-methoxyphenyl)ethylamine
        self.rho_3MPEA = 1.038*1e3 # [g/L]

        # ----------------------------------------------------------- Ac ------------------------------------------------------------------------------------- 
        
        # acetone removal/evaporation rate constant
        self.k_ACE = 1e-1 # [1/s]
        # molar mass of acetone
        self.M_Ac = 58.08*1e-3 # [g/mmol]
        # density of liquid acetone
        self.rho_l_Ac = 0.79*1e3 # [g/L]

        # ----------------------------------------------------------- 3MAP ----------------------------------------------------------------------------------- 
        
        # mass transfer rate constant for ketone 3MAP
        self.k_3MAP = 1e-2 # [1/s]
        # molar mass of 3-methoxyphenylacetone
        self.M_3MAP = 150.177*1e-3 # [g/mmol]
        # aqueous solubility of 3-methoxyphenylacetone
        self.Cs_3MAP = 15 # [mM]
        # density of 3-methoxyphenylacetone
        self.rho_3MAP = 1.094*1e3 # [g/L]

        # ----------------------------------------------------------- 3DPPA ---------------------------------------------------------------------------------- 

        # molar mass of 3,3-diphenylpropionic acid
        self.M_3DPPA = 226.27*1e-3 # [g/mmol]
        # density of 3,3-diphenylpropionic acid
        self.rho_3DPPA = 1.100*1e3 # [g/L]

        # ----------------------------------------------------------- IPA-3DPPA ------------------------------------------------------------------------------ 

        # molar mass of donor salt IPA-3DPPA
        self.M_IPA_3DPPA = self.M_IPA + self.M_3DPPA # [g/mmol]
        # solubility of donor salt IPA-3DPPA
        self.Cs_IPA_3DPPA = 55 # [mM]
        # solubility product of donor salt IPA-3DPPA
        self.Ks_IPA_3DPPA = self.Cs_IPA_3DPPA**2 # [mM^2]


        # ----------------------------------------------------------- 3MPEA-3DPPA ---------------------------------------------------------------------------- 

        # molar mass of product salt 3MPEA-3DPPA
        self.M_3MPEA_3DPPA = self.M_3MPEA + self.M_3DPPA # [g/mmol]
        # solubility of product salt 3MPEA-3DPPA
        self.Cs_3MPEA_3DPPA = 4.7 # [mM]
        # solubility product of product salt 3MPEA-3DPPA
        self.Ks_3MPEA_3DPPA = self.Cs_3MPEA_3DPPA**2 # [mM^2]
        # crystal growth rate power exponent
        self.g = 1 # [-]

        # concentrations of feed:
        # isopropylamine
        self.cin_IPA_sc = 1000 # [mmol/L]
        #  3,3-diphenylpropionic acid
        self.cin_3DPPA_sc = 1000 # [mmol/L]
        # 3-methoxyphenylacetone
        self.cin_3MAP_sc= self.rho_3MAP/self.M_3MAP # [mmol/L]

        ################################################################ universal constants ################################################################# 

        # Avogadro number
        self.N_A = 6.02214076e23 # [1/mol]


        # universal gas constant 
        self.R_gas = 8.314 # [J/(mol K)]

        ########################################################### chemical reaction - constants ############################################################ 
        # taken from DOI 10.1002/btpr.1588

        # ----------------------------------------------------------- inhibition constants [mM] -------------------------------------------------------------- 
        # inhibition constant of (S)-1-(3-methoxyphenyl)ethylamine
        self.K_i_MPEA = 1e5 # [mM]
        # inhibition constant of isopropylamine
        self.K_i_IPA = 4.281 # [mM]
        # inhibition constant of acetone
        self.K_i_Ac = 0.11 # [mM]
        # inhibition constant of 3-methoxyphenylacetone
        self.K_i_MAP = 0.09 # [mM]
        # ----------------------------------------------------------- Michaelis constant [mM] ---------------------------------------------------------------- 

        # Michaelis constant of (S)-1-(3-methoxyphenyl)ethylamine
        self.K_m_MPEA = 0.12 # [mM]
        # Michaelis constant of isopropylamine
        self.K_m_IPA = 101.28 # [mM]
        # Michaelis constant of acetone
        self.K_m_Ac = 148.99 # [mM]
        # Michaelis constant of 3-methoxyphenylacetone
        self.K_m_MAP = 1.85 # [mM]
        # ----------------------------------------------------------- Equilibrium constant ------------------------------------------------------------------- 

        # reaction equilibrium constant
        self.K_eq = 0.033 # [-]

        ############################################################### Crystallization physical properties ################################################## 
  
        # molar mass of product salt
        self.molar_mass = self.M_3MPEA_3DPPA # [g/mmol]
        # unit cell volume
        self.V_cry0_M = 1029.43*(1e-7)**3. # [mm^3]
        # crystal length-to-width ratio
        self.shape_P1 = 10 # [-]
        # crystal width-to-depth ratio
        self.shape_P2 = 1. # [-]
        # crystal dimensions array
        self.shape = np.array([self.shape_P1,self.shape_P2]) # [-]
        # number of molecules per unit cell
        self.Z_p = 2. # [-]
        # weight of one product salt molecule
        self.weight_molecule = (self.molar_mass*1e3)/self.N_A # [g/molecule]
        # density of product crystal
        self.density = (self.Z_p*self.weight_molecule)/self.V_cry0_M # [g/mm^3]
        # saturation concentration of product salt
        self.c_s = self.Cs_3MPEA_3DPPA # [mM]
   
        # shape factors of crystals
        self.shape_fac_S = 2/self.shape[0]*(1+1/self.shape[0]+1/(self.shape[0]*self.shape[1])) # crystal surface area factor [-]
        self.shape_fac_V = 1/(self.shape[0]**2*self.shape[1]) # crystal volume shape factor [-]

        # length of initial seeds
        self.initial_L3 = 1e-6 # volume of initial seed crystal [mm^3] 

        ############################################################### Stoichiometric vectors ############################################################### 
        
        # chemical reaction - in polar phase
        nue_p = np.transpose([[-1., 1., 1., -1., 0, 0]]) # stoichiometry in polar phase 
        nue_u = [[0]] # stoichiometry in non-polar phase 
        self.nue = np.concatenate((nue_p,  nue_u), axis = 0) # stoichiometry vector for chemical reaction 

        # diffusion of 3MAP - from polar to unpolar phase
        nue_3MAP_p = np.transpose([[0, 0, 0, 1, 0, 0]]) # 3MAP polar stoichiometry 
        nue_3MAP_u = [[-1]] # 3MAP non-polar stoichiometry 
        self.nue_3MAP = np.concatenate((nue_3MAP_p,  nue_3MAP_u), axis = 0) # stoichiometry vector for 3MAP mass transfer 

        # IPA Feed
        nue_IPA_3DPPA_p = np.transpose([[1, 0, 0, 0, 1, 0]]) # donor salt dissolution polar stoichiometry 
        nue_IPA_3DPPA_u = [[0]] # donor salt dissolution non-polar stoichiometry 
        self.nue_IPA_3DPPA = np.concatenate((nue_IPA_3DPPA_p,  nue_IPA_3DPPA_u), axis = 0) # stoichiometry vector for donor salt dissolution 

        # Enzyme decay
        nue_E_p = np.transpose([[0, 0, 0, 0, 0, -1]]) # enzyme decay polar stoichiometry 
        nue_E_u = [[0]] # enzyme decay non-polar stoichiometry 
        self.nue_E = np.concatenate((nue_E_p,  nue_E_u), axis = 0) # stoichiometry vector for enzyme decay 

        # evaporation
        nue_evap_p = np.transpose([[0, 0, -1, 0, 0, 0]]) # acetone evaporation polar stoichiometry 
        nue_evap_u = [[0]] # acetone evaporation non-polar stoichiometry 
        self.nue_evap = np.concatenate((nue_evap_p,  nue_evap_u ), axis = 0) # stoichiometry vector for acetone evaporation 

        # Product salt formation
        nue_Cry_p_p = np.transpose([[0, -1, 0, 0, -1, 0]]) # product crystallization polar stoichiometry 
        nue_Cry_p_u = [[0]] # product crystallization non-polar stoichiometry 
        self.nue_Cry = np.concatenate((nue_Cry_p_p, nue_Cry_p_u), axis = 0) # stoichiometry vector for product crystallization 

        ################################################################## index positions ################################################################### 

        # liquid - polar phase
        self.idx_IPA = 0 # index position for isopropylamine in polar phase 
        self.idx_3MPEA = 1 # index position for (S)-1-(3-methoxyphenyl)ethylamine in polar phase 
        self.idx_Ac = 2 # index position for acetone in polar phase 
        self.idx_3MAP_p = 3 # index position for 3-methoxyphenylacetone in polar phase 
        self.idx_3DPPA = 4 # index position for 3,3-diphenylpropionic acid in polar phase 
        self.idx_E = 5 # index position for enzyme in polar phase 

        self.idx_polar = np.array([self.idx_IPA,self.idx_3MPEA,self.idx_Ac,self.idx_3MAP_p,self.idx_3DPPA,self.idx_E]) # array of polar phase indices 

        # liquid - non polar phase
        self.idx_3MAP_np = 6 # index position for 3-methoxyphenylacetone in non-polar phase 

        # solid - donor salt
        self.idx_IPA_3DPPA = self.idx_3MAP_np + 1 # index position for solid donor salt (IPA-3DPPA) 

        # volume
        self.idx_V = self.idx_IPA_3DPPA + 1 # index position for polar phase volume 

        # crystal length
        self.idx_L = self.idx_V + 1 # index position for crystal volume (L^3) 

        ###################################################################### for feed ###################################################################### 
        
        # number of states in liquid phases
        n_pl = 7

        # define array for feed of:
        # IPA
        array_IPA = np.zeros((n_pl ,1)) # preallocate feed array for IPA
        array_IPA[self.idx_IPA] = 1 # set IPA position in feed array
        self.cin_IPA = self.cin_IPA_sc*array_IPA # feed concentration vector for IPA [mmol/L]

        # 3DPPA
        array_3DPPA = np.zeros((n_pl ,1)) # preallocate feed array for 3DPPA
        array_3DPPA[self.idx_3DPPA] = 1 # set 3DPPA position in feed array
        self.cin_3DPPA = self.cin_3DPPA_sc*array_3DPPA # feed concentration vector for 3DPPA [mmol/L]

        # 3MAP
        array_3MAP = np.zeros((n_pl ,1)) # preallocate feed array for 3MAP
        array_3MAP[self.idx_3MAP_np] = 1 # set 3MAP position in feed array
        self.cin_3MAP = self.cin_3MAP_sc*array_3MAP # feed concentration vector for 3MAP [mmol/L]    
    
    def initial_L(self,molar_np0):
        """
        Description: Calculates initial seed crystal volume L^3 (using self.initial_L3)
                     and computes the corresponding crystal count self.N_0 from molar amount.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - molar_np0: Initial product salt seed molar amount [mmol]
        Outputs:
            - L30: Initial seed crystal length (L) [mm]
            - self.N_0: Computed initial seed crystal count [-]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 

        # initial L^3
        L03 =  np.array([self.initial_L3]) # initial crystal lengthL [mm]

        # calculate initial seed number based on molar amount and length
        self.N_0 = self.mass_2_N0(molar_np0,L03) # initial seed crystal count [-]

        return L03**(1/3),self.N_0    
    
    def L_2_mass(self,L,N0):
        """
        Description: Converts mean crystal length L and crystal count N0 to molar amount.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - L3: Mean crystal length (L) [mm]
            - N0: Crystal count [-]
        Outputs:
            - molar_mass: Molar amount of product salt [mmol]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 

        # calculate molar mass from L and N0 with density, molar weight and shape factor
        molar_mass = (L**3*N0*self.shape_fac_V*self.density)/(self.molar_mass) # molar amount [mmol]

        return molar_mass
    
    def mass_2_L(self,molar_amount,N0):
        """
        Description: Converts product salt molar amount and crystal count N0 to mean crystal volume L.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - molar_amount: Molar amount of product salt [mmol]
            - N0: Crystal count [-]
        Outputs:
            - L: Mean crystal length (L) [mm]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # calculate L from molar mass and N0 with density, molar weight and shape factor
        L = ((self.molar_mass * molar_amount) / (N0 * self.shape_fac_V * self.density)) ** (1/3) # crystal length L [mm]

        return L  
    
    def mass_2_N0(self,molar_amount,L3):
        """
        Description: Converts product salt molar amount and mean crystal length L to crystal count.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - molar_amount: Molar amount of product salt [mmol]
            - L: Mean crystal length (L) [mm]
        Outputs:
            - N0: Crystal count [-]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # calculate N0 from L, molar mass and N0 with density, molar weight and shape factor
        N0 = (self.molar_mass*molar_amount)/(L3*self.shape_fac_V*self.density) # crystal count [-]

        return N0  
         
    def initial_cond(self):
            """
            Description: Sets the initial conditions for states and seeds for the simulation.
            #----------------------------------------------------------------------------------------------------------------------------------------------------- 
            Inputs:
                - None
            Outputs:
                - initial_cond: 1D array of initial state values [various units]
                - molar0_seeds: Initial seed crystal molar amount [mmol]
            """
            #----------------------------------------------------------------------------------------------------------------------------------------------------- 
            #____________________________________________________________ Volume _________________________________________________________________________________ 

            # polar phase volume
            V0 = 0.1 # [L] 

            ## molar amounts and concentrations - solid and liquid phase
            #____________________________________________________________ IPA ____________________________________________________________________________________ 

            # IPA - isopropylamine
            c0_IPA =  0 # [mM] 

            #____________________________________________________________ 3MPEA __________________________________________________________________________________ 

            # 3MPEA - (S)-1-(3-methoxyphenyl)ethylamine 
            c0_3MPEA = 0 # [mM]

            #____________________________________________________________ Ac _____________________________________________________________________________________ 

            # Ac - Acetone
            c0_Ac = 0 # [mM] 

            #____________________________________________________________ 3DPPA __________________________________________________________________________________ 

            # 3DPPA - 3,3-diphenylpropionic acid 
            c0_3DPPA = 0 # [mM] 

            #____________________________________________________________ Enzyme _________________________________________________________________________________ 

            # Calculate initial enzyme concentration
            U_ml = 80 # initial enzyme activity in solution [U/ml]
            self.U_ml = U_ml
            A_E = 170  # specific activity of enzyme [U/mg]
            self.A_E = A_E
            c0_E = (U_ml/A_E)/self.M_E  # initial 'concentration' of enzyme [mM]

            #____________________________________________________________ 3MAP ___________________________________________________________________________________ 

            # 3MAP - polar and non-polar phase
            c0_3MAP_aq =  0 # 3MAP in polar aqueous phase [mM] 
            c0_3MAP_u = 0 # 3MAP in non-polar phase [mM] 

            # initial molar amount of 3MAP in non-polar phase
            n0_3MAP_u = np.array([c0_3MAP_u*V0]) # [mmol] 

            #____________________________________________________________ initial condition - polar phase ________________________________________________________ 

            # polar phase - molar amounts
            initial_cond_p = np.array([c0_IPA,c0_3MPEA,c0_Ac,c0_3MAP_aq,c0_3DPPA,c0_E])*V0  # initial molar amounts in polar phase [mmol]

            #____________________________________________________________ Product salt 3MPEA-3DPPA _______________________________________________________________ 

            # initial molar amount for seed crystals
            molar0_seeds =  1e-3 # [mmol]

            #____________________________________________________________ Donor salt IPA-3DPPA ___________________________________________________________________ 
            #  initial molar amount of external donor salt
            n_IPA_3DPPA_ext = np.array([0]) # [mmol]

            #____________________________________________________________ concatenate initial conditions _________________________________________________________ 

            # concatenate all initial conditions
            initial_cond = np.concatenate((initial_cond_p,n0_3MAP_u,n_IPA_3DPPA_ext,np.array([V0]))) # [various units]

            return initial_cond, molar0_seeds

    def setup_smooth_functions(self):
        """
        Creates symbolic CasADi Function handles for fmax, fmin, and switch operations.
        """
        x_sym = SX.sym('x')
        y_sym = SX.sym('y')
        eps_sym = SX.sym('eps')
        
        fmax_expr = 0.5 * (x_sym + y_sym + sqrt((x_sym - y_sym)**2 + eps_sym**2))
        self._fmax_fun = Function('smooth_fmax', [x_sym, y_sym, eps_sym], [fmax_expr])
        
        fmin_expr = 0.5 * (x_sym + y_sym - sqrt((x_sym - y_sym)**2 + eps_sym**2))
        self._fmin_fun = Function('smooth_fmin', [x_sym, y_sym, eps_sym], [fmin_expr])
        
        cond_sym = SX.sym('cond')
        val_true_sym = SX.sym('val_true')
        val_false_sym = SX.sym('val_false')
        k_sym = SX.sym('k')
        switch_expr = 0.5 * (1 + tanh(k_sym * cond_sym))
        smooth_switch_expr = switch_expr * val_true_sym + (1 - switch_expr) * val_false_sym
        self._switch_fun = Function('smooth_switch', [cond_sym, val_true_sym, val_false_sym, k_sym], [smooth_switch_expr])

    def smooth_fmax(self, x, y, eps=1e-5):
        return self._fmax_fun(x, y, eps)

    def smooth_fmin(self, x, y, eps=1e-9):
        return self._fmin_fun(x, y, eps)

    def smooth_switch(self, condition_expr, val_true, val_false, k=5.0):
        return self._switch_fun(condition_expr, val_true, val_false, k)
    
    def get_model(self):
        """
        Description: Formulates the CasADi symbolic Ordinary Differential Equations (ODEs)
                     for the crystallization process and constructs the step integrator 
                     and RHS equation.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - None
        Outputs:
            - F_int_step: CasADi integrator function for single-step integration
            - F_RHS: CasADi function representing the right hand side (Ts * 3600 * Xdot)
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 

        # states: 
        # molar amount of IPA in polar phase
        n_IPA = SX.sym('n_IPA',1) # [mmol]
        # molar amount of 3MPEA in polar phase
        n_3MPEA = SX.sym('n_3MPEA',1) # [mmol]
        # molar amount of acetone in polar phase
        n_Ac = SX.sym('n_Ac',1) # [mmol]
        # molar amount of 3MAP in polar phase
        n_3MAP_p = SX.sym('n_3MAP_p',1) # [mmol]
        # molar amount of enzyme in polar phase
        n_E = SX.sym('n_E',1) # [mmol]
        # molar amount of 3DPPA in polar phase
        n_3DPPA = SX.sym('n_3DPPA',1) # [mmol]
        # molar amount of 3MAP in non-polar phase
        n_3MAP_np = SX.sym('n_3MAP_np',1) # [mmol]
        # molar amount of donor salt (solid phase)
        n_d = SX.sym('n_d',1) # [mmol]
        # volume of polar phase
        V_sym = SX.sym('V',1) # [L]
        # volume + jitter to prevent zero division
        V = V_sym + 1e-12   # [L]
        # volume of a single crystal (L^3)
        L = SX.sym('L',1)  # [mm^3]

        # control actions for ...
        # IPA feed
        U_in_IPA = SX.sym('U_in_IPA') # [-]
        # 3MAP feed
        U_in_3MAP = SX.sym('U_in_3MAP')  # [-]
        # 3DPPA feed
        U_in_3DPPA = SX.sym('U_in_3DPPA')  # [-]

        # volumetric feed flow rate of ...
        # IPA
        V_in_IPA = U_in_IPA*self.V_max_in # [L/s]
        # 3MAP
        V_in_3MAP = U_in_3MAP*self.V_max_in # [L/s]
        # 3DPPA
        V_in_3DPPA = U_in_3DPPA*self.V_max_in # [L/s]

        # donor crystal growth rate parameter 
        k_d = SX.sym('k_d')   # [1/s] 

        # crystal growth rate parameter 
        k_G_eff = SX.sym('k_G_eff')   # [mm/s] 

        # enzyme forward reaction rate constant 
        K_cat_f  = SX.sym('K_cat_f') # [1/s] 

        # enzyme reverse reaction rate constant 
        K_cat_r = K_cat_f/np.sqrt(self.K_eq/((self.K_m_MPEA*self.K_m_Ac)/(self.K_m_IPA*self.K_m_MAP))) # [1/s] 
        # number of crystals 
        N0 = SX.sym('N0',1) # [-] 

        # state vector
        self.X = vertcat(n_IPA, n_3MPEA, n_Ac, n_3MAP_p, n_3DPPA, n_E, n_3MAP_np, n_d, V_sym, L) # [various units]

        # parameter vector
        self.Para = vertcat(K_cat_f, k_G_eff,k_d, N0) # [1/s, mm/s, -]

        # control variables
        self.U = vertcat(U_in_IPA,U_in_3MAP,U_in_3DPPA) # [-]

        ######################################################### from molar amount to concentration ######################################################### 

        # calculate concentrations of...
        # IPA
        IPA = n_IPA/V # [mM]
        # 3MAP in polar phase
        MAP_p = n_3MAP_p/V # [mM]
        # 3MPEA
        MPEA = n_3MPEA/V # [mM]
        # Acetone
        Ac = n_Ac/V # [mM]
        # Enzyme 
        E = n_E/V # [mM]
        # 3DPPA 
        DPPA = n_3DPPA/V # [mM]

        ################################################################## reaction kinetic ################################################################## 
        
        # forward reaction velocity
        v_f = K_cat_f*E # [mM/s] 

        # reverse reaction velocity
        v_r = K_cat_r*E # [mM/s]

        # numerator of reaction rate expression
        N = v_f*v_r*(IPA*MAP_p-(Ac*MPEA)/self.K_eq) # [mM^2/s]

        # denominator of reaction rate expression
        D = v_r*self.K_m_IPA*MAP_p+ v_r*self.K_m_MAP*IPA + v_f*self.K_m_MPEA/self.K_eq*Ac + v_f*self.K_m_Ac/self.K_eq*MPEA \
            + v_r*IPA*MAP_p+ v_f*self.K_m_MPEA/(self.K_i_IPA*self.K_eq)*IPA*Ac + v_r*self.K_m_IPA/self.K_i_MPEA*MPEA*MAP_p+ v_f/self.K_eq*MPEA*Ac # [mM]

        # total reaction rate in polar phase
        Gamma = (N / (D + 1e-8)) * V # [mmol/s], epsilon added to avoid division by zero 

        #################################################################### enzyme decay #################################################################### 

        # enzyme activity decay rate
        dE = self.k_E*E # [mM/s]

        ############################################################### donor salt dissolution ############################################################### 

        # 1. Saturation and limit
        S_IPA_3DPPA = (IPA * DPPA) / self.Ks_IPA_3DPPA
        min_IPA_3DPPA = self.smooth_fmin(IPA, DPPA)

        # 2. Dissolution (active when 1 - S > 0)
        R_diss = k_d * n_d * self.smooth_fmax(1.0 - S_IPA_3DPPA, 0.0)

        # 3. Precipitation (active when S - 1 > 0)
        R_prec = k_d * min_IPA_3DPPA * V * self.smooth_fmax(S_IPA_3DPPA - 1.0, 0.0)

        # 4. Net dissolution rate
        r_IPA_3DPPA = R_diss - R_prec

        # external donor salt change rate
        dn_IPA_3DPPA_extdt = -r_IPA_3DPPA # [mmol/s]

        ######################################################## 3MAP diffusion from 2nd liquid phase ######################################################## 

        # 3MAP Diffusion from 2nd liquid Phase
        S_3MAP = MAP_p/self.Cs_3MAP
        switch2 = self.smooth_switch(S_3MAP - 1.0, 1.0, 0.0,k=10) #if_else(S_3MAP<1,0,1)

        # Only allow diffusion if there is actually substance in the non-polar phase
        driving_force_3MAP =  (1 - switch2)*n_3MAP_np + switch2*n_3MAP_p 

        # rate of 3MAP 
        d_3MAP = self.k_3MAP * (1 - S_3MAP) * driving_force_3MAP

        ############################################################### product crystal growth ############################################################### 

        # product salt saturation ratio
        S = (MPEA*DPPA)/self.Ks_3MPEA_3DPPA #[-]

        # growth rate - saturation (S>1) triggers growth
        k_G = k_G_eff / (N0**(1/3))
        G_p = (k_G *(S-1)**self.g) # linear growth rate [mm/s]
        G = - self.smooth_fmin(-G_p,0) #fmin(-G_p,0) # no negative growth (dissolution) [mm/s]

        # change in mean crystal volume
        dL = G  # [mm^3/s]

        # Moment dynamics of third moment (total volume change rate)
        dmoment_3 = 3*dL*N0*L**2*self.shape_fac_V # [mm^3/s]

        # flux from liquid to solid
        r_dot = (dmoment_3*self.density)/self.molar_mass # [mmol/s]

        ################################################################ Evaporation acetone ################################################################# 

        # acetone evaporation rate 
        N_dot_ACE = self.k_ACE*Ac # [mmol/s] 

        #################################################################### all dynamics #################################################################### 

        # add all dynamics with stochiometric vectors together
        dM =self.nue*Gamma +\
            self.nue_E*dE +\
            self.nue_Cry*r_dot +\
            self.nue_evap*N_dot_ACE + \
            self.nue_IPA_3DPPA *r_IPA_3DPPA +\
            self.nue_3MAP*d_3MAP # [mmol/s]
        
        # add input
        dM_u = dM + V_in_IPA*self.cin_IPA + V_in_3DPPA*self.cin_3DPPA + V_in_3MAP*self.cin_3MAP # ODEs for liquid phase molar amounts [mmol/s]

        ######################################################## volume balance - polar liquid phase ######################################################### 

        # volume change rate due to donor salt dissolution/formation
        dV_d = r_IPA_3DPPA*self.M_IPA/self.rho_IPA + r_IPA_3DPPA*self.M_3DPPA/self.rho_3DPPA # [L/s]
        # volume change rate due to product precipitation 
        dV_p = r_dot*self.M_3MPEA/self.rho_3MPEA + r_dot*self.M_3DPPA/self.rho_3DPPA #  [L/s]
        # volume change rate due to acetone evaporation
        dV_Ac = N_dot_ACE*self.M_Ac/self.rho_l_Ac # [L/s]
        # volume change rate due to 3MAP mass transfer
        dV_3MAP = d_3MAP*self.M_3MAP/self.rho_3MAP # [L/s]

        # net polar volume change rate for polar phase volume
        dV = V_in_IPA  + V_in_3DPPA + dV_d  + dV_3MAP - dV_p - dV_Ac # [L/s]

        ################################################################## Whole ODE system ################################################################## 

        # state time derivative vector
        self.Xdot = vertcat(dM_u, dn_IPA_3DPPA_extdt, dV, dL) # [various units/s]

        ################################################################## integrator setup ################################################################## 

        # CasADi ODE system logic dict
        mySys_logic = {'x':self.X,'ode':self.dt*3600*self.Xdot,'p':self.Para,'u':self.U} # CasADi ODE system logic dict
        # integrator options - integration is time scaled
        opt = {'t0':0, 'tf':1, 'reltol':1e-10, 'abstol':1e-12} # integrator options
         # CasADi step integrator function 
        F_int_step = integrator('F','cvodes',mySys_logic,opt) # CasADi step integrator function 

        # Right-hand side ODE function: dt * 3600 * Xdot
        F_RHS = Function('F_RHS',[self.X,self.U,self.Para],[self.dt*3600*self.Xdot]) 

        return F_int_step,F_RHS

    def Measurement_Eq_casadi(self):
        """
        Description: Formulates the symbolic CasADi measurement equations,
                     mapping the state vector to measured variables (converting crystal L^3 to molar mass).
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - None
        Outputs:
            - Meas_Eq: CasADi Function mapping states to measured output variables
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # symbolic state vector
        X_sym = SX.sym('X', self.n_s) # [various units]

        # every state is measured
        X_meas = X_sym

        # CasADi measurement function
        Meas_Eq = Function('Meas_Eq', [X_sym], [X_meas])

        return Meas_Eq

    def Measurement_Eq(self, X, X_lb,k,use_noise = 1):
        """
        Description: Evaluates the measurement equation for state matrix X in numpy,
                     optionally adding pre-calculated measurement noise and clipping below lower bounds.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - X: State matrix/array of shape (n_s, N) or (n_s,), unit [various units]
            - X_lb: Lower bound array of shape (n_s,), unit [various units]
            - k: Current sample step index for accessing noise sequence [-]
            - use_noise: Boolean flag to enable or disable standard measurement noise addition [-]
        Outputs:
            - y_states: Measured output matrix/array of same shape as X, unit [various units, index L converted to mmol]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # clipped states on lower bounds 
        X_lb_clip = np.maximum(0, X_lb)

        # checking if input is 1D vector
        is_vector = (X.ndim == 1)  
        if is_vector:
            X = X[:, None] # convert to 2D column vector

        # every state is measured
        y_states = X.copy()

        # Measurement equation - add noise if use_noise is true
        if use_noise:
            # Ensure noise sequence matches X shape for correct broadcasting 
            noise = self.noise_sequence[:, k] # noise vector for current step
            if noise.ndim == 1:
                noise = noise[:, None] # convert to 2D column vector
            y_states = y_states + noise # add noise to state measurements

        # check for feasible states
        y_states = np.maximum(y_states, X_lb_clip[:, None]) # enforce lower bounds on measurements

        # Return same shape as input
        if is_vector:
            return y_states[:, 0]  # return as 1D vector if input was 1D
        
        return y_states
    
    def full_int(self,F_int,x0,U,para,p):
        """
        Description: Integrates the state equations over the entire simulation horizon step-by-step.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - F_int: CasADi step integrator function
            - x0: Initial state vector of shape (n_s,), unit [various units]
            - U: Control input history matrix of shape (n_u, N_sample), unit [-]
            - para: Parameter vector (Kcat_f, k_G, N0), unit [1/s, mm/s, -]
            - p: crystallization parameters container object
        Outputs:
            - X_res: Simulated state trajectory matrix of shape (n_s, N_sample+1), unit [various units]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # state trajectory storage 
        X_res = np.zeros((x0.shape[0],p.N_sample+1))
        X_res[:,0] = x0.copy().squeeze() # store initial conditions

        for i in range(p.N_sample): # loop through simulation steps

            # solve problem by integrator
            X_res[:,i+1] = self.step_int(F_int,X_res[:,i],U[:,i],para)  # integrate one step ahead

        return X_res
    
    def step_int(self,F_int,x0,U,para):
        """
        Description: Integrates state equations one sample step forward and sets negative values to zero.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - F_int: CasADi step integrator function
            - x0: Current state vector of shape (n_s,), unit [various units]
            - U: Control input vector of shape (n_u,), unit [-]
            - para: Parameter vector, unit [1/s, mm/s, -]
        Outputs:
            - X_int: Next step state vector of shape (n_s,), unit [various units] 
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
     
        # call CasADi integrator- integrate one step ahead
        x_int = F_int(x0 = x0, u=U,p=para)

        # extract state vector from output
        X_int = x_int['xf'].full().reshape(x0.shape)

        # make negative elements zero
        X_int = np.maximum(X_int, 0)

        return X_int

# Backward-compatibility alias &!
crystallization = Crystallization  # &!