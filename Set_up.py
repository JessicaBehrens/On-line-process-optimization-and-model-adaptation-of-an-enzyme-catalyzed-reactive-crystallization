"""
Simulation environment setup and global parameter configuration.
Initializes process models, horizons, initial conditions, noise covariances, and PL grids.
"""
# load packages
import solver_config  
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.gridspec import GridSpec

# call self-written classes  # &!
from class_Process_model import Crystallization, crystallization # process model  # &!
from class_Collocation import Collocation # collocation method  # &!
from class_MPC import MPC_class # Model predictive Control  problem 
from class_PL import PL_class # Profile likelihood 
from class_MHE import MHE_class # Moving Horizon Estimation 

#--------------------------------------------------Initialize seed for random numbers ----------------------------------------------
# set seed for random numbers
np.random.seed(seed=16)

#____________________________________________________________________________________________________________________________________
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: all parameters ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# simulation time
t_end = 6 # h

# sampling time 
Ts = 10/60 # every 10 min
#____________________________________________________________________________________________________________________________________
# load model
p = Crystallization(t_end = t_end,Ts = Ts) # in h

#____________________________________________________________________________________________________________________________________

# Define measurement noise standard deviation
scale_std = np.ones(p.n_s)*0.01 # 1% of mean value 

# number of collocation points
n_coll = 5
# Number of parameter grid points for Profile Likelihood (full dataset)
N_LL_full = 99
# Number of parameter grid points for Profile Likelihood (during MHE calculation)
N_LL = 39

# length of MHE horizon
N_MHE = round(2/p.dt) 

# length of MPC horizon
N_MPC = round(t_end /p.dt)

#____________________________________________________________________________________________________________________________________
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: initial condition/define model ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# initial condition
initial_cond, molar0_seeds = p.initial_cond()

# get initial number density dist
molar_n0 = np.array([molar0_seeds])

# get initial length
L03,N0 = p.initial_L(molar_n0)

# initial condition
X0 = np.concatenate((initial_cond,L03))

# get integrator - nominal model
F_int,F_RHS = p.get_model()

# Define the correct plant parameters using k_G_eff = k_G * N0^(1/3)
para_plant = np.array([p.Kcat_f, p.k_G * N0[0]**(1/3),p.k_d])

# concentration of feed
c_in = np.stack((p.cin_IPA_sc,p.cin_3MAP_sc,p.cin_3DPPA_sc))#

# time for plot in h
t = p.tspan/3600

#____________________________________________________________________________________________________________________________________
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: bounds on states and parameters  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# -----------------------------------------------------------------------
# lower bounds for states
X_lb_states = np.ones(p.n_s)*(-1)

# State lower bounds
X_lb = X_lb_states

# -----------------------------------------------------------------------
#  upper bounds for states
x_max = np.ones(p.n_s)*1e5
x_max[p.idx_L] = 1e3
x_max[p.idx_V]= p.V_max

# -----------------------------------------------------------------------
# bound for parameter - log10 scale!
lb_para = np.ones(p.n_p)*(-5)
ub_para =  np.zeros(p.n_p)
#____________________________________________________________________________________________________________________________________
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: define noise ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# -----------------------------------------------------------------------

# simulate model with full input
U_feed = np.ones((p.n_u,p.N_sample))*np.array([0.1,0.01,0.1])[:,None] 

X_plant = p.full_int(F_int=F_int,x0=X0,U=U_feed,para=np.concatenate((para_plant.squeeze(),N0)),p=p)
Mean_X = np.mean(p.Measurement_Eq(X_plant, X_lb,k=[],use_noise=0),axis = 1)

std_mess = Mean_X*scale_std


# initialize noise sequence for measurements
p.initialize_noise(std_mess)

# Measurement noise covariance matrix R
R = std_mess**2 
R[R== 0] = 1e-4 

# invert
R_inv =  1/R 

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: for testing of methods - in log-space!/Parameter for profile likelihood (PL)   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# create different parameter initial values
n_para_var = 4
initial_para = np.zeros((p.n_p,n_para_var))

# create parameter vector for samples of PL
var_para = np.zeros((p.n_p,N_LL))

# create parameter vector for samples of PL
var_para_full = np.zeros((p.n_p,N_LL_full))

# Starting points for parameter perturbation cases (e.g. +/- 10%)
spread = np.array([0.1,0.5,0.1])
lb_list = []
ub_list = []
for i in range(p.n_p):
    p_opt = para_plant[i]
    lb_val = np.clip(p_opt-p_opt * spread[i], a_min=10.**lb_para[i], a_max=10.**ub_para[i])
    ub_val = np.clip(p_opt+p_opt * spread[i], a_min=10.**lb_para[i], a_max=10.**ub_para[i])
    lb_list.append(lb_val)
    ub_list.append(ub_val)

lb_log = np.log10(lb_list)
ub_log = np.log10(ub_list)

# Case 1: Both below plant values [lb_0, lb_1]
initial_para[0,0] = lb_log[0]
initial_para[1,0] = lb_log[1]
initial_para[2,0] = ub_log[2]

# Case 2: Both above plant values [ub_0, ub_1]
initial_para[0,1] = ub_log[0]
initial_para[1,1] = ub_log[1]
initial_para[2,1] = lb_log[2]

# Case 3: Mixed case 1 [lb_0, ub_1]
initial_para[0,2] = lb_log[0]
initial_para[1,2] = ub_log[1]
initial_para[2,2] = lb_log[2]

# Case 4: Mixed case 2 [ub_0, lb_1]
initial_para[0,3] = ub_log[0]
initial_para[1,3] = lb_log[1]
initial_para[2,3] = ub_log[2]

# for PL calculation
for i in range(p.n_p):

    p_opt = para_plant[i] #para_opt[idx_not_opt] #

    lb = np.clip( p_opt * (1 - spread[i]),a_min=10.**lb_para[i],a_max=10.**ub_para[i])
    ub =  np.clip(p_opt * (1 + spread[i]),a_min=10.**lb_para[i],a_max=10.**ub_para[i])

    # Discretize parameter values for PL calculation
    var_para[i,:] = np.log10(np.linspace(lb, ub, N_LL)).T.squeeze()

# for PL calculation - full dataset
for i in range(p.n_p):

    p_opt = para_plant[i] #para_opt[idx_not_opt] #

    lb = np.clip( p_opt * (1 - spread[i]),a_min=10.**lb_para[i],a_max=10.**ub_para[i])
    ub =  np.clip(p_opt * (1 + spread[i]),a_min=10.**lb_para[i],a_max=10.**ub_para[i])

    # Discretize parameter values for PL calculation with full dataset
    var_para_full[i,:] = np.log10(np.linspace(lb, ub, N_LL_full)).T.squeeze()

#____________________________________________________________________________________________________________________________________
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: Parameter for moving horizon estimation (MHE)  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Initial covariance matrix P for parameter estimation
P = np.eye(p.n_p)

#____________________________________________________________________________________________________________________________________
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: Parameter for model predictive control (MPC) ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# initial controls 
u0 = np.array([0.1,0.1,0.1]) # initial controls
