"""
Closed-loop simulation for integrated NMPC, MHE, and Profile Likelihood.
Links the crystallization plant model, state/parameter estimation, and controller.
"""  
import sys
import numpy as np
from casadi import *
import time

# Append path to import project modules
from pathlib import Path  
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  
import solver_config  

from Set_up_general import *  
from class_MHE import MHE_class  
from class_MPC import MPC_class  


def run_simulation(initial_para_log,N_MPC,N_MHE,P,use_MHE=0, use_PL=0):
    """
    Description: Executes closed-loop control simulation with optional MHE and Profile Likelihood.
    #----------------------------------------------------------------------------------------------------------------------------------------------------- 
    Inputs: 
       - initial_para_log: Initial parameter values in log10 scale
       - N_MPC: horizon length for MPC
       - N_MHE: estimation horizon length for MHE
       - P: Prior arrival cost covariance/weighting matrix for parameters
       - use_MHE: Binary flag to enable Moving Horizon Estimation (1: on, 0: off) 
       - use_PL: Binary flag to enable Profile Likelihood calculation within MHE (1: on, 0: off)
    Outputs: 
       - Y_measurements: Measured process outputs 
       - U_applied: Control inputs 
       - Para_MHE: Estimated parameter trajectories from MHE in linear scale
       - X_real_tk: True plant states from 
       - Broken_MHE_tk: Solver status flags for MHE across steps (0: converged, 1: broken)
       - Broken_MPC_tk: Solver status flags for MPC across steps (0: converged, 1: broken)
       - Obj_tk: Profile likelihood objective values over parameter grid
       - Para_PL: Tested parameter grid points in linear scale for PL
       - Broken_PL_tk: Solver status flags for PL optimizations (0: converged, 1: broken)
       - t_matrix: Execution time per step for [MPC, MHE, PL]
       - t_mean: Average timing metrics summary array (total per step and per solver call)
    Notes / Math: 
       - Integrates dynamic Radau collocation models for NMPC and MHE NLP formulations
       - Handles horizon shrinking at batch termination
    """  

    # initialize matrices - applied controls
    U_applied = np.zeros((p.n_u,p.N_sample))

    # initialize timing matrix (MPC, MHE, PL)
    t_matrix = np.zeros((p.N_sample+1, 3))

    # Data from complex model
    Y_measurements = np.zeros((p.n_s,p.N_sample+1)) 
    Y_measurements[:,0] = p.Measurement_Eq(X0,X_lb,k=0)

    # real state matrix
    X_real_tk = np.zeros((p.n_s,p.N_sample+1))
    X_real_tk[:,0] = X0

    # matrix for MPC solver status flag 
    Broken_MPC_tk = np.ones((p.N_sample))

    # value of objective function
    Obj_tk = np.zeros((p.n_p,p.N_sample+1,N_LL+1))
    Obj_opt_tk = np.zeros((p.N_sample+1))
    Para_PL = np.zeros((p.n_p,p.N_sample+1,N_LL+1))
    Broken_PL_tk = np.ones((p.n_p,p.N_sample+1,N_LL+1))

    # initial value of MHE
    X_traj = np.zeros((p.n_s,p.N_sample+1))
    X_traj[:,0] = X0

    # initialize matrices - controls MHE
    Para_MHE = np.zeros((p.n_p,p.N_sample+1))

    # matrix for solver status
    Broken_MHE_tk = np.ones((p.N_sample+1))

    # initial parameter prediction
    para_pred = 10**initial_para_log

    # initialize boolean - shrink horizon if at end of process operation
    shrink = 0
    N_MHE_i = 1

    # initial value for MPC horizon (cap by simulation samples to avoid dimension mismatch)
    N_MPC_i = N_MPC

    # get MPC problem
    MPC = MPC_class(Collocation=Collocation,Model = p,x_max = x_max,n_coll=n_coll,N_hor=N_MPC_i,F = F_RHS,F_int=F_int,X_lb = X_lb,c_in=c_in,max_iter=200)

    # initial control estimate
    u0_mat = np.repeat(u0[:, None], N_MPC_i - 1, 1)

    # arrival cost weighting
    Pi = P.copy()

    # reshape P for initial step
    P_double_reshaped = reshape(inv(Pi),p.n_p*p.n_p,1).full().squeeze()

    # number of multistart optimizations 
    n_opt = 1
    n_opt_PL = 3

    # variables for MPC solver fallback
    u_MPC_last = None
    k_last_success = -1

    for k in range(p.N_sample+1):

        ################################################################################# MHE ############################################################################################
        ##################################################################################################################################################################################
        if use_MHE:
            if k == 0:
                # Save initial nominal parameter guess
                para_pred = 10**initial_para_log.squeeze()
                Para_MHE[:, 0] = para_pred
                Obj_opt_tk[0] = 0.0
                Broken_MHE_tk[0] = 0
                t_matrix[0, 1] = 0.0
                
                # If PL is used, save initial parameter guess for PL too
                if use_PL:
                    t_matrix[0, 2] = 0.0
                    for idx_not_opt in range(p.n_p):
                        var_para_i = np.sort(np.concatenate((var_para, np.log10(para_plant[:, None])), axis=1), axis=1)
                        for j in range(N_LL + 1):
                            Obj_tk[idx_not_opt, 0, j] = 0.0
                            Para_PL[idx_not_opt, 0, j] = 10**var_para_i[idx_not_opt, j]
                            Broken_PL_tk[idx_not_opt, 0, j] = 0
            else:
                if k<=N_MHE: # fill up estimation window when not enough measurements are available 
                    N_MHE_i = k+1 # estimation horizon
                    # get MHE problem
                    MHE = MHE_class(Collocation=Collocation,Model = p,x_max = x_max,n_coll=n_coll,N_hor=N_MHE_i,F =F_RHS,F_int=F_int,X_lb = X_lb,c_in=c_in,lb_para=lb_para, ub_para=ub_para) 
                    # index of estimation horizon
                    idx_th_MHE = np.arange(k+1) 
                else:
                    # shift index of estimation horizon
                    idx_th_MHE = np.arange(k+1 - N_MHE_i , k+1 ) 

                # get data
                Y_th = Y_measurements[:,idx_th_MHE].T.reshape(-1,1).squeeze()

                # controls of estimation horizon
                U_th_vec = U_applied[:,idx_th_MHE[:-1]].T.reshape(-1,1).squeeze()
                
                # get initial conditions for solver
                if k == 1:
                    para0 = initial_para_log.squeeze()
                    x_est_tk_MHE = X0
                else:
                    para0 = para_opt.squeeze()  # warm start parameters from previous step
                    if idx_th_MHE[0] == 0:
                        x_est_tk_MHE = X0
                    else:
                        x_est_tk_MHE = X_traj[:, idx_th_MHE[0]]

                # set up matrices
                Obj_opt_n = np.ones(n_opt)*1e9 # for objective
                broken_n = np.zeros(n_opt) # for solver flag
                para_opt_n = np.zeros((p.n_p,n_opt)) # for parameter values

                # initial value for multistart
                para0_n  = para0

                t_mhe_start = time.perf_counter()
                for idx_opt_run in range(n_opt): # run over number of multistarts
                    # solve MHE problem
                    x_MHE, para_opt,Obj_opt,broken =  MHE.MHE_step(x_est_tk_MHE,Y_th,U_th_vec,n_coll,para0_n,P_double_reshaped,R_inv,N0) 

                    # save optimal values
                    Obj_opt_n[idx_opt_run] = Obj_opt
                    broken_n[idx_opt_run] = broken
                    para_opt_n[:,idx_opt_run] = para_opt.squeeze()

                    # perturb for multistart (perturb relative to nominal guess, prevent dimension accumulation) 
                    para0_n = para0 + (np.random.rand(p.n_p) - 0.5) * 0.1
                t_matrix[k, 1] = time.perf_counter() - t_mhe_start

                # find optimal values from multistart
                idx_opt_val = np.where(np.min(Obj_opt_n)==Obj_opt_n)[0][0] # index of optimal value
                para_opt = para_opt_n[:,idx_opt_val][:,None] # optimal parameter values (log-space)
                Obj_opt =  Obj_opt_n[idx_opt_val] # optimal objective value

                # get optimal values
                para_pred = 10**para_opt.squeeze() # optimal parameter
                Obj_opt_tk[k] = Obj_opt # optimal objective
                
                # save next point in trajectory
                X_traj[:,idx_th_MHE] = x_MHE

                # parameter vector
                Para_MHE[:,k] = para_pred

                # save if optimization broke
                Broken_MHE_tk[k] = broken_n[idx_opt_val]

                if use_PL:
                    t_pl_start = time.perf_counter()
                    for idx_not_opt in range(p.n_p): # run over number of parameters 
                        # get PL problem
                        PL = PL_class(Collocation=Collocation,Model = p,x_max = x_max,n_coll=n_coll,N_hor=N_MHE_i,F_int =F_int,F =F_RHS,X_lb = X_lb,c_in=c_in,idx_not_opt=[idx_not_opt],lb_para=lb_para, ub_para=ub_para) 

                        # index of parameter
                        idx_all = np.arange(p.n_p)
                        idx_opt = np.delete(idx_all, idx_not_opt)

                        # add optimal parameter to PL testing vector 
                        var_para_i =  np.sort(np.concatenate((var_para,np.log10(para_plant[:,None])),axis = 1),axis=1)

                        for j in range(N_LL+1):
                            # fixed parameter value
                            para_fix = var_para_i[idx_not_opt,j] # in log space
                            
                            # estimated initial condition
                            para0 = np.clip(para_opt[idx_opt],a_min=-5,a_max=6)#.squeeze() # in log scale

                            # estimated initial condition
                            x_est_tk_PL =  x_MHE[:,0] #estimate based on fitted model

                            # set up matrices for PL multistart
                            Obj_opt_n_PL = np.ones(n_opt_PL)*1e9 # for objective function value
                            broken_n_PL = np.zeros(n_opt_PL) # for solver flag
                            para_opt_n_PL = np.zeros((p.n_p-1,n_opt_PL)) # for optimized parameters (log-space) 

                            # initial parameter value for loop
                            para0_n  = para_opt[idx_opt]
                            for n_multi in range(n_opt_PL): # multistart
                                # solve PL problem
                                para_opt_PL, Obj_opt_PL, broken = PL.PL_step(x_est_tk=x_est_tk_PL, y_data=Y_th, u=U_th_vec, n_coll=n_coll, para0=para0_n, R=R_inv, N0=N0, para_fix=para_fix)

                                # save optimal values
                                Obj_opt_n_PL[n_multi] = Obj_opt_PL.full() # objective value
                                broken_n_PL[n_multi] = broken # solver flag
                                para_opt_n_PL[:,n_multi] = para_opt_PL.squeeze() # optimal parameter value (log-space) 

                                # perturb multistart value 
                                para0_n = para_opt[idx_opt] + np.random.rand(p.n_p-1,1)

                            # find optimal values from multistart
                            idx_opt_multi = np.where(np.min(Obj_opt_n_PL)==Obj_opt_n_PL)[0][0] # index of minimal objective
                            para_opt_PL = para_opt_n_PL[:,idx_opt_multi] # get optimal parameter based on index
                            Obj =  Obj_opt_n_PL[idx_opt_multi] # get optimal parameter based on index

                            # refill parameter vector
                            para_full = np.zeros(p.n_p)
                            para_full[idx_opt] = 10**para_opt_PL.squeeze()
                            para_full[idx_not_opt] = 10**para_fix
                            
                            # objective function value
                            Obj_tk[idx_not_opt,k,j] = Obj
                            
                            # parameter vector
                            Para_PL[idx_not_opt,k,j] = 10**para_fix

                            # save if optimization broke
                            Broken_PL_tk[idx_not_opt, k, j] = broken_n_PL[idx_opt_multi]
                    t_matrix[k, 2] = time.perf_counter() - t_pl_start
        ################################################################################# MPC ############################################################################################
        ##################################################################################################################################################################################
        
        if k==p.N_sample:
            break

        if k>(p.N_sample+1)-N_MPC_i: # shrink horizon

            N_MPC_i = (p.N_sample+1)-k # control horizon length

            # boolean flag for shrunken horizon 
            shrink = 1 

            # get MPC problem
            MPC = MPC_class(Collocation=Collocation,Model = p,x_max = x_max,n_coll=n_coll,N_hor=N_MPC_i,F = F_RHS,F_int=F_int,X_lb = X_lb,c_in=c_in,max_iter=200)

        #---------------------------------------------------------------- run MPC ---------------------------------------------------------------------------------------------------
        # estimated initial condition
        if k == 0: # for startup
            uk_last = u0 # initial control
            x_est_tk_MPC = X0.copy() # start with known initial condition in state space (L3)
            weight = np.array([0]) # weight to use control change in objective   
        else:
            if use_MHE:
                x_est_tk_MPC =  np.maximum(x_MHE[:, -1].copy(), 0) # use MHE filtered state as initial condition (already in state space L3)
            else:
                x_est_tk_MPC  = Y_measurements[:,k].copy() # use measurement as initial condition 
                #x_est_tk_MPC[p.idx_L] = p.mass_2_L(x_est_tk_MPC[p.idx_L], N0) # convert back to state space (L3)

            weight = np.array([1]) # weight to use control change in objective 
            if shrink:
                    u0_mat = u_MPC[:,1:] # initial control estimate - use last prediction, shrink horizon
                    uk_last = U_applied[:,k-1]
            else:
                    u0_mat = np.zeros((p.n_u,N_MPC_i-1))
                    u0_mat[:,:-1] = u_MPC[:,1:]  # initial control estimate - use last prediction
                    uk_last = U_applied[:,k-1]

        # reshape controls to row vector
        u0_reshaped = u0_mat.T.reshape(-1,1)

        # compute MPC step
        t_mpc_start = time.perf_counter()
        _, u_MPC_calc, broken =  MPC.MPC_step(x_est_tk_MPC,u0_reshaped,para_pred,n_coll,N0,uk_last,weight)
        t_matrix[k, 0] = time.perf_counter() - t_mpc_start

        if broken:
            # Solver broke! Fallback to the according control from the last successful trajectory.
            if u_MPC_last is not None:
                elapsed = k - k_last_success
                u_MPC = np.zeros((p.n_u, N_MPC_i - 1))
                remaining_len = u_MPC_last.shape[1] - elapsed
                if remaining_len > 0:
                    copy_len = min(remaining_len, N_MPC_i - 1)
                    u_MPC[:, :copy_len] = u_MPC_last[:, elapsed : elapsed + copy_len]
                    if (N_MPC_i - 1) > copy_len:
                        u_MPC[:, copy_len:] = u_MPC_last[:, -1:]
                else:
                    u_MPC[:, :] = u_MPC_last[:, -1:]
            else:
                # No previous successful MPC step, fallback to the initial control estimate uk_last
                u_MPC = np.zeros((p.n_u, N_MPC_i - 1))
                u_MPC[:, :] = uk_last[:, None]
        else:
            # Solver converged successfully! Save the trajectory and index.
            u_MPC = u_MPC_calc
            u_MPC_last = u_MPC.copy()
            k_last_success = k

        # save applied control
        U_applied[:,k] = np.squeeze(u_MPC[:,0])

        # save solver flag
        Broken_MPC_tk[k] = broken

        # -------------------------------------------next measurement-------------------------------------------------------------------------   
        # Apply the control to real dynamic system
        X_real_tk[:,k+1] = p.step_int(F_int = F_int,x0 = X_real_tk[:,k],U = U_applied[:,k],para = np.concatenate((para_plant.squeeze(),N0)))

        # get measurement
        Y_measurements[:,k+1] = p.Measurement_Eq(X_real_tk[:,k+1],X_lb,k=k+1)    

    # Calculate medium (average) iteration time for all optimizations.
    steps_mpc = p.N_sample
    steps_mhe = p.N_sample if use_MHE else 0
    steps_pl = p.N_sample if (use_MHE and use_PL) else 0

    mean_mpc_step = np.mean(t_matrix[:p.N_sample, 0]) if steps_mpc > 0 else 0.0
    mean_mhe_step = np.mean(t_matrix[:, 1]) if steps_mhe > 0 else 0.0
    mean_pl_step = np.mean(t_matrix[:, 2]) if steps_pl > 0 else 0.0

    mean_mpc_single = mean_mpc_step
    mean_mhe_single = mean_mhe_step / n_opt if (steps_mhe > 0 and n_opt > 0) else 0.0
    mean_pl_single = mean_pl_step / (p.n_p * (N_LL + 1) * n_opt_PL) if (steps_pl > 0 and n_opt_PL > 0) else 0.0

    # t_mean:
    # Row 0: Mean total time spent per simulation step.
    # Row 1: Mean time per single solver call (taking into account multistart).
    
    t_mean = np.array([
        [mean_mpc_step, mean_mhe_step, mean_pl_step],
        [mean_mpc_single, mean_mhe_single, mean_pl_single]])

    return Y_measurements, U_applied, Para_MHE, X_real_tk, Broken_MHE_tk, Broken_MPC_tk, Obj_tk, Para_PL, Broken_PL_tk, t_matrix, t_mean

