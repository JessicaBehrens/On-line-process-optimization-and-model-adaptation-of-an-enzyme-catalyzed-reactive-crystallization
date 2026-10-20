import numpy as np
from casadi import *
from solver_config import HSL_LIB_PATH 


class MPC_class():
    """
    Nonlinear Model Predictive Control (NMPC) formulation using Radau orthogonal collocation.
    Constructs the CasADi NLP with stage/terminal costs, volume constraints, and IPOPT interface.
    """
    def __init__(self,Collocation,F,Model,F_int,N_hor,n_coll,x_max,X_lb,c_in,max_iter=100):
        """
        Description: Initializes the Model Predictive Control (MPC) setup with parameters,
                        horizons, bounds, cost weights, and optimization solver.
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        Inputs:
            - Collocation: Collocation class helper function for constructing dynamic constraints
            - F: CasADi function representing the process ODE right-hand side (RHS)
            - Model: Process model object containing parameter definitions and state specifications
            - F_int: CasADi step integrator function
            - N_hor: MPC prediction horizon length [-]
            - n_coll: Number of collocation points used per interval [-]
            - x_max: State upper bound array [various units]
            - X_lb: State lower bound array [various units]
            - c_in: Feed concentrations vector [mmol/L] 
            - max_iter: Maximum number of IPOPT iterations [-] 
        Outputs:
            - None (Initializes instance attributes)
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        #____________________________________________________________ Setup problem __________________________________________________________________________ 
        # model
        self.p = Model  

        # step integrator
        self.F_int = F_int  

        # prediction horizon
        self.N_hor = N_hor  

        # parameter collocation points count
        self.n_coll = n_coll  

        # state upper bounds
        self.x_max = x_max 

        # state lower bounds
        self.X_lb = X_lb 

        # solver max_iter and tolerance
        self.max_iter = max_iter

        # feed concentrations vector
        self.c_in = c_in # [mmol/L] 

        # initial state and seed values
        self.initial_cond,self.molar0_seeds = self.p.initial_cond()

        # collocation function
        self.Coll =  Collocation(RHS = F, model = self.p,n_s = self.p.n_s)  

        # set up solver
        self.solver = self.Problem_setup(n_coll = self.n_coll)  

    def constraints(self,n_coll,x_max):
        """
        Description: Formulates lower and upper bounds for states, controls, and volume constraints
                        along with decision variable indices for MPC collocation.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - n_coll: Number of collocation points used per interval [-]
            - x_max: State upper bound array [various units]
        Outputs:
            - args: Dictionary containing dimensions (n_eq, n_ineq, n_x, n_u) and bounds (lb, ub, lbX, ubX, lbu, ubu, lbg, ubg)
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        #____________________________________________________________ bounds _________________________________________________________________________________ 
        # model
        p = self.p 

        # structure for bounds
        args = {} 

        # number of equality constraints
        n_eq = (p.n_s+p.n_s*(n_coll))*self.N_hor - p.n_s*(n_coll) # for Collocation 

        # save equality constraints dimension
        args.update({"n_eq": n_eq}) 

        # number of inequality constraints (total volume at each of the N_hor steps)
        n_ineq = self.N_hor 

        # save inequality constraints dimension
        args.update({"n_ineq": n_ineq}) 

        # number of states - decision variable
        n_x = (self.N_hor)*p.n_s + p.n_s*(n_coll)*(self.N_hor-1) # for Collocation 
        
        # save state variable dimension
        args.update({"n_x": n_x}) 

        # number of control - decision variable
        n_u = (self.N_hor-1)*p.n_u 

        # save control variable dimension
        args.update({"n_u": n_u}) 

        # lower bounds for constraints: equalities are 0, inequalities are [-inf, V_max]
        lbg_eq = np.zeros((n_eq, 1))
        lbg_ineq = np.zeros((n_ineq, 1)) # for total volume [L] 

        # save constraint - lower bounds
        args.update({"lbg": np.vstack((lbg_eq, lbg_ineq))}) 

        # upper bounds for constraints: equalities are 0, inequalities are [-inf, V_max]
        ubg_eq = np.zeros((n_eq, 1))
        ubg_ineq = p.V_max * np.ones((n_ineq, 1)) # for total volume [L] 

        # save constraint - upper bounds
        args.update({"ubg": np.vstack((ubg_eq, ubg_ineq))}) 

        # bounds for states
        lb_states = self.X_lb # lower
        ub_states = x_max[:p.n_s,None] # upper

        # flatten bounds
        lb_X = lb_states.flatten() 
        ub_X = ub_states.flatten() 

        # state bounds -  expand for collocation
        args.update({"lbX": np.tile(lb_X,int(n_x/p.n_s))}) # lower bound 
        args.update({"ubX": np.tile(ub_X,int(n_x/p.n_s))}) # upper bound 

        # for controls
        args.update({"lbu": np.zeros((n_u,1))})  # lower bound 
        args.update({"ubu": np.ones((n_u,1))})   # upper bound 

        # for optimization variables
        args.update({"lb": np.vstack((np.reshape(args.get("lbX"),(n_x,1)),np.reshape(args.get("lbu"),(n_u,1))))}) # lower bounds
        args.update({"ub": np.vstack((np.reshape(args.get("ubX"),(n_x,1)),np.reshape(args.get("ubu"),(n_u,1))))}) # upper bounds
        
        # idx of sampled states
        self.idx_x_sampled = [np.arange(p.n_s)]  # Initial range for the first sampled state 

        # Define the index range to loop through
        for i in range(p.n_s + p.n_s * n_coll, n_x, p.n_s * (n_coll + 1)):
            self.idx_x_sampled.append(np.arange(i , i + p.n_s ))  # Append next state indices 

        return args

    def perturb_state(self,rand_states,X_traj,n_coll):
        """
        Description: Replicates states at collocation points and reshapes the trajectory,
                    adding random perturbations to form the initial decision variables vector.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - rand_states: Random perturbation array to add to the reshaped states vector
            - X_traj: Warm-start state trajectory matrix of shape (n_s, N_hor) [various units]
            - n_coll: Number of collocation points used per interval [-]
        Outputs:
            - X0_reshaped: Column vector of initial state decision variables [various units]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        tau = collocation_points(n_coll, "radau")
        X0_list = []
        for i in range(self.N_hor - 1):
            x_start = X_traj[:, i]
            x_end = X_traj[:, i+1]
            X0_list.append(x_start[:, None])
            for j in range(n_coll):
                x_j = x_start + tau[j] * (x_end - x_start)
                X0_list.append(x_j[:, None])
        X0_list.append(X_traj[:, -1][:, None])
        X0_reshaped = np.vstack(X0_list) + rand_states 

        return X0_reshaped
   
    def calc_warm_startup(self,X0,U,para,N0):
        """
        Description: Integrates the process model step-by-step from initial condition X0
                     to compute a warm-start state trajectory over the prediction horizon.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - X0: Initial state vector at the start of the horizon [various units]
            - U: Control inputs history matrix of shape (n_u, N_hor-1) [-]
            - para: Parameters vector [various units]
            - N0: Seed crystal number parameter [-]
        Outputs:
            - X_traj_init: Warm-start state trajectory matrix of shape (n_s, N_hor) [various units]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # model
        p = self.p 

        # Simulate MPC trajectory for warm-start step-by-step &!
        X_traj_init = np.zeros((p.n_s,self.N_hor)) 
        X_traj_init[:, 0] = X0 
        x_current = X0.copy() 
        for step in range(self.N_hor-1):
           x_next = p.step_int(
                 F_int=self.F_int,
                 x0=x_current,
                 U=U[:, step],
                 para=np.concatenate((para.squeeze(), N0))
           ) 
           X_traj_init[:, step+1] = x_next.squeeze() 
           x_current = x_next.squeeze().copy() 

        return X_traj_init 
       
    def MPC_step(self,x_est_tk,u0_mat,para,n_coll,N0,Uk_last,weight_dU):
        """
        Description: Executes a single MPC optimization step, solving the NLP collocation problem
                     and dynamically adjusting collocation order if the solver fails to converge.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - x_est_tk: Initial state estimate vector [various units]
            - u0_mat: Warm-start control inputs guess matrix [-]
            - para: Parameters vector [various units]
            - n_coll: Current collocation order [-]
            - N0: Seed crystal number parameter [-]
            - Uk_last: Control input vector from previous step [-]
        Outputs:
            - x: Predicted state trajectory matrix of shape (n_s, N_hor) [various units]
            - u: Optimized control inputs history matrix of shape (n_u, N_hor-1) [-]
            - broken: Boolean flag indicating if solver failed to converge
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # get model
        p = self.p 

        # values of previous interval
        para_opt = np.concatenate((x_est_tk.flatten(), para.flatten(), np.array([N0]).flatten(), Uk_last.flatten(),weight_dU)) 

        # dictionary for bounds
        args = self.constraints(n_coll = self.n_coll,x_max = self.x_max) 

        # get solver
        solver = self.solver 

        # count number of iterations in loop
        count_while = 0 # counter for while loop 

        # booleans
        solver_success = 0 # flag of solver 
        # collocation order change flag
        inc_n_coll = 0 # 1-number of collocation points changed 
        # convergence failure flag
        broken = 0 # 1- solver broke 
        # reset flag
        reset = 0 # collocation probelm was changed- needs to be reset

        while solver_success != 1:  

            # calculate warm startup
            X_traj = self.calc_warm_startup(x_est_tk,u0_mat.reshape((self.N_hor-1,p.n_u)).T,para,N0) 

            # decision variable - perturbate in case of multiple iterations to help solver
            if count_while > 0:
                # Do not perturb initial state variables (indices 0 to p.n_s-1)
                X0_unpert = self.perturb_state(np.zeros((args.get('n_x'), 1)), X_traj, n_coll)
                rand_states = np.zeros_like(X0_unpert)
                rand_states[p.n_s:] = np.random.randn(*X0_unpert[p.n_s:].shape) * (0.02 * np.maximum(np.abs(X0_unpert[p.n_s:]), 1e-4))
                X0_pert = X0_unpert + rand_states
            else:
                X0_pert = self.perturb_state(np.zeros((args.get('n_x'), 1)), X_traj, n_coll)

            # Clip perturbed states to the bounds to ensure initial feasibility
            X0_pert = np.clip(X0_pert, args.get('lbX')[:, None], args.get('ubX')[:, None])

            # Control perturbation: additive uniform in [-0.05, 0.05], clipped to bounds
            if count_while > 0:
                rand_u = np.random.uniform(-0.05, 0.05, size=u0_mat.shape)
            else:
                rand_u = np.zeros_like(u0_mat)
            u_init = u0_mat + rand_u # perturb initial condition
            u_iter = np.clip(u_init, 0.0, 1.0) 

            # concatenate decision variables
            x0  = np.squeeze(np.vstack((X0_pert,u_iter))) # perturbed initial condition 

            # solve
            if not hasattr(self, 'lam_x') or inc_n_coll or reset or self.lam_x.shape[0] != x0.shape[0]:
                sol = solver(x0 = x0, p=para_opt,ubx= args.get("ub"),lbx= args.get("lb"),
                                lbg= args.get("lbg"), ubg= args.get("ubg")) 
            else:
                sol = solver(x0 = x0, p=para_opt,ubx= args.get("ub"),lbx= args.get("lb"),
                                lbg= args.get("lbg"), ubg= args.get("ubg"),
                                lam_x0=self.lam_x, lam_g0=self.lam_g)

            # solution
            W = sol['x'].full() 

            # flat indices for states
            flat_indices = np.concatenate(self.idx_x_sampled) 

            # optimized state trajectories
            X = W[flat_indices] 

            # optimized control inputs
            U = W[-(self.N_hor-1)*p.n_u:] 

            # check solver status
            Solver_stats = solver.stats() 
            solver_success = Solver_stats['success'] 
            if solver_success:
                self.lam_x = sol['lam_x']
                self.lam_g = sol['lam_g']
            else:
                if hasattr(self, 'lam_x'): delattr(self, 'lam_x')
                if hasattr(self, 'lam_g'): delattr(self, 'lam_g')

            # increase while loop counter
            count_while = count_while+1 

            # redefine MPC problem, if solver is stuck - introduce more collocation points
            if count_while == 2: 
                # order of polynomial
                n_coll = 6   
                # set booleans to true
                inc_n_coll = 1
                reset = 1
            elif count_while == 4:
                # order of polynomial
                n_coll = 9
                # set booleans to true
                inc_n_coll = 1

            elif count_while==6 and solver_success!=1 :
                # stop calculation, solver does not converge
                broken = 1

            if broken :
                break
                
            if inc_n_coll == 1:  # update collocation problem
                # reset 
                inc_n_coll = 0

                # get new constraints
                args = self.constraints(n_coll=n_coll,x_max = self.x_max)

                # values of previous interval
                para_opt = np.concatenate((x_est_tk.flatten(), para.flatten(), np.array([N0]).flatten(), Uk_last.flatten(),weight_dU))

                # create solver with new number of collocation points
                solver = self.Problem_setup(n_coll = n_coll)

        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        # get best solution - states
        x = X.reshape((self.N_hor,p.n_s)).T 

        # get best solution - controls
        u = U.reshape((self.N_hor-1,p.n_u)).T 

        # reset Coll problem to n_coll = 3
        if reset == 1:

           # order of polynomial
           n_coll = 3
           solver = self.Problem_setup(n_coll = n_coll)
           args = self.constraints(n_coll,x_max = self.x_max)

        return x, u, broken 
      
    def Costs(self, U, U_last, X, N0, w_move_val):
        """
        Description: Computes the stage cost penalizing control changes and rewarding solid product yield.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - U: Current control input vector [-]
            - U_last: Previous control input vector [-]
            - X: Current state vector [various units]
            - N0: Seed crystal number parameter [-]
            - w_move_val: Move penalty weight value for this step [-]
        Outputs:
            - J: Calculated stage cost value [-]
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # model
        p = self.p 

        # penalty for control change
        dU = U - U_last 
        # move penalty
        move_penalty = sum(w_move_val*5e2 * dU**2) 

        # Solid product salt (mmol) at current stage
        salt = p.L_2_mass(X[p.idx_L], N0) - self.molar0_seeds

        # Stage cost tracking error to pull the salt yield up towards the high reference
        yield_reward = salt

        # reduce second phases
        J_contaminants =  X[p.idx_IPA_3DPPA] + X[p.idx_3MAP_np] # use zero as lower bounds, because states can be negative (for solver)

        # Total stage cost
        J =  move_penalty - 1e2*yield_reward + J_contaminants + 1e2*sum(fmin(X,0)**2) # additional term to penalize negative states

        return J   
     
    def Terminal_Costs(self, Xk, N0):
        """
        Description: Computes the terminal cost penalizing wasted reactants and rewarding product yield.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - Xk: Final state vector at the end of the horizon [various units]
            - N0: Seed crystal number parameter [-]
        Outputs:
            - J_STY: Log Space-time yield terminal cost value [-]
            - J_contaminants: Log terminal penalty for solid donor salt and non-polar ketone [-]
        """

        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        p = self.p

        # Solid product salt (mmol)
        salt = fmax(p.L_2_mass(Xk[p.idx_L], N0) - self.molar0_seeds,0.)

        # maximize Log Space-time yield
        V_tot = self.total_volume(Xk, N0)
        eps = 1e-3  # Regularization constant to prevent log(0)
        
        # Log-transformed STY cost (replaces -10 * salt / (V_tot * p.t_end))
        J_STY = - 1e3 * (log(salt + eps) - log(V_tot) - log(p.t_end))

        # reduce second phases
        J_contaminants = 1e2 *(Xk[p.idx_IPA_3DPPA] + Xk[p.idx_3MAP_np]) # use zero as lower bounds, because states can be negative (for solver)

        return J_STY, J_contaminants    
       
    def total_volume(self,Xk,N0):
        """
        Description: calculate total reactor volume
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - Xk: Final state vector at the end of the horizon [various units]
            - N0: Seed crystal number parameter [-]
        Outputs:
            - V_tot: total reactor volume [L]
        """    
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 

        # model 
        p = self.p 

        # total volume - # use zero as lower bounds, because states can be negative (for solver)
        V_tot = Xk[p.idx_V] + Xk[p.idx_3MAP_np]*p.M_3MAP/p.rho_3MAP + (Xk[p.idx_L]**3)*N0*p.shape_fac_V*1e-6 + Xk[p.idx_IPA_3DPPA]*(p.M_IPA/p.rho_IPA + p.M_3DPPA/p.rho_3DPPA)

        return V_tot
    
    def Problem_setup(self,n_coll):
        """
        Description: Formulates the CasADi MPC optimization problem using collocation and volume constraints
                     and configures the IPOPT solver with custom Jacobian and Hessian evaluations.
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        Inputs:
            - n_coll: Number of collocation points used per interval [-]
        Outputs:
            - solver: CasADi NLP solver function
        """
        #----------------------------------------------------------------------------------------------------------------------------------------------------- 
        # model 
        p = self.p 

        # objective function
        obj = 0 

        # equality constraints vector
        g_eq = []  

        # inequality constraints vector
        g_ineq = [] 

        # decision variables vector
        w = [] 

        # Collocation Problem
        Coll =  self.Coll.Problem_setup(n_coll = n_coll) 
      
        # parameter value
        Para = SX.sym('Para',p.n_p) 

        # seed crystal count
        N0 = SX.sym('N0',1) 

        # boolean to use penalty
        w_move_val = SX.sym('w_move_val',1) 

        # initial state
        X0 = SX.sym('X0',p.n_s) 

        # Initial conditions - decision variable
        Xk = SX.sym('Xk0',p.n_s,1) 
        w.append(Xk) 

        # Initial controls 
        Uk = SX.sym('Uk',p.n_u,self.N_hor-1) 

        # initial previous control input
        Uk_last_init = SX.sym('Uk_last_init',p.n_u,1) 

        # initial equality constraints
        g_eq.append(X0-Xk) 

        # Total volume constraint at step 0
        g_ineq.append(self.total_volume(Xk,N0)) # [L] 

        # initialize last control input
        Uk_last = Uk_last_init 

        # define problem in loop 
        for k in range(self.N_hor-1):

           # evalualte costs
           cost = self.Costs(Uk[:,k],Uk_last,Xk,N0,w_move_val)
           obj += cost

           # create new collocation state 
           X_coll = SX.sym('Xk'+ str(k)+ '_coll', p.n_s*n_coll,1) 
           w.append(X_coll) # save variable in decision variable vector 

           # append collocation problem
           g_eq.append(Coll(X_coll,Xk,Uk[:,k],Para,N0)) 

           # for equality condition at end of interval
           Xk_end = X_coll[-p.n_s:] 

           # create new state 
           Xk = SX.sym('Xk'+str(k+1),p.n_s,1) 
           w.append(Xk) # save variable in decision variable vector 

           # collocation condition
           g_eq.append(Xk_end -Xk) 

           # Total volume constraint at step k
           g_ineq.append(self.total_volume(Xk,N0)) # [L] 

           # save last control input value
           Uk_last = Uk[:,k] 

        # Terminal reward
        J_STY,J_contaminants = self.Terminal_Costs(Xk,N0) 

        # divided terminal cost by number of iterations
        obj = obj/(self.N_hor-1) 

        # add to objective
        obj +=  J_STY + J_contaminants 

        # make the decision variable one column vector
        U_reshaped = reshape(Uk,p.n_u*(self.N_hor-1),1) 

        # make the decision variable one column vector
        OPT_variables = vertcat(*w,U_reshaped) 

        # Parameters to pass to problem
        params = vertcat(X0,Para,N0,Uk_last_init,w_move_val) 

        # equality constraints for collocation, inequality constraint for total volume 
        g = vertcat(*g_eq, *g_ineq) 

        # create MPC problem
        nlp_prob = {'f': obj,'x': OPT_variables,'g': g,'p':params} 

        #  calculate jacobian + hessian
        w = nlp_prob['x'] 
        J = nlp_prob['f'] 
        g = nlp_prob['g'] 
        lam_f = SX.sym("lam_f") 
        lam_g = SX.sym("lam_g", g.sparsity()) 

        # gradient of f
        grad_f = Function("grad_f",[w, params],[J, gradient(J,w)]) 

        # jacobian
        jac_g = Function("jac_g",[w, params],[g, jacobian(g,w)]) 

        # calculate hessian
        L = lam_f*J + dot(lam_g, g) 
        H, _ = hessian(L,w) 
        Htriu = triu(H) 
        hess_lag = Function("hess_lag", [w, params, lam_f, lam_g], [Htriu]) 

        # solver options
        opts = {
        "ipopt.tol": 1e-5,
        "ipopt.max_iter": self.max_iter,
        "ipopt.print_level": 0,
        "ipopt.acceptable_tol": 1e-5,
        "ipopt.acceptable_obj_change_tol": 1e-5,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.nlp_scaling_method": 'gradient-based',
        "ipopt.bound_relax_factor": 1e-8,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.warm_start_bound_push": 1e-6,
        "ipopt.warm_start_mult_bound_push": 1e-6,
        'grad_f': grad_f,
        'jac_g':jac_g,
        'hess_lag':hess_lag} 

        # Configure linear solver dynamically based on HSL library availability 
        if HSL_LIB_PATH:  
            opts["ipopt.linear_solver"] = "ma57"  
            opts["ipopt.hsllib"] = HSL_LIB_PATH  
        else:  
            opts["ipopt.linear_solver"] = "mumps"  

        # create solver
        solver = nlpsol('solver','ipopt',nlp_prob,opts) 

        return solver