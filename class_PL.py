import numpy as np
from casadi import *
from solver_config import HSL_LIB_PATH  

class PL_class():
   """
   Profile Likelihood (PL) formulation using Radau orthogonal collocation. 
   Evaluates parameter identifiability by profiling targeted parameters while optimizing the rest.
   """
   def __init__(self,Collocation,F,Model,N_hor,F_int,n_coll,x_max,X_lb,c_in, idx_not_opt,ub_para,lb_para):
      """
      Description: Initializes the parameter estimation / profile likelihood setup with decision
                  variables, horizon, bounds, and collocation solver configurations.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - Collocation: Collocation class helper function for constructing dynamic constraints
         - F: CasADi function representing the process ODE right-hand side (RHS)
         - Model: Process model object containing parameter definitions and state specifications
         - N_hor:  estimation horizon length [-]
         - F_int: CasADi step integrator function
         - n_coll: Number of collocation points used per interval [-]
         - x_max: State upper bound array [various units]
         - X_lb: State lower bound array [various units]
         - c_in: Feed concentrations vector [mmol/L]
         - idx_not_opt: Indices of parameters that are held constant [-]
         - ub_para: Upper bounds of parameter values (log scale) [various units]
         - lb_para: Lower bounds of parameter values (log scale) [various units]
      Outputs:
         - None (Initializes instance attributes)
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      #____________________________________________________________ Setup problem __________________________________________________________________________ 
      # process model
      self.p = Model

      # estimation horizon
      self.N_hor = N_hor

      # step integrator
      self.F_int = F_int

      # parameter number of collocation points
      self.n_coll = n_coll

      # state upper bounds
      self.x_max = x_max # [various units] 

      # state lower bounds
      self.X_lb = X_lb # [various units] 

      # feed concentrations
      self.c_in = c_in # [mmol/L] 

      # index of not optimized parameter
      self.idx_not_opt = idx_not_opt

      # parameter upper bounds (log scale)
      self.ub_para = ub_para # [various units] 

      # parameter lower bounds (log scale)
      self.lb_para = lb_para # [various units] 

      # collocation function
      self.Coll =  Collocation(RHS = F, model = self.p,n_s = self.p.n_s)

      # index of not optimized parameter
      idx_all = np.arange(self.p.n_p)

      # remove not optimized indices
      self.idx_opt = np.delete(idx_all, idx_not_opt)

      # set up solver
      self.solver = self.Problem_setup(n_coll = self.n_coll) # [-] 
   
   def constraints(self,n_coll,x_max):
      """
      Description: Calculates the lower and upper bounds of states, parameters, and constraints
                  along with the decision variable indices for PL collocation.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - n_coll: Number of collocation points used per interval [-]
         - x_max: State upper bound array [various units]
      Outputs:
         - args: Dictionary containing dimensions (n_eq, n_x, n_para_opt) and bounds (lb, ub, lbX, ubX, lbp, ubp, lbg, ubg)
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      #____________________________________________________________ bounds _________________________________________________________________________________ 
      # model
      p = self.p # [-] 

      # structure for bounds
      args = {} # [-] 

      # number of equality constraints
      n_eq = (p.n_s+p.n_s*(n_coll))*self.N_hor - p.n_s*(n_coll) # for Collocation 

      # save dimensions
      args.update({"n_eq": n_eq}) # [-] 

      # total number of constraints
      n_constr = n_eq # [-] 

      # Number of state decision variables
      n_x = (self.N_hor)*p.n_s + p.n_s*(n_coll)*(self.N_hor-1) # for Collocation 
   
      # save state dimensions
      args.update({"n_x": n_x}) # [-] 

      # reduced parameter optimization size
      n_para_opt_red = p.n_p - len(self.idx_not_opt) # [-] 
      # save parameter dimensions
      args.update({"n_para_opt": n_para_opt_red}) # [-] 

      # equality and inequality constraints
      args.update({"lbg": np.zeros((n_constr,1))}) # Equality constraints 
      args.update({"ubg": np.zeros((n_constr,1))}) # Equality constraints 

      # for states
      lb_states = self.X_lb # [various units] 
      ub_states = x_max[:p.n_s,None] # [various units] 

      # flatten bounds
      lb_X = lb_states.flatten() # [various units] 
      ub_X = ub_states.flatten() # [various units] 

      # state lower bounds
      args.update({"lbX": np.tile(lb_X,int(n_x/p.n_s))}) # state x lower bound 
      # state upper bounds
      args.update({"ubX": np.tile(ub_X,int(n_x/p.n_s))}) # state x upper bound 

      # for parameter
      lb_all_p = self.lb_para # [various units] 
      ub_all_p =  self.ub_para # [various units] 

      # Reduce bounds by removing fixed non-optimized parameters
      lb_red_p = np.delete(lb_all_p,self.idx_not_opt) # [various units] 
      ub_red_p = np.delete(ub_all_p,self.idx_not_opt) # [various units] 
                           
      # parameter lower bounds (log scale)
      args.update({"lbp": lb_red_p})  # parameter lower bound (log scale) 
      # parameter upper bounds (log scale)
      args.update({"ubp": ub_red_p})  # parameter upper bound (log scale) 

      # for optimization variables
      args.update({"lb": np.vstack((np.reshape(args.get("lbX"),(n_x,1)),np.reshape(args.get("lbp"),(n_para_opt_red,1))))}) # [-] 
      # for optimization variables
      args.update({"ub": np.vstack((np.reshape(args.get("ubX"),(n_x,1)),np.reshape(args.get("ubp"),(n_para_opt_red,1))))})  # [-] 
   
      # idx of sampled states
      self.idx_x_sampled = [np.arange(p.n_s)]  # Initial range for the first sampled state 

      # Define the index range to loop through
      for i in range(p.n_s + p.n_s * n_coll, n_x, p.n_s * (n_coll + 1)):
         self.idx_x_sampled.append(np.arange(i , i + p.n_s ))  # Append next state indices 

      return args

   def calc_warm_startup(self,X0,U,para_log,N0):
      """
      Description: Integrates the process model step-by-step from initial condition X0
                  to compute a warm-start trajectory over the estimation horizon.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - X0: Initial state vector at the start of the horizon [various units]
         - U: Control inputs history matrix of shape (n_u, N_hor-1) [-]
         - para_log: Log10 of parameters vector (log scale) [various units]
         - N0: Seed crystal number parameter [-]
      Outputs:
         - X_traj_init: Warm-start state trajectory matrix of shape (n_s, N_hor) [various units]
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      # model
      p = self.p # [-] 

      # Storage for warm-start trajectory
      X_traj_init = np.zeros((p.n_s,self.N_hor))

      # store initial condition
      X_traj_init[:, 0] = X0.copy()
      x_current = X0.copy()

      # get parameter
      para = np.concatenate((np.array([10**para_log]).flatten(), N0))

      # integrate step by step
      for step in range(self.N_hor-1):
         x_next = p.step_int(F_int=self.F_int,x0=x_current,U=U[:, step],para=para)

         # save state
         X_traj_init[:, step+1] = x_next.squeeze()
         x_current = x_next.squeeze().copy()

      return X_traj_init

   def pertubate_state(self,rand_states,X_traj,n_coll):
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
      # model
      p = self.p

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

   def PL_step(self,x_est_tk,y_data,u,n_coll,para0,R,N0,para_fix):
      """
      Description: Performs a Profile Likelihood step, attempting to solve the NLP collocation problem
                  and dynamically adjusting collocation order if the solver fails to converge.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - x_est_tk: Initial state estimate vector [various units]
         - y_data: Measured output data vector [various units]
         - u: Control input vector history [-]
         - n_coll: Current collocation order [-]
         - para0: Initial parameter guesses (log scale) [various units]
         - R: Measurement noise variance vector [various units]
         - N0: Seed crystal number parameter [-]
         - para_fix: Fixed parameter values not optimized (log scale) [various units]
      Outputs:
         - Para: Optimized parameter values (log scale) [various units]
         - Obj: Objective function value at solution [-]
         - broken: Boolean flag indicating if solver failed to converge
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      #_______________________________________________________________  step _______________________________________________________________________________ 

      # get model
      p = self.p

      # values of previous interval
      para_opt = np.hstack((x_est_tk.T,y_data,u,R,N0,para_fix)) #

      # dictionary for bounds
      args = self.constraints(n_coll = self.n_coll,x_max = self.x_max) #

      # get solver
      solver  = self.solver

      # count number of iterations in loop
      count_while = 0

      # booleans
      solver_success = 0 # flag of solver
      inc_n_coll = 0 # number of collocation points changed 
      broken = 0 # convergence failure flag 
      reset = 0 # reset flag: collocation problem was changed and needs to be reset

      while solver_success != 1:  # while solver did not converge

         # Parameter perturbation
         if count_while > 0:
            rand_para_opt = np.random.randn(*para0.shape) * para0 * 0.01
         else:
            rand_para_opt = np.zeros_like(para0)

         p_init = para0 + rand_para_opt # perturb initial condition 
         # parameter iteration guess
         p_iter = p_init # [various units] 

         # create whole parameter vector
         para_log = np.zeros(p.n_p) # [various units] 
         para_log[self.idx_opt] = p_init.squeeze() # [various units] 
         para_log[self.idx_not_opt] =  para_fix # [various units] 

         # calculate warm startup
         X_traj = self.calc_warm_startup(x_est_tk,u.reshape((self.N_hor-1,p.n_u)).T,para_log,N0) # [various units] 

         # decision variable - perturbate in case of multiple iterations to help solver
         if count_while > 0:
            # Do not perturb initial state variables (indices 0 to p.n_s-1)
            X0_unpert = self.pertubate_state(np.zeros((args.get('n_x'), 1)), X_traj, n_coll)
            rand_states = np.zeros_like(X0_unpert)
            rand_states[p.n_s:] = np.random.randn(*X0_unpert[p.n_s:].shape) * 0.001
            X0_pert = X0_unpert + rand_states
         else:
            X0_pert = self.pertubate_state(np.zeros((args.get('n_x'), 1)), X_traj, n_coll)

         # Clip perturbed states to the bounds to ensure initial feasibility
         X0_pert = np.clip(X0_pert, args.get('lbX')[:, None], args.get('ubX')[:, None])

         # concat decision variables
         x0  = np.squeeze(np.vstack((X0_pert,p_iter))) # perturbed initial condition 

         # solve optimization problem
         if not hasattr(self, 'lam_x') or inc_n_coll or reset or self.lam_x.shape[0] != x0.shape[0]:
             sol = solver(x0 = x0, p=para_opt,ubx= args.get("ub"),lbx= args.get("lb"),
                               lbg= args.get("lbg"), ubg= args.get("ubg"))
         else:
             sol = solver(x0 = x0, p=para_opt,ubx= args.get("ub"),lbx= args.get("lb"),
                               lbg= args.get("lbg"), ubg= args.get("ubg"),
                               lam_x0=self.lam_x, lam_g0=self.lam_g)

         # solution - get optimal parameter
         W = sol['x'].full() # states
         Para = W[-(p.n_p-len(self.idx_not_opt)):] # parameter

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

         # Redefine PL problem if solver is stuck - introduce more collocation points
         if count_while == 2: 
            # Order of polynomial
            n_coll = 2    
            # set booleans to true
            inc_n_coll = 1
            reset = 1
         elif count_while == 4:
            # Order of polynomial
            n_coll = 5
            # set booleans to true
            inc_n_coll = 1

         elif count_while ==6 and solver_success!=1:
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
            para_opt = np.hstack((x_est_tk.T,y_data,u,R,N0,para_fix))

            # create solver with new number of collocation points
            solver = self.Problem_setup(n_coll = n_coll)

      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      # get best solution
      Obj = sol['f'] # value of objective function 

      # reset Coll problem to n_coll = 3
      if reset == 1:

         # Order of polynomial
         n_coll = 3
         solver = self.Problem_setup(n_coll = n_coll)
         args = self.constraints(n_coll,x_max = self.x_max)

      return Para, Obj, broken
   
   def Problem_setup(self,n_coll):
      """
      Description: Formulates the CasADi PL optimization problem using collocation constraints
                  and configures the IPOPT solver with custom Jacobian and Hessian evaluations.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - n_coll: Number of collocation points used per interval [-]
      Outputs:
         - solver: CasADi NLP solver function
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      #____________________________________________________________ Problem setup __________________________________________________________________________ 
      # model 
      p = self.p 

      # get measurement Equation
      Meas_Eq = p.Measurement_Eq_casadi()

      # objective function accumulator
      obj = 0
      # constraints vector
      g = [] 
      # decision variables vector
      w = []

      # Collocation Problem
      Coll =  self.Coll.Problem_setup(n_coll = n_coll)

      # Initial measurement tensor
      Y = SX.sym('Y',p.n_y,self.N_hor) # [various units]

      # parameter value - fixed
      Para_fix = SX.sym('Para',len(self.idx_not_opt)) # [log scale] 
      # fixed parameter value rescaled
      Para_fix_rescaled = 10.**Para_fix

      # parameter value - optimized
      Para_opt = SX.sym('Para',len(self.idx_opt)) # [log scale] 

      # optimized parameter value rescaled
      Para_opt_rescaled = 10.**Para_opt 

      # seed crystal count
      N0 = SX.sym('N0',1) # [-] 

      # create whole parameter vector
      Para_rescaled = SX.zeros(p.n_p) 
      Para_rescaled[self.idx_opt] = Para_opt_rescaled
      Para_rescaled[self.idx_not_opt] = Para_fix_rescaled

      # measurement covariance matrix
      R_vec = SX.sym('R',p.n_y) # n_y = n_s, full measurements 

      # Inverse of measurement noise variance matrix 
      R = diag(R_vec)

      # control inputs
      Uk = SX.sym('Uk',p.n_u,self.N_hor-1) # [-] 

      # Initial conditions
      X0 = SX.sym('X0',p.n_s) # [various units] 

      # Initial conditions - decision variable
      Xk = SX.sym('Xk0',p.n_s,1) # [various units] 
      w.append(Xk) # save variable in decision variable vector 

      # initial equality constraints
      g.append(X0-Xk)

      # define problem in loop 
      k =-1 # in case the loop is not used
      for k in range(self.N_hor-1):

         # measurement error
         e_y = Y[:,k] - Meas_Eq(Xk)  # full measurement information 
         obj +=  e_y.T@R@e_y # [-] 

         # create new collocation state 
         X_coll = SX.sym('Xk'+ str(k)+ '_coll', p.n_s*n_coll,1) # [various units] 
         w.append(X_coll) # save variable in decision variable vector 

         # append collocation problem
         g.append(Coll(X_coll,Xk,Uk[:,k],Para_rescaled,N0)) # [various units] 

         # for equality condition at end of interval - collocation condition
         Xk_end = X_coll[-p.n_s:] # [various units] 

         # create new state 
         Xk = SX.sym('Xk'+str(k+1),p.n_s,1) # [various units] 
         w.append(Xk) # save variable in decision variable vector 

         # equality constraints
         g.append(Xk_end - Xk) 

      # final measurement error
      e_y = Y[:,k+1] - Meas_Eq(Xk) # full measurement information 
      obj +=  e_y.T@R@e_y # [-] 

      # make the decision variable one column vector
      OPT_variables = vertcat(*w,Para_opt)

      # Reshape measurement and control matrices into column vectors
      Y_reshaped = reshape(Y,p.n_s*self.N_hor,1)
      U_reshaped = reshape(Uk,p.n_u*(self.N_hor-1),1)

      # Parameters to pass to problem
      params = vertcat(X0,Y_reshaped,U_reshaped,R_vec,N0,Para_fix)

      # equality constraints
      g = vertcat(*g)

      # Create PL NLP problem 
      nlp_prob = {'f': obj,'x': OPT_variables,'g': g,'p':params} 

      # Calculate gradient, Jacobian, and Hessian for solver
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
      "ipopt.tol": 1e-6,
      "ipopt.max_iter": 300,
      "ipopt.print_level": 0,
      "ipopt.acceptable_tol": 1e-4,
      "ipopt.acceptable_iter": 5,
      "ipopt.acceptable_obj_change_tol": 1e-5,
      "ipopt.constr_viol_tol": 1e-8,
      "ipopt.nlp_scaling_method": 'gradient-based',
      "ipopt.warm_start_init_point": "yes",
      "ipopt.warm_start_bound_push": 1e-9,
      "ipopt.warm_start_mult_bound_push": 1e-9,
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