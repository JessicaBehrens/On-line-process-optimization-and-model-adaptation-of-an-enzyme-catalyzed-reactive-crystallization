import numpy as np
from casadi import *
from solver_config import HSL_LIB_PATH 

class MHE_class():
    """
    Moving Horizon Estimation (MHE) formulation using Radau orthogonal collocation.
    Performs joint state and parameter estimation in log10 space with arrival costs.
    """
    def __init__(self,Collocation,F,Model,F_int,N_hor,n_coll,x_max,X_lb,c_in,ub_para,lb_para):
      """
      Description: Initializes the Moving Horizon Estimation (MHE) setup with parameters,
                  horizons, bounds, collocation constraints, and optimization solver.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - Collocation: Collocation class helper function for constructing dynamic constraints
         - F: CasADi function representing the process ODE right-hand side (RHS)
         - Model: Process model object containing parameter definitions and state specifications
         - F_int: CasADi step integrator function
         - N_hor: MHE estimation horizon length
         - n_coll: Number of collocation points used per interval
         - x_max: State upper bound array
         - X_lb: State lower bound array
         - c_in: Feed concentrations vector [mmol/L]
         - ub_para: Upper bounds of parameter values (log scale)
         - lb_para: Lower bounds of parameter values (log scale)
      Outputs:
         - None (Initializes instance attributes)
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      #____________________________________________________________ Setup problem __________________________________________________________________________ 
      # process model
      self.p = Model 

      # step integrator
      self.F_int = F_int 

      # estimation horizon length
      self.N_hor = N_hor 

      # parameter number of collocation points
      self.n_coll = n_coll 

      # state upper bounds
      self.x_max = x_max 

      # state lower bounds
      self.X_lb = X_lb 

      # feed concentrations
      self.c_in = c_in # [mmol/L] 

      # parameter upper bounds (log scale)
      self.ub_para = ub_para 

      # parameter lower bounds (log scale)
      self.lb_para = lb_para 

      # collocation function
      self.Coll =  Collocation(RHS = F, model = self.p,n_s = self.p.n_s) 

      # set up solver
      self.solver = self.Problem_setup(n_coll = self.n_coll) 
    
    def constraints(self,n_coll,x_max):
      """
      Description: Formulates lower and upper bounds for states, parameters, and constraints
                  along with decision variable indices for MHE collocation.
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
      p = self.p 

      # structure for bounds
      args = {} 

      # number of equality constraints
      n_eq = (p.n_s+p.n_s*(n_coll))*self.N_hor - p.n_s*(n_coll) # for Collocation 

      # save dimensions
      args.update({"n_eq": n_eq}) 

      # total number of constraints
      n_constr = n_eq 

      # number of states - decision variable
      n_x = (self.N_hor)*p.n_s + p.n_s*(n_coll)*(self.N_hor-1) # for Collocation 
   
      # save state dimensions
      args.update({"n_x": n_x}) 

      # number of parameters to optimize
      n_para_opt = p.n_p 

      # save parameter dimensions
      args.update({"n_para_opt": n_para_opt}) 

      # equality constraints
      args.update({"lbg": np.zeros((n_constr,1))}) # lower bound 
      args.update({"ubg": np.zeros((n_constr,1))}) # upper bound

      # bounds for states
      lb_states = self.X_lb 
      ub_states = x_max[:p.n_s,None] 

      # flatten bounds
      lb_X = lb_states.flatten() 
      ub_X = ub_states.flatten() 

      # state bounds -  expand for collocation
      args.update({"lbX": np.tile(lb_X,int(n_x/p.n_s))}) # lower bounds
      args.update({"ubX": np.tile(ub_X,int(n_x/p.n_s))}) # upper bounds

      # parameter bounds (log scale)
      args.update({"lbp": self.lb_para}) # lower bound
      args.update({"ubp": self.ub_para}) # upper bounds

      # for optimization variables
      args.update({"lb": np.vstack((np.reshape(args.get("lbX"),(n_x,1)),np.reshape(args.get("lbp"),(n_para_opt,1))))}) # lower bounds
      args.update({"ub": np.vstack((np.reshape(args.get("ubX"),(n_x,1)),np.reshape(args.get("ubp"),(n_para_opt,1))))}) # upper bounds
   
      # idx of sampled states
      self.idx_x_sampled = [np.arange(p.n_s)]  # Initial range for the first sampled state

      # Define the index range to loop through
      for i in range(p.n_s + p.n_s * n_coll, n_x, p.n_s * (n_coll + 1)):
         self.idx_x_sampled.append(np.arange(i , i + p.n_s ))  # Append next state indices

      return args
   
    def calc_warm_startup(self,X0,U,para_log,N0):
      """
      Description: Integrates the process model step-by-step from initial condition X0
                  to compute a warm-start state trajectory over the estimation horizon.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - X0: Initial state vector at the start of the horizon 
         - U: Control inputs history matrix of shape 
         - para_log: Log10 of parameters vector (log scale) 
         - N0: Seed crystal number parameter 
      Outputs:
         - X_traj_init: Warm-start state trajectory matrix of shape (n_s, N_hor) 
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      # model
      p = self.p 

      # Simulate MHE trajectory for warm-start step-by-step
      X_traj_init = np.zeros((p.n_s,self.N_hor)) 
      X_traj_init[:, 0] = X0
      x_current = X0.copy() 
      for step in range(self.N_hor-1):# over whole horizon
         x_next = p.step_int(F_int=self.F_int,x0=x_current,U=U[:, step],para=np.concatenate((10**para_log.squeeze(), N0)))
         # save state
         X_traj_init[:, step+1] = x_next.squeeze() 
         # next state for iteration
         x_current = x_next.squeeze().copy() 

      return X_traj_init
   
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

    def MHE_step(self,x_est_tk,y_data,u,n_coll,para0,Pmat,R,N0):
      """
      Description: Executes a single MHE optimization step, solving the NLP collocation problem
                  and dynamically adjusting collocation order if the solver fails to converge.
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      Inputs:
         - x_est_tk: Initial state estimate vector
         - y_data: Measured output data vector 
         - u: Control input vector history 
         - n_coll: Current collocation order 
         - para0: Initial parameter guesses (log scale)
         - Pmat: Covariance matrix of parameter estimation 
         - R: Measurement noise variance vector 
         - N0: Seed crystal number parameter
      Outputs:
         - x: Estimated state trajectory matrix of shape (n_s, N_hor) 
         - Para: Optimized parameter values (log scale)
         - Obj: Objective function value at solution 
         - broken: Boolean flag indicating if solver failed to converge
      """
      #----------------------------------------------------------------------------------------------------------------------------------------------------- 

      # get model
      p = self.p 

      # values of previous interval
      para_opt = np.concatenate((x_est_tk.flatten(), para0.flatten(), y_data.flatten(), u.flatten(), R.flatten(), np.array([N0]).flatten(), Pmat.flatten())) 

      # dictionary for bounds
      args = self.constraints(n_coll = self.n_coll,x_max = self.x_max) 

      # MHE solver
      solver = self.solver 

      # count number of iterations in loop
      count_while = 0 # counter for while loop 

      # booleans
      solver_success = 0 # flag of solver 
      # collocation order change flag
      inc_n_coll = 0 # 1-number of collocation points changed 
      # convergence failure flag
      broken = 0 # 1- solver broke 
      # reset flag: collocation problem was changed and needs to be reset
      reset = 0

      while solver_success != 1:  

         # Parameter perturbation
         if count_while > 0:
            rand_para_opt = np.random.randn(*para0.shape) * para0 * 0.01
         else:
            rand_para_opt = np.zeros_like(para0)

         p_init = para0 + rand_para_opt
         p_iter = reshape(p_init, -1, 1)

         # calculate warm startup
         X_traj = self.calc_warm_startup(x_est_tk, u.reshape((self.N_hor-1, p.n_u)).T, p_init, N0)

         # decision variable - perturbate in case of multiple iterations to help solver
         if count_while > 0:
            # Do not perturb initial state variables (indices 0 to p.n_s-1)
            X0_unpert = self.perturb_state(np.zeros((args.get('n_x'), 1)), X_traj, n_coll)
            rand_states = np.zeros_like(X0_unpert)
            rand_states[p.n_s:] = np.random.randn(*X0_unpert[p.n_s:].shape) * 0.001
            X0_pert = X0_unpert + rand_states
         else:
            X0_pert = self.perturb_state(np.zeros((args.get('n_x'), 1)), X_traj, n_coll)

         # Clip perturbed states to the bounds to ensure initial feasibility
         X0_pert = np.clip(X0_pert, args.get('lbX')[:, None], args.get('ubX')[:, None])

         # concat decision variables
         x0  = np.squeeze(np.vstack((X0_pert, p_iter)))  

         # solve
         if not hasattr(self, 'lam_x') or inc_n_coll or reset or self.lam_x.shape[0] != x0.shape[0]:
             sol = solver(x0 = x0,lbx= args.get("lb"), ubx= args.get("ub"),
                         lbg= args.get("lbg"), ubg= args.get("ubg"), p=para_opt) 
         else:
             sol = solver(x0 = x0,lbx= args.get("lb"), ubx= args.get("ub"),
                         lbg= args.get("lbg"), ubg= args.get("ubg"), p=para_opt,
                         lam_x0=self.lam_x, lam_g0=self.lam_g)

         # solution
         W = sol['x'].full() 

         # flat indices for states
         flat_indices = np.concatenate(self.idx_x_sampled) 

         # optimized state trajectories
         X = W[flat_indices] 

         # optimized parameters
         Para = W[-p.n_p:] # [log scale] 

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

         # redefine MHE problem, if solver is stuck - introduce more collocation points
         if count_while == 2: 
            # order of polynomial
            n_coll = 2    
            # set booleans to true
            inc_n_coll = 1
            reset = 1
         elif count_while == 4:
            # order of polynomial
            n_coll = 5
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
            para_opt = np.concatenate((x_est_tk.flatten(), para0.flatten(), y_data.flatten(), u.flatten(), R.flatten(), np.array([N0]).flatten(), Pmat.flatten()))

            # create solver with new number of collocation points
            solver = self.Problem_setup(n_coll = n_coll)

      #----------------------------------------------------------------------------------------------------------------------------------------------------- 
      # get optimal objective
      Obj = sol['f'] # value of objective function 

      # get best solution - reshape states
      x = X.reshape((self.N_hor,p.n_s)).T 

      # reset Coll problem to n_coll = 3
      if reset == 1:

         # order of polynomial
         n_coll = 3
         solver,_ = self.Problem_setup(n_coll = n_coll)
         args = self.constraints(n_coll,x_max = self.x_max)

      return x, Para, Obj, broken
   
    def Problem_setup(self,n_coll):
      """
      Description: Formulates the CasADi MHE optimization problem using collocation constraints
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

      # get measurement equation
      Meas_Eq = p.Measurement_Eq_casadi() 

      # objective function accumulator
      obj = 0 

      # constraints list
      g = []  

      # decision variables list
      w = [] 

      # Collocation Problem
      Coll =  self.Coll.Problem_setup(n_coll = n_coll) 

      # measurements matrix over the horizon
      Y = SX.sym('Y',p.n_y,self.N_hor) 

      # prior parameter guess
      Para0 = SX.sym('Para0',p.n_p) # [log scale] 

      # parameter values to optimize
      Para = SX.sym('Para',p.n_p) # [log scale] 

      # rescaled parameter values
      Para_rescaled = 10.**Para 

      # seed crystal count
      N0 = SX.sym('N0',1) 

      # AC weighting matrix
      P_mat = SX.sym('P_mat',p.n_p,p.n_p)

      # measurement noise covariance elements
      R_vec = SX.sym('R',p.n_y) 
      # measurement noise covariance matrix inverse
      R = diag(R_vec) 

      # control inputs history
      Uk = SX.sym('Uk',p.n_u,self.N_hor-1) 

      # initial state prior estimate
      X0 = SX.sym('X0',p.n_s) 

      # initial state decision variable
      Xk = SX.sym('Xk0',p.n_s,1) 
      w.append(Xk) 

      # initial equality constraints
      g.append(X0-Xk) 

      # define problem in loop 
      k=-1
      for k in range(self.N_hor-1):

         # measurement error
         e_y = Y[:,k] - Meas_Eq(Xk)  # full measurement information 
         obj +=  e_y.T@R@e_y 

         # create new collocation state 
         X_coll = SX.sym('Xk'+ str(k)+ '_coll', p.n_s*n_coll,1) 
         w.append(X_coll) # save variable in decision variable vector 

         # append collocation problem
         g.append(Coll(X_coll,Xk,Uk[:,k],Para_rescaled,N0)) 

         # for equality condition at end of interval - collocation condition
         Xk_end = X_coll[-p.n_s:] 

         # create new state 
         Xk = SX.sym('Xk'+str(k+1),p.n_s,1) 
         w.append(Xk) # save variable in decision variable vector 

         # equality constraints
         g.append(Xk_end - Xk) 

      # final measurement error
      e_y = Y[:,k+1] - Meas_Eq(Xk) # full measurement information 
      obj +=  e_y.T@R@e_y 

      # arrival cost
      e_para = Para0 - Para # [log scale] 
      obj += e_para.T@P_mat@e_para 

      # make the decision variable one column vector
      OPT_variables = vertcat(*w,Para) 

      # Reshape measurement, control, and covariance matrices into column vectors
      Y_reshaped = reshape(Y,p.n_s*self.N_hor,1) 
      U_reshaped = reshape(Uk,p.n_u*(self.N_hor-1),1) 
      P_reshaped = reshape(P_mat.T,p.n_p*p.n_p,1)

      # Parameters to pass to problem
      params = vertcat(X0,Para0,Y_reshaped,U_reshaped,R_vec,N0,P_reshaped) 

      # equality constraints
      g = vertcat(*g) 

      # create MHE problem
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
      "ipopt.tol": 1e-6,
      "ipopt.max_iter": 150,
      "ipopt.print_level": 0,
      "ipopt.acceptable_tol": 1e-5,
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
      if HSL_LIB_PATH:  #
          opts["ipopt.linear_solver"] = "ma57" 
          opts["ipopt.hsllib"] = HSL_LIB_PATH 
      else:  #
          opts["ipopt.linear_solver"] = "mumps"  

      # create solver
      solver = nlpsol('solver','ipopt',nlp_prob,opts) 

      return solver