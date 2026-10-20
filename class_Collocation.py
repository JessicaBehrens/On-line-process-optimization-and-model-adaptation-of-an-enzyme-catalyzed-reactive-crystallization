## Collocation discretization &!
from casadi import *

class Collocation():
    """
    Orthogonal collocation on finite elements using Radau quadrature.
    Transcribes continuous ODEs dx/dt = f(x,u,p) into algebraic equality
    constraints for NLP optimization in MPC, MHE, and Profile Likelihood.
    """
    def __init__(self,RHS,model,n_s):
        """
        Description: Initializes the Collocation helper class with the system equations RHS,
                     the ODE model, and the state dimension.
        #------------------------------------------------------------------------------------------ 
        Inputs:
            - RHS: CasADi Function representing the Ordinary Differential Equation right-hand side
            - model: Process model object containing parameter configurations and state definitions
            - n_s: State vector dimension [-]
        Outputs:
            - None (Initializes instance attributes)
        """
        #------------------------------------------------------------------------------------------ 
        # Right-hand side ODE function: dx/dt = f(x, u, p)
        self.F = RHS 

        # Process model object containing parameters and dimensions
        self.p = model 

        # Number of differential states (dimension of x)
        self.n_s = n_s 

    def Problem_setup(self,n_coll):
        """
        Description: Formulates the Radau collocation equations, computing the Lagrange polynomial
                     coefficients and setting up the CasADi function for dynamic collocation constraints.
        #------------------------------------------------------------------------------------------ 
        Inputs:
            - n_coll: Number of collocation points per interval [-]
        Outputs:
            - Coll: CasADi Function returning collocation constraints given state trajectory and inputs
        """
        #------------------------------------------------------------------------------------------ 
        # Number of differential states
        n_s = self.n_s 

        # Continuous normalized time variable tau in [0, 1]
        t = SX.sym('t') 

        # Interior and terminal Radau collocation points
        tau = collocation_points(n_coll, "radau") 

        # Collocation points including element starting point (tau_0 = 0)
        tau_root = [0] + tau 

        # Total collocation points per finite element (K = n_coll + 1)
        n_K = len(tau_root) 

        # State vector at the start of the finite element
        X_coll0 = SX.sym('X_coll0',n_s) 

        # Model parameter vector
        Para = SX.sym('Para',self.p.n_p)

        # Seed crystal count parameter
        N0 = SX.sym('N0',1) 

        # Combined parameter vector including seed crystal count
        Para_N0 = vertcat(Para,N0) 

        # Control input vector
        Uk = SX.sym('Uk',self.p.n_u) 

        # Lagrange polynomial differentiation matrix
        C = np.zeros((n_K, n_K)) 

        # Compute Lagrange basis polynomials and evaluate derivatives at collocation nodes
        for i in range(n_K):
            # Initialize Lagrange polynomial basis function
            P = 1 
            for j in range(n_K):
                if j != i:
                    P *= (t - tau_root[j])/(tau_root[i] - tau_root[j]) # Product formulation

            # Compute time derivative of Lagrange basis polynomial dL_i/dt via CasADi Jacobian
            dP = jacobian(P, t)

            # Create CasADi Function to evaluate dL_i/dt
            dP_fun = Function(f'dP_{i}', [t], [dP]) 

            for j in range(n_K):
                C[j,i] = dP_fun(tau_root[j]) # Evaluate derivative at collocation node tau_root[j]

        # Decision variables for states at interior/terminal collocation nodes
        X_opt_coll = SX.sym('X_opt_coll', n_s,n_coll) 

        # List to accumulate collocation defect / residual constraints
        g_coll = [] 

        for j in range(1, n_K):

            # State time derivative approximation: dx/dt = sum_i C[j, i] * X_i
            xp = 0 

            for i in range(n_K):

                # State at node i: initial condition if i=0, decision variable otherwise
                Xi = X_coll0 if i == 0 else X_opt_coll[:, i-1]

                # Accumulate contribution to time derivative
                xp += C[j,i] * Xi 

            # State at collocation node j-1
            X_j = X_opt_coll[:,j-1] 

            # Collocation defect equation: dx/dt - f(x_j, u_k, p) = 0
            g_j = xp - self.F(X_j, Uk, Para_N0) 
            g_coll.append(g_j) 

        # Concatenate collocation constraints into a column vector
        g_coll = vertcat(*g_coll) 

        # Vectorize collocation state decision variables (dimension: n_s * n_coll)
        X_opt_coll_vec = reshape(X_opt_coll,n_s*n_coll,1) 

        # Store collocation states decision vector
        self.in_Coll_x = X_opt_coll_vec 

        # Store collocation parameters input vector
        self.in_Coll_p = vertcat(X_coll0,Uk,Para_N0) 

        # Store collocation constraints vector
        self.g_coll = g_coll 
        
        # CasADi Function mapping [X_coll, X0, Uk, Para, N0] -> [g_coll]
        Coll = Function('Coll',[X_opt_coll_vec,X_coll0,Uk,Para,N0],[g_coll]) 

        return Coll