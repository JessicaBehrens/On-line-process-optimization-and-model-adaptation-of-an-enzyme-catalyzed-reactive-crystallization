# On-line process optimization and model adaptation of an enzyme catalyzed reactive crystallization

Official code repository for the publication: 
> **"On-line process optimization and model adaptation of an enzyme catalyzed reactive crystallization"** 
> *Jessica Behrens et al.* 
 
--- 
 
## Overview 
 
This repository implements an integrated optimization, control, and parameter estimation framework combining: 
1. Nonlinear Model Predictive Control (NMPC)
2. Moving Horizon Estimation (MHE) 
3. Profile Likelihood (PL)
 
The framework is demonstrated on a continuous enzymatic synthesis and reactive crystallization process of (S)-(3 methoxyphenyl)-ethylamine (3MPEA). 
 
``` 
   ┌────────────────────────────────────────────────────────┐ 
   │                       Plant                            │ 
   └───────┬────────────────────────────────────────▲───────┘ 
           │ Measurements y(tk)                     │ Control u(tk) 
           ▼                                        │ 
   ┌───────────────┐         p_est          ┌───────┴────────┐ 
   │      MHE      ├───────────────────────►│  Adaptive MPC  │ 
   └───────┬───────┘                        └────────────────┘ 
           │ 
           ▼ 
   ┌───────────────┐ 
   │      PL       │ ──► Online Identifiability
   └───────────────┘ 
``` 
 
--- 
 
## Repository Architecture 
 
```
├── class_Process_model.py         # Nonlinear ODE dynamics, enzyme kinetics, and crystallization model 
├── class_Collocation.py           # Orthogonal collocation on finite elements (Radau polynomial roots) 
├── class_MPC.py                   # NMPC formulation (CasADi NLP, constraints, Ipopt interface) 
├── class_MHE.py                   # MHE formulation (arrival cost approximation, parameter estimation) 
├── class_PL.py                    # Profile Likelihood optimizer and identifiability analyzer 
├── Set_up_general.py              # Central simulation parameters, initial conditions, physical constants 
├── solver_config.py               # CasADi solver configuration and HSL DLL path resolution 
├── Set_up_figures.py              # Publication-ready figure generation and styling routines 
├── Main_Create_Tables.ipynb       # Jupyter notebook generating paper LaTeX tables (Tabs. 1, 2, 4, 7, 8) 
├── MPC_MHE_PL/ 
│   ├── simulation_loop.py         # Closed-loop simulation loop (Plant + MHE + PL + Adaptive MPC) 
│   ├── Main_Evaluate_results.ipynb# Post-processing, benchmark comparison, and performance metrics 
│   ├── dataset_MPC_benchmark.npz  # Full-order plant benchmark dataset 
│   ├── dataset_MPC.npz            # Standard (unadapted) MPC simulation results 
│   └── dataset_MPC_MHE_PL.npz     # Adaptive MPC (with MHE + PL) simulation results 
├── PL_calculations/               # Standalone scripts and notebooks for Profile Likelihood evaluation 
├── Compare_plant_controller_model/# Analysis comparing plant model with controller models 
└── Figures/                       # Export directory storing generated figures 
``` 
 
--- 
 
## Installation & Prerequisites 
 
### 1. Environment Setup 
Recommended Python version: **Python 3.10 to 3.12**. 
 
### 2. Install Dependencies 
```bash 
pip install casadi numpy scipy matplotlib pandas ipykernel 
``` 
 
### 3. Linear Solver Configuration (Ipopt / HSL) 
For high-performance execution of NLP and MHE problems, HSL linear solvers (such as `ma27` or `ma57`) are recommended. 
- **Windows**: Place `libhsl.dll` (Coin-HSL) in your path or configure the directory in `solver_config.py`. 
- **Linux/macOS**: Ensure `libhsl.so` or `libhsl.dylib` is on `LD_LIBRARY_PATH`. 
- If HSL is unavailable, Ipopt automatically falls back to MUMPS. 
 
--- 
 
## Quickstart & Reproducing Results 
 
### 1. Generating Paper LaTeX Tables 
Run the notebook [`Main_Create_Tables.ipynb`](Main_Create_Tables.ipynb) to print the LaTeX tables: 
- **Table 1**: Plant and controller controller model parameters ($K_{\mathrm{f}}$, $k_{G,\mathrm{eff}}$, $k_{\mathrm{d}}$). 
- **Table 2**: Initial conditions for process simulation and data generation. 
- **Table 4**: Parameter, state, and control variable bounds for MHE and MPC. 
- **Table 7 (Appendix)**: Complete list of physical, kinetic, and simulation constants. 
- **Table 8 (Appendix)**: Measurement noise standard deviation values. 
 
### 2. Evaluating Closed-Loop Performance & Exporting Figures 
Open and run [`MPC_MHE_PL/Main_Evaluate_results.ipynb`](MPC_MHE_PL/Main_Evaluate_results.ipynb): 
- Plots state trajectories, controlled inputs, and parameter estimates against the benchmark plant. 
- Evaluates computational timing metrics across MPC, MHE, and Profile Likelihood. 
- Computes percentage performance improvements (Space-Time Yield, product salt, excess reagents). 
- Generates `result_percentage_table.tex` comparing standard MPC vs. adaptive MPC. 
- Produces graphical abstract figures via `plot_graph_abstract`. 
 
### 3. Running Closed-Loop Simulations from Scratch (Optional) 
To re-simulate the benchmark, controller MPC, and adaptive MPC runs: 
```bash 
python MPC_MHE_PL/simulation_loop.py 
``` 
*(Note: Running the full Profile Likelihood multi-starts online for all controller models requires several hours depending on available CPU cores).* 
 
--- 
 
## Citation 
 
If you use this code or datasets in your research, please cite: 
 
```bibtex 
@article{Behrens2026AdaptiveMPC, 
  title     = {On-line process optimization and model adaptation of an enzyme catalyzed reactive crystallization}, 
  author    = {Behrens, Jessica; Kunde, Tom; von Langermann, Jan; Kienle, Achim}, 
  journal   = {Chemical Engineering Journal}, 
  year      = {2026}, 
  volume    = {XX}, 
  number    = {X}, 
  pages     = {XXX--XXX}, 
  doi       = {10.XXXX/XXXXX} 
} 
``` 
 