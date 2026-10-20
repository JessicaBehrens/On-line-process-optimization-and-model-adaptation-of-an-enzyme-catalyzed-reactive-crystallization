""" 
Post-processing, publication visualization, and figure generation routines. 
Provides state trajectories, control plots, profile likelihood heatmaps, and performance bar charts. 
""" 
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import chi2 
from matplotlib.ticker import MaxNLocator, FormatStrFormatter 

# ==============================================================================
# 1. Global Setup & Reusable Helpers
# ==============================================================================

def set_plot_style():
    """Configures global matplotlib publication styling (Computer Modern math, serif font).""" 
    matplotlib.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

set_plot_style()


def prepare_data(Data):
    """Ensures input data array is formatted as a 3D tensor (n_vars, n_steps, n_datasets).""" 
    if Data is None or (hasattr(Data, '__len__') and len(Data) == 0):
        return np.empty((0, 0, 0))
    Data = np.asarray(Data)
    if Data.ndim == 2:
        return Data[:, :, np.newaxis]
    elif Data.ndim == 1:
        return Data[:, np.newaxis, np.newaxis]
    return Data


def get_nom_color(j, n_ds):
    """Returns a distinct or viridis color mapped to dataset index j out of n_ds sets.""" 
    if n_ds == 1:
        return 'k'
    distinct_colors = ['k', 'forestgreen', 'royalblue', 'darkorange', 'purple', 'darkcyan', 'olive']
    if j < len(distinct_colors):
        return distinct_colors[j]
    return plt.cm.viridis(j / n_ds)


def format_sci(val):
    """Formats float value into clean scientific notation LaTeX string. """ 
    if val is None:
        return ""
    s = f"{val:.2e}"
    if s.split('e')[0].endswith('0'):
        return f"{val:.1e}"
    return s


def save_figure(fig, save_name, default_dir='Figures'):
    """Saves matplotlib figure to the project Figures directory. """ 
    if save_name is None:
        return
    try: 
        from solver_config import PROJECT_DIR 
        root_dir = str(PROJECT_DIR) 
    except Exception: 
        root_dir = os.getcwd() 
    target_dir = os.path.join(root_dir, default_dir)
    os.makedirs(target_dir, exist_ok=True)
    fig.savefig(os.path.join(target_dir, os.path.basename(save_name)), bbox_inches='tight', dpi=150)


def plot_reduced_line(t, data, ax, label=None, linestyle='-', color=None, **kwargs):
    """Plots 1D data against time vector t, interpolating if dimensions differ. """ 
    if len(t) != len(data):
        t = np.linspace(t[0], t[-1], len(data))
    return ax.plot(t, data, color=color, linestyle=linestyle, label=label, **kwargs)


def _create_param_labels(label_base, n_ds, p_vals=None, default_vals=None, include_params=False):
    """Generates LaTeX formatted legend labels with parameter values (Kf, kG, kd). """ 
    labels = []
    if p_vals is not None:
        p_arr = np.asarray(p_vals)
        if np.any(p_arr < 0):
            p_arr = 10**p_arr
    else:
        p_arr = None

    for j in range(max(n_ds, 1)):
        lbl_escaped = label_base.replace("_", "\\_")
        lbl_math = f"\\text{{{lbl_escaped}}}"
        if n_ds > 1:
            lbl_math += f"\\text{{ {j+1}}}"

        if include_params:
            kf, kg, kd = None, None, None
            if p_arr is not None:
                if p_arr.ndim == 2 and j < p_arr.shape[1]:
                    kf, kg = p_arr[0, j], p_arr[1, j]
                    if p_arr.shape[0] > 2:
                        kd = p_arr[2, j]
                elif p_arr.ndim == 1:
                    kf, kg = p_arr[0], p_arr[1]
                    if len(p_arr) > 2:
                        kd = p_arr[2]
            elif default_vals is not None:
                kf, kg = default_vals[0], default_vals[1]
                if len(default_vals) > 2:
                    kd = default_vals[2]

            if kf is not None and kg is not None:
                lbl_math += f", K_{{\\text{{f}}}} = {format_sci(kf)}\\text{{ s}}^{{-1}}, k_{{G,\\text{{eff}}}} = {format_sci(kg)}\\text{{mm}}\\text{{ s}}^{{-1}}"
                if kd is not None:
                    lbl_math += f", k_{{\\text{{d}}}} = {format_sci(kd)}\\text{{ s}}^{{-1}}"

        labels.append(f"${lbl_math}$")
    return labels


def _extract_broken_mask(t_len, broken_mpc=None, broken_mhe=None):
    """Combines MPC and MHE solver status masks into an array of divergence indices. """ 
    mask = np.zeros(t_len, dtype=bool)
    if broken_mpc is not None:
        mask[:len(broken_mpc)] |= np.asarray(broken_mpc, dtype=bool)
    if broken_mhe is not None:
        mask[:len(broken_mhe)] |= np.asarray(broken_mhe, dtype=bool)
    return np.where(mask)[0]


def _extract_state_trajectory(D, idx, j, p, N0=None): 
    """Extracts trajectory for state idx and dataset j, normalizing concentration by volume. """ 
    if idx == p.idx_V or idx == p.idx_3MAP_np: 
        return D[idx, :, j] 
    elif idx == p.idx_L: 
        n0_val = N0 if N0 is not None else getattr(p, 'N_0', 1.0) 
        return p.mass_2_L(D[idx, :, j], n0_val) 
    else: 
        if p.idx_V < D.shape[0]: 
            vol = D[p.idx_V, :, j] 
            return D[idx, :, j] / np.where(vol == 0, 1.0, vol) 
        return D[idx, :, j] 


def _get_dataset_color(j, n_ds, col_base): 
    """Returns color for dataset j out of n_ds, with colormap shading for multi-dataset curves. """ 
    if n_ds <= 1: 
        return col_base 
    if col_base in ('C0', 'b'): 
        cmap = plt.cm.Blues 
    elif col_base in ('r', 'C3'): 
        cmap = plt.cm.Reds 
    else: 
        cmap = plt.cm.Greens 
    return cmap(0.4 + 0.5 * j / (n_ds - 1)) 


# ==============================================================================
# 2. Primary State & Control Plotting Functions
# ==============================================================================

def plot_states_all(t, p, Data, Data2=None, label1='plant', label2='contr. model',
                    para_vals1=None, para_vals2=None, save_name=None, para_plant=None):
    """Plots full 10-state trajectory comparison between plant and controller models. """ 
    set_plot_style()
    n_rows = int(np.ceil(p.n_s / 2))
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 18))
    axs = axs.flatten()

    Data = prepare_data(Data)
    Data2 = prepare_data(Data2)
    n_ds1 = Data.shape[2] if Data.ndim == 3 else 0
    n_ds2 = Data2.shape[2] if Data2.ndim == 3 else 0

    t_data = np.linspace(t[0], t[-1], Data.shape[1]) if Data.size > 0 else t
    t_data2 = np.linspace(t[0], t[-1], Data2.shape[1]) if Data2.size > 0 else t

    label_plant_list = _create_param_labels(label1, n_ds1, para_vals1, default_vals=para_plant, include_params=True)
    label_nom_list = _create_param_labels(label2, n_ds2, para_vals2, default_vals=getattr(p, 'para', None), include_params=True)

    state_symbols = {
        p.idx_V: r'$V(t)$',
        p.idx_L: r'$L(t)$',
        p.idx_3MAP_np: r'$n^{\prime\circ}_{\text{3MAP}}(t)$',
        getattr(p, 'idx_IPA', 0): r'$c^{\prime\bullet}_{\text{IPA}}(t)$',
        getattr(p, 'idx_3MPEA', 1): r'$c^{\prime\bullet}_{\text{3MPEA}}(t)$',
        getattr(p, 'idx_ACE', 2): r'$c^{\prime\bullet}_{\text{ACE}}(t)$',
        getattr(p, 'idx_3MAP_p', 3): r'$c^{\prime\bullet}_{\text{3MAP}}(t)$',
        getattr(p, 'idx_3DPPA', 4): r'$c^{\prime\bullet}_{\text{3DPPA}}(t)$',
        getattr(p, 'idx_E', 5): r'$c^{\prime\bullet}_{\text{E}}(t)$',
        getattr(p, 'idx_IPA_3DPPA', 7): r'$c^{\prime\prime}_{\text{d}}(t)$',
        getattr(p, 'idx_3MPEA_3DPPA', 10): r'$c^{\prime\prime}_{\text{p}}(t)$',
        11: r'$c^{\prime\circ}_{\text{3MAP}}(t) + c^{\prime\bullet}_{\text{3MAP}}(t)$',
        12: r'$c^{\prime\prime}_{\text{p}}(t) + c^{\prime\bullet}_{\text{3MPEA}}(t)$'
    }

    for i, ax in enumerate(axs):
        if i >= p.n_s:
            ax.axis('off')
            continue

        # Plot Data 1 (Plant) 
        for j in range(n_ds1): 
            if i < Data.shape[0]: 
                c_plant = _get_dataset_color(j, n_ds1, 'r') 
                y = _extract_state_trajectory(Data, i, j, p) 
                ax.plot(t_data, y, color=c_plant, linestyle='-', label=label_plant_list[j]) 
                if i == p.idx_V and p.idx_3MAP_np < Data.shape[0]: 
                    ax.plot(t_data, y + Data[p.idx_3MAP_np, :, j] * p.M_3MAP / p.rho_3MAP, color=c_plant, linestyle='--') 

        # Plot Data 2 (Controller Model) 
        for j in range(n_ds2): 
            if i < Data2.shape[0]: 
                c_nom = get_nom_color(j, n_ds2) 
                y2 = _extract_state_trajectory(Data2, i, j, p) 
                ax.plot(t_data2, y2, color=c_nom, linestyle='-', label=label_nom_list[j]) 
                if i == p.idx_V and p.idx_3MAP_np < Data2.shape[0]: 
                    ax.plot(t_data2, y2 + Data2[p.idx_3MAP_np, :, j] * p.M_3MAP / p.rho_3MAP, color=c_nom, linestyle='--') 

        symbol = state_symbols.get(i, rf'$c_{{{i+1}}}(t)$')
        ax.set_ylabel(symbol, fontsize=13)
        ax.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True)
        if i == 0 and ((n_ds1 > 0 and i < Data.shape[0]) or (n_ds2 > 0 and i < Data2.shape[0])):
            ax.legend(fontsize=11)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()


def _plot_control_feed(ax_bottom, t, U_bench_data, U1_data, U2_data, i_u,
                       label_b_list, label_plant_list, label_nom_list,
                       broken_mpc1, broken_mpc2, y_label, y_lim):
    """Plots step control profile for a single input channel across benchmark, plant, and controller datasets. """ 
    ax_bottom.grid(True)
    ax_bottom.set_ylim(*y_lim)
    ax_bottom.set_ylabel(y_label, fontsize=13)
    ax_bottom.set_xlabel(r'$t \text{ (h)}$', fontsize=13)

    n_u_ds_bench = U_bench_data.shape[2] if U_bench_data.ndim == 3 else 0
    n_u_ds1 = U1_data.shape[2] if U1_data.ndim == 3 else 0
    n_u_ds2 = U2_data.shape[2] if U2_data.ndim == 3 else 0

    for j in range(n_u_ds_bench):
        u_val = U_bench_data[i_u, :, j]
        u_plot = np.append(u_val, u_val[-1]) if len(u_val) == len(t) - 1 else u_val
        lbl = label_b_list[j] if j < len(label_b_list) else label_b_list[0]
        ax_bottom.step(t, u_plot, where='post', color='k', linestyle='-', label=lbl)

    for j in range(n_u_ds1):
        c_plant = 'C0' if n_u_ds1 == 1 else plt.cm.Blues(0.4 + 0.5 * j / (n_u_ds1 - 1) if n_u_ds1 > 1 else 0.9)
        u_val = U1_data[i_u, :, j]
        u_plot = np.append(u_val, u_val[-1]) if len(u_val) == len(t) - 1 else u_val
        ax_bottom.step(t, u_plot, where='post', color=c_plant, linestyle='--', label=label_plant_list[j])

    for j in range(n_u_ds2):
        c_nom = 'g' if n_u_ds2 == 1 else plt.cm.Greens(0.4 + 0.5 * j / (n_u_ds2 - 1) if n_u_ds2 > 1 else 0.9)
        u_val = U2_data[i_u, :, j]
        u_plot = np.append(u_val, u_val[-1]) if len(u_val) == len(t) - 1 else u_val
        ax_bottom.step(t, u_plot, where='post', color=c_nom, linestyle='-.', label=label_nom_list[j])

    for broken_mpc, u_data, n_u in [(broken_mpc1, U1_data, n_u_ds1), (broken_mpc2, U2_data, n_u_ds2)]:
        if broken_mpc is not None:
            broken_indices = np.where(broken_mpc)[0]
            for j in range(n_u):
                u_val = u_data[i_u, :, j]
                t_arr = t[:-1] if len(u_val) == len(t) - 1 else t
                valid_idx = broken_indices[broken_indices < len(u_val)]
                ax_bottom.scatter(t_arr[valid_idx], u_val[valid_idx], color='r', marker='x', s=100, zorder=5)

    if i_u == 0:
        ax_bottom.legend(loc='best', fontsize=11)


def plot_red_states(N0, p, t, Data, Data2=None, U1=None, U2=None,
                    label1='plant', label2='controller model',
                    Para1=None, Para2=None, para_vals1=None, para_vals2=None,
                    broken_mhe1=None, broken_mhe2=None,
                    broken_mpc1=None, broken_mpc2=None,
                    Obj=None, Para_PL=None, Broken_PL=None,
                    Data_bench=None, U_bench=None, label_bench='Benchmark',
                    save_name=None, para_plant=None):
    """Plots reduced 4-species state trajectories (IPA, 3MAP, 3DPPA, 3MPEA), feed controls, and PL heatmaps. """ 
    set_plot_style()

    # Normalize positional swap arguments if strings/arrays were passed positionally
    if isinstance(U1, str): label1, U1 = U1, None
    if isinstance(U2, str): label2, U2 = U2, None
    if not isinstance(label1, str) and U1 is None: U1, label1 = label1, 'plant'
    if not isinstance(label2, str) and U2 is None: U2, label2 = label2, 'controller model'

    use_bottom_subplots = (U1 is not None or U2 is not None or Para1 is not None or Para2 is not None or U_bench is not None)
    n_p_to_plot = p.n_p

    if use_bottom_subplots:
        height_ratios = [1.8, 1, 1.8, 1, 1]
        fig = plt.figure(figsize=(14, 15 if n_p_to_plot <= 2 else 16))
        gs = gridspec.GridSpec(5, 2, height_ratios=height_ratios)
        ax_ipa = fig.add_subplot(gs[0, 0])
        ax_u_ipa = fig.add_subplot(gs[1, 0])
        ax_3map = fig.add_subplot(gs[0, 1])
        ax_u_3map = fig.add_subplot(gs[1, 1])
        ax_3dppa = fig.add_subplot(gs[2, 0])
        ax_3mpea = fig.add_subplot(gs[2, 1])

        gs_bottom_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[3:, 0], width_ratios=[1, 0.04], wspace=0.03)
        gs_bottom_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[3:, 1], width_ratios=[1, 0.04], wspace=0.03)
        ax_u_3dppa = fig.add_subplot(gs_bottom_left[0, :])

        ax_bottom_list = [None] * n_p_to_plot
        ax_cbar_left = None
        if n_p_to_plot == 3:
            ax_bottom_list[0] = fig.add_subplot(gs_bottom_right[0, 0])
            ax_bottom_list[1] = fig.add_subplot(gs_bottom_right[1, 0])
            ax_bottom_list[2] = fig.add_subplot(gs_bottom_left[1, 0])
            ax_cbar_left = fig.add_subplot(gs_bottom_left[1, 1])
        elif n_p_to_plot == 2:
            ax_bottom_list[0] = fig.add_subplot(gs_bottom_right[0, 0])
            ax_bottom_list[1] = fig.add_subplot(gs_bottom_right[1, 0])
        else:
            for ip in range(n_p_to_plot):
                ax_bottom_list[ip] = fig.add_subplot(gs_bottom_right[min(ip, 1), 0])
        ax_cbar = fig.add_subplot(gs_bottom_right[:, 1])
    else:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

    Data = prepare_data(Data)
    Data2 = prepare_data(Data2)
    Data_bench = prepare_data(Data_bench)
    n_ds1 = Data.shape[2] if Data.ndim == 3 else 0
    n_ds2 = Data2.shape[2] if Data2.ndim == 3 else 0
    n_ds_bench = Data_bench.shape[2] if Data_bench.ndim == 3 else 0

    broken_idx1 = _extract_broken_mask(len(t), broken_mpc1, broken_mhe1)
    broken_idx2 = _extract_broken_mask(len(t), broken_mpc2, broken_mhe2)

    label_plant_list = _create_param_labels(label1, n_ds1)
    label_nom_list = _create_param_labels(label2, n_ds2)
    label_bench_list = _create_param_labels(label_bench, n_ds_bench)
    label_b = label_bench_list[0] if len(label_bench_list) > 0 else f"${label_bench}$"

    U1_data = prepare_data(U1)
    U2_data = prepare_data(U2)
    U_bench_data = prepare_data(U_bench)

    is_conc = (Data.size > 0 and p.idx_V < Data.shape[0]) or \
              (Data2.size > 0 and p.idx_V < Data2.shape[0]) or \
              (Data_bench.size > 0 and p.idx_V < Data_bench.shape[0])

    # ---------------- 1. IPA Subplot ----------------
    ax = ax_ipa if use_bottom_subplots else axs[0]
    ax.grid(True)
    ax_vol = ax.twinx() if is_conc else None
    if ax_vol:
        ax_vol.set_ylabel(r'$V(t)$', fontsize=13)

    for D, n_ds, col_base, lbl_list, brk_idx in [
        (Data_bench, n_ds_bench, 'k', label_bench_list, []),
        (Data, n_ds1, 'C0', label_plant_list, broken_idx1),
        (Data2, n_ds2, 'g', label_nom_list, broken_idx2)
    ]:
        for j in range(n_ds): 
            if p.idx_IPA < D.shape[0]: 
                col = _get_dataset_color(j, n_ds, col_base) 
                if p.idx_V < D.shape[0]: 
                    vol = D[p.idx_V, :, j]
                    y = D[p.idx_IPA, :, j] / np.where(vol == 0, 1.0, vol)
                    if ax_vol:
                        plot_reduced_line(t, vol, ax_vol, rf'$V(t)$ - {lbl_list[j]}', '--', col)
                else:
                    y = D[p.idx_IPA, :, j]
                plot_reduced_line(t, y, ax, lbl_list[j], '-', col)
                if len(brk_idx) > 0:
                    ax.scatter(t[brk_idx], y[brk_idx], color='r', marker='x', s=100, zorder=5)

    if use_bottom_subplots and any(u.size > 0 for u in [U1_data, U2_data, U_bench_data]):
        _plot_control_feed(ax_u_ipa, t, U_bench_data, U1_data, U2_data, 0,
                           label_bench_list, label_plant_list, label_nom_list,
                           broken_mpc1, broken_mpc2, r'$u_{\text{IPA}}$', (-0.05, 1.1))

    ax.set_ylabel(r'$c^{\prime\bullet}_{\text{IPA}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{IPA}}(t)$', fontsize=13)
    ax.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
    ax.tick_params(labelsize=11)
    if ax_vol:
        ax_vol.tick_params(labelsize=11)

    color_handles = []
    if n_ds_bench > 0: color_handles.append(Line2D([0], [0], color='k', linestyle='-', label=label_b))
    if n_ds1 > 0: color_handles.append(Line2D([0], [0], color='C0', linestyle='-', label=label_plant_list[0]))
    if n_ds2 > 0: color_handles.append(Line2D([0], [0], color='g', linestyle='-', label=label_nom_list[0]))
    if color_handles:
        leg1 = ax.legend(handles=color_handles, loc='lower right', fontsize=11)
        ax.add_artist(leg1)
        ls_handles = [
            Line2D([0], [0], color='k', linestyle='-', label=r'$c^{\prime\bullet}_{\text{IPA}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{IPA}}(t)$'),
            Line2D([0], [0], color='k', linestyle='--', label=r'$V(t)$')
        ]
        ax.legend(handles=ls_handles, loc='lower center', fontsize=11)

    # ---------------- 2. 3MAP Subplot ----------------
    ax = ax_3map if use_bottom_subplots else axs[1]
    ax.grid(True)
    ax2 = ax.twinx()

    for D, n_ds, col_base, lbl_list, brk_idx in [
        (Data_bench, n_ds_bench, 'k', label_bench_list, []),
        (Data, n_ds1, 'C0', label_plant_list, broken_idx1),
        (Data2, n_ds2, 'g', label_nom_list, broken_idx2)
    ]:
        for j in range(n_ds): 
            col = _get_dataset_color(j, n_ds, col_base) 
            vol_safe = np.where(D[p.idx_V, :, j] == 0, 1.0, D[p.idx_V, :, j]) if p.idx_V < D.shape[0] else 1.0 
            if p.idx_3MAP_p < D.shape[0]:
                y_p = D[p.idx_3MAP_p, :, j] / vol_safe if is_conc else D[p.idx_3MAP_p, :, j]
                plot_reduced_line(t, y_p, ax, f'polar, {lbl_list[j]}', '-', col)
                if len(brk_idx) > 0: ax.scatter(t[brk_idx], y_p[brk_idx], color='r', marker='x', s=100, zorder=5)
            if p.idx_3MAP_np < D.shape[0]:
                y_u = D[p.idx_3MAP_np, :, j]
                plot_reduced_line(t, y_u, ax2, f'non-polar, {lbl_list[j]}', '--', col)
                if len(brk_idx) > 0: ax2.scatter(t[brk_idx], y_u[brk_idx], color='r', marker='x', s=100, zorder=5)

    if use_bottom_subplots and any(u.size > 0 for u in [U1_data, U2_data, U_bench_data]):
        _plot_control_feed(ax_u_3map, t, U_bench_data, U1_data, U2_data, 1,
                           label_bench_list, label_plant_list, label_nom_list,
                           broken_mpc1, broken_mpc2, r'$u_{\text{3MAP}}$', (-0.02, 0.3))

    ax.set_ylabel(r'$c^{\prime\bullet}_{\text{3MAP}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{3MAP}}(t)$', fontsize=13)
    ax2.set_ylabel(r'$n^{\prime\circ}_{\text{3MAP}}(t)$', fontsize=13)
    ax.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
    ax.tick_params(labelsize=11)
    ax2.tick_params(labelsize=11)
    custom_lines = [Line2D([0], [0], color='k', linestyle='-'), Line2D([0], [0], color='k', linestyle='--')]
    ax.legend(custom_lines, [r'$c^{\prime\bullet}_{\text{3MAP}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{3MAP}}(t)$', r'$n^{\prime\circ}_{\text{3MAP}}(t)$'], loc='lower center', fontsize=11)

    # ---------------- 3. 3DPPA Subplot ----------------
    ax = ax_3dppa if use_bottom_subplots else axs[2]
    ax.grid(True)
    ax_right = ax.twinx()

    for D, n_ds, col_base, lbl_list, brk_idx in [
        (Data_bench, n_ds_bench, 'k', label_bench_list, []),
        (Data, n_ds1, 'C0', label_plant_list, broken_idx1),
        (Data2, n_ds2, 'g', label_nom_list, broken_idx2)
    ]:
        for j in range(n_ds): 
            col = _get_dataset_color(j, n_ds, col_base) 
            vol_safe = np.where(D[p.idx_V, :, j] == 0, 1.0, D[p.idx_V, :, j]) if p.idx_V < D.shape[0] else 1.0 
            if p.idx_3DPPA < D.shape[0]:
                y = D[p.idx_3DPPA, :, j] / vol_safe if is_conc else D[p.idx_3DPPA, :, j]
                plot_reduced_line(t, y, ax, lbl_list[j], '-', col)
                if len(brk_idx) > 0: ax.scatter(t[brk_idx], y[brk_idx], color='r', marker='x', s=100, zorder=5)
            if p.idx_IPA_3DPPA < D.shape[0]:
                y_ipa = D[p.idx_IPA_3DPPA, :, j]
                plot_reduced_line(t, y_ipa, ax_right, f'IPA-3DPPA, {lbl_list[j]}', '--', col)
                if len(brk_idx) > 0: ax_right.scatter(t[brk_idx], y_ipa[brk_idx], color='r', marker='x', s=100, zorder=5)

    if use_bottom_subplots and any(u.size > 0 for u in [U1_data, U2_data, U_bench_data]):
        _plot_control_feed(ax_u_3dppa, t, U_bench_data, U1_data, U2_data, 2,
                           label_bench_list, label_plant_list, label_nom_list,
                           broken_mpc1, broken_mpc2, r'$u_{\text{3DPPA}}$', (0.0, 1.1))

    ax.set_ylabel(r'$c^{\prime\bullet}_{\text{3DPPA}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{3DPPA}}(t)$', fontsize=13)
    ax_right.set_ylabel(r'$n^{\prime\prime}_{\text{d}}(t)$', fontsize=13)
    ax.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
    ax.tick_params(labelsize=11)
    ax_right.tick_params(labelsize=11)
    custom_lines = [Line2D([0], [0], color='k', linestyle='-'), Line2D([0], [0], color='k', linestyle='--')]
    ax.legend(custom_lines, [r'$c^{\prime\bullet}_{\text{3DPPA}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{3DPPA}}(t)$', r'$n^{\prime\prime}_{\text{d}}(t)$'], loc='best', fontsize=11)

    # ---------------- 4. 3MPEA Subplot & Parameters ----------------
    ax = ax_3mpea if use_bottom_subplots else axs[3]
    ax.grid(True)
    ax_right = ax.twinx()

    for D, n_ds, col_base, lbl_list, brk_idx in [
        (Data_bench, n_ds_bench, 'k', label_bench_list, []),
        (Data, n_ds1, 'C0', label_plant_list, broken_idx1),
        (Data2, n_ds2, 'g', label_nom_list, broken_idx2)
    ]:
        for j in range(n_ds): 
            col = _get_dataset_color(j, n_ds, col_base) 
            vol_safe = np.where(D[p.idx_V, :, j] == 0, 1.0, D[p.idx_V, :, j]) if p.idx_V < D.shape[0] else 1.0 
            if p.idx_3MPEA < D.shape[0]:
                y_3mpea = D[p.idx_3MPEA, :, j] / vol_safe if is_conc else D[p.idx_3MPEA, :, j]
                plot_reduced_line(t, y_3mpea, ax, f'3MPEA, {lbl_list[j]}', '-', col)
                if len(brk_idx) > 0: ax.scatter(t[brk_idx], y_3mpea[brk_idx], color='r', marker='x', s=100, zorder=5)
                if p.idx_L < D.shape[0]:
                    n_p = p.L_2_mass(D[p.idx_L, :, j], N0)
                    plot_reduced_line(t, n_p, ax_right, f'3MPEA-3DPPA, {lbl_list[j]}', '--', col)
                    plot_reduced_line(t, D[p.idx_3MPEA, :, j] + n_p, ax_right, f'3MPEA total, {lbl_list[j]}', ':', col)
                    if len(brk_idx) > 0:
                        ax_right.scatter(t[brk_idx], n_p[brk_idx], color='r', marker='x', s=100, zorder=5)
                        ax_right.scatter(t[brk_idx], (D[p.idx_3MPEA, :, j] + n_p)[brk_idx], color='r', marker='x', s=100, zorder=5)

    ax.set_ylabel(r'$c^{\prime\bullet}_{\text{3MPEA}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{3MPEA}}(t)$', fontsize=13)
    ax_right.set_ylabel(r'$n^{\prime\prime}_{\text{p}}(t)$ / $(n^{\prime\prime}_{\text{p}}(t) + n^{\prime\bullet}_{\text{3MPEA}}(t))$', fontsize=13)
    ax.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
    ax.tick_params(labelsize=11)
    ax_right.tick_params(labelsize=11)
    custom_lines = [Line2D([0], [0], color='k', linestyle='-'), Line2D([0], [0], color='k', linestyle='--'), Line2D([0], [0], color='k', linestyle=':')]
    ax.legend(custom_lines, [r'$c^{\prime\bullet}_{\text{3MPEA}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{3MPEA}}(t)$', r'$n^{\prime\prime}_{\text{p}}(t)$', r'$n^{\prime\prime}_{\text{p}}(t) + n^{\prime\bullet}_{\text{3MPEA}}(t)$'], loc='upper center', fontsize=11)

    # ---------------- 5. Bottom Parameter Plots & PL Heatmaps ----------------
    if use_bottom_subplots and (Para1 is not None or Para2 is not None or Para_PL is not None):
        names_para = [r'$K_{\mathrm{f}}$', r'$k_{G,\text{eff}}$', r'$k_{\mathrm{d}}$']
        if len(names_para) < p.n_p:
            names_para += [f'p_{ip+1}' for ip in range(len(names_para), p.n_p)]

        has_heatmap = (Obj is not None and Para_PL is not None)
        pl_para_limits = {}
        if has_heatmap:
            Obj_data = np.asarray(Obj)
            Para_PL_data = np.asarray(Para_PL)
            Obj_active = Obj_data[:, 1:, :, :] if Obj_data.ndim == 4 else Obj_data[:, 1:, :]
            Obj_log = np.log10(Obj_active + 1e-5) if np.min(Obj_active) >= 0 else Obj_active

            v_kfg = (np.min(Obj_log[0:2]), np.max(Obj_log[0:2]))
            v_kd = (np.min(Obj_log[2]), np.max(Obj_log[2])) if p.n_p == 3 else (np.min(Obj_log), np.max(Obj_log))

            for idx_p in range(p.n_p):
                p_slice = Para_PL_data[idx_p, 0, ...]
                pl_para_limits[idx_p] = (np.min(p_slice), np.max(p_slice))

            n_pl_ds = Obj_data.shape[3] if Obj_data.ndim == 4 else 1
            for ds_idx in range(n_pl_ds):
                for idx_p in range(p.n_p):
                    ax_p = ax_bottom_list[idx_p]
                    z = Obj_data[idx_p, 1:, :, ds_idx] if Obj_data.ndim == 4 else Obj_data[idx_p, 1:, :]
                    p_val = Para_PL_data[idx_p, 0, :, ds_idx] if Obj_data.ndim == 4 else Para_PL_data[idx_p, 0, :]
                    if np.min(z) >= 0: z = np.log10(z + 1e-5)
                    x_grid, y_grid = np.meshgrid(t[1:], p_val)

                    vmin, vmax = (v_kd if idx_p == 2 else v_kfg) if p.n_p == 3 else (np.min(Obj_log), np.max(Obj_log))
                    if vmax - vmin <= 1e-5: vmax = vmin + 1.0
                    mesh = ax_p.pcolormesh(x_grid, y_grid, z.T, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax, zorder=0)

                    if Broken_PL is not None:
                        b_data = np.asarray(Broken_PL)
                        b_z = b_data[idx_p, 1:, :, ds_idx] if Obj_data.ndim == 4 else b_data[idx_p, 1:, :]
                        b_mask = np.asarray(b_z, dtype=bool)
                        if np.any(b_mask):
                            ax_p.scatter(x_grid[b_mask.T], y_grid[b_mask.T], color='r', marker='x', s=20, zorder=5)

                    if ds_idx == n_pl_ds - 1:
                        if idx_p == 2 and p.n_p == 3 and ax_cbar_left:
                            cbar = fig.colorbar(mesh, cax=ax_cbar_left)
                            cbar.set_label(r'$\log_{10}(\text{PL}(\theta_i))$', rotation=270, labelpad=15)
                        elif (idx_p == 1 and p.n_p == 3) or (idx_p == p.n_p - 1 and p.n_p != 3):
                            cbar = fig.colorbar(mesh, cax=ax_cbar)
                            cbar.set_label(r'$\log_{10}(\text{PL}(\theta_i))$', rotation=270, labelpad=15)

                    ax_p.set_ylim(pl_para_limits[idx_p])
                    ax_p.set_xlim(0, t[-1])

        # Plot true plant parameters
        if para_plant is not None:
            for idx_p in range(p.n_p):
                if idx_p < len(para_plant):
                    ax_bottom_list[idx_p].axhline(y=para_plant[idx_p], color='r', linestyle='-', label=f'true {names_para[idx_p]}')

        # Plot estimated parameters
        col_est = 'w' if has_heatmap else 'k'
        for Para_in, n_ds, lbl_base, brk in [(Para1, n_ds1, label1, broken_mhe1), (Para2, n_ds2, label2, broken_mhe2)]:
            if Para_in is not None:
                P_data = prepare_data(Para_in)
                n_p_ds = P_data.shape[2]
                for j in range(n_p_ds):
                    for idx_p in range(p.n_p):
                        ax_p = ax_bottom_list[idx_p]
                        val = P_data[idx_p, :, j]
                        lbl = rf'{names_para[idx_p]} ({lbl_base} {j+1})' if n_p_ds > 1 else rf'{names_para[idx_p]} ({lbl_base})'
                        c_p = col_est if n_p_ds == 1 else plt.cm.Greens(0.4 + 0.5 * j / (n_p_ds - 1))
                        t_step = t[:-1] if len(val) == len(t) - 1 else t
                        ax_p.step(t_step, val, where='post', color=c_p, linestyle='--', label=lbl)

                        if has_heatmap and idx_p in pl_para_limits:
                            ylim_min, ylim_max = pl_para_limits[idx_p]
                            ax_p.set_ylim(ylim_min, ylim_max)
                        elif para_plant is not None and idx_p < len(para_plant):
                            p_true = para_plant[idx_p]
                            ylim_min = max(0.0, p_true - 2 * p_true)
                            ylim_max = p_true + 2 * p_true
                            ax_p.set_ylim(ylim_min, ylim_max)

                if brk is not None:
                    brk_idx = np.where(brk)[0]
                    for j in range(n_p_ds):
                        for idx_p in range(p.n_p):
                            val = P_data[idx_p, :, j]
                            t_arr = t[:-1] if len(val) == len(t) - 1 else t
                            valid_idx = brk_idx[brk_idx < len(val)]
                            ax_bottom_list[idx_p].scatter(t_arr[valid_idx], val[valid_idx], color='r', marker='x', s=100, zorder=5)

        for idx_p in range(p.n_p):
            ax_p = ax_bottom_list[idx_p]
            ax_p.set_ylabel(names_para[idx_p], fontsize=13)
            ax_p.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
            ax_p.tick_params(labelsize=11)
            ax_p.grid(True)
            if has_heatmap and idx_p in pl_para_limits:
                ax_p.set_ylim(pl_para_limits[idx_p])

        handles_legend = []
        if para_plant is not None:
            handles_legend.append(Line2D([0], [0], color='r', linestyle='-', label='true parameter value'))
        handles_legend.append(Line2D([0], [0], color=col_est, linestyle='--', label='estimated parameter value'))
        ax_bottom_list[0].legend(handles=handles_legend, loc="best", fontsize=11)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()


def plot_states_all_controls_para(N0, p, t, Data, Data2=None, U1=None, U2=None,
                                  label1='plant', label2='nominal model',
                                  Para1=None, Para2=None, para_vals1=None, para_vals2=None,
                                  broken_mhe1=None, broken_mhe2=None,
                                  broken_mpc1=None, broken_mpc2=None,
                                  Obj=None, Para_PL=None, Broken_PL=None,
                                  Data_bench=None, U_bench=None, label_bench='Benchmark',
                                  save_name=None, **kwargs):
    """Plots multi-panel overview comparing all 10 states, inputs, and estimated parameters. """ 
    set_plot_style()
    Data = prepare_data(Data)
    Data2 = prepare_data(Data2)
    Data_bench = prepare_data(Data_bench)

    n_ds1 = Data.shape[2] if Data.ndim == 3 else 0
    n_ds2 = Data2.shape[2] if Data2.ndim == 3 else 0
    n_ds_bench = Data_bench.shape[2] if Data_bench.ndim == 3 else 0

    t_data = np.linspace(t[0], t[-1], Data.shape[1]) if Data.size > 0 else t
    t_data2 = np.linspace(t[0], t[-1], Data2.shape[1]) if Data2.size > 0 else t
    t_bench = np.linspace(t[0], t[-1], Data_bench.shape[1]) if Data_bench.size > 0 else t

    label_plant_list = _create_param_labels(label1, n_ds1)
    label_nom_list = _create_param_labels(label2, n_ds2)
    label_bench_list = _create_param_labels(label_bench, n_ds_bench)

    fig, axs = plt.subplots(5, 2, figsize=(12, 13))
    axs = axs.flatten()

    state_symbols = [
        r'$c^{\prime\bullet}_{\text{IPA}}(t)$',
        r'$c^{\prime\bullet}_{\text{3MPEA}}(t)$',
        r'$c^{\prime\bullet}_{\text{ACE}}(t)$',
        r'$c^{\prime\bullet}_{\text{3MAP}}(t)$',
        r'$c^{\prime\bullet}_{\text{3DPPA}}(t)$',
        r'$c^{\prime\bullet}_{\text{E}}(t)$',
        r'$n^{\prime\circ}_{\text{3MAP}}(t)$',
        r'$n^{\prime\prime}_{\text{d}}(t)$',
        r'$V(t)$',
        r'$L(t)$'
    ]

    for i in range(10):
        ax = axs[i]
        ax.grid(True)
        ax.set_ylabel(state_symbols[i], fontsize=13)
        ax.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
        ax.tick_params(labelsize=11)

        for j in range(n_ds_bench): 
            if i < Data_bench.shape[0]: 
                plot_reduced_line(t_bench, _extract_state_trajectory(Data_bench, i, j, p, N0), ax, label_bench_list[j], '-', 'k') 

        for j in range(n_ds1): 
            if i < Data.shape[0]: 
                col = _get_dataset_color(j, n_ds1, 'C0') 
                plot_reduced_line(t_data, _extract_state_trajectory(Data, i, j, p, N0), ax, label_plant_list[j], '-', col) 

        for j in range(n_ds2): 
            if i < Data2.shape[0]: 
                col = _get_dataset_color(j, n_ds2, 'g') 
                plot_reduced_line(t_data2, _extract_state_trajectory(Data2, i, j, p, N0), ax, label_nom_list[j], '-', col) 

    custom_colors, custom_labels = [], []
    if n_ds_bench > 0: custom_colors.append('k'); custom_labels.append(label_bench)
    if n_ds1 > 0: custom_colors.append('C0'); custom_labels.append(label1)
    if n_ds2 > 0: custom_colors.append('g'); custom_labels.append(label2)
    color_handles = [Line2D([0], [0], color=c, linestyle='-') for c in custom_colors]
    axs[0].legend(color_handles, custom_labels, loc='best', fontsize=11)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()


def plot_u(p, t, U1, U2=None, label1='dataset 1', label2='dataset 2', names=None):
    """Plots control input profiles across time for one or two datasets. """ 
    set_plot_style()
    U1 = prepare_data(U1)
    U2 = prepare_data(U2)
    if U1.size == 0:
        print("Warning: U1 is empty. Nothing to plot.")
        return

    n_vars = U1.shape[0]
    n_ds1 = U1.shape[2] if U1.ndim == 3 else 0
    n_ds2 = U2.shape[2] if U2.ndim == 3 else 0

    fig, axs = plt.subplots(n_vars, 1, figsize=(8, 2.5 * n_vars), squeeze=False)
    axs = axs.flatten()

    if names is None:
        names = [r'$u_{\mathrm{IPA}}$', r'$u_{\mathrm{3MAP}}$', r'$u_{\mathrm{3DPPA}}$'] if n_vars == p.n_u else [f'u_{i+1}' for i in range(n_vars)]

    for i, ax in enumerate(axs):
        for D, n_ds, col_base, lbl in [(U1, n_ds1, 'b', label1), (U2, n_ds2, 'r', label2)]:
            for j in range(n_ds): 
                col = _get_dataset_color(j, n_ds, col_base) 
                l = f'{lbl} {j+1}' if n_ds > 1 else lbl
                u_val = D[i, :, j]
                if len(u_val) == len(t):
                    ax.plot(t, u_val, color=col, label=l)
                elif len(u_val) == len(t) - 1:
                    ax.step(t[:-1], u_val, where='post', color=col, label=l)
                else:
                    plot_reduced_line(t, u_val, ax, label=l, color=col)

        if n_ds1 > 0 or n_ds2 > 0:
            ax.legend()
        ax.set_ylabel(names[i])
        ax.set_xlabel('t in h')
        ax.grid(True)
        if n_vars == p.n_u:
            ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.show()


def plot_p(p, t, P, names=None):
    """Plots parameter estimation trajectories across time. """ 
    set_plot_style()
    P = prepare_data(P)
    if P.size == 0:
        print("Warning: P is empty. Nothing to plot.")
        return

    n_vars = P.shape[0]
    n_ds = P.shape[2] if P.ndim == 3 else 0
    if names is None:
        names = [f'p_{i+1}' for i in range(n_vars)]

    n_rows = int(np.ceil(n_vars / 2)) if n_vars > 1 else 1
    n_cols = 2 if n_vars > 1 else 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 2.5 * n_rows), squeeze=False)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i >= n_vars:
            ax.axis('off')
            continue
        for j in range(n_ds):
            col = 'k' if n_ds == 1 else plt.cm.viridis(j / (n_ds - 1) if n_ds > 1 else 0.5)
            plot_reduced_line(t, P[i, :, j], ax, label=f'dataset {j+1}' if n_ds > 1 else None, color=col)
        if n_ds > 1:
            ax.legend()
        ax.set_ylabel(names[i])
        ax.set_xlabel('t in h')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_3MPEA(t, p, N0, Data, Data2=None, label1='plant', label2='contr. model',
               para_vals1=None, para_vals2=None, save_name=None, para_plant=None):
    """Plots target 3MPEA product concentration and crystal mass trajectories. """ 
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax_right = ax.twinx()
    ax.grid(True)

    Data = prepare_data(Data)
    Data2 = prepare_data(Data2)
    n_ds1 = Data.shape[2] if Data.ndim == 3 else 0
    n_ds2 = Data2.shape[2] if Data2.ndim == 3 else 0

    label_plant_list = _create_param_labels(label1, n_ds1)
    label_nom_list = _create_param_labels(label2, n_ds2)

    is_conc = (Data.size > 0 and p.idx_V < Data.shape[0]) or (Data2.size > 0 and p.idx_V < Data2.shape[0])

    for D, n_ds, col_fn, lbl_list in [
        (Data, n_ds1, lambda j: _get_dataset_color(j, n_ds1, 'r'), label_plant_list), 
        (Data2, n_ds2, lambda j: get_nom_color(j, n_ds2), label_nom_list)
    ]:
        for j in range(n_ds):
            if p.idx_3MPEA < D.shape[0]:
                col = col_fn(j)
                vol_safe = np.where(D[p.idx_V, :, j] == 0, 1.0, D[p.idx_V, :, j]) if p.idx_V < D.shape[0] else 1.0
                y = D[p.idx_3MPEA, :, j] / vol_safe if is_conc else D[p.idx_3MPEA, :, j]
                plot_reduced_line(t, y, ax, f'3MPEA, {lbl_list[j]}', '-', col)
                if p.idx_L < D.shape[0]:
                    n_p = p.L_2_mass(D[p.idx_L, :, j], N0)
                    plot_reduced_line(t, n_p, ax_right, f'3MPEA-3DPPA, {lbl_list[j]}', '--', col)
                    plot_reduced_line(t, D[p.idx_3MPEA, :, j] + n_p, ax_right, f'3MPEA total, {lbl_list[j]}', ':', col)

    lbl_3mpea = r'$c^{\prime\bullet}_{\text{3MPEA}}(t)$' if is_conc else r'$n^{\prime\bullet}_{\text{3MPEA}}(t)$'
    ax.set_ylabel(lbl_3mpea, fontsize=13)
    ax_right.set_ylabel(r'$n^{\prime\prime}_{\text{p}}(t)$ / $(n^{\prime\prime}_{\text{p}}(t) + n^{\prime\bullet}_{\text{3MPEA}}(t))$', fontsize=13)
    ax.set_xlabel(r'$t \text{ (h)}$', fontsize=13)
    ax.tick_params(labelsize=11)
    ax_right.tick_params(labelsize=11)

    style_lines = [
        Line2D([0], [0], color='gray', linestyle='-', label=lbl_3mpea),
        Line2D([0], [0], color='gray', linestyle='--', label=r'$n^{\prime\prime}_{\text{p}}(t)$'),
        Line2D([0], [0], color='gray', linestyle=':', label=r'$n^{\prime\prime}_{\text{p}}(t) + n^{\prime\bullet}_{\text{3MPEA}}(t)$')
    ]
    color_lines = [Line2D([0], [0], color=_get_dataset_color(j, n_ds1, 'r'), linestyle='-', label=label_plant_list[j]) for j in range(n_ds1)] 
    color_lines += [Line2D([0], [0], color=get_nom_color(j, n_ds2), linestyle='-', label=label_nom_list[j]) for j in range(n_ds2)]

    ax.legend(handles=style_lines + color_lines, loc='upper center', fontsize=11)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()


def plot_profile_likelihood(p, Para, Para_opt, Obj, Obj_opt, CI, PL_broken=None, save_name=None):
    """Plots profile likelihood curves with 95% confidence intervals and MLE optimum markers. """ 
    set_plot_style()
    n_rows = p.n_p
    fig, axs = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows + 1))
    axs = [axs] if n_rows == 1 else axs.flatten()

    names_control = [r'$K_{\mathrm{f}}$', r'$k_{G,\text{eff}}$', r'$k_{\mathrm{d}}$']
    if len(names_control) < p.n_p:
        names_control += [f'p_{i+1}' for i in range(len(names_control), p.n_p)]

    Obj_CI = Obj_opt + chi2.ppf(0.95, 1)
    threshold_log = float(np.atleast_1d(np.log10(Obj_CI))[0])

    for i, ax in enumerate(axs):
        para_rescaled = 10**Para[i, :]
        obj_log = np.log10(Obj[i, :])
        opt_val = float(np.atleast_1d(10**Para_opt[i])[0])
        opt_obj_log = float(np.atleast_1d(np.log10(Obj_opt))[0])
        ci_lb = float(np.atleast_1d(10**CI[i, 0])[0])
        ci_ub = float(np.atleast_1d(10**CI[i, 1])[0])

        color_theme, color_opt, color_ci = '#1f77b4', '#d62728', '#2ca02c'

        broken_mask = np.asarray(PL_broken[i, :], dtype=bool) if PL_broken is not None else np.zeros(len(para_rescaled), dtype=bool) 
        lbl_conv = 'Converged PL' if PL_broken is not None else 'PL points' 
        ax.plot(para_rescaled[~broken_mask], obj_log[~broken_mask], color=color_theme, linestyle='-', alpha=0.5) 
        ax.scatter(para_rescaled[~broken_mask], obj_log[~broken_mask], color=color_theme, marker='o', s=40, label=lbl_conv) 
        if np.any(broken_mask): 
            ax.scatter(para_rescaled[broken_mask], obj_log[broken_mask], color=color_opt, marker='x', s=60, label='Non-converged PL') 

        ax.scatter(opt_val, opt_obj_log, color=color_opt, marker='*', s=150, zorder=5, label=f'MLE Optimum: {opt_val:.5f}')
        ax.axhline(y=threshold_log, color='black', linestyle='-', alpha=0.7, label='95% CI Threshold')
        ax.axvspan(ci_lb, ci_ub, color=color_ci, alpha=0.15, label='95% CI')
        ax.axvline(x=ci_lb, color='black', linestyle='-', alpha=0.8)
        ax.axvline(x=ci_ub, color='black', linestyle='-', alpha=0.8)

        # Inset zoom for first 2 parameters (Kf, kG_eff)
        if i < 2:
            ax_inset = ax.inset_axes([0.35, 0.45, 0.3, 0.45])
            ax_inset.plot(para_rescaled[~broken_mask], obj_log[~broken_mask], color=color_theme, linestyle='-', alpha=0.5) 
            ax_inset.scatter(para_rescaled[~broken_mask], obj_log[~broken_mask], color=color_theme, marker='o', s=30) 
            if np.any(broken_mask): 
                ax_inset.scatter(para_rescaled[broken_mask], obj_log[broken_mask], color=color_opt, marker='x', s=45) 

            ax_inset.scatter(opt_val, opt_obj_log, color=color_opt, marker='*', s=100, zorder=5)
            ax_inset.axhline(y=threshold_log, color='black', linestyle='-', alpha=0.7)
            ax_inset.axvspan(ci_lb, ci_ub, color=color_ci, alpha=0.2)
            ax_inset.axvline(x=ci_lb, color='black', linestyle='-', alpha=0.8)
            ax_inset.axvline(x=ci_ub, color='black', linestyle='-', alpha=0.8)

            x_span = ci_ub - ci_lb
            x_margin = max(x_span * 2.0, 1e-5 * opt_val)
            x_min_main, x_max_main = np.min(para_rescaled), np.max(para_rescaled)
            x_span_main = x_max_main - x_min_main
            ax_inset.set_xlim(max(opt_val - x_margin, x_min_main + 0.02 * x_span_main),
                              min(opt_val + x_margin, x_max_main - 0.02 * x_span_main))
            ax_inset.set_ylim(opt_obj_log - 0.002, threshold_log + 0.015)

            y_pos_ci = threshold_log + 0.002
            ax_inset.text(ci_lb - 0.08 * x_span, y_pos_ci, f'{ci_lb:.6f}', rotation=90, va='bottom', ha='right', fontsize=7.5, color='black')
            ax_inset.text(ci_ub + 0.08 * x_span, y_pos_ci, f'{ci_ub:.6f}', rotation=90, va='bottom', ha='left', fontsize=7.5, color='black')
            ax_inset.text(opt_val - x_margin * 0.95, threshold_log + 0.0003, f'{threshold_log:.4f}', va='bottom', ha='left', fontsize=7.5, color='black')

            ax_inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
            decimals = int(max(2, -np.log10(x_span) + 1))
            ax_inset.xaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals}f'))
            ax_inset.tick_params(labelsize=8)
            ax_inset.grid(True, which='both', linestyle=':', alpha=0.5)
            ax_inset.set_title("CI Zoom-in", fontsize=8, pad=2)
            ax.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=0.3)

        x_min_main, x_max_main = np.min(para_rescaled), np.max(para_rescaled)
        x_span_main = x_max_main - x_min_main
        ax.set_xlim(x_min_main - 0.03 * x_span_main, x_max_main + 0.03 * x_span_main)
        y_min_main, y_max_main = np.min(obj_log), np.max(obj_log)
        y_span_main = y_max_main - y_min_main
        ax.set_ylim(y_min_main - 0.05 * y_span_main, y_max_main + 0.05 * y_span_main)

        ax.set_xlabel(names_control[i], fontsize=11)
        ax.set_ylabel(r'$\log_{10}(\mathrm{PL}(\theta_i))$', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()

nice_plot_PL = plot_profile_likelihood


def compute_sty(X_real, p, N0, t, molar0_seeds):
    """Calculates space-time yield trajectory (mmol/(L*h)) from real states and reactor volume. """ 
    N0_val = N0.item() if hasattr(N0, 'item') else N0
    molar0_seeds_val = molar0_seeds.item() if hasattr(molar0_seeds, 'item') else molar0_seeds
    V_polar = X_real[p.idx_V, :]
    V_nonpolar = X_real[p.idx_3MAP_np, :] * p.M_3MAP / p.rho_3MAP
    V_crystal = X_real[p.idx_L, :]**3 * N0_val * p.shape_fac_V * 1e-6
    V_donor = X_real[p.idx_IPA_3DPPA, :] * (p.M_IPA / p.rho_IPA + p.M_3DPPA / p.rho_3DPPA)
    V_tot = V_polar + V_nonpolar + V_crystal + V_donor

    true_salt = p.L_2_mass(X_real[p.idx_L, :], N0_val) - molar0_seeds_val
    n_steps = X_real.shape[1]
    t_active = np.asarray(t[:n_steps], dtype=float).copy()
    if np.max(t_active) > 200.0:
        t_active /= 3600.0
    sty = np.zeros_like(t_active)
    sty[1:] = true_salt[1:] / (V_tot[1:] * t_active[1:])
    return sty


# ==============================================================================
# 3. Bar Plotting Shared Utilities & Performance Visualizations
# ==============================================================================

def align_subplot_zeros(left_ax, right_ax):
    """Aligns the y=0 horizontal gridlines across left and right twin axes. """ 
    ylim_l, ylim_r = left_ax.get_ylim(), right_ax.get_ylim()
    ymin_l, ymax_l = min(ylim_l), max(ylim_l)
    ymin_r, ymax_r = min(ylim_r), max(ylim_r)

    span_l = ymax_l - ymin_l if ymax_l != ymin_l else 1.0
    span_r = ymax_r - ymin_r if ymax_r != ymin_r else 1.0
    ymin_l -= 0.15 * span_l
    ymax_l += 0.10 * span_l
    ymin_r -= 0.15 * span_r
    ymax_r += 0.15 * span_r

    ymin_l = min(ymin_l, -1e-5)
    ymax_l = max(ymax_l, 0.0)
    ymin_r = min(ymin_r, 0.0)
    ymax_r = max(ymax_r, 1e-5)

    r_l = ymax_l / (-ymin_l)
    r_r = (-ymin_r) / ymax_r

    if r_l > r_r:
        ymin_r = -r_l * ymax_r
    else:
        ymax_l = r_r * (-ymin_l)

    left_ax.set_ylim(ymin_l, ymax_l)
    right_ax.set_ylim(ymax_r, ymin_r)


def annotate_axis_bars(axis, bar_groups_list):
    """Annotates bar chart patches with formatted percentage values, resolving vertical overlaps. """ 
    ylim = axis.get_ylim()
    is_inverted = ylim[0] > ylim[1]
    span = abs(ylim[1] - ylim[0])
    threshold = 0.05 * span

    annotations = []
    for item in bar_groups_list:
        if len(item) == 2:
            rects_d, rects_k = item
            for idx in range(len(rects_d)):
                rb, rk = rects_d[idx], rects_k[idx]
                d_val, k_val = rb.get_height(), rk.get_height()
                y_bottom = rk.get_y()
                x_pos = rb.get_x() + rb.get_width() / 2

                if abs(y_bottom) > 1e-5:
                    val_k = d_val + k_val
                    should_go_up_k = (val_k >= 0) if not is_inverted else (val_k < 0)
                    annotations.append({'x': x_pos, 'y': val_k, 'text': f'{k_val:+.2f}%',
                                        'offset_y': 3 if should_go_up_k else -3,
                                        'va': 'bottom' if should_go_up_k else 'top',
                                        'should_go_up': should_go_up_k})
                    should_go_up_d = not should_go_up_k
                    annotations.append({'x': x_pos, 'y': d_val if abs(d_val) >= 8.0 else 0.0, 'text': f'{d_val:+.2f}%',
                                        'offset_y': 3 if should_go_up_d else -3,
                                        'va': 'bottom' if should_go_up_d else 'top',
                                        'should_go_up': should_go_up_d})
                else:
                    for val in [d_val, k_val]:
                        if abs(val) >= 1e-5:
                            should_go_up = (val >= 0) if not is_inverted else (val < 0)
                            annotations.append({'x': x_pos, 'y': val, 'text': f'{val:+.2f}%',
                                                'offset_y': 3 if should_go_up else -3,
                                                'va': 'bottom' if should_go_up else 'top',
                                                'should_go_up': should_go_up})
        else:
            rects = item[0]
            for rb in rects:
                val = rb.get_height()
                should_go_up = (val >= 0) if not is_inverted else (val < 0)
                annotations.append({'x': rb.get_x() + rb.get_width() / 2, 'y': val, 'text': f'{val:+.2f}%',
                                    'offset_y': 3 if should_go_up else -3,
                                    'va': 'bottom' if should_go_up else 'top',
                                    'should_go_up': should_go_up})

    groups = {}
    for ann in annotations:
        g_idx = int(round(ann['x']))
        groups.setdefault(g_idx, []).append(ann)

    for group_anns in groups.values():
        group_anns.sort(key=lambda a: a['x'])
        for j in range(1, len(group_anns)):
            prev, curr = group_anns[j-1], group_anns[j]
            if abs(prev['x'] - curr['x']) < 0.05 and (prev['y'] >= 0) == (curr['y'] >= 0) and abs(prev['y'] - curr['y']) < threshold:
                curr['offset_y'] = (15 if prev['offset_y'] == 3 else (27 if prev['offset_y'] == 15 else 3)) if curr['should_go_up'] else \
                                   (-15 if prev['offset_y'] == -3 else (-27 if prev['offset_y'] == -15 else -3))

    for ann in annotations:
        axis.annotate(ann['text'], xy=(ann['x'], ann['y']), xytext=(0, ann['offset_y']),
                      textcoords="offset points", ha='center', va=ann['va'],
                      fontsize=8, fontweight='semibold', rotation=90)


def _compute_benchmark_metrics(N0_val, molar0_seeds_val, p, t, results_MPC, results_MHE_MPC_PL, results_MPC_benchmark):
    """Computes relative percentage differences in STY, salt, product, and excess species vs. benchmark. """ 
    x_bench = results_MPC_benchmark['X_real_tk'] 
    sty_bench = compute_sty(x_bench, p, N0_val, t, molar0_seeds_val)[-1] 
    salt_bench = p.L_2_mass(x_bench[p.idx_L, -1], N0_val) 
    prod_bench = salt_bench + x_bench[p.idx_3MPEA, -1] 
    exc_bench = x_bench[p.idx_IPA_3DPPA, -1] + x_bench[p.idx_3MAP_np, -1] 
    bench_donor_final = x_bench[p.idx_IPA_3DPPA, -1] 
    bench_ketone_final = x_bench[p.idx_3MAP_np, -1] 
    bench_donor_avg = np.mean(x_bench[p.idx_IPA_3DPPA, :]) 
    bench_ketone_avg = np.mean(x_bench[p.idx_3MAP_np, :]) 
    avg_exc_bench = bench_donor_avg + bench_ketone_avg 

    metrics = {
        'final_sty_mpc': [], 'final_donor_mpc': [], 'final_ketone_mpc': [],
        'final_sty_mhe': [], 'final_donor_mhe': [], 'final_ketone_mhe': [],
        'diff_sty_mpc': [], 'diff_salt_mpc': [], 'diff_prod_mpc': [], 'diff_exc_mpc': [],
        'diff_exc_donor_mpc': [], 'diff_exc_ketone_mpc': [],
        'diff_avg_exc_mpc': [], 'diff_avg_exc_donor_mpc': [], 'diff_avg_exc_ketone_mpc': [],
        'yerr_avg_exc_mpc': [],
        'diff_sty_mhe': [], 'diff_salt_mhe': [], 'diff_prod_mhe': [], 'diff_exc_mhe': [],
        'diff_exc_donor_mhe': [], 'diff_exc_ketone_mhe': [],
        'diff_avg_exc_mhe': [], 'diff_avg_exc_donor_mhe': [], 'diff_avg_exc_ketone_mhe': [],
        'yerr_avg_exc_mhe': []
    }

    n_sets = len(results_MPC)
    for i in range(n_sets):
        for res, prefix in [(results_MPC[i], 'mpc'), (results_MHE_MPC_PL[i], 'mhe')]:
            x = res['X_real_tk'] 
            sty = compute_sty(x, p, N0_val, t, molar0_seeds_val)[-1] 
            salt = p.L_2_mass(x[p.idx_L, -1], N0_val) 
            prod = salt + x[p.idx_3MPEA, -1] 
            d_final = x[p.idx_IPA_3DPPA, -1] 
            k_final = x[p.idx_3MAP_np, -1] 
            exc = d_final + k_final 
            d_avg = np.mean(x[p.idx_IPA_3DPPA, :]) 
            k_avg = np.mean(x[p.idx_3MAP_np, :]) 
            avg_exc = d_avg + k_avg 
            std_exc = np.std(x[p.idx_IPA_3DPPA, :] + x[p.idx_3MAP_np, :]) 

            metrics[f'final_sty_{prefix}'].append(float(sty))
            metrics[f'final_donor_{prefix}'].append(float(d_final))
            metrics[f'final_ketone_{prefix}'].append(float(k_final))

            metrics[f'diff_sty_{prefix}'].append((sty - sty_bench) / sty_bench * 100)
            metrics[f'diff_salt_{prefix}'].append((salt - salt_bench) / salt_bench * 100)
            metrics[f'diff_prod_{prefix}'].append((prod - prod_bench) / prod_bench * 100)
            metrics[f'diff_exc_{prefix}'].append((exc - exc_bench) / exc_bench * 100)
            # Normalized to each component's individual benchmark excess
            metrics[f'diff_exc_donor_{prefix}'].append((d_final - bench_donor_final) / bench_donor_final * 100)
            metrics[f'diff_exc_ketone_{prefix}'].append((k_final - bench_ketone_final) / bench_ketone_final * 100)
            metrics[f'diff_avg_exc_{prefix}'].append((avg_exc - avg_exc_bench) / avg_exc_bench * 100)
            metrics[f'diff_avg_exc_donor_{prefix}'].append((d_avg - bench_donor_avg) / bench_donor_avg * 100)
            metrics[f'diff_avg_exc_ketone_{prefix}'].append((k_avg - bench_ketone_avg) / bench_ketone_avg * 100)
            metrics[f'yerr_avg_exc_{prefix}'].append(std_exc / avg_exc_bench * 100)

    return metrics


def plot_bar_difference_to_benchmark(N0, p, t, molar0_seeds, results_MPC, results_MHE_MPC_PL, results_MPC_benchmark, save_name=None):
    """Plot percentage differences in STY, product salt, and final excess contaminants vs. benchmark. """ 
    set_plot_style()
    N0_val = N0.item() if hasattr(N0, 'item') else N0
    molar0_seeds_val = molar0_seeds.item() if hasattr(molar0_seeds, 'item') else molar0_seeds
    m = _compute_benchmark_metrics(N0_val, molar0_seeds_val, p, t, results_MPC, results_MHE_MPC_PL, results_MPC_benchmark)

    n_sets = len(results_MPC)
    x = np.arange(n_sets)
    width = 0.10
    labels = [f'contr. model {i+1}' for i in range(n_sets)]

    fig, ax1 = plt.subplots(figsize=(10, 8.5))
    rects1 = ax1.bar(x - 3.5 * width, m['diff_sty_mpc'], width, label='MPC STY', color='C0')
    rects2 = ax1.bar(x - 2.5 * width, m['diff_sty_mhe'], width, label='adaptive MPC STY', color='g')
    rects5 = ax1.bar(x - 1.5 * width, m['diff_salt_mpc'], width, label='MPC Product Salt', color='C0', alpha=0.6, hatch='||', edgecolor='C0')
    rects6 = ax1.bar(x - 0.5 * width, m['diff_salt_mhe'], width, label='adaptive MPC Product Salt', color='g', alpha=0.6, hatch='||', edgecolor='g')

    ax1.set_ylabel('Space-Time Yield & Product Salt Difference (%)', color='black', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1.2)

    ax1_twin = ax1.twinx()
    r3_d = [ax1_twin.bar(x[i] + 0.5 * width, m['diff_exc_donor_mpc'][i], width, color='C0', alpha=0.5, hatch='//', edgecolor='C0')[0] for i in range(n_sets)]
    r3_k = [ax1_twin.bar(x[i] + 2.5 * width, m['diff_exc_ketone_mpc'][i], width, color='C0', alpha=0.8, hatch='\\\\', edgecolor='C0')[0] for i in range(n_sets)]
    r4_d = [ax1_twin.bar(x[i] + 1.5 * width, m['diff_exc_donor_mhe'][i], width, color='g', alpha=0.5, hatch='//', edgecolor='g')[0] for i in range(n_sets)]
    r4_k = [ax1_twin.bar(x[i] + 3.5 * width, m['diff_exc_ketone_mhe'][i], width, color='g', alpha=0.8, hatch='\\\\', edgecolor='g')[0] for i in range(n_sets)]
    ax1_twin.set_ylabel('Final Excess Difference (%)', color='black', fontsize=11)

    align_subplot_zeros(ax1, ax1_twin)
    annotate_axis_bars(ax1, [(rects1,), (rects2,), (rects5,), (rects6,)])
    annotate_axis_bars(ax1_twin, [(r3_d,), (r4_d,), (r3_k,), (r4_k,)])

    patch_mpc = mpatches.Patch(color='C0', alpha=0.9, label='MPC')
    patch_adapted = mpatches.Patch(color='g', alpha=0.9, label='adaptive MPC')
    leg_col = ax1.legend(handles=[patch_mpc, patch_adapted], loc='upper center', frameon=True)
    ax1.add_artist(leg_col)

    line_bench = Line2D([0], [0], color='black', linestyle='-', linewidth=1.2, label='Benchmark')
    p_sty = mpatches.Patch(facecolor='gray', edgecolor='none', alpha=0.9, label='Space-Time Yield (STY)')
    p_salt = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.6, hatch='||', label='Product Salt')
    p_d = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.5, hatch='//', label='Excess IPA-3DPPA')
    p_k = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.8, hatch='\\\\', label='Excess 3MAP')
    ax1.legend(handles=[line_bench, p_sty, p_salt, p_d, p_k], loc='upper right', frameon=True)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()


def plot_bar_combined_performance(N0, p, t, molar0_seeds, results_MPC, results_MHE_MPC_PL, results_MPC_benchmark, save_name=None):
    """Plot 2x2 comparison grid of absolute performance and relative percentage differences vs. benchmark. """ 
    set_plot_style()
    N0_val = N0.item() if hasattr(N0, 'item') else N0
    molar0_seeds_val = molar0_seeds.item() if hasattr(molar0_seeds, 'item') else molar0_seeds
    m = _compute_benchmark_metrics(N0_val, molar0_seeds_val, p, t, results_MPC, results_MHE_MPC_PL, results_MPC_benchmark)

    n_sets = len(results_MPC)
    labels = [f'contr. model {i+1}' for i in range(n_sets)]
    x = np.arange(n_sets)
    width = 0.35
    width_diff = 0.2

    fig, axs = plt.subplots(2, 2, figsize=(18, 16.5))

    # --- 1A: Final Absolute STY ---
    ax1 = axs[0, 0]
    ax1.bar(x - width/2, m['final_sty_mpc'], width, label='MPC', color='C0')
    ax1.bar(x + width/2, m['final_sty_mhe'], width, label='adaptive MPC', color='g')
    ax1.set_ylabel('Space-Time Yield (mmol / (L * h))')
    ax1.set_title('Final Space-Time Yield (STY) Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 1B: Final Absolute Contaminants ---
    ax2 = axs[0, 1]
    r3_d = ax2.bar(x - width/2, m['final_donor_mpc'], width, label='MPC (IPA-3DPPA)', color='C0', alpha=0.5, hatch='//', edgecolor='C0')
    r3_k = ax2.bar(x - width/2, m['final_ketone_mpc'], width, bottom=m['final_donor_mpc'], label='MPC (3MAP)', color='C0', alpha=0.8, hatch='\\\\', edgecolor='C0')
    r4_d = ax2.bar(x + width/2, m['final_donor_mhe'], width, label='adaptive MPC (IPA-3DPPA)', color='g', alpha=0.5, hatch='//', edgecolor='g')
    r4_k = ax2.bar(x + width/2, m['final_ketone_mhe'], width, bottom=m['final_donor_mhe'], label='adaptive MPC (3MAP)', color='g', alpha=0.8, hatch='\\\\', edgecolor='g')
    ax2.set_ylabel('Contaminant Amount (mmol)')
    ax2.set_title('Final Contaminant Yield breakdown', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    p_solid = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.5, hatch='//', label='IPA-3DPPA')
    p_hatch = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.8, hatch='\\\\', label='3MAP')
    ax2.legend(handles=[p_solid, p_hatch], title='Species', loc='upper right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    for r_bot, r_top in [(r3_d, r3_k), (r4_d, r4_k)]:
        for rb, rt in zip(r_bot, r_top):
            tot = rb.get_height() + rt.get_height()
            ax2.annotate(f'{tot:.2f}', xy=(rb.get_x() + rb.get_width() / 2, tot), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='semibold')

    # --- 2A: Final STY & Final Excess % Diff ---
    ax3 = axs[1, 0]
    rects1_diff = ax3.bar(x - 1.5 * width_diff, m['diff_sty_mpc'], width_diff, label='MPC STY', color='C0')
    rects2_diff = ax3.bar(x - 0.5 * width_diff, m['diff_sty_mhe'], width_diff, label='adaptive MPC STY', color='g')
    ax3.set_ylabel('Space-Time Yield (STY) Difference (%)', color='black', fontsize=11)
    ax3.set_title('Final STY & Final Excess Contaminants (% Diff)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.2)

    ax3_twin = ax3.twinx()
    r3_diff_d, r3_diff_k, r4_diff_d, r4_diff_k = [], [], [], []
    for i in range(n_sets):
        d_val, k_val = m['diff_exc_donor_mpc'][i], m['diff_exc_ketone_mpc'][i]
        b_mpc = d_val if (d_val >= 0) == (k_val >= 0) else 0.0
        r3_diff_d.append(ax3_twin.bar(x[i] + 0.5 * width_diff, d_val, width_diff, color='C0', alpha=0.5, hatch='//', edgecolor='C0')[0])
        r3_diff_k.append(ax3_twin.bar(x[i] + 0.5 * width_diff, k_val, width_diff, bottom=b_mpc, color='C0', alpha=0.8, hatch='\\\\', edgecolor='C0')[0])

        d_mhe, k_mhe = m['diff_exc_donor_mhe'][i], m['diff_exc_ketone_mhe'][i]
        b_mhe = d_mhe if (d_mhe >= 0) == (k_mhe >= 0) else 0.0
        r4_diff_d.append(ax3_twin.bar(x[i] + 1.5 * width_diff, d_mhe, width_diff, color='g', alpha=0.5, hatch='//', edgecolor='g')[0])
        r4_diff_k.append(ax3_twin.bar(x[i] + 1.5 * width_diff, k_mhe, width_diff, bottom=b_mhe, color='g', alpha=0.8, hatch='\\\\', edgecolor='g')[0])
    ax3_twin.set_ylabel('Final Excess Difference (%)', color='black', fontsize=11)

    # --- 2B: Total Product & Avg Excess % Diff ---
    ax4 = axs[1, 1]
    rects5_diff = ax4.bar(x - 1.5 * width_diff, m['diff_prod_mpc'], width_diff, label='MPC Total Prod', color='C0', alpha=0.6, hatch='..', edgecolor='C0')
    rects6_diff = ax4.bar(x - 0.5 * width_diff, m['diff_prod_mhe'], width_diff, label='adaptive MPC Total Prod', color='g', alpha=0.6, hatch='..', edgecolor='g')
    ax4.set_ylabel('Total Product (3MPEA + Salt) Difference (%)', color='black', fontsize=11)
    ax4.set_title('Total Product & Avg Excess Contaminants (% Diff)', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.axhline(0, color='black', linestyle='-', linewidth=1.2)

    ax4_twin = ax4.twinx()
    r7_diff_d, r7_diff_k, r8_diff_d, r8_diff_k = [], [], [], []
    for i in range(n_sets):
        d_val, k_val = m['diff_avg_exc_donor_mpc'][i], m['diff_avg_exc_ketone_mpc'][i]
        b_mpc = d_val if (d_val >= 0) == (k_val >= 0) else 0.0
        r7_diff_d.append(ax4_twin.bar(x[i] + 0.5 * width_diff, d_val, width_diff, color='C0', alpha=0.5, hatch='//', edgecolor='C0')[0])
        r7_diff_k.append(ax4_twin.bar(x[i] + 0.5 * width_diff, k_val, width_diff, bottom=b_mpc, color='C0', alpha=0.8, hatch='\\\\', edgecolor='C0')[0])

        d_mhe, k_mhe = m['diff_avg_exc_donor_mhe'][i], m['diff_avg_exc_ketone_mhe'][i]
        b_mhe = d_mhe if (d_mhe >= 0) == (k_mhe >= 0) else 0.0
        r8_diff_d.append(ax4_twin.bar(x[i] + 1.5 * width_diff, d_mhe, width_diff, color='g', alpha=0.5, hatch='//', edgecolor='g')[0])
        r8_diff_k.append(ax4_twin.bar(x[i] + 1.5 * width_diff, k_mhe, width_diff, bottom=b_mhe, color='g', alpha=0.8, hatch='\\\\', edgecolor='g')[0])
    ax4_twin.set_ylabel('Time-Average Excess Difference (%)', color='black', fontsize=11)
    ax4_twin.errorbar(x + 0.5 * width_diff, m['diff_avg_exc_mpc'], yerr=m['yerr_avg_exc_mpc'], fmt='none', ecolor='black', capsize=4, elinewidth=1.0)
    ax4_twin.errorbar(x + 1.5 * width_diff, m['diff_avg_exc_mhe'], yerr=m['yerr_avg_exc_mhe'], fmt='none', ecolor='black', capsize=4, elinewidth=1.0)

    align_subplot_zeros(ax3, ax3_twin)
    align_subplot_zeros(ax4, ax4_twin)
    annotate_axis_bars(ax3, [(rects1_diff,), (rects2_diff,)])
    annotate_axis_bars(ax3_twin, [(r3_diff_d, r3_diff_k), (r4_diff_d, r4_diff_k)])
    annotate_axis_bars(ax4, [(rects5_diff,), (rects6_diff,)])
    annotate_axis_bars(ax4_twin, [(r7_diff_d, r7_diff_k), (r8_diff_d, r8_diff_k)])

    patch_mpc = mpatches.Patch(color='C0', alpha=0.9, label='MPC')
    patch_adapted = mpatches.Patch(color='g', alpha=0.9, label='adaptive MPC')
    leg_col = ax3.legend(handles=[patch_mpc, patch_adapted], loc='lower right', frameon=True)
    ax3.add_artist(leg_col)

    line_bench = Line2D([0], [0], color='black', linestyle='-', linewidth=1.2, label='Benchmark')
    p_sty = mpatches.Patch(facecolor='gray', edgecolor='none', alpha=0.9, label='Space-Time Yield (STY)')
    p_exc_d = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.5, hatch='//', label='Final Excess (IPA-3DPPA)')
    p_exc_k = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.8, hatch='\\\\', label='Final Excess (3MAP)')
    ax3.legend(handles=[line_bench, p_sty, p_exc_d, p_exc_k], loc='upper center', frameon=True)

    p_prod = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.6, hatch='..', label='Total Product')
    p_avg_d = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.5, hatch='//', label='Time-Avg Excess (IPA-3DPPA)')
    p_avg_k = mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.8, hatch='\\\\', label='Time-Avg Excess (3MAP)')
    err_handle = Line2D([0], [0], color='black', marker='|', linestyle='', markersize=8, label='Std. Dev. Error Bar')
    ax4.legend(handles=[line_bench, p_prod, p_avg_d, p_avg_k, err_handle], loc='upper center', frameon=True)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()