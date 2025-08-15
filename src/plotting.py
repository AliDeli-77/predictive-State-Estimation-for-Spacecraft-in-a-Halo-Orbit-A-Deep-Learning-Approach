import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

def plot_state_comparisons(true_data, pred_data, save_dir=None):
    rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.1,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    
    labels = [r'$x$', r'$y$', r'$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
    time = np.arange(len(true_data))
    
    for i, lbl in enumerate(labels):
        plt.figure(figsize=(3.5, 2.2))
        plt.plot(time, true_data[:, i], label='True', color='k')
        plt.plot(time, pred_data[:, i], label='Predicted', linestyle='--', color='C0')
        plt.xlabel('Time step')
        plt.ylabel(lbl)
        plt.legend(loc='best', framealpha=0.95)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / f'state_{i}.png', dpi=600)
            plt.close()
        else:
            plt.show()

def plot_trajectories(chaser_states, target_states, T, save_dir=None):
    rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.1,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    
    # CORRECTED: Move tensor data to CPU and convert to NumPy array before plotting
    chaser_states_cpu = chaser_states.cpu().numpy()
    target_states_cpu = target_states.cpu().numpy()
    
    time = np.linspace(0, T, len(chaser_states_cpu))
    state_tags = ['x', 'y', 'z', 'dx', 'dy', 'dz']
    ylabels = [r'$x$', r'$y$', r'$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
    
    for i, tag in enumerate(state_tags):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(time, chaser_states_cpu[:, i], label='Chaser', color='C0')
        ax.plot(time, target_states_cpu[:, i], label='Target', color='C1')
        ax.set_xlabel('Time (TU)')
        ax.set_ylabel(ylabels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)
        fig.tight_layout()
        
        if save_dir:
            fig.savefig(save_dir / f'{tag}.png', dpi=600)
            plt.close(fig)
        else:
            plt.show()
    
    # 3D trajectory plot
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(chaser_states_cpu[:, 0], chaser_states_cpu[:, 1], chaser_states_cpu[:, 2], 
            color='C0', label='Chaser')
    ax.plot(target_states_cpu[:, 0], target_states_cpu[:, 1], target_states_cpu[:, 2], 
            color='C1', label='Target')
    
    ax.scatter(*chaser_states_cpu[0, :3], color='C0', marker='o', s=20)
    ax.scatter(*chaser_states_cpu[-1, :3], color='C0', marker='^', s=40)
    ax.scatter(*target_states_cpu[0, :3], color='C1', marker='o', s=20)
    ax.scatter(*target_states_cpu[-1, :3], color='C1', marker='^', s=40)
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=22, azim=135)
    
    # Set equal axes
    x = np.concatenate([chaser_states_cpu[:, 0], target_states_cpu[:, 0]])
    y = np.concatenate([chaser_states_cpu[:, 1], target_states_cpu[:, 1]])
    z = np.concatenate([chaser_states_cpu[:, 2], target_states_cpu[:, 2]])
    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    fig.tight_layout()
    
    if save_dir:
        fig.savefig(save_dir / '3d_trajectory.png', dpi=600)
        plt.close(fig)
    else:
        plt.show()

def plot_comparison_with_kalman():
    DATA_DIR = PROJECT_ROOT / 'data' 
    excel_path = DATA_DIR / 'sim_state.xlsx'
    true_path = DATA_DIR / 'true_values.npy'
    lstm_path = DATA_DIR / 'predicted_values.npy'
    fig_dir = RESULTS_DIR  
    
    kalman_df = pd.read_excel(excel_path)  
    kalman_data = kalman_df[['x', 'y', 'z', 'vx', 'vy', 'vz']].to_numpy()
    
    true_data = np.load(true_path)[:, 0, :]  
    lstm_data = np.load(lstm_path)[:, 0, :]  
    
    N = min(true_data.shape[0], lstm_data.shape[0], kalman_data.shape[0])
    true_data = true_data[:N]
    lstm_data = lstm_data[:N]
    kalman_data = kalman_data[:N]
    time = np.arange(N)
    
    rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.1,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    ylabels = [r'$x$', r'$y$', r'$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
    
    for i, state in enumerate(state_names):
        plt.figure(figsize=(3.5, 2.2))  
        
        plt.plot(time, kalman_data[:, i], label='Kalman', color='C1', alpha=0.4, linewidth=1.1)
        plt.plot(time, lstm_data[:, i], label='LSTM', color='C0', linewidth=1.1)
        plt.plot(time, true_data[:, i], label='True', color='k', linewidth=1.1)
        
        plt.xlabel('Time step')
        plt.ylabel(ylabels[i])
        plt.legend(loc='best', framealpha=0.95)
        plt.tight_layout()
        
        plt.savefig(fig_dir / f'state_{state}.png', dpi=600)
        plt.close()
    
    # RMSE plots
    err_kalman = kalman_data - true_data  
    err_lstm = lstm_data - true_data
    
    cum_mse_k = np.cumsum(err_kalman**2, axis=0)
    cum_mse_l = np.cumsum(err_lstm**2, axis=0)
    samples = np.arange(1, N + 1)[:, None]  
    
    rmse_k = np.sqrt(cum_mse_k / samples)  
    rmse_l = np.sqrt(cum_mse_l / samples)
    
    for i, state in enumerate(state_names):
        plt.figure(figsize=(3.5, 2.2))
        
        plt.plot(time, rmse_k[:, i], label='Kalman', color='C1', linewidth=1.1)
        plt.plot(time, rmse_l[:, i], label='LSTM', color='C0', linewidth=1.1)
        
        plt.xlabel('Time step')
        plt.ylabel(f'RMSE {ylabels[i]}')
        plt.legend(loc='best', framealpha=0.95)
        plt.tight_layout()
        
        plt.savefig(fig_dir / f'rmse_{state}.png', dpi=600)
        plt.close()
