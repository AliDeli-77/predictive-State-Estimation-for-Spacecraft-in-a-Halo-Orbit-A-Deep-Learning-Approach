import numpy as np
import torch
from pathlib import Path
from .data_generation import generate_control_data, generate_control_data_zero_u
from .environment import SpacecraftRendezvousEnv
from .lstm_model import Model
from .control import Control
from .plotting import plot_state_comparisons, plot_trajectories, plot_comparison_with_kalman
from .utils import load_data, save_results

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
 
    LOAD_EXISTING_DATA = True
    LOAD_EXISTING_MODEL = True

    if LOAD_EXISTING_DATA:
        print("Attempting to load existing training data...")
        try:
            x_train, y_train = load_data()
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Data not found. Generating new training data instead...")
            x_train, y_train = generate_control_data(t_end=10, dt=0.00001)
            save_results(x_train, y_train)
            print("New data generated and saved.")
    else:
        print("Generating new training data...")
        x_train, y_train = generate_control_data(t_end=10, dt=0.00001)
        save_results(x_train, y_train)
        print("New data generated and saved.")

    # Initialize environment and model
    dt = 0.0001
    env = SpacecraftRendezvousEnv(dt)
    model = Model(dt=dt)

    # Generate test data
    test_inputs, test_outputs = generate_control_data_zero_u(t_end=1.0, dt=dt)
    sigma = np.sqrt(1.7e-8)
    test_inputs[:, :, 0:3] += sigma * np.random.randn(*test_inputs[:, :, 0:3].shape)

    # Preprocess data
    x_tv, y_tv = generate_control_data(t_start=0, t_end=0.1, dt=dt)
    model.pre_process(x_train, y_train, x_tv, y_tv)

    # Train or load model based on the boolean flag
    if LOAD_EXISTING_MODEL:
        print("Attempting to load existing model...")
        try:
            model.load_model()
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model not found. Training a new model instead...")
            model.train()
    else:
        print("Training a new model...")
        model.train()

    # Test predictions
    x_test_tensor = torch.tensor(test_inputs, dtype=torch.float32).to(model.device)
    y_pred = model.predict(x_test_tensor).cpu().numpy()
    y_true = test_outputs[:, 0, :]

    # Plot results
    plot_state_comparisons(y_true, y_pred, save_dir=RESULTS_DIR)

    # Run control simulation
    control_method = Control(env, model)
    Q = 1 * np.array([
        [100, 0, 0, 0, 0, 0],
        [0, 100, 0, 0, 0, 0],
        [0, 0, 100, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    R = 0.001 * np.diag([1, 1, 1])

    chaser_states, target_states, actions, y_hats = control_method.LQR(Q, R)
    plot_trajectories(chaser_states, target_states, control_method.T, save_dir=RESULTS_DIR)

    # Plot comparison with Kalman filter
    plot_comparison_with_kalman()

if __name__ == "__main__":
    main()