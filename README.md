

```markdown
# LSTM-Based State Estimation for Spacecraft Rendezvous Control

## Overview
This project implements a simulation of a spacecraft rendezvous maneuver using a **Linear-Quadratic Regulator (LQR)** for control.  
The key feature of this simulation is the use of a **Long Short-Term Memory (LSTM)** neural network as a state estimator, predicting the spacecraft's full state (position and velocity) from a sequence of partial measurements.

The system is designed to be modular:
- Train a new LSTM model or load a pre-trained one.
- Compare the LSTM estimator’s performance against a traditional **Kalman filter**.

---

## Project Structure
```

.
├── data/
│   ├── predicted\_values.npy
│   ├── sim\_state.xlsx
│   ├── true\_values.npy
│   ├── x\_train.npy
│   └── y\_train.npy
├── model/
│   └── model.pth
├── results/
│   └── (Generated plots are saved here)
├── src/
│   ├── constants.py
│   ├── control.py
│   ├── data\_generation.py
│   ├── environment.py
│   ├── lstm\_model.py
│   ├── main.py
│   └── plotting.py
└── requirements.txt

```

- **data/**: Training data (`x_train.npy`, `y_train.npy`) and comparison data (`sim_state.xlsx`, `true_values.npy`).  
- **model/**: Stores the pre-trained PyTorch LSTM model (`model.pth`).  
- **results/**: Directory for all generated plots and figures.  
- **src/**: All Python source code for the simulation.  



---

## Dependencies
All required Python packages are listed in `requirements.txt`:

```

numpy>=1.21.0
torch>=1.9.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
scipy>=1.7.0
pandas>=1.3.0
openpyxl>=3.0.0

````

---

## Setup
1. Clone the repository (or place the project in your desired location).  
2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
````

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

The main entry point is `src/main.py`.

### Execution Command

Run the simulation from the root directory:

```bash
python -m src.main
```

This ensures relative imports work correctly.

---

## Configuration

Modify the boolean flags at the top of `src/main.py`:

* **LOAD\_EXISTING\_DATA**

  * `True`: Load `x_train.npy` and `y_train.npy` from `data/`.
  * `False`: Generate new training data.

* **LOAD\_EXISTING\_MODEL**

  * `True`: Load the pre-trained `model.pth` from `model/`.
  * `False`: Train a new LSTM model and save it to `model/model.pth`.

---

## File Descriptions

* **main.py**: Orchestrates data loading, model training/loading, simulation, and plotting.
* **lstm\_model.py**: Defines the PyTorch `LSTMModel` and wrapper class for training, loading, and prediction.
* **control.py**: Implements the `Control` class with the LQR solver and main simulation loop.
* **environment.py**: Defines `SpacecraftRendezvousEnv`, simulating spacecraft dynamics.
* **plotting.py**: Functions for generating plots (state comparisons, 2D/3D trajectories, Kalman filter comparison).
* **data\_generation.py**: Functions to generate synthetic training and test data.
* **utils.py**: Helper functions for loading and saving data.
* **constants.py**: Stores constant matrices (A, B) for spacecraft dynamics.

```
---

## Data Availability
The training data (`x_train.npy`, `y_train.npy`) is too large to be hosted on GitHub.  
You can download it from "https://drive.google.com/drive/folders/1De8sjXW8I1D_3Cj6rGE5yJR3Ln38j0TK?usp=sharing" and place it inside the `data/` folder.

```