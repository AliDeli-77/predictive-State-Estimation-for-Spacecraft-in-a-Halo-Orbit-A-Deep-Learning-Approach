import numpy as np
import math
from .constants import MU, X0_CHASER, X0_TARGET, K

def generate_control_data(x0_c=X0_CHASER, x0_t=X0_TARGET, t_start=0, t_end=10, dt=0.00001):
    input_samples = []
    output_samples = []
    input_sequence = []
    
    X_t = x0_t
    X_c = x0_c
    x, y, z, dx, dy, dz = X_c
    
    for _ in np.arange(t_start, t_end, dt):
        error = X_t - X_c
        
        if np.random.rand() < 0.9:
            u = K @ error
        else:
            u = np.random.uniform(-10, 10, size=(3,))
        
        # Dynamics update
        r1 = np.array([x + MU, y, z])
        r2 = np.array([x - (1 - MU), y, z])
        nr1 = np.linalg.norm(r1)
        nr2 = np.linalg.norm(r2)
        
        ddx = 2 * dy + x - ((1 - MU) / nr1**3) * r1[0] - (MU / nr2**3) * r2[0] + u[0]
        ddy = -2 * dx + y - ((1 - MU) / nr1**3) * r1[1] - (MU / nr2**3) * r2[1] + u[1]
        ddz = -((1 - MU) / nr1**3) * r1[2] - (MU / nr2**3) * r2[2] + u[2]
        
        dx += dt * ddx
        dy += dt * ddy
        dz += dt * ddz
        
        x += dt * dx
        y += dt * dy
        z += dt * dz
        
        X_c = np.array([x, y, z, dx, dy, dz])
        
        # Target dynamics
        r1_t = np.array([X_t[0] + MU, X_t[1], X_t[2]])
        r2_t = np.array([X_t[0] - (1 - MU), X_t[1], X_t[2]])
        nr1_t = np.linalg.norm(r1_t)
        nr2_t = np.linalg.norm(r2_t)
        
        ddx_t = 2 * X_t[4] + X_t[0] - ((1 - MU) / nr1_t**3) * r1_t[0] - (MU / nr2_t**3) * r2_t[0]
        ddy_t = -2 * X_t[3] + X_t[1] - ((1 - MU) / nr1_t**3) * r1_t[1] - (MU / nr2_t**3) * r2_t[1]
        ddz_t = -((1 - MU) / nr1_t**3) * r1_t[2] - (MU / nr2_t**3) * r2_t[2]
        
        X_t[3] += dt * ddx_t
        X_t[4] += dt * ddy_t
        X_t[5] += dt * ddz_t
        
        X_t[0] += dt * X_t[3]
        X_t[1] += dt * X_t[4]
        X_t[2] += dt * X_t[5]
        
        # Store sequences
        input_state = np.hstack((X_c[:3] + 0.0001 * np.random.normal(0, 1, (3,)), u))
        input_sequence.append(input_state)
        
        if len(input_sequence) >= 100:
            input_samples.append(np.array(input_sequence))
            output_samples.append(X_c)
            input_sequence = input_sequence[1:]
    
    return np.array(input_samples), np.array(output_samples)[:, np.newaxis, :]

def generate_control_data_zero_u(x0_c=X0_CHASER, t_start=0, t_end=1.0, dt=0.0001):
    """Generate data with zero control input and no measurement noise."""
    inputs = []
    outputs = []
    window = []
    
    x, y, z, dx, dy, dz = x0_c.copy()
    
    for _ in np.arange(t_start, t_end, dt):
        r1 = np.array([x + MU, y, z])
        r2 = np.array([x - 1 + MU, y, z])
        nr1, nr2 = np.linalg.norm(r1), np.linalg.norm(r2)
        
        ddx = 2 * dy + x - ((1 - MU) / nr1**3) * r1[0] - (MU / nr2**3) * r2[0]
        ddy = -2 * dx + y - ((1 - MU) / nr1**3) * r1[1] - (MU / nr2**3) * r2[1]
        ddz = -((1 - MU) / nr1**3) * r1[2] - (MU / nr2**3) * r2[2]
        
        dx += dt * ddx
        dy += dt * ddy
        dz += dt * ddz
        
        x += dt * dx
        y += dt * dy
        z += dt * dz
        
        state = np.array([x, y, z, dx, dy, dz])
        window.append(state)
        
        if len(window) == 100:
            inputs.append(np.array(window))
            outputs.append(state)
            window.pop(0)
    
    return np.array(inputs), np.array(outputs)[:, None, :]