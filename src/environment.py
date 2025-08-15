import numpy as np
import torch
from .constants import MU, X0_CHASER, X0_TARGET

class SpacecraftRendezvousEnv:
    def __init__(self, dt):
        self.mu = MU
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()
        
        # Unit conversions
        self.DU = 384402  # km
        self.TU = 27.3216 * 3600 * 24 / (2 * np.pi)  # sec

    def step(self, action, estimate=False):
        x, y, z, dx, dy, dz = np.array(self.chaser_state.cpu())
        u1, u2, u3 = action

        # Chaser dynamics
        r1 = np.array([x + self.mu, y, z])
        r2 = np.array([x - (1 - self.mu), y, z])
        nr1 = np.linalg.norm(r1)
        nr2 = np.linalg.norm(r2)

        ddx = 2 * dy + x - ((1 - self.mu) / nr1**3) * r1[0] - (self.mu / nr2**3) * r2[0] + u1
        ddy = -2 * dx + y - ((1 - self.mu) / nr1**3) * r1[1] - (self.mu / nr2**3) * r2[1] + u2
        ddz = -((1 - self.mu) / nr1**3) * r1[2] - (self.mu / nr2**3) * r2[2] + u3

        dx += self.dt * ddx
        dy += self.dt * ddy
        dz += self.dt * ddz

        x += self.dt * dx
        y += self.dt * dy
        z += self.dt * dz

        self.chaser_state = torch.tensor([x, y, z, dx, dy, dz], device=self.device, dtype=torch.float32)

        # Target dynamics
        target_x, target_y, target_z, target_dx, target_dy, target_dz = np.array(self.target_state.cpu())
        
        r1_t = np.array([target_x + self.mu, target_y, target_z])
        r2_t = np.array([target_x - (1 - self.mu), target_y, target_z])
        nr1_t = np.linalg.norm(r1_t)
        nr2_t = np.linalg.norm(r2_t)

        target_ddx = 2 * target_dy + target_x - ((1 - self.mu) / nr1_t**3) * r1_t[0] - (self.mu / nr2_t**3) * r2_t[0]
        target_ddy = -2 * target_dx + target_y - ((1 - self.mu) / nr1_t**3) * r1_t[1] - (self.mu / nr2_t**3) * r2_t[1]
        target_ddz = -((1 - self.mu) / nr1_t**3) * r1_t[2] - (self.mu / nr2_t**3) * r2_t[2]

        target_dx += self.dt * target_ddx
        target_dy += self.dt * target_ddy
        target_dz += self.dt * target_ddz

        target_x += self.dt * target_dx
        target_y += self.dt * target_dy
        target_z += self.dt * target_dz

        self.target_state = torch.tensor(
            [target_x, target_y, target_z, target_dx, target_dy, target_dz],
            device=self.device, dtype=torch.float32
        )

    def reset(self):
        self.chaser_state = torch.tensor(X0_CHASER, device=self.device, dtype=torch.float32)
        self.target_state = torch.tensor(X0_TARGET, device=self.device, dtype=torch.float32)