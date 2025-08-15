import numpy as np
import torch
from scipy import linalg
from .environment import SpacecraftRendezvousEnv
from .lstm_model import Model
from .constants import A, B

class Control:
    def __init__(self, env, model_object):
        self.stimator = model_object
        self.A = A
        self.B = B
        self.T = 3.5  
        self.dt = env.dt
        self.environment = env
        self.device = self.stimator.device
        self.measure = 'chaser'

    def LQR(self, Q, R):
        steps = int(self.T / self.dt)
        chaser_states = torch.zeros(steps, 6).to(self.device)
        target_states = torch.zeros(steps, 6).to(self.device)
        y_hats = torch.zeros(steps, 6).to(self.device)
        actions = np.zeros((steps, 3))
        
        y_hats[0] = self.environment.chaser_state
        chaser_states[0] = self.environment.chaser_state
        target_states[0] = self.environment.target_state
        
        # Solve LQR
        P = linalg.solve_continuous_are(self.A, self.B, Q, R)
        K = np.linalg.inv(R) @ (self.B.T @ P)
        
        chaser_measures = []
        
        for i in range(steps - 1):
            if self.measure == 'chaser':
                chaser_measure = self.environment.chaser_state.clone()
                
                if len(chaser_measures) < 100:
                    chaser_measures.append(chaser_measure)
                    y_hat = self.environment.chaser_state.clone()
                    error = self.environment.target_state - y_hat
                    action = K @ np.transpose(error.detach().cpu().numpy())
                    actions[i + 1] = action
                    y_hats[i + 1] = y_hat
                    self.environment.step(action)
                else:
                    chaser_measures = chaser_measures[1:]
                    chaser_measures.append(chaser_measure)
                    
                    measures_tensor = torch.stack(chaser_measures)
                    measures_tensor = measures_tensor.unsqueeze(0)
                    #y_hat = self.stimator.predict(measures_tensor).squeeze(0)
                    y_hat = self.environment.chaser_state

                    error = self.environment.target_state - y_hat
                    action = K @ np.transpose(error.detach().cpu().numpy())
                    actions[i + 1] = action
                    y_hats[i + 1] = y_hat
                    self.environment.step(action)
            
            chaser_states[i + 1] = self.environment.chaser_state
            target_states[i + 1] = self.environment.target_state
        
        return chaser_states, target_states, actions, y_hats
