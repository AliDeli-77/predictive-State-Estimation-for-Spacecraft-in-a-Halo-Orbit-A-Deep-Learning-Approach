import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size[0], num_layers=num_layers, batch_first=True, dropout=0.8)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size[0])
        self.fc1 = nn.Linear(hidden_size[0], output_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size[1])
        self.fc2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.batch_norm3 = nn.BatchNorm1d(hidden_size[2])
        self.fc3 = nn.Linear(hidden_size[2], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        return out

class Model:
    def __init__(self, dt=0.0001, model_path='model/model.pth'):
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [8.274, 0, -1.821, 0, 2, 0],
            [0, -3.019, 0, -2, 0, 0],
            [-1.821, 0, -3.255, 0, 0, 0]
        ])

    def pre_process(self, x_train, y_train, x_tv, y_tv, test_size=0.4, noise_power=0.000001):
        x_val, x_test, y_val, y_test = train_test_split(x_tv, y_tv, test_size=test_size, shuffle=False)
        self.noise_power = noise_power

        self.x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        self.x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)

    def train(self, hidden_size=[32, 16, 8], learning_rate=0.005, num_epochs=2000, batch_size=512):
        input_size = self.x_val_tensor.shape[-1]
        output_size = self.y_test_tensor.shape[-1]
        self.model = LSTMModel(input_size, hidden_size, output_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        train_dataset = TensorDataset(self.x_train_tensor, self.y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(self.x_val_tensor, self.y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in train_dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y.squeeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(train_dataloader)
            
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.model_path)
        
        return self.model

    def load_model(self, hidden_size=[32, 16, 8]):
        input_size = self.x_val_tensor.shape[-1]
        output_size = self.y_test_tensor.shape[-1]
        
        self.model = LSTMModel(input_size, hidden_size, output_size).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        self.model.eval()
        return self.model

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x).to(self.device)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
