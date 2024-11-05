import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import os


PATH_POINT_NUM = 5
STATE_DIM = PATH_POINT_NUM * 2 + 2
ACTION_DIM = 1
HIDDEN_SIZE = 64
MAX_ACCELERATION = 2
MAX_BRAKING = 4
GAMMA = 0.99
BATCH_SIZE = 32
LR = 1e-3
TAU = 0.005
MAX_MEMORY = 10000
PARAM_PATH = 'params/speed_profiler.pth'


class Actor(nn.Module):
    """
    Actor network predicting speed change based on the current state
    """
    def __init__(self) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, ACTION_DIM)

        self.softsign = nn.Softsign()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.tensor) -> None:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softsign(self.fc3(x))

        return self._scale(x)

    @staticmethod
    def _scale(x: torch.tensor) -> torch.tensor:
        """
        Scale from [-1, 1] to [-MAX_BRAKING, MAX_ACCELERATION]
        """
        scale_factor = (MAX_ACCELERATION + MAX_BRAKING) / 2
        offset = (MAX_ACCELERATION - MAX_BRAKING) / 2
        return scale_factor * x + offset


class SpeedProfiler:
    def __init__(self, train: bool = False, logging: bool = False) -> None:
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=MAX_MEMORY)

        # Actor
        self.model = Actor()

        self.optimizer = None
        if os.path.exists(PARAM_PATH):
            self.load_model()
        else:
            self.model.apply(self._initialize_weights)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, betas=(0.85, 0.999), weight_decay=1e-5)

        # Critic
        self.target_model = Actor()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.loss = nn.MSELoss()

        self.logging = logging
        self.train = train

        self.last_state = None
        self.last_action = None
        self.iter = 0

    @staticmethod
    def _initialize_weights(layer: nn.Module) -> None:
        """
        Initialize the weights of a new network using Kaiming initialization
        """
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def load_model(self) -> None:
        """
        Load an already stored model and optimizer state
        """
        checkpoint = torch.load(PARAM_PATH, map_location=self.model.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model(self) -> None:
        """
        Save the model and optimizer state to a file
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, PARAM_PATH)

    @staticmethod
    def _state(path: np.array, current_speed: float, steering_angle: float) -> torch.tensor:
        """
        Convert the path, current speed, and steering angle to a state tensor serving as an input to the model
        """
        # Too many points, truncate the path
        if len(path) > PATH_POINT_NUM:
            path = path[:PATH_POINT_NUM]

        # Too few points, repeat the last point to fill the gap
        elif len(path) < PATH_POINT_NUM:
            last_point = path[-1] if len(path) > 0 else np.zeros(2)
            path = np.vstack((path, np.tile(last_point, (PATH_POINT_NUM - len(path), 1))))

        return torch.tensor(np.concatenate((path.flatten(), [current_speed], [steering_angle])), dtype=torch.float)

    def predict(self, path: np.array, current_speed: float, steering_angle: float, reward: float | None = None) -> float:
        """
        Predict the speed change based on the current state
        """
        self.iter += 1

        # Convert the input to a tensor
        state = self._state(path, current_speed, steering_angle).to(self.model.device)

        # Predict the speed change
        with torch.no_grad():
            speed = self.model(state)

        # Update the model if training
        if self.train and reward is not None and self.last_state is not None:
            self._update_reward(self.last_state, self.last_action, reward, state)
            self.last_state = state
            self.last_action = speed

        # Log the predicted speed
        if self.logging and self.iter % 100 == 0:
            print(f"Predicted speed: {speed.item()}")

        return speed.item()

    def _update_reward(self, state: torch.tensor, action: torch.tensor, reward: float, next_state: torch.tensor) -> None:
        # Store the transition in memory
        self.memory.append((state, action, reward, next_state))

        # Update the model
        if len(self.memory) > self.batch_size:
            self._replay()

    def _update_target_network(self) -> None:
        """
        Update the target network from our model using a soft update
        """
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(TAU * model_param.data + (1.0 - TAU) * target_param.data)

    def _replay(self) -> None:
        """
        Replay the memory and update the model
        """
        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states).to(self.model.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.model.device)
        next_states = torch.stack(next_states).to(self.model.device)

        current_q = self.model(states).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).squeeze()

        target_q = rewards + self.gamma * next_q

        loss = self.loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        self._update_target_network()
