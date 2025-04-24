"""
Deep Q-Learning implementation using PyTorch
"""
import math
import os
import tempfile
import zipfile
import cloudpickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils import save_state, load_state


class DQNAgent:
    def __init__(self, model, target_model, num_actions, device='cpu'):
        """Initialize Deep Q-Network agent.
        
        Parameters
        ----------
        model: torch.nn.Module
            The Q-Network model
        target_model: torch.nn.Module
            The target Q-Network model
        num_actions: int
            Number of possible actions
        device: str
            PyTorch device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.num_actions = num_actions
        
        # Copy weights from model to target_model
        self.update_target_network(1.0)

    def act(self, observation, epsilon=0.0):
        """Select an action based on current observation using epsilon-greedy policy.
        
        Parameters
        ----------
        observation: numpy.ndarray
            Current observation
        epsilon: float
            Random action probability
            
        Returns
        -------
        action: int
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(observation)
        return q_values.argmax(dim=1).item()
    
    def compute_td_error(self, obs_t, action, reward, obs_tp1, done, gamma):
        """Compute TD-error for a single transition.
        
        Parameters
        ----------
        obs_t: numpy.ndarray
            Current observation
        action: int
            Action taken
        reward: float
            Reward received
        obs_tp1: numpy.ndarray
            Next observation
        done: bool
            Whether the episode is done
        gamma: float
            Discount factor
            
        Returns
        -------
        td_error: float
            TD-error for the transition
        """
        obs_t = torch.tensor(obs_t, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done = torch.tensor([done], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            q_tp1 = self.target_model(obs_tp1)
            q_t_selected = self.model(obs_t).gather(1, action.unsqueeze(1)).squeeze()
            q_tp1_best = q_tp1.max(1)[0]
            q_tp1_best = (1.0 - done) * q_tp1_best
            target = reward + gamma * q_tp1_best
            td_error = target - q_t_selected
        
        return td_error.abs().item()
    
    def update_target_network(self, tau=1.0):
        """Update target network by copying from model.
        
        Parameters
        ----------
        tau: float
            Interpolation parameter - if 1.0, target is set equal to model
        """
        if tau == 1.0:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class ActWrapper:
    def __init__(self, agent, act_params):
        """Wrapper for agent's act function.
        
        Parameters
        ----------
        agent: DQNAgent
            DQN agent
        act_params: dict
            Parameters for the act function
        """
        self._agent = agent
        self._act_params = act_params
    
    def __call__(self, observation, stochastic=True, update_eps=-1):
        """Select an action based on the observation.
        
        Parameters
        ----------
        observation: numpy.ndarray
            Current observation
        stochastic: bool
            Whether to use stochastic policy
        update_eps: float
            Epsilon value to use, if negative, uses stored epsilon value
            
        Returns
        -------
        action: int
            Selected action
        """
        epsilon = self._act_params.get('epsilon', 0.1) if update_eps < 0 else update_eps
        return self._agent.act(observation, epsilon=epsilon * stochastic)
    
    @staticmethod
    def load(path, act_params_new=None):
        """Load agent from file.
        
        Parameters
        ----------
        path: str
            Path to the saved agent
        act_params_new: dict
            New parameters for the act function
            
        Returns
        -------
        act: ActWrapper
            Wrapper for the agent's act function
        """
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        
        if act_params_new:
            for (k, v) in act_params_new.items():
                act_params[k] = v
        
        device = act_params.get('device', 'cpu')
        model = act_params['q_func'](act_params['num_actions']).to(device)
        target_model = act_params['q_func'](act_params['num_actions']).to(device)
        agent = DQNAgent(model, target_model, act_params['num_actions'], device)
        
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)
            
            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            
            state_dict = torch.load(os.path.join(td, "model"), map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            target_model.load_state_dict(state_dict['target_state_dict'])
        
        return ActWrapper(agent, act_params)
    
    def save(self, path=None):
        """Save agent to a file.
        
        Parameters
        ----------
        path: str
            Path to save the agent to
        """
        if path is None:
            path = os.path.join(os.getcwd(), "model.pkl")
        
        with tempfile.TemporaryDirectory() as td:
            save_path = os.path.join(td, "model")
            torch.save({
                'model_state_dict': self._agent.model.state_dict(),
                'target_state_dict': self._agent.target_model.state_dict()
            }, save_path)
            
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            
            with open(arc_name, "rb") as f:
                model_data = f.read()
        
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path, act_params=None):
    """Load agent from file.
    
    Parameters
    ----------
    path: str
        Path to the saved agent
    act_params: dict
        Parameters for the act function
        
    Returns
    -------
    act: ActWrapper
        Wrapper for the agent's act function
    """
    return ActWrapper.load(path, act_params)


def learn(env,
          q_func,
          lr=5e-4,
          lr_decay_factor=0.99,
          lr_growth_factor=1.01,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          double_q=False,
          epoch_steps=20000,
          eval_logger=None,
          save_dir='.',
          test_eps=0.05,
          gpu_memory=1.0,
          render=False,
          device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train a Deep Q-Network model.
    
    Parameters
    ----------
    env: gym.Env
        Environment to train on
    q_func: function
        Function that takes observation_shape and num_actions and returns a torch.nn.Module
    lr: float
        Learning rate for optimizer
    lr_decay_factor: float
        Factor to decrease learning rate by
    lr_growth_factor: float
        Factor to increase learning rate by
    max_timesteps: int
        Number of env steps to optimize for
    buffer_size: int
        Size of the replay buffer
    exploration_fraction: float
        Fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        Final value of random action probability
    train_freq: int
        Update the model every `train_freq` steps
    batch_size: int
        Size of batches sampled from replay buffer for training
    print_freq: int
        How often to print out training progress
    checkpoint_freq: int
        How often to save the model
    checkpoint_path: str
        Path to save the model
    learning_starts: int
        How many steps of the model to collect transitions for before learning starts
    gamma: float
        Discount factor
    target_network_update_freq: float
        Update the target network every `target_network_update_freq` steps
    prioritized_replay: bool
        If True, use prioritized replay buffer
    prioritized_replay_alpha: float
        Alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        Initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        Number of iterations over which beta will be annealed
    prioritized_replay_eps: float
        Epsilon to add to the TD errors when updating priorities
    param_noise: bool
        Whether to use parameter noise
    callback: function
        Called every step with state of the algorithm
    double_q: bool
        Whether to use double Q-learning
    epoch_steps: int
        Number of steps per epoch
    eval_logger: Logger
        Logger for evaluation
    save_dir: str
        Path for saving results
    test_eps: float
        Epsilon value for testing
    gpu_memory: float
        Fraction of GPU memory to use
    render: bool
        Whether to render the environment
    device: str
        PyTorch device to use
    
    Returns
    -------
    act: ActWrapper
        Wrapper around act function
    """
    # Create optimizers
    observation_shape = env.observation_space.shape
    
    # Create target model
    model = q_func(env.action_space.n).to(device)
    target_model = q_func(env.action_space.n).to(device)
    
    agent = DQNAgent(model, target_model, env.action_space.n, device)
    
    # Create replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha, device=device)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = np.linspace(prioritized_replay_beta0, 1.0, prioritized_replay_beta_iters)
    else:
        replay_buffer = ReplayBuffer(buffer_size, device=device)
        beta_schedule = None
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize variables
    num_episodes = 0
    episode_rewards = deque(maxlen=100)
    saved_mean_reward = -math.inf
    
    # Function to train the model
    def train():
        nonlocal saved_mean_reward
        
        if len(replay_buffer) < batch_size:
            return 0
        
        # Sample from replay buffer
        if prioritized_replay:
            beta = beta_schedule[min(t, prioritized_replay_beta_iters - 1)]
            batch = replay_buffer.sample(batch_size, beta)
            obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = batch
        else:
            batch = replay_buffer.sample(batch_size)
            obses_t, actions, rewards, obses_tp1, dones = batch
            weights, batch_idxes = torch.tensor(np.ones_like(rewards.cpu()),device=device,dtype=torch.float32), None
        
        # Compute current Q-values
        q_t = model(obses_t)
        q_t_selected = q_t.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q-values using target network
        with torch.no_grad():
            if double_q:
                # Double Q-learning: Select actions using the main network
                q_tp1 = model(obses_tp1)
                _, a_tp1 = q_tp1.max(dim=1)
                
                # Evaluate those actions using the target network
                q_tp1_target = target_model(obses_tp1)
                q_tp1_best = q_tp1_target.gather(1, a_tp1.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: Select best actions using the target network
                q_tp1 = target_model(obses_tp1)
                q_tp1_best = q_tp1.max(1)[0]
            
            # Zero out Q-values for terminal states
            q_tp1_best = q_tp1_best * (1.0 - dones)
            
            # Compute the TD target
            td_target = rewards + gamma * q_tp1_best
        
        # Compute Huber loss
        td_error = q_t_selected - td_target
        loss = F.smooth_l1_loss(q_t_selected, td_target, reduction='none')
        
        # Apply importance weights from prioritized replay
        weighted_loss = torch.mean(loss * weights)
        
        # Optimize the model
        optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        
        # Update target network if it's time
        if target_network_update_freq < 1:
            # Soft target update
            agent.update_target_network(target_network_update_freq)
        else:
            # Hard target update
            if t % target_network_update_freq == 0:
                agent.update_target_network()
        
        # Update priorities in prioritized replay buffer
        if prioritized_replay:
            new_priorities = np.abs(td_error.detach().cpu().numpy()) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)
        
        return (loss * weights).mean().item()
    
    # Create the schedule for exploration
    exploration = np.linspace(1.0, exploration_final_eps, int(exploration_fraction * max_timesteps))
    
    # Initialize the parameters and copy them to the target network
    agent.update_target_network()
    
    # Create action function
    act_params = {
        'epsilon': exploration_final_eps,
        'q_func': q_func,
        'num_actions': env.action_space.n,
        'device': device
    }
    act = ActWrapper(agent, act_params)
    
    # Initialization variables
    obs = env.reset()
    episode_reward = 0
    episode_step = 0
    loss = 0
    t = 0
    episode_rewards_history = []
    
    # Main training loop
    for t in range(max_timesteps):
        # Select action
        action = act(obs, stochastic=True, update_eps=exploration[min(t, len(exploration) - 1)])
        
        # Execute action and observe next state
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        # Store transition in the replay buffer
        replay_buffer.add(obs, action, reward, next_obs, float(done))
        
        # Update statistics
        episode_reward += reward
        episode_step += 1
        
        # Train the network
        if t > learning_starts and t % train_freq == 0:
            loss = train()
        
        # Update observation
        obs = next_obs
        
        # End of episode
        if done:
            # Update episode statistics
            num_episodes += 1
            episode_rewards.append(episode_reward)
            episode_rewards_history.append(episode_reward)
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            episode_step = 0
        
        # Evaluate and save model
        if t > learning_starts and checkpoint_freq is not None and t % checkpoint_freq == 0:
            # Compute mean reward
            mean_100ep_reward = np.mean(episode_rewards)
            
            # Print progress
            if print_freq is not None and len(episode_rewards) > 1 and t % print_freq == 0:
                print(f"Steps: {t}")
                print(f"Episodes: {num_episodes}")
                print(f"Mean 100 episode reward: {mean_100ep_reward:.2f}")
                print(f"% time spent exploring: {int(100 * exploration[min(t, len(exploration) - 1)])}")
                print(f"Learning rate: {lr:.5f}")
            
            # Save best model
            if checkpoint_path is not None:
                if mean_100ep_reward > saved_mean_reward:
                    print(f"Saving model due to mean reward increase: {saved_mean_reward:.2f} -> {mean_100ep_reward:.2f}")
                    act.save(checkpoint_path)
                    saved_mean_reward = mean_100ep_reward
        
        # Call callback if provided
        if callback is not None:
            if callback(locals(), globals()):
                break
    
    return act
