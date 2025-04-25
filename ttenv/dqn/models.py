import torch
import torch.nn as nn
import torch.nn.functional as F

from ttenv.metadata import DEVICE


class MLP(nn.Module):
    def __init__(self, input_dim, hiddens, num_actions, layer_norm=False):
        """MLP network for DQN algorithm.
        
        Parameters
        ----------
        input_dim: int
            Dimension of input observation
        hiddens: [int]
            List of sizes of hidden layers
        num_actions: int
            Number of possible actions
        layer_norm: bool
            Whether to use layer normalization
        """
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.layer_norm = layer_norm
        
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden in hiddens:
            self.layers.append(nn.Linear(prev_dim, hidden))
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden))
            prev_dim = hidden
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_actions)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        q_values: torch.Tensor
            Output tensor of shape (batch_size, num_actions)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.layer_norms[i](x)
            x = F.relu(x)
        
        return self.output_layer(x)


class CNNToMLP(nn.Module):
    def __init__(self, observation_shape, convs, hiddens, num_actions, dueling=False, layer_norm=False):
        """CNN + MLP network for DQN algorithm.
        
        Parameters
        ----------
        observation_shape: tuple
            Shape of the input observation (channels, height, width)
        convs: [(int, int, int)]
            List of convolutional layers in form of
            (num_outputs, kernel_size, stride)
        hiddens: [int]
            List of sizes of hidden layers
        num_actions: int
            Number of possible actions
        dueling: bool
            If true, use dueling DQN architecture
        layer_norm: bool
            Whether to use layer normalization
        """
        super(CNNToMLP, self).__init__()
        
        self.dueling = dueling
        self.layer_norm = layer_norm
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        
        in_channels = observation_shape[0]
        for out_channels, kernel_size, stride in convs:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            )
            in_channels = out_channels
        
        # Calculate size of CNN output
        # This is a placeholder - in real implementation you'd calculate the actual size
        self.feature_size = self._get_conv_output_size(observation_shape)
        
        # Action value stream
        self.action_layers = nn.ModuleList()
        self.action_layer_norms = nn.ModuleList()
        
        prev_dim = self.feature_size
        for hidden in hiddens:
            self.action_layers.append(nn.Linear(prev_dim, hidden))
            if layer_norm:
                self.action_layer_norms.append(nn.LayerNorm(hidden))
            prev_dim = hidden
        
        self.action_output = nn.Linear(prev_dim, num_actions)
        
        # State value stream (for dueling)
        if dueling:
            self.state_layers = nn.ModuleList()
            self.state_layer_norms = nn.ModuleList()
            
            prev_dim = self.feature_size
            for hidden in hiddens:
                self.state_layers.append(nn.Linear(prev_dim, hidden))
                if layer_norm:
                    self.state_layer_norms.append(nn.LayerNorm(hidden))
                prev_dim = hidden
            
            self.state_output = nn.Linear(prev_dim, 1)
    
    def _get_conv_output_size(self, shape):
        """Calculate output size of convolutional layers."""
        o = torch.zeros(1, *shape)
        for conv in self.conv_layers:
            o = conv(o)
        return int(torch.prod(torch.tensor(o.size())))
    
    def forward(self, x):
        """Forward pass through the network.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns
        -------
        q_values: torch.Tensor
            Output tensor of shape (batch_size, num_actions)
        """
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        conv_out = x.view(x.size(0), -1)  # Flatten
        
        # Action value stream
        action_out = conv_out
        for i, layer in enumerate(self.action_layers):
            action_out = layer(action_out)
            if self.layer_norm:
                action_out = self.action_layer_norms[i](action_out)
            action_out = F.relu(action_out)
        
        action_scores = self.action_output(action_out)
        
        if self.dueling:
            # State value stream
            state_out = conv_out
            for i, layer in enumerate(self.state_layers):
                state_out = layer(state_out)
                if self.layer_norm:
                    state_out = self.state_layer_norms[i](state_out)
                state_out = F.relu(state_out)
            
            state_value = self.state_output(state_out)
            
            # Combine streams
            action_mean = action_scores.mean(dim=1, keepdim=True)
            action_centered = action_scores - action_mean
            q_out = state_value + action_centered
        else:
            q_out = action_scores
        
        return q_out


class ParticleDeepSetMLP(nn.Module):
    def __init__(self, target_dim, agent_dim, output_dim):
        super(ParticleDeepSetMLP, self).__init__()
        self.target_dim = target_dim
        self.agent_dim = agent_dim
        self.output_dim = output_dim
        # agent embedding
        hidden_dim = 10 * target_dim
        # self.agent_embedding = nn.Linear(agent_dim, hidden_dim//3)
        self.phi_func = nn.Sequential(nn.Linear(target_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                      nn.Linear(hidden_dim // 2, hidden_dim // 2))
        reg_dim = hidden_dim // 2 + agent_dim
        self.regressor = nn.Sequential(nn.Linear(reg_dim, reg_dim), nn.ReLU(), nn.Linear(reg_dim, reg_dim), nn.ReLU(),
                                       nn.Linear(reg_dim, output_dim))

    def forward(self, target_belief,agent):
        agent_batch_size, agent_set_size, agent_input_dim = agent.shape
        agent_reshaped = agent.view(-1, agent_input_dim)
        # agent_rep = self.agent_embedding(agent_reshaped)
        # agent_rep = self.phi_func(agent_rep)
        batch_size, set_size, input_dim = target_belief.shape
        target_belief_reshaped = target_belief.view(-1, input_dim)
        target_rep = self.phi_func(target_belief_reshaped)
        target_rep = target_rep.view(batch_size, set_size, -1)
        sum_pooled = torch.cat((target_rep.sum(dim=1), agent_reshaped), dim=1)
        return self.regressor(sum_pooled)

def get_mlp_model(input_dim, hiddens=[], layer_norm=False):
    """Factory function to create MLP model.
    
    Parameters
    ----------
    input_dim: int
        Dimension of the input observation
    hiddens: [int]
        List of sizes of hidden layers
    layer_norm: bool
        Whether to use layer normalization
        
    Returns
    -------
    model_fn: function
        Function that creates an MLP model
    """
    def model_fn(num_actions):
        return MLP(input_dim, hiddens, num_actions, layer_norm)
    
    return model_fn


def get_cnn_model(observation_shape, convs, hiddens, dueling=False, layer_norm=False):
    """Factory function to create CNN+MLP model.
    
    Parameters
    ----------
    observation_shape: tuple
        Shape of the observation (channels, height, width)
    convs: [(int, int int)]
        List of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        List of sizes of hidden layers
    dueling: bool
        If true, use dueling DQN architecture
    layer_norm: bool
        Whether to use layer normalization
        
    Returns
    -------
    model_fn: function
        Function that creates a CNN+MLP model
    """
    def model_fn(num_actions):
        return CNNToMLP(observation_shape, convs, hiddens, num_actions, dueling, layer_norm)
    
    return model_fn


def get_deepsetmlp_model(target_dim,agent_dim):
    """Factory function to create MLP model.

    Parameters
    ----------
    target_dim: int
        Dimension of a particle, calculated as 1+target state dim, 1 for the weight of the particle
    agent_dim: int
        Dimension of the agent state

    Returns
    -------
    model_fn: function
        Function that creates an ParticleDeepSetMLP model
    """

    def model_fn(num_actions):
        return ParticleDeepSetMLP(target_dim, agent_dim, num_actions)

    return model_fn