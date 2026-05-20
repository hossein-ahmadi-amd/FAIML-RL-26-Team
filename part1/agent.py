import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 128
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.35
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)  # outputs scalar V(s)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        # TASK 3: forward pass through critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu', algo='reinforce', baseline=None):
        """
        Args:
            policy:   Policy network instance
            device:   'cpu' or 'cuda'
            algo:     'reinforce' or 'ac'
            baseline: None (no baseline) or float (constant baseline, e.g. 20.0)
                      Only used when algo='reinforce'
        """
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.algo = algo
        self.baseline = baseline  # None or constant float

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.state_values = []  # used only by Actor-Critic

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # ------------------------------------------------------------------
        # TASK 2: REINFORCE
        # ------------------------------------------------------------------
        if self.algo == "reinforce":

            # Compute discounted returns G_t
            returns = discount_rewards(rewards, self.gamma)

            # Normalize returns to reduce variance
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            if self.baseline is None:
                # REINFORCE without baseline
                # loss = -sum( log_prob(a_t) * G_t )
                policy_loss = -(action_log_probs * returns).sum()
            else:
                # REINFORCE with constant baseline b
                # loss = -sum( log_prob(a_t) * (G_t - b) )
                policy_loss = -(action_log_probs * (returns - self.baseline)).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        # ------------------------------------------------------------------
        # TASK 3: Actor-Critic (TD targets, not Monte Carlo)
        # ------------------------------------------------------------------
        elif self.algo == "ac":

            # Stack state values collected during the episode
            state_values = torch.stack(self.state_values).squeeze().to(self.train_device)
            self.state_values = []

            # TD targets: r_t + gamma * V(s_{t+1}) * (1 - done_t)
            with torch.no_grad():
                _, next_state_values = self.policy(next_states)
                next_state_values = next_state_values.squeeze()

            targets = rewards + self.gamma * next_state_values * (1 - done)

            # Advantages: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            advantages = targets - state_values.detach()

            # Actor loss: -sum( log_prob(a_t) * delta_t )
            actor_loss = -(action_log_probs * advantages).sum()

            # Critic loss: MSE between V(s_t) and TD targets
            critic_loss = F.mse_loss(state_values, targets.detach())

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:
            return normal_dist.mean, None
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()

            # Always store state_value; only used by AC in update_policy
            self.state_values.append(state_value)

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
