import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor
LR_CRITIC = 1e-4       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 4
UPDATE_NUMBER = 15
ACTOR_UPDATE = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''
    The agent who interacts with the environment and learns it
    '''

    def __init__(self, state_space, action_space, random_seed):

        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(random_seed)
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.lr_adjust = 0.999
        self.lr_min_actor = 1e-7
        self.lr_min_critic = 1e-6

        # The actor network
        self.actor_local = Actor(state_space, action_space).to(device)
        self.actor_target = Actor(state_space, action_space).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # The critic network
        self.critic_local = Critic(state_space, action_space).to(device)
        self.critic_target = Critic(state_space, action_space).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=WEIGHT_DECAY)


        # Noise which could be added but is not added in the initial step
        # self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_space, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done, action_space, sigma, num_agents):
        '''
        Get the current state, action, reward, next_state and done tuple and store it in the replay buffer.
        Also check if already enough samples have been collected in order to start a training step
        :param state: Current state of the environment
        :param action: Action that has been chosen in current state
        :param reward: The reward that has been received in current state
        :param next_state: The next state that has been reached due to the current action
        :param done: Parameter to see if an episode has finished
        :return: -
        '''
        #Save experience, reward, and next state in the replay buffer
        # actions_next = actions_next + torch.from_numpy(noise)
        # actions_next = np.clip(actions_next, -1,1)
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # Learn only if enough samples have already been collected
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, action_space, sigma, num_agents)

    def act(self, state, sigma, action_space, num_agents):
        '''
        Functionality to determine the action in the current state.
        Additionally to the current action which is determined by the actor a normal distribution will be added to the
        action space to have enough exploration at least in the beginning of the training. The normal distribution will
        get smaller with time expressed through the parameter sigma
        :param state: Current state of the environment
        :param sigma: Parameter which decays over time to make the normal distribution smaller
        :return: action which shall be performed in the environment
        '''
        noise = np.array([np.random.normal(0, sigma, action_space), np.random.normal(0, sigma, action_space)])
        #noise = np.random.normal(0, sigma, action_space)
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = action + noise
        return np.clip(action, -1, 1)

    def act_rand(self, sigma, action_space, num_agents):
        '''
        Functionality to determine the action in the current state.
        Additionally to the current action which is determined by the actor a normal distribution will be added to the
        action space to have enough exploration at least in the beginning of the training. The normal distribution will
        get smaller with time expressed through the parameter sigma
        :param action_space:
        :param state: Current state of the environment
        :param sigma: Parameter which decays over time to make the normal distribution smaller
        :return: action which shall be performed in the environment
        '''
        rand_act = np.array([np.random.normal(0, sigma, action_space), np.random.normal(0, sigma, action_space)])

        return np.clip(rand_act, -1, 1)

    def learn(self, experiences, gamma, action_space, sigma, num_agents):
        '''
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r+ gamma * critic_target(next_state, actor_target(next_state)
            actor_target(state) --> action
            critic_target(state, action) --> Q-value
        Also update the the actor with gradient ascent by comparing the loss between the actor and the critiv actions.
        Perform the learning multiple times expressed by the parameter UPDATE_NUMBER

        IMPORTANT TRICK:
        A important trick has been introduced to make the learning more stable.
        The learning rate decays over time. After every learning step the learning rate will be decayed.
        This makes the agent in the beginning more aggressive and more passive the longer it trains
        The function for this is called exp_lr_scheduler
        :param experiences: A sample of states, actions, rewards, next_states, dones tuples to learn from
        :param gamma: Value to determine the reward discount
        :return: -
        '''

        states, actions, rewards, next_states, dones = experiences


        '''-------------------- Update critic -----------------------'''
        # Get predicted next-state actions and Q values from target models
        for i in range(UPDATE_NUMBER):

            actions_next = self.actor_target(next_states)
            #noise = torch.randn_like(actions_next).clamp(-1*sigma, 1*sigma)
            #actions_next = actions_next + noise
            q1_target, q2_target = self.critic_target(next_states, actions_next)
            q_target = torch.min(q1_target, q2_target)
            # Get the Q-targets
            Q_targets = rewards + (gamma * q_target * (1 - dones))
            # Compute the critic loss
            q1_current, q2_current = self.critic_local(states, actions)
            critic_loss = F.mse_loss(q1_current, Q_targets) + F.mse_loss(q2_current, Q_targets)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            '''-------------------- Update actor -----------------------'''
            # Compute the actor loss but do an delayed update only
            if self.t_step % ACTOR_UPDATE == 0:
                actions_pred = self.actor_local(states)
                q1_current, _ = self.critic_local(states, actions_pred)
                actor_loss = -q1_current.mean()

                # Minimize the loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                '''-------------------- Update target networks -----------------------'''
                self.soft_update(self.critic_local, self.critic_target, TAU)
                self.soft_update(self.actor_local, self.actor_target, TAU)

            '''-------------------- Adjust learning rate -----------------------'''
        self.lr_actor = max(self.lr_actor*self.lr_adjust, self.lr_min_actor)
        self.lr_critic = max(self.lr_critic*self.lr_adjust, self.lr_min_critic)
        self.actor_optimizer = self.exp_lr_scheduler(self.actor_optimizer, self.lr_actor)
        self.critic_optimizer = self.exp_lr_scheduler(self.critic_optimizer, self.lr_critic)

    def exp_lr_scheduler(self, optimizer, decayed_lr):
        '''
        Set the learning rate to a decayed learning rate, without initializing the optimizer from scratch
        :param optimizer: the optimizer in which the learning rate shall be adjusted
        :param decayed_lr: the decaed learning rate to be set
        :return: optimizer with new learning rate
        '''

        for param_group in optimizer.param_groups:
            param_group['lr'] = decayed_lr
        return optimizer

    def soft_update(self, local_model, target_model, tau):
        '''
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_mode:
        :param target_model:
        :param tau:
        :return:
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

class ReplayBuffer():
    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''
        Initialize replay buffer
        :param action_size: action size of environment
        :param buffer_size: buffer size for replay buffer
        :param batch_size: batch size to learn from
        :param seed: random seed
        '''

        self.action_space = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''
        Adding a nre state, action, reward, nect_state, done tuplt to the replay memory
        :param state: Current state
        :param action: Action taken in current state
        :param reward: Reward that has been granted
        :param next_state: Next state reached
        :param done: Information if environment has finished
        :return: -
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''
        Radnomly sample a batch
        :return: A random selected batch of the memory
        '''

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)













































