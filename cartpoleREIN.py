import sys
from torch.distributions import Categorical
import torch
from collections import deque
import gym
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyperparameters
max_epsilon = 1
min_epsilon = 0.005
epsilon_decay = 0.005
batch_size = 500
replay_capacity = 200
lr = 0.001
gamma = 0.99
hid = 20
num_episodes = 5000
steps = 400
success_reward = 1
fail_reward = -20
target_update = 20
epoch = 50
torch.manual_seed(100)
random.seed(100)

# Agent class
class Agent():
    def __init__(self, action_dim, device) -> None:
        self.current_step = 0
        
        # Current stretagy is epsilon greedy stretagy
        self.action_dim = action_dim
        self.device = device

    # Get actions according to current state and model
    def get_action(self, state, policy_net): 
        self.current_step += 1
        pred = policy_net(torch.tensor(state).to(device).unsqueeze(0))
        distribution = Categorical(pred)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

# Deep Q network
class model(torch.nn.Module): 
    def __init__(self, state_dim, action_dim) -> None:
        # Quite simple fully connected network copied from Assignment 1
        super(model, self).__init__()
        self.in_to_hid1 = torch.nn.Linear(state_dim, hid)
        self.hid1_to_hid2 = torch.nn.Linear(hid, hid)
        self.hid2_to_out = torch.nn.Linear(hid, action_dim)

    # Forward mathod
    # Input: Current state
    # Output: Q value for each action
    def forward(self, input):
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.relu(hid1_sum)
        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = torch.relu(hid2_sum)
        output_sum = self.hid2_to_out(self.hid2)
        output = F.softmax(output_sum, dim=1)
        # print('output:', output)
        # print('------------------')
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    agent = Agent(action_dim, device)

    # policy net is used for training
    # target net is used for calculating Q next value to evaluate Q value gotten from policy net
    policy_net = model(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=lr)

    # total reward is for testing purpose
    for episode in range(num_episodes):
        state = env.reset()
        t = 0
        acc_reward = 0
        total_reward = []
        reward_batch = deque(maxlen=batch_size)
        log_prob_batch = deque(maxlen=batch_size)

        # Try to play game, if done, the loop ends
        for t in range(steps):
            action, log_prob = agent.get_action(state, policy_net)
            
            state, reward, done, _ = env.step(action)
            if done:
                reward = fail_reward
            else:
                reward = success_reward
            acc_reward += reward
            reward_batch.append(reward)
            log_prob_batch.append(log_prob)
            
            # env.render()

            t += 1
            if done:
                total_reward.append(acc_reward)
                break
                
            if t == steps:
                print('Trained successfully! Step: {}'.format(t))
                print('Episode: {}'.format(episode))
                exit

            # Training
        discount_batch = [gamma ** i for i in range(len(reward_batch))]
        expected_reward = 0
        for index in range(len(reward_batch)):
            expected_reward += discount_batch[index] * reward_batch[index]
            
        loss_batch = [-lp * expected_reward for lp in log_prob_batch]
        loss = torch.cat(loss_batch).sum()
        # print('loss batch: {}'.format(loss_batch))
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        # For testing purpose
        if episode > 0 and episode % epoch == 0:
            print("episode {}, avg reward: {:.2f}".format(episode, np.mean(total_reward)))
            total_reward = []
    env.close()

    plt.plot(range(num_episodes), total_reward)
    plt.xlabel('Episode')
    plt.ylabel('Round')
    plt.show()