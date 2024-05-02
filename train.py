import numpy as np
import pandas as pd
from IPython.display import display
import cProfile 
from tqdm import tqdm
class connect_x:

    def __init__(self):
        self.board_height = 6
        self.board_width = 7
        self.board_state = np.zeros([self.board_height, self.board_width], dtype=np.int8)
        self.players = {'p1': 1, 'p2': 2}
        self.isDone = False
        self.reward = {'win': 1, 'draw': 0.5, 'lose': -1}
    
    def render(self):
        rendered_board_state = self.board_state.copy().astype(np.str)
        rendered_board_state[self.board_state == 0] = ' '
        rendered_board_state[self.board_state == 1] = 'O'
        rendered_board_state[self.board_state == 2] = 'X'
        display(pd.DataFrame(rendered_board_state))
    
    def reset(self):
        self.__init__()
        
    def get_available_actions(self):
        available_cols = []
        for j in range(self.board_width):
            if np.sum([self.board_state[:, j] == 0]) != 0:
                available_cols.append(j)
        return available_cols
    
    def check_game_done(self, player):
        if player == 'p1':
            check = '1 1 1 1'
        else:
            check = '2 2 2 2'
        
        # check vertically then horizontally
        for j in range(self.board_width):
            if check in np.array_str(self.board_state[:, j]):
                self.isDone = True
        for i in range(self.board_height):
            if check in np.array_str(self.board_state[i, :]):
                self.isDone = True
        
        # check left diagonal and right diagonal
        for k in range(0, self.board_height - 4 + 1):
            left_diagonal = np.array([self.board_state[k + d, d] for d in \
                            range(min(self.board_height - k, min(self.board_height, self.board_width)))])
            right_diagonal = np.array([self.board_state[d + k, self.board_width - d - 1] for d in \
                            range(min(self.board_height - k, min(self.board_height, self.board_width)))])
            if check in np.array_str(left_diagonal) or check in np.array_str(right_diagonal):
                self.isDone = True
        for k in range(1, self.board_width - 4 + 1):
            left_diagonal = np.array([self.board_state[d, d + k] for d in \
                            range(min(self.board_width - k, min(self.board_height, self.board_width)))])
            right_diagonal = np.array([self.board_state[d, self.board_width - 1 - k - d] for d in \
                            range(min(self.board_width - k, min(self.board_height, self.board_width)))])
            if check in np.array_str(left_diagonal) or check in np.array_str(right_diagonal):
                self.isDone = True
        
        if self.isDone:
            return self.reward['win']
        # check for draw
        elif np.sum([self.board_state == 0]) == 0:
            self.isDone = True
            return self.reward['draw']
        else:
            return 0.
        
    def make_move(self, a, player):
        # check if move is valid
        if a in self.get_available_actions():
            i = np.sum([self.board_state[:, a] == 0]) - 1
            self.board_state[i, a] = self.players[player]
        else:
            print('Move is invalid')
            self.render()

        reward = self.check_game_done(player)
        
        # give feedback as new state and reward
        return self.board_state.copy(), reward

env = connect_x()
import random

# memory block for deep q learning
class replayMemory:
    def __init__(self):
        self.memory = []
        
    def dump(self, transition_tuple):
        self.memory.append(transition_tuple)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
memory = replayMemory()

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, outputs):
        super(DQN, self).__init__()
        # 6 by 7, 10 by 11 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        linear_input_size = 6 * 7 * 32
        self.MLP1 = nn.Linear(linear_input_size, 50)
        self.MLP2 = nn.Linear(50, 50)
        self.MLP3 = nn.Linear(50, 50)
        self.MLP4 = nn.Linear(50, outputs)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        # flatten the feature vector except batch dimension
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.MLP1(x))
        x = F.leaky_relu(self.MLP2(x))
        x = F.leaky_relu(self.MLP3(x))
        return self.MLP4(x)
device = torch.device("mps:0")
# Assuming that we are on a CUDA machine, this should print a CUDA device:

class Model():
    def __init__(self,network,target):
        self.network = DQN(n_actions).to(device) 
        self.network.load_state_dict(network.state_dict())
        self.memory = replayMemory()
        self.optimizer = optim.Adam(self.network.parameters())
        self.target_net = target
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), \
                                            [m[1]], m[2], np.expand_dims(m[3], axis=0)) for m in transitions])
        # tensor wrapper
        state_batch = torch.tensor(state_batch, dtype=torch.float, device=device)
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
        
        # for assigning terminal state value = 0 later
        non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=device)
        non_final_next_state = torch.cat([torch.tensor(s_, dtype=torch.float, device=device).unsqueeze(0) for s_ in next_state_batch if s_[0] is not None])
        
        # prediction from policy_net
        state_action_values = self.network(state_batch).gather(1, action_batch)
        
        # truth from target_net, initialize with zeros since terminal state value = 0
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad
        next_state_values[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()
        # compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # torch.tensor.unsqueeze returns a copy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


import matplotlib.pyplot as plt
# epilson decay graph
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

steps_done = np.arange(20000)
eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1 * steps_done / EPS_DECAY)


import torch.optim as optim
import math

BATCH_SIZE = 256
GAMMA = 0.999

# get max no. of actions from action space
n_actions = env.board_width

height = env.board_height
width = env.board_width

policy_net = DQN(n_actions).to(device)
# target_net will be updated every n episodes to tell policy_net a better estimate of how far off from convergence
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
# set target_net in testing mode
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())

def select_action(state, available_actions, steps_done=None, training=True):
    # batch and color channel
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
    epsilon = random.random()
    if training:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    else:
        eps_threshold = 0
    
    # follow epsilon-greedy policy
    if epsilon > eps_threshold:
        with torch.no_grad():
            # action recommendations from policy net
            r_actions = policy_net(state)[0, :]
            state_action_values = [r_actions[action] for action in available_actions]
            argmax_action = np.argmax(state_action_values)
            greedy_action = available_actions[argmax_action]
            return greedy_action
    else:
        return random.choice(available_actions)


def select_actionNetwork(state, available_actions, policy_net,steps_done=None, training=False):
    # batch and color channel
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
        with torch.no_grad():
            # action recommendations from policy net
            r_actions = policy_net(state)[0, :]
            state_action_values = [r_actions[action].item() for action in available_actions]
            argmax_action = np.argmax(state_action_values)
            greedy_action = available_actions[argmax_action]
            return greedy_action
def play(agent1, agent2, env, render= False):
    env.reset()
    if render : env.render()

    while not env.isDone:
        state = env.board_state.copy()
        available_actions = env.get_available_actions()
        action = select_actionNetwork(state, available_actions,agent1, training=False)
        # trained agent's move is denoted by O
        state, reward = env.make_move(action, 'p1')
        if render : env.render()

        if reward == 1:
            return 'p1'
        elif reward == 0.5:
            return 'draw'

        available_actions = env.get_available_actions()
        action =  select_actionNetwork(state, available_actions,agent2, training=False)
        state, reward = env.make_move(action, 'p2')
        if reward == 1:
            return 'p1'
        elif reward == 0.5:
            return 'draw'
        if render:  env.render()

def train_step_p2(model, env,opponent):
    env.reset()
    state_p1 = env.board_state.copy()
    state_p2 = env.board_state.copy()
    steps_done = 0
    while True:
        available_actions = env.get_available_actions()
        action_p1 = select_actionNetwork(state_p1, available_actions, opponent.network,steps_done)
        
        state_p1, reward_p1 = env.make_move(action_p1, 'p1')
        
        if env.isDone:
            if reward_p1 == 1:
                # reward p1 for p1's win
                memory.dump([state_p2, action_p1, -1, None])
            else:
                # state action value tuple for a draw
                memory.dump([state_p2, action_p1, 0.5, None])
            break
    
        state_p2 = state_p1

        available_actions = env.get_available_actions()
        action_p2 = select_actionNetwork(state_p2, available_actions,model.network, steps_done)
        state_p2_, reward_p2 = env.make_move(action_p2, 'p2')
        steps_done += 1
        if env.isDone:
            if reward_p2 == 1:
                # punish p1 for (random agent) p2's win 
                model.memory.dump([state_p2, action_p1, 1, None])
            else:
                # state action value tuple for a draw
                model.memory.dump([state_p2, action_p1, 0.5, None])
            break
        
        # punish for taking too long to win
        model.memory.dump([state_p1, action_p1, -0.05, state_p2_])
        state_p1 = state_p2_
        
        # Perform one step of the optimization (on the policy network)
        model.optimize_model()
def train_step_p1(model, env,opponent):
    env.reset()
    state_p1 = env.board_state.copy()
    state_p2 = env.board_state.copy()
    steps_done = 0
    while True:
        available_actions = env.get_available_actions()
        action_p1 = select_actionNetwork(state_p1, available_actions, model.network,steps_done)
        steps_done += 1
        state_p1_, reward_p1 = env.make_move(action_p1, 'p1')
        state_p2 = state_p1
        if env.isDone:
            if reward_p1 == 1:
                # reward p1 for p1's win
                model.memory.dump([state_p1, action_p1, 1, None])
            else:
                # state action value tuple for a draw
                model.memory.dump([state_p1, action_p1, 0.5, None])
            break
        
        available_actions = env.get_available_actions()
        action_p2 = select_actionNetwork(state_p2, available_actions,opponent.network, steps_done)
        state_p2_, reward_p2 = env.make_move(action_p2, 'p2')
        
        if env.isDone:
            if reward_p2 == 1:
                # punish p1 for (random agent) p2's win 
                model.memory.dump([state_p1, action_p1, -1, None])
            else:
                # state action value tuple for a draw
                model.memory.dump([state_p1, action_p1, 0.5, None])
            break
        
        # punish for taking too long to win
        memory.dump([state_p1, action_p1, -0.05, state_p2_])
        state_p1 = state_p2_
        
        # Perform one step of the optimization (on the policy network)
        model.optimize_model()
class EvolutionaryTrainer:
    def __init__(self,Pm, Pc, N, env,initial_networkp1 = None, initial_networkp2 = None, tournament_size = None):
        self.Pm = Pm 
        self.Pc = Pc
        self.N = N
        self.initial_networkp1 = initial_networkp1
        self.initial_networkp2 = initial_networkp2
        if(initial_networkp1 == None):
            print("Please provide an initial network for player 1")
            print("Please provide an initial network for player 2")
        self.individualsp1 = [Model(initial_networkp1, initial_networkp1) for i in range(N)]
        self.individualsp2 = [Model(initial_networkp2, initial_networkp2) for i in range(N)]
        self.tournament_size = tournament_size
        self.best_individualp1 = Model(initial_networkp1, initial_networkp1)
        self.best_individualp2 = Model(initial_networkp2, initial_networkp2)
        self.env = env
        self.keep = 5
        self.mutation_size = 5
        self.nbMutation = 4
        self.mutation_pool1 = []
        self.mutation_pool2 = []
#on commence à 15
#on en selectionne 5 (parmis 10)

#on en fait muter 5

    def selection(self,keep):
        #
        recordsp1 = []
        recordsp2 = []
        for i in range(self.tournament_size):
            recordsp1.append(0)
            recordsp2.append(0)
        index_p1 = np.random.choice(self.N, size=self.tournament_size, replace=False)
        index_p2 = np.random.choice(self.N, size=self.tournament_size, replace=False)
        

        for i in range(self.tournament_size):
            for j in range(self.tournament_size):
              
                res = play(self.individualsp1[index_p1[i]].network,self.individualsp2[index_p2[j]].network, self.env)
                if res == 'p1':
                    recordsp1[i] += 1
                elif res == 'p2':
                    recordsp2[i] += 1
                else:
                    recordsp1[i] += 0.5
                    recordsp2[i] += 0.5
        
        best_index_p1 = (np.argsort(recordsp1)[::-1])[:keep]
        best_index_p2 = (np.argsort(recordsp2)[::-1])[:keep]
     
        index_p1 = np.delete(index_p1, best_index_p1)
        index_p2 = np.delete(index_p2, best_index_p2)
        
        self.individualsp1 = list(np.delete(self.individualsp1, index_p1))
        self.individualsp2 = list(np.delete(self.individualsp2, index_p2))
        
    def crossover(self):
        #None for the moment 
        pass 
    def mutation(self):
        for network in self.mutation_pool1:
            copy = Model(network.network, self.best_individualp1.network)
            for i in range(self.nbMutation):
                
                train_step_p1(copy, env, self.best_individualp2)
               
            self.individualsp1.append(copy)
        for network in self.mutation_pool2:
            copy = Model(network.network, self.best_individualp2.network)
            for i in range(self.nbMutation):
                
                train_step_p2(copy, env, self.best_individualp1)
               
            self.individualsp2.append(copy)
        
    def fitness_function(self):
        #Number of win vs the last best network
        pass


    def train(self, nbTrain):
        for i in range(nbTrain):
            self.selection( 5)
            print(len(self.individualsp1))
            index_p1 = np.random.choice(5, size=self.mutation_size, replace=False)
            index_p2 = np.random.choice(5, size=self.mutation_size, replace=False)
            self.mutation_pool1 = [self.individualsp1[index] for index in index_p1]
            self.mutation_pool2 = [self.individualsp2[index] for index in index_p2]
            self.mutation()



agent1= DQN(n_actions).to(device) 
agent2 = DQN(n_actions).to(device) 
agent1.load_state_dict(torch.load("DQN_plainCNN.pth",map_location=torch.device('cpu')))
agent2.load_state_dict(torch.load("DQN_plainCNN.pth",map_location=torch.device('cpu')))

evo = EvolutionaryTrainer(1,1,15,env, agent1, agent2,10)


cProfile.run('evo.train(10)')