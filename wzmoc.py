import gymnasium as gym
import numpy as np
from minigrid.envs import FourRoomsEnv
import matplotlib.pyplot as plt
import random

def create_fixed_env(max_steps):
    env = gym.make("MiniGrid-FourRooms-v0", max_steps=max_steps, 
                   goal_pos=(2, 2), agent_pos=(7, 7))
    return env

class QLearningAgent:
    def __init__(self, n_actions, beta, gamma, epsilon):
        self.q_table = {} 
        self.n_actions = n_actions
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, env):
        return (env.unwrapped.agent_pos[0], env.unwrapped.agent_pos[1], env.unwrapped.agent_dir)

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.get_q_values(state))

    def learn(self, state, action, reward, next_state, terminated):
        old_q = self.get_q_values(state)[action]
        next_max_q = 0 if terminated else np.max(self.get_q_values(next_state))
        
        self.q_table[state][action] = old_q + self.beta * (reward + self.gamma * next_max_q - old_q)

def train(episodes, beta, gamma, epsilon):
    env = create_fixed_env(max_steps=300)
    agent = QLearningAgent(env.action_space.n, beta, gamma, epsilon)
    rewards_history = []

    for ep in range(episodes):
        env.reset()
        state = agent.get_state_key(env)
        total_reward = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.choose_action(state)
            _, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.get_state_key(env)

            current_reward = reward if reward > 0 else -0.01
            
            agent.learn(state, action, current_reward, next_state, terminated)
            
            state = next_state
            total_reward += reward

        agent.epsilon = max(0.05, agent.epsilon * 0.99)
        
        rewards_history.append(total_reward)
        
        if (ep + 1) % 100 == 0:
            success_rate = sum(1 for r in rewards_history[-100:] if r > 0)
            print(f"Epizod {ep+1}: Sukcesy w tej setce: {success_rate}%")
            
    return rewards_history

experiments = [
    {"beta": 0.1, "gamma": 0.1, "epsilon": 0.1, "label": "Gamma = 0.1"},
    {"beta": 0.1, "gamma": 0.3, "epsilon": 0.1, "label": "Gamma = 0.3"},
    {"beta": 0.1, "gamma": 0.6, "epsilon": 0.1, "label": "Gamma = 0.6"},
    {"beta": 0.1, "gamma": 0.9, "epsilon": 0.1, "label": "Gamma = 0.9"}
]

plt.figure(figsize=(12, 6))

for exp in experiments:
    print(f"Obecny zestaw: {exp['label']}...")
    results = train(episodes=500, beta=exp['beta'], gamma=exp['gamma'], epsilon=exp['epsilon'])
    
    smoothed = np.convolve(results, np.ones(20)/20, mode='valid')
    plt.plot(smoothed, label=exp['label'])

plt.title("Wpływ parametrów na proces uczenia")
plt.xlabel("Epizody")
plt.ylabel("Nagroda")
plt.legend()
plt.grid(True)
plt.show()