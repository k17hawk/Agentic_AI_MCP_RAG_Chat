"""
Reinforcement Learning - RL-based trading agent
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import random
from collections import deque
from datetime import datetime
import json
import os
from utils.logger import logger as logging

class ReinforcementLearning:
    """
    Reinforcement Learning - RL-based trading agent using Q-learning
    
    Features:
    - Q-learning with experience replay
    - Epsilon-greedy exploration
    - Discounted future rewards
    - State discretization
    - Multiple action support
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # RL parameters
        self.learning_rate = config.get("learning_rate", 0.1)
        self.discount_factor = config.get("discount_factor", 0.95)
        self.epsilon = config.get("epsilon", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        
        # Experience replay
        self.replay_memory_size = config.get("replay_memory_size", 10000)
        self.batch_size = config.get("batch_size", 32)
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        
        # State and action spaces
        self.state_size = config.get("state_size", 10)
        self.action_size = config.get("action_size", 3)  # 0: hold, 1: buy, 2: sell
        
        # Q-table (for discrete states)
        self.q_table = {}
        
        # Storage
        self.data_dir = config.get("data_dir", "data/rl")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Training stats
        self.training_stats = {
            "episodes": 0,
            "total_rewards": [],
            "avg_rewards": [],
            "epsilon_history": []
        }
        
        logging.info(f"✅ ReinforcementLearning initialized")
    
    def discretize_state(self, state: np.ndarray) -> str:
        """
        Convert continuous state to discrete representation
        """
        # Simplified discretization - in production, use proper binning
        discrete = []
        for i, value in enumerate(state):
            if i < 3:  # Price-based features
                if value > 0.01:
                    discrete.append("up")
                elif value < -0.01:
                    discrete.append("down")
                else:
                    discrete.append("flat")
            elif i < 6:  # Volume-based
                if value > 1.5:
                    discrete.append("high_vol")
                elif value < 0.5:
                    discrete.append("low_vol")
                else:
                    discrete.append("norm_vol")
            else:  # Other indicators
                if value > 0.7:
                    discrete.append("high")
                elif value < 0.3:
                    discrete.append("low")
                else:
                    discrete.append("med")
        
        return "_".join(discrete)
    
    def get_q_value(self, state_key: str, action: int) -> float:
        """
        Get Q-value for state-action pair
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        return self.q_table[state_key][action]
    
    def update_q_value(self, state_key: str, action: int, value: float):
        """
        Update Q-value for state-action pair
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        self.q_table[state_key][action] = value
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        """
        state_key = self.discretize_state(state)
        
        if not evaluate and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: best action
            q_values = self.q_table.get(state_key, [0.0] * self.action_size)
            return int(np.argmax(q_values))
    
    def store_experience(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory
        """
        self.replay_memory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })
    
    def experience_replay(self):
        """
        Train using experience replay
        """
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.replay_memory, self.batch_size)
        
        for experience in batch:
            state = experience["state"]
            action = experience["action"]
            reward = experience["reward"]
            next_state = experience["next_state"]
            done = experience["done"]
            
            state_key = self.discretize_state(state)
            next_state_key = self.discretize_state(next_state)
            
            # Current Q-value
            current_q = self.get_q_value(state_key, action)
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = self.q_table.get(next_state_key, [0.0] * self.action_size)
                max_next_q = max(next_q_values)
                target_q = reward + self.discount_factor * max_next_q
            
            # Update
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.update_q_value(state_key, action, new_q)
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Train for one episode
        """
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Select action
            action = self.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Experience replay
            self.experience_replay()
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            "episode": self.training_stats["episodes"],
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": self.epsilon
        }
    
    def train(self, env, n_episodes: int = 1000, 
             log_interval: int = 100) -> Dict[str, Any]:
        """
        Train the RL agent
        """
        for episode in range(n_episodes):
            result = self.train_episode(env)
            
            self.training_stats["episodes"] += 1
            self.training_stats["total_rewards"].append(result["total_reward"])
            self.training_stats["epsilon_history"].append(self.epsilon)
            
            if len(self.training_stats["total_rewards"]) >= 100:
                avg_reward = np.mean(self.training_stats["total_rewards"][-100:])
                self.training_stats["avg_rewards"].append(avg_reward)
            
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.training_stats["total_rewards"][-log_interval:])
                logging.info(f"Episode {episode + 1}: Avg Reward={avg_reward:.2f}, Epsilon={self.epsilon:.3f}")
        
        return {
            "episodes_trained": n_episodes,
            "final_epsilon": self.epsilon,
            "avg_reward_last_100": np.mean(self.training_stats["total_rewards"][-100:]) if len(self.training_stats["total_rewards"]) >= 100 else 0,
            "q_table_size": len(self.q_table)
        }
    
    def evaluate(self, env, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate the trained agent
        """
        original_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during evaluation
        
        total_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, evaluate=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        self.epsilon = original_epsilon
        
        return {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards),
            "n_episodes": n_episodes
        }
    
    def save_model(self, filename: str = None):
        """
        Save Q-table to disk
        """
        if filename is None:
            filename = f"rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "training_stats": self.training_stats,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logging.info(f"💾 Saved RL model to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self, filename: str):
        """
        Load Q-table from disk
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.q_table = data["q_table"]
            self.epsilon = data["epsilon"]
            self.training_stats = data["training_stats"]
            
            logging.info(f"📂 Loaded RL model from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")