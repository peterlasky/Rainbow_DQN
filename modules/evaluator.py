import gymnasium as gym
from collections import deque
import numpy as np
import pandas as pd
from typing import Dict
from types import SimpleNamespace
import torch

from modules.action_handlers import ActionHandler # for typing

class Evaluator:
    ''' Helper class to evaluate the policy network
        In my implementation, I instantiate 2 separate evaluators:
            1. For evaluating the policy network, we pass in the basic evalation environment that allows for 
            multiple evaluations.
            2. For video recording, we pass in a modified evalation environment that records video using
                gymnasium's video wrapper. 
    '''
    def __init__(self, 
                 params:    SimpleNamespace,
                 env:       gym.Env, 
                 ah:        ActionHandler,):
        self.p = params  # Store the full params object
        self.n = self.p.n_games_per_eval
        self.trail = self.p.trailing_avg_trail
        self.action_handler = ah
        assert self.n <= self.trail
        self.env = env
        self.trailing_scores = deque([0], maxlen=self.trail)
        self.single_history = {
                          'steps':          0, 
                          'best_score':     0.0, 
                          'eval_avg':       0.0, 
                          'trailing_avg':   0.0, 
                          'loss':           0.
                          }
        self._history_df = pd.DataFrame(self.single_history, index=[0])

    def evaluate(self, steps: int, recent_loss: float):
        ''' Evaluate the policy network '''
        self.single_history['steps'] = steps
        self.single_history['loss'] = recent_loss

        for _ in range(self.n):
            frames, _ = self.env.reset()
            done, truncated = False, False
            score = 0
            while not (done or truncated):
                state = frames[1:]
                state = torch.from_numpy(state).to(self.p.device)
                action = self.action_handler.get_action(state, eval=True)  
                frames, reward, done, truncated, _ = self.env.step(action)
                score += reward

            # Update trailing scores
            self.trailing_scores.append(score)      
            # Update best score
            self.single_history['best_score'] = max(self.single_history['best_score'], score)

        # Calculate averages
        self.single_history['trailing_avg'] = np.mean(self.trailing_scores)
        self.single_history['eval_avg'] = np.array(self.trailing_scores)[-self.n:].mean()
        
        # add the single history to the history dataframe
        self._history_df = pd.concat([self._history_df, pd.DataFrame(self.single_history, index=[0])], ignore_index=True)
        
    @property
    def best_score(self) -> float:
        return self.single_history['best_score']
    
    @property
    def avg(self) -> float:
        return self.single_history['eval_avg']
    
    @property
    def trailing_avg(self) -> float:
        return self.single_history['trailing_avg']
      
    @property
    def history_df(self) -> pd.DataFrame:
        return self._history_df