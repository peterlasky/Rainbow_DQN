"""Policy network evaluation and performance tracking.

This module provides functionality for evaluating reinforcement learning policies
and tracking their performance metrics over time. It supports both standard
evaluation and video recording modes through gymnasium environments.

The evaluator maintains running statistics including:
    - Best score achieved
    - Average score over evaluation episodes
    - Trailing average over a configurable window
    - Training loss
    - Performance history in a pandas DataFrame
"""

from collections import deque
from types import SimpleNamespace
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from modules.action_handlers import ActionHandler


class Evaluator:
    """Evaluates and tracks performance of reinforcement learning policies.
    
    Can be used in two modes:
    1. Standard evaluation: Uses basic environment for performance assessment
    2. Video recording: Uses wrapped environment for capturing gameplay videos
    
    Attributes:
        n_games: Number of episodes per evaluation
        trail_length: Length of trailing average window
        env: Gymnasium environment for evaluation
        action_handler: Handler for policy action selection
        trailing_scores: Deque of recent scores for averaging
        metrics: Current evaluation metrics
        history: DataFrame of all evaluation results
    """
    
    def __init__(
            self,
            params: SimpleNamespace,
            env: gym.Env,
            action_handler: ActionHandler,
    ) -> None:
        """Initialize the evaluator.
        
        Args:
            params: Configuration parameters including:
                - n_games_per_eval: Episodes per evaluation
                - trailing_avg_trail: Length of trailing average window
                - device: Device for tensor operations
            env: Gymnasium environment for evaluation
            action_handler: Policy action selection handler
            
        Raises:
            ValueError: If n_games_per_eval > trailing_avg_trail
        """
        if params.n_games_per_eval > params.trailing_avg_trail:
            raise ValueError(
                "n_games_per_eval must be <= trailing_avg_trail"
            )
            
        self.n_games = params.n_games_per_eval
        self.trail_length = params.trailing_avg_trail
        self.env = env
        self.action_handler = action_handler
        self.device = params.device
        
        # Initialize tracking containers
        self.trailing_scores = deque([0.0], maxlen=self.trail_length)
        self.metrics = {
            'steps': 0,
            'best_score': 0.0,
            'eval_avg': 0.0,
            'trailing_avg': 0.0,
            'loss': 0.0
        }
        self.history = pd.DataFrame(self.metrics, index=[0])

    def evaluate(self, steps: int, recent_loss: float) -> None:
        """Evaluate the policy over multiple episodes.
        
        Runs n_games episodes, updates metrics including:
        - Best score achieved
        - Average score over current evaluation
        - Trailing average over recent evaluations
        - Training loss
        
        Args:
            steps: Current training step count
            recent_loss: Recent training loss value
        """
        self.metrics['steps'] = steps
        self.metrics['loss'] = recent_loss
        
        # Run evaluation episodes
        episode_scores = []
        for _ in range(self.n_games):
            score = self._run_episode()
            episode_scores.append(score)
            self.trailing_scores.append(score)
            self.metrics['best_score'] = max(
                self.metrics['best_score'],
                score
            )
            
        # Update averages
        self.metrics['trailing_avg'] = np.mean(self.trailing_scores)
        self.metrics['eval_avg'] = np.mean(episode_scores)
        
        # Update history
        self.history = pd.concat([
            self.history,
            pd.DataFrame(self.metrics, index=[0])
        ], ignore_index=True)
        
    def _run_episode(self) -> float:
        """Run a single evaluation episode.
        
        Returns:
            Total episode reward
        """
        frames, _ = self.env.reset()
        done = truncated = False
        total_reward = 0.0
        
        while not (done or truncated):
            state = torch.from_numpy(frames[1:]).to(self.device)
            action = self.action_handler.get_action(state, eval=True)
            frames, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            
        return total_reward

    @property
    def best_score(self) -> float:
        """Best score achieved during evaluation."""
        return self.metrics['best_score']
    
    @property
    def avg(self) -> float:
        """Average score over most recent evaluation."""
        return self.metrics['eval_avg']
    
    @property
    def trailing_avg(self) -> float:
        """Average score over trailing window."""
        return self.metrics['trailing_avg']
      
    @property
    def history_df(self) -> pd.DataFrame:
        """Complete history of evaluation metrics."""
        return self.history