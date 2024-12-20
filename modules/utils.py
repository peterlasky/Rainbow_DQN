"""Utility classes and functions for experiment management.

This module provides utilities for:
    - Progress tracking
    - Experiment logging
    - Plotting
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import torch
from IPython.display import clear_output
from tqdm import tqdm

from .filepath_handler import FilePathHandler
__all__ = [
    'PBar',
    'Logger',
    'Plotter'
]

class PBar:
    """Progress bar for experiment tracking.
    
    Provides a customized tqdm progress bar for tracking:
        - Training steps
        - Episodes completed
        - Evaluation metrics
        - Training speed
        
    Attributes:
        max_steps: Maximum number of training steps
        increment: Steps per update
        elapsed_time: Time elapsed since start
    """
    
    def __init__(self, max_steps: int, increment: int = 1) -> None:
        """Initialize progress bar.
        
        Args:
            max_steps: Total number of steps to track
            increment: Number of steps per update
        """
        self.max_steps = max_steps
        self.increment = increment
        self.bar_format="{desc}:{percentage:3.0f}%|{bar}|{n:,}/{total:,}[t:{elapsed}/{remaining}]{postfix}"
        self._pbar = None
    
    def start(self) -> None:
        """Initialize progress bar with zero values."""
        self._pbar = tqdm(
            total=self.max_steps,
            desc="steps",
            ncols=140,
            bar_format=self.bar_format
        )
        self.update(steps=0, eps=0, update_count=0, avg=0., trailing_avg=0.)

    def update(
            self,
            steps: int,
            update_count: int,
            eps: int,
            avg: float,
            trailing_avg: float
    ) -> None:
        """Update progress bar with current metrics.
        
        Args:
            steps: Current step count
            eps: Episodes completed
            avg: Current evaluation average
            trailing_avg: Trailing evaluation average
        """
        elapsed_time = self.elapsed_time
        rate = steps/elapsed_time if elapsed_time > 0 else 0.0

        self._pbar.set_postfix({
            'eps': f'{eps:,}',
            'ev_avg': f'{avg:.1f}',
            'pol_updts': f'{update_count:,}',
            'tr_avg': f'{trailing_avg:.1f}',
            'rate': f'{rate:.1f} stp/s'
        })
        
        if steps < self.max_steps:
            self._pbar.update(self.increment)
            
    @property
    def elapsed_time(self) -> float:
        """Time elapsed since progress bar start."""
        return self._pbar.format_dict['elapsed']


class Logger:
    """Experiment logger for tracking training progress.
    
    Handles:
        - Experiment notes
        - Parameter logging
        - Training metrics
        - Timing information
        
    Attributes:
        filepaths: Manager for log file paths
    """
    
    def __init__(
            self,
            filepaths: FilePathHandler,
            note: str,
            params: Dict
    ) -> None:
        """Initialize logger.
        
        Args:
            filepaths: File path manager
            note: Experiment description
            params: Parameter dictionary to log
        """
        self.filepaths = filepaths
   
        # Save experiment note
        self.filepaths.note_filepath.write_text(
            f'File created: {datetime.now().strftime("%m-%d-%Y %H:%M")}\n\n{note}'
        )
        
        # Save parameters
        param_text = '\n'.join(f'{k}: {v}' for k, v in params.items())
        self.filepaths.params_filepath.write_text(param_text)

    def save_to_log(
            self,
            history_df: pd.DataFrame,
            overwrite: bool = True
    ) -> None:
        """Save training history to log file.
        
        Args:
            history_df: DataFrame containing training history
            overwrite: Whether to overwrite existing log
        """
        history_df.to_csv(self.filepaths.log_filepath, index=False)

    def append_elapsed_time(
            self,
            time_str: str,
            steps: Optional[int] = None
    ) -> None:
        """Append elapsed time to experiment note.
        
        Args:
            time_str: Elapsed time in HH:MM:SS format
            steps: Optional step count to log
        """
        with open(self.filepaths.note_filepath, 'a') as f:
            timestamp = datetime.now().strftime('%m-%d-%Y %H:%M')
            message = f'\nElapsed time: {time_str}'
            if steps is not None:
                message += f' ({steps:,} steps)'
            message += f' - {timestamp}'
            f.write(message)

    def save_checkpoint(self, 
                        model:      torch.nn,
                        optimizer:  torch.optim, 
                        steps:      int):
        checkpoint = {
            'model': model,  # Save the entire model
            'optimizer': optimizer.state_dict(),
            'steps': steps
            }
        full_path = os.path.join(self.filepaths.checkpoints_dir, f'checkpoint_{steps}.pth')
        torch.save(checkpoint, full_path)


class Plotter:
    ''' Helper class to plot training progress '''
    
    def __init__(self, plot_filepath: str):
        self.plot_filepath = plot_filepath

    def plot_data(self, 
                  history_df: pd.DataFrame,
                  save_plot:  bool =False):
        df = history_df
        xvals = df['steps'] / 1000
        n_steps = df['steps'].max()
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(xvals, df['eval_avg'], label='Eval Average')
        ax[0].plot(xvals, df['trailing_avg'], label='Trailing Average')
        ax[0].plot(xvals, df['best_score'], label='Best Game Score')
        ax[0].set_xlabel('Steps (,000)')
        ax[0].set_ylabel('Average Score')
        ax[0].legend()
        ax[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        ax[1].plot(xvals, df['loss'])
        ax[1].set_xlabel('Steps (,000)')
        ax[1].set_ylabel('Loss')
        ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        clear_output()
        if save_plot:
            plt.savefig(self.plot_filepath)
        plt.show()


def validate_device(device) -> torch.device:
    device = device.lower()
    # Throw error if device is an invalid string, convert to torch.device otherwise.
    if isinstance(device, str):
        try:
            t_device = torch.device(device)
        except RuntimeError as e: 
            print(f"Error: {e}")

    # Throw error if device is not a torch device
    elif isinstance(device, torch.device):
        t_device = device
    else:
        raise ValueError(f"Invalid device {device} specified.")
    
    string_repr = str(t_device)
    if string_repr == 'cpu':
        self._p.device = torch.device('cpu')
        return
    if string_repr.split(':')[0] == 'cuda':
        available = torch.cuda.is_available()
    elif string_repr.split(':')[0] == 'mps':
        available = torch.backends.mps.is_available()
    elif string_repr.split(':')[0] == 'xpu':
        available = torch.xpu.is_available()
    elif string_repr.split(':')[0] == 'hip':
        available = torch.hip.is_available()
    elif string_repr.split(':')[0] == 'cpu':
        available = True
    else:
        raise ValueError(f"Device {device} is not compatible or not recognized.")
        
    if not available:
        raise ValueError(f"Device {device} is not available.")
        
    return t_device


import IPython
def ipynb() -> bool:
    '''
    Simple program to determin if the code is running in a Jupyter notebook
    '''
    try:
        # Check if IPython is running and if the environment is a Jupyter notebook
        from IPython import get_ipython
        return "ipykernel" in str(type(get_ipython()))
    except ImportError:
        return False


def plot_multiple_results(names: List[str],
                          data_col: str = 'eval_avg'):
    assert data_col in ['best_score','eval_avg','trailing_avg','loss']

    # search log dir for each sub directory.  In each sub directory, look for the csv file
    # and load it into a pd datframe. 
    # append the dataframes to a list of dataframes
    dfs = []
    for name in names:
        file = os.path.split(name)[-1] + '.csv'
        csv_path = os.path.join(name, file)
        print(f'Looking for {csv_path}')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
        else:
            print(f'No csv file found for {csv_path}')
    if len(dfs) == 0:
        print(f'No csv files found in {log_dir}')
        return
    # use first dataframe to get the steps column
    main_df = dfs[0]['steps']

    # build combined dataframe
    for i, df in enumerate(dfs):
        # take each eval_avg column from each df, rename it eval_eval_{i} and add it to the main df, matching the steps column
        # first rename the eval_avg column to eval_avg_{i}
        df.rename(columns={data_col: f'{names[i]}'}, inplace=True)
        # then merge the dataframes on the steps column
        main_df = pd.merge(main_df, df[['steps', f'{names[i]}']], on='steps', how='outer')

    main_df['steps'] = main_df['steps'] / 1_000.0
    plt.figure(figsize=(10, 5))
    plt.title(data_col)
    for col in main_df.columns[1:]:
        plt.plot(main_df['steps'], main_df[col], label=col)
    plt.xlabel('thousand steps')
    plt.legend()
    plt.show()
    print(main_df)
