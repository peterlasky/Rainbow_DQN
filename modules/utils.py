import os, shutil, datetime
from tqdm import tqdm
import torch
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import clear_output

__all__ = ['FilePathManager', 
           'PBar', 
           'Logger', 
           'Plotter']

class FilePathManager:
    '''
    simple class that builds the sub-directories and file paths
    
    '''
    def __init__(self, 
                 log_dir:            str, 
                 name:               str,
                 overwrite_previous: bool = False):
        
        self._log_dir = log_dir
        self.sub_dir = os.path.join(log_dir, name)

        # throw an error if the log_dir does not exist
        if not os.path.exists('logs'):
            raise FileNotFoundError('"logs" folder does not exist. Create before using it.')
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # if the sub_dir already exists, delete it and all its contents
        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir)
        else:   
            if overwrite_previous:
                shutil.rmtree(self.sub_dir)
                os.makedirs(self.sub_dir)
            else:
                i = 1
                while os.path.exists(self.sub_dir + '_' + str(i)):
                    i += 1
                self.sub_dir += ('_' + str(i))
                os.makedirs(self.sub_dir)
        with open(os.path.join(self.sub_dir,'created_' + datetime.datetime.now().strftime('%m-%d_%H_%M') + '.txt'), 'w') as f:
            f.write('Folder created ' + datetime.datetime.now().strftime('%m-%d-%Y %H:%M'))


        # create the sub_dir again
        os.makedirs(self.sub_dir, exist_ok=True)
    
        # make the checkpoint sub_dir and the video subdir
        self._checkpoints_dir = os.path.join(self.sub_dir, f'{name}_checkpoints')
        os.makedirs(self._checkpoints_dir)
        self._video_dir = os.path.join(self.sub_dir, f'{name}_video')
        os.makedirs(self._video_dir)

        # logging directory (csv file)
        self._log_filepath = os.path.join(log_dir, name, f'{name}.csv')
        self._note_filepath = os.path.join(log_dir, name, f'{name}_notes.txt')
        self._params_filepath = os.path.join(log_dir, name, f'{name}_params.txt')
        self._plot_filepath = os.path.join(log_dir, name, f'{name}_plot.png')
    
    @property
    def checkpoints_dir(self) -> str:
        return self._checkpoints_dir
    
    @property
    def video_dir(self) -> str:
        return self._video_dir
    
    @property
    def log_filepath(self) -> str:
        return self._log_filepath
    
    @property
    def note_filepath(self) -> str:
        return self._note_filepath
    
    @property
    def params_filepath(self) -> str:
        return self._params_filepath
    
    @property
    def plot_filepath(self) -> str:
        return self._plot_filepath
    
    @property
    def log_dir(self) -> str:
        return self._log_dir
# ---- end of FilePathManager class ----

class PBar:
    ''' simple helper class to manage the pbar object '''
    def __init__(self, 
                 max_steps: int, 
                 increment: int = 1):
        
        self.max_steps = max_steps
        self.increment = increment

        # create pbar object
        self._pbar = tqdm(
            total=     self.max_steps,
            desc=      "steps", 
            ncols=     120,  
            bar_format="{desc}:{percentage:3.0f}%|{bar}|{n:,}/{total:,}[t:{elapsed}/{remaining}]{postfix}")
    
    def start(self):
        self.update(steps=0, eps=0, avg=0., trailing_avg=0.)

    def update(self, 
               steps:        int, 
               eps:          int, 
               avg:          float, 
               trailing_avg: float) -> None:
        
        elapsed_time = self.elapsed_time
        rate = steps/elapsed_time if elapsed_time > 0 else 0.0

        self._pbar.set_postfix(
            {'eps'   : f'{eps:,}',
             'ev_avg': f'{avg:.1f}', 
             'tr_avg': f'{trailing_avg:.1f}',
             'rate'  : f'{rate:.1f} stp/s'})
        if steps < self.max_steps:
            self._pbar.update(self.increment)
            
    @property
    def elapsed_time(self):
        return self._pbar.format_dict['elapsed']
# ---- end of PBar class ----


class Logger:
    ''' Helper class to log training progress '''
    def __init__(self,
                 filepaths:         FilePathManager,
                 note:              str,
                 params:            Dict):

        self.filepaths = filepaths
   
        # save note to note_filepath
        with open(self.filepaths.note_filepath, 'w') as f:
            f.write(f'file created: {datetime.datetime.now().strftime("%m-%d-%Y %H:%M")}\n\n')
            f.write(note)
        
        # save params to params_filepath
        with open(self.filepaths.params_filepath, 'w') as f:
            for key, value in params.items():
                f.write(f'{key}: {value}\n')

    def save_to_log(self, 
                    history_df: pd.DataFrame,
                    overwrite:  bool=True): 
        
        # Overwrite the file if it already exists
        history_df.to_csv(self.filepaths.log_filepath, index=False)

    def append_elapsed_time(self, hh_mm_ss: str, steps: int = None):
        h, m, s = hh_mm_ss.split(':')
        steps_per_min = steps/(int(h)*60 + int(m) + int(s)/60)
        with open(self.filepaths.note_filepath, 'a') as f:
            f.write(f'\nElapsed Time: {hh_mm_ss}\n')
            if steps is not None:
                f.write(f'Steps: {steps:,}\n')
                f.write(f'1000 steps per minute: {steps_per_min:,.4f}\n')

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
# ---- end of Logger class ----

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
# ---- end of Plotter class ----

import IPython
def ipynb():
    try:
        # Check if IPython is running and if the environment is a Jupyter notebook
        from IPython import get_ipython
        return "ipykernel" in str(type(get_ipython()))
    except ImportError:
        return False