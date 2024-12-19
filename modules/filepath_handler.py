"""File and directory management for reinforcement learning experiments.

This module provides the FilePathManager class for managing experiment outputs:
    - Log directories
    - Checkpoint directories
    - Video recording directories
    - Parameter and note files
    - Plot outputs
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Union
from types import SimpleNamespace
import shutil
import json

class FilePathHandler:
    """Manages experiment directories and file paths.
    
    Creates and manages directory structure for experiment outputs including:
        - Logs
        - Checkpoints
        - Videos
        - Parameter files
        - Notes
        - Plots
        
    Attributes:
        main_log_dir: Root directory for all experiments
        group_dir: Directory for experiment group
        checkpoints_dir: Directory for model checkpoints
        video_dir: Directory for recorded videos
        log_filepath: Path to experiment log file
        note_filepath: Path to experiment notes
        params_filepath: Path to parameter file
        plot_filepath: Path to plot output file
    """
    
    def __init__(self, p: SimpleNamespace) -> None:
        """Initialize directory structure for experiment.
        
        Args:
            p: Parameters namespace containing:
                - main_log_dir: Root log directory
                - group_dir: Group directory name
                - name: Experiment name
                - overwrite_previous: Whether to overwrite existing directories
                - record_interval: Video recording interval (optional)
        """
        # Create main log directory
        self._main_log_dir = Path(p.main_log_dir)
        self._main_log_dir.mkdir(exist_ok=True)
        
        # Setup group directory
        self._group_dir = self._setup_group_dir(p)
        
        # Create experiment directories
        self._video_dir = self._group_dir / p.name / 'videos'
        self._checkpoints_dir = self._group_dir / f'{p.name}_checkpoints'
        
        # Setup file paths
        self._log_filepath = self._group_dir / f'{p.name}.csv'
        self._note_filepath = self._group_dir / f'{p.name}_notes.txt'
        self._params_filepath = self._group_dir / f'{p.name}_params.txt'
        self._plot_filepath = self._group_dir / f'{p.name}_plot.png'
        self._user_params_filepath = self._group_dir / f'{p.name}_user_params.json'
        
        # Create required directories
        # self._create_directories(p)

    def _setup_group_dir(self, p: SimpleNamespace) -> Path:
        """Setup group directory with proper naming and permissions.
        
        Args:
            p: Parameters namespace
            
        Returns:
            Path to group directory
        """
        group_dir = self._main_log_dir / p.group_dir
        
        if group_dir.exists():
            if p.overwrite_previous:
                print(f'Removing existing directory: {group_dir}')
                shutil.rmtree(group_dir)
            else:
                i = 1
                while (group_dir.parent / f'{group_dir.name}_{i}').exists():
                    i += 1
                group_dir = group_dir.parent / f'{group_dir.name}_{i}'
                
        print(f'Creating new directory: {group_dir}')
        group_dir.mkdir(exist_ok=True)
        return group_dir

    # @public
    def _create_directories(self, p: SimpleNamespace) -> None:
        """Create all required subdirectories.
        
        Args:
            p: Parameters namespace
        """
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        if p.record_interval is not None:
            self._video_dir.mkdir(parents=True, exist_ok=True)
            
        # Create timestamp file
        timestamp = datetime.now().strftime('%m-%d_%H_%M')
        timestamp_file = self._group_dir / f'created_{timestamp}.txt'
        timestamp_file.write_text(
            f'Folder created {datetime.now().strftime("%m-%d-%Y %H:%M")}'
        )
    
    def save_user_parameters(self, user_parameters: Union[Dict, SimpleNamespace]):
        """Save user parameters as a JSON file.
        
        Args:
            user_parameters: Parameters to save, either a dict or SimpleNamespace
        """
        if isinstance(user_parameters, SimpleNamespace):
            user_parameters = vars(user_parameters)

        # write as json file
        with open(self._user_params_filepath, 'w') as f:
            json.dump(user_parameters, f, indent=4)

        
    @property
    def filepaths(self) -> SimpleNamespace:
        return SimpleNamespace(
            main_log_dir=self._main_log_dir,
            group_dir=self._group_dir,
            checkpoints_dir=self._checkpoints_dir,
            video_dir=self._video_dir,
            log_filepath=self._log_filepath,
            note_filepath=self._note_filepath,
            params_filepath=self._params_filepath,
            plot_filepath=self._plot_filepath
        )

    @property
    def main_log_dir(self) -> Path:
        """Root directory for all experiments."""
        return self._main_log_dir

    @property
    def group_dir(self) -> Path:
        """Directory for experiment group."""
        return self._group_dir

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for model checkpoints."""
        return self._checkpoints_dir
    
    @property
    def video_dir(self) -> Path:
        """Directory for recorded videos."""
        return self._video_dir
    
    @property
    def log_filepath(self) -> Path:
        """Path to experiment log file."""
        return self._log_filepath
    
    @property
    def note_filepath(self) -> Path:
        """Path to experiment notes."""
        return self._note_filepath
    
    @property
    def params_filepath(self) -> Path:
        """Path to parameter file."""
        return self._params_filepath
    
    @property
    def plot_filepath(self) -> Path:
        """Path to plot output file."""
        return self._plot_filepath

    @property
    def user_params_filepath(self) -> Path:
        return self._user_params_filepath
