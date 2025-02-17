"""
Utility functions for data handling in PTR experiments
@author: Lies Hadjadj
All rights reserved
"""

import pickle
from pathlib import Path
from typing import Any


class DataManager:
    """Class for handling data serialization and loading"""

    def __init__(self, base_dir: str = "../../obj"):
        """Initialize with base directory for data storage
        
        Args:
            base_dir: Base directory path for data files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_obj(self, obj: Any, name: str) -> None:
        """Save object to pickle file
        
        Args:
            obj: Object to save
            name: Name of the file (without extension)
        """
        file_path = self.base_dir / f"{name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name: str) -> Any:
        """Load object from pickle file
        
        Args:
            name: Name of the file (without extension)
            
        Returns:
            Loaded object
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.base_dir / f"{name}.pkl"
        with open(file_path, "rb") as f:
            return pickle.load(f)
