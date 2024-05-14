from abc import abstractmethod
from pathlib import Path

class Experiment: 
    def __init__(self, name, display_name, create_folders = True):
        self.name = name
        self.display_name = display_name
        self.scores = None

        if not create_folders: 
            return

        # Create the folders in the current save path
        Path(f"data/saved_models/{self.name}/").mkdir(parents=True, exist_ok=True)
        Path(f"data/figures/{self.name}/").mkdir(parents=True, exist_ok=True)
        Path(f"data/results_data/{self.name}/").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self, arguments): 
        raise NotImplementedError("Run is not implemented")
