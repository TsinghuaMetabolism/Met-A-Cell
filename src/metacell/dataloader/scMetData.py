import os
import pandas as pd
import dill
from pathlib import Path
from typing import Union
from metacell.dataloader._logger import setup_logger


class scMetData(object):
    def __init__(self, file: Union[Path, str], filename=None):
        self.file = file
        if filename is not None:
            self.filename = filename
        else:
            self.filename = os.path.basename(file)

        self.logger, self.memory_handler = setup_logger()

        self.mz_data = {}
        self.intensity_data = {}
        self.raw_scm_data = pd.DataFrame()
        self.cell_marker_eic = {}
        self.scm_events_index = {}
        self.scm_events = pd.DataFrame()
        self.cell_feature_matrix = pd.DataFrame()

        self.cell_marker = {}
        self.main_cell_marker = ""

        self.cell_type_marker_df = pd.DataFrame()
        self.cell_type_marker_eic = {}
        self.cell_type_marker_apex_index = {}

        self.cell_channel_df = pd.DataFrame()

        self.scm_type = ""
        # self.lif_data_dir = {}
        # self.lif_peak_data_dir = {}
        self.processing_status = {
            'scm_events_extraction_strategy': None,
            'feature_extraction_strategy': None
        }

    def show_processing_status(self):
        print("Current processing status:")
        if self.processing_status['scm_events_extraction_strategy'] is not None:
            print(f"SCM events extraction strategy: {self.processing_status['scm_events_extraction_strategy']}")
        else:
            print("No single-cell event extraction has been performed.")

        if self.processing_status['feature_extraction_strategy'] is not None:
            print(f"Feature extraction strategy: {self.processing_status['feature_extraction_strategy']}")
        else:
            print("No feature extraction has been performed.")
    # 当调用 mdata.scm_events['marker']时，返回 mdata.raw_scm_data[scm_events[marker]]

    def get_processing_status(self):
        log = "Current processing status:\n"
        if self.processing_status['scm_events_extraction_strategy'] is not None:
            log += f"SCM events extraction strategy: {self.processing_status['scm_events_extraction_strategy']}\n"
        else:
            log += "No single-cell event extraction has been performed.\n"

        if self.processing_status['feature_extraction_strategy'] is not None:
            log += f"Feature extraction strategy: {self.processing_status['feature_extraction_strategy']}\n"
        else:
            log += "No feature extraction has been performed.\n"
        return log

    def save(self, path: Union[str, Path]):
        """
        Save the scMetData object to a file using dill.

        Parameters:
        - path: The file path to save the object (should end in .pkl or .dill)
        """
        with open(path, 'wb') as f:
            dill.dump(self, f)
        print(f"scMetData object saved to {path} using dill.")
    
    @staticmethod
    def load(path: Union[str, Path]):
        """
        Load a scMetData object from a file using dill.

        Parameters:
        - path: The file path to load the object from

        Returns:
        - scMetData object
        """
        with open(path, 'rb') as f:
            obj = dill.load(f)
        print(f"scMetData object loaded from {path} using dill.")
        return obj
    

