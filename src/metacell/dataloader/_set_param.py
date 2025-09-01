import os
import json
import pandas as pd

# === load config.json to set parameter. ===
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    merged_config = {}
    for key in config:
        merged_config.update(config.get(key, {}))
    return merged_config

# === check config dictionary ===
def check_config(config_dict: dict):
    """
    Description：
    ------------
    check config dictionary

    Parameter:
    ----------
    config_dict: config dictionary

    Return:
    -------
    None
    """
    # ===== Paths  =====
    # To determine if there are files with the suffixes .txt or .mzML in the input path.
    # 1.Check if the directory path config_dict['input_path'] exists.
    if not os.path.exists(config_dict['input_path']):
        raise FileNotFoundError(f"Directory {config_dict['input_path']} does not exist.")
    file_type = ['txt', 'mzML']
    all_files = set()
    for ext in file_type:
        all_files.update([
            os.path.join(config_dict['input_path'], file)
            for file in os.listdir(config_dict['input_path'])
            if file.endswith(f'.{ext}')
        ])
    if not all_files:
        raise FileNotFoundError(f"No files with {file_type} suffix found in {config_dict['input_path']}.")
    all_files_list = list(all_files)

    # 2.Check and output the filenames that have both .txt and .mzML extensions.
    for file_path in all_files_list:
        filename = os.path.basename(file_path)
        if filename.endswith('.txt') and (
                filename.replace('.txt', '.mzML') in [os.path.basename(file_paths) for file_paths in
                                                      all_files_list]):
            print(
                f"Warning: Both '{filename}.txt' and '{filename}.mzML' exist. Only '{filename}.txt' will be used.")
            all_files.remove(file_path.replace('.txt', '.mzML'))
    config_dict['all_files'] = list(all_files)

    # 3.Check if the file path config_dict['MAL_path'] exists.
    if os.path.exists(config_dict['MAL_path']):
        # Read the xlsx file.
        df = pd.read_excel(config_dict['MAL_path'])
        # Check if it contains columns named 'Metabolites_name' and 'Theoretical_value'.
        if 'Metabolites_name' not in df.columns or 'Theoretical_value' not in df.columns:
            raise ValueError(
                "The file should contain columns named 'Metabolites_name' and 'Theoretical_value'.")
    else:
        config_dict['MAL_path'] = None

    # 4.Check if config_dict['excluded_data_path'] exists, then read the txt file.
    if config_dict['excluded_data_path'] is not None:
        if os.path.exists(config_dict['excluded_data_path']):
            # Read the txt file
            poor_signal_range = pd.read_excel(config_dict['excluded_data_path'], names=['filename', 'start', 'end'])

            # Check if 'start' is greater than 'end' for each row
            errors = poor_signal_range[poor_signal_range['start'] > poor_signal_range['end']]

            # Raise an error if any incorrect data is found
            if not errors.empty:
                raise ValueError(
                    "An error occurred while reading or checking the excluded_data file: There are rows where 'start' is greater than 'end'. Please check the data.")
        else:
            # If the file does not exist, set it to None
            config_dict['excluded_data_path'] = None
    else:
        config_dict['excluded_data_path'] = None

    # 5.Check if config_dict['cell_type_mark_path'] exists, then read the file.
    if os.path.exists(config_dict['cell_type_marker_path']):
        # Read the xlsx file.
        df = pd.read_excel(config_dict['cell_type_marker_path'])
        # Check if it contains columns named 'Metabolites_name' and 'Theoretical_value'.
        if 'marker_name' not in df.columns or 'mz' not in df.columns or 'color_code' not in df.columns:
            raise ValueError(
                "The file should contain columns named 'marker_name', 'mz' and 'color'.")
    else:
        config_dict['cell_type_marker_path'] = None

    # ===== Parameters =====
    # 1.Check if the specific key in config_dict is a positive integer.
    keys_to_int = ['offset', 'interval', 'ppm_threshold']
    for key in keys_to_int:
        if not isinstance(config_dict[key], int) or config_dict[key] <= 0:
            raise ValueError(f"{key} should be a positive integer.")

    # 2.Check if the specific key in config_dict is a positive float.
    keys_to_float = ['sn_ratio', 'sn_ratio_lif', 'signal_threshold_ratio']
    for key in keys_to_float:
        if not isinstance(config_dict[key], float) or config_dict[key] <= 0:
            raise ValueError(f"{key} should be a positive float.")

    # 3.Check the range of values for config_dict['save_file_type'].
    if config_dict['save_file_type'] not in ['csv', 'xlsx']:
        raise ValueError("config_dict['save_file_type'] should be either 'csv' or 'xlsx'.")

    # 4.Check if config_dict['cell_marker'] is None.
    if not isinstance(config_dict.get('cell_marker'), dict) or not config_dict['cell_marker']:
        raise ValueError("The 'cell_marker' key in config_dict must be a non-empty dictionary.")

    # 5.Check if config_dict['main_cell_marker'] is a key in config_dict['cell_marker'].
    if config_dict['main_cell_marker'] not in config_dict.get('cell_marker', {}):
        raise KeyError("'main_cell_marker' is not a key in the 'cell_marker' dictionary.")

    # Print the parameters in a more prominent way
    print("The program will run with the following configuration settings:")
    print("-" * 50)  # Separator line to make the section stand out
    for key, value in config_dict.items():
        print(f"{key}: {value}")
    print("-" * 50)  # Separator line to make the section stand out

    return config_dict

