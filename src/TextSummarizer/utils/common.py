import os
from box.exceptions import BoxValueError
import yaml
from src.TextSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


# 1) Read yaml module
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """ reads yaml file and returns 
    Args:
         Path_to_yaml (str): Path like input
    
    Raises:
           valueError: if file is empty
           e: empty file
           
    Returns:
            ConfigBox: ConfigBox type
    """
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml_file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
# 2) Create directories module

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """ create a list of directories
    
    Args:
         Path_to_directories (list): list of path of directories
         ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to false.
    """
    
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
            

# 3) Getsize module

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in kb
    Args: 
         path (Path): path of the file
         
    Returns:
            str: size in kb
    """
    
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"               