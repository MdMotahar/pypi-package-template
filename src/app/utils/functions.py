import re
import gdown
import os
from app.utils.model_config import download_url
import re
import shutil
from pathlib import Path
from app.config import CACHE_DIR
from typing import Union, List, Tuple
from zipfile import ZipFile
import random
import numpy as np
import torch
    

def download_file_from_google_drive(fileid:str, filename:str) -> None:
    """
    Download a file from Google Drive using its file ID.
    
    Args:
        fileid (str): The Google Drive file ID.
        filename (str): The name of the file to save.
    
    Returns:
        None
    """
    gdown.download(fileid, filename, quiet=False)


def download_models()->Path:
    """
    Download and extract model files from a Google Drive link.
    
    This function downloads a ZIP file from a Google Drive link using the download_file_from_google_drive function,
    and then extracts its contents to a specified cache directory. The extracted model folder is detected and returned.
    
    Returns:
        Path: The path to the detected model folder.
    """
    cache_dir = CACHE_DIR
    # purge cache directory from the saved model files before downloading
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir,ignore_errors=False)
    else:
        os.makedirs(cache_dir)

    download_file_from_google_drive(download_url,os.path.join(cache_dir,'models.zip'))
    unzip_file(os.path.join(cache_dir,'models.zip'),os.path.join(cache_dir))

    #remove the zip file
    os.remove(os.path.join(cache_dir,'models.zip'))
    return detect_model_folder(cache_dir)


def unzip_file(file: Path, unzip_to: Path) -> None:
    """
    Unzip a file to a specified directory.
    
    Args:
        file (Path): The path to the ZIP file.
        unzip_to (Path): The path to the directory to unzip to.
    
    Returns:
        None
    """
    with ZipFile(file, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(unzip_to)

def detect_model_folder(model_cache_dir:Union[str,Path])->Union[str,Path]:
    """
    Detect the newly created model folder inside the model_cache_dir.
    
    This function detects the newly created model folder inside the provided model_cache_dir after zip extraction.
    It returns the path to the detected model folder.
    
    Args:
        model_cache_dir (Union[str, Path]): The path to the model cache directory.
    
    Returns:
        Path: The path to the detected model folder.
        
    """
    if isinstance(model_cache_dir,str):
        model_cache_dir = Path(model_cache_dir)

    all_files = list(model_cache_dir.glob("*"))
    all_paths = [str(i.absolute()) for i in all_files]
    for path in all_paths:
        if os.path.isdir(path):
            model_cache_dir = path
            break

    if isinstance(model_cache_dir,str):
        model_cache_dir = Path(model_cache_dir)

    return model_cache_dir
