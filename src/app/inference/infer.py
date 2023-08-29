from pathlib import Path
import os
import app
from app.utils.functions import load_cached_file
from app.utils.download_config import MODEL_URL_MAP
from typing import Union,List,Tuple

class Infer:
    """
    Attributes:
        model_name (str): Name of the pre-trained NER model.
        infer_class (Infer_LDA): Inference class for lda or other topic model.
        model_cache_dir (Path): Path to the cached model files.

    Methods:
        __init__(self, model_name: str = None,force_redownload: bool = False) -> None:
            Initializes the Infer class instance.
            
        infer(self, texts: List[str] = '...') -> dict:
            Perform topic modelling on the input texts.
    """
    def __init__(self,
                 model_name: str = None,
                 force_redownload: bool = False,
                 ) -> None:
        """
        Initializes the Infer class instance.

        Args:
            model_name (str): Name of the pre-trained NER model.
            pretrained_model_path (str): Path to a custom pre-trained model directory.
            force_redownload (bool): Flag to force redownload if the model is not cached.
        """
        self.model_name = model_name
        cache_dir = Path(app.CACHE_DIR)
        model_cache_dir = cache_dir / f'{model_name}'

        if self.model_name == 'MODEL_NAME':
            self.infer_class = MODEL_CLASS

        if model_name not in MODEL_URL_MAP and not os.path.exists(model_cache_dir):
            raise Exception('model name not found in download map and no model cache directory found')
        
        # load the model files from cache directory or force download and save in cache
        self.model_cache_dir = load_cached_file(model_name,force_redownload=force_redownload)
        self.infer_class = self.infer_class(self.model_cache_dir)
             

    def infer(self,texts: List[str]) -> dict:
        """
        Perform topic modelling on the input texts.

        Args:
            texts (List[str]): Input texts for topic modelling.

        Returns:
            dict: Dictionary containing NER inference results.
        """
        result = self.infer_class.infer(texts)
        return result


