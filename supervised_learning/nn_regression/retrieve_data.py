"""

Code for retrieving experiment data and paper data. Used in jupyter notebooks
for post-processing analysis of folders labeled "experiment_##".

Updated: 2022-04-21

"""
import os
from ae_functions import get_folder_pickle_files
import pickle
from model_architectures import NeuralNetwork_01, NeuralNetwork_02
import torch

def load_experiment_results(experiment_num):
    """
    
    Retrieve data from "experiment_##" folders. Each experiment folder is 
    created upon running a test. Within each folder are pickle files which 
    store the parameters of the test in addition to the output metrics, and
    feature data. 

    Parameters
    ----------
    experiment_num : int
        This value corresponds to the subdirectory folder to extract exp from.

    Returns
    -------
    data : dict
        Dictionary of the parameters for single pickle file.

    """
    experiment_num = str(experiment_num).zfill(2)
    pickle_files = get_folder_pickle_files('.\experiment_'+experiment_num)
    experiment_folder = os.getcwd() + "/experiment_" + experiment_num + "/" 
    
    print(f"Retrieved data ({len(pickle_files)} pickle files)",
          f"from {experiment_folder}")
    
    for pickle_file in pickle_files:
        with open(experiment_folder + pickle_file, "rb") as file:
            data = pickle.load(file)
    
    return  data

def load_model(pth_path, model_num, feature_dim, num_classes=5):
    """
    
    Loads up a trained model.

    Parameters
    ----------
    pth_path : str
        File path to pth, ex: './experiment_01/10_3000_0.001_adam_mse.pth'.
    model_num : int
        Specific model to load up, refer to model architectures for types.
    feature_dim : int
        Dimension of input, can be acquired from associated pickle file.
    num_classes : int, optional
        Dimension of output, associated with classification. The default is 5.

    Returns
    -------
    model : torch object
        Loaded model with trained parameters acquired from pth file.

    """
    if model_num == 1: # classification
        model = NeuralNetwork_01(feature_dim,num_classes)
    elif model_num == 2: # regression
        model = NeuralNetwork_02(feature_dim)
        
    model.load_state_dict(torch.load(pth_path))
    
    print(model)
    
    return model


    
