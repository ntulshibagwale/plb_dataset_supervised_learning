"""

acoustic_emission_dataset

Loads in waveform files and performs any transforms on the data. To be used 
with pytorch data loaders.

Nick Tulshibagwale

Updated: 2022-04-07

"""
import torch
from torch.utils.data import Dataset
from ae_measure2 import load_PLB
from torch import tensor
from ae_measure2 import wave2vec

class AcousticEmissionDataset(Dataset):
    
    # Constructor
    def __init__(self,json_data_file,sig_len,dt,low_pass,high_pass,fft_units,
                 num_bins,n_fft,hop_length):
      
        self.json_data_file = json_data_file
        self.sig_len = sig_len
        self.dt = dt
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.fft_units = fft_units
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bins = num_bins
        
        # Load in AE Data
        data = load_PLB(json_data_file)
        
        # Separate dict into arrays
        waves = data['data']           # List of raw waveforms
        targets = data['target']       # waveform labels
        self.angles = data['target_angle']  # angle, one hot encoded
        targets = tensor(targets,dtype=torch.int64,requires_grad=False)
        targets = torch.nn.functional.one_hot(targets.long())    
        self.targets = targets.float()
        self.n_samples = self.targets.shape[0]    # Number of samples/labels
        self.waves = tensor(waves,dtype=torch.float32,requires_grad=False)                          

        print(f"Shape of waves is: {self.waves.shape}")
        print(f"Datatype of waves is: {self.waves.dtype}")
        print("waves requires grad:", self.waves.requires_grad)
        print(f"Shape of targets is: {self.targets.shape}")
        print(f"Datatype of targets is: {self.targets.dtype}")
        print("targets requires grad:", self.targets.requires_grad)
        print("")
        
    def __getitem__(self,index):
        """
        
        Function called when object is indexed. The transformation of data 
        occurs in this getter function. In other words, the constructor reads
        in the raw data filtered by hand, and this function contains the 
        sequence of remaining processing on the data to extract features used
        in ML models.
        
        index (int): the index of the sample and label
        
        return:
        (x,y): feature vector and label corresponding to single event
        
        """
        x = self.waves[index]      
        y = self.targets[index]
        
        # Perform transformations here
        # i.e. spectrogram, partial power, fft etc..
        #x = x.numpy()
        #x, freq_bounds, spacing = wave2vec(self.dt, x, self.low_pass,
        #                                                self.high_pass,
        #                                                self.num_bins,
        #                                                self.fft_units)
        #x = tensor(x,dtype=torch.float32,requires_grad=False)
        
        return x, y # input example, label
    
    def __len__(self): # number of samples
        return self.n_samples
  
