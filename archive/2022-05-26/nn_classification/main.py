"""

main

Program Flow:
1. Loads in AE data
2. Loads in model
3. Trains model
4. Evaluates model accuracy on test set
4. Saves log, model results, parameters, etc. to working directory

User should move files to separate folder, zip code used to generate data.

NB: Nothing will be output from console, all goes to log .txt file.

Nick Tulshibagwale

Updated: 2022-05-25

"""
import numpy as np
from acoustic_emission_dataset import AcousticEmissionDataset
from model_architectures import NeuralNetwork_01
import pickle
import sys 
import datetime
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# SIGNAL PROCESSING CONSTANTS
SIG_LEN = 1024           # [samples / signal] ;
DT = 10**-7              # [seconds] ; sample period / time between samples
LOW_PASS = 0#50*10**3    # [Hz] ; low frequency cutoff
HIGH_PASS = 100*10**4    # [Hz] ; high frequency cutoff
FFT_UNITS = 1000         # FFT outputs in Hz, this converts to kHz
NUM_BINS = 26            # For partial power

# ML HYPERPARAMETERS
EPOCHS = 1000            # training iterations
LEARNING_RATE = 1e-3     # step size for optimizer
BATCH_SIZE = 20          # for train and test loaders
ARCHITECTURE = 1
# NB: To vary autoencoder architecture, must do in class definition file

# FILE I/O
JSON_DATA_FILE = 'E:/file_cabinet/phd/projects/plb_dataset_supervised_learning/Data/220426_PLB_data.json'

if __name__ == "__main__":
    
    # Make a log for output statements (errors will not output here however)
    now = datetime.datetime.now()
    time_stamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
    stdoutOrigin=sys.stdout 
    sys.stdout = open(time_stamp+"_log.txt", "w")
    print(f"{time.ctime()}\n")
    
    # Load AE data
    ae_dataset = AcousticEmissionDataset(JSON_DATA_FILE,SIG_LEN,DT,LOW_PASS,
                                         HIGH_PASS,FFT_UNITS,NUM_BINS,0,0)
    angles = ae_dataset.angles # what the one hot encoded targets map to
    num_classes = len(angles)  # how many diff angles, for model output dim
    example_feature_vec, _ = ae_dataset[0] # to determine feature dim
    feature_dim = example_feature_vec.shape[0] # for model creation input dim
    
    # Separate data into training data and test data  
    total_count = len(ae_dataset)
    train_percent = 0.80
    test_percent = 1 - train_percent
    train_count = int(train_percent * total_count)
    test_count = total_count - train_count
    train_seed = 41 # THIS ENSURES SAME SPLIT FOR TEST DATA EVERY RUN
    torch.manual_seed(train_seed) 
    train_data, test_data = torch.utils.data.random_split(ae_dataset,
                                                          (train_count,
                                                            test_count))
    
    # Create data loaders
    test_data_loader = DataLoader(test_data)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                   shuffle=True) # shuffles training data
    
    # Store parameters in dictionary for log
    model_data = { 
        'time_stamp'    : time_stamp,
        'SIG_LEN'       : SIG_LEN,
        'DT'            : DT,
        'LOW_PASS'      : LOW_PASS,
        'HIGH_PASS'     : HIGH_PASS,
        'FFT_UNITS'     : FFT_UNITS,
        'NUM_BINS'      : NUM_BINS,
        'EPOCHS'        : EPOCHS,
        'LEARNING_RATE' : LEARNING_RATE,
        'BATCH_SIZE'    : BATCH_SIZE,
        'train_seed'    : train_seed, 
        'train_percent' : train_percent,
        'test_percent'  : test_percent,
        'total_count'   : total_count,
        'train_count'   : train_count,
        'test_count'    : test_count,
        'angles'        : angles,
        'feature_dim'   : feature_dim,
        'num_classes'   : num_classes
        }
    
    print("Parameters:")
    for key,value in model_data.items():
    	print(key, ':', value)
    print("")
    
    # Look at shapes of batches
    for ex_idx, (batch_x,batch_y) in enumerate(train_data_loader):
        print(f"Shape of x batch is: {batch_x.shape}")
        print(f"Datatype of x batch is: {batch_x.dtype}\n")
        print(f"Shape of y batch is: {batch_y.shape}")
        print(f"Datatype of y batch is: {batch_y.dtype}\n")
        break
    
    # Training Model       
    print('------------------------------------------------------------------')
    print("Begin model training...\n")
    
    # Create model
    model = NeuralNetwork_01(feature_dim, num_classes) #.to(device)       
    print(model)
    print("")
    
    # Optimizer Algorithm
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    # Loss Function
    loss_func = torch.nn.MSELoss()
    loss_history = []
    show_every = 50
    
    # Loop over training data and optimize model, record loss
    print("Begin Training...\n")
    for i in range(EPOCHS+1):
        for step, (batch_x,batch_y) in enumerate(train_data_loader):
            predictions = model(batch_x)  # Perform forward pass
            loss = loss_func(predictions, batch_y) # Compute loss 
            optimizer.zero_grad() # clear gradients
            loss.backward()       # compute loss gradients wrt parameters
            optimizer.step()      # apply gradients to optimizer
        if i % show_every ==0: # output loss
            with torch.no_grad(): # turn off gradient tracking 
                total_loss = 0 # sum over all batches
                for step, (batch_x,batch_y) in enumerate(train_data_loader):
                    predictions = model(batch_x)
                    loss = loss_func(predictions, batch_y)
                    total_loss = total_loss+loss
                print("Testing loss at Epoch # ", i , "is : ",
                      total_loss.item())
                loss_history.append(total_loss) # record loss
    
    model_data['Loss'] = np.array(loss_history) # save training loss
    model_data['show_every'] = show_every # training loss freq
    print("Training completed.\n")
       
    print('------------------------------------------------------------------')
    
    print("Begin model evaluation...\n")
    print("\nEvaluate results on test data...\n")
    
    # Evaluate model accuracy on test data set
    model.eval()
    with torch.no_grad():
        test_predicted = []
        test_actual = []
        total_examples = 0
        total_correct = 0
        for inputs, targets in test_data_loader:
            prediction = model(inputs) # Tensor (1,3) -> ex. [0.1, 0.01,0.6]
            prediction = prediction[0].argmax()
            actual = targets[0].argmax()
            test_predicted.append(prediction) # ex. [0,0,1]
            test_actual.append(actual)
            total_examples = total_examples + 1
            if prediction == actual:
                total_correct = total_correct + 1
            print(f"class_prediction= {prediction} , target={actual}")  
    model_data['test_predicted'] = test_predicted
    model_data['test_actual'] = test_actual
    model_data['test_accuracy'] = 100 * total_correct/total_examples
    
    print("")
    print("Test Set Metrics:")
    model_data.update(classification_report(test_predicted,test_actual,
                                target_names=angles,output_dict=True))
    print(classification_report(test_predicted,test_actual,
                                target_names=angles))
    
    # Saving trained model parameters so it can be reloaded
    model_name = 'nn'+str(ARCHITECTURE).zfill(2) + '_' + str(EPOCHS) + '_' + \
        str(LEARNING_RATE) + '_' + 'adam_mse' 
    torch.save(model.state_dict(), model_name + '.pth')
    print("\nSaved PyTorch Model State to " + model_name + ".pth")
    print("")
    
    # Save data (input parameters, results) associated with trained model 
    f = open(model_name + ".pkl","wb")
    pickle.dump(model_data,f)
    f.close()
    print("Saved data to " + model_name + ".pkl")
    print("")
    print("Completed training and evaluation.\n")
        
    # Close log file
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
    
