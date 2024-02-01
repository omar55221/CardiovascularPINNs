# Physics-Informed Neural Networks (PINNs) for Patient-Specific Cardiovascular Flows
## Introduction
Physics-Informed Neural Networks (PINNs) are a novel approach in the field of machine learning that integrate physical laws into the training process of neural networks. This integration is achieved by incorporating differential equations that govern physical processes directly into the loss function used to train the network. As a result, PINNs are capable of learning and making predictions that are consistent with known physical principles. This makes them particularly valuable for solving complex scientific and engineering problems where traditional data-driven models might struggle due to the scarcity of training data or the complexity of the underlying physical processes. By leveraging the structure and constraints provided by physical theories, PINNs can efficiently predict outcomes, simulate processes, and even discover new insights within fields such as fluid dynamics, material science, and climate modeling, among others.
Here, we coded a PINNs framework that uses Navier-Stroke equation, some known data points, and a boundary condition to simulate blood flow through 2D and 3D geometries. The architecture can be shown as:
![Figure_Methods_0](https://github.com/Owais-Khan/CardiovascularPINNs/assets/79001778/aaf9659c-991c-4fbc-a2dc-2b984ca70242)

## Python Files
### main.py
This file is the main code. All the hyperparameters are in this file. The hyperparameters are listed below:

Parameters for model and the neural network

Flag_GPU = 1                                                        # 0 use cpu and 1 use gpu and 2 use mps

Model_dim = 3                                                       # Model dimension 2D or 3D?

TimeVaryingModelFlag = 1                                            # if 0 that means the model is steady, if it is 1 that means the model is time-varying

ActivationFunctions = ['sinusResNet', 'sinus', 'swish', 'tanh']     # (Sinus+skip connections = sinusResNet) | sinus AF FF network = sinus | swish AF FF network = swish | tanh AF FF 
network = tanh

ActivationFunction = ActivationFunctions[3]                         # Choose one of the listed neural networks.

NumberOfSensorData = [200, 400, 600, 800, 1000, 1200, 1400, 1600]   # number of sensor points (you can add any number here)

if Model_dim == 2:                                                  # This is the number of sensor points for the 2D stenosis case that we used
    NumberOfSensorData = [25, 100, 225, 400]
    
sensor_num = NumberOfSensorData[7]                                  # Choose the number of sensor points you want

num_layers = 4                                                      # Choose the PINNs' number of layers

dim_hidden = 128                                                    # Choose the number of neurons per layer

w0 = 10                                                             # w0 hyperparameter is only for sinusoidal AF. W0=10 recommended for 3D Aorta and W0=30 recommended for 2D stenosis case.

epoches = 400                                                       # number of epochs

batchsize = 512                                                     # batch size

shuffle = True                                                      # If you want the input data to be shuffled enter True, if not enter False.

Parameters for the adaptive coefficient on PINNs loss function

Lambda = 0.9                                                        # smoothing factor for moving average on Lambda_bc and Lambda_data in Loss Function

Parameters of the Navier Stoke equation
Diff = 0.000452638                                                  # Diff - Navier Stroke Equation. Diff = 0.000452638 we used for our Aorta 3D case and Diff = 8e-6 for 2D for our stenosis case
Rho = 1.                                                            # Rho - Navier Stroke Equation.

Parameters for adaptive learning rate
Flag_schedule = True                                                # True: change the learning rate during training, False: constant learning rate
learning_rate = 1e-3                                                # Starting learning rate
step_epoches = 70                                                   # After how many epochs you want to change the LR
decay_rate = 0.1                                                    # The LR multiplies by this number after every step_epoches

Input and output paths
folder_path_velocity = 'Results_SimVascular_Coarse'                 # All the velocity files are in this folder for every time step. If the model is steady this folder contains only one file.
NameOfVelocityField = 'velocity'                                    # For 3D Aorta the velocity field is saved into 'velocity'. For 2D stenosis data is 'Assigned Vector Function'
vtu_files, directories = sort_vtuFiles_WallFolder(folder_path_velocity)

Parameters for Time Varying problem
sampling_rate = 5                                                   # What is the sampling rate when the network is time-varying? For the 3D Aorta case, it is 5, and for the 2D stenosis case it is 35.
SampleFileNumber = len(vtu_files)                                   # How many Velocity files do we have for time-varying problem
TimeOfSampling = 1.0                                                # time of time-varying data in seconds. It is 1 for 3D Aorta and 5.0 for 2D stenosis
TotalSampleOfData = 100                                             # How many files do we have on the whole data? It is 100 for 3D Aorta and 2500 for 2D stenosis
