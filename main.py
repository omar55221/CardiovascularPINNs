import numpy as np

from ReadMeshComplete import *
import vtk
from matplotlib import pyplot as plt
from utilities import *
from openpyxl.workbook import Workbook
import torch
import os
from train import *
import itertools
from numpy import arange

# Parameters for model and the neural network
Flag_GPU = 1                                                        # 0 use cpu and 1 use gpu and 2 use mps
Model_dim = 3                                                       # Model dimension 2D or 3D?
TimeVaryingModelFlag = 1                                            # if 0 that means model is steady, if it is 1 that means the model is time-varying
ActivationFunctions = ['sinusResNet', 'sinus', 'swish', 'tanh']     # (Sinus+skip connections = sinusResNet) | sinus AF FF network = sinus | swish AF FF network = swish | tanh AF FF network = tanh
ActivationFunction = ActivationFunctions[3]                         # Choose the network?
NumberOfSensorData = [200, 400, 600, 800, 1000, 1200, 1400, 1600]   # number of sensor points (you can add any number here)
if Model_dim == 2:                                                  # This is the number of sensor points for the 2D stenosis case
    NumberOfSensorData = [25, 100, 225, 400]
sensor_num = NumberOfSensorData[7]                                  # choose the number of sensor points you want
num_layers = 4                                                      # choose the PINNs number of layers of
dim_hidden = 128                                                    # choose the number of neurons per layer
w0 = 10                                                             # w0 hyperparameter only for sinusoidal AF. W0=10 recommended for 3D Aorta and W0=30 recommended for 2D stenosis case.
epoches = 400                                                       # number of epochs
batchsize = 512                                                     # batch size
shuffle = True                                                      # If you want the input data to be shuffled enter True, if not enter False

# Parameters for adaptive coefficent on PINNs loss function
Lambda = 0.9                                                        # smoothing factor for moving average on Lambda_bc and Lambda_data in Loss Function

# Parameteres of the Navier Stoke equation
Diff = 0.000452638                                                  # Diff - Navier Stroke Equation. Diff = 0.000452638 for Aorta 3D case and Diff = 8e-6 for 2D stenosis case
Rho = 1.                                                            # Rho - Navier Stroke Equation.

# Parameters for adaptive learning rate
Flag_schedule = True                                                # True=change the learning rate during training, False = constant learning rate
learning_rate = 1e-3                                                # Starting learning rate
step_epoches = 70                                                   # After how many epochs you want to change the LR
decay_rate = 0.1                                                    # The LR multiplies by this number after every step_epoches

# Input and output paths
folder_path_velocity = 'Results_SimVascular_Coarse'      # All the velocity files are in this folder for every time step. If the model is steady this folder contains only one file.
NameOfVelocityField = 'velocity'                                    # For 3D Aorta the velocity field is saved into 'velocity'. For 2D stenosis data is 'Assigned Vector Function'
#folder_path_velocity = 'MeshFolder2D'
#NameOfVelocityField = 'Assigned Vector Function'
vtu_files, directories = sort_vtuFiles_WallFolder(folder_path_velocity)

# Parameters for Time Varying problem
sampling_rate = 5                                                   # What is the sampling rate when the network is time varying. For 3D Aorta case it is 5, and for 2D stenosis case it is 35
#sampling_rate = 35
SampleFileNumber = len(vtu_files)                                   # How many Velocity file do we have for time varying problem
TimeOfSampling = 1.0                                                # time of time-varying data in second. It is 1 for 3D Aorta and 5.0 for 2D stenosis
#TimeOfSampling = 5.0
TotalSampleOfData = 100                                             # How many file do we have on the whole data. It is 100 for 3D Aorta and 2500 for 2D stenosis
#TotalSampleOfData = 2500

if TimeVaryingModelFlag == 1:
    NumberOfInputs = Model_dim + 1
else:
    NumberOfInputs = Model_dim
    SampleFileNumber = 1
############################
# Iterate over all files and directories in the folder
print("--- Reading the Mesh, Wall, Inlet and Outlet Coordinates")
if Model_dim == 3:
    x, y, z, T, xb_wall, yb_wall, zb_wall, T_walls, Sensor_coord_x, Sensor_coord_y, Sensor_coord_z, T_sensors, data_vel_u, data_vel_v, data_vel_w, NumberOfMechCoordinates, MeshCompleteVTU = Read_Input_3D_Data(SampleFileNumber, folder_path_velocity, vtu_files, NameOfVelocityField, sensor_num, sampling_rate, TotalSampleOfData, TimeOfSampling)

if Model_dim == 2:
    x, y, T, xb_wall, yb_wall, T_walls, Sensor_coord_x, Sensor_coord_y, Sensor_coord_z, T_sensors, data_vel_u, data_vel_v, NumberOfMechCoordinates, MeshCompleteVTU = Prepare_2D_stenosis_Data(folder_path_velocity, vtu_files, sensor_num, NameOfVelocityField, SampleFileNumber, sampling_rate, TotalSampleOfData, TimeOfSampling)
    z = np.zeros(x.shape)

x_data = Sensor_coord_x
y_data = Sensor_coord_y
z_data = Sensor_coord_z

# Define boundry Condition
print ("--- Defining Boundary Conditions of Zero for the Wall")
u_wall_BC = np.linspace(0., 0., len(xb_wall))                                #wall boundry condition in direction u
v_wall_BC = np.linspace(0., 0., len(xb_wall))                              #wall boundry condition in direction v
if Model_dim == 3:
    w_wall_BC = np.linspace(0., 0., len(xb_wall))                              #wall boundry condition in direction w
u_wall_BC = u_wall_BC.reshape(-1, 1) #need to reshape to get 2D array
v_wall_BC = v_wall_BC.reshape(-1, 1) #need to reshape to get 2D array
if Model_dim == 3:
    w_wall_BC = w_wall_BC.reshape(-1, 1) #need to reshape to get 2D array

if Model_dim == 3:
    print('--- Shape of coordinates in the wall boundry: x{} y{} z{}'.format(xb_wall.shape, yb_wall.shape, zb_wall.shape))
if Model_dim == 3:
    print('--- Shape of velocity of the wall boundry (BC) in direction of u{} , v{} , w{})'.format(u_wall_BC.shape, v_wall_BC.shape, w_wall_BC.shape))
if Model_dim == 2:
    print('--- Shape of coordinates in the wall boundry: x{} y{}'.format(xb_wall.shape, yb_wall.shape))
if Model_dim == 2:
    print('--- Shape of velocity of the wall boundry (BC) in direction of u{} , v{} )'.format(u_wall_BC.shape, v_wall_BC.shape))

## Define hyper parameters

if not os.path.isdir("result/"+ActivationFunction+"_layern%d"%num_layers+"_hiddenlayern%d"%dim_hidden+"_SensNum%d"%sensor_num+"_Rho%d"%Rho+"_Diff"+str(Diff)+"_w0siren"+str(w0)+"/"):
    os.makedirs("result/"+ActivationFunction+"_layern%d"%num_layers+"_hiddenlayern%d"%dim_hidden+"_SensNum%d"%sensor_num+"_Rho%d"%Rho+"_Diff"+str(Diff)+"_w0siren"+str(w0)+"/")

path_NetWeights = "result/"+ActivationFunction+"_layern%d"%num_layers+"_hiddenlayern%d"%dim_hidden+"_SensNum%d"%sensor_num+"_Rho%d"%Rho+"_Diff"+str(Diff)+"_w0siren"+str(w0)+"/"		# saving network

if Model_dim == 2:
    plt.figure()
    plt.scatter(x, y, cmap='rainbow')
    plt.scatter(x_data, y_data, cmap = 'red')
    plt.title('Sensor location')
    plt.colorbar()
    plt.savefig(path_NetWeights+'sensor_location.png')
    plt.figure()
if Model_dim == 3:
    plt.figure()
    plt.scatter(x, y, cmap='rainbow')
    plt.scatter(x_data, y_data, cmap = 'red')
    plt.title('Sensor location')
    plt.colorbar()
    plt.savefig(path_NetWeights+'sensor_location.png')
    plt.figure()

if not os.path.isfile(path_NetWeights+"loss.xlsx"):
    headers= ['Loss_eqn', 'Loss_BC', 'Loss_Data', 'Loss_total', 'Time']
    workbook_name = path_NetWeights+'loss.xlsx'
    wb = Workbook()
    page = wb.active
    page.title = 'Siren-Based-3Daorta'
    page.append(headers) # write the headers to the first line
    wb.save(filename=workbook_name)

# ---------------------------------------------
# working on GPU or CPU
if Flag_GPU == 0:
    device = torch.device("cpu")
    processor = "cpu"
if Flag_GPU == 1:
    device = torch.device("cuda:0")
    processor = "cuda:0"
if Flag_GPU == 2:
    device = torch.device("mps")
    processor = "mps"

if TimeVaryingModelFlag == 0:  # make T if the model is steady state to avoid error
    T_walls = np.zeros(shape=xb_wall.shape)
    T_sensors = np.zeros(shape=x_data.shape)
    T = np.zeros(shape=x.shape)
if Model_dim == 2:
    InputParameters={"device":          device,
            "processor":                processor,
            "dim":                      Model_dim,
            "NumberOfMechCoordinates":  NumberOfMechCoordinates,           # How many MechCoordinates do we have
            "NumberOfInputs":           NumberOfInputs,                    # How many inputs do we have for neural network x , y, z ,T
            "MeshCompleteVTU":          MeshCompleteVTU,                   #Mesh volume file
            "xyz":                      [x,y,z],                           #Mesh Coordinates
            "xyzb_wall":                [xb_wall,yb_wall],                 #Wall Coordinates
            "uvw_wall_BC":              [u_wall_BC,v_wall_BC],             #Boundary Conditions (Velocity=0)
            "xyz_data":                 [x_data,y_data],                   #Sensor Coordinates
            "data_vel":                 [data_vel_u,data_vel_v],           #Sensor Velocities
            "batchsize":                batchsize,                         #Batch Size, the number of data to show the network in a single iteration
            "learning_rate":            learning_rate,                     #learning rate of the network
            "decay_rate":               decay_rate,                        #decay rate of the network
            "epoches":                  epoches,                           #number of epoches
            "step_epoches":             step_epoches,                      #the epoches at which to decrease the learning rate
            "Flag_schedule":            Flag_schedule,                     #Whether to use decreased learning rate or constant
            "Diff":                     Diff,                              #Differene to compute the learning rate
            "rho":                      Rho,                               #The density of the fluid
            "Lambda":                   Lambda,                            #Coefficent factor of boundary condition in the loss function
            "Path_NetWeights":          path_NetWeights,                   #Path for saving the weights
            "ActivationFunction":       ActivationFunction,                # Which AF neural net
            "NumberOfLayers":           num_layers,                        # Number of layers
            "NumberOfHiddenNeurons":    dim_hidden,                        # Number of Neurons in each layer
            "W0_Siren":                 w0,                                # W0 hyperparameter of Siren
            "Time":                     T,                                 # Time of the model
            "Time_walls":               T_walls,                           # Time of the model with the shape of walls inputs
            "Time_data":                T_sensors,                         # Time of the model with the shape of sensor data inputs
            "sampling_rate":            sampling_rate,                     # What is Sampling rate?
            "NumberOfSampleFiles":      SampleFileNumber,                  # How many sample file do we have
            "shuffle":                  shuffle,                           # Input data should be shuffled or not?
            "input_files":              folder_path_velocity,              # Input data address
            "vtu_files":                vtu_files
            }

if Model_dim == 3:
    InputParameters={"device":          device,
            "processor":                processor,
            "dim":                      Model_dim,
            "NumberOfMechCoordinates":  NumberOfMechCoordinates,           # How many MechCoordinates do we have
            "NumberOfInputs":           NumberOfInputs,                    # How many inputs do we have for neural network x , y, z ,T
            "MeshCompleteVTU":          MeshCompleteVTU,                   #Mesh volume file
            "xyz":                      [x, y, z],                             #Mesh Coordinates
            "xyzb_wall":                [xb_wall, yb_wall, zb_wall],                 #Wall Coordinates
            "uvw_wall_BC":              [u_wall_BC, v_wall_BC, w_wall_BC],             #Boundary Conditions (Velocity=0)
            "xyz_data":                 [x_data, y_data, z_data],                   #Sensor Coordinates
            "data_vel":                 [data_vel_u, data_vel_v, data_vel_w],           #Sensor Velocities
            "batchsize":                batchsize,                         #Batch Size, the number of data to show the network in a single iteration
            "learning_rate":            learning_rate,                     #learning rate of the network
            "decay_rate":               decay_rate,                        #decay rate of the network
            "epoches":                  epoches,                           #number of epoches
            "step_epoches":             step_epoches,                      #the epoches at which to decrease the learning rate
            "Flag_schedule":            Flag_schedule,                     #Whether to use decreased learning rate or constant
            "Diff":                     Diff,                              #Differene to compute the learning rate
            "rho":                      Rho,                               #The density of the fluid
            "Lambda":                   Lambda,                            #Coefficent factor of boundary condition in the loss function
            "Path_NetWeights":          path_NetWeights,                   #Path for saving the weights
            "ActivationFunction":       ActivationFunction,                # Which AF neural net
            "NumberOfLayers":           num_layers,                        # Number of layers
            "NumberOfHiddenNeurons":    dim_hidden,                        # Number of Neurons in each layer
            "W0_Siren":                 w0,                                # W0 hyperparameter of Siren
            "Time":                     T,                                 # Time of the model
            "Time_walls":               T_walls,                           # Time of the model with the shape of walls inputs
            "Time_data":                T_sensors,                         # Time of the model with the shape of sensor data inputs
            "sampling_rate":            sampling_rate,                     # What is Sampling rate?
            "NumberOfSampleFiles":      SampleFileNumber,                  # How many sample file do we have
            "shuffle":                  shuffle,                           # Input data should be shuffled or not?
            "input_files":              folder_path_velocity,              # Input data address
            "vtu_files":                vtu_files
            }

net = geo_train(InputParameters)
