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


class CardiovascularPINNs():
	def __init__(self):
		self.Args=Args
	
		#Make an output directory to store the PINNs Predictions
		if self.Args.OutputFolder is None:
			self.Args.OutputFolder=os.path.join(self.Args.InputFolder+"../PINNsPredictions/")
			os.makedirs(self.Arg.OutputFolder)
			print ("--- Output Folder is: %s"%self.Args.OutputFolder)
	
		#Create a name for the output file
		if self.Args.OutputFileName is None:
			self.OutputFileName=self.Args.ActivationFunction+"_layern%d"%num_layers+"_hiddenlayern%d"%dim_hidden+"_SensNum%d"%sensor_num+"_Rho%d"%Rho+"_Diff"+str(Diff)+"_w0siren"+str(w0)
			self.Args.OutputFileName=os.path.join(self.Args.OutputFolder,self.OutputFileName)

		if self.Args.Shuffle is 1: self.Args.Shuffle=True
		else: self.Args.Shuffle=False
		
		if self.Args.DynamicLearningRate is 1: self.Args.DynamicLearningRate=True
		else: self.Args.DynamicLearningRate=False

		if self.Args.TimeVarying is 1: self.NumberOfInputs=self.Args.Dimension+1
		else: self.NumberOfInputs=self.Args.Dimension
		

                # Define device and processor for CPU, GPU or MacOS
		if self.Args.FlagGPU == 0:
			self.device = torch.device("cpu")
			self.processor = "cpu"
		if Flag_GPU == 1:
			self.device = torch.device("cuda:0")
			self.processor = "cuda:0"
		if Flag_GPU == 2:
			self.device = torch.device("mps")
			self.processor = "mps"


	def main(self):

		#Read the VTU Files and Directories
		vtu_files, directories = sort_vtuFiles_WallFolder(self.Args.InputFolder)

		#Define how many vtu files to read
		self.NumberOfFiles=len(vtu_files) 
		
#---------------------------------- I DONT UNDERSTAND THIS -----------------------
		# Parameters for Time Varying problem
		sampling_rate = 5                                                   # What is the sampling rate when the network is time varying. For 3D Aorta case it is 5, and for 2D stenosis case it is 35
		#sampling_rate = 35
		SampleFileNumber = len(vtu_files)                                   # How many Velocity file do we have for time varying problem
		TotalSampleOfData = 100                                             # How many file do we have on the whole data. It is 100 for 3D Aorta and 2500 for 2D stenosis

#---------------------------------------------------------------------------------

		############################
		# Iterate over all files and directories in the folder
		print("--- Reading the Mesh, Wall, Inlet and Outlet Coordinates")
		x, y, z, T, xb_wall, yb_wall, zb_wall, T_walls, Sensor_coord_x, Sensor_coord_y, Sensor_coord_z, T_sensors, data_vel_u, data_vel_v, data_vel_w, NumberOfMechCoordinates, MeshCompleteVTU = Read_Input_3D_Data(SampleFileNumber, folder_path_velocity, vtu_files, self.Args.VelocityArrayName, self.Args.NumberOfSensorPoints, sampling_rate, self.NumberOfFiles, self.Args.Period)

		x_data = Sensor_coord_x
		y_data = Sensor_coord_y
		z_data = Sensor_coord_z

		# Define boundry Condition
		print ("--- Defining Boundary Conditions of Zero for the Wall")
		u_wall_BC = np.linspace(0., 0., len(xb_wall))                                #wall boundry condition in direction u
		v_wall_BC = np.linspace(0., 0., len(xb_wall))                              #wall boundry condition in direction v
		w_wall_BC = np.linspace(0., 0., len(xb_wall))                              #wall boundry condition in direction w
		
		#Define vectors for velocity
		u_wall_BC = u_wall_BC.reshape(-1, 1) #need to reshape to get 2D array
		v_wall_BC = v_wall_BC.reshape(-1, 1) #need to reshape to get 2D array
		w_wall_BC = w_wall_BC.reshape(-1, 1) #need to reshape to get 2D array

		#Print out the shape of the velocity field 
		print('--- Shape of coordinates in the wall boundry: x{} y{} z{}'.format(xb_wall.shape, yb_wall.shape, zb_wall.shape))
		print('--- Shape of velocity of the wall boundry (BC) in direction of u{} , v{} , w{})'.format(u_wall_BC.shape, v_wall_BC.shape, w_wall_BC.shape))


		#Output the loss function in excel format	
		if not os.path.isfile(path_NetWeights+"loss.xlsx"):
		    headers= ['Loss_eqn', 'Loss_BC', 'Loss_Data', 'Loss_total', 'Time']
		    workbook_name = os.path.join(self.Args.OutputFolder,"loss.xlsx")
		    wb = Workbook()
		    page = wb.active
		    page.title = 'Siren-Based-3Daorta'
		    page.append(headers) # write the headers to the first line
		    wb.save(filename=workbook_name)


		if self.Args.TimeVarying == 0:  # make T if the model is steady state to avoid error
		    T_walls = np.zeros(shape=xb_wall.shape)
		    T_sensors = np.zeros(shape=x_data.shape)
		    T = np.zeros(shape=x.shape)
		    


		InputParameters={"device":              self.device,
		            "processor":                self.processor,
		            "dim":                      self.Args.Dimension,
		            "NumberOfMechCoordinates":  NumberOfMechCoordinates,              # How many MechCoordinates do we have
		            "NumberOfInputs":           self.NumberOfInputs,                  # How many inputs do we have for neural network x , y, z ,T
		            "MeshCompleteVTU":          MeshCompleteVTU,                      #Mesh volume file
		            "xyz":                      [x, y, z],                            #Mesh Coordinates
		            "xyzb_wall":                [xb_wall, yb_wall, zb_wall],          #Wall Coordinates
		            "uvw_wall_BC":              [u_wall_BC, v_wall_BC, w_wall_BC],    #Boundary Conditions (Velocity=0)
		            "xyz_data":                 [x_data, y_data, z_data],             #Sensor Coordinates
		            "data_vel":                 [data_vel_u, data_vel_v, data_vel_w], #Sensor Velocities
		            "batchsize":                self.Args.BatchSize,                  #Batch Size, the number of data to show the network in a single iteration
		            "learning_rate":            self.Args.LearningRate,               #learning rate of the network
		            "decay_rate":               self.Args.DecayRate,                  #decay rate of the network
		            "epoches":                  self.Args.NumberOfEpoches,            #number of epoches
		            "step_epoches":             self.Args.StepEpoches,                #the epoches at which to decrease the learning rate
		            "Flag_schedule":            self.Args.DynamicLearningRate,        #Whether to use decreased learning rate or constant
		            "Diff":                     self.Args.Viscosity,                  #Differene to compute the learning rate
		            "rho":                      self.Args.Density,                    #The density of the fluid
		            "Lambda":                   self.Args.Lambda,                     #Coefficent factor of boundary condition in the loss function
		            "Path_NetWeights":          self.Args.OutputFolder,               #Path for saving the weights
		            "ActivationFunction":       self.Args.ActivationFunction,         # Which AF neural net
		            "NumberOfLayers":           self.Args.NumberOfLayers,             # Number of layers
		            "NumberOfHiddenNeurons":    self.Args.NumberOfNeurons,            # Number of Neurons in each layer
		            "W0_Siren":                 self.Args.Omega0,                     # W0 hyperparameter of Siren
		            "Time":                     T,                                    # Time of the model
		            "Time_walls":               T_walls,                              # Time of the model with the shape of walls inputs
		            "Time_data":                T_sensors,                            # Time of the model with the shape of sensor data inputs
		            "sampling_rate":            sampling_rate,                        # What is Sampling rate?
		            "NumberOfSampleFiles":      SampleFileNumber,                     # How many sample file do we have
		            "shuffle":                  self.Args.Shuffle,                    # Input data should be shuffled or not?
		            "input_files":              folder_path_velocity,                 # Input data address
		            "vtu_files":                vtu_files
		            }

		net = geo_train(InputParameters)



if __name__=="__main__":
        #Description
        
	parser = argparse.ArgumentParser(description="This script will generate tecplot files from perfusion territory vtu files.")
                        
        #Input filename of the perfusion map
	parser.add_argument('-InputFolder', '--InputFolder', type=str, required=True, dest="InputFolder",help="The folder containing the velocity files and wall nodes.")
	
	parser.add_argument('-VelocityArrayName', '--VelocityArrayName', type=str, required=False,default="velocity", dest="ArrayName",help="Name for the velocity array in the data files.")
	
	parser.add_argument('-GPUFlag', '--GPUFlag', type=int, required=False,default=1, dest="GPUFlag",help="Flag to use GPU rather than CPU. Default is 1. [0=CPU, 1=GPU, 2=MPS].")               
	
	parser.add_argument('-Dimension', '-Dimension', type=int, required=False, default=3, dest="Dimension", help="The dimension of the geometric problem. Could be 2 or 3. Default is 3.")
	
	parser.add_argument('-TimeVarying', '-TimeVarying', type=int, required=False, default=1, dest="TimeVarying", help="Assign whether the problem is steady-state of time-varying. Default is time-varying.")
	
	parser.add_argument('-ActivationFunction', '-ActivationFunction', type=str, required=False, default="tanh", dest="ActivationFunction", help="Assign the activation function: tanh, swish, sinus or sinusResNet. Default is tanh.")
	
	parser.add_argument('-NumberOfSensorPoints', '-NumberOfSensorPoints', type=int, required=False, default=800, dest="NumberOfSensorPoints", help="Assign the number of sensor data points to sample from the vtu velocity files. Default is 800.")
	
	parser.add_argument('-NumberOfLayers', '-NumberOfLayers', type=int, required=False, default=4, dest="NumberOfLayers", help="Assign the number of layers for the neural network. Default is 4.")
	
	parser.add_argument('-NumberOfNeurons', '-NumberOfNeurons', type=int, required=False, default=128, dest="NumberOfNeurons", help="Assign the number of neurons per layer. Default is 128.")
	
	parser.add_argument('-Omega0', '-Omega0', type=int, required=False, default=25, dest="Omega0", help="Only specified for the sinus activation and sinusResNet model. This parameter assigns the frequency for the first layer to improve performance. Default is 25 for 3D Aorta.")
	
	parser.add_argument('-Period', '-Period', type=float, required=False, default=1.0, dest="Period", help="Assign the period of the cardiac cycle. Default is 1.0.")
	
	parser.add_argument('-BatchSize', '-BatchSize', type=float, required=False, default=512, dest="BatchSize", help="Assign the batch size.")
	parser.add_argument('-Shuffle', '-Shuffle', type=int, required=False, default=1, dest="Shuffle", help="Assign if input data needs to be shuffled. Default is 1.")
	
	parser.add_argument('-Lambda', '-Lambda', type=float, required=False, default=0.9, dest="Lambda", help="Assign smoothing factor for moving average on Lambda_bc and Lambda_data in the loss function. Default is 0.9.")
	
	parser.add_argument('-Viscosity', '-Viscosity', type=float, required=False, default=0.04, dest="Viscosity", help="Assign the viscosity of blood. Default is 0.04 poise. Use 0.000452638 for sample data in github repository.")
	
	parser.add_argument('-Density', '-Density', type=float, required=False, default=1.06, dest="Density", help="Assign the density of blood. Default is 1.06 g/cm3. Use 1.0 for sample data in github repository.")
	
	parser.add_argument('-DynamicLearningRate', '-DynamicLearningRate', type=int, required=False, default=1, dest="DynamicLearningRate", help="Assign whether to use constant or dynamic learning rate. Default is 1. [1=Dynamic Learning Rate, 0=Constant Learning Rate].")
	
	parser.add_argument('-LearningRate', '-LearningRate', type=float, required=False, default=1e-3, dest="LearningRate", help="Assign the learning rate for the training process. Default is 1e-3.")
	
	parser.add_argument('-NumberOfEpoches', '-NumberOfEpoches', type=int, required=False, default=400, dest="NumberOfEpoches", help="Assign the number of epoches to run the training process. Default is 400.")
	
	parser.add_argument('-StepEpoches', '-StepEpoches', type=int, required=False, default=70, dest="StepEpoches", help="Assign after how many epoches you want to change the learning rate. Default is 70.")
	
	parser.add_argument('-DecayRate', '-DecayRate', type=float, required=False, default=0.1, dest="DecayRate", help="Assign the decay rate (i.e., learning rate multiplies by this number after StepEpoches. Default is 0.1.")
	
	parser.add_argument('-OutputFolder', '-OutputFolder', type=str, required=False, dest="OutputFolder", help="Assign the name of the output folder. By default, it will be one folder up from the velocity data folder.")
	
	args=parser.parse_args() 
        
	CariovascularPINNs(args).main()    

