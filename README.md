# Introduction

CardiovascularPINNs can be used to inversely train a neural network to simulate blood flow in vascular models. For more details about this framework, please refer to the following papers:

[Aghaee A. and Khan MO., "Performance of Fourier-based Activation Function in Physics-Informed Neural Netoworks for Patient-specific Cardiovascular Flows", Computer Methods and Programs in Biomedicine, 2024.](https://scholar.google.ca/citations?view_op=view_citation&hl=en&user=KAfm-70AAAAJ&sortby=pubdate&citation_for_view=KAfm-70AAAAJ:ZeXyd9-uunAC) 

You will need the following pre-requisite libraries to use the framework: 
1. vtk
2. pytorch
3. numpy
4. matplotlib
5. openpyxl
6. itertools

We have provided a sample CFD dataset in this github repository; however, a more comprehensive dataset is available publically [elsewhere](https://www.kaggle.com/datasets/khanmu11/aortapinnsdata/data?select=Aorta_100_coarse). This public dataset consists of an aortic models with synthetically-induced stenosis, raning in severity from 0-70% in increments of 5\% (see Figure below). The CFD simulations were performed using SimVascular (7 million elements, 10,000 timesteps) and projected onto a coarser mesh of approximately 340,000 elements. CardiovascularPINNs framework is fully compatible with the dataset included above and should work out-of-the-box.  

![alt text](https://github.com/Owais-Khan/CardiovascularPINNs/blob/main/Figures/Figure1_Stenosis_Mapping.png)

# Steps to use the CardiovascularPINNs framework
## Step 1. Preparing Training Data and Input Points
You will need a folder that contains the CFD simulation or experimental data (i.e., velocity and pressures). The data needs to be in vtkXMLUnstructured format (i.e., .vtu file format). You may use SimVascular to run CFD simulations to obtain your own "ground-truth" CFD data that can seamlessly be used with this framework. We have provided sample data in the the subfolder ```VelocityData3D``` that contains velocity and pressure fields obtained from SimVascular CFD simulations.  

You will also need to store wall boundaries in vtkPolyData format (i.e., .vtp surface files), which will be used to prescribe zero-velocity on the mesh wall. We have added a subfolder in ```VelocityData3D/WallMesh/wall.vtp``` that contains the wall mesh. If you are using SimVascular, you can easily obtain this file from the mesh-complete folder (e.g., ```mesh-complete/mesh-surfaces/walls.vtp```)

## Step2. Run CardiovascularPINNs 
To run the framework, you need to run the following command:
```console
foo@bar:~$ python main_3D.py -InputFolder [/path/to/VelocityData3D]
```
To run the framework using the sample data provided with this repository, you can run the following command:
```console
foo@bar:~$ python main_3D.py -InputFolder ~/VelocityData3D/ -Viscosity 0.000452638 -Density 1
```
Optional argumens are provided below:

| Argument | Type | Description | Default |
| ---      | ---  |  ---        | ---     |
| VelocityArrayName           | str   | Assign the name of velocity array in velocity data files | velocity |
| Period                      | int   | Assign the period of the cardiac cycle. | 1 |
| Viscosity                   | float | Assign the dynamic viscosity of blood. | 0.04 |
| Density                     | float | Assign the density of blood. | 1.06 |
| SkipFiles                   | int   | Assign how many files to skip in the velocity data files | 1 |
| SaveAfter                   | int   | Assign after which epoch to start saving the data. | 100 |
| GPUFlag                     | int   | Assing whether to use GPU rather than CPU. Default is 1. [0=CPU, 1=GPU, 2=MPS]. | 1 |
| Dimension                   | int   | The dimension of the geometric problem. Could be 2 or 3. | 1 |
| TimeVarying                 | int   | Assign whether the problem is steady-state [=0] of time-varying [=1] | 1 |
| ActivationFunction          | str   | Assign the activation function: tanh, swish, sinus or sinusResNet. | tanh |
| NumberOfSensorPoints        | int   | Assign the number of sensor data points to sample from velocity data files. | 800 |
| NumberOfLayers              | int   | Assign the number of layers for the neural network. | 4 |
| NumberOfNeurons             | int   | Assign the number of neurons per layer. | 128 |
| Omega0                      | int   | Only specified for the sinus activation and sinusResNet model. Assign frequency for first layer. | 25 |
| BatchSize                   | int   | Assign the batch size | 512 |
| Shuffle                     | int   | Assign if the input data needs to be shuffled [=1] or remain unshuffled [0] | 1 | 
| Lambda                      | float | Assign smoothing factor for moving average on Lambda_bc and Lambda_data in the loss function. | 0.9 |
| DynamicLearningRate         | int   | Assign whether to use dynamic [=1] or constant [=0] learning rate for loss function. | 1 |
| LearningRate                | float | Assign the learning rate for the training process. | 1e-3 |
| NumberOfEpoches             | int   | Assign the number of epoches to run the training process. | 400 |
| StepEpoches                 | int   | Assign after how many epoches you want to change the learning rate. | 70 |
| DecayRate                   | float | Assign the decay rate (i.e., learning rate multiplies by this number after StepEpoches. | 0.1 |
| OutputFolder                | str   | Assign the name of the output folder. By default, it will be one folder up from the velocity data folder.| 0 |

--- 
## File Description

### main.py
This file is the main file that specifies all of the parameters. All the hyperparameters are in this file, including the input data paths.

### train.py
This file contains the training process function.

### utilities.py
The is file contains commonly used functions, such as those related to I/O.

### ReadMeshComplete.py
This file contains the function to read the Mesh files.

### SirenNN.py - TanhNN.py - SwishNN.py
These files contain feed-forward neural networks with different activation functions.

### SirenResNetNN.py
This file contains the coding for feed-forward neural networks with sinusoidal activation functions and skip connections.

### JobScript.sh
This file submits a job for running the code on the compute cluster. The script is defined for the MIST cluster at the SciNet HPC Consortium.

### VelocityData2D
This folder contains the 2D stenosis case data

### VelocityData3D
This folder contains velocity field in a 3D aortic geometry obtained from CFD simulations run with SimVascular. The simulatations were run at ~7 million tetrahedral elements with 10,000 timesteps per cardiac cycle, and subsequently projected to a coarse mesh. 

