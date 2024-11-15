CardiovascularPINNs can be used to inversely train a neural network to solve blood flow simulation problems. You will need the following pre-requisite libraries to use the framework: 
1. vtk
2. pytorch
3. numpy
4. matplotlib
5. openpyxl
6. itertools

To get help for any script, please type:
```console
foo@bar:~$ python [ScriptName.py] -h
```

# Physics-Informed Neural Networks (PINNs) for Patient-Specific Cardiovascular Flows
## Introduction
Physics-Informed Neural Networks (PINNs) are a novel approach in the field of machine learning that integrate physical laws into the training process of neural networks. This integration is achieved by incorporating differential equations that govern physical processes directly into the loss function used to train the network. As a result, PINNs are capable of learning and making predictions that are consistent with known physical principles. This makes them particularly valuable for solving complex scientific and engineering problems where traditional data-driven models might struggle due to the scarcity of training data or the complexity of the underlying physical processes. By leveraging the structure and constraints provided by physical theories, PINNs can efficiently predict outcomes, simulate processes, and even discover new insights within fields such as fluid dynamics, material science, and climate modeling, among others.
Here, we coded a PINNs framework that uses Navier-Stroke equation, some known data points, and a boundary condition to simulate blood flow through 2D and 3D geometries. The architecture can be shown as:
![Figure_Methods_0](https://github.com/Owais-Khan/CardiovascularPINNs/assets/79001778/aaf9659c-991c-4fbc-a2dc-2b984ca70242)

# Steps to use the CardiovascularPINNs framework
## Step 1. Preparing Training Data
You will need a folder that contains the CFD simulation or experimental data (i.e., velocity and pressures). Currently, the volumetric velocity and pressure has to be in VTK format as vtkXMLUnstructured data (i.e., .vtu file format). 

Within the folder, you also need to specific the location of wall nodes to assign "zero-velocity" at the wall mesh points. This has to be a vtk file in vtkPolyData format (i.e., .vtp). A typical folder will look like this: [INSERT TREE STRUCTURE FOR mesh-surface folder]
 

### main.py
This file is the main code. All the hyperparameters are in this file, including the input data paths.
### train.py
This file contains the training process function.
### utilities.py
This file contains several functions we defined to be used during the training process.
### ReadMeshComplete.py
This file contains the function for reading the Mesh files.
### SirenNN.py - TanhNN.py - SwishNN.py
These files contain feed-forward neural networks with different activation functions.
### SirenResNetNN.py
This file contains the coding for feed-forward neural networks with sinusoidal activation functions and skip connections.

## JobScript.sh
This file submits a job for running the code on Mist (Compute Canada)

## MeshFolder2D
This folder contains the 2D stenosis case data

## Results_SimVascular_Coarse
This folder contains the 3D stenosis case data
