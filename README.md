CardiovascularPINNs can be used to inversely train a neural network to simulate blood flow in vascular models. For more details about this framework, please refer to the following papers:

[Aghaee A. and Khan MO., "Performance of Fourier-based Activation Function in Physics-Informed Neural Netoworks for Patient-specific Cardiovascular Flows", Computer Methods and Programs in Biomedicine, 2024.](https://scholar.google.ca/citations?view_op=view_citation&hl=en&user=KAfm-70AAAAJ&sortby=pubdate&citation_for_view=KAfm-70AAAAJ:ZeXyd9-uunAC) 

You will need the following pre-requisite libraries to use the framework: 
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

# Steps to use the CardiovascularPINNs framework
## Step 1. Preparing Training Data and Input Points
You will need a folder that contains the CFD simulation or experimental data (i.e., velocity and pressures). The data needs to be in vtkXMLUnstructured format (i.e., .vtu file format). You can use SimVascular to run CFD simulations to obtain your own "ground-truth" CFD data that can seamlessly be used with this framework. 

We have provided sample data in the the subfolder Velocity3DData that contains velocity and pressure data obtained from CFD simulations. The CFD simulations were run for 4 cycles with 10,000 timesteps per cardiac cycle. The data from the last cardiac cycle was projected onto a coarser mesh of approximately 240,000 tetrahedral cells. 

You will also need to store wall boundaries in vtkPolyData format (i.e., .vtp surface files), which will be used to prescribe zero-velocity on the mesh wall. We have added a subfolder in VelocityData3D/WallMesh/wall.vtp. If you are using SimVascular, you can easily obtain this file from the mesh-complete folder (e.g., mesh-complete/mesh-surfaces/walls.vtp)

## Step2. Run CardiovascularPINNs to inversely obtained blood flow data.
To run the framework, you need to run the following command:
```console
foo@bar:~$ python main.py -InputFolder [/path/to/VelocityData3D]
```
Optional argumens are provided below:

| Argument | Type | Description | Default |
| ---      | ---  |  ---        | ---     |
| VelocityArrayName | str | Assign the name of velocity array in velocity data files | velocity |  

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
