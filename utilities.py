import vtk
#from vmtk import vtkvmtk, vmtkscripts
import numpy as np
from glob import glob
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from ReadMeshComplete import *
import os

############ Read Dicom Folder ############
def ReadDicomFiles(FolderName):
	FileList=sorted(glob("%s/*.dcm"%FolderName))
	print (FileList[0])
	Image=vmtkscripts.vmtkImageReader()
	print (dir(Image))
	exit(1)
	Image.InputFileName(FileList[0])
	Image.ImageOutputFileName("/Users/mokhan/GoogleDrive/Owais/Research_Postdoc/perfusion_project/Simvascular/CABG1A/Images/abc.vti")
	Image.Update()

#ReadDicomFiles("/Users/mokhan/GoogleDrive/Owais/Research_Postdoc/perfusion_project/Simvascular/CABG1A/Images/CTA")


############ Input/Output ##################
def ReadVTUFile(FileName):
	reader=vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(FileName)
	reader.Update()
	return reader.GetOutput()

def ReadVTKFile(FileName):
	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(FileName)
	reader.Update()
	data_vtk = reader.GetOutput()
	n_points = data_vtk.GetNumberOfPoints()

	# define two empty numpy array in order to move the inlet boundry coordinates in them
	x_vtk_mesh = np.zeros((n_points,1))
	y_vtk_mesh = np.zeros((n_points,1))

	VTKpoints = vtk.vtkPoints()

# move the inlet boundry coordinates in x_vtk_mesh & y_vtk_mesh
	for i in range(n_points):
		pt_iso  =  data_vtk.GetPoint(i)
		x_vtk_mesh[i] = pt_iso[0]
		y_vtk_mesh[i] = pt_iso[1]
		VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
	point_data = vtk.vtkUnstructuredGrid()
	point_data.SetPoints(VTKpoints)

	xb_in  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh[:]),1))    # move x_vtk_mesh into xb_in
	yb_in  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh[:]),1))    # move y_vtk_mesh into yb_in
	return  xb_in, yb_in

def ReadVTPFile(FileName):
	reader=vtk.vtkXMLPolyDataReader()
	reader.SetFileName(FileName)
	reader.Update()
	return reader.GetOutput()

def ReadVTIFile(FileName):
	reader = vtk.vtkXMLImageDataReader()
	reader.SetFileName(FileName)
	reader.Update()
	return reader.GetOutput()

def WriteVTUFile(FileName,Data):
	writer=vtk.vtkXMLUnstructuredGridWriter()
	writer.SetFileName(FileName)
	writer.SetInputData(Data)
	writer.Update()

def WriteVTPFile(FileName,Data):
	writer=vtk.vtkXMLPolyDataWriter()
	writer.SetFileName(FileName)
	writer.SetInputData(Data)
	writer.Update()

############# Mesh Morphing Functions ###############
        #Create a line from apex and centroid of the myocardium

def CreateLine(Point1,Point2,Length):
	line0=np.array([Point1[0]-Point2[0],Point1[1]-Point2[1],Point1[2]-Point2[2]])
	line1=-1*line0
	line0=(line0/np.linalg.norm(line0))*(Length/2.)
	line1=(line1/np.linalg.norm(line1))*(Length/2.)
	return line0,line1

def CreatePolyLine(Coords):
	# Create a vtkPoints object and store the points in it
	points = vtk.vtkPoints()
	for i in range(len(Coords)): points.InsertNextPoint(Coords[i])

	#Create a Polyline
	polyLine = vtk.vtkPolyLine()
	polyLine.GetPointIds().SetNumberOfIds(len(Coords))
	for i in range(len(Coords)): polyLine.GetPointIds().SetId(i, i)

	# Create a cell array to store the lines in and add the lines to it
	cells = vtk.vtkCellArray()
	cells.InsertNextCell(polyLine)

	# Create a polydata to store everything in
	polyData = vtk.vtkPolyData()

	# Add the points to the dataset
	polyData.SetPoints(points)

	# Add the lines to the dataset
	polyData.SetLines(cells)

	return polyData

def ClosestPoint(Point, Array):
	dist_2 = np.sum((Array - Point)**2, axis=1)
	return Array[np.argmin(dist_2)],np.argmin(dist_2)

def FurthestPoint(Point, Array):
        dist_2 = np.sum((Array - Point)**2, axis=1)
        return Array[np.argmax(dist_2)],np.argmax(dist_2)


def ClippedSlices(Origin,Norm,Volume):
	plane=vtk.vtkPlane()
	plane.SetOrigin(Origin)
	plane.SetNormal(Norm)
	Slice=vtk.vtkCutter()
	Slice.GenerateTrianglesOff()
	Slice.SetCutFunction(plane)
	Slice.SetInputData(Volume)
	Slice.Update()
	return Slice.GetOutput()



def CutPolyData(Point1,Point2,Slice,Norm1):
	#Get the two in-plane normals
	Norm2_slice=(Point1-Point2)/np.linalg.norm(Point1-Point2)
	Norm3_slice=np.cross(Norm1,Norm2_slice)
	#Generate the two planes
	plane_N2=vtk.vtkPlane()
	plane_N2.SetOrigin(Point2)
	plane_N2.SetNormal(Norm2_slice)
	plane_N3=vtk.vtkPlane()
	plane_N3.SetOrigin(Point2)
	plane_N3.SetNormal(Norm3_slice)
	#Clip the plane to get a line across the diameter
	Line =vtk.vtkCutter()
	Line.GenerateTrianglesOff()
	Line.SetCutFunction(plane_N3)
	Line.SetInputData(Slice)
	Line.Update()

	#Separate the line into only one quarter (i.e. half the line)
	Line1=vtk.vtkClipPolyData()
	Line1.SetClipFunction(plane_N2)
	Line1.SetInputData(Line.GetOutput())
	Line1.Update()
	Line1_data=Line1.GetOutput()

	return Line1

#Get Centroid of the VTK dataset
def GetCentroid(Surface):
	Centroid=vtk.vtkCenterOfMass()
	Centroid.SetInputData(Surface)
	Centroid.SetUseScalarsAsWeights(False)
	Centroid.Update()
	return Centroid.GetCenter()


def ExtractSurface(volume):
	#Get the outer surface of the volume
	surface=vtk.vtkDataSetSurfaceFilter()
	surface.SetInputData(volume)
	surface.Update()
	return surface.GetOutput()

#Print the progress of the loop
def PrintProgress(i,N,progress_old):
	progress_=(int((float(i)/N*100+0.5)))
	if progress_%10==0 and progress_%10!=progress_old: print ("    Progress: %d%%"%progress_)
	return progress_%10

def MeshDifferences(Mesh1,Mesh2,ArrayName,Vector=True):
	Npts=Mesh1.GetNumberOfPoints()
	Diff=np.zeros(Npts)

	if Vector is True:
		for i in range(Npts):
				Vx1_=Mesh1.GetPointData().GetArray(ArrayName).GetValue(i*3)
				Vy1_=Mesh1.GetPointData().GetArray(ArrayName).GetValue(i*3+1)
				Vz1_=Mesh1.GetPointData().GetArray(ArrayName).GetValue(i*3+2)
				Vx2_=Mesh2.GetPointData().GetArray(ArrayName).GetValue(i*3)
				Vy2_=Mesh2.GetPointData().GetArray(ArrayName).GetValue(i*3+1)
				Vz2_=Mesh2.GetPointData().GetArray(ArrayName).GetValue(i*3+2)
				Diff[i]=np.sqrt((Vx1_-Vx2_)**2+(Vy1_-Vy2_)**2+(Vz1_-Vz2_)**2)
	return Diff

################ Direction Function ###########
#This function will extract the direction from a
#block of receptive field.
#def DirectionVector(Image,Point,BoxSize):

def coord_to_xyz(area):
	n_points = len(area[0])
	x_vtk_mesh = np.zeros((n_points,1))
	y_vtk_mesh = np.zeros((n_points,1))
	z_vtk_mesh = np.zeros((n_points,1))
	#VTKpoints = vtk.vtkPoints()
	for i in range(n_points):
		#pt_iso  =  area.GetPoint(i)  # pt_iso has three dim that the first is x-coordinates, y-coordinates, and z-coordinates
		x_vtk_mesh[i] = area[0][i][0]
		y_vtk_mesh[i] = area[0][i][1]
		z_vtk_mesh[i] = area[0][i][2]

	x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1))    # move x_vtk_mesh into x
	y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))    # move y_vtk_mesh into y
	z  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))    # move y_vtk_mesh into y

	return x, y, z

def coord_to_xy(area):
	n_points = len(area[0])
	x_vtk_mesh = np.zeros((n_points,1))
	y_vtk_mesh = np.zeros((n_points,1))
	#VTKpoints = vtk.vtkPoints()
	for i in range(n_points):
		#pt_iso  =  area.GetPoint(i)  # pt_iso has three dim that the first is x-coordinates, y-coordinates, and z-coordinates
		x_vtk_mesh[i] = area[0][i][0]
		y_vtk_mesh[i] = area[0][i][1]

	x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1))    # move x_vtk_mesh into x
	y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))    # move y_vtk_mesh into y
	return x, y

def concatenate(ListOfWalls,BoundaryWallCoords):
	for i in range(len(ListOfWalls)):
		xwall, ywall, zwall = coord_to_xyz(BoundaryWallCoords[ListOfWalls[i]])
		if i==0:
			xb_wall = xwall
			yb_wall = ywall
			zb_wall = zwall
		if i>0:
			xb_wall = np.concatenate((xb_wall,xwall),axis=0)
			yb_wall = np.concatenate((yb_wall,ywall),axis=0)
			zb_wall = np.concatenate((zb_wall,ywall),axis=0)
	return xb_wall,yb_wall,zb_wall

# Save the sensor data file as a txt file in Meshfolder
def ReadSensorDataFile(SensorDataFileName="sensordata.txt"):
	f = open(SensorDataFileName, "r")
	sensorData = f.read().splitlines() #puts the file into an array
	print(f.read())
	f.close()
	sensorData = sensorData[1:]
	my_array = np.array(sensorData)
	x_data = []
	y_data =[]
	z_data  = []
	data_vel_u= []
	data_vel_v= []
	data_vel_w= []
	data_pressure= []

	for i in range(len(sensorData)):
  		splited=sensorData[i].split()
  		x_data.append([float(splited[0])])
  		y_data.append([float(splited[1])])
  		z_data.append([float(splited[2])])
  		data_vel_u.append([float(splited[3])])
  		data_vel_v.append([float(splited[4])])
  		data_vel_w.append([float(splited[5])])
  		data_pressure.append([float(splited[6])])

	x_data = np.asarray(x_data , dtype=np.float32)
	y_data = np.asarray(y_data, dtype=np.float32)
	z_data = np.asarray(z_data, dtype=np.float32)
	data_vel_u = np.asarray(data_vel_u, dtype=np.float32)
	data_vel_v = np.asarray(data_vel_v, dtype=np.float32)
	data_vel_w = np.asarray(data_vel_w, dtype=np.float32)
	data_pressure = np.asarray(data_pressure, dtype=np.float32)
	#print('sensor coordinates x:{} y:{} z:{} '.format(x_data.reshape(1, -1),y_data.reshape(1, -1),z_data.reshape(1, -1)))
	#print('Sensor value in directio of u: ',data_vel_u.reshape(1, -1))
	#print('Sensor value in directio of v: ',data_vel_v.reshape(1, -1))
	#print('Sensor value in directio of w: ',data_vel_w.reshape(1, -1))
	print('the number of sensor data is: ',len(data_vel_w))

	return x_data, y_data, z_data, data_vel_u, data_vel_v, data_vel_w, data_pressure


def ExtractVelocitySensorDataFromVTU(sensor_coordinate_x, sensor_coordinate_y, sensor_coordinate_z, GrandTruthVTUpath = 'MeshFolder/velocity_0.vtu', NameOfVelocityField = 'Assigned Vector Function'):
	#Loading the file data (ground truth) to assign to our sensors in variables data_vel_u (u direction) & data_vel_v(v direction)
	reader = vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(GrandTruthVTUpath)
	reader.Update()
	data_vtk = reader.GetOutput()
	n_points = data_vtk.GetNumberOfPoints()
	VTKpoints = vtk.vtkPoints()
	for i in range(len(sensor_coordinate_x)):
		VTKpoints.InsertPoint(i, sensor_coordinate_x[i], sensor_coordinate_y[i], sensor_coordinate_z[i])
	point_data = vtk.vtkUnstructuredGrid()
	point_data.SetPoints(VTKpoints)
	probe = vtk.vtkProbeFilter()
	probe.SetInputData(point_data)
	probe.SetSourceData(data_vtk)
	probe.Update()
	array = probe.GetOutput().GetPointData().GetArray(NameOfVelocityField)
	data_vel = VN.vtk_to_numpy(array)  # the value of u and v direction in the 5 sensor locations
	data_vel_u = data_vel[:,0]
	data_vel_v = data_vel[:,1]
	x_data= sensor_coordinate_x.reshape(-1, 1) #need to reshape to get 2D array
	y_data= sensor_coordinate_y.reshape(-1, 1) #need to reshape to get 2D array
	data_vel_u= data_vel_u.reshape(-1, 1) #need to reshape to get 2D array
	data_vel_v= data_vel_v.reshape(-1, 1) #need to reshape to get 2D array
	return x_data, y_data, data_vel_u, data_vel_v

def ExtractVelocitySensorDataFromVTU_3D(sensor_coordinate_x, sensor_coordinate_y, sensor_coordinate_z, GrandTruthVTUpath = 'MeshFolder/velocity_0.vtu', NameOfVelocityField = 'Assigned Vector Function'):
	#Loading the file data (ground truth) to assign to our sensors in variables data_vel_u (u direction) & data_vel_v(v direction)
	reader = vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(GrandTruthVTUpath)
	reader.Update()
	data_vtk = reader.GetOutput()
	n_points = data_vtk.GetNumberOfPoints()
	VTKpoints = vtk.vtkPoints()
	for i in range(len(sensor_coordinate_x)):
		VTKpoints.InsertPoint(i, sensor_coordinate_x[i], sensor_coordinate_y[i], sensor_coordinate_z[i])
	point_data = vtk.vtkUnstructuredGrid()
	point_data.SetPoints(VTKpoints)
	probe = vtk.vtkProbeFilter()
	probe.SetInputData(point_data)
	probe.SetSourceData(data_vtk)
	probe.Update()
	array = probe.GetOutput().GetPointData().GetArray(NameOfVelocityField)
	data_vel = VN.vtk_to_numpy(array)  # the value of u and v direction in the 5 sensor locations
	data_vel_u = data_vel[:, 0]
	data_vel_v = data_vel[:, 1]
	data_vel_w = data_vel[:, 2]
	x_data = sensor_coordinate_x.reshape(-1, 1) #need to reshape to get 2D array
	y_data = sensor_coordinate_y.reshape(-1, 1) #need to reshape to get 2D array
	z_data = sensor_coordinate_z.reshape(-1, 1) #need to reshape to get 2D array
	data_vel_u = data_vel_u.reshape(-1, 1) #need to reshape to get 2D array
	data_vel_v = data_vel_v.reshape(-1, 1) #need to reshape to get 2D array
	data_vel_w = data_vel_w.reshape(-1, 1) #need to reshape to get 2D array
	return x_data, y_data, z_data, data_vel_u, data_vel_v, data_vel_w

# Two Helper functions for Siren
def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

def SaveVTU_TimeVarying(output_pinns, Model_dim, SampleNum_InTime, path, epoch, input_files, vtu_files):
	output_u = output_pinns[:, 0]
	output_v = output_pinns[:, 1]
	if Model_dim == 3: output_w = output_pinns[:, 2]
	output_p = output_pinns[:, Model_dim]
	output_u = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
	output_v = output_v.cpu().data.numpy()
	if Model_dim == 3: output_w = output_w.cpu().data.numpy()
	output_p = output_p.cpu().data.numpy()
	v = np.float64(output_v)
	u = np.float64(output_u)
	if Model_dim == 3: w = np.float64(output_w)
	p = np.float64(output_p)
	#UVW in a single array
	U_vector = np.zeros(shape=(len(u), Model_dim))
	U_vector[:, 0] = u
	U_vector[:, 1] = v
	if Model_dim == 3: U_vector[:, 2] = w
	U_vector = np.atleast_2d(U_vector)
	U_vectorVTK = numpy_to_vtk(U_vector, deep=True)
	U_vectorVTK.SetName("velocity")
	VolumeCoords, BoundaryWallCoords, BoundaryInletCoords, BoundaryOutletCoords, MeshCompleteVTU = ReadMeshComplete(input_files).Main(vtu_files[SampleNum_InTime])
	MeshCompleteVTU.GetPointData().AddArray(U_vectorVTK)
	MeshCompleteVTU.Modified()
	P_vector = np.zeros(shape=(len(p), 1))
	P_vector = np.atleast_2d(P_vector)
	P_vectorVTK = numpy_to_vtk(P_vector, deep=True)
	P_vectorVTK.SetName("pressure")
	MeshCompleteVTU.GetPointData().AddArray(U_vectorVTK)
	MeshCompleteVTU.Modified()
	if not os.path.exists(path+"output_epoch_%.04d"%epoch+'/'):
		os.mkdir(path+"output_epoch_%.04d"%epoch+'/')
	WriteVTUFile(path+"output_epoch_%.04d"%epoch+'/' + 'sample'+vtu_files[SampleNum_InTime], MeshCompleteVTU)

def SaveVTU_SteadyModel(output_pinns, Model_dim, MeshCompleteVTU, path, epoch):
	output_u = output_pinns[:,0]
	output_v = output_pinns[:,1]  #evaluate model
	if Model_dim==3: output_w = output_pinns[:,2]
	output_p = output_pinns[:,Model_dim]

	output_u = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
	output_v = output_v.cpu().data.numpy()
	if Model_dim==3: output_w = output_w.cpu().data.numpy()
	output_p = output_p.cpu().data.numpy()

	v=np.float64(output_v)
	u=np.float64(output_u)
	if Model_dim==3: w=np.float64(output_w)
	p = np.float64(output_p)

	#UVW in a single array
	U_vector=np.zeros(shape=(len(u),Model_dim))
	U_vector[:,0]=u
	U_vector[:,1]=v
	if Model_dim==3: U_vector[:,2]=w
	U_vector = np.atleast_2d(U_vector)
	U_vectorVTK= numpy_to_vtk(U_vector, deep=True)
	U_vectorVTK.SetName("velocity")
	MeshCompleteVTU.GetPointData().AddArray(U_vectorVTK)
	MeshCompleteVTU.Modified()
	P_vector = np.zeros(shape=(len(p), 1))
	P_vector = np.atleast_2d(P_vector)
	P_vectorVTK = numpy_to_vtk(P_vector, deep=True)
	P_vectorVTK.SetName("pressure")
	MeshCompleteVTU.GetPointData().AddArray(U_vectorVTK)
	MeshCompleteVTU.Modified()
	#Write the output file
	if not os.path.exists(path+"steadymodel/"):
		os.mkdir(path+"steadymodel/")
	WriteVTUFile(path+"steadymodel/"+"output_epoch_%.04d"%epoch+'.vtu',MeshCompleteVTU)

def sort_vtuFiles_WallFolder(folder_path_velocity):
	vtu_files = []
	directories = []
	
	for item in os.listdir(folder_path_velocity):
		item_path = os.path.join(folder_path_velocity, item)
		if os.path.isfile(item_path) and item.endswith('.vtu'):
			vtu_files.append(item)
		elif os.path.isdir(item_path):
			directories.append(item)
	return sorted(vtu_files), sorted(directories)


def Read_Input_3D_Data(SampleFileNumber, folder_path_velocity, vtu_files, NameOfVelocityField, sensor_num, TimeOfSampling):
	for i in range(SampleFileNumber):    # how many file do we have? now 20
		if i == 0:
			VolumeCoords, BoundaryWallCoords, BoundaryInletCoords, BoundaryOutletCoords, MeshCompleteVTU = ReadMeshComplete(folder_path_velocity).Main(vtu_files[i])
			x, y, z = coord_to_xyz(VolumeCoords) # Preparing the mesh coordinates as x, y ,z
			NumberOfMechCoordinates = len(x)
			T = np.ones(shape=x.shape)
			T = T * (TimeOfSampling/SampleFileNumber * i)
			# rename the wall vtp file to wall.vtp and put it in Results_SimVascular_Coarse/mesh-surfaces
			xb_wall, yb_wall, zb_wall = coord_to_xyz(BoundaryWallCoords["wall.vtp"]) # Preparing the wall coordinates as xb_wall, yb_wall ,zb_wall
			T_walls = np.ones(shape=xb_wall.shape)
			T_walls = T_walls * (TimeOfSampling/SampleFileNumber * i)
			SensorCoords = np.linspace(len(xb_wall), len(x)-1, num=sensor_num, dtype=int)
			x_data = x[SensorCoords]
			y_data = y[SensorCoords]
			z_data = z[SensorCoords]
			Sensor_coord_x, Sensor_coord_y, Sensor_coord_z, data_vel_u, data_vel_v, data_vel_w = ExtractVelocitySensorDataFromVTU_3D(x_data, y_data, z_data, GrandTruthVTUpath=folder_path_velocity+"/"+vtu_files[i], NameOfVelocityField=NameOfVelocityField)
			T_sensors = np.ones(shape=Sensor_coord_x.shape)
			T_sensors = T_sensors * (TimeOfSampling/SampleFileNumber * i)
		else:
			VolumeCoords, BoundaryWallCoords, BoundaryInletCoords, BoundaryOutletCoords, MeshCompleteVTU = ReadMeshComplete(folder_path_velocity).Main(vtu_files[i])
			x_nextT, y_nextT, z_nextT = coord_to_xyz(VolumeCoords)
			T_nextT = np.ones(shape=x_nextT.shape)
			T_nextT = T_nextT * (TimeOfSampling/SampleFileNumber * i)
			x = np.concatenate((x, x_nextT), axis=0)
			y = np.concatenate((y, y_nextT), axis=0)
			z = np.concatenate((z, z_nextT), axis=0)
			T = np.concatenate((T, T_nextT), axis=0)
			# wall data
			xb_wallnext, yb_wallnext, zb_wallnext = coord_to_xyz(BoundaryWallCoords["wall.vtp"])  # Preparing the wall coordinates as xb_wall, yb_wall ,zb_wall
			xb_wall = np.concatenate((xb_wall, xb_wallnext), axis=0); yb_wall = np.concatenate((yb_wall, yb_wallnext), axis=0); zb_wall = np.concatenate((zb_wall, zb_wallnext), axis=0)
			T_wallsnext = np.ones(shape=xb_wallnext.shape)
			T_wallsnext = T_wallsnext * (TimeOfSampling/SampleFileNumber * i); 
			T_walls = np.concatenate((T_walls, T_wallsnext), axis=0)
			# sensor data
			Sensor_coord_nextx, Sensor_coord_nexty, Sensor_coord_nextz, data_vel_nextu, data_vel_nextv, data_vel_nextw = ExtractVelocitySensorDataFromVTU_3D(x_data, y_data, z_data, GrandTruthVTUpath=folder_path_velocity+"/"+vtu_files[i], NameOfVelocityField=NameOfVelocityField)
			Sensor_coord_x = np.concatenate((Sensor_coord_x, Sensor_coord_nextx), axis=0); Sensor_coord_y = np.concatenate((Sensor_coord_y, Sensor_coord_nexty), axis=0); Sensor_coord_z = np.concatenate((Sensor_coord_z, Sensor_coord_nextz), axis=0)
			data_vel_u = np.concatenate((data_vel_u, data_vel_nextu), axis=0); data_vel_v = np.concatenate((data_vel_v, data_vel_nextv), axis=0); data_vel_w = np.concatenate((data_vel_w, data_vel_nextw), axis=0)
			T_sensorsnext = np.ones(shape=Sensor_coord_nextx.shape)
			T_sensorsnext = T_sensorsnext * (TimeOfSampling/SampleFileNumber * i)
			T_sensors = np.concatenate((T_sensors, T_sensorsnext), axis=0)
	return x, y, z, T, xb_wall, yb_wall, zb_wall, T_walls, Sensor_coord_x, Sensor_coord_y, Sensor_coord_z, T_sensors, data_vel_u, data_vel_v, data_vel_w, NumberOfMechCoordinates, MeshCompleteVTU



def Prepare_2D_stenosis_Data(folder_path_velocity, vtu_files, sensor_num, NameOfVelocityField, SampleFileNumber, sampling_rate, TotalSampleOfData, TimeOfSampling):
	if sensor_num == 25:
		x_data = list(np.linspace(0, 1, sensor_num, endpoint=True))
		y_data = [0]*sensor_num
	if sensor_num == 100:
		x_data = list(np.linspace(0, 1, 50, endpoint=True))*2
		y_data = [(0.04/3)+(-0.02)]*50 + [(0.02)-(0.04/3)]*50
	if sensor_num == 225:
		x_data = list(np.linspace(0, 1, 75, endpoint=True))*3
		y_data = [-0.01]*75 + [0]*75 + [0.01]*75
	if sensor_num == 400:
		x_data = list(np.linspace(0, 1, 100, endpoint=True))*4
		y_data = [-0.012]*100 + [-0.004]*100 + [0.004]*100 + [0.012]*100
	z_data = [0] * sensor_num
	x_data = np.asarray(x_data)  # convert to numpy
	y_data = np.asarray(y_data)  # convert to numpy
	z_data = np.asarray(z_data)  # convert to numpy
	for i in range(SampleFileNumber):    # how many file do we have? now 20
		if i == 0:
			VolumeCoords, BoundaryWallCoords, BoundaryInletCoords, BoundaryOutletCoords, MeshCompleteVTU = ReadMeshComplete(folder_path_velocity).Main(vtu_files[i])
			x, y = coord_to_xy(VolumeCoords)
			NumberOfMechCoordinates = len(x)
			T = np.ones(shape=x.shape)
			T = T * (TimeOfSampling/SampleFileNumber * i)
			# sensor data
			# Sensor_coord_x, Sensor_coord_y, data_vel_u, data_vel_v = ExtractVelocitySensorDataFromVTU(x_data, y_data, z_data, GrandTruthVTUpath = 'MeshFolder_timevarying/'+"velocity_%d"%(i*sampling_rate)+'.vtu', NameOfVelocityField = 'Assigned Vector Function')
			Sensor_coord_x, Sensor_coord_y, data_vel_u, data_vel_v = ExtractVelocitySensorDataFromVTU(x_data, y_data, z_data, GrandTruthVTUpath = folder_path_velocity+'/'+vtu_files[i], NameOfVelocityField = NameOfVelocityField)
			T_sensors = np.ones(shape=Sensor_coord_x.shape)
			T_sensors = T_sensors * (TimeOfSampling/SampleFileNumber * i)
			# walls
			xb_wall, yb_wall = ReadVTKFile(folder_path_velocity+'/WallMesh'+'/Walls.vtk')
			T_walls = np.ones(shape=xb_wall.shape)
			T_walls = T_walls * (TimeOfSampling/SampleFileNumber * i)
		else:
			VolumeCoords, BoundaryWallCoords, BoundaryInletCoords, BoundaryOutletCoords, MeshCompleteVTU = ReadMeshComplete(folder_path_velocity).Main(vtu_files[i])
			x_nextT, y_nextT = coord_to_xy(VolumeCoords)
			T_nextT = np.ones(shape=x_nextT.shape)
			T_nextT = T_nextT * (TimeOfSampling/SampleFileNumber * i)
			x = np.concatenate((x, x_nextT), axis=0)
			y = np.concatenate((y, y_nextT), axis=0)
			T = np.concatenate((T, T_nextT), axis=0)

			# walls
			xb_wallnext, yb_wallnext = ReadVTKFile(folder_path_velocity+'/WallMesh'+'/Walls.vtk')
			xb_wall = np.concatenate((xb_wall, xb_wallnext), axis=0); yb_wall = np.concatenate((yb_wall, yb_wallnext), axis=0)
			T_wallsnext = np.ones(shape=xb_wallnext.shape)
			T_wallsnext = T_wallsnext * (TimeOfSampling/SampleFileNumber * i); 
			T_walls = np.concatenate((T_walls, T_wallsnext), axis=0)
			# sensor data
			Sensor_coord_nextx, Sensor_coord_nexty, data_vel_nextu, data_vel_nextv = ExtractVelocitySensorDataFromVTU(x_data, y_data, z_data, GrandTruthVTUpath = folder_path_velocity+'/'+vtu_files[i], NameOfVelocityField = NameOfVelocityField)
			Sensor_coord_x = np.concatenate((Sensor_coord_x, Sensor_coord_nextx), axis=0)
			Sensor_coord_y = np.concatenate((Sensor_coord_y, Sensor_coord_nexty), axis=0)
			data_vel_u = np.concatenate((data_vel_u, data_vel_nextu), axis=0)
			data_vel_v = np.concatenate((data_vel_v, data_vel_nextv), axis=0)
			T_sensorsnext = np.ones(shape=Sensor_coord_nextx.shape)
			T_sensorsnext = T_sensorsnext * (TimeOfSampling/SampleFileNumber * i)
			T_sensors = np.concatenate((T_sensors, T_sensorsnext), axis=0)
	Sensor_coord_z = np.zeros(shape=Sensor_coord_x.shape)

	return x, y, T, xb_wall, yb_wall, T_walls, Sensor_coord_x, Sensor_coord_y, Sensor_coord_z, T_sensors, data_vel_u, data_vel_v, NumberOfMechCoordinates, MeshCompleteVTU
