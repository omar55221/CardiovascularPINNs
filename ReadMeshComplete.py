import vtk
import numpy as np
from glob import glob
import sys
# output would be a dictionary that the 0 is whole mesh, 1 is wall, 2 is inlet, 3 is outlet
class ReadMeshComplete():
	def __init__(self,MeshFolder):
		self.MeshFolder=MeshFolder

	def Main(self, velocity_file_path = "Velocity_Re2000.vtu"):
		BoundaryFileNames=glob(self.MeshFolder+"/mesh-surfaces/*.vtp")


		#Create a dictionary to store the surface coordinates
		BoundaryWallCoords  ={}#Wall
		BoundaryInletCoords ={}#Inlet
		BoundaryOutletCoords={}#Outlet

		#Separate the Surfaces into walls, inlet and outlets
		for FileName in BoundaryFileNames:
			SurfaceName=FileName.split("/")[-1]
			SurfaceData =self.ReadVTPFile(FileName) #Load Surface File
			#n_points_meshcomplete= n_points
			#Read Wall Coordinates and Store them
			if FileName.find("wall")>=0:
				BoundaryWallCoords[SurfaceName]=self.GetCoordinates(SurfaceData)
			#Read Inlet Coordinates and Store them
			elif FileName.find("in")>=0:
				BoundaryInletCoords[SurfaceName]=self.GetCoordinates(SurfaceData)
			#Read Outlet Coordinates and Store them
			else:
				BoundaryOutletCoords[SurfaceName]=self.GetCoordinates(SurfaceData)

		#Load the Mesh Coordinates
		MeshCompleteVTU=self.ReadVTUFile(self.MeshFolder+"/"+velocity_file_path)
		VolumeCoords = self.GetCoordinates(MeshCompleteVTU)

		return VolumeCoords,BoundaryWallCoords,BoundaryInletCoords,BoundaryOutletCoords,MeshCompleteVTU
	def ReadVTPFile(self,FileName):
		reader=vtk.vtkXMLPolyDataReader()
		reader.SetFileName(FileName)
		reader.Update()
		reader.GetOutput()

		return reader.GetOutput()

	def ReadVTUFile(self,FileName):
		reader=vtk.vtkXMLUnstructuredGridReader()
		reader.SetFileName(FileName)
		reader.Update()
		return reader.GetOutput()

	def GetCoordinates(self,Data):
		Npts=Data.GetNumberOfPoints() #Number of Points
		Coords=np.zeros(shape=(Npts,3))  #Define Array to Store Coordinates
		for i in range(Npts): Coords[i,:]=Data.GetPoints().GetPoint(i)[:]
		return Coords, Npts;

#if __name__=="__main__":
	#sReadMeshComplete(sys.argv[1]).Main()
