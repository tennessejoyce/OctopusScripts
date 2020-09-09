'''A class to help with handling Octopus output files in dx format.'''


import numpy as np
import pandas as pd
from glob import glob

class Mesh:
	'''Object to store the information about the mesh used for
	the simulation, which might apply to multiple dx files.'''
	def __init__(self,sample_dx_file):
		'''Parses the header of a dx file to get the mesh information.
		Note that even if you have a non-rectangular simulation box in
		Octopus, the dx file always has a full 3d rectanglular grid of data.
		This is accomplished by padding with zeros the points outside the
		simulation box.'''
		with open(sample_dx_file) as f:
			s1 = f.readline()
			s2 = f.readline()
		#Shape of the 3d array (number of points along each axis x,y,z).
		self.wf_shape = [int(n) for n in s1.split()[-3:]]
		#
		self.box_lengths = [-float(n) for n in s2.split()[-3:]]
		#Total number of grid points.
		self.num_points = np.prod(wfShape)
		#Grid spacing, assumed to be uniform in all three directions.
		self.dx = 2*boxLengths[0]/(wfShape[0]-1)
		self.mesh1D = [np.linspace(-b,b,n) for b,n in zip(boxLengths,wfShape)]
		
	def construct_3d_mesh(self):
		'''Construct numpy arrays with the x,y,z coordinates of each gridpoint.
		This isn't done by default in the constructor because these arrays might
		take up a considerable amount of memory.'''
		self.x,self.y,self.z = np.meshgrid(self.mesh1D[0],self.mesh1D[1],self.mesh1D[2],indexing='ij')

	def read_dx(self,dx_filename,real=True):
		'''Reads a dx file into a numpy array. It's much faster to do this in pandas
		than to parse the file line-by-line in python, but we need to know the structure
		of the file beforehand (i.e., how many gridpoints). That's why this a method
		of the Mesh object.
		The parameter 'real' tells us whether the dx file contains real data (like charge
		density) or complex data (like a time-dependent orbital).'''
		if real:
			out = pd.read_csv(dx_filename,header=None,skiprows=7,nrows=self.num_points).values	
			out = np.reshape(out,self.wf_shape)
		else:
			out = pd.read_csv(dx_filename,header=None,skiprows=7,nrows=self.num_points,delim_whitespace=True).values	
			out = np.reshape(out[:,0]+1j*out[:,1],self.wf_shape)
		return out

	def read_current_density(self,directory):
		'''Reads a vector field in three separate dx files.'''
		return np.stack([self.read_dx(f'{directory}/current-{i}.dx') for i in ('x','y','z')])

	def read_ks_orbitals(self,directory,real=True):
		'''Reads in all the orbitals in a given directory, and keep them in
		a dictionary indexed by	the wavefunction file identifiers.'''
		return {f.split('/')[-1]:self.read_dx(f,real) for f in glob(f'{directory}/wf-*.dx')}


def td_projections(directory,verbose=True):
	'''Example usage of this class to compute the projections of the
	time-dependent Kohn-Sham orbitals onto the field-free Kohn-Sham orbitals.
	This function assumes that ground state and time-dependent Octopus calculations
	have been run in 'directory'. The orbitals from the ground state calculation
	will have been saved to 'directory/static' and the time-dependent orbitals will
	have been saved to 'directory/output_iter', both in dx format.'''

	#The example dx file from which the mesh is loaded.
	field_free_orbital_files = sorted(glob(f'{directory}/static/wf-*.dx'))
	if verbose:
		print(f'{len(field_free_orbital_files)} field-free orbitals found in {directory}/static')
	if len(field_free_orbital_files)==0:
		#No field-free orbitals to project onto.
		return
	if verbose:
		print('Reading mesh information from {field_free_orbital_files[0]}')
	mesh = Mesh(field_free_orbital_files[0])
	if verbose:
		print('Reading in field-free orbitals')
	field_free_orbitals = mesh.read_ks_orbitals(directory)

	td_folders = sorted(glob(f'{directory}/output_iter/*'))
	if verbose:
		print(f'{len(td_folders)} time-dependent output folders found in {directory}/output_iter')
	if len(td_folders)==0:
		return
	out = []
	for time_folder in td_folders:
		td_orbitals = mesh.read_ks_orbitals(directory,real=False)
		temp = []
		for key in td_orbitals:
			if key in field_free_orbitals:
				temp.append(np.sum(td_orbitals[key]*field_free_orbitals[key]))
		out.append(temp)
	return np.array(out)




