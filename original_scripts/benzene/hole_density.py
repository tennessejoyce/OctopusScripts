from gridData import Grid
import numpy as np
import pandas as pd

sampleDxFile = '2/output_iter/td.0000000/density.dx'

#Open a wavefunction file to find the mesh.
with open(sampleDxFile) as f:
  s1 = f.readline()
  s2 = f.readline()
  wfShape = [int(n) for n in s1.split()[-3:]]
  boxLengths = [-float(n) for n in s2.split()[-3:]]
  numPoints = np.prod(wfShape)
  dx = 2*boxLengths[0]/(wfShape[0]-1)
print(f'Mesh size: {wfShape}')
print(f'Box size: {boxLengths}')
print(f'Spacing: {dx}')
print('Constructing mesh...')
#Construct the mesh
mesh1D = [np.linspace(-b,b,n) for b,n in zip(boxLengths,wfShape)]
x,y,z = np.meshgrid(mesh1D[0],mesh1D[1],mesh1D[2],indexing='ij')
rho = np.sqrt(x**2 + y**2+1e-8)
lzop = np.array([y,-x])
mask = x**2+y**2+z**2 <= 6**2



base_density = pd.read_csv(sampleDxFile,header=None,skiprows=7,nrows=numPoints).values	
base_density = np.reshape(base_density,wfShape)

stamp = str(400*10)
tdDxFile = f'4/output_iter/td.{stamp.zfill(7)}/density.dx'

td_density = pd.read_csv(tdDxFile,header=None,skiprows=7,nrows=numPoints).values	
td_density = np.reshape(td_density,wfShape)

#Write hole density to a file.
hole_density = td_density - base_density
np.save(f'hd_4_{stamp}.npy',hole_density)

