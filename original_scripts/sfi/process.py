import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import interp1d,RectBivariateSpline,barycentric_interpolate
from scipy.signal import convolve
from PIL import Image
import io
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.switch_backend('agg')

#Function to calculate the gradient of an nparray with a fourth order stencil.
def gradient4(a,dx=1,axis=-1):
  #Construct the stencil for a 4th order gradient in the specified axis.
  stencilShape = np.ones_like(a.shape)
  stencilShape[axis] = 5
  stencil = np.array([1.0,-8.0,0.0,8.0,-1.0])/(12.0*dx)
  stencil = np.reshape(stencil,stencilShape)
  #Use a convolution to calculate the gradient (with zero padding).
  return convolve(a,stencil,mode='same')

sampleDxFile = 'output_iter/td.0000000/density.dx'

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


read_dx=True
if read_dx:
	density_XY = []
	ionization = []
	lz = []
	for time_folder in glob(f'output_iter/*'):
		print(time_folder)
		#Compute density integrated over z
		# density = pd.read_csv(time_folder+'/density.dx',header=None,skiprows=7,nrows=numPoints).values	
		# density = np.reshape(density,wfShape)
		# density_XY.append(np.sum(density,axis=-1)*dx)

		#Compute ionization (leaving box) maybe with different masks.
		#ionization.append(np.sum(density*mask)*dx**3)
		
		#Compute lz, maybe with different masks.
		curX = pd.read_csv(time_folder+'/current-x.dx',header=None,skiprows=7,nrows=numPoints).values
		curX = np.reshape(curX,wfShape)
		curY = pd.read_csv(time_folder+'/current-y.dx',header=None,skiprows=7,nrows=numPoints).values
		curY = np.reshape(curY,wfShape)
		lz.append(np.sum(y*curX - x*curY)*dx**3)

	lz = np.array(lz)

	#np.save('density_XY_all.npy',density_XY_all)
	#np.save('ionization_all.npy',ionization_all)
	np.save('lz.npy',lz)
else:
	#density_XY_all = np.load('density_XY_all.npy',allow_pickle=True)
	#ionization_all = np.load('ionization_all.npy',allow_pickle=True)
	lz = np.load('lz.npy',allow_pickle=True)

plot_diff = False
if plot_diff:
	#Compute the density difference.
	density_diff = []
	#Finer mesh to interpolate onto.
	mesh1D_high = [np.linspace(-b,b,2*n) for b,n in zip(boxLengths,wfShape)]
	for d in density_XY_all[-1]:
		#Subtract off the initial density.
		d_low = d-density_XY_all[-1][0]
		#Upsample by cubic spline.
		d_high = RectBivariateSpline(mesh1D[0],mesh1D[1],d_low)(mesh1D_high[0],mesh1D_high[1])
		density_diff.append(d_high)
	density_diff = np.array(density_diff)
	#Upsample in time
	print(density_diff.shape)
	t_low = np.linspace(0,202,density_diff.shape[0])
	t_high = np.linspace(0,202,60)
	density_diff = interp1d(t_low,density_diff,axis=0)(t_high)

	vmax = np.max(np.log(1e-8+np.abs(density_diff)))
	vmin = np.min(np.log(1e-8+np.abs(density_diff)))

	positive_mask = (density_diff>0)

	positive = np.log(np.maximum(density_diff,1e-8))
	negative = np.log(np.maximum(-density_diff,1e-8))

	positive[1-positive_mask] = np.nan
	negative[positive_mask] = np.nan


	#Density movie for a particular intensity.
	images = []
	#Transparent colormaps to overlay positive and negative plots.
	transparent_reds = copy(plt.cm.get_cmap('Reds'))
	transparent_blues = copy(plt.cm.get_cmap('Blues'))
	transparent_reds.set_bad(alpha=0)
	transparent_blues.set_bad(alpha=0)
	for i,(p,n) in enumerate(zip(positive,negative)):

		plt.imshow(p,cmap=transparent_reds,vmax=vmax,vmin=vmin)
		plt.imshow(n,cmap=transparent_blues,vmax=vmax,vmin=vmin)
		plt.xticks([])
		plt.yticks([])
		filename = f'frames/{i}.png'
		plt.savefig(filename)#,dpi=50,optimize=True)
		#plt.savefig(buf,format='png')
		plt.close()
		#buf.seek(0)
		#images.append(copy(Image.open(buf)))
		im = Image.open(filename)
		im.info['duration']=1
		images.append(im)

	images[0].save('density.gif',save_all=True,
				append_images=images[1:], optimize=False, duration=1, loop=0)


print('lz: ',lz.shape)
plt.plot(lz)
plt.savefig('lz.png')
