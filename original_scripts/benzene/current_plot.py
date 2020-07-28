import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import RectBivariateSpline
from scipy.signal import convolve
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

sampleDxFile = '1/output_iter/td.0000000/current-x.dx'

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


def plot_jphi(current,ax):
	#Integrate Jphi over phi.
	jphi3D = (y*current[0]-x*current[1])/rho
	#For each layer of z,
	nrad = 100
	radii = np.linspace(0,6,nrad)[1:]
	jphiXZ = np.zeros([21,2*nrad-1])
	phi = np.linspace(0,2*np.pi,100)[:-1]
	for i,z in zip(range(50,71),np.linspace(-4,4,21)):
		#do a 2D interpolation
		jphiInterp = RectBivariateSpline(mesh1D[0],mesh1D[1],jphi3D[:,:,i])
		#and integrate the interpolant over phi.
		for j,r in enumerate(radii):
			jphiXZ[i-50,j+nrad] = np.mean(jphiInterp(r*np.cos(phi),r*np.sin(phi),grid=False))
			jphiXZ[i-50,nrad-j-2] = jphiXZ[i-50,j+nrad]
	#Plot onto ax.
	#Contour map of J_phi in XZ plane.
	maxjphiXZ = np.max(np.abs(jphiXZ))
	ax.imshow(jphiXZ,cmap='seismic',vmin=-maxjphiXZ,
			vmax=maxjphiXZ,interpolation='bicubic',aspect=(2*nrad-1)/31)
	ax.set_xticks(np.linspace(0,2*nrad-1,5))
	ax.set_xticklabels(np.linspace(-6,6,5))
	ax.set_yticks(np.linspace(0,20,5))
	ax.set_yticklabels(-np.linspace(-4,4,5))

def load_current_dx(intens):
	dx_dir = f'{intens}/output_iter/td.0002022'
	curX = pd.read_csv(dx_dir+'/current-x.dx',header=None,skiprows=7,nrows=numPoints).values	
	curX = np.reshape(curX,wfShape)
	curY = pd.read_csv(dx_dir+'/current-y.dx',header=None,skiprows=7,nrows=numPoints).values	
	curY = np.reshape(curY,wfShape)
	curZ = pd.read_csv(dx_dir+'/current-z.dx',header=None,skiprows=7,nrows=numPoints).values	
	curZ = np.reshape(curZ,wfShape)
	#Symmetrize to extract the ring current part.
	curX -= np.flip(np.flip(curX,axis=0),axis=1)
	curX /= 2
	curY -= np.flip(np.flip(curY,axis=0),axis=1)
	curY /= 2
	current = np.array([curX,curY,curZ])
	return current

def load_current_npy():
	current = -0.5*np.load('j13_14.npy')
	#Zero pad up to the full size.
	current = np.pad(current,((0,0),(38,38),(37,37),(37,37)),mode='constant')
	return current


fig, axs = plt.subplots(2,2,sharex=True,sharey=True)
plot_jphi(load_current_npy(),axs[0,0])
plot_jphi(load_current_dx(1),axs[0,1])
plot_jphi(load_current_dx(4),axs[1,0])
plot_jphi(load_current_dx(8),axs[1,1])
#Write the axis labels only on the external axes.
axs[0,0].set_ylabel('z (a.u.)')
axs[1,0].set_ylabel('z (a.u.)')
axs[1,0].set_xlabel('x (a.u.)')
axs[1,1].set_xlabel('x (a.u.)')
#Write the titles.
axs[0,0].set_title('Linear response')
axs[0,1].set_title(r'$3.8\times 10^{11}$ W/cm$^2$')
axs[1,0].set_title(r'$5\times 10^{12}$ W/cm$^2$')
axs[1,1].set_title(r'$10^{13}$ W/cm$^2$')

plt.savefig('jphi.png',dpi=500)
plt.close()