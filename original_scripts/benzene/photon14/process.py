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

sampleDxFile = '1/output_iter/td.0000000/current-x.dx'

#Open a wavefunction file to find the mesh.
with open(sampleDxFile) as f:
  s1 = f.readline()
  s2 = f.readline()
  wfShape = [int(n) for n in s1.split()[-3:]]
  boxLengths = [-float(n) for n in s2.split()[-3:]]
  numPoints = np.prod(wfShape)
  dx = 2*boxLengths[0]/(wfShape[0]-1)
#print(f'Mesh size: {wfShape}')
#print(f'Box size: {boxLengths}')
#print(f'Spacing: {dx}')
print('Constructing mesh...')
#Construct the mesh
mesh1D = [np.linspace(-b,b,n) for b,n in zip(boxLengths,wfShape)]
x,y,z = np.meshgrid(mesh1D[0],mesh1D[1],mesh1D[2],indexing='ij')
rho = np.sqrt(x**2 + y**2+1e-8)
lzop = np.array([y,-x])
mask = x**2+y**2+z**2 <= 6**2


read_dx=False
if read_dx:
	density_XY_all = []
	ionization_all = []
	lz_all = []
	for intens in range(1,9):
		density_XY = []
		ionization = []
		lz = []
		for time_folder in sorted(glob(str(intens)+'/output_iter/*'))[-2:]:
			print(time_folder)
			#Compute density integrated over z
			#density = pd.read_csv(time_folder+'/density.dx',header=None,skiprows=7,nrows=numPoints).values	
			#density = np.reshape(density,wfShape)
			#density_XY.append(np.sum(density,axis=-1)*dx)

			#Compute ionization (leaving box) maybe with different masks.
			#ionization.append(np.sum(density)*dx**3)
			multi = np.loadtxt(str(intens)+'/td.general/multipoles')
			ionization.append(multi[-1,2])
			
			#Compute lz, maybe with different masks.
			curX = pd.read_csv(time_folder+'/current-x.dx',header=None,skiprows=7,nrows=numPoints).values
			curX = np.reshape(curX,wfShape)
			curY = pd.read_csv(time_folder+'/current-y.dx',header=None,skiprows=7,nrows=numPoints).values
			curY = np.reshape(curY,wfShape)
			lz.append(np.sum(y*curX - x*curY)*dx**3)
		#density_XY_all.append(density_XY)
		ionization_all.append(ionization)
		lz_all.append(lz)


	#density_XY_all = np.array(density_XY_all)
	ionization_all = np.array(ionization_all)
	lz_all = np.array(lz_all)

	#np.save('density_XY_test.npy',density_XY_all)
	np.save('ionization_test.npy',ionization_all)
	np.save('lz_test.npy',lz_all)
else:
	#density_XY_all = np.load('density_XY_all.npy',allow_pickle=True)
	ionization_test = np.load('ionization_test.npy',allow_pickle=True)
	lz_test = np.load('lz_test.npy',allow_pickle=True)

ionization_all = np.load('ionization_all.npy',allow_pickle=True)
lz_all = np.load('lz_all.npy',allow_pickle=True)




lz_all = lz_all[:,-1]
ionization_all = 30-ionization_all[:,-1]
lz_test = lz_test[:,-1]
ionization_test = 30-ionization_test[:,-1]
print(ionization_all)
print('lz_all',lz_all.shape)
print('ionization_all',ionization_all.shape)

lz_all = np.concatenate([np.flip(lz_all,axis=0),[0],lz_all])
ionization_all = np.concatenate([np.flip(ionization_all,axis=0),[0],ionization_all])

intens_test = 1e13*np.sin(0.5*np.pi*np.array(range(1,9))/8)**4

efield_low = -np.cos(0.5*np.pi*np.array(range(0,17))/8)
intens_low = 1e13*efield_low**2
intens_high = np.geomspace(1e11,1e13,100)
efield_high = np.sqrt(intens_high/1e13)

lz_high = barycentric_interpolate(efield_low,lz_all,efield_high)
ionization_high = barycentric_interpolate(efield_low,ionization_all,efield_high)

ratio=0.8
plt.figure(figsize=(6.4*ratio,4.8*ratio))
#final_ionization = np.sum(ionization_high,axis=1)*time_step
#final_lz = lzAll_high[:,-1]
duration = 202.58
rabi_freq = efield_high*1.81*np.sqrt(1e13/(3.51e16))
few_level = 0.78*np.sin(0.25*duration*rabi_freq)**2
plt.semilogx(intens_high,lz_high)
plt.semilogx(intens_high,few_level,linestyle='--')
plt.semilogx(intens_high,ionization_high,linestyle=':')
plt.semilogx(intens_low[-8:],ionization_all[-8:],linestyle='None',marker='x',color='black')
plt.semilogx(intens_test[1:4],lz_test[1:4],linestyle='None',marker='+',color='red')
plt.semilogx(intens_test[1:4],ionization_test[1:4],linestyle='None',marker='+',color='red')
plt.semilogx(intens_low[-8:],lz_all[-8:],linestyle='None',marker='x',color='black')
plt.legend([r'$L_z$ (Interpolated)',r'$L_z$ (Few level)','Ionization probability (Interpolated)','TDDFT calculations','Test points'])
plt.xlabel(r'Peak intensity (W/cm$^2$)')
plt.ylabel(r'$L_z$ (a.u.) or probability')
plt.tight_layout()
plt.savefig('comparison_test.eps')
plt.close()

# ratio=0.8
# plt.figure(figsize=(6.4*ratio,4.8*ratio))
# #final_ionization = np.sum(ionization_high,axis=1)*time_step
# #final_lz = lzAll_high[:,-1]
# duration = 202.58
# rabi_freq = efield_high*1.81*np.sqrt(1e13/(3.51e16))
# few_level = 0.78*np.sin(0.25*duration*rabi_freq)**2
# plt.semilogx(intens_high,lz_high)
# plt.semilogx(intens_high,few_level,linestyle='--')
# plt.semilogx(intens_high,ionization_high,linestyle=':')
# plt.semilogx(intens_low[-8:],lz_all[-8:],linestyle='None',marker='x',color='black')
# plt.semilogx(intens_low[-8:],ionization_all[-8:],linestyle='None',marker='x',color='black')
# plt.legend([r'$L_z$ (TDDFT)',r'$L_z$ (Few level)','Ionization probability','Samples'])
# plt.xlabel(r'Peak intensity (W/cm$^2$)')
# plt.ylabel(r'$L_z$ (a.u.) or probability')
# plt.tight_layout()
# plt.savefig('comparison.png')
# plt.close()
