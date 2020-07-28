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

sampleDxFile = '1/output_iter/td.0000000/density.dx'

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


read_dx=False
if read_dx:
	density_XY_all = []
	ionization_all = []
	lz_all = []
	for intens in range(1,9):
		density_XY = []
		ionization = []
		lz = []
		for time_folder in glob(f'{intens}/output_iter/*')[-2:]:
			print(time_folder)
			#Compute density integrated over z
			density = pd.read_csv(time_folder+'/density.dx',header=None,skiprows=7,nrows=numPoints).values	
			density = np.reshape(density,wfShape)
			density_XY.append(np.sum(density,axis=-1)*dx)

			#Compute ionization (leaving box) maybe with different masks.
			ionization.append(np.sum(density)*dx**3)
			
			#Compute lz, maybe with different masks.
			curX = pd.read_csv(time_folder+'/current-x.dx',header=None,skiprows=7,nrows=numPoints).values
			curX = np.reshape(curX,wfShape)
			curY = pd.read_csv(time_folder+'/current-y.dx',header=None,skiprows=7,nrows=numPoints).values
			curY = np.reshape(curY,wfShape)
			lz.append(np.sum(y*curX - x*curY)*dx**3)
		density_XY_all.append(density_XY)
		ionization_all.append(ionization)
		lz_all.append(lz)


	density_XY_all = np.array(density_XY_all)
	ionization_all = np.array(ionization_all)
	lz_all = np.array(lz_all)

	np.save('density_XY_all.npy',density_XY_all)
	np.save('ionization_all.npy',ionization_all)
	np.save('lz_all.npy',lz_all)
else:
	density_XY_all = np.load('density_XY_all.npy',allow_pickle=True)
	ionization_all = np.load('ionization_all.npy',allow_pickle=True)
	lz_all = np.load('lz_all.npy',allow_pickle=True)

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


compute_occupations = False
if compute_occupations:
	#Read in field-free orbitals.
	field_free_wfs = []
	for i in range(1,48):
		wf = pd.read_csv(f'1/static/wf-st00{str(i).zfill(2)}.dx',header=None,skiprows=7,nrows=numPoints).values	
		wf = np.reshape(wf,wfShape)
		field_free_wfs.append(wf)

	homo_proj_right = []
	lumo_proj_right = []
	ionization_right = []
	homo_proj_left = []
	lumo_proj_left = []
	ionization_left = []
	proj_all_orbitals = []
	for intens in range(1,9):
		#Read in the wavefunctions after the pulse.
		wf14 = pd.read_csv(f'{intens}/output_iter/td.0004044/wf-st0014.dx',header=None,skiprows=7,nrows=numPoints,delim_whitespace=True).values	
		print(wf14.shape)
		wf14 = np.reshape(wf14[:,0]+1j*wf14[:,1],wfShape)
		wf15 = pd.read_csv(f'{intens}/output_iter/td.0004044/wf-st0015.dx',header=None,skiprows=7,nrows=numPoints,delim_whitespace=True).values	
		wf15 = np.reshape(wf15[:,0]+1j*wf15[:,1],wfShape)
		wf_right = (wf14+1j*wf15)/np.sqrt(2)
		wf_left = (wf14-1j*wf15)/np.sqrt(2)
		#Project onto field-free orbitals
		temp_right = []
		temp_left = []
		for ff in field_free_wfs:
			temp_right.append(np.abs(np.sum(wf_right*ff)*dx**3)**2 )
			temp_left.append(np.abs(np.sum(wf_left*ff)*dx**3)**2 )
		proj_all_orbitals.append([temp_right,temp_left])
		homo_proj_right.append(temp_right[14-1]+temp_right[15-1])
		homo_proj_left.append(temp_left[14-1]+temp_left[15-1])
		lumo_proj_right.append(temp_right[16-1]+temp_right[17-1])
		lumo_proj_left.append(temp_left[16-1]+temp_left[17-1])
		ionization_right.append(1-np.sum(np.abs(wf_right)**2)*dx**3)
		ionization_left.append(1-np.sum(np.abs(wf_left)**2)*dx**3)
		# homo_proj_right.append(np.abs(np.sum(wf_right*field_free_wfs[0])*dx**3)**2 + np.abs(np.sum(wf_right*field_free_wfs[1])*dx**3)**2)
		# homo_proj_left.append(np.abs(np.sum(wf_left*field_free_wfs[0])*dx**3)**2 + np.abs(np.sum(wf_left*field_free_wfs[1])*dx**3)**2)
		# lumo_proj_right.append(np.abs(np.sum(wf_right*field_free_wfs[2])*dx**3)**2 + np.abs(np.sum(wf_right*field_free_wfs[3])*dx**3)**2)
		# lumo_proj_left.append(np.abs(np.sum(wf_left*field_free_wfs[2])*dx**3)**2 + np.abs(np.sum(wf_left*field_free_wfs[3])*dx**3)**2)
		# ionization_right.append(1-np.sum(np.abs(wf_right)**2)*dx**3)
		# ionization_left.append(1-np.sum(np.abs(wf_left)**2)*dx**3)
	proj_all_orbitals = np.array(proj_all_orbitals)
	print(proj_all_orbitals.shape)
	proj_all = np.array([homo_proj_right,lumo_proj_right,ionization_right,homo_proj_left,lumo_proj_left,ionization_left])
	np.save('proj_all.npy',proj_all)
	np.save('proj_all_orbitals.npy',proj_all_orbitals)
else:
	proj_all = np.load('proj_all.npy')
	proj_all_orbitals = np.load('proj_all_orbitals.npy')




plot_occupations_new = False
if plot_occupations_new:
	proj_low = []
	proj_low.append(proj_all_orbitals[:,:,11-1])
	proj_low.append(proj_all_orbitals[:,:,14-1]+proj_all_orbitals[:,:,15-1])
	proj_low.append(proj_all_orbitals[:,:,16-1]+proj_all_orbitals[:,:,17-1])
	proj_low.append(proj_all_orbitals[:,:,21-1])
	ionization = np.transpose(np.array([proj_all[2],proj_all[5]]))
	proj_low.append(ionization)
	#proj_low.append(1-np.sum(proj_low,axis=0))
	proj_low = np.array(proj_low)
	proj_low= np.concatenate([np.transpose(np.array([[0,1,0,0,0],[0,1,0,0,0]]))[:,None,:],proj_low],axis=1)
	print(proj_low.shape)
	
	efield_low = np.sin(0.5*np.pi*np.array(range(0,9))/8)
	intens_low = 1e13*efield_low**2
	intens_high = np.linspace(0,1e13,100)

	proj_high = barycentric_interpolate(intens_low,proj_low,intens_high,axis=1)

	fig,axs = plt.subplots(1,2,sharey=True)
	axs[0].plot(intens_high,proj_high[1,:,0],label='HOMO')
	axs[0].plot(intens_high,proj_high[2,:,0],label='LUMO',linestyle='--')
	axs[0].plot(intens_high,proj_high[4,:,0],label='Ionized',linestyle=':')
	other_right = 1 - proj_high[1,:,0] - proj_high[2,:,0] - proj_high[4,:,0]
	axs[0].plot(intens_high,other_right,label='Other',linestyle='-.')
	axs[0].plot(intens_low[1:],proj_low[1,1:,0],linestyle='None',marker='x',color='black',label='Samples')
	axs[0].set_ylim([0,1])
	#axs[0].axhline(y=0,color='black',linewidth=1)
	axs[0].legend()
	
	axs[0].set_title('Co-rotating electron')
	axs[0].set_ylabel('probability')
	axs[0].set_xlabel(r'intensity [W/cm$^2$]')

	#axs[1].plot(intens_high,proj_high[0,:,1],label='11')	
	axs[1].plot(intens_high,proj_high[1,:,1],label='HOMO')	
	#axs[1].plot(intens_high,proj_high[2,:,1],label='LUMO',linestyle=':')	
	axs[1].plot(intens_high,proj_high[3,:,1],label='21',linestyle='--')
	axs[1].plot(intens_high,proj_high[4,:,1],label='Ionized',linestyle=':')
	other_left = 1 - proj_high[1,:,1] - proj_high[3,:,1] - proj_high[4,:,1]
	axs[1].plot(intens_high,other_left,label='Other',linestyle='-.')
	axs[1].set_ylim([0,1])
	#axs[1].axhline(y=0,color='black',linewidth=1)
	axs[1].plot(intens_low[1:],proj_low[1,1:,1],linestyle='None',marker='x',color='black',label='Samples')
	axs[1].legend()

	axs[1].set_title('Counter-rotating electron')
	axs[1].set_xlabel(r'intensity [W/cm$^2$]')
	plt.savefig('new_proj.png')


plot_occupations = False
if plot_occupations:
	#Add the projections at zero intensity.
	print(proj_all.shape)
	proj_low= np.concatenate([np.array([1,0,0,1,0,0])[:,None],proj_all],axis=1)
	print(proj_low.shape)
	#Add the other category.
	other_right = 1-proj_low[0] - proj_low[1] - proj_low[2]
	other_left = 1-proj_low[3] - proj_low[4] - proj_low[5]
	proj_low= np.concatenate([proj_low,other_right[None,:],other_left[None,:]],axis=0)
	print(proj_low.shape)
	#Interpolate to higher intensity resolution.
	efield_low = np.sin(0.5*np.pi*np.array(range(0,9))/8)
	intens_low = 1e13*efield_low**2
	intens_high = np.linspace(0,1e13,100)
	proj_high = barycentric_interpolate(intens_low,proj_low,intens_high,axis=1)
	#Plot the projections
	fig,axs = plt.subplots(1,2)
	axs[0].plot(intens_high[1:],proj_high[0,1:],label='HOMO')
	axs[0].plot(intens_high[1:],proj_high[1,1:],label='LUMO',linestyle='--')
	axs[0].plot(intens_high[1:],proj_high[2,1:],label='ionization',linestyle=':')
	axs[0].plot(intens_high[1:],proj_high[6,1:],label='other',linestyle='-.')
	axs[0].legend()
	axs[0].plot(intens_low[1:],proj_low[0,1:],linestyle='None',marker='x',color='black')
	axs[0].set_title('Co-rotating electron')
	axs[0].set_ylabel('probability')
	axs[0].set_xlabel(r'intensity [W/cm$^2$]')
	#axs[0].set_ylim(1e-4,1)
	#axs[0].set_xlim(1e11,1e13)

	axs[1].plot(intens_high[1:],proj_high[3,1:],label='HOMO')
	axs[1].plot(intens_high[1:],proj_high[4,1:],label='LUMO',linestyle='--')
	axs[1].plot(intens_high[1:],proj_high[5,1:],label='ionization',linestyle=':')
	axs[1].plot(intens_high[1:],proj_high[7,1:],label='other',linestyle='-.')
	axs[1].plot(intens_low[1:],proj_low[3,1:],linestyle='None',marker='x',color='black')
	axs[1].set_title('Counter-rotating electron')
	axs[1].set_xlabel(r'intensity [W/cm$^2$]')
	#axs[1].set_ylim(-0.1,1)
	#axs[1].set_xlim(1e11,1e13)
	axs[1].legend()
	plt.savefig('proj.png')
	plt.close()


lz_all = lz_all[:,-1]
ionization_all = 30-ionization_all[:,-1]
print(ionization_all)
print('lz_all',lz_all.shape)
print('ionization_all',ionization_all.shape)

lz_all = np.concatenate([np.flip(lz_all,axis=0),[0],lz_all])
ionization_all = np.concatenate([np.flip(ionization_all,axis=0),[0],ionization_all])

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
#plt.semilogx(intens_low[-8:],lz_all[-8:],linestyle='None',marker='x',color='black')
#plt.semilogx(intens_low[-8:],ionization_all[-8:],linestyle='None',marker='x',color='black')
plt.legend([r'$L_z$ (TDDFT)',r'$L_z$ (Few level)','Ionization probability (TDDFT)'])
plt.xlabel(r'Peak intensity (W/cm$^2$)')
plt.ylabel(r'$L_z$ (a.u.) or probability')
plt.tight_layout()
plt.savefig('comparison.eps')
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
