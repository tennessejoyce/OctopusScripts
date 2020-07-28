#Calculates Macroscopic HHG 

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,LogLocator
from matplotlib.colors import LogNorm
import scipy.signal as sig
import scipy.special as sp
from scipy.interpolate import interp1d as interp
plt.switch_backend('agg')

#Physical parameters
omega =  0.072014	#Fundamental frequency
beam_width = 5.67e5 #30 micrometers converted to atomic units
thickness = 2*9.45e6	#0.5 millimeters, thickness of gas jet.
kappa_factor = beam_width*omega/137  #kappa = kappa_factor * harmonic_number * sin(theta)
g_factor = thickness*omega/137
mu_dipole = 0.36 #Transition dipole for Rabi frequency
excited_energy = (1.681640 - 1.059454)/omega

#Numerical parameters
Ntdse = 16  		#How many TDSE calculations to read in
maxKappa = 50		#Maximum value of kappa to calculate.
num_kappa = 1000
dt = 0.1			#Time step for the dipole moment data from TDSE
xMax = 500			#Cutoff for Hankel transform
numPoints = 100		#Number of points for Hankel transform
num_levels = 40

#Plotting parameters
max_theta = 5e-3	#Maximum value of theta (macroscopic angle) to plot.
low = 4
high = 30
which_helicity = 'right'

#Which plots to make
plot_raw_sweep = False
plotTDSE = True		#Plots the microscopic HHG spectra from the individual TDSE calculations.
plotFwd = False		#Plot the macroscopic signal in the forward direction

calcTM=False		#Whether to calculate t_m(kappa) from scratch, or read from a file.
calcFT=False	#Whether to calculate the microscopic HHG spectra, or read from a file.
recalculate_macroscopic = True


def bary_cheb2(x):
	#Construct the interpolation matrix onto field amplitudes x.
	#Chebyshev nodes of the second kind on the half interval [0,1].
	k = np.array(range(2*Ntdse+1))
	nodes = np.cos(np.pi*k/(2*Ntdse))
	#Weights for barycentric interpolation.
	weights = (-1.0)**k
	weights[0] /= 2.0
	weights[-1] /= 2.0
	#Numerator of the barycentric interpolation formula.
	numerator = weights[np.newaxis,:]/(x[:,np.newaxis] - nodes[np.newaxis,:])
	#Denominator of the barycentric interpolation formula.
	denominator = np.sum(numerator,axis=1)
	#Combine into the full barycentric interpolation matrix.
	bary = numerator/denominator[:,np.newaxis]
	#Since our functions are antisymmetric, we only need samples k=0 through k=Ntdse-1.
	bary_antisym = bary[:,:Ntdse] - bary[:,-Ntdse:][:,::-1]
	#Window
	#window = 1-np.exp(-100*nodes[:Ntdse]**2)
	return bary_antisym#*window[np.newaxis,:]


kappa = np.linspace(1e-3,maxKappa,num_kappa)
if calcTM:
	zeros = sp.jn_zeros(0,numPoints)
	int_nodes = zeros/np.pi/np.sqrt(numPoints)
	int_weights = 2/(np.pi**2*numPoints)/(sp.jv(1,zeros)**2)
	bessel_eval = sp.jv(0,np.outer(kappa,int_nodes))
	interp_matrix = bary_cheb2(np.exp(-int_nodes**2/2))
	tInt = bessel_eval.dot(int_weights[:,np.newaxis]*interp_matrix)
	np.savetxt("tInt.dat",tInt)
else:
	tInt = np.loadtxt('tInt.dat')


#Calculates the harmonic spectrum from the dipole moment.
def getSpec(data,maxHN=20):
	tmax = 10*dt*len(data)
	dw = 2*np.pi/tmax
	freq = np.arange(0,maxHN,dw/omega)
	vel = np.gradient(np.gradient(data))*sig.blackman(len(data)) #Take a derivative to get velocity, then multiply by a window function
	spec=np.fft.rfft(vel,n=10*len(vel))
	return freq[1:],spec[1:len(freq)]

#Calculate microsocpic HHG spectra
if calcFT:
	print("Calculating microsocpic HHG spectra...")
	allDipoles=np.load('dipole.npy')
	if not os.path.isdir('hhgSpec'):
		os.mkdir('hhgSpec')
	total_left=[]
	total_right=[]
	for i in range(Ntdse):
		dipoleY = allDipoles[0,i,:]
		dipoleZ = allDipoles[1,i,:]
		freq, specY = getSpec(dipoleY,maxHN=50)
		freq, specZ = getSpec(dipoleZ,maxHN=50)
		spec_left = (specY + 1j*specZ)/np.sqrt(2)
		spec_right = (specY - 1j*specZ)/np.sqrt(2)
		fig,(ax1,ax2) = plt.subplots(2,1,figsize=[7,5])
		if plotTDSE:
			#Plot the harmonic spectrum
			ax2.semilogy(freq,np.abs(spec_left)**2,color='red')
			ax2.semilogy(freq,np.abs(spec_right)**2,color='blue')
			ax2.set_xticks(range(1,14,2))
			ax2.set_yticks([])
			ax2.grid(True,axis="x")
			ax2.set_xlabel("harmonic number")
			ax2.set_ylabel("log yield (arb. units)")
			plt.tight_layout()
			plt.savefig("hhgSpec/"+str(i)+".png",dpi=500)
			plt.close()
		total_left.append(spec_left)
		total_right.append(spec_right)	
	total_left = np.array(total_left)
	total_right = np.array(total_right)
	np.save("HHG_left.npy",total_left)
	np.save("HHG_right.npy",total_right)
	np.save("HHGfreq.npy",freq)
else:
	print("Loading microsocpic HHG spectra from file...")
	total_left = np.load("HHG_left.npy")
	total_right = np.load("HHG_right.npy")
	freq = np.load("HHGfreq.npy")


#Shift the phases to be relative to the peak of the pulse.
t_peak = 872.5/2
total_left *= np.exp(1j*t_peak*freq[None,:]*omega)
total_right *= np.exp(1j*t_peak*freq[None,:]*omega)

#Upsample as a function of electric field amplitude.
x_up = np.sqrt(np.linspace(0,1,1000))-1e-8
hhg_up_left = bary_cheb2(x_up).dot(total_left)
hhg_up_right = bary_cheb2(x_up).dot(total_right)


def macro_response(freq,spec,theta):
	#Apply the transformation matrix to get H(omega,kappa)
	hwk = tInt.dot(spec)
	#Next, we change variables from kappa to theta by interpolating
	#onto the desired values of theta provided in the argument.
	macro_hhg = []
	for hn,col in zip(freq,np.transpose(hwk)): #Iterate over frequencies.
		#Response is an even function of r.
		double_col = np.concatenate([col,col])
		double_kappa = np.concatenate([kappa,-kappa])
		#The values of theta correponding to kappa for this particular frequency.
		sample_theta = np.arcsin(double_kappa/kappa_factor/hn)
		#Do the interpolation.
		macro_hhg.append(interp(sample_theta,double_col,kind='cubic',fill_value=0.0,bounds_error=False)(theta))
	return np.transpose(np.array(macro_hhg))


print('Macroscopic response...')
#Values for outgoing angles
theta = np.linspace(0,max_theta,1000)
if recalculate_macroscopic:
	#Compute H(omega,theta)
	macro_hhg_left = macro_response(freq,total_left,theta)
	macro_hhg_right = macro_response(freq,total_right,theta)
	#Multiply by G(kappa)
	g_exp = g_factor*np.outer(freq,np.sin(theta/2)**2)
	g = np.exp(-0.5*g_exp**2)
	macro_hhg_left *= np.transpose(g)
	macro_hhg_right *= np.transpose(g)

	#Write to a file
	np.save('macro_hhg_left.npy',macro_hhg_left)
	np.save('macro_hhg_right.npy',macro_hhg_right)
else:
	macro_hhg_left = np.load('macro_hhg_left.npy')
	macro_hhg_right = np.load('macro_hhg_right.npy')



def phase_heatmap(x,y,z,ax,noise_threshold=0.2):
	z_mask = np.abs(z)/np.max(np.abs(z)) < np.sqrt(noise_threshold)
	phase = np.angle(z)
	phase[z_mask] = np.nan#np.pi-1e-10
	color_map = ax.contourf(y,x,np.transpose(phase),100,cmap='hsv')
	color_map.get_cmap().set_bad(color='grey')
	ax.set_ylabel('harmonic order')
	ticks = np.arange(0,20,0.5)
	ticks_mask = (ticks>=x[0]-0.1)*(ticks<=x[-1]+0.1)
	ticks = ticks[ticks_mask]
	#ax.set_xticks(ticks)
	ax.yaxis.set_major_locator(MultipleLocator(1))
	ax.yaxis.set_minor_locator(MultipleLocator(0.25))
	ax.tick_params(axis='y',right=True,which='both')
	plt.colorbar(color_map,ax=ax,ticks=[-np.pi,-np.pi/2,0,np.pi/2,np.pi-1e-10])

def delay_heatmap(x,y,z,ax,noise_threshold=0.2):
	z_mask = np.abs(z)/np.max(np.abs(z)) < np.sqrt(noise_threshold)
	delay = -np.gradient(np.unwrap(np.angle(z)),axis=1)/(x[1]-x[0])/(2*np.pi)
	delay = np.maximum(np.minimum(delay,2),-2)
	delay[z_mask] = np.nan#np.min(delay[1-z_mask])
	color_map = ax.contourf(y,x,np.transpose(delay),100,cmap='jet')
	color_map.get_cmap().set_bad(color='grey')
	ax.set_ylabel('harmonic order')
	ticks = np.arange(0,20,0.5)
	ticks_mask = (ticks>=x[0]-0.1)*(ticks<=x[-1]+0.1)
	ticks = ticks[ticks_mask]
	#ax.set_xticks(ticks)
	ax.yaxis.set_major_locator(MultipleLocator(1))
	ax.yaxis.set_minor_locator(MultipleLocator(0.25))
	ax.tick_params(axis='y',right=True,which='both')
	plt.colorbar(color_map,ax=ax,ticks=range(-3,4))

def heatmap(x,y,z,ax,noise_threshold=0.2):
	z = np.abs(z)**2
	z = np.maximum(np.abs(z)/np.max(np.abs(z)),noise_threshold)
	#Divide by column-wise maximum.
	#z = np.abs(z)/np.amax(np.abs(z),axis=1)[:,None]
	#Truncate below the noise threshold.
	z = np.maximum(z,noise_threshold)
	color_map = ax.contourf(y,x,np.transpose(z),levels=np.geomspace(noise_threshold,1,num_levels),norm=LogNorm(),cmap='jet')
	ax.set_ylabel('harmonic order')
	ticks = np.arange(0,20,0.5)
	ticks_mask = (ticks>=x[0]-0.1)*(ticks<=x[-1]+0.1)
	ticks = ticks[ticks_mask]
	#ax.set_xticks(ticks)
	ax.yaxis.set_major_locator(MultipleLocator(1))
	ax.yaxis.set_minor_locator(MultipleLocator(0.25))
	ax.tick_params(axis='y',right=True,which='both')
	plt.colorbar(color_map,ax=ax,ticks=[1.0,0.1,0.01,0.001])

print('Plotting...')

if which_helicity == 'right':
	macro_hhg = macro_hhg_right
	hhg_up = hhg_up_right
else:
	macro_hhg = macro_hhg_left
	hhg_up = hhg_up_left


#Microscopic plots
micro_fig,micro_axs = plt.subplots(1,2,figsize=(8,6))
micro_phase_ax = micro_axs[0]
micro_delay_ax = None#micro_axs[1]
micro_magnitude_ax = micro_axs[1]

noise_threshold = 1e-4

mask = (freq>=low)*(freq<=high)
if micro_phase_ax!=None:
	phase_heatmap(freq[mask],3*x_up**2,hhg_up[:,mask],micro_phase_ax,noise_threshold)
	micro_phase_ax.set_xlabel(r'peak intensity ($10^{14}$ W/cm$^2$)')
	micro_phase_ax.set_title('Single molecule phase')
if micro_delay_ax!=None:
	delay_heatmap(freq[mask],3*x_up**2,hhg_up[:,mask],micro_delay_ax,noise_threshold)
	micro_delay_ax.set_xlabel(r'peak intensity ($10^{14}$ W/cm$^2$)')
	micro_delay_ax.set_title('Single molecule group delay (in cycles)')
if micro_magnitude_ax!=None:
	heatmap(freq[mask],3*x_up**2,hhg_up[:,mask],micro_magnitude_ax,noise_threshold)
	# for hn in range(1,int(high),3):
	# 	if hn>low:
	# 		micro_magnitude_ax.plot(3*x_up**2,hn+x_up*0.09245*mu_dipole/omega,color='red',linestyle='--')
	# 		micro_magnitude_ax.plot(3*x_up**2,hn-x_up*0.09245*mu_dipole/omega,color='red',linestyle='--')
	#micro_magnitude_ax.axhline(excited_energy,color='blue',linestyle=':')
	micro_magnitude_ax.set_xlabel(r'peak intensity ($10^{14}$ W/cm$^2$)')
	micro_magnitude_ax.set_title('Single molecule spectrum')

#Save all the plots together.
plt.tight_layout()
plt.savefig('micro_all.png',dpi=500)
plt.close()

#Macroscopic plots
macro_fig,macro_axs = plt.subplots(1,2,figsize=(8,6))
macro_phase_ax = macro_axs[0]
macro_delay_ax = None#macro_axs[1]
macro_magnitude_ax = macro_axs[1]

theta_mask = theta < max_theta
scale_mask = (freq>10)
scale_factor = 1*(1-scale_mask) + np.sqrt(10)*scale_mask
scaled_macro_hhg = macro_hhg[theta_mask] #scale_factor[np.newaxis,:]*macro_hhg[theta_mask]
if macro_magnitude_ax!=None:
	heatmap(freq[mask],theta[theta_mask]*1000,scaled_macro_hhg[:,mask],macro_magnitude_ax,noise_threshold)
	macro_magnitude_ax.set_xlabel('outgoing angle (mrad)')
	macro_magnitude_ax.set_title('Macroscopic spectrum')
if macro_phase_ax!=None:
	phase_heatmap(freq[mask],theta[theta_mask]*1000,scaled_macro_hhg[:,mask],macro_phase_ax,noise_threshold)
	macro_phase_ax.set_xlabel('outgoing angle (mrad)')
	macro_phase_ax.set_title('Macroscopic phase')
if macro_delay_ax!=None:
	delay_heatmap(freq[mask],theta[theta_mask]*1000,scaled_macro_hhg[:,mask],macro_delay_ax,noise_threshold)
	macro_delay_ax.set_xlabel('outgoing angle (mrad)')
	macro_delay_ax.set_title('Macroscopic group delay (in cycles)')


#Save all the plots together.
plt.tight_layout()
plt.savefig('macro_all.png',dpi=500)
plt.close()
exit()


# Plot the forward direction
# Mask to select relevant frequency components.
low_fwd = 4.2
high_fwd = 12
mask = (freq>=low_fwd)*(freq<=high_fwd)
freq_mask = freq[mask]
#Forward direction
hhg_fwd = macro_hhg[0,mask]
#Angle integrated
weight = (1+np.cos(theta)**2)*np.sin(theta)
hhg_ai = np.transpose(macro_hhg[:,mask]).dot(weight)
#Maximum intensity
hhg_max = total[0,mask]
#Match at the specified harmonic
match_harmonic = 5
match_index = np.argmin(np.abs(freq_mask-match_harmonic))
#Loop over each one, match, and plot.
#plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.figsize': (0.8*6,0.8*4)})
hhg_types = [hhg_max,hhg_ai]
styles = ['-','--']
labels = ['single-molecule','macroscopic, angle-integrated']
for h,s,l in zip(hhg_types,styles,labels):
	h /= h[match_index]
	h = np.abs(h)**2
	plt.semilogy(freq_mask,h,linestyle=s,label=l)
plt.xticks(range(int(low_fwd)+1,int(high_fwd)+1,2))
plt.legend()
plt.grid(axis='x')
plt.xlabel('harmonic number')
plt.ylabel('HHG yield (arb. units)')
plt.tight_layout()
plt.savefig('compare3.png',dpi=500)
plt.close()






