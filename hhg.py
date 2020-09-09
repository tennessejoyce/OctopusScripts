import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import get_window
from matplotlib.ticker import MultipleLocator,LogLocator
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d as interp

def fourier_transform(dipole,dt,omega,max_harmonic=30,zero_padding_factor=10,window='blackman',t_phase=0,time_axis=0):
	'''Handles the Fourier transform for HHG with all the extra subtleties.
			'dipole' is the time-dependent dipole moment (as a numpy array)
			'dt' is the time step
			'omega' is the reference frequency, to measure the harmonics against
			'max_harmonic' is the highest frequency harmonic to return
			'zero_padding_factor' is the factor by which the signal duration is increased via zero padding
			'window' is the name of the window to use (see scipy.signal for the options)
			't_phase' is the time relative to which the phase of the harmonics is measured. Typically the peak of the laser pulse.
			'axis' is used when 'dipole' is a multidimensional array to indicate which axis is time
		This function returns both the frequencies (in units of omega) and the (complex) hhg spectrum.
	'''
	#Total duration of the signal, after zero padding
	num_samples = dipole.shape[time_axis]
	t_max = zero_padding_factor*dt*num_samples
	#Spacing in frequency domain
	dw = 2*np.pi/t_max
	#Frequencies in units of omega
	freq = np.arange(0,max_harmonic,dw/omega)
	#Calculate the dipole acceleration in time domain by taking the second derivative.
	dipole = np.gradient(np.gradient(dipole,dt,axis=time_axis),dt,axis=time_axis)
	#Multiply by the window function
	dipole *= get_window(window,num_samples)
	#Compute the Fourier transform with zero-padding
	spec=np.fft.rfft(dipole,n=zero_padding_factor*num_samples,axis=time_axis)
	#Shift the phase to be measured relative to 't_phase'.
	spec = multiply_along_axis(spec,np.exp(1j*t_phase*freq*omega))
	#Truncate the spectrum at the max harmonic, and also remove
	#the first point to avoid division by zero later
	return freq[1:],spec[1:len(freq)]


def multiply_along_axis(a,b,axis=0):
	'''Multiply an array 'a' of arbitrary dimension by a 1d array 'b' along a given axis.
	Surprisingly there's not a built-in numpy function for this.'''
	dim_array = np.ones(a.ndim)
	dim_array[axis]=-1
	return a*np.reshape(b,dim_array)



def plot_hhg_spectrum(freq,spec):
	'''Example plotting script for an HHG spectrum.'''
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


def magnitude_heatmap(x,y,z,ax,noise_threshold=0.2):
	'''
		Heatmap for the magnitude of the HHG spectrum.
			'x' is either intensity or macroscopic angle (works for either)
			'y' is the harmonic number ('freq')
			'z' is the harmonic spectrum (complex amplitude)
			'ax' is the axes on which the plot will be drawn.
	'''
	z = np.abs(z)**2
	z = np.maximum(np.abs(z)/np.max(np.abs(z)),noise_threshold)
	#Truncate below the noise threshold.
	z = np.maximum(z,noise_threshold)
	color_map = ax.contourf(y,x,np.transpose(z),levels=np.geomspace(noise_threshold,1,num_levels),norm=LogNorm(),cmap='jet')
	ax.set_ylabel('harmonic order')
	ticks = np.arange(0,20,0.5)
	ticks_mask = (ticks>=x[0]-0.1)*(ticks<=x[-1]+0.1)
	ticks = ticks[ticks_mask]
	ax.yaxis.set_major_locator(MultipleLocator(1))
	ax.yaxis.set_minor_locator(MultipleLocator(0.25))
	ax.tick_params(axis='y',right=True,which='both')
	plt.colorbar(color_map,ax=ax,ticks=[1.0,0.1,0.01,0.001])


def phase_heatmap(x,y,z,ax,noise_threshold=0.2):
	z_mask = np.abs(z)/np.max(np.abs(z)) < np.sqrt(noise_threshold)
	phase = np.angle(z)
	phase[z_mask] = np.nan
	color_map = ax.contourf(y,x,np.transpose(phase),100,cmap='hsv')
	color_map.get_cmap().set_bad(color='grey')
	ax.set_ylabel('harmonic order')
	ticks = np.arange(0,20,0.5)
	ticks_mask = (ticks>=x[0]-0.1)*(ticks<=x[-1]+0.1)
	ticks = ticks[ticks_mask]
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
