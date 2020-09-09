import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import get_window
from matplotlib.ticker import MultipleLocator,LogLocator
from matplotlib.colors import LogNorm

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


def hhg_intensity_heatmap(dipole,intensties,interpolation_matrix,intensity_axis=1,**kwargs):
	'''Example of how to create a heatmap showing HHG spectra as a function of intensity.
	You can specify in the arguments which axis is intensity and which is time.'''
	freq,spec = fourier_transform(dipole,**kwargs)






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