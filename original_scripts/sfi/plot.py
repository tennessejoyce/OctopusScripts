import numpy as np
from scipy.interpolate import barycentric_interpolate
from scipy.optimize import minimize
from decimal import Decimal
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

omega = 0.07165
time_factor = (2*np.pi/omega) * 0.364056 / 41.34
mu = 0.36

#Gets the intensites for the sweep
def getChebNodes(n):
    k = np.array(range(n))
    x = np.sin(np.pi*k/(2*(n-1)))**2
    return x



min_cycles =  4 #Minimum number of cycles for the sweep (command line argument 1)
max_cycles =  16 #Maximum number of cycles for the sweep (command line argument 2)
order = 8         #How many pulse durations to sweep over (command line argument 3)

cycles_sweep = min_cycles +  (max_cycles - min_cycles) * getChebNodes(order) 

wf_example = '8/0/output_iter/td.0000000/wf-st0001.dx'
with open(wf_example) as f:
	s1 = f.readline()
	s2 = f.readline()
	wfShape = [int(n) for n in s1.split()[-3:]]
	boxLengths = [-float(n) for n in s2.split()[-3:]]
	numPoints = np.prod(wfShape)
	dx = 2*boxLengths[0]/(wfShape[0]-1)
	print(f'Mesh size: {wfShape}')
	print(f'Box size: {boxLengths}')
	print(f'Spacing: {dx}')

def read_wf(name):
	df = pd.read_csv(name,header=None,skiprows=7,nrows=numPoints,delim_whitespace=True)
	wf = df.values[:,0] + 1j*df.values[:,1]
	return np.reshape(wf,wfShape)
	


recalculate = False
if recalculate:

	#Read in initial wavefunctions.
	initial_wfs = []
	for name in glob('8/0/output_iter/td.0000000/*.dx'):
		wfi = read_wf(name)
		initial_wfs.append(wfi)


	ion = []
	lz = []
	dm = []
	for i in range(8):
		print(i)
		#Read in multipoles file.
		a = np.loadtxt(f'8/{i}/td.general/multipoles')
		ion.append(a[0,2] - a[-1,2])
		b = np.loadtxt(f'8/{i}/td.general/angular')
		lz.append(b[-1,-1])

		#Read in initial wavefunctions.
		density_matrix = np.zeros([5,5],dtype=complex)
		for name in glob(f'8/{i}/output_iter/*/*.dx')[-5:]:
			wfj = read_wf(name)
			projections = []
			for wfi in initial_wfs:
				proj = np.sum(np.conjugate(wfi)*wfj)*dx**3
				projections.append(proj)
			projections = np.array(projections)
			density_matrix += np.outer(np.conjugate(projections),projections)
		dm.append(density_matrix)


	ion = np.array(ion)
	lz = np.array(lz)
	dm = np.array(dm)

	np.save('ion.npy',ion)
	np.save('lz.npy',lz)
	np.save('dm.npy',dm)
	dm = 1-dm
else:
	ion = np.load('ion.npy')
	lz = np.load('lz.npy')
	dm = 1-np.load('dm.npy')

trace = np.einsum('abb->a',dm)
for a,b in zip(ion,trace):
	print(f'{a}  :  {2*b.real}  :  {a-2*b.real}')

cycles_up = np.linspace(min_cycles,max_cycles,1000)
ion_up = barycentric_interpolate(cycles_sweep,ion,cycles_up)
lz_up = barycentric_interpolate(cycles_sweep,lz,cycles_up)
dm_up = barycentric_interpolate(cycles_sweep,dm,cycles_up,axis=0)

diagonal = np.einsum('abb->ab',dm_up).real
right = (diagonal[:,2] + diagonal[:,3])/2 + dm_up[:,2,3].imag
left = (diagonal[:,2] + diagonal[:,3])/2 - dm_up[:,2,3].imag
diagonal[:,2] = right
diagonal[:,3] = left

labels = [r'2$\sigma_g$',r'2$\sigma_u$',r'1$\pi_{u+}$',r'1$\pi_{u-}$',r'3$\sigma_g$']
linestyles = reversed(['-','--',':','-.','-'])
for d,l,ls in reversed(list(zip(np.transpose(diagonal),labels,linestyles))):
	plt.plot(time_factor*cycles_up,d,linestyle = ls,label=l)
plt.legend()
#plt.legend(['1','2','3','4','5'])
plt.ylabel('Ionization probability')
plt.xlabel('Pulse duration (FWHM in fs)')
plt.savefig('diagonal.png')
plt.close()
print(dm_up.shape)

#Explanatory model
def model(param,T):
	a,b,c,d = param
	t1 = np.linspace(0,T,100)
	dt = t1[1]-t1[0]
	exp_factor = np.exp(-c*(t1/T-0.5)**2)
	exp_factor /= np.sum(exp_factor)*dt
	x = np.pi*t1/T
	sin_factor = (12*(np.pi-x) + 8*np.sin(2*x) - np.sin(4*x))/32
	rabi_freq = mu*np.sqrt(3e-2/3.51)*sin_factor/np.pi
	rabi_factor = np.sin(d + T*rabi_freq)
	return a+b*np.sum(exp_factor*rabi_factor)*dt

chi_exact = -lz/ion
def objective(param):
	chi_model = np.array([model(param,(2*np.pi/0.07165) * ncyc) for ncyc in cycles_sweep])
	error = np.linalg.norm(chi_model-chi_exact)
	print(error)
	return error

guess_param = [  0.29718332 ,  0.34044058, 108.16813124,  -0.76964426]
#result = minimize(objective,guess_param)
#param = result.x
#print(param)
param = guess_param

chi_model = [model(param,(2*np.pi/0.07165) * ncyc) for ncyc in cycles_up]

fig,axs = plt.subplots(2,1,sharex=True)
axs[0].plot(time_factor*cycles_up,ion_up)
axs[0].plot(time_factor*cycles_up,-lz_up)
axs[0].plot(time_factor*cycles_sweep,ion,linestyle='None',marker='x',color='black')
axs[0].plot(time_factor*cycles_sweep,-lz,linestyle='None',marker='x',color='black')
axs[0].legend(['Ionization','Lz'])
axs[0].set_ylim([0,np.max(ion_up)+0.01])
#axs[0].set_xlabel('Pulse duration (FWHM in fs)')

axs[1].plot(time_factor*cycles_up,-lz_up/ion_up)
#axs[1].plot(time_factor*cycles_up,chi_model,linestyle='--',color='red')
axs[1].plot(time_factor*cycles_sweep,-lz/ion,linestyle='None',marker='x',color='black')
axs[1].set_ylim([0,1])
axs[1].legend(['Helicity selectivity'])
#plt.plot(cycles_sweep,lz)
axs[1].set_xlabel('Pulse duration (FWHM in fs)')
axs[0].set_title(r'N$_2$')
plt.tight_layout()
plt.savefig('sfi.png')