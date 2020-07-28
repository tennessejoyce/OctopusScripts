import numpy as np
from scipy.interpolate import barycentric_interpolate,interp1d
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#Build matrix for trig interpolation
def vandermonde(theta):
	#Multiply this by a vector of coefficients to evaluate a trig series.
	out = [np.cos(theta*(2*j+1))for j in range(4)]
	return np.transpose(np.array(out))

theta_low = (np.pi/8)*np.array([0,1,2,3])
print(vandermonde(theta_low))
theta_high = np.linspace(-np.pi/2,np.pi/2,100)
trig_interpolation = vandermonde(theta_high).dot(np.linalg.inv(vandermonde(theta_low)))
print('trig_interpolation',trig_interpolation.shape)

#Read in current over time and intensity
lzAll = np.loadtxt('lzAll.txt')
lzAll = np.transpose(np.reshape(lzAll,[2,4]))
#ihpAll = np.loadtxt('ihpAll.txt')
#First axis is intensity, second axis is time.
print('lzAll',lzAll.shape)
#Times
times_low = np.linspace(-400,400,17)
times_high = np.linspace(0,400,100)
#Upsample over time (cubic interpolation)
lzAll_high = trig_interpolation.dot(lzAll)
#Contour plot over both
# vmax = np.max(np.abs(lzAll_high))
# cmap = plt.contourf(theta_high*180/np.pi,times_high/41.31,np.transpose(lzAll_high),30,vmax=vmax,vmin=-vmax,cmap='seismic')
# plt.colorbar()
# plt.ylabel('time [fs]')
# plt.xlabel('alignment angle [deg]')
# plt.axhline(y=200/41.34,linestyle='--',color='black')
# plt.xticks(np.linspace(-90,90,13))
# plt.title(r'$<L_z>$ [a.u.]')
# plt.tight_layout()
# plt.savefig('contour.png')
# plt.close()

#Plot only at the final time (twice pulse duration).
plt.figure(figsize=1*np.array([6,4]))
plt.plot(theta_high*180/np.pi,lzAll_high[:,0],color='blue')
plt.plot(theta_high*180/np.pi,lzAll_high[:,1],color='red',linestyle='--')
plt.legend([r'$1.47\times 10^{12}$',r'$10^{13}$'],title='Intensity (W/cm$^2$)',loc='lower left')
plt.plot(theta_low*180/np.pi,lzAll[:,0],marker='x',linestyle='None',color='black')
plt.plot(theta_low*180/np.pi,lzAll[:,1],marker='x',linestyle='None',color='black')
plt.xlabel('Alignment angle (deg)')
plt.xticks(np.linspace(-90,90,9))
plt.ylabel(r'$L_z$ (a.u.)')
plt.tight_layout()
plt.savefig('final_orientation.eps')

