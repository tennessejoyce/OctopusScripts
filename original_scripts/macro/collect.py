import numpy as np
import sys

#This script will collect the results from a sweep into a
#single convenient output file.
#By default, intensities are chosen at Chebyshev nodes, which
#allow for efficient polynomial interpolation of the results.

#Gets the intensites for the sweep
def getChebNodes(n):
    k = np.array(range(n))
    x = np.cos(np.pi*k/(2*n))
    return x

order = int(sys.argv[1])       #How many intensities to calculate (command line argument 2)

dipoleY=[]
dipoleZ=[]
for i in range(order):
	dataName = str(order)+"/"+str(i) +"/"+"td.general/multipoles"
	multipoles = np.loadtxt(dataName)
	dipoleY.append(multipoles[:,4] + multipoles[:,8])
	dipoleZ.append(multipoles[:,5] + multipoles[:,9])
dipole = np.array([np.array(dipoleY),np.array(dipoleZ)])
print(dipole.shape)
np.save('dipole.npy',dipole)
