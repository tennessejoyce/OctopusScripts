import os
import shutil
import sys
import numpy as np

#This script will generate a folder structure for a sweep over
#intensity in octopus. It will also submit each job to the queue.
#By default, intensities are chosen at Chebyshev nodes, which
#allow for efficient polynomial interpolation of the results.

#Gets the intensites for the sweep
def getChebNodes(n):
    k = np.array(range(n))
    x = np.cos(np.pi*k/(2*n))
    return x

peakEamp =  float(sys.argv[1]) #Peak electric field (command line argument 1)
order = int(sys.argv[2])       #How many intensities to calculate (command line argument 2)
intensitiesSweep = peakEamp * getChebNodes(order) #List of intensities to calculate. This can be changed.

if not os.path.exists(str(order)):
	os.mkdir(str(order))

for i,intens in enumerate(intensitiesSweep):
    folderName = str(order)+"/"+str(i)
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    #In the top directory, you should prepare an input file and the
    #restart info for the ground state calculation.
    #These are copied into each subfolder.
    shutil.copy('inp',folderName)
    shutil.copy('octo32.sh',folderName)
    shutil.copytree('restart',folderName+'/restart')

    os.chdir(folderName)
    os.system(f"sed -i '1 i\\amplitude={intens}' inp")
    os.system("sbatch octo32.sh") #Submits the octopus job to the queue
    os.chdir('../..')
