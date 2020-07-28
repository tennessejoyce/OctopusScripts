import os
import shutil
import sys
import numpy as np

#This script will generate a folder structure for a sweep over
#cycles in octopus. It will also submit each job to the queue.
#By default, cyclesities are chosen at Chebyshev nodes, which
#allow for efficient polynomial interpolation of the results.

#Gets the intensites for the sweep
def getChebNodes(n):
    k = np.array(range(n))
    x = np.sin(np.pi*k/(2*(n-1)))**2
    return x

min_cycles =  float(sys.argv[1]) #Minimum number of cycles for the sweep (command line argument 1)
max_cycles =  float(sys.argv[2]) #Maximum number of cycles for the sweep (command line argument 2)
order = int(sys.argv[3])         #How many pulse durations to sweep over (command line argument 3)

cycles_sweep = min_cycles +  (max_cycles - min_cycles) * getChebNodes(order) 

if not os.path.exists(str(order)):
	os.mkdir(str(order))

for i,cycles in enumerate(cycles_sweep):
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
    os.system(f"sed -i '1 i\\ncyc={cycles}' inp")
    os.system("sbatch octo32.sh") #Submits the octopus job to the queue
    os.chdir('../..')
