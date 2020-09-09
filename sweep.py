import os
import shutil
import sys
import numpy as np

#This script will generate a folder structure for a sweep over
#intensity in octopus. It will also submit each job to the queue.
#By default, intensities are chosen at Chebyshev nodes, which
#allow for efficient polynomial interpolation of the results.


def run_sweep(parameter_name,parameter_values,main_directory_name='',submit_jobs=True,copy_ground_state=True,bash_script_name='octo32.sh'):
    '''Creates the folder structure to run multiple Octopus calculations
    sweeping over some parameter. In the top directory, you
    should prepare an input file, a bash script that submits a single job,
    and the restart folder from the ground state calculation.
    calculation. These are copied into each subfolder, and the input file is
    modified by adding a line that sets 'parameter_name' to one of the values
    in 'parameter_values'. Then the jobs are automatically submitted to the queue.
    '''
    if main_directory_name=='':
        #The default name for the folder containing the sweep.
        main_directory_name = f'{parameter__name}_{len(parameter_values)}'
    #Create the 
    if not os.path.exists(main_directory_name):
        os.mkdir(main_directory_name)
    #Create the subfolders, each of which will do an Octopus calculation.
    for i,v in enumerate(intensities):
        #Create each subfolder, each of
        folder_name = f'{main_directory_name}/{i}'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        #In the top directory, you should prepar
        shutil.copy('inp',folder_name)
        if submit_jobs:
            shutil.copy(bash_script_name,folder_name)
        if copy_ground_state:
            shutil.copytree('restart',folder_name+'/restart')

        os.chdir(folder_name)
        #Edits the 
        os.system(f"sed -i '1 i\\a{parameter_name}={v}' inp")
        if submit_jobs
            #Submits the Octopus job to the queue
            os.system(f'sbatch {bash_script_name}')
        os.chdir('../..')
    return main_directory_name

def collect_sweep(main_directory_name,parameter_values):
    '''Consolidates information from the parameter sweeps td.general files,
    like ionization and time-dependent dipole moment.'''
    subfolders = sorted(glob('main_directory_name/*'))
    return np.stack([np.loadtxt(s+'/td.general/multipoles') for s in subfolders])

#Functions for interpolating over parameter values.
#That includes Chebyshev interpolation, but also things designed
#for molecular orientation, like evenly spaced angles appropriately
#designed for symmetry.


#Gets the intensites for the sweep
def cheb_nodes(n):
    k = np.array(range(n))
    x = np.cos(np.pi*k/(2*n))
    return x

def barycentric_interpolation_matrix(low_sample,high_sample):
    '''Matrix for barycentric interpolation from the parameter values onto others.'''

if __name__=='__main__':
    peakEamp =  float(sys.argv[1]) #Peak electric field (command line argument 1)
    order = int(sys.argv[2])       #How many intensities to calculate (command line argument 2)
    intensitiesSweep = peakEamp * getChebNodes(order) #List of intensities to calculate. This can be changed.
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

order = int(sys.argv[1])       #How many intensities to calculate (command line argument 1)

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
