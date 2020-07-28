#!/bin/bash
#SBATCH --job-name octopus
#SBATCH --partition=shas --qos=normal
#SBATCH --nodes 1
#SBATCH --ntasks 24
#SBATCH --time 24:00:00
#SBATCH --output Octo.o%j
#SBATCH --export=ALL

module purge
module load intel/17.4
module load impi
module load hdf5
module load perl fftw gsl netcdf mkl cmake
export LD_LIBRARY_PATH=/projects/anmo3668/OCTOPUS_TEST_ANDY1/pfft-1.0.8-alpha/lib:$LD_LIBRARY_PATH


export PATH=$PATH:/projects/anmo3668/OCTOPUS_TEST_ANDY1/octopus-7.1/bin
export MPIEXEC=`which mpirun`

mpirun -n $SLURM_NTASKS octopus >& out.log






