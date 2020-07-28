#!/bin/bash
#SBATCH -J octopus
#SBATCH -p jila
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 96:00:00
module load gcc openmpi gsl fftw libxc blas lapack octopus
export OCT_PARSE_ENV=1
#export OCT_CalculationMode='gs'
#mpiexec -n 16 octopus >& outgs.log
export OCT_CalculationMode='td'
mpiexec -n 16 octopus >& outtd.log
