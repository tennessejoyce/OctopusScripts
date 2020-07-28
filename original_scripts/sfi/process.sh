#!/bin/bash
#SBATCH -J octopus
#SBATCH -p jila
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH -t 24:00:00
python process.py
