#!/bin/bash

#SBATCH --job-name barriers
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 32gb
#SBATCH --time 72:00:00

module add anaconda3
venv/bin/python neb.py