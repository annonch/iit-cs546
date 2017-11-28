#!/bin/bash
#SBATCH -J GameOfLife          # Job name
#SBATCH -o GOLS.%j.out   # stdout; %j expands to jobid
#SBATCH -e GOLS.%j.err   # stderr; skip to combine stdout and stderr
#SBATCH -p gpu         # specify queue
#SBATCH --gres=gpu:4  
#SBATCH --ntasks-per-node=24
#SBATCH -t 00:05:00       # max time 5mins

#SBATCH --mail-user=channon@hawk.iit.edu
#SBATCH --mail-type=ALL

#SBATCH -A TG-CIE170044        # project/allocation number;

#module load cuda         # Load any necessary modules (these are examples)
#module list

./GoLS    # TACC systems use "ibrun", not "mpirun" 