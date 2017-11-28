#!/bin/bash
#SBATCH -J GameOfLife          # Job name
#SBATCH -o GOL.%j.out   # stdout; %j expands to jobid
#SBATCH -e GOL.%j.err   # stderr; skip to combine stdout and stderr
#SBATCH -p gpu         # specify queue
#SBATCH --gres=gpu:4  
#SBATCH --ntasks-per-node=24
#SBATCH -t 00:15:00       # max time 5mins

#SBATCH --mail-user=channon@hawk.iit.edu
#SBATCH --mail-type=ALL

#SBATCH -A TG-CIE170044        # project/allocation number;

module load cuda         # Load any necessary modules (these are examples)
#module list

./GoLCuda    # TACC systems use "ibrun", not "mpirun" 