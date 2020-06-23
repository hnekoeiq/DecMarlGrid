#!/bin/bash
#SBATCH --partition=long                     # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=24G                             # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                        # The job will run for 3 hours
#SBATCH -o /network/home/nekoeiha/job_results/marlgrid/slurm-%j.out  # Write the log on tmp1

python3 run.py --train
