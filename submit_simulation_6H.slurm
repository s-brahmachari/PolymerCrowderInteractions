#!/bin/bash -l

#SBATCH --job-name=polycol
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1            # this can be more, up to 22 on aries
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --export=ALL

module purge
module load foss/2020b Launcher_GPU OpenMPI 
source $HOME/anaconda3/bin/activate
conda activate openmm

max_replicas=$((2*SLURM_NNODES*8))
# Controling Launcher and showing some job info
export LAUNCHER_WORKDIR=`pwd`
export LAUNCHER_JOB_FILE=$PWD/launcher_jobs_sim
export LAUNCHER_BIND=1
launcher_str=$1
# Each iteration is an inversion
rm ${LAUNCHER_WORKDIR}/launcher_jobs_sim &> /dev/null
date
echo "${launcher_str}" >> ${LAUNCHER_WORKDIR}/launcher_jobs_sim
$LAUNCHER_DIR/paramrun
wait
date






