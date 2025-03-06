#!/bin/bash -l

counter=0
counter2=0
out_str=""
code_home=/home/sb95/PolymerCrowderInteractions
data_home=/work/cms16/sb95/Polymer_collapse

mkdir -p -v $data_home

cp -r $code_home/modules $data_home
cp $code_home/run_rg.py $code_home/submit_simulation_6H.slurm $data_home
cd $data_home
mkdir input
for ka in 0.0 5.0 10.0 20.0 50.0 100.0 200.0; do
for chi in 0.0 -0.05 -0.1 -0.15 -0.2 -0.25 -0.3 -0.35 -0.4; do
for ecut in 0.0 4.0 8.0; do
savefolder=output_ka${ka}_chi${chi}_ecut${ecut}
mkdir -p -v $savefolder

if [[ -z "$out_str" ]]; then
    out_str=$"python run_rg.py ${ka} ${chi} ${ecut} ${savefolder} > ${savefolder}/output.log"
else
    out_str+=$'\n'"python run_rg.py ${ka} ${chi} ${ecut} ${savefolder} > ${savefolder}/output.log"
fi
# out_str+=$'\n'"python run_rg.py ${ka} ${chi} ${ecut} ${savefolder} > ${savefolder}/output.log"

((counter++))
((counter2++))

if (( counter == 16 )); then
# echo "$out_str"
sbatch_file="#!/bin/bash -l

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
source \$HOME/anaconda3/bin/activate
conda activate openmm

# Controlling Launcher and showing some job info
export LAUNCHER_WORKDIR=\`pwd\`
export LAUNCHER_JOB_FILE=\$PWD/launcher_jobs_sim$counter2
export LAUNCHER_BIND=1

# Each iteration is an inversion
# rm \${LAUNCHER_WORKDIR}/launcher_jobs_sim &> /dev/null
date
echo \"${out_str}\" >> \${LAUNCHER_WORKDIR}/launcher_jobs_sim$counter2

\$LAUNCHER_DIR/paramrun

wait
date"
echo "$sbatch_file">>submit_sim_$counter2.slurm
sbatch submit_sim_$counter2.slurm
echo "--------------------------------------"
counter=0
out_str=""
fi
done
done
done

# Print remaining commands if any
if [[ -n "$out_str" ]]; then
echo "$out_str"
sbatch_file="#!/bin/bash -l

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
source \$HOME/anaconda3/bin/activate
conda activate openmm

# Controlling Launcher and showing some job info
export LAUNCHER_WORKDIR=\`pwd\`
export LAUNCHER_JOB_FILE=\$PWD/launcher_jobs_sim
export LAUNCHER_BIND=1

# Each iteration is an inversion
rm \${LAUNCHER_WORKDIR}/launcher_jobs_sim &> /dev/null
date
echo \"${out_str}\" >> \${LAUNCHER_WORKDIR}/launcher_jobs_sim
\$LAUNCHER_DIR/paramrun
wait
date"
echo "$sbatch_file">>submit_sim_$counter2.slurm
sbatch submit_sim_$counter2.slurm
fi