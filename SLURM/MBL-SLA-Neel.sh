#!/bin/bash
# current 2018- SC-RTP SLURM system
#SBATCH --nodes=2
#SBATCH --tasks-per-node=16
#SBATCH --mem-per-cpu=3882
#SBATCH --time=10:00:00
#SBATCH --job-name=MYJOBNAME
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.benini@warwick.ac.uk

# ----------------------------------------------------------------
# Module load
module purge
. ~/modules.sh

# Set jobname
jobname=MYJOBNAME
echo $jobname

# Executable directory and file
EXECSRC="/home/physics/phrgmr/Projects/FidMBL"
cd $EXECSRC

# Temporary directory for each job
mkdir ~/RUNS/$jobname
cp Run_Neel.py params.py h_functions.py ~/RUNS/$jobname/

cd ~/RUNS/$jobname
execfile="Run_Neel.py"
inpfile="params.py"

# Input parameters
echo "L               = SIZE ">$inpfile #
echo "D               = DISORDER ">>$inpfile #
echo "Jz              = 1.0 ">>$inpfile #
echo "BC              = 1 ">>$inpfile #
echo "Format_flag     = 0 ">>$inpfile #
echo "Epsilon         = EPSILON">>$inpfile #
echo "In_flag	      = 1">>$inpfile #
cat $inpfile

# Parallel options
MY_PARALLEL_OPTS="-N 1 --delay .8 -j $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.log --resume --compress"
MY_SRUN_OPTS="-N 1 -n 1 --exclusive"

time parallel $MY_PARALLEL_OPTS srun $MY_SRUN_OPTS "python $execfile"  ::: {0..1279}
