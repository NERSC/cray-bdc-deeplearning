#!/bin/bash
#SBATCH --job-name=ResNet
#SBATCH --time=03:00:00
#SBATCH --nodes=1024
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
##DW jobdw capacity=1500GB access_mode=striped type=scratch
##DW stage_in source=/scratch/snx3000/pjm/ImageNet destination=$DW_JOB_STRIPED/ImageNet type=directory

##SBATCH --account=g107

module load /scratch/snx3000/pjm/modulefiles/cosmoflow
export PYTHONPATH=/scratch/snx3000/pjm/cray-tensorflow/tf_models/models:${PYTHONPATH}

ulimit -s unlimited

MDLDIR=checkpoint
BATCH_SIZE=32
DATA_DIR=/scratch/snx3000/pjm/ImageNet
#DATA_DIR=$DW_JOB_STRIPED/ImageNet

echo "DATA_DIR = $DATA_DIR"
ls $DATA_DIR | wc -l
du $DATA_DIR

rm -rf ${MDLDIR}
mkdir ${MDLDIR}
lfs setstripe -c 16 ${MDLDIR}

srun -u -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 --cpu_bind=none python imagenet_main.py \
  --resnet_size=50 \
  --enable_ml_comm=1 \
  --model_dir=${MDLDIR} \
  --benchmark_log_dir=${MDLDIR} \
  --export_dir=${MDLDIR} \
  --batch_size=${BATCH_SIZE} \
  --data_dir=${DATA_DIR} \
  --num_parallel_calls=8 \
  --intra_op_parallelism_threads=8 \
  --inter_op_parallelism_threads=2 \
  --epochs_between_evals=40 \
  --global_perf_log_freq=20 |& tee logfile

