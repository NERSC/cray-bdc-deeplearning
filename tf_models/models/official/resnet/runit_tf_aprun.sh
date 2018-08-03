#!/bin/bash
#PBS -l nodes=128:INTELBW36
#PBS -l walltime=2:00:00
#PBS -N resnet-mlc
#PBS -j oe
#PBS -o resnet50-dist.out

#export OMP_NUM_THREADS=36

ulimit -s unlimited
ulimit -c unlimited
#cudatoolkit/8.0.61_2.4.9-6.0.6.0_4.1__g899857c
#source /cray/css/users/jbalma/bin/setup_env_cuda9.sh
#source /cray/css/users/jbalma/bin/env_python.sh
source ./setup_env.sh
#module load cray-python
#module use /cray/css/perfeng/ml-plugin/modulefiles
#module load craype-ml-plugin-py2/1.0.1
#module load cray-python
#module load craype-ml-plugin-py2
#module list
#module load cray-python
#module load /cray/css/perfeng/pjm/TFbuilds/tmp_inst/modulefiles/craype-ml-plugin-py2/1.0.1
#module load /cray/css/users/jbalma/Innovation-Proposals/ML-Comm-Plugin/ml-mpi/tmp_inst/modulefiles/craype-ml-plugin-py2/1.1.0
module load craype-ml-plugin-py2/1.1.1
module list

#export CRAYPE_ML_PLUGIN_BASEDIR=/cray/css/users/jbalma/Innovation-Proposals/ML-Comm-Plugin/ml-mpi

export PYTHONPATH="$CRAYPE_ML_PLUGIN_BASEDIR/lib:$CRAYPE_ML_PLUGIN_BASEDIR/lib/ml_comm:/home/users/jbalma/.local/lib/python2.7/site-packages:$PYTHONPATH"

#export PYTHONPATH="$PYTHONINCLUDE/cray/css/users/jbalma/Innovation-Proposals/ML-Comm-Plugin/ml_comm_plugin_examples/tf_models:$PYTHONPATH"

export SCRATCH=/lus/scratch/jbalma
#source ${SCRATCH}/TensorFlow-v1/ML_Dock_tests/env.sh
#export CUDA_CACHE_PATH=${SCRATCH}/.nv/ComputeCache

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_CPUMASK_DISPLAY=1
#export MPICH_COLL_SYNC=1 #force everyone into barrier before each collective
#export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_MAX_THREAD_SAFETY=multiple
#export CRAY_CUDA_MPS=1
#export CUDA_VISIBLE_DEVICES=0
#export CRAY_CUDA_PROXY=0

echo "Running..."

#module load /cray/css/perfeng/pjm/TFbuilds/tmp_inst/modulefiles/craype-ml-plugin-py2/1.0.1


export LD_LIBRARY_PATH=$CRAYPE_ML_PLUGIN_BASEDIR/lib:$CRAYPE_ML_PLUGIN_BASEDIR/lib/ml_comm:$CUDATOOLKIT_HOME/lib64:$CUDNN_PATH/lib64:$LD_LIBRARY_PATH

WKDIR=/lus/scratch/jbalma/temp/junk_mlmpi_run
rm -rf $WKDIR
mkdir -p $WKDIR
#SCRATCH=$WKDIR
#cp ./*.py $WKDIR
#cp -r /cray/css/users/jbalma/Innovation-Proposals/ML-Comm-Plugin/ml-mpi/examples/tf_models/models $WKDIR/
cp -r /cray/css/users/jbalma/Innovation-Proposals/ML-Comm-Plugin/cray-bdc-deeplearning/tf_models/models $WKDIR/
cd $WKDIR/models
#rm -r checkpoint
mkdir checkpoint
#export SLURM_WORKING_DIR=$WKDIR/models

export PBS_O_WORKDIR=${WKDIR}
echo "Currently in "
echo $PWD

export PYTHONPATH="$PYTHONPATH:${WKDIR}/models"

NP=256
BATCH_SIZE=64
NUM_IMG_TRAIN=960831
NUM_IMG_TEST=320336

NUM_EPOCHS=1

let "NUM_BATCHES=NUM_EPOCHS * NUM_IMG_TRAIN/BATCH_SIZE"

#DATA_DIR=/lus/scratch/jbalma/DataSets/Uber/imagenet_tfrecords_224
#DATA_DIR=/lus/scratch/jbalma/ImageNet/ILSVRC2016
DATA_DIR=/lus/scratch/pjm/ImageNet_data
HYPNO_CMD="hypno --audit --gpu --plot=gpu_power,node_power"

export OMP_NUM_THREADS=16
#INIT_LR=0.08
#$(echo "scale=2; $LR" | bc)
#srun -n 10 -N 10 -C P100 --gres=gpu python dist_stacked.py  > dist_test.out
#srun -N 1 -n 1 --gres=gpu -C P100 --exclusive python stacked.py ${DATA_PATH} ${FULL_DATAPATH}
#srun -N 1 -n 1 --gres=gpu -C K40 --exclusive python stacked.py ${DATA_PATH} ${FULL_DATAPATH}
#srun -N 1 -n 1 --gres=gpu -C P100 --exclusive python dual_stacked.py ${DATA_PATH} ${FULL_DATAPATH}
#srun --time=1:00:00 -l --ntasks=${NP} --ntasks-per-node=1 -C P100 --gres=gpu --exclusive --cpu_bind=none -u nvidia-smi
aprun -r 2 -n ${NP} -N 2 -j 1 -S 1 -d $OMP_NUM_THREADS -cc numa_node python ${WKDIR}/models/official/resnet/imagenet_main.py \
  --resnet_size=50 \
  --enable_ml_comm=1 \
  --train_epochs 100 \
  --init_lr 0.1 \
  --base_lr 40.0 \
  --warmup_epochs 12 \
  --weight_decay 0.0005 \
  --model_dir=${WKDIR}/models/checkpoint \
  --benchmark_log_dir=${WKDIR}/models/checkpoint \
  --export_dir=${WKDIR}/models/checkpoint \
  --batch_size=${BATCH_SIZE} \
  --intra_op_parallelism_threads=2 \
  --inter_op_parallelism_threads=2 \
  --data_dir=${DATA_DIR} \
  --global_perf_log_freq=20 |& tee logfile


#	--resnet_size=50 \
#	--enable_ml_comm=1 \
#	--epochs_between_evals=1 \
#	--ml_comm_validate_init=1 \
#	--batch_size=$BATCH_SIZE \
#	--train_epochs=90 \
#	--max_train_steps=1024 \
#	--epochs_between_evals=1 \
#	--intra_op_parallelism_threads=1 \
#	--inter_op_parallelism_threads=1 \
#	--num_parallel_calls=1 \
#	--benchmark_log_dir="${WKDIR}/models/checkpoint/" \
#	--model_dir="${WKDIR}/models/checkpoint/" \
#	--export_dir="${WKDIR}/models/checkpoint/" \
#	--data_dir="${DATA_DIR}"

#	--use_synthetic_data
	

#	--intra_op_parallelism_threads=1 \
#	--inter_op_parallelism_threads=1 \
#	--resnet_size=50 \
#	--batch_size=$BATCH_SIZE \
#	--train_epochs=90 \
#	--num_parallel_calls=1 \
#	--use_synthetic_data
	#--benchmark_log_dir="${WKDIR}/checkpoint" \
	#--data_dir=${DATA_DIR} \
#	--eval_dir="${WKDIR}/eval" \

#0: usage: imagenet_main.py [-h] [--data_dir <DD>] [--model_dir <MD>]
#0:                         [--train_epochs <TE>] [--epochs_between_evals <EBE>]
#0:                         [--stop_threshold <ST>] [--batch_size <BS>]
#0:                         [--multi_gpu] [--hooks <HK> [<HK> ...]]
#0:                         [--num_parallel_calls <NPC>]
#0:                         [--inter_op_parallelism_threads <INTER>]
#0:                         [--intra_op_parallelism_threads <INTRA>]
#0:                         [--use_synthetic_data] [--max_train_steps <MTS>]
#0:                         [--dtype <DT>] [--loss_scale LOSS_SCALE]
#0:                         [--data_format <CF>] [--export_dir <ED>]
#0:                         [--benchmark_log_dir <BLD>] [--gcp_project <GP>]
#0:                         [--bigquery_data_set <BDS>]
#0:                         [--bigquery_run_table <BRT>]
#0:                         [--bigquery_metric_table <BMT>] [--version {1,2}]
#0:                         [--resnet_size {18,34,50,101,152,200}]

#    dataset: A Dataset representing raw records
#    is_training: A boolean denoting whether the input is for training.
#    batch_size: The number of samples per batch.
#    shuffle_buffer: The buffer size to use when shuffling records. A larger
#      value results in better randomness, but smaller values reduce startup
#      time and use less memory.
#    parse_record_fn: A function that takes a raw record and returns the
#      corresponding (image, label) pair.
#    num_epochs: The number of epochs to repeat the dataset.
#    num_parallel_calls: The number of records that are processed in parallel.
#      This can be optimized per data set but for generally homogeneous data
#      sets, should be approximately the number of available CPU cores.
#    examples_per_epoch: The number of examples in the current set that
#      are processed each epoch. Note that this is only used for multi-GPU mode,
#      and only to handle what will eventually be handled inside of Estimator.
#    multi_gpu: Whether this is run multi-GPU. Note that this is only required
#      currently to handle the batch leftovers (see below), and can be removed
#      when that is handled directly by Estimator.

echo "Done..."



