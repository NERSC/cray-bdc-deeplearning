Examples in this directory are ported from https://github.com/tensorflow/models.git

To run the baseline resnet-50 using gRPC, (from https://github.com/tensorflow/models/tree/master/official/resnet)
do the following:

	cd tf_models/models/official/
	mv /resnet_orig resnet/
	The follow directions below (steps 1,2,3) for everything but remove the --enable_ml_comm=1 and --ml_comm_validate_init=1 options

To run the cray version, do the following:

	cd tf_models/models/official/
	mv resnet_cray/ resnet


1) Setup from the root repo directory:
	pip install -r tf_models/models/official/requirements.txt --user
	export PYTHONPATH="$PYTHONPATH:/full/path/to/tf_models/models"

2) Modify the runscript 
        cd models/official/resnet_cray
	edit runit_tf_sanity.sh to point to your local paths and dataset
	

3) Run the benchmark

	To use real data:
	
		python imagenet_main.py \
                        --resnet_size=50 \
                        --enable_ml_comm=1 \
                        --ml_comm_validate_init=1 \
                        --batch_size=32
                        --train_epochs=90 \
			--data_dir=/lus/scratch/username/Datasets/imagenet

	To use synthetic data:

		python imagenet_main.py \
		        --resnet_size=50 \
		        --enable_ml_comm=1 \
		        --ml_comm_validate_init=1 \
		        --batch_size=32 \
		        --train_epochs=90 \
		        --use_synthetic_data

	
	

	




