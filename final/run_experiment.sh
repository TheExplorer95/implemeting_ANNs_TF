#!/usr/bin/env bash

declare -a models=("2dconv_gru/")
#, "1dconv_transformer/" "2dconv_gru/")

# run entire script for each model in models
for model in ${models[@]}
do
	# train cpc_model
		#python scripts/train_cpc.py -m $model
   	# get embeddings
   	python scripts/generate_embeddings.py -m $model
   	# train classifier and plot results
   	#python scripts/train_classifiers.py -m $model
done
