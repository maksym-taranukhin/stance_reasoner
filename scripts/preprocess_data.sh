#!/bin/bash

DATASET_NAME=semeval2016

poetry run python src/data_preprocessing.py --dataset_name $DATASET_NAME