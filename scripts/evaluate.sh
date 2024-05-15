#!/bin/bash

DATASET_NAME=semeval2016

poetry run python src/inference.py --dataset_name $DATASET_NAME