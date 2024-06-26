#!/bin/bash
# run.sh

# Define distance metric and weight
# DISTANCE_METRIC="euclidean" # "euclidean" and "manhattan"
WEIGHTS=("uniform" "distance")  # "uniform" and "distance"

# Define the list of datasets
DATASETS=("Arrhythmia_Data_Set" "gene_expression_cancer_RNA-Seq_Data_Set")
# DATAPATH='dataset/Arrhythmia_Data_Set'  #'dataset/gene_expression_cancer_RNA-Seq_Data_Set'
SAVE_DIR="./trainlog"
SAVE_FIG_DIR="./fig"
k=13
THRESHOLD=0.8
MAX_ITER=200

# arrhythmia dataset
TRAIN_NUM_CLASSES=8
UNKNOWN_NUM_CLASSES=5
N_CLUSTERS=$UNKNOWN_NUM_CLASSES

CLASSIFY_ALGO="KNN"
# Define k start, end, and step
K_START=3
K_END=59
K_STEP=2

# Loop over datasets for KNN
for DATASET in "${DATASETS[@]}"
do
    DATAPATH="dataset/$DATASET"
    echo "Running experiments for $DATASET"

    # Adjust settings based on the dataset
    if [ "$DATASET" == "Arrhythmia_Data_Set" ]; then
        TRAIN_NUM_CLASSES=8
        UNKNOWN_NUM_CLASSES=5
    elif [ "$DATASET" == "gene_expression_cancer_RNA-Seq_Data_Set" ]; then
        TRAIN_NUM_CLASSES=3
        UNKNOWN_NUM_CLASSES=2
    fi
    N_CLUSTERS=$UNKNOWN_NUM_CLASSES


    for WEIGHT in "${WEIGHTS[@]}"
    do
        # Loop over a range of k values generated by seq
        for k in $(seq $K_START $K_STEP $K_END)
        do
            echo "Running experiment with k=$k"
            python main.py --data_path $DATAPATH --k_value $k --weight $WEIGHT --save_dir $SAVE_DIR --save_fig_dir $SAVE_FIG_DIR --classify_algo $CLASSIFY_ALGO --threshold $THRESHOLD --n_clusters $N_CLUSTERS --max_iter $MAX_ITER --train_num_classes $TRAIN_NUM_CLASSES --unknown_num_classes $UNKNOWN_NUM_CLASSES # --k_start $K_START --k_end $K_END
        done
    done
done


CLASSIFY_ALGO="SVM"
KERNELS=("linear" "poly" "rbf" "sigmoid")

for DATASET in "${DATASETS[@]}"
do
    DATAPATH="dataset/$DATASET"
    echo "Running experiments for $DATASET"

    # Adjust settings based on the dataset
    if [ "$DATASET" == "Arrhythmia_Data_Set" ]; then
        TRAIN_NUM_CLASSES=8
        UNKNOWN_NUM_CLASSES=5
    elif [ "$DATASET" == "gene_expression_cancer_RNA-Seq_Data_Set" ]; then
        TRAIN_NUM_CLASSES=3
        UNKNOWN_NUM_CLASSES=2
    fi
    N_CLUSTERS=$UNKNOWN_NUM_CLASSES


    for KERNEL in "${KERNELS[@]}"
    do
        # Loop over a range of k values generated by seq
        echo "Running experiment with SVM kernel=$KERNEL"
        python main.py --data_path $DATAPATH --kernel $KERNEL --save_dir $SAVE_DIR --save_fig_dir $SAVE_FIG_DIR --classify_algo $CLASSIFY_ALGO --threshold $THRESHOLD --n_clusters $N_CLUSTERS --max_iter $MAX_ITER --train_num_classes $TRAIN_NUM_CLASSES --unknown_num_classes $UNKNOWN_NUM_CLASSES # --k_start $K_START --k_end $K_END
        
    done
done


CLASSIFY_ALGO="XGB"

for DATASET in "${DATASETS[@]}"
do
    DATAPATH="dataset/$DATASET"
    echo "Running experiments for $DATASET"

    # Adjust settings based on the dataset
    if [ "$DATASET" == "Arrhythmia_Data_Set" ]; then
        TRAIN_NUM_CLASSES=8
        UNKNOWN_NUM_CLASSES=5
    elif [ "$DATASET" == "gene_expression_cancer_RNA-Seq_Data_Set" ]; then
        TRAIN_NUM_CLASSES=3
        UNKNOWN_NUM_CLASSES=2
    fi
    N_CLUSTERS=$UNKNOWN_NUM_CLASSES

    
    # Loop over a range of k values generated by seq
    echo "Running experiment with XGBoost"
    python main.py --data_path $DATAPATH --save_dir $SAVE_DIR --save_fig_dir $SAVE_FIG_DIR --classify_algo $CLASSIFY_ALGO --threshold $THRESHOLD --n_clusters $N_CLUSTERS --max_iter $MAX_ITER --train_num_classes $TRAIN_NUM_CLASSES --unknown_num_classes $UNKNOWN_NUM_CLASSES # --k_start $K_START --k_end $K_END
    
done

# python main.py --data_path 'dataset/Arrhythmia_Data_Set' --k_value 3 --weight 'uniform' --save_dir "./trainlog" --save_fig_dir "./fig" --classify_algo "KNN" --threshold 0.8 --n_clusters 5 --max_iter 100 --train_num_classes 8 --unknown_num_classes 5
# python main.py --data_path $DATAPATH --k_value $k --weight $WEIGHT --save_dir $SAVE_DIR --threshold $THRESHOLD --n_clusters $N_CLUSTERS --max_iter $MAX_ITER --train_num_classes $TRAIN_NUM_CLASSES --unknown_num_classes $UNKNOWN_NUM_CLASSES