# Arrhythmia and RNA-seq Data Analysis

## Project Overview

This project focuses on the analysis of arrhythmia and gene expression cancer RNA-seq datasets using various machine learning techniques. The aim is to develop predictive models that can effectively classify different types of biological samples based on their gene expression profiles and electrocardiogram (ECG) data. Given the presence of unknown classes in the test sets that are not present in the training sets, the project employs a strategy of initial classification followed by clustering. This approach helps in handling the prediction of unknown categories effectively.

We use several classification algorithms, including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and XGBoost, to explore their performance in distinguishing between known classes. After classification, a clustering step using K-means is applied to group low-confidence predictions, which may correspond to unknown classes, thereby enhancing the generalization of our model against unseen data during training.

## Getting Started

The dataset can be downloaded [here](https://drive.google.com/drive/folders/1nIkIWn8YyIao1qwO1LiKIQB6AwuBQIR_?usp=sharing). 
Please ensure to place the datasets in the correct directory structure as follows for the scripts to function correctly:

```bash
Data_Mining/final
|----dataset
  |----Arrhythmia_Data_Set
    |----train_data.csv
    |----train_label.csv
    |----test_data.csv
    |----test_label.csv
  |----gene_expression_cancer_RNA-Seq_Data_Set
    |----train_data.csv
    |----train_label.csv
    |----test_data.csv
    |----test_label.csv
|----run.sh
```

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.x
- pip 

### Installation

Clone this repository to your local machine to get started:

```bash
git clone https://github.com/Terry01001/Data_Mining.git
cd Data_Mining
```


### Run experiment

- Generate the results using
```bash
bash ./final/run.sh
```


