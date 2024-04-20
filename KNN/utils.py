import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd 

def load_data(traindf, testdf):
    X_traindf = traindf.drop(columns=['Outcome']).values
    X_testdf = testdf.drop(columns=['Outcome']).values
    y_traindf = traindf['Outcome'].values
    y_testdf = testdf['Outcome'].values

    return X_traindf, y_traindf, X_testdf, y_testdf

def plot_confusion_matrix(y_true, y_pred):
    # confusion matrix
    c_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 7))  
    sns.heatmap(c_matrix, annot=True, cmap='Blues', fmt='d', linewidths=.5)
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Show the plot
    plt.show()