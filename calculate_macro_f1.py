#!/usr/bin/env python3
"""
This script reads an output file where each non-empty line contains space-separated values.
The fourth value in each line is the true tag and the fifth value is the predicted tag.
It calculates the macro F1 score for each tag and prints a classification report.
Additionally, it computes and plots a confusion matrix.

Usage:
    python calculate_macro_f1.py <data_file>
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def read_data(file_path):
    """
    Reads the data file and extracts true and predicted tags.
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        tuple: Two lists containing true tags and predicted tags respectively.
    """
    true_tags = []
    pred_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            parts = line.split()
            # Check that there are at least 5 columns
            if len(parts) >= 5:
                true_tags.append(parts[3])
                pred_tags.append(parts[4])
    
    return true_tags, pred_tags

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        classes (list): List of class labels.
        title (str): Title of the plot.
        cmap: Colormap for the plot.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Print numbers inside the cells
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("figures/" + sys.argv[2])

def main():
    if len(sys.argv) != 3:
        print("Usage: python calculate_macro_f1.py <data_file> <output_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    true_tags, pred_tags = read_data(file_path)
    
    # Generate and print the classification report (includes macro F1 scores for each tag)
    report = classification_report(true_tags, pred_tags, digits=4)
    print("Classification Report:")
    print(report)
    
    # Compute confusion matrix
    labels = sorted(set(true_tags + pred_tags))
    cm = confusion_matrix(true_tags, pred_tags, labels=labels)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, classes=labels)
    plt.show()

if __name__ == '__main__':
    main()
