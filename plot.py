#!/usr/bin/env python3

import matplotlib.pyplot as plt
from collections import Counter

def plot_bio_distribution(filename: str):
    """
    Reads text data from the provided file and plots the distribution
    of BIO tags. The BIO tag is expected to be the fourth value on each line.
    """
    tags = []
    ordered_tags = ["O", "I-PER", "I-LOC", "I-ORG", "I-MIS", "B-LOC", "B-ORG", "B-MIS"]
    # Initialize tag counts with 0 for each tag in the ordered list
    tag_counts = {tag: 0 for tag in ordered_tags}
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines (denote sentence breaks)
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                tag = parts[3]
                # Map "I-MISC" to "I-MIS" if encountered
                if tag == "I-MISC":
                    tag = "I-MIS"
                # Only count the tag if it's in our ordered list
                if tag in tag_counts:
                    tag_counts[tag] += 1

    # Prepare data for plotting using the ordered tag list
    counts = [tag_counts[tag] for tag in ordered_tags]
    
    # Plot the distribution
    plt.bar(ordered_tags, counts)
    plt.xlabel('BIO Tags')
    plt.ylabel('Frequency')
    plt.title('BIO Tagging Distribution')
    plt.savefig('figures/dist_test.jpg')

if __name__ == '__main__':
    # Replace 'data.txt' with the path to your file containing BIO-tagged data
    plot_bio_distribution('ner.test')
