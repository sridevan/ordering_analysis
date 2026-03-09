# ordering_analysis

Python utilities for ordering identifiers based on pairwise dissimilarity values and visualizing the result as a heatmap.

## Input

The input file should contain three whitespace-separated columns:

Query Target Dissimilarity

Example:

7LWW 7LWW 0.0  
7LWW 9BBK 5.418  
7LWW 8UKD 4.781  

## Workflow

1. Read pairwise dissimilarity values from a file
2. Construct a symmetric dissimilarity matrix
3. Order identifiers using `treePenalizedPathLength`
4. Visualize the ordered matrix as a heatmap

## Requirements

- Python 3
- numpy
- matplotlib
- scipy

## Author

Sri Devan Appasamy
Bioinformatician,EMBL-EBI (PDBe)
