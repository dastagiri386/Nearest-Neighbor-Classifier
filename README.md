# Nearest-Neighbor-Classifier
kNN implementation ( with leave-one-out cross-validation) in python

The kNN learner uses Euclidean distance to compute distances between instances. This is a basic kNN (does not include distance weighting, edited nearest neighbor, k-d tree lookup). Both classification and regression are considered (identified by the last attribute in the ARFF file being 'class' or 'response')

Another version kNN-select chooses optimal k for test set based on leave-one-out cross-validation performed on training set. k is chosen from 3 values passed as parameters to the program.

## Running the program
```sh
kNN train-set-file test-set-file k
kNN-select train-set-file test-set-file k1 k2 k3
```
