# Sequence Classification using CNN 


### Task Description
This small project involves predicting species class labels using DNA sequences as features using 1D CNN. The DNA string comprises characters {A, C, G, T, -}, with "-" representing undetected positions. 

### Data Description
- `train_features.csv` : contains DNA sequences (not necessarily of the same length).
- `train_labels.csv` : contains the corresponding labels for the DNA sequences (range is between 1 - 1213).

There are no missing values in either of them.

### Feature Engineering
- one hot encoded all the DNA sequences, i.e 'A': [1, 0, 0, 0, 0],   'C': [0, 1, 0, 0, 0],  'G': [0, 0, 1, 0, 0], 'T': [0, 0, 0, 1, 0],    '-': [0, 0, 0, 0, 1]
- The features had a few missing values denoted by '-' in the sequence, I encoded it too as shown above.
- all the DNA sequences were not of the same length, so padded with [0,0,0,0,0] to bring them to a common length of 1058.
- added the extra dimension(channels) so as to bring the data to a format to be fed into a CNN.

### Model Architecture
1-D Convolutional Neural Network is used.

- **input channels** : 5, corresponding to one-hot encoded DNA bases.
- **1st conv layer** : 32 output channels, 3 kernel size, 1 stride, 1 padding.
- **2nd conv layer** : 64 output channels, same kernel size, stride, and padding.
- **Pooling** : Max pooling with a window of 2
- **Fully connected layers** : Two layers transitioning from 64*264 inputs to 128, then to `num_classes (1213)`.
- **Activation** : ReLU
- **Output**: Final layer outputs to 1213 classes.
- **Loss Function**: Cross-Entropy Loss.
- **optimizer** : Adam with `lr = 0.0001`

### Results
Hyperparameter tuning has been performed. The best hyperparameters turn out to be `no_of_epochs = 50` and `lr = 0.0001`. The same model is used for predictions on the test data. 

