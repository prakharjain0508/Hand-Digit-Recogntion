# Hand Digit Recogntion

Implemented Hand digit recognition using fully-connected feed-forward neural networks and convolutional networks (CNNs). Experimented with two different error functions; Sum-of-squares error function and Cross-entropy error function. Also, tried out two different hidden units; ReLU and tanh. 

## How to run the code?

1. To run Fully-connected Feed-forward Neural Network, change "mode = 0"
2. To run Convolutional Neural Network, change "mode = 1"
3. To change activation function, change "activation='relu'" to "activation='tanh'" and vice-versa
4. To change error function, change "loss='categorical_crossentropy'" to "loss='mean_squared_error'" and vice-versa

## Results

### Fully-connected Feed-forward Neural Network

#### Sum-of-squares Error Function VS. Cross-entropy Error Function
| Configuration | Results |
| -- | -- |
| Error Function = Sum-of-squares | Execution time: 179.1158 Secs |
| Activation function = "ReLU" | Accuracy (train data): 0.9977 |
| Num of hidden layers = 3 | Loss (train data): 0.0012 |
| Num of hidden layer units = 800 | Accuracy (test data): 0.9666 |
| Learning rates = 0.05 | Loss (test data): 0.0057 |
| Momentum rates = 0.75 |
| Patience = 5 |
| Epoch = 89 | 

| Configuration | Results |
| -- | -- |
| Error Function = Cross-entropy | Execution time: 56.1165 Secs |
| Activation function = "ReLU" | Accuracy (train data): 1 |
| Num of hidden layers = 3 | Loss (train data): 0.0045 |
| Num of hidden layer units = 800 | Accuracy (test data): 0.9705 |
| Learning rates = 0.009 | Loss (test data): 0.09423 |
| Momentum rates = 0.95 |
| Patience = 5 |
| Epoch = 27 |

Concluded that Cross-entropy error function is better than sum-of-squares error function as the accuracy with cross-entropy function is higher. Also, the number of epochs/iterations required before early stopping is quite high for sum-of-squares error function.

#### ReLU VS. tanh Hidden Units

| Configuration | Results |
| -- | -- |
| Activation function = "ReLU" | Execution time: 56.1165 Secs |
| Num of hidden layers = 3 | Accuracy (train data): 1 |
| Num of hidden layer units = 800 | Loss (train data): 0.0045 |
| Epoch = 27 | Accuracy (test data): 0.9705 |
| Learning rates = 0.009 | Loss (test data): 0.09423 |
| Momentum rates = 0.95 |
| Patience = 5 |

| Configuration | Results |
| -- | -- |
| Activation function = "tanh" | Execution time: 73.5298 Secs |
| Num of hidden layers = 3 | Accuracy (train data): 1 |
| Num of hidden layer units = 600 | Loss (train data): 0.0028 |
| Epoch = 69 | Accuracy (test data): 0.9677 |
| Learning rates = 0.05 | Loss (test data): 0.0961 |
| Momentum rates = 0.95 |
| Patience = 5 |

Concluded that using ReLU as the activation function is better than using tanh function. Also, execution time reduces when using ReLU as activation functions as compared to tanh activation functions.


### Convolutional Neural Network (CNN)

| Configuration | Results |
| -- | -- |
| Activation function = "ReLU" | Execution time: 48.29061 Secs |
| Num of hidden layers = 2 | Accuracy (train data): 0.9931 |
| Num of hidden layer units = 500, 300 | Loss (train data): 0.0194 |
| Learning rates = 0.05 | Accuracy (test data): 0.9788 |
| Momentum rates = 0.75 | Loss (test data): 0.0736 |
| Patience = 5 |
| Epoch = 34 |
| Filter Size = 64 |
| Kernel size = (2,2) |
| Dropout = 0.3 |

* Convolutional Networks has higher accuracy than the fully connected neural network
* CNNs have higher accuracy due to reduced overfitting problem as compared to fully connected neural networks.
* CNNS build the model within less number of epochs as compared to fully connected neural networks
