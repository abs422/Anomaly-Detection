# Anomaly-Detection

The first multivariate LSTM has following structure, explained in words (block diagram will be available soon) - 

The code defines an autoencoder model using the Tensorflow framework. An autoencoder is a type of neural network that is trained to reconstruct its inputs. The code uses the Long Short-Term Memory (LSTM) architecture, which is a type of Recurrent Neural Network (RNN) architecture specifically designed to handle sequences of data.

The code starts by defining the inputs to the model, which is expected to have the shape (X.shape[1], X.shape[2]). The input shape specifies that the model is expecting a 2-dimensional array where the first dimension is X.shape[1] and the second dimension is X.shape[2].

Next, the code defines the first LSTM layer (L1) with 16 neurons and a 'relu' activation function. The return_sequences parameter is set to True, which means that the outputs of this layer will be passed on to the next layer as a sequence. The kernel_regularizer parameter specifies that L2 regularization will be used to prevent overfitting.

The second LSTM layer (L2) has 4 neurons and a 'relu' activation function. The return_sequences parameter is set to False, which means that the outputs of this layer will not be passed on to the next layer as a sequence.

The third layer (L3) is a RepeatVector layer, which takes the outputs from L2 and repeats them X.shape[1] times. This is necessary for the autoencoder to work correctly as it requires that the output of the encoding part of the network have the same shape as the input.

The fourth and fifth LSTM layers (L4 and L5) have 4 and 16 neurons, respectively, and a 'relu' activation function. The return_sequences parameter is set to True, which means that the outputs of these layers will be passed on to the next layer as a sequence.

The final layer (output) is a TimeDistributed dense layer, which applies a dense (fully connected) layer to each time step of the input. The number of neurons in this layer is specified to be the same as the number of columns in X.shape[2].

# Graph Adversarial Networks
Sources - https://github.com/gusty1g/TadGAN


# Reconstruction Loss Logic

