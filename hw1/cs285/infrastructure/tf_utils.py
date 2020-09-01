# import tensorflow as tf
import os

# added by Fangda @2020/8
import torch
import torch.nn as nn


############################################
############################################

def build_mlp(output_size, n_layers, size, output_activation=None):
    """
        Builds a feedforward neural network

        arguments:
            
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_model: the tensorflow model of a forward pass through the hidden layers + the output layer
    """
    
    model_list=[]

    for _ in range(n_layers):
            # HINT: use torch.nn.Linear() + torch.nn.Tanh()
        # model_list.append(tf.keras.layers.Dense(size, activation='tanh'))
        model_list.append(nn.Linear(size, size))
        model_list.append(nn.Tanh())
    
    model_list.append(nn.Linear(size, output_size))
    return nn.Sequential(*model_list)


############################################
############################################

