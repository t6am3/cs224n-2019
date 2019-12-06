#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """Highway module:
            - projection
            - gate
            - skip-connection
            X_proj = ReLU(W_proj.matmul(X_conv_out) + b_proj)
            X_gate = Sigmoid(W_gate.matmul(X_conv_out) + b_gate)
            X_highway = X_gate * X__proj + (1 - X_gate) * X_conv_out
    """
    def __init__(self, 
             word_embed_size: int=256):
        """ Init Highway model
        @param word_embed_size (int): Embedding size of word
        """
        super(Highway, self).__init__()
            
        self.word_embed_size = word_embed_size
        
        self.projection = nn.Sequential(
            nn.Linear(self.word_embed_size, self.word_embed_size),
            nn.ReLU()
        )
        self.gate = nn.Sequential(
            nn.Linear(self.word_embed_size, self.word_embed_size),
            nn.Sigmoid()
        )
        
    def forward(self, X_conv_out):
        """Compute the projection of inputs and gate_value, combine them together to get final word embeddings.
        
        @param X_conv_out (Tensor): X_conv_out, (max_sentence_length, batch_size, word_embed_size)
        @return X_highway (Tensor): X_highway, (max_sentence_length, batch_size, word_embed_size)
        """
        X_proj = self.projection(X_conv_out)        
        X_gate = self.gate(X_conv_out)        
        X_highway = X_gate * X_proj + (1 - X_gate) * X_conv_out
        return X_highway
### END YOUR CODE 

