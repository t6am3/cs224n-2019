#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn


### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """ CNN module:
        - Conv1d layer
        - MaxPool layer
    """
    def __init__(self,
             embed_char_size: int=50,
             filters_size: int=256,
             kernel_size: int=5, 
             max_word_length: int=21):
        """ Init 1d cnn
        
        @param embed_char_size (int): embed_char_size, e_char in handout, and in_channels_size in Conv1d layer
        @param filters_size (int): filters_size, f in handout, and out_channels_size in Conv1d layer
        @param kernel_size (int): kernel_size, k in handout, kernel_size of Conv1d
        @max_word_length (int): max_word_length after padding & truncating, m_word in handout
        
        docs:
        nn.Conv1d: https://pytorch.org/docs/stable/nn.html#conv1d
        nn.MaxPool1d: https://pytorch.org/docs/stable/nn.html#maxpool1d
        """
        super(CNN, self).__init__()
        
        self.in_channels_size = embed_char_size
        self.out_channels_size = filters_size
        self.kernel_size = kernel_size
        self.max_word_length = max_word_length
        
        self.conv = nn.Conv1d(
            in_channels=self.in_channels_size, 
            out_channels=self.out_channels_size, 
            kernel_size=self.kernel_size
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_word_length-kernel_size+1)
        )
    
    def forward(self, X_reshaped):
        """ Map X_reshaped to X_conv_out
        
        @param X_reshaped (Tensor): X_reshaped in handout, (max_sentence_length, batch_size, embed_char_size, max_word_length)
        @return X_conv_out (Tensor): X_conv_out in handout, (max_sentence_length, batch_size, embed_word_size)
        """
        # We cant directly use self.conv to fit X_reshaped, it will raise a shape dismatch error
        X_conv = [self.conv(X_reshaped_i) for X_reshaped_i in X_reshaped]
        
        # Same attention as Conv1d operation above, because our pooling operation eliminate a dimension, so use torch.squeeze
        X_conv_out = torch.stack([self.output(X_conv_i) for X_conv_i in X_conv], dim=0).squeeze(-1)
        return X_conv_out
### END YOUR CODE

