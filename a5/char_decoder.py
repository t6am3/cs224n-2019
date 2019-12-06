#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.target_vocab = target_vocab
        self.char_embedding_size = char_embedding_size
        
        self.charDecoder = nn.LSTM(
            input_size=self.char_embedding_size,
#             num_layers=2,
            hidden_size=self.hidden_size
        )
        self.char_output_projection = nn.Linear(self.hidden_size, len(self.target_vocab.char2id))
        
        self.decoderCharEmb = nn.Embedding(
            num_embeddings=len(self.target_vocab.char2id),
            embedding_dim=self.char_embedding_size,
            padding_idx=self.target_vocab.char2id['<pad>']
        )
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embeddings = self.decoderCharEmb(input)
        output, dec_hidden = self.charDecoder(char_embeddings, dec_hidden)
        scores = self.char_output_projection(output)
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        # Shape of scores: (length, batch, vocab_size)
        scores, dec_hidden = self(char_sequence[:-1, :], dec_hidden)
        cel = torch.nn.CrossEntropyLoss(
            ignore_index=self.target_vocab.char2id['<pad>'],
            reduction='sum'
        )     
        cross_entropy_loss = cel(scores.permute(1, 2, 0), char_sequence[1:, :].transpose(0, 1))
        return cross_entropy_loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].size(1)
        decodeWords = output_words = [[] for _ in range(batch_size)]
        
        # Current_chars as the input of our forward method, shape is (length, batch_size), for there we have(1, batch_size)
        current_chars = torch.tensor([[self.target_vocab.start_of_word for _ in range(batch_size)]], device=device)
        
        # You are such a dummmmmmie, if you dont hidden_states, what's the usage of lstm？
        dec_hidden = initialStates

        for _ in range(max_length):
            # Shape of s_tp : (1, batch_size, self.vocab_size)
            s_tp, dec_hidden = self(current_chars, dec_hidden)
            # Because e**x is a Monotonically increasing function we could just use the argmax but not argmax(softmax)
#             current_chars = torch.argmax(F.softmax(s_tp, dim=2), dim=2)
            current_chars = s_tp.argmax(2)
            # Append chars
            end_batch = []
            current_len = len(current_chars[0])
            for i in range(current_len):
                if current_chars[0][i] == self.target_vocab.end_of_word:
                    end_batch.append(i)
                else:
                    output_words[i].append(self.target_vocab.id2char[int(current_chars[0][i])])
                    if len(output_words[i]) == max_length:
                        end_batch.append(i)
            left_idxs = [i for i in range(current_len) if i not in end_batch]
            
            # Early Stop
            if len(left_idxs) == 0:
                break
            # Attention, if use this method, dont compute those end word, we should keep everything aligned
            current_chars = current_chars[:, left_idxs]
            output_words = [output_words[i] for i in left_idxs]
            dec_hidden = dec_hidden[0][:, left_idxs, :], dec_hidden[1][:, left_idxs, :]
            
        decodeWords = [''.join(i) for i in decodeWords]
        return decodeWords

