"""
Enhanced Pointer Network based on the architecture proposed at: https://arxiv.org/abs/1506.03134
This version includes deeper LSTM layers, refined attention mechanism, and gradient clipping.
"""

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_new import sample, batch

HIDDEN_SIZE = 256
NUM_LAYERS = 2  # Deeper LSTM layers

BATCH_SIZE = 32
STEPS_PER_EPOCH = 500
EPOCHS = 10


class Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)  # Deeper LSTM
    
    def forward(self, x: torch.Tensor):
        return self.lstm(x)  # Encoder LSTM layer outputs


class Attention(nn.Module):
    def __init__(self, hidden_size, units):
        super(Attention, self).__init__()
        # Refined attention mechanism
        self.W1 = nn.Linear(hidden_size, units * 2, bias=False)  # More attention units for better focus
        self.W2 = nn.Linear(hidden_size, units * 2, bias=False)
        self.V = nn.Linear(units * 2, 1, bias=False)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        # Add time axis to decoder hidden state to make operations compatible with encoder output
        decoder_hidden_time = decoder_hidden.unsqueeze(1)
        uj = torch.tanh(self.W1(encoder_out) + self.W2(decoder_hidden_time))  # Attention score
        uj = self.V(uj)

        # Attention mask over inputs
        aj = F.softmax(uj, dim=1)

        # Calculate the attention-weighted encoder output
        di_prime = aj * encoder_out
        di_prime = di_prime.sum(1)

        return di_prime, uj.squeeze(-1)


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, attention_units: int = 10, num_layers: int = 1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size + 1, hidden_size, num_layers=num_layers, batch_first=True)  # Deeper LSTM
        self.attention = Attention(hidden_size, attention_units)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor], encoder_out: torch.Tensor):
        ht = hidden[0][0]  # Decoder hidden state

        # Attention-aware hidden state
        di, att_w = self.attention(encoder_out, ht)

        # Append attention-aware hidden state to input
        x = torch.cat([di.unsqueeze(1), x], dim=2)

        # Generate next hidden state
        _, hidden = self.lstm(x, hidden)
        return hidden, att_w


class PointerNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(PointerNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, y: torch.Tensor, teacher_force_ratio=.5):
        encoder_in = x.unsqueeze(-1).type(torch.float)

        # Encode input sequence
        out, hs = self.encoder(encoder_in)

        # Save outputs at each timestep
        outputs = torch.zeros(out.size(1), out.size(0), dtype=torch.long)

        # First decoder input is always 0
        dec_in = torch.zeros(out.size(0), 1, 1, dtype=torch.float)

        loss = 0
        for t in range(out.size(1)):
            hs, att_w = self.decoder(dec_in, hs, out)
            predictions = F.softmax(att_w, dim=1).argmax(1)

            # Teacher forcing
            teacher_force = random.random() < teacher_force_ratio
            idx = y[:, t] if teacher_force else predictions
            dec_in = torch.stack([x[b, idx[b].item()] for b in range(x.size(0))])
            dec_in = dec_in.view(out.size(0), 1, 1).type(torch.float)

            # Calculate loss
            loss += F.cross_entropy(att_w, y[:, t])
            outputs[t] = predictions

        batch_loss = loss / y.size(0)
        return outputs, batch_loss


def train(model, optimizer, epoch, clip=1.):
    """Train for a single epoch"""
    print(f'Epoch [{epoch}] -- Train')
    for step in range(STEPS_PER_EPOCH):
        optimizer.zero_grad()

        # Fetch a batch of data
        x, y = batch(BATCH_SIZE)

        # Forward pass through the model
        out, loss = model(x, y)

        # Backward pass (backpropagation)
        loss.backward()

        # Clip gradients to avoid explosion
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update model parameters
        optimizer.step()

        # Logging the loss every 100 steps
        if (step + 1) % 100 == 0:
            print(f'Epoch [{epoch}] loss: {loss.item()}')


@torch.no_grad()
def evaluate(model, epoch):
    """Evaluate the model after each epoch"""
    print(f'Epoch [{epoch}] -- Evaluate')

    # Fetch a small batch for evaluation
    x_val, y_val = batch(4)
  
    # Pass the validation data through the model without teacher forcing
    out, _ = model(x_val, y_val, teacher_force_ratio=0.)
    
    # Permute to match expected output format
    out = out.permute(1, 0)

    # Print the input, predicted, and actual sequences for each example in the batch
    for i in range(out.size(0)):
        print(f'{x_val[i]} --> {x_val[i].gather(0, out[i])} --> {x_val[i].gather(0, y_val[i])}')


# Instantiate encoder, decoder, and the pointer network model
encoder = Encoder(HIDDEN_SIZE, NUM_LAYERS)
decoder = Decoder(HIDDEN_SIZE, num_layers=NUM_LAYERS)
ptr_net = PointerNetwork(encoder, decoder)

# Use Adam optimizer for efficient gradient updates
optimizer = optim.Adam(ptr_net.parameters())

# Training and evaluation loop
for epoch in range(EPOCHS):
    train(ptr_net, optimizer, epoch + 1)
    evaluate(ptr_net, epoch + 1)
