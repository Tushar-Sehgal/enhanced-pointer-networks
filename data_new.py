import torch

def sample(array_len=10, max_val=100):
    """Generate a random sequence of integers."""
    return torch.randint(0, max_val, (array_len,))

def batch(batch_size, min_len=5, max_len=10, max_val=100):
    """Generate a batch of random sequences and their sorted indices."""
    # Generate a random sequence length for each batch
    array_len = torch.randint(min_len, max_len + 1, (1,)).item()
    
    # Create a batch of random sequences
    x = torch.randint(0, max_val, (batch_size, array_len))
    
    # Generate the corresponding sorted indices for each sequence
    y = x.argsort(dim=1)  # Get sorted indices
    
    return x, y
