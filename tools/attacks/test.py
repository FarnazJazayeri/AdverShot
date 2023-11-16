import numpy as np
import torch
from pgd import PGD


# Given numpy array
numpy_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# Given torch tensor of indices
indices_tensor = torch.tensor([0, 3, 3, 4, 6, 7]) + 2 * torch.ones_like(torch.tensor([0, 3, 3, 4, 6, 7]))

# Extract elements based on indices using numpy
selected_elements = numpy_array[indices_tensor]

# Convert the numpy array back to a torch tensor
selected_tensor = torch.from_numpy(selected_elements)

print(selected_tensor)