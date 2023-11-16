import torch


def class_tensor_extract(batch_tensor, label_tensor, num_features=0, con_features=[(0, 1)]):
    # Initialize class tensors dictionary
    package = [0] * num_features
    class_tensors = {class_idx: [torch.zeros(0, batch_tensor.size(1), batch_tensor.size(2), batch_tensor.size(3))] for class_idx in range(11)}
    
    # Iterate through the batch tensor and label tensor
    for idx, label in enumerate(label_tensor):
        instance = batch_tensor[idx]  # Get the instance from the batch tensor
        class_idx = label.item()  # Get the class index from the label tensor
        
        # Concatenate the instance to the corresponding class tensor
        class_tensors[class_idx] = torch.cat((class_tensors[class_idx], instance.unsqueeze(0)), dim=0)
    return class_tensors
    

'''
# Example batch tensor and label tensor
batch_tensor = torch.randn(10, 3, 32, 32)  # Example shape (B x C x H x W)
label_tensor = torch.randint(0, 11, (10, 1))  # Example shape (B x 1)

# Initialize class tensors dictionary
class_tensors = {class_idx: torch.zeros(0, batch_tensor.size(1), batch_tensor.size(2), batch_tensor.size(3)) for class_idx in range(11)}

# Iterate through the batch tensor and label tensor
for idx, label in enumerate(label_tensor):
    instance = batch_tensor[idx]  # Get the instance from the batch tensor
    class_idx = label.item()  # Get the class index from the label tensor
    
    # Concatenate the instance to the corresponding class tensor
    class_tensors[class_idx] = torch.cat((class_tensors[class_idx], instance.unsqueeze(0)), dim=0)

# Display the shapes of the class tensors
for class_idx, tensor in class_tensors.items():
    print(f"Class {class_idx} tensor shape: {tensor.shape}")
'''

#batch_tensor = torch.randn(10, 3, 32, 32)  # Example shape (B x C x H x W)
#label_tensor = torch.randint(0, 11, (10, 1))  # Example shape (B x 1)
#class_tensors = class_tensor_extract(batch_tensor, label_tensor, 3)
#print(class_tensors)