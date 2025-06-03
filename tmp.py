import torch
import torch.nn.functional as F

logits = torch.tensor([[-6.9946, -3.6217],
        [-6.4035, -2.9512],
        [-6.6786, -3.0779],
        [-7.4148, -3.5390],
        [-3.1820, -0.4259],
        [-7.2820, -4.4349],
        [-6.7757, -3.3081],
        [-7.7839, -3.4579]])
labels = torch.tensor([3919, 3919, 3919, 3919, 3919, 8505, 8505, 3919])

print(f"logits.shape=${logits.shape}")
print(f"labels.shape=${labels.shape}")

# On CPU: Target 3919 is out of bounds.
#out = F.cross_entropy(logits, labels, reduction='mean')

# On MPS: out is tensor(0., device='mps:0')
out = F.cross_entropy(logits.to(torch.device("mps")), labels.to(torch.device("mps")), reduction='mean')

print(out)
