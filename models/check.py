import torch

path = "stg_nf_latest.pth"
data = torch.load(path, map_location="cpu")

print(type(data))
