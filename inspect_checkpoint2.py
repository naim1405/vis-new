import torch

checkpoint = torch.load("models/stg_nf_latest.pth", map_location='cpu')
state_dict = checkpoint['state_dict']

# Check what's actually in layer 0
print("=== Flow Layer 0 Keys ===")
layer0_keys = [k for k in state_dict.keys() if 'flow.layers.0' in k]
for key in sorted(layer0_keys):
    print(f"  {key}")

# Check if there are invconv layers
print("\n=== Invconv Layers ===")
invconv_keys = [k for k in state_dict.keys() if 'invconv' in k]
print(f"Found {len(invconv_keys)} invconv keys")
if invconv_keys:
    for key in invconv_keys[:5]:
        print(f"  {key}")

# Check if there are block.1 layers
print("\n=== Block.1 Layers ===")
block1_keys = [k for k in state_dict.keys() if 'block.1' in k]
print(f"Found {len(block1_keys)} block.1 keys")

# Check actnorm layers
print("\n=== Actnorm Layers ===")
actnorm_keys = [k for k in state_dict.keys() if 'actnorm' in k]
print(f"Found {len(actnorm_keys)} actnorm keys")
if actnorm_keys:
    for key in actnorm_keys[:5]:
        print(f"  {key}")

# Check edge_importance
print("\n=== Edge Importance ===")
edge_keys = [k for k in state_dict.keys() if 'edge_importance' in k]
print(f"Found {len(edge_keys)} edge_importance keys")

print("\n=== All Flow Layers Structure ===")
for i in range(8):
    layer_keys = [k for k in state_dict.keys() if f'flow.layers.{i}.' in k]
    print(f"Layer {i}: {len(layer_keys)} parameters")
    # Show first 3 keys as example
    for key in sorted(layer_keys)[:3]:
        print(f"    {key}")
