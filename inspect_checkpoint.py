import torch

checkpoint = torch.load("models/stg_nf_latest.pth", map_location='cpu')

print("Checkpoint keys:", checkpoint.keys())
print("\nEpoch:", checkpoint.get('epoch', 'N/A'))

# Inspect state_dict to infer model configuration
state_dict = checkpoint['state_dict']

# Look at key shapes to infer configuration
print("\n=== Key Model Parameters ===")
for key in ['prior_h', 'prior_h_normal', 'prior_h_abnormal']:
    if key in state_dict:
        print(f"{key}: {state_dict[key].shape}")

# Check flow layer 0 to understand dimensions
print("\n=== Flow Layer 0 Dimensions ===")
for key in state_dict.keys():
    if 'flow.layers.0.block.0' in key:
        print(f"{key}: {state_dict[key].shape}")
        
# Count total flow layers
flow_layers = set()
for key in state_dict.keys():
    if 'flow.layers.' in key:
        layer_num = int(key.split('flow.layers.')[1].split('.')[0])
        flow_layers.add(layer_num)
        
print(f"\n=== Total Flow Layers: {len(flow_layers)} ===")
print(f"Layer indices: {sorted(flow_layers)}")

# Check for learn_top_fn
print("\n=== Learn Top Function ===")
for key in state_dict.keys():
    if 'learn_top_fn' in key:
        print(f"{key}: {state_dict[key].shape}")
