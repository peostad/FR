import torch
from EdgeFace.backbones import get_model

# Load the model
model = get_model("edgeface_s_gamma_05")
checkpoint_path = 'EdgeFace/checkpoints/edgeface_s_gamma_05.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

# Example: if the model expects a batch of 1 image of size (3, 112, 112)
dummy_input = torch.randn(1, 3, 112, 112)

# Export the model
torch.onnx.export(
    model,                       # The PyTorch model to be exported
    dummy_input,                 # A dummy input for tracing the model
    "edgeface_s_gamma_05.onnx", # Path where the ONNX model will be saved
    export_params=True,          # Store the trained parameter weights
    opset_version=11,            # ONNX version to export the model to
    do_constant_folding=True,    # Whether to execute constant folding for optimization
    input_names=['input'],       # Input tensor names (optional)
    output_names=['output'],     # Output tensor names (optional)
    dynamic_axes={'input': {0: 'batch_size'},  # Enable dynamic axes for the input
                  'output': {0: 'batch_size'}} # Enable dynamic axes for the output
)

print("Model has been converted to ONNX format and saved as edgeface_s_gamma_05.onnx")
