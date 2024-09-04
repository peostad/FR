import torch
import onnxruntime
import numpy as np
from EdgeFace.backbones import get_model

def load_pytorch_model():
    model = get_model("edgeface_s_gamma_05")
    checkpoint_path = 'EdgeFace/checkpoints/edgeface_s_gamma_05.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def run_pytorch_inference(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy()

def run_onnx_inference(onnx_path, input_array):
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    ort_output = ort_session.run(None, ort_inputs)
    return ort_output[0]

def compare_outputs(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5):
    np.testing.assert_allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    print("PyTorch and ONNX outputs are similar within the specified tolerance.")

def main():
    # Load PyTorch model
    pytorch_model = load_pytorch_model()

    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 112, 112)

    # Run inference with PyTorch model
    pytorch_output = run_pytorch_inference(pytorch_model, input_tensor)

    # Run inference with ONNX model
    onnx_path = "edgeface_xs_gamma_06.onnx"
    onnx_output = run_onnx_inference(onnx_path, input_tensor.numpy())

    # Compare outputs
    compare_outputs(pytorch_output, onnx_output)

    print("Model verification completed successfully!")

if __name__ == "__main__":
    main()
