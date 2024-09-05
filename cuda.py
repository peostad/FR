import onnxruntime as ort

# Print available execution providers
print("Available Execution Providers:")
print(ort.get_available_providers())

# Initialize ONNX Runtime session to see if CUDA is used
session = ort.InferenceSession("yolov7_tiny_threeclass.onnx")

# Check the execution providers used in the session
print("Execution Providers used in the session:")
print(session.get_providers())
