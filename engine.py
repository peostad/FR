import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path, precision='fp16'):
    """
    Converts an ONNX model to a TensorRT engine and saves it.
    
    Args:
    onnx_file_path (str): Path to the ONNX model file
    engine_file_path (str): Path where the TensorRT engine should be saved
    precision (str): Precision mode, either 'fp32', 'fp16', or 'int8'
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX file
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    
    config.max_workspace_size = 1 << 30  # 1GB

    # Build and save the engine
    engine = builder.build_engine(network, config)
    
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine has been built and saved to {engine_file_path}")

if __name__ == "__main__":
    onnx_model_path = "yolov7_tiny_threeclass.onnx"
    engine_path = "yolov7_tiny_threeclass.engine"
    
    build_engine(onnx_model_path, engine_path)
