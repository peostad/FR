import onnxruntime as ort
import numpy as np
import cv2
import time

class Face:
    def __init__(self, xmin, ymin, xmax, ymax, score):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score

class Anchor:
    def __init__(self, cx, cy, width, height):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

class FaceDetector:
    def __init__(self, model_path):
        # Initialize ONNX Runtime environment with CUDA support
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CUDAExecutionProvider']  # Specify CUDA as the provider
        self.ort_session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        
        # Get input and output tensor names
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        
        # Initialize model input dimensions
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        # Placeholder for anchors
        self.anchors = []
        self.create_anchors()

    def create_anchors(self):
        self.anchors.clear()
        # Example strides and anchor sizes (Modify as per your model)
        strides = [8, 16, 32]
        anchor_sizes = [[32, 64], [128, 256], [512]]
        
        # Generate anchors
        for k, stride in enumerate(strides):
            for y in range(0, self.input_height, stride):
                for x in range(0, self.input_width, stride):
                    cx = x + stride / 2
                    cy = y + stride / 2
                    for size in anchor_sizes[k]:
                        anchor = Anchor(cx, cy, size, size)
                        self.anchors.append(anchor)

    def non_maximum_suppression(self, faces, iou_threshold):
        # Sort faces by score
        faces = sorted(faces, key=lambda x: x.score, reverse=True)
        selected_faces = []

        while faces:
            face = faces.pop(0)
            selected_faces.append(face)
            faces = [f for f in faces if self.iou(face, f) < iou_threshold]
        
        return selected_faces

    def iou(self, face1, face2):
        # Calculate intersection over union (IoU)
        inter_xmin = max(face1.xmin, face2.xmin)
        inter_ymin = max(face1.ymin, face2.ymin)
        inter_xmax = min(face1.xmax, face2.xmax)
        inter_ymax = min(face1.ymax, face2.ymax)

        inter_width = max(inter_xmax - inter_xmin + 1, 0)
        inter_height = max(inter_ymax - inter_ymin + 1, 0)
        inter_area = inter_width * inter_height

        area1 = (face1.xmax - face1.xmin + 1) * (face1.ymax - face1.ymin + 1)
        area2 = (face2.xmax - face2.xmin + 1) * (face2.ymax - face2.ymin + 1)
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area

    def process_input_image(self, image):
        # Preprocess image to fit the model input size
        scale_x = self.input_width / image.shape[1]
        scale_y = self.input_height / image.shape[0]
        scale = min(scale_x, scale_y)

        resized_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        # Create input tensor
        input_tensor = np.zeros((3, self.input_height, self.input_width), dtype=np.float32)
        # Fill in the resized image data in BGR order, normalized
        input_tensor[0, :resized_image.shape[0], :resized_image.shape[1]] = resized_image[:, :, 0] - 104.0
        input_tensor[1, :resized_image.shape[0], :resized_image.shape[1]] = resized_image[:, :, 1] - 117.0
        input_tensor[2, :resized_image.shape[0], :resized_image.shape[1]] = resized_image[:, :, 2] - 123.0

        return input_tensor, scale

    def process_bounding_boxes(self, locations, classifications, score_threshold, scale):
        faces = []
        for i, anchor in enumerate(self.anchors):
            score = classifications[i * 2 + 1]
            if score < score_threshold:
                continue

            pred_dx, pred_dy, pred_w, pred_h = locations[i*4:i*4+4]

            bbox_cx = anchor.cx + pred_dx * 0.1 * anchor.width
            bbox_cy = anchor.cy + pred_dy * 0.1 * anchor.height
            bbox_w = np.exp(pred_w * 0.2) * anchor.width
            bbox_h = np.exp(pred_h * 0.2) * anchor.height

            xmin = (bbox_cx - bbox_w / 2.0) / scale
            ymin = (bbox_cy - bbox_h / 2.0) / scale
            xmax = (bbox_cx + bbox_w / 2.0) / scale
            ymax = (bbox_cy + bbox_h / 2.0) / scale

            faces.append(Face(xmin, ymin, xmax, ymax, score))

        return self.non_maximum_suppression(faces, iou_threshold=0.4)

    def detect_faces(self, image, score_threshold=0.5):
        # Preprocess input image
        input_tensor, scale = self.process_input_image(image)

        # Run inference
        ort_inputs = {self.input_name: np.expand_dims(input_tensor, axis=0)}
        start_time = time.time()  # Start time for detection
        ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
        detection_time = (time.time() - start_time) * 1000  # Calculate detection time in milliseconds
        
        # Process bounding boxes
        locations = ort_outputs[0].flatten()
        classifications = ort_outputs[1].flatten()
        faces = self.process_bounding_boxes(locations, classifications, score_threshold, scale)

        return faces, detection_time

# Main function for live camera feed with FPS and detection time display
def main():
    model_path = 'models/face_detector.onnx'  # Replace with your ONNX model path
    
    # Initialize the detector
    detector = FaceDetector(model_path)

    # Open the default camera (change the index if using an external camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize variables for FPS calculation
    prev_time = time.time()
    fps = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Detect faces in the current frame
        faces, detection_time = detector.detect_faces(frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Draw detected faces on the frame
        for face in faces:
            cv2.rectangle(frame, (int(face.xmin), int(face.ymin)), (int(face.xmax), int(face.ymax)), (255, 0, 0), 2)
            cv2.putText(frame, f"{face.score:.2f}", (int(face.xmin), int(face.ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Display FPS and detection time on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detection Time: {detection_time:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Live Face Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
