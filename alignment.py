import cv2
import numpy as np
from mmpose.apis import init_pose_model, inference_pose_model
from mmpose.datasets import DatasetInfo

def align_face(image, face_bbox):
    # Initialize the face keypoint detection model
    config_file = 'mmpose/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256.py'
    checkpoint_file = 'mmpose/checkpoints/hrnetv2_w18_coco_wholebody_face_256x256-4f9176cf_20210909.pth'
    pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')

    # Prepare the dataset info
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get('dataset_info', None))
    
    # Extract face from bounding box
    x1, y1, x2, y2 = face_bbox
    face_img = image[y1:y2, x1:x2]
    
    # Perform inference
    pose_results, _ = inference_pose_model(pose_model, face_img, dataset_info=dataset_info)

    if len(pose_results) == 0:
        return None  # No face keypoints detected

    keypoints = pose_results[0]['keypoints']
    
    # Extract eye and mouth coordinates
    left_eye = keypoints[36].astype(int)
    right_eye = keypoints[45].astype(int)
    left_mouth = keypoints[48].astype(int)
    right_mouth = keypoints[54].astype(int)

    # Calculate angle for rotation
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Rotate the image
    center = tuple(np.array([face_img.shape[1] / 2, face_img.shape[0] / 2]))
    rotation_matrix = cv2.getRotationMatrix2D(center, eye_angle, 1)
    rotated_image = cv2.warpAffine(face_img, rotation_matrix, (face_img.shape[1], face_img.shape[0]))

    # Recalculate keypoints after rotation
    for i in range(keypoints.shape[0]):
        keypoints[i] = np.dot(rotation_matrix, [keypoints[i][0], keypoints[i][1], 1])[:2]

    # Calculate the desired size and position
    left_eye = keypoints[36].astype(int)
    right_eye = keypoints[45].astype(int)
    left_mouth = keypoints[48].astype(int)
    right_mouth = keypoints[54].astype(int)

    eye_center = ((left_eye + right_eye) / 2).astype(int)
    mouth_center = ((left_mouth + right_mouth) / 2).astype(int)

    desired_eye_center_y = int(rotated_image.shape[0] * 0.35)
    desired_mouth_center_y = int(rotated_image.shape[0] * 0.65)

    scale = (desired_mouth_center_y - desired_eye_center_y) / (mouth_center[1] - eye_center[1])

    # Perform scaling and translation
    translation_matrix = np.array([
        [scale, 0, rotated_image.shape[1]/2 - eye_center[0]*scale],
        [0, scale, desired_eye_center_y - eye_center[1]*scale]
    ])

    aligned_image = cv2.warpAffine(rotated_image, translation_matrix, (rotated_image.shape[1], rotated_image.shape[0]))

    return aligned_image



