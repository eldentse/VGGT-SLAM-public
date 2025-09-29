import cv2
import numpy as np
from zapvision_py import ZapvisionTracker
import logging
from scipy.spatial.transform import Rotation as R

class Calibrator():
    def __init__(self) -> None:
        self.tracker_ = ZapvisionTracker()

    def detect_markers(self, img):
        height, width = img.shape[:2]

        self.tracker_.process(img, width, height, width)

        count = self.tracker_.result_count()
        # self.logger.info(f"number of detections: {count}")

        markers = []
        for i in range(count):
            type = self.tracker_.result_type(i)

            if type == 0: #qr code
                value = self.tracker_.result_qr_code(i)
            elif type == 1: #dense code
                value = self.tracker_.result_dense_code_value(i)

            landmarks = self.tracker_.result_landmarks(i) # [x1, y1, x2, y2, ... , xn, yn]

            if len(value) <= 0:
                value = None

            markers.append(
                {
                    'type': type,
                    'value': value,
                    'landmarks': landmarks
                }
            )
        return markers

    def preprocess_image(self, img, target_size=2056):
        height, width, channels = img.shape
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Determine the scale factor
        max_side = max(width, height)

        scale_factor = target_size / max_side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return img, scale_factor

    def camera_pose(self, portal, corners, camera_matrix, dist_coeffs):
        # self.logger.debug(f"portal_size: {portal['size']}")
        half_size = portal['size'] / 2
        # 3D points of the QR code in its own coordinate system (Z=0 plane)
        qr_world_coords = np.array([
            [-half_size,  half_size, 0],  # Top-left
            [ half_size,  half_size, 0],  # Top-right
            [ half_size, -half_size, 0],  # Bottom-right
            [-half_size, -half_size, 0]   # Bottom-left
        ], dtype=np.float32)
        # qr_world_coords = np.array([
        #     [-half_size,  -half_size, 0],  # Top-left
        #     [ half_size,  -half_size, 0],  # Top-right
        #     [ half_size, half_size, 0],  # Bottom-right
        #     [-half_size, half_size, 0]   # Bottom-left
        # ], dtype=np.float32)
        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(qr_world_coords, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        # success, rvec, tvec, _ = cv2.solvePnPRansac(qr_world_coords, corners, camera_matrix, dist_coeffs, 
        #                                             iterationsCount=100, reprojectionError=1.0, flags=cv2.SOLVEPNP_EPNP)
        # self.logger.debug(f"solvepnp:")
        # self.logger.debug(f"\ttvec: \n{tvec}")
        # self.logger.debug(f"\trvec: \n{rvec}")

        if not success:
            return None
        
        R, _ = cv2.Rodrigues(rvec)
        T_Camera_QR = np.eye(4, dtype=np.float64)  # Initialize as identity
        T_Camera_QR[:3, :3] = R
        T_Camera_QR[:3, 3] = tvec.flatten()

        # self.logger.debug(f"T_Camera_QR: \n{T_Camera_QR}")
        # self.logger.debug(f"Portal {portal['short_id']} Pose: \n{portal['pose']}")
        
        # T Domain Camera = T_Domain_QR * T_QR_Camera
        T_Domain_Camera = portal['pose'] @ np.linalg.inv(T_Camera_QR)

        return T_Domain_Camera # T_Reference_Object


def average_transforms(transforms):
    rotations = []
    translations = []

    for T in transforms:
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]
        rotations.append(R_mat)
        translations.append(t_vec)

    # Convert to rotation objects
    rot_objs = R.from_matrix(rotations)

    # Average the rotations using quaternion averaging
    mean_rot = rot_objs.mean().as_matrix()

    # Average the translations
    mean_trans = np.mean(translations, axis=0)

    # Construct average transform
    T_avg = np.eye(4)
    T_avg[:3, :3] = mean_rot
    T_avg[:3, 3] = mean_trans
    return T_avg