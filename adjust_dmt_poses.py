import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import glob 
import os
import yaml
import sys
import cv2
from tqdm.contrib import tenumerate

from domain import Domain 
from marker_calibrator import Calibrator, average_transforms

import vggt_slam.slam_utils as utils

parser = argparse.ArgumentParser(description="load_cam_pose")
parser.add_argument("--pose_file", type=str, required=True, help="Path to folder containing images")
parser.add_argument("--intrinsics", type=str, required=True, help="Path to folder containing images")
parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
parser.add_argument("--output", type=str, required=True, help="Path to folder containing images")
parser.add_argument("--domain_config", type=str, default="domain_config.yaml", help="configuration file for login auki domain")

def aqr_calibration(domain, calibrator, image_path, 
                    predicted_intrinsics=None,
                    dist_coeffs=None, ):
    
    image_sizes = [3480, 2560, 1920, 1280, 640]
    
    image_origin = cv2.imread(image_path)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    tfs = []
    for size in image_sizes:
        # Preprocess Image
        image, scale = calibrator.preprocess_image(image_origin, size)

        # only scale up
        if scale > 1.0:
            continue

        # Detect Markers
        markers = calibrator.detect_markers(image)

        height, width = image.shape

        if predicted_intrinsics is not None:
            camera_matrix = predicted_intrinsics.copy()
            camera_matrix[0, 0] *= scale  # fx
            camera_matrix[0, 2] *= scale  # cx
            camera_matrix[1, 1] *= scale  # fy
            camera_matrix[1, 2] *= scale  # cy
        else:
            focal_length = width * 0.7
            camera_matrix = np.array([
                [focal_length,   0, width/2],
                [  0, focal_length, height/2],
                [  0,   0,   1]
            ], dtype=np.float64)

        for marker in markers:
            if marker['value'] is None:
                print(f"unable to decode qr, value: {marker['value']}")
                continue

            if "HTTPS://R8.HR/" not in marker['value']:
                print(f"detected qr is not a portal, decoding: { marker['value']}")
                continue

            short_id = marker['value'].replace("HTTPS://R8.HR/", "")
            print(f"Detected Portal: {short_id}")
            portal = domain.portals().get(short_id, None)
            if portal is None:
                print(f"{short_id} not found in domain:")
                continue
            
            if len(marker['landmarks']) == 14:
                qr_corners = marker['landmarks'][6:]
                corners = np.array(list(zip(qr_corners[::2], qr_corners[1::2])))
                tf_matrix = calibrator.camera_pose(portal, np.array(corners), camera_matrix, dist_coeffs)
                if tf_matrix is not None:
                    tfs.append(tf_matrix)
                else:
                    print(f"failed to calibrate for portal {short_id}")
            else:
                print(f"AQR's QR code not detected {len(marker['landmarks'])}")

    if len(tfs) < 1:
        # print(f"No Valid Markers Detected")
        return None
        
    T_Map_Camera_avg = average_transforms(tfs)

    return T_Map_Camera_avg


def load_camera_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 8:
                continue  # skip malformed lines
            _, x, y, z, qx, qy, qz, qw = parts
            quat = [float(qx), float(qy), float(qz), float(qw)]
            trans = [float(x), float(y), float(z)]

            # Convert quaternion to rotation matrix
            rot = R.from_quat(quat).as_matrix()

            # Change-of-basis matrix from ARKit to OpenCV (Camera)
            theta = np.deg2rad(-90)
            R_z90 = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])

            theta_2 = np.deg2rad(180)  # 180 degrees in radians
            R_x180 = np.array([
                [1,             0,              0],
                [0, np.cos(theta_2), -np.sin(theta_2)],
                [0, np.sin(theta_2),  np.cos(theta_2)]
            ])

            # Apply rotation: rotate camera frame
            rot = rot @ R_x180 @ R_z90

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = trans
            poses.append(T)
    return poses

def load_camera_info(file_path):
    infos = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue  # skip malformed lines
            _, fx, fy, cx, cy, _, _ = parts
            camera_matrix = np.array([
                [float(fx),   0, float(cx)],
                [  0, float(fy), float(cy)],
                [  0,   0,   1],
            ])


            infos.append(camera_matrix)
    return infos

def write_pose_ace0fmt(poses, file_path, focal_length):
    with open(file_path, 'w') as f:
        for i, pose in enumerate(poses):
            x, y, z = pose[0:3, 3]
            rotation_matrix = pose[0:3, 0:3]
            quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w

            pose_str = f"image_{i} " \
            f"{quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} " \
            f"{x} {y} {z} {focal_length} {2000}\n"
            f.write(pose_str)

if __name__ == "__main__":
    args = parser.parse_args()

    # Auki Domain
    with open(args.domain_config) as stream:
        try:
            domain_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"load yaml error: {exc}")
            sys.exit(1)

    domain = Domain(domain_config)
    ret, msg = domain.auth()
    if not ret:
        print(f"domain authentication failed. message: {msg}")
        sys.exit(1)
    print(f"domain authenticated {ret} {msg}")
    domain.fetch_portal_poses()


    # AQR Calibrator
    marker_calibrator = Calibrator()

    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    print(f"number of images: {len(image_names)}")

    poses = load_camera_poses(args.pose_file)
    print(f"number of poses: {len(poses)}")

    infos = load_camera_info(args.intrinsics)
    print(f"number of infos: {len(infos)}")

    adjustment_matrics = []
    adjustment_matrix = None
    ref_image = None
    T_map_ref_image = None
    index = None
    for i, image_name in tenumerate(image_names):
        if adjustment_matrix is not None:
            break
        T_map_image = aqr_calibration(domain, marker_calibrator, image_name, infos[i])

        if T_map_image is not None:
            index = i
            T_dmt_image = poses[i]
            # Calculate difference and apply adjustment matrix
            adjustment_matrix = T_map_image @ np.linalg.inv(T_dmt_image)
            ref_image = image_name
            adjustment_matrics.append(adjustment_matrix)



    print(f"adjusted based on image: {ref_image}")
    print(f"T_dmt_image: \n{T_dmt_image}")
    print(f"T_map_ref_image: \n{T_map_ref_image}")
    print(f"adjustment_matrix: \n{adjustment_matrix}")
    print(f"Adjusted Ref Image: \n{adjustment_matrix @ T_dmt_image}")

    avg_adj_matrix = average_transforms(adjustment_matrics)
    print(f"avg_adj_matrix: \n{avg_adj_matrix}")
    adj_poses = [adjustment_matrix @ pose for pose in poses]
    print(f"Adjusted Ref Image2 : \n{adj_poses[index]}")
    write_pose_ace0fmt(adj_poses, args.output, 1424.674)

