import os
import glob
import yaml
import sys
import argparse

from domain import Domain 
from marker_calibrator import Calibrator, average_transforms

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt.models.vggt import VGGT

from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--save_intrinsics", action="store_true", help="save intrinsics")

parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")

parser.add_argument("--domain_config", type=str, default="domain_config.yaml", help="configuration file for login auki domain")

def aqr_calibration(domain, calibrator, image_path, 
                    predicted_intrinsics=None, predicted_intrinsics_image_width=518, 
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
        # print(f"image: {image_path}\timage_size: {size}\tnumber of markers: {len(markers)}")

        height, width = image.shape

        instrinsic_scale = width / predicted_intrinsics_image_width

        if predicted_intrinsics is not None:
            camera_matrix = predicted_intrinsics.copy()
            camera_matrix[0, 0] *= instrinsic_scale  # fx
            camera_matrix[0, 2] *= instrinsic_scale  # cx
            camera_matrix[1, 1] *= instrinsic_scale  # fy
            camera_matrix[1, 2] *= instrinsic_scale  # cy
        else:
            height, width = image.shape[:2]
            focal_length = width * 0.7
            camera_matrix = np.array([
                [focal_length,   0, width/2],
                [  0, focal_length, height/2],
                [  0,   0,   1]
            ], dtype=np.float64)

        # print(f"processing image shape: {image.shape}\t camera_matrix: {camera_matrix}")

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



def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
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
    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False
    )

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []
    intrinsics = []
    adjustment_matrix = None
    ref_image = None
    orig_image_width = 0
    scale_factor = 1
    for image_name in tqdm(image_names):
        img = cv2.imread(image_name)
        _, orig_image_width = img.shape[:2]
        scale_factor = orig_image_width/518
        if use_optical_flow_downsample:
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(f"processing image subset:\n{image_names_subset}\n")

            predictions = solver.run_predictions(image_names_subset, model, args.max_loops)

            intrinsics.extend(predictions["intrinsic"].tolist())

            data.append(predictions["intrinsic"][:,0,0])

            solver.add_points(predictions)

            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Detect AQR for each image
            for i, image_name in enumerate(image_names_subset):
                if adjustment_matrix is not None:
                    break
                T_map_image = aqr_calibration(domain, marker_calibrator, image_name, 
                                              predictions["intrinsic"][i])
                if T_map_image is not None:
                    print(f"{image_name}\naqr calibration pose: {T_map_image}\n extrinsics: {predictions["extrinsic"][i]}")
                    predicted_pose = predictions["extrinsic"][i]
                    T_vggt_image = np.eye(4)  # start with identity
                    T_vggt_image[0:3, 0:3] = predicted_pose[0:3, 0:3]
                    T_vggt_image[0:3, 3] = predicted_pose[0:3, 3] * scale_factor

                    # Calculate difference and apply adjustment matrix
                    adjustment_matrix = T_map_image @ np.linalg.inv(T_vggt_image)
                    ref_image = image_name

            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]


    # Find adjusted pose at the end
    for i, submap in enumerate(solver.map.ordered_submaps_by_key()):
        poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
        frame_ids = submap.get_frame_ids()


    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())
    print(f"reference image: {ref_image}")
    print(f"Original matrix pose: \n{T_vggt_image}")
    print(f"Adjusted ref image pose: \n{T_map_image}")
    print(f"Adjustment Matrix: \n{adjustment_matrix}")

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        # VGGT Pose File
        solver.map.write_poses_to_file(args.log_path)

        # VGGT Pose File in ACE0 Format
        
        print(f"Scale factor is {scale_factor}")
        print(f"number of images: {len(image_names)}")
        solver.map.write_poses_to_acezero_pose_file(args.log_path.replace(".txt", "_ace0fmt.txt"), image_names, data, scale_factor)

        # Log the full point cloud as one file, used for visualization.
        # solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))

        # Save adjusted pose
        f_domain_coords = open(args.log_path.replace(".txt", "_domain_scaled.txt"), "w")
        f_domain_coords_ace = open(args.log_path.replace(".txt", "_domain_scaled_ace0fmt.txt"), "w")

        for i, submap in enumerate(solver.map.ordered_submaps_by_key()):
            poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
            frame_ids = submap.get_frame_ids()
            focal_length_list = data[i]
            inner_counter = 0
            assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
            for frame_id, pose in zip(frame_ids, poses):

                # Get Image Info
                rgb_file = image_names[int(frame_id)]
                focal_length = focal_length_list[inner_counter]*scale_factor
                inner_counter += 1

                # Calculate Adjusted Poses
                T_vggt_pose = np.eye(4)  # start with identity
                T_vggt_pose[0:3, 0:3] = pose[0:3, 0:3]
                T_vggt_pose[0:3, 3] = pose[0:3, 3] * scale_factor
                T_domain_pose = adjustment_matrix @ T_vggt_pose

                if rgb_file == ref_image:
                    print(f"image at vggt world pose: \n{T_vggt_pose}")
                    print(f"image at domain world pose: \n{T_domain_pose}")
                    print(f"image at domain world pose inv: \n{np.linalg.inv(T_domain_pose)}")

                x, y, z = T_domain_pose[0:3, 3]
                rotation_matrix = T_domain_pose[0:3, 0:3]
                quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                
                output = np.array([float(frame_id), x, y, z, *quaternion])
                f_domain_coords.write(" ".join(f"{v:.8f}" for v in output) + "\n")

                pose_str = f"{rgb_file} " \
                    f"{quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} " \
                    f"{x} {y} {z} {focal_length} {2000}\n"
                f_domain_coords_ace.write(pose_str)


        f_domain_coords.close()
        f_domain_coords_ace.close()


    if args.plot_focal_lengths:
        # Define a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values  # Y-values from the list
            x = [i] * len(values)  # X-values (same for all points in the list)
            plt.scatter(x, y, color=colors[i], label=f'List {i+1}')

        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.savefig(args.log_path.replace(".txt", ".png"))
        plt.close()

    # for i, intrinsic in enumerate(intrinsics):
    #     print(f"id: {i} \t intrinsics: \t{np.array(intrinsic).shape}")

    # # if args.save_intrinsics:
    # #     np.savetxt(args.log_path.replace(".txt", "_intrinsics.txt"), np.array(intrinsics), fmt="%d")

if __name__ == "__main__":
    main()
