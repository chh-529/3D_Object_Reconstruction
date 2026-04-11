import argparse
import os
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from camera_config import CAMERAS

def rgbd_to_pcd(count, object_name, cam, args):

    source_color = o3d.io.read_image(f'./train/{object_name}/rgb/align_test{count}.png')
    source_depth = o3d.io.read_image(f'./train/{object_name}/depth/align_test_depth{count}.png')

    K = np.array(
         [[cam['fx'], 0.0,        cam['cx']],
          [0.0,       cam['fy'],  cam['cy']],
          [0.0,       0.0,        1.0      ]], dtype=np.float64)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = K

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth,
        depth_scale=cam['depth_scale'],
        convert_rgb_to_intensity=False,
        depth_trunc=cam['depth_trunc'])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)

    # Plane Segmentation (optional)
    if not args.no_plane_removal:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
    else:
        print("Plane removal skipped.")
        outlier_cloud = pcd

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            outlier_cloud.cluster_dbscan(eps=args.dbscan_eps,
                               min_points=args.dbscan_min_pts,
                               print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    if args.all_clusters or max_label < 0:
        # Keep all valid clusters (exclude noise label -1)
        indexes = np.where(labels >= 0)
    else:
        # Keep only the largest cluster (label 0)
        indexes = np.where(labels == 0)

    # Extract Interest point clouds
    interest_pcd = o3d.geometry.PointCloud()
    interest_pcd.points = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.points, np.float32)[indexes])
    interest_pcd.colors = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.colors, np.float32)[indexes])

    # Plane Segmentation for floor
    # plane_model, inliers = interest_pcd.segment_plane(distance_threshold=0.001,
    #                                          ransac_n=3,
    #                                          num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # inlier_cloud = interest_pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = interest_pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # Optional Y-axis height filter (camera-space, Y is downward in Open3D)
    # Useful for horizontal circular scans; disable for spiral/scene data.
    if args.y_max is not None:
        y = np.asarray(outlier_cloud.points)[:, 1]
        idx = np.where(y < args.y_max)[0]
        idx = np.asarray(idx, dtype=int)
        interest_pcd = outlier_cloud.select_by_index(list(idx))
        print(f"Y-axis filter (y < {args.y_max}): kept {len(idx)} / {len(y)} points")
    else:
        interest_pcd = interest_pcd  # already set by DBSCAN above

    print("Statistical outlier removal")
    cl, ind = interest_pcd.remove_statistical_outlier(nb_neighbors=args.stat_neighbors, std_ratio=args.stat_ratio)

    print("Radius outlier removal")
    cl, ind = cl.remove_radius_outlier(nb_points=args.radius_pts, radius=args.radius)

    out_dir = f'./pcd_o3d/{object_name}'
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(f'{out_dir}/{object_name}{count}.pcd', cl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RGBD images to preprocessed point clouds')
    parser.add_argument('--object', type=str, required=True,
                        help='Object name (folder under train/ and pcd_o3d/)')
    parser.add_argument('--camera', type=str, default='realsense_d415',
                        choices=list(CAMERAS.keys()),
                        help='Camera intrinsics preset (default: realsense_d415)')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames to process (default: auto-detect)')

    # --- Plane removal ---
    parser.add_argument('--no_plane_removal', action='store_true',
                        help='Skip plane segmentation. Use for scene-level data (e.g. desk).')

    # --- DBSCAN clustering ---
    parser.add_argument('--dbscan_eps', type=float, default=0.02,
                        help='DBSCAN epsilon (neighbor distance, default: 0.02 m)')
    parser.add_argument('--dbscan_min_pts', type=int, default=500,
                        help='DBSCAN min_points per cluster (default: 500)')
    parser.add_argument('--all_clusters', action='store_true',
                        help='Keep all DBSCAN clusters (not just the largest). '
                             'Use for scene-level or spiral-scan data.')

    # --- Y-axis height filter ---
    parser.add_argument('--y_max', type=float, default=None,
                        help='Keep only points with camera-space Y < Y_MAX (meters). '
                             'Default: 0.18 for original circular-scan objects, '
                             'disabled (None) for spiral/scene data. '
                             'Set to 0.18 explicitly to reproduce original behaviour.')

    # --- Outlier removal ---
    parser.add_argument('--stat_neighbors', type=int, default=200,
                        help='Statistical outlier removal: nb_neighbors (default: 200)')
    parser.add_argument('--stat_ratio', type=float, default=2.0,
                        help='Statistical outlier removal: std_ratio (default: 2.0)')
    parser.add_argument('--radius_pts', type=int, default=20,
                        help='Radius outlier removal: min nb_points (default: 20)')
    parser.add_argument('--radius', type=float, default=0.05,
                        help='Radius outlier removal: search radius in meters (default: 0.05)')

    args = parser.parse_args()

    cam = CAMERAS[args.camera]
    object_name = args.object

    from glob import glob
    import re
    def natural_sort_key(s):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(s))]

    rgb_files = sorted(glob(f'./train/{object_name}/rgb/align_test*.png'), key=natural_sort_key)
    n = len(rgb_files) if args.n_frames is None else min(len(rgb_files), args.n_frames)
    if n == 0:
        raise FileNotFoundError(f'No align_test*.png found in ./train/{object_name}/rgb/. '
                                 f'Run prepare_dataset.py first if using TUM/Redwood data.')

    print(f'Object: {object_name}, camera: {args.camera}, frames: {n}')
    for i in range(1, n + 1):
        print(f'Processing frame {i}/{n}...')
        rgbd_to_pcd(i, object_name, cam, args)