import argparse
import os
import re
import numpy as np
import open3d as o3d
from SIFT import SIFT_Transformation # SIFT feature points based registration
from ORB import ORB_Transformation # ORB feature points based registration
from LoFTR import LoFTR_Transformation # LoFTR method based registration
from glob import glob
import matplotlib.pyplot as plt
from camera_config import CAMERAS


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(s))]

# Load point clouds
def load_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print('len demo_icp_pcds_paths:', len(pcds_paths))
    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

def load_orginal_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print('len demo_icp_pcds_paths:', len(pcds_paths))
    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcds.append(pcd)
    return pcds

def relative_camera_poses_all(rgb_lists, depth_lists, pcd_lists):

    pose_list = []
    num_multiview = len(rgb_lists)

    for i in range(num_multiview-1):
        j = i + 1
        transformation, pcd1_features, source_pcd1_features, pts1, pts_source_1, pts1_3d, pts_source1_3d = SIFT_Transformation(
            rgb_lists[i], rgb_lists[j],
            depth_lists[i], depth_lists[j],
            pcd_lists[i], pcd_lists[j],
            distance_ratio=0.7)
        pose_list.append(transformation)

    return np.asarray(pose_list)

def relative_camera_poses_select(start_idx, end_idx, pose_list):

    result = np.identity(4)
    for idx in range(start_idx, end_idx):
        result = pose_list[idx] @ result
    return result

def pairwise_registration(source, target, init_trans):
    print("Apply point-to-plane ICP")
    source.estimate_normals()
    target.estimate_normals()
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine, relative_camera_poses=None):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds) # 16
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print('source id:', source_id)
            print('target id:', target_id)
            threshold = 0.001

            init_trans = np.identity(4)
            transformation_icp, information_icp = pairwise_registration(pcds_down[source_id], pcds_down[target_id],
                                                                        init_trans)

            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case

                sift_trans, pcd1_features, source_pcd1_features, pts1, pts_source_1, pts1_3d, pts_source1_3d = SIFT_Transformation(
                    rgb_path[source_id], rgb_path[target_id],
                    depth_path[source_id], depth_path[target_id],
                    origin_pcds[source_id], origin_pcds[target_id],
                    distance_ratio=0.9, camera=camera)
                if sift_trans is None:
                    print(f'SIFT failed for pair ({source_id}, {target_id}), falling back to identity')
                    init_trans = np.identity(4)
                else:
                    init_trans = np.array(sift_trans)

                icp_fine = o3d.pipelines.registration.registration_icp(
                    origin_pcds[source_id], origin_pcds[target_id], threshold,
                    init_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                transformation_icp = icp_fine.transformation

                information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    origin_pcds[source_id], origin_pcds[target_id], threshold,
                    transformation_icp)

                # visualize transformation icp result
                # draw_registration_result(pcds_down[source_id], pcds_down[target_id], transformation_icp, mode='rgb')

                odometry = np.dot((transformation_icp), odometry)

                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # Connect any non-neighboring nodes

                # transformation_icp = relative_camera_poses_select(start_idx=source_id, end_idx=target_id, pose_list=relative_camera_poses)
                # information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                #     origin_pcds[source_id], origin_pcds[target_id], threshold,
                #     transformation_icp)

                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    # Loop closure
    source_id = n_pcds - 1
    target_id = 0
    sift_trans, pcd1_features, source_pcd1_features, pts1, pts_source_1, pts1_3d, pts_source1_3d = SIFT_Transformation(
        rgb_path[source_id], rgb_path[target_id],
        depth_path[source_id], depth_path[target_id],
        origin_pcds[source_id], origin_pcds[target_id],
        distance_ratio=0.9, camera=camera)
    if sift_trans is None:
        print(f'SIFT failed for loop closure ({source_id} -> {target_id}), falling back to identity')
        loop_init_trans = np.identity(4)
    else:
        loop_init_trans = np.array(sift_trans)

    origin_pcds[source_id].estimate_normals()
    origin_pcds[target_id].estimate_normals()

    threshold = 0.01
    icp_fine = o3d.pipelines.registration.registration_icp(
        origin_pcds[source_id], origin_pcds[target_id], threshold,
        loop_init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        origin_pcds[source_id], origin_pcds[target_id], threshold,
        transformation_icp)

    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(n_pcds-1,
                                                 0,
                                                 transformation_icp,
                                                 information_icp,
                                                 uncertain=True))
    return pose_graph

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='3D Object Reconstruction using Feature-based (SIFT) + ICP')
    parser.add_argument('--object', type=str, required=True,
                        help='Object name (must match folder names under train/ and pcd_o3d/)')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames to use (default: auto-detect)')
    parser.add_argument('--voxel_size', type=float, default=0.001,
                        help='Voxel size for downsampling (default: 0.001)')
    parser.add_argument('--camera', type=str, default='realsense_d415',
                        choices=list(CAMERAS.keys()),
                        help='Camera intrinsics preset (default: realsense_d415)')
    args = parser.parse_args()

    camera = args.camera
    object_name = args.object
    out_dir = f'./results/{object_name}'
    os.makedirs(out_dir, exist_ok=True)

    rgb_path   = sorted(glob(f'./train/{object_name}/rgb/align_test*.png'),          key=natural_sort_key)
    depth_path = sorted(glob(f'./train/{object_name}/depth/align_test_depth*.png'),  key=natural_sort_key)
    pcds_paths = sorted(glob(f'./pcd_o3d/{object_name}/*.pcd'),                       key=natural_sort_key)

    n = min(len(rgb_path), len(depth_path), len(pcds_paths))
    if args.n_frames is not None:
        n = min(n, args.n_frames)

    if n == 0:
        raise FileNotFoundError(f'No data found for object "{object_name}". '
                                 f'Check train/{object_name}/ and pcd_o3d/{object_name}/')

    rgb_path   = rgb_path[:n]
    depth_path = depth_path[:n]
    pcds_paths = pcds_paths[:n]

    print(f'Object: {object_name}, frames: {n}')

    # Define voxel size to Downsample
    voxel_size = args.voxel_size
    origin_pcds = load_orginal_point_clouds(voxel_size, pcds_paths)
    pcds_down = load_point_clouds(voxel_size, pcds_paths)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 10
    max_correspondence_distance_fine = voxel_size * 1
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine,
                                       )

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("Transform points and accumulate")
    accumulated_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        accumulated_pcd += pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

    out_path = f'{out_dir}/accumulated_feature.pcd'
    o3d.io.write_point_cloud(out_path, accumulated_pcd)
    print(f'Saved accumulated point cloud to {out_path}')