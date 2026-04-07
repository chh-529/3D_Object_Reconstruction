import argparse
import os
import re
import numpy as np
import open3d as o3d
from SIFT import SIFT_Transformation
from registration import draw_registration_result
from glob import glob
import matplotlib.pyplot as plt


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

def pairwise_registration(source, target, init_trans):

    source.estimate_normals()
    target.estimate_normals()

    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds) # 16
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print('source id:', source_id)
            print('target id:', target_id)

            init_trans = np.identity(4)
            transformation_icp, information_icp = pairwise_registration(pcds_down[source_id], pcds_down[target_id],
                                                                        init_trans)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case

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
            else:  # loop closure case -> connect any non-neighboring nodes
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    # Loop closure
    init_trans = np.identity(4)
    transformation_icp, information_icp = pairwise_registration(pcds_down[n_pcds-1], pcds_down[0],
                                                                init_trans)
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(n_pcds-1,
                                                 0,
                                                 transformation_icp,
                                                 information_icp,
                                                 uncertain=False))
    return pose_graph

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='3D Object Reconstruction using ICP')
    parser.add_argument('--object', type=str, required=True,
                        help='Object name (must match folder names under train/ and pcd_o3d/)')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames to use (default: auto-detect from pcd_o3d/<object>/)')
    parser.add_argument('--voxel_size', type=float, default=0.001,
                        help='Voxel size for downsampling (default: 0.001)')
    args = parser.parse_args()

    object_name = args.object
    out_dir = f'./results/{object_name}'
    os.makedirs(out_dir, exist_ok=True)

    pcds_paths = sorted(glob(f'./pcd_o3d/{object_name}/*.pcd'), key=natural_sort_key)
    if not pcds_paths:
        raise FileNotFoundError(f'No .pcd files found in ./pcd_o3d/{object_name}/')

    if args.n_frames is not None:
        pcds_paths = pcds_paths[:args.n_frames]

    print(f'Object: {object_name}, frames: {len(pcds_paths)}')

    # Define voxel size to Downsample
    voxel_size = args.voxel_size
    origin_pcds = load_orginal_point_clouds(voxel_size, pcds_paths)
    pcds_down = load_point_clouds(voxel_size, pcds_paths)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)

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

    out_path = f'{out_dir}/accumulated_icp.pcd'
    o3d.io.write_point_cloud(out_path, accumulated_pcd)
    print(f'Saved accumulated point cloud to {out_path}')