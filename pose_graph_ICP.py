import argparse
import os
from glob import glob

import numpy as np
import open3d as o3d

from dataset_presets import DATASET_CONFIGS, get_dataset_config


def load_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print("len demo_icp_pcds_paths:", len(pcds_paths))
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size) if voxel_size > 0 else pcd
        pcds.append(pcd_down)
    return pcds


def load_original_point_clouds(pcds_paths=None):
    pcds = []
    print("len demo_icp_pcds_paths:", len(pcds_paths))
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcds.append(pcd)
    return pcds


def pairwise_registration(source, target, init_trans, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    source.estimate_normals()
    target.estimate_normals()

    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        max_correspondence_distance_fine,
        icp_fine.transformation,
    )
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    print("Starting full registration with pairwise ICP ...")

    for source_id in range(len(pcds) - 1):
        target_id = source_id + 1
        print(f"Aligning: Frame {source_id} -> Frame {target_id}")

        init_trans = np.identity(4)
        transformation_icp, information_icp = pairwise_registration(
            pcds[source_id],
            pcds[target_id],
            init_trans,
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine,
        )

        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id,
                target_id,
                transformation_icp,
                information_icp,
                uncertain=False,
            )
        )

    return pose_graph


def resolve_pcd_paths(dataset_name, source_dir=None, pattern="cloud_bin_*.pcd"):
    dataset_config = get_dataset_config(dataset_name)
    pcd_dir = source_dir if source_dir is not None else dataset_config["output_dir"]
    pcd_files = glob(os.path.join(pcd_dir, pattern))
    pcd_files.sort(key=lambda path: int(os.path.basename(path).split("_")[-1].split(".")[0]))
    return pcd_files, pcd_dir


def run_pose_graph_icp(dataset_name, source_dir=None, voxel_size=0.02, output_path=None):
    pcd_files, pcd_dir = resolve_pcd_paths(dataset_name, source_dir=source_dir)
    print(f"Found {len(pcd_files)} point cloud files ready for registration.")

    origin_pcds = load_original_point_clouds(pcd_files)
    pcds_down = load_point_clouds(voxel_size, pcd_files)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 10
    max_correspondence_distance_fine = voxel_size * 2
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        pose_graph = full_registration(
            pcds_down,
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine,
        )

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

    print("Transform points and display")
    accumulated_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(origin_pcds)):
        pcd_temp = origin_pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        accumulated_pcd += pcd_temp

    if output_path is None:
        output_path = os.path.join("results", dataset_name, "reconstructed_result.ply")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(pcd_files)} point cloud files ready for registration.")
    o3d.io.write_point_cloud(output_path, accumulated_pcd)
    print(f"Successfully saved! File name is {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ICP-based pose graph optimization on preprocessed point clouds")
    parser.add_argument(
        "--dataset",
        default="redwood_stool",
        choices=sorted(DATASET_CONFIGS.keys()),
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Override the preset point-cloud directory",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling before ICP",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ply path. Defaults to <preset output_dir>/reconstructed_result.ply",
    )

    args = parser.parse_args()
    run_pose_graph_icp(
        dataset_name=args.dataset,
        source_dir=args.source_dir,
        voxel_size=args.voxel_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()