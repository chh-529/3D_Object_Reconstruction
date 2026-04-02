import argparse
import os
from glob import glob

import numpy as np
import open3d as o3d
from dataset_presets import DATASET_CONFIGS, build_intrinsic_from_config, get_dataset_config


def rgbd_to_pcd(
    color_path,
    depth_path,
    intrinsic,
    depth_scale=1000.0,
    depth_trunc=2.0,
    convert_rgb_to_intensity=False,
):
    """Convert one RGB-D pair to a point cloud."""
    if not os.path.exists(color_path):
        raise FileNotFoundError(f"Color image not found: {color_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    source_color = o3d.io.read_image(color_path)
    source_depth = o3d.io.read_image(depth_path)

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color,
        source_depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)


def preprocess_pointcloud(
    pcd,
    voxel_size=0.01,
    plane_distance_threshold=0.015,
    plane_ransac_n=3,
    plane_num_iterations=1000,
    dbscan_eps=0.04,
    dbscan_min_points=15,
    y_max_threshold=None,
    sor_nb_neighbors=20,
    sor_std_ratio=2.0,
):
    """Reusable object-focused preprocessing pipeline for a single point cloud."""
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    _, inliers = pcd_down.segment_plane(
        distance_threshold=plane_distance_threshold,
        ransac_n=plane_ransac_n,
        num_iterations=plane_num_iterations,
    )
    pcd_object = pcd_down.select_by_index(inliers, invert=True)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.array(
            pcd_object.cluster_dbscan(
                eps=dbscan_eps,
                min_points=dbscan_min_points,
                print_progress=False,
            )
        )

    valid_labels = labels[labels >= 0]
    if valid_labels.size > 0:
        largest_cluster_idx = np.argmax(np.bincount(valid_labels))
        target_indices = np.where(labels == largest_cluster_idx)[0]
        pcd_clean = pcd_object.select_by_index(target_indices)
    else:
        pcd_clean = pcd_object

    if y_max_threshold is not None and len(pcd_clean.points) > 0:
        y = np.asarray(pcd_clean.points)[:, 1]
        keep_idx = np.where(y < y_max_threshold)[0]
        pcd_clean = pcd_clean.select_by_index(keep_idx.tolist())

    _, ind = pcd_clean.remove_statistical_outlier(
        nb_neighbors=sor_nb_neighbors,
        std_ratio=sor_std_ratio,
    )
    return pcd_clean.select_by_index(ind)


def process_rgbd_pair(
    color_path,
    depth_path,
    output_path,
    intrinsic,
    depth_scale=1000.0,
    depth_trunc=2.0,
    preprocess_kwargs=None,
):
    """One-stop helper: RGB-D pair -> point cloud -> preprocessing -> save."""
    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    pcd = rgbd_to_pcd(
        color_path=color_path,
        depth_path=depth_path,
        intrinsic=intrinsic,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    pcd_final = preprocess_pointcloud(pcd, **preprocess_kwargs)
    o3d.io.write_point_cloud(output_path, pcd_final)
    return pcd_final


def iter_rgbd_pairs(dataset_config):
    dataset_path = dataset_config["dataset_path"]
    dataset_type = dataset_config["dataset_type"]

    if dataset_type == "folder_pairs":
        rgb_files = sorted(glob(os.path.join(dataset_path, dataset_config["rgb_glob"])))
        depth_files = sorted(glob(os.path.join(dataset_path, dataset_config["depth_glob"])))

        if len(rgb_files) != len(depth_files):
            print("⚠️ Warning: RGB and Depth file counts do not match!")

        for index, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
            yield index, rgb_path, depth_path

    elif dataset_type == "association_file":
        association_file = os.path.join(dataset_path, dataset_config["association_file"])
        frame_index = 0
        with open(association_file, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    continue

                data = line.strip().split()
                if len(data) != 4:
                    continue

                rgb_path = os.path.join(dataset_path, data[1])
                depth_path = os.path.join(dataset_path, data[3])
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    continue

                yield frame_index, rgb_path, depth_path
                frame_index += 1

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def clear_existing_pcds(output_dir):
    existing_pcds = sorted(glob(os.path.join(output_dir, "*.pcd")))
    if not existing_pcds:
        return 0

    for pcd_path in existing_pcds:
        os.remove(pcd_path)
    return len(existing_pcds)


def run_dataset_preprocess(dataset_name, sample_every=None, limit=None, clean_output=True):
    dataset_config = get_dataset_config(dataset_name)
    output_dir = dataset_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if clean_output:
        removed_count = clear_existing_pcds(output_dir)
        print(f"Cleaned {removed_count} existing .pcd files from '{output_dir}'.")

    intrinsic = build_intrinsic_from_config(dataset_config["intrinsic"])
    rgbd_config = dataset_config.get("rgbd", {})
    preprocess_config = dataset_config.get("preprocess", {})

    step = sample_every if sample_every is not None else dataset_config.get("sample_every", 1)
    if step < 1:
        raise ValueError("sample_every must be >= 1")
    processed_count = 0

    for index, rgb_path, depth_path in iter_rgbd_pairs(dataset_config):
        if step > 1 and index % step != 0:
            continue

        output_name = f"cloud_bin_{processed_count:04d}.pcd"
        output_path = os.path.join(output_dir, output_name)

        print(f"Processing {processed_count}: {os.path.basename(rgb_path)}")
        process_rgbd_pair(
            color_path=rgb_path,
            depth_path=depth_path,
            output_path=output_path,
            intrinsic=intrinsic,
            depth_scale=rgbd_config.get("depth_scale", 1000.0),
            depth_trunc=rgbd_config.get("depth_trunc", 2.0),
            preprocess_kwargs=preprocess_config,
        )

        processed_count += 1
        if limit is not None and processed_count >= limit:
            break

    print(f"Preprocessing complete! Processed {processed_count} point clouds saved to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess RGB-D datasets into object point clouds")
    parser.add_argument(
        "--dataset",
        default="redwood_stool",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset preset to run",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="Process every N-th frame instead of the preset value",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after processing this many point clouds",
    )
    parser.add_argument(
        "--clean-output",
        dest="clean_output",
        action="store_true",
        help="Delete existing .pcd files in output_dir before preprocessing (default: enabled)",
    )
    parser.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Keep existing .pcd files in output_dir",
    )
    parser.set_defaults(clean_output=True)

    args = parser.parse_args()
    run_dataset_preprocess(
        args.dataset,
        sample_every=args.sample_every,
        limit=args.limit,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()