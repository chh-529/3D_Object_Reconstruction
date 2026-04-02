import os
from glob import glob
import open3d as o3d
from dataset_presets import DATASET_CONFIGS, build_intrinsic_from_config, get_dataset_config


def load_pointclouds_from_pairs(
    dataset_path,
    intrinsic,
    rgb_glob="rgb/*.jpg",
    depth_glob="depth/*.png",
    depth_scale=1000.0,
    depth_trunc=2.0,
):
    pcds = []

    rgb_files = sorted(glob(os.path.join(dataset_path, rgb_glob)))
    depth_files = sorted(glob(os.path.join(dataset_path, depth_glob)))

    if len(rgb_files) != len(depth_files):
        print("⚠️ Warning: RGB and Depth file counts do not match!")
    else:
        print(f"Start transforming {len(rgb_files)} RGB-D pairs into point clouds...")

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        color_raw = o3d.io.read_image(rgb_path)
        depth_raw = o3d.io.read_image(depth_path)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_raw,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcds.append(pcd)

    return pcds


def load_pointclouds_from_association(
    dataset_path,
    association_file,
    intrinsic,
    depth_scale=5000.0,
    depth_trunc=3.0,
):
    pcds = []

    association_path = association_file
    if not os.path.isabs(association_path):
        association_path = os.path.join(dataset_path, association_file)

    with open(association_path, "r", encoding="utf-8") as file_handle:
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

            color_raw = o3d.io.read_image(rgb_path)
            depth_raw = o3d.io.read_image(depth_path)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw,
                depth_raw,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            pcds.append(pcd)

    return pcds


def load_dataset_pointclouds(dataset_name, sample_every=None, limit=None):
    dataset_config = get_dataset_config(dataset_name)
    intrinsic = build_intrinsic_from_config(dataset_config["intrinsic"])
    rgbd_config = dataset_config.get("rgbd", {})

    if dataset_config["dataset_type"] == "folder_pairs":
        pcds = load_pointclouds_from_pairs(
            dataset_path=dataset_config["dataset_path"],
            intrinsic=intrinsic,
            rgb_glob=dataset_config.get("rgb_glob", "rgb/*.jpg"),
            depth_glob=dataset_config.get("depth_glob", "depth/*.png"),
            depth_scale=rgbd_config.get("depth_scale", 1000.0),
            depth_trunc=rgbd_config.get("depth_trunc", 2.0),
        )
    elif dataset_config["dataset_type"] == "association_file":
        pcds = load_pointclouds_from_association(
            dataset_path=dataset_config["dataset_path"],
            association_file=dataset_config["association_file"],
            intrinsic=intrinsic,
            depth_scale=rgbd_config.get("depth_scale", 5000.0),
            depth_trunc=rgbd_config.get("depth_trunc", 3.0),
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_config['dataset_type']}")

    step = sample_every if sample_every is not None else dataset_config.get("sample_every", 1)
    if step < 1:
        raise ValueError("sample_every must be >= 1")
    if step > 1:
        pcds = pcds[::step]

    if limit is not None:
        pcds = pcds[:limit]

    return pcds


def load_tum_pointclouds(dataset_path, association_file):
    intrinsic = build_intrinsic_from_config(DATASET_CONFIGS["tum_fr1_desk"]["intrinsic"])
    rgbd_config = DATASET_CONFIGS["tum_fr1_desk"]["rgbd"]
    return load_pointclouds_from_association(
        dataset_path=dataset_path,
        association_file=association_file,
        intrinsic=intrinsic,
        depth_scale=rgbd_config.get("depth_scale", 5000.0),
        depth_trunc=rgbd_config.get("depth_trunc", 3.0),
    )


def load_redwood_pointclouds(dataset_path):
    intrinsic = build_intrinsic_from_config(DATASET_CONFIGS["redwood_stool"]["intrinsic"])
    rgbd_config = DATASET_CONFIGS["redwood_stool"]["rgbd"]
    return load_pointclouds_from_pairs(
        dataset_path=dataset_path,
        intrinsic=intrinsic,
        rgb_glob=DATASET_CONFIGS["redwood_stool"]["rgb_glob"],
        depth_glob=DATASET_CONFIGS["redwood_stool"]["depth_glob"],
        depth_scale=rgbd_config.get("depth_scale", 1000.0),
        depth_trunc=rgbd_config.get("depth_trunc", 2.0),
    )