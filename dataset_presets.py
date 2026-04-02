import numpy as np
import open3d as o3d


DATASET_CONFIGS = {
    "redwood_stool": {
        "dataset_type": "folder_pairs",
        "dataset_path": "dataset/Redwood/stool",
        "rgb_glob": "rgb/*.jpg",
        "depth_glob": "depth/*.png",
        "output_dir": "pcd_o3d/redwood_stool",
        "intrinsic": {
            "width": 640,
            "height": 480,
            "fx": 525.0,
            "fy": 525.0,
            "cx": 319.5,
            "cy": 239.5,
        },
        "rgbd": {
            "depth_scale": 1000.0,
            "depth_trunc": 2.0,
        },
        "preprocess": {
            "voxel_size": 0.01,
            "plane_distance_threshold": 0.015,
            "dbscan_eps": 0.04,
            "dbscan_min_points": 15,
            "sor_nb_neighbors": 20,
            "sor_std_ratio": 2.0,
        },
        "sample_every": 2,
    },
    "tum_fr1_desk": {
        "dataset_type": "association_file",
        "dataset_path": "dataset/tum_fr1_desk",
        "association_file": "associated.txt",
        "output_dir": "pcd_o3d/tum_fr1_desk",
        "intrinsic": {
            "width": 640,
            "height": 480,
            "fx": 517.3,
            "fy": 516.5,
            "cx": 318.6,
            "cy": 255.3,
        },
        "rgbd": {
            "depth_scale": 5000.0,
            "depth_trunc": 3.0,
        },
        "preprocess": {
            "voxel_size": 0.01,
            "plane_distance_threshold": 0.02,
            "dbscan_eps": 0.04,
            "dbscan_min_points": 20,
            "sor_nb_neighbors": 30,
            "sor_std_ratio": 2.0,
        },
        "sample_every": 2,
    },
}


def build_intrinsic(width, height, fx=None, fy=None, cx=None, cy=None, k_matrix=None):
    """Create an Open3D intrinsic from either fx/fy/cx/cy or a 3x3 K matrix."""
    intrinsic = o3d.camera.PinholeCameraIntrinsic()

    if k_matrix is not None:
        k_matrix = np.asarray(k_matrix, dtype=np.float64)
        if k_matrix.shape != (3, 3):
            raise ValueError("k_matrix must be shape (3, 3)")
        intrinsic.intrinsic_matrix = k_matrix
        intrinsic.width = int(width)
        intrinsic.height = int(height)
        return intrinsic

    required = [fx, fy, cx, cy]
    if any(value is None for value in required):
        raise ValueError("Provide either k_matrix or fx/fy/cx/cy")

    intrinsic.set_intrinsics(
        int(width),
        int(height),
        float(fx),
        float(fy),
        float(cx),
        float(cy),
    )
    return intrinsic


def build_intrinsic_from_config(intrinsic_config):
    return build_intrinsic(
        width=intrinsic_config["width"],
        height=intrinsic_config["height"],
        fx=intrinsic_config.get("fx"),
        fy=intrinsic_config.get("fy"),
        cx=intrinsic_config.get("cx"),
        cy=intrinsic_config.get("cy"),
        k_matrix=intrinsic_config.get("k_matrix"),
    )


def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(sorted(DATASET_CONFIGS.keys()))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    return DATASET_CONFIGS[dataset_name]