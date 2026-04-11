"""
camera_config.py
----------------
Camera intrinsic parameters for each supported sensor.

depth_scale : raw depth value -> meters  (z_meters = pixel_value / depth_scale)
depth_trunc : depth values beyond this distance (meters) are discarded when
              building the RGBD point cloud in preprocess_pcd.py
"""

CAMERAS = {
    'realsense_d415': {
        'fx': 597.522, 'fy': 597.522,
        'cx': 312.885, 'cy': 239.870,
        'depth_scale': 1000,
        'depth_trunc': 1.5,
    },
    # TUM RGB-D benchmark, freiburg1 sequence
    'tum_fr1': {
        'fx': 517.3, 'fy': 516.5,
        'cx': 318.6, 'cy': 255.3,
        'depth_scale': 5000,
        'depth_trunc': 4.0,
    },
    # Redwood indoor LIDAR scan dataset (SUN3D-style)
    'redwood': {
        'fx': 525.0, 'fy': 525.0,
        'cx': 319.5, 'cy': 239.5,
        'depth_scale': 1000,
        'depth_trunc': 4.0,
    },
}
