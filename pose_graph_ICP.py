import numpy as np
import open3d as o3d
from SIFT import SIFT_Transformation
from registration import draw_registration_result
from glob import glob
import matplotlib.pyplot as plt

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
    n_pcds = len(pcds)
    
    print("開始建立 Pose Graph (純里程計模式)...")
    
    # 🚨 關鍵修改：移除雙重迴圈，只讓 source_id 跟 source_id + 1 (相鄰幀) 做 ICP
    for source_id in range(n_pcds - 1):
        target_id = source_id + 1
        print(f'正在對齊: Frame {source_id} -> Frame {target_id}')

        init_trans = np.identity(4)
        transformation_icp, information_icp = pairwise_registration(
            pcds[source_id], pcds[target_id], init_trans)
            
        # 累積位姿 (Odometry accumulation)
        odometry = np.dot(transformation_icp, odometry)

        # 加入 Node (節點：相機當下的絕對位置)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)))
                
        # 加入 Edge (邊：相機移動的相對位置)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id, target_id, transformation_icp,
                information_icp, uncertain=False))
                
    return pose_graph

# def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
#     pose_graph = o3d.pipelines.registration.PoseGraph()
#     odometry = np.identity(4)
#     pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
#     n_pcds = len(pcds) # 16
#     for source_id in range(n_pcds):
#         for target_id in range(source_id + 1, n_pcds):
#             print('source id:', source_id)
#             print('target id:', target_id)

#             init_trans = np.identity(4)
#             transformation_icp, information_icp = pairwise_registration(pcds_down[source_id], pcds_down[target_id],
#                                                                         init_trans)
#             print("Build o3d.pipelines.registration.PoseGraph")
#             if target_id == source_id + 1:  # odometry case

#                 odometry = np.dot((transformation_icp), odometry)

#                 pose_graph.nodes.append(
#                     o3d.pipelines.registration.PoseGraphNode(
#                         np.linalg.inv(odometry)))
#                 pose_graph.edges.append(
#                     o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                              target_id,
#                                                              transformation_icp,
#                                                              information_icp,
#                                                              uncertain=False))
#             else:  # loop closure case -> connect any non-neighboring nodes
#                 pose_graph.edges.append(
#                     o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                              target_id,
#                                                              transformation_icp,
#                                                              information_icp,
#                                                              uncertain=True))
#     # Loop closure
#     # init_trans = np.identity(4)
#     # transformation_icp, information_icp = pairwise_registration(pcds_down[n_pcds-1], pcds_down[0],
#     #                                                             init_trans)
#     # pose_graph.edges.append(
#     #     o3d.pipelines.registration.PoseGraphEdge(n_pcds-1,
#     #                                              0,
#     #                                              transformation_icp,
#     #                                              information_icp,
#     #                                              uncertain=False))
#     return pose_graph

if __name__ == "__main__":

    # object = "spyderman2"

    # if object == "castard":
    #     depth_path = ['./train/castard/depth/align_test_depth%d.png' % i for i in range(1, 21)]
    #     rgb_path = ['./train/castard/rgb/align_test%d.png' % i for i in range(1, 21)]
    #     pcds_paths = ['./pcd_o3d/castard/box%d.pcd' % i for i in range(1, 19)]
    # elif object == "new_box":
    #     depth_path = ['./train/new_box2/depth/align_test_depth%d.png' % i for i in range(1, 19)]
    #     rgb_path = ['./train/new_box2/rgb/align_test%d.png' % i for i in range(1, 19)]
    #     pcds_paths = ['./pcd_o3d/new_box2/box1%d.pcd' % i for i in range(1, 19)]
    # elif object == "spyderman":
    #     depth_path = ['./train/spyderman/depth/align_test_depth%d.png' % i for i in range(1, 17)]
    #     rgb_path = ['./train/spyderman/rgb/align_test%d.png' % i for i in range(1, 17)]
    #     pcds_paths = ['./pcd_o3d/spyderman/spyderman%d.pcd' % i for i in range(1, 17)]
    # elif object == "spyderman2":
    #     depth_path = ['./train/spyderman/depth/align_test_depth%d.png' % i for i in range(1, 23)]
    #     rgb_path = ['./train/spyderman/rgb/align_test%d.png' % i for i in range(1, 23)]
    #     pcds_paths = ['./pcd_o3d/spyderman2/spyderman2%d.pcd' % i for i in range(1, 23)]

    # 1. 改用 glob 自動讀取 pcd_o3d/ 裡面所有的 .pcd 檔案
    # 注意：我們需要確保檔案是照順序排列的 (cloud_bin_0.pcd, cloud_bin_1.pcd ...)
    import os
    from glob import glob
    
    pcd_files = glob('./pcd_o3d/cloud_bin_*.pcd')
    # 確保照數字順序排序，否則 Pose Graph 的相鄰關係會全錯！
    pcd_files.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
    
    pcds_paths = pcd_files
    print(f"共找到 {len(pcds_paths)} 個點雲檔案準備進行配準。")

    # Define voxel size to Downsample
    # voxel_size = 0.001
    voxel_size = 0.02
    origin_pcds = load_orginal_point_clouds(voxel_size, pcds_paths)
    pcds_down = load_point_clouds(voxel_size, pcds_paths)
    # o3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    # max_correspondence_distance_coarse = voxel_size * 15
    # max_correspondence_distance_fine = voxel_size * 1
    max_correspondence_distance_coarse = voxel_size * 10
    max_correspondence_distance_fine = voxel_size * 2
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

    print("Transform points and display")
    accumulated_pcd = o3d.geometry.PointCloud()
    # for point_id in range(len(pcds_down)):
    #     print(pose_graph.nodes[point_id].pose)
    #     accumulated_pcd += pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    for point_id in range(len(origin_pcds)):
        # 複製一份原始點雲，避免改到原本的資料
        pcd_temp = origin_pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        accumulated_pcd += pcd_temp

    # o3d.visualization.draw_geometries([accumulated_pcd])
    # o3d.io.write_point_cloud('accumulated_%s.pcd'%object, accumulated_pcd)

    # y = np.asarray(accumulated_pcd.points)[:, 1]
    # y_mean = np.mean(y)
    # plt.plot(y)
    # plt.show()
    # # idx = np.array([i for i in range(len(z))], dtype=np.int)
    # idx = np.where(y < 0.138)[0]
    # idx = np.asarray(idx, dtype=np.int)
    # interest_pcd = accumulated_pcd.select_by_index(list(idx))
    # o3d.visualization.draw_geometries([interest_pcd])

    # Render
    # vis = o3d.visualization.Visualizer()
    # vis.create_window('3DReconstructed')

    # for p in pcds_down:
    #     vis.add_geometry(p)

    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # vis.add_geometry(axis)

    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([1, 1, 1])
    # opt.point_size = 1.5

    # vis.run()
    # vis.destroy_window()

    # 假設最終合併好的點雲變數叫做 pcd_combined
    # 將點雲存成 .ply 格式 (這是在 3D 領域最通用且支援顏色的格式)
    print("正在儲存重建結果...")
    o3d.io.write_point_cloud("reconstructed_result.ply", accumulated_pcd)
    print("儲存成功！檔案名稱為 reconstructed_result.ply")