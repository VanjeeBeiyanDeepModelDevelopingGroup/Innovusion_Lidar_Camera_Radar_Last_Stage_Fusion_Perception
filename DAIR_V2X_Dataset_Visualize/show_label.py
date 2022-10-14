import os
import math
import numpy as np
import torch
import open3d as o3d
import visualize_utils as V
import time
import json

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(vis, view_json_filename):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(view_json_filename)
    ctr.convert_from_pinhole_camera_parameters(param)


def custom_draw_geometry(vis, pcd, linesets, view_json_filename):
    vis.add_geometry(pcd, reset_bounding_box=False)
    for i in linesets:
        vis.add_geometry(i)
    # 定义窗口参数
    render_option = vis.get_render_option()
    render_option.point_size = 0.5
    render_option.background_color = np.asarray([0, 0, 0])
    load_view_point(vis, view_json_filename)

    vis.run()
    time.sleep(0.1)
    vis.clear_geometries()



def main(pointcloud_path, label_path, view_json_filename):
    pc_bins = os.listdir(pointcloud_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(pc_bins)):
        pc_file = os.path.join(pointcloud_path, '%06d.pcd'%i)
        label_file = os.path.join(label_path, '%06d.json'%i)
        print(pc_file)
        points = o3d.io.read_point_cloud(pc_file)
        points = np.array(points.points)
        linesets = []

        # draw label
        result = []
        with open(label_file, 'r') as f:
            label_dict = json.load(f)
            for object in label_dict:
                cls = object['type']
                x = float(object['3d_location']['x'])
                y = float(object['3d_location']['y'])
                z = float(object['3d_location']['z'])
                l = float(object['3d_dimensions']['l'])
                w = float(object['3d_dimensions']['w'])
                h = float(object['3d_dimensions']['h'])
                alp = float(object['rotation'])
                result.append([x, y, z, l, w, h, alp])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        gt_boxes_corners = V.boxes_to_corners_3d(torch.Tensor(result))

        for j in range(gt_boxes_corners.shape[0]):
            gt_points_box = gt_boxes_corners[j]
            center = torch.mean(gt_points_box, 0).reshape((-1, 3))
            gt_points_box = torch.cat((gt_points_box, center), 0)
            points_cen = torch.mean(gt_points_box[[0, 1, 4, 5], :], 0).reshape((-1, 3))  # 车头中心
            gt_points_box = torch.cat((gt_points_box, points_cen), 0)
            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7], [8, 9], [0, 5], [1, 4]])
            colors = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0],
                               [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0],
                               [1, 0, 0], [1, 0, 0], [1, 0, 0]])
            gt_line_set = o3d.geometry.LineSet()
            gt_line_set.points = o3d.utility.Vector3dVector(np.array(gt_points_box.cpu()))
            gt_line_set.lines = o3d.utility.Vector2iVector(lines_box)
            gt_line_set.colors = o3d.utility.Vector3dVector(colors)
            linesets.append(gt_line_set)

        custom_draw_geometry(vis, point_cloud, linesets, view_json_filename)


if __name__ == "__main__":

    # pointcloud_path = '/data/RADIal/data/WJ-Multi-Frame-Dataset/pointcloud-multi-frame/20211216084030/pcd/'
    # label_path = '/data/RADIal/data/WJ-Multi-Frame-Dataset/pointcloud-multi-frame/20211216084030/label_7/'

    pointcloud_path = '/data/RADIal/data2/Baidu_DAIR_V2X_Dataset/RoadSide_DAIR_V2X_I/single-infrastructure-side-example/velodyne'
    label_path = '/data/RADIal/data2/Baidu_DAIR_V2X_Dataset/RoadSide_DAIR_V2X_I/single-infrastructure-side-example/label/virtuallidar'

    # 保存视角
    # pcd_file = os.path.join(pointcloud_path,'000000.pcd')
    # pcd = o3d.io.read_point_cloud(pcd_file)
    # save_view_point(pcd, "viewpoint.json")
    # load_view_point(pcd, "viewpoint.json")
    main(pointcloud_path, label_path, './viewpoint.json')
