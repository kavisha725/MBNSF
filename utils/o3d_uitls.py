# Point cloud processing utility functions.

import numpy as np
import open3d as o3d
import copy
from matplotlib import pyplot as plt

def make_open3d_point_cloud(xyz, color=None):
    if not isinstance(xyz, np.ndarray):
        xyz = xyz.cpu().detach().numpy()
        xyz = np.squeeze(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color)==3: pcd.paint_uniform_color([1, 0.706, 0])
        else: pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def draw_registration_result(source, target, transformation=np.eye(4), source_color=None, target_color=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not isinstance(source_temp, o3d.geometry.PointCloud):
        source_temp = make_open3d_point_cloud(source_temp)
        source_temp.paint_uniform_color([1, 0.706, 0])
    else:
        if source_temp.colors is None or len(source_temp.colors) == 0:
            source_temp.paint_uniform_color([1, 0.706, 0])
    if not isinstance(target_temp, o3d.geometry.PointCloud):
        target_temp = make_open3d_point_cloud(target_temp)
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    else:
        if target_temp.colors is None or len(target_temp.colors) == 0:
            target_temp.paint_uniform_color([1, 0.706, 0])
    
    if not source_color is None:
        source_temp.paint_uniform_color(source_color)
    if not target_color is None:
        target_temp.paint_uniform_color(target_color)
    target_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def transfrom_cloud(xyz, transformation):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.transform(transformation)
    return np.asarray(pcd.points)

def visualize_cloud(cloud, color=None, viewpoint=None):
    if not isinstance(cloud, o3d.geometry.PointCloud):
        cloud = make_open3d_point_cloud(cloud, color)
    if not viewpoint:
        o3d.visualization.draw_geometries([cloud])
    else:
        load_view_point(cloud, viewpoint)

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

def load_view_point_multi(pcds, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    for pcd in pcds:
        vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

def extract_clusters_dbscan(cloud, eps = 0.9, min_points=10, return_clusters= False, return_colored_pcd=False):
    pcl = copy.deepcopy(cloud)
    pcl = make_open3d_point_cloud(pcl)
    labels = np.array(
            pcl.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    if return_colored_pcd:
        cmap = plt.get_cmap("tab20")
        max_label = labels.max()
        print("Has %d clusters" % (max_label + 1))
        colors = cmap(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([pcl])
        # save_view_point(pcl, 'pcd_viewpoint.json')
        load_view_point(pcl, 'pcd_viewpoint.json')
        
    clusters = []
    if return_clusters:
        label_ids = np.delete(np.unique(labels), 0)
        for id in label_ids:
            clusters.append(cloud[labels == id])
        clusters = np.asarray(clusters)

        if return_colored_pcd: 
            return labels, clusters, pcl
        return labels, clusters
    else:
        if return_colored_pcd: 
            return labels, pcl
        return labels

def viz_clusters(clusters, return_pcds=False):
    cmap = plt.get_cmap("tab20")
    max_label = len(clusters)
    labels = np.arange(max_label)
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0

    pcds = []
    for id in range(max_label):
        cluster = clusters[id]
        color = colors[id][:3]
        pcds.append(make_open3d_point_cloud(cluster, np.tile(color,(len(cluster),1))))
    o3d.visualization.draw_geometries(pcds)
    if return_pcds:
        return pcds

def get_bb(xyz, axis_aligned = False, bev = False):
    pc = make_open3d_point_cloud(xyz)
    if axis_aligned:
        bb_aa = pc.get_axis_aligned_bounding_box()
        bb_aa_vertices = np.asarray(bb_aa.get_box_points())
        if bev:
            # z_min = min(bb_aa_vertices[:,2])
            # bb_bev_vertices = bb_aa_vertices[bb_aa_vertices[:,2] == z_min]
            x_min = min(bb_aa_vertices[:,0])
            x_max = max(bb_aa_vertices[:,0])
            y_min = min(bb_aa_vertices[:,1])
            y_max = max(bb_aa_vertices[:,1])
            bb_bev_vertices = np.asarray([x_min, y_min, x_max, y_max])
            return bb_bev_vertices
        else:
            return bb_aa_vertices
    else:
        bb = pc.get_oriented_bounding_box()
        return np.asarray(bb.get_box_points())

def extract_single_cluster_info(cluster, axis_aligned = False, bev = False):
    cluster_center = np.mean(cluster, axis=0)

    cluster_bb = get_bb(cluster, axis_aligned = axis_aligned, bev = bev)
    bb_x_diff = cluster_bb[2] - cluster_bb[0]
    bb_y_diff = cluster_bb[3] - cluster_bb[1]
    if bb_x_diff >= bb_y_diff:
        delta_x = 2.5
        delta_y = delta_x *(bb_y_diff/bb_x_diff)
    else:
        delta_y = 2.5
        delta_x = delta_y * (bb_x_diff/bb_y_diff)
    cluster_bb_dash = np.asarray([cluster_bb[0] - delta_x, 
                                cluster_bb[1] - delta_y, 
                                cluster_bb[2] + delta_x, 
                                cluster_bb[3] + delta_y])
    
    return cluster_center, cluster_bb_dash

def extract_cluster_info(clusters, axis_aligned = False, bev = False):
    cluster_centers, cluster_bbs = [], []
    for id in range(len(clusters)):
        cluster = clusters[id]
        cluster_center, cluster_bb_dash = extract_single_cluster_info(cluster, axis_aligned = axis_aligned, bev = bev)
        cluster_centers.append(cluster_center)
        cluster_bbs.append(cluster_bb_dash)
    return np.asarray(cluster_centers), np.asarray(cluster_bbs)

def get_inlier_points(cloud, bb):
    cloud_temp = copy.deepcopy(cloud)

    cloud_temp = cloud_temp[cloud_temp[:,0] > bb[0]]
    cloud_temp = cloud_temp[cloud_temp[:,1] > bb[1]]
    cloud_temp = cloud_temp[cloud_temp[:,0] < bb[2]]
    cloud_temp = cloud_temp[cloud_temp[:,1] < bb[3]]
    return cloud_temp
