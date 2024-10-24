import open3d as o3d
import cv2 
import numpy as np
import glob
import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image
from sklearn.neighbors import NearestNeighbors



# set intrincsic function
WIDTH = 512
HEIGHT = 512
FOV = 0.5 * np.pi 
FX = WIDTH / 2 * np.tan(FOV / 2)
FY = HEIGHT / 2 * np.tan(FOV / 2)
CX = WIDTH / 2
CY = HEIGHT / 2
INTRINSIC = [WIDTH, HEIGHT, FX, FY, CX, CY]

# util function
def get_transform_by_quaternion_and_pos(pose):

    # extract info
    [x, y, z, rw, rx, ry, rz] = pose

    # convert quaternion to rotation
    m00 = 1 - 2 * (ry * ry + rz * rz)
    m01 = 2 * (rx * ry - rw * rz)
    m02 = 2 * (rx * rz + rw * ry)

    m10 = 2 * (rx * ry + rw * rz)
    m11 = 1 - 2 * (rx * rx + rz * rz)
    m12 = 2 * (ry * rz - rw * rx)

    m20 = 2 * (rx * rz - rw * ry)
    m21 = 2 * (ry * rz + rw * rx)
    m22 = 1 - 2 * (rx * rx + ry * ry)

    # transform matrix
    transform = np.array([[m00, m01, m02,   x],
                          [m10, m11, m12,   y],
                          [m20, m21, m22,   z],
                          [  0,   0,   0,   1]]).astype(np.float64)

    return transform

# major function
def depth_image_to_point_cloud(rgb, depth, intrinsic):

    adjust = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0,-1, 0],
                    [0, 0, 0, 1]])

    [w, h, fx, fy, cx, cy] = intrinsic

    ix, iy =  np.meshgrid(range(w), range(h))
    
    x_ratio = (ix.ravel() - cx) / fx
    y_ratio = (iy.ravel() - cy) / fy

    z = depth.ravel()
    x = z * x_ratio
    y = z * y_ratio

    points = np.vstack((x, y, z)).T
    colors = np.reshape(rgb,(512 * 512, 3))
    colors = np.array([colors[:,2], colors[:,1], colors[:,0]]).T / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(adjust)

    return pcd

def get_all_pcds(num=-1):
    # get point clouds
    num = len(glob.glob('{}/*.png'.format(rgb_path))) if num == -1 else num
    pose_file = open(pose_path, 'r')
    pcds = []
    poses = []
    for id in range(num):

        path_id = 1 + id

        # get pose info from corresponding image in file
        data_raw = pose_file.readline()
        pose_raw = np.asarray([np.float64(x) for x in str(data_raw).split('\n')[0].split()])
        pose = get_transform_by_quaternion_and_pos(pose_raw)

        # read image file and get point cloud
        demo_color = cv2.imread('{}{}.png'.format(rgb_path, path_id ))
        demo_depth = np.array(Image.open('{}{}.png'.format(depth_path, path_id ))).astype(np.float32) / 65535.0 * 10.0
        pcd = depth_image_to_point_cloud(demo_color, demo_depth, [WIDTH, HEIGHT, FX, FY, CX, CY])

        print('{}{}.png'.format(rgb_path, path_id), pcd)
        
        # write to buffer
        poses.append(pose)
        pcds.append(pcd)

    return pcds, poses
    
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def ransac_transform(kpt1, kpt2, matches, sample_num=3, thresh=10, iteration_num=100):
    random_generator = np.random.default_rng()

    # initialize np array for points (homogeneous corordinate)
    points_1 = np.ones((3, len(matches)))
    points_2 = np.ones((3, len(matches)))
    points_1[:2,:] = np.array([[int(kpt1[item.queryIdx].pt[0] + 0.5), int(kpt1[item.queryIdx].pt[1] + 0.5)] for item in matches]).T
    points_2[:2,:] = np.array([[int(kpt2[item.trainIdx].pt[0] + 0.5), int(kpt2[item.trainIdx].pt[1] + 0.5)] for item in matches]).T

    best_inliers = 0
    best_transform = None
    best_pt1 = None
    best_pt2 = None

    thresh = thresh ** 2

    itr = 0
    while itr < iteration_num:
        while True:
            sample_ids = random_generator.choice(len(matches), sample_num, replace=False)
            pts_1 = points_1[:,sample_ids]
            pts_2 = points_2[:,sample_ids]
            # check is not singular
            if np.linalg.det(pts_1) != 0 and np.linalg.det(pts_2) != 0:
                break
        
        tranform_12 = np.dot(pts_2, np.linalg.inv(pts_1))
        
        diff = (np.dot(tranform_12, points_1) - points_2).T

        pt1_tmp = []
        pt2_tmp = []
        inliers = 0
        for i, item in enumerate(diff):
            if np.sum(item * item) < thresh:
                inliers += 1
                pt1_tmp.append(points_1[:,i])
                pt2_tmp.append(points_2[:,i])

        if inliers > best_inliers:
            best_inliers = inliers
            best_pt1 = pt1_tmp 
            best_pt2 = pt2_tmp 
            best_transform = tranform_12

        itr += 1
    return best_pt1, best_pt2, best_transform, best_inliers

def compute_best_match(src, dst, prefix=''):

    img1 = cv2.imread('Data_collection/{}/rgb/{}.png'.format(prefix, src))
    img2 = cv2.imread('Data_collection/{}/rgb/{}.png'.format(prefix, dst))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kpt1, desp1 = sift.detectAndCompute(gray1, None)
    kpt2, desp2 = sift.detectAndCompute(gray2, None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(desp1,desp2)

    return ransac_transform(kpt1, kpt2, matches)

#  (implement by myself)
def local_icp_algorithm(pc1, pc2, init_trans=None, thresh=0.001, iteration=1000, verbose=False):
    pc1_array = np.asarray(pc1.points)
    pc2_array = np.asarray(pc2.points)

    # remove point by height
    min_height, max_height = np.min(pc1_array[:,1]), np.max(pc1_array[:,1])
    diff = max_height - min_height
    remove_ratio = 0.6
    low_thresh, high_thresh = min_height + (remove_ratio / 2) * diff, max_height - (remove_ratio / 2) * diff
    cond1, cond2 = np.where(pc1_array[:,1] < low_thresh), np.where(pc2_array[:,1] < low_thresh)
    pc1_array = np.delete(pc1_array, cond1, axis=0)
    pc2_array = np.delete(pc2_array, cond2, axis=0)
    cond1, cond2 = np.where(pc1_array[:,1] > high_thresh), np.where(pc2_array[:,1] > high_thresh)
    pc1_array = np.delete(pc1_array, cond1, axis=0)
    pc2_array = np.delete(pc2_array, cond2, axis=0)

    # homogenous coordinate form
    pc1_copy = np.ones((4, pc1_array.shape[0]))
    pc2_copy = np.ones((4, pc2_array.shape[0]))
    pc1_copy[:3, :] = np.copy(pc1_array.T)
    pc2_copy[:3, :] = np.copy(pc2_array.T)

    nn = NearestNeighbors(n_neighbors=1)
    err = np.inf
    
    # initialize return matrix
    ret_T = np.identity(4)

    # multiply initialize matrix
    if init_trans is not None:
        pc1_copy = init_trans @ pc1_copy
        ret_T = init_trans

    for i in range(iteration):
        
        # for point to point match
        nn.fit(pc1_copy[:3,:].T)
        distance, idx = nn.kneighbors(pc2_copy[:3,:].T)
        distance = distance.flatten()

        cond = np.where(distance > 1.5 * thresh) # difference is 3cm
        idx = np.delete(idx, cond)
        pc1_nn = pc1_copy[:,idx] 
        pc2_nn = np.delete(pc2_copy.T, cond, axis=0).T 

        # ICP core algorithm
        pc1_center = np.array([np.mean(pc1_nn[0]), np.mean(pc1_nn[1]), np.mean(pc1_nn[2])])
        pc2_center = np.array([np.mean(pc2_nn[0]), np.mean(pc2_nn[1]), np.mean(pc2_nn[2])])

        pc1_reletive = pc1_nn.T[:,:3] - pc1_center
        pc2_reletive = pc2_nn.T[:,:3] - pc2_center

        U, S, Dt = np.linalg.svd(pc2_reletive.T @ pc1_reletive)

        R =  U @ Dt
        if np.linalg.det(R) < 0:
            Dt[2,:] *= -1
            R = U @ Dt

        t = np.array([pc2_center.T - R @ pc1_center.T])
        T = np.identity(4)
        T[:3,:3] = R
        T[:3,3] = t

        pc1_copy = T @ pc1_copy 
        ret_T = T @ ret_T

        mean_err = np.mean(np.abs(distance))

        if verbose == True:
            print('iteration {} -> mean error = {}'.format(i, mean_err))
        if err - mean_err < 0.000001 or err < thresh:
            break
        
        err = mean_err

    return ret_T


# open3d icp algorithm
def open3d_icp_algorithm(pc1, pc2, init_trans=np.identity(4), thresh=0.001, iteration=1000):
    pc1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    icp_fine = o3d.pipelines.registration.registration_icp(
        pc1, pc2, thresh,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))
    transformation_icp = icp_fine.transformation
    return transformation_icp


def full_registration(pcds, thresh):
    
    n_pcds = len(pcds)
    global_threshold = 1.5 * thresh
    local_threshold = 0.5 * thresh

    odometry_o3d = np.identity(4)
    odometry_mine = np.identity(4)
    odometry_global = np.identity(4)

    transform_list_o3d = []
    transform_list_mine = []
    transform_list_global = []
    transform_list_o3d.append(odometry_o3d)
    transform_list_mine.append(odometry_mine)
    transform_list_global.append(odometry_global)
    for id in range(n_pcds - 1):
        print("Processing id =  {}/{}".format(id, n_pcds - 1))
        src = pcds[id]
        dst = pcds[id + 1]

        # down sample and get features
        src_down, src_fpfh = preprocess_point_cloud(src, thresh)
        dst_down, dst_fpfh = preprocess_point_cloud(dst, thresh)

        # global registration
        icp_rough = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down, dst_down, src_fpfh, dst_fpfh, True,
            global_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    global_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        # my function / open3d function to process icp algorithm
        transform_o3d = open3d_icp_algorithm(src_down, dst_down, icp_rough.transformation, local_threshold)
        transform_mine = local_icp_algorithm(src_down, dst_down, icp_rough.transformation, local_threshold)
        
        # add transform matrix to list
        odometry_o3d = np.dot(transform_o3d, odometry_o3d)
        odometry_mine = np.dot(transform_mine, odometry_mine)
        odometry_global = np.dot(icp_rough.transformation, odometry_global)
        transform_list_o3d.append(np.linalg.inv(odometry_o3d))
        transform_list_mine.append(np.linalg.inv(odometry_mine))
        transform_list_global.append(np.linalg.inv(odometry_global))
    return transform_list_o3d, transform_list_mine, transform_list_global

# constant config

FLOOR = 0 # 0: first, 1: second
CURRENT_DIR = ['first_floor', 'second_floor']

pose_path = 'Data_collection/{}/GT_Pose.txt'.format(CURRENT_DIR[FLOOR])
rgb_path = 'Data_collection/{}/rgb/'.format(CURRENT_DIR[FLOOR])
depth_path = 'Data_collection/{}/depth/'.format(CURRENT_DIR[FLOOR])

VOXEL_SIZE = 0.03

# get all point cloud
pcds, transforms = get_all_pcds(-1)
pcds_mine = copy.deepcopy(pcds)

print("Full registration ...")
thresh = VOXEL_SIZE
transform_list_o3d, transform_list_mine, transform_list_global = full_registration(pcds, thresh)
adjust_transform = transforms[0]
origin = np.array([0, 0, 0, 1]).T

# initialize trajectory points 
traj_ground_points = []
traj_ground_lines = [[i, i+1] for i in range(len(pcds)-1)]
traj_ground_colors = [[0, 1, 0] for i in range(len(pcds)-1)]
traj_icp_points_o3d = []
traj_icp_lines_o3d = [[i, i+1] for i in range(len(pcds)-1)]
traj_icp_colors_o3d = [[1, 0, 0] for i in range(len(pcds)-1)]
traj_icp_points_mine = []
traj_icp_lines_mine = [[i, i+1] for i in range(len(pcds)-1)]
traj_icp_colors_mine = [[1, 0, 0] for i in range(len(pcds)-1)]
traj_global_points = []

height_thresh_list = [0.75, 3.3]

# multiway registration
for point_id in range(len(pcds)):
    # process scenes
    pcds[point_id].transform(adjust_transform @ transform_list_o3d[point_id])
    pcds_mine[point_id].transform(adjust_transform @ transform_list_mine[point_id])

    # remove roof
    height_thresh = height_thresh_list[FLOOR]
    # icp by open3d
    points_o3d = np.array(pcds[point_id].points)
    colors_o3d = np.array(pcds[point_id].colors)
    cond_o3d = np.where(points_o3d[:,1] > height_thresh)
    pcd_without_roof = np.delete(points_o3d, cond_o3d, axis=0)
    pcd_colors = np.delete(colors_o3d, cond_o3d, axis=0)
    pcds[point_id].points = o3d.utility.Vector3dVector(pcd_without_roof)
    pcds[point_id].colors = o3d.utility.Vector3dVector(pcd_colors)
    # icp by myself
    points_mine = np.array(pcds_mine[point_id].points)
    colors_mine = np.array(pcds_mine[point_id].colors)
    cond_mine = np.where(points_mine[:,1] > height_thresh)
    pcd_without_roof = np.delete(points_mine, cond_mine, axis=0)
    pcd_colors = np.delete(colors_mine, cond_mine, axis=0)
    pcds_mine[point_id].points = o3d.utility.Vector3dVector(pcd_without_roof)
    pcds_mine[point_id].colors = o3d.utility.Vector3dVector(pcd_colors)


    # process trajectory
    ground_pos = transforms[point_id] @ origin
    icp_pos_o3d = adjust_transform @ transform_list_o3d[point_id] @ origin
    icp_pos_mine = adjust_transform @ transform_list_mine[point_id] @ origin
    icp_pos_global = adjust_transform @ transform_list_global[point_id] @ origin
    traj_ground_points.append(ground_pos.T[:3])
    traj_icp_points_o3d.append(icp_pos_o3d.T[:3])
    traj_icp_points_mine.append(icp_pos_mine.T[:3])
    traj_global_points.append(icp_pos_global.T[:3])


# draw lines
line_ground_set = o3d.geometry.LineSet()
line_ground_set.points = o3d.utility.Vector3dVector(np.array(traj_ground_points))
line_ground_set.lines = o3d.utility.Vector2iVector(np.array(traj_ground_lines))
line_ground_set.colors = o3d.utility.Vector3dVector(np.array(traj_ground_colors))

line_icp_set_o3d = o3d.geometry.LineSet()
line_icp_set_o3d.points = o3d.utility.Vector3dVector(np.array(traj_icp_points_o3d))
line_icp_set_o3d.lines = o3d.utility.Vector2iVector(np.array(traj_icp_lines_o3d))
line_icp_set_o3d.colors = o3d.utility.Vector3dVector(np.array(traj_icp_colors_o3d))
line_icp_set_mine = o3d.geometry.LineSet()
line_icp_set_mine.points = o3d.utility.Vector3dVector(np.array(traj_icp_points_mine))
line_icp_set_mine.lines = o3d.utility.Vector2iVector(np.array(traj_icp_lines_mine))
line_icp_set_mine.colors = o3d.utility.Vector3dVector(np.array(traj_icp_colors_mine))

o3d_diff = np.array(traj_icp_points_o3d) - np.array(traj_ground_points)
o3d_square_err = [np.dot(line, line) for line in o3d_diff]
o3d_err = np.mean(np.abs(o3d_diff))
mine_diff = np.array(traj_icp_points_mine) - np.array(traj_ground_points)
mine_square_err = [np.dot(line, line) for line in mine_diff]
mine_err = np.mean(np.abs(mine_diff))
global_diff = np.array(traj_global_points) - np.array(traj_ground_points)
global_err = np.mean(np.abs(global_diff))

print('--------------------------- Reconstruction Result ---------------------------')
print('| The trajectory error of global registration only = {}'.format(global_err))
print('| The trajectory error of Open3d local ICP = {}'.format(o3d_err))
print('| The trajectory error of My local ICP = {}'.format(mine_err))
print('-----------------------------------------------------------------------------')
print('process completed.')

pcds.append(line_ground_set)
pcds.append(line_icp_set_o3d)
pcds_mine.append(line_ground_set)
pcds_mine.append(line_icp_set_mine)

o3d.visualization.draw_geometries(pcds, zoom=0.3412, front=[0, 1, 0], lookat=[0, 10, 5], up=[1, 0, 0])
o3d.visualization.draw_geometries(pcds_mine, zoom=0.3412, front=[0, 1, 0], lookat=[0, 10, 5], up=[1, 0, 0])
