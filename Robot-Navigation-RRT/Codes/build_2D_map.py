import argparse
import numpy as np
import matplotlib.pyplot as plt

def transform(pos, trans_mat=np.identity(4)):
    padding_zero = np.ones((pos.shape[0], 1))
    pts = np.hstack((pos, padding_zero)).T
    ret = trans_mat @ pts
    print(ret[:3,:].T)
    return ret[:3,:].T

def main(args):
    color = np.load(args.color_path).astype(int)
    point = np.load(args.point_path).astype(float)

    # remove ceiling and floor
    cond_high = np.where(point[:,1] > -0.005)
    point_tmp = np.delete(point, cond_high, axis=0)
    color_tmp = np.delete(color, cond_high, axis=0)
    cond_low = np.where(point_tmp[:,1] < -0.032)
    point_wall = np.delete(point_tmp, cond_low, axis=0)
    color_wall = np.delete(color_tmp, cond_low, axis=0)
    
    color_hex = np.array(['#%02x%02x%02x' % (c[0], c[1], c[2]) for c in color_wall])

    point_x = point_wall[:,0] * 10000. / 255
    point_z = point_wall[:,2] * 10000. / 255

    # coord_x = np.array([0., 1.])
    # coord_z = np.array([0., 0.])
    # coord_color = np.array(['#ffff00', '#ffff00'])
    
    # point_x = np.hstack((point_x, coord_x))
    # point_z = np.hstack((point_z, coord_z))
    # color_hex = np.hstack((color_hex, coord_color))

    # x_range = np.ceil(np.max(point_x) - np.min(point_x))
    # z_range = np.ceil(np.max(point_z) - np.min(point_z))
    # print(x_range, z_range)

    # plt.figure(figsize=(z_range, x_range))
    fig, ax = plt.subplots()
    ax.axis('equal')
    plt.axis('off')
    plt.scatter(point_z, point_x, s=1., c=color_hex)
    plt.savefig('../Map/map.png')
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_path', '-c', default='semantic_3d_pointcloud/color0255.npy', type=str)
    parser.add_argument('--point_path', '-p', default='semantic_3d_pointcloud/point.npy', type=str)
    args = parser.parse_args()
    
    main(args)