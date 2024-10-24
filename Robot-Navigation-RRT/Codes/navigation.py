import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from numpy.core.einsumfunc import _optimal_path
from sklearn.neighbors import NearestNeighbors

COORDINATE_RATIO = 30.0

def find_target_by_color(map, color):
    (h, w, _) = map.shape
    ret = []
    for x in range(h):
        for z in range(w):
            if ((map[x, z] == color).all()):
                ret.append((x, z))
    print(ret)
    return np.array(ret)

start_pos = None
def get_click_pose(event, x, y, flags, param):
    global start_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pos = [y, x]
        print(start_pos)

def get_start_pos(map):
    cv2.imshow('image', map)
    cv2.setMouseCallback('image', get_click_pose)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return start_pos

def not_repeat(rand, pt_set):
    for pt in pt_set:
        if pt==rand:
            return False
    return True

def random_sample(map_shape, pt_set, target, rand_thresh=0.5, margin=1.0):
    h, w, _ = map_shape
    h, w = int(np.round(h - margin * COORDINATE_RATIO)), int(np.round(w - margin * COORDINATE_RATIO))
    rand = random.random()
    while True:
        if rand < rand_thresh:
            rand_x = int(np.round(random.random() * h))
            rand_z = int(np.round(random.random() * w))
            ret = [rand_x, rand_z]
        else :
            ret = target 

        if not_repeat(ret, pt_set):
            break

    return ret

def distance(x, y):
    x_diff = x[0] - y[0]
    z_diff = x[1] - y[1]
    return np.sqrt(x_diff ** 2 + z_diff ** 2)

def steer(x_nearest, x_rand, step_size=1.):
    pixel_step = COORDINATE_RATIO * step_size
    x_diff = x_rand[0] - x_nearest[0]
    z_diff = x_rand[1] - x_nearest[1]
    length = np.sqrt(x_diff ** 2 + z_diff ** 2)
    assert length > 0 

    ratio = pixel_step / length

    x_step = int(ratio * x_diff)
    z_step = int(ratio * z_diff)

    ret = [x_nearest[0] + x_step, x_nearest[1] + z_step]
    return ret

def is_obstacle_free(map, x_src, x_dst, radius=.2, step_num=20):
    pixel_radius = int(np.round(COORDINATE_RATIO * radius))

    # discrete radius map center and 8 direction 
    check_pts =  [[                 0,                  0],
                  [                 0,       pixel_radius],
                  [ pixel_radius // 2,  pixel_radius // 2],
                  [      pixel_radius,                  0],
                  [ pixel_radius // 2, -pixel_radius // 2],
                  [                 0,      -pixel_radius],
                  [-pixel_radius // 2, -pixel_radius // 2],
                  [     -pixel_radius,                  0],
                  [-pixel_radius // 2,  pixel_radius // 2]]
    
    
    x_diff = x_dst[0] - x_src[0]
    z_diff = x_dst[1] - x_src[1]

    for i in range(1, step_num):
        x = int(np.round(x_src[0] + i / step_num * x_diff))
        z = int(np.round(x_src[1] + i / step_num * z_diff))

        for check_pt in check_pts:
            color = map[x + check_pt[0],z + check_pt[1]]
            if (color!=[255, 255, 255]).any():
                return False

    return True

def create_path(line_set, start, end):
    path_set = []

    path_set.append(end)
    tmp = end
    while tmp != start:
        for line in line_set:
            if tmp == line[1]:
                path_set.append(tmp)
                tmp = line[0]
                break
    path_set.append(start)
    
    return path_set

def optimize_path(map, path_set):
    optimal_path_set = [path_set[0]]
    size = len(path_set)
    assert size > 1

    i = 0
    while i < size - 1:
        tmp = path_set[i]
        for check_i in range(size - 1, i, -1):
            if is_obstacle_free(map, tmp, path_set[check_i]):
                optimal_path_set.append(path_set[check_i])
                i = check_i
                break
    
    return optimal_path_set

def rrt_algo(map, begin, target, step_size=1.0, rand_thresh=0.5):
    pt_set = [begin]
    line_set = []

    nn = NearestNeighbors(n_neighbors=1)
    cnt = 0
    while True:
        x_rand = random_sample(map.shape, pt_set, target, 0.9)
        
        # find the nearest point in point set to the new sample point 
        nn.fit(pt_set)
        _, index = nn.kneighbors([x_rand])
        x_nearest = pt_set[index[0,0]]
        x_new = steer(x_nearest, x_rand, step_size)
        
        cnt += 1
        print('iteration : {}'.format(cnt))
        
        if is_obstacle_free(map, x_nearest, x_new):
            pt_set.append(x_new)
            line_set.append([x_nearest, x_new])

            # if the new point can go straight to the target point without collision, then end
            if is_obstacle_free(map, x_new, target):
                pt_set.append(target)
                line_set.append([x_new, target])
                break

    return line_set

def save_path(path_set, original=(290, 252), scale_ratio=COORDINATE_RATIO):
    f = open('route.txt', 'w')
    for i in range(len(path_set) - 1, -1, -1):
        x = -(path_set[i][0] - original[0]) / scale_ratio
        z = (path_set[i][1] - original[1]) / scale_ratio
        f.write('{} {}\n'.format(x, z))
    print('route.txt saved.')
    return

def main(args):
    '''
    260 252 -> (1, 0)
    290 252 -> (0, 0)
    '''

    # const information
    category_dict = {'0':'refrigerator', '1':'rack', '2':'cushion', '3':'lamp', '4':'cooktop'}
    color_dict = {'refrigerator':(255, 0, 0), 'rack':(0, 255, 133), 'cushion':(255, 9, 92), 'lamp':(160, 150, 20), 'cooktop':(7, 255, 224)}
    pos_dict = {'refrigerator':[246, 257], 'rack':[178, 368], 'cushion':[261, 495], 'lamp':[342, 441], 'cooktop':[297, 218]}
    
    # ret = find_target_by_color(map, color_dict[category_dict[choose_index]])
    # x = ret[:,0]
    # z = ret[:,1]
    # plt.scatter(x, z)
    # plt.show()
    
    # color dict of 4 objects
    choose_index = input('Please input target (refrigerator -> 0, rack -> 1, cushion -> 2, lamp -> 3, cooktop -> 4) :')
    while len(choose_index) != 1 or choose_index not in '01234':
        choose_index = input('WRONG INPUT!, Please input target (refrigerator -> 0, rack -> 1, cushion -> 2, lamp -> 3, cooktop -> 4) :')
    
    target = pos_dict[category_dict[choose_index]]

    # load map
    map = cv2.imread(args.map)
    begin = get_start_pos(map)
    print('start : {}, tar : {}'.format(begin, target))
    
    lines = rrt_algo(map, begin, target, step_size=0.5, rand_thresh=0.5)
    
    path_set = create_path(lines, begin, target)
    optimal_path_set = optimize_path(map, path_set)

    print('before : {}, after : {}'.format(len(path_set), len(optimal_path_set)))

    # add lines
    for line in lines:
        x = (line[0][1], line[0][0])
        y = (line[1][1], line[1][0])
        cv2.line(map, x, y, (200, 200, 20), 1)
    
    # add optimal path
    for i in range(len(optimal_path_set) - 1):
        x = (optimal_path_set[i][1], optimal_path_set[i][0])
        y = (optimal_path_set[i + 1][1], optimal_path_set[i + 1][0])
        cv2.line(map, x, y, (120, 205, 20), 2)
    
    # show result
    cv2.imshow('res', map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('../Map/route.png', map)
    print('route.png saved.')

    save_path(optimal_path_set)

#refrigerator, rack, cushion, lamp, cooktop
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', '-m', default='../Map/map.png', type=str)
    args = parser.parse_args()
    
    main(args)