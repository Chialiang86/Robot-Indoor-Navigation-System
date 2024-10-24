# Robot Indoor Navigation System 
This project contains the following three parts:
1. ICP Implementation for 3D Indoor Scene Reconstruction
2. Semantic Segmentation For Object Detection (under maintenance)
3. RRT Path-Planning for Robot Navigation 

## 1. 3d Indoor Scene Reconstruction

<img width="513" alt="first_floor_open3d" src="https://github.com/user-attachments/assets/b2eed1f3-38c5-4e4f-bd4f-c643112833e4">

### dependency
- python 3.6 (anaconda)
- open3d 0.12.0
- opencv-contrib-python 4.5.3.56
- opencv-python         4.5.3.56

### Important information 
- the scene is provided by the file 'mesh_semantic.ply'. I put it in root directory so that it can read directly in load.py

### first floor
- load.py -> capture first floor image (rgb, depth)
    1. change "FLOOR" to 0 # 0 : first, 1 : second
    2. set sim_settings = 
        {
            "scene": test_scene,  # Scene path
            "default_agent": 0,  # Index of the default agent
            "sensor_height": 0,  # Height of sensors in meters, relative to the agent
            "width": 512,  # Spatial resolution of the observations
            "height": 512,
            "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
        }
    3. execution : python load.py

- reconstruction.py 
    1. FLOOR = 0 # 0 : first, 1 : second
    2. execution : python reconstruction.py

### second floor
- load.py -> capture first floor image (rgb, depth)
    1. change "FLOOR" to 1 # 0 : first, 1 : second
    2. set sim_settings = 
        {
            "scene": test_scene,  # Scene path
            "default_agent": 0,  # Index of the default agent
            "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
            "width": 512,  # Spatial resolution of the observations
            "height": 512,
            "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
        }
    3. execution : python load.py

- reconstruction.py 
    1. FLOOR = 1 # 0 : first, 1 : second
    2. execution : python reconstruction.py

### about reconstruction.py 
- local_icp_algorithm is the function written by myself, it can align two point cloud by point-to-point ICP local_icp_algorithm
- depth_image_to_point_cloud is the function that can create point cloud by given rgb/depth image and the camera intrinsic

### console log information : about trajectory error
- an example result  (demo with 285 first-floor images)
--------------------------- Reconstruction Result ---------------------------
| The trajectory error of global registration only = 0.020334287968358646
| The trajectory error of Open3d local ICP = 0.001608008832216784
| The trajectory error of My local ICP = 0.00294859923219839
-----------------------------------------------------------------------------

## 2. Semantic Segmentation for Object Detection
- The corresponding codebase is under maintenance.

## 3. A Robot Navigation Framework

> Perception and Decision Making in Intelligent Systems 

- GitHub repo : https://github.com/Chialiang86/Robot-Navigation-RRT

### Run `build_2D_map.py` in `Codes/`
arguments :
- color_path :  default:semantic_3d_pointcloud/color0255.npy
- point_path : default:semantic_3d_pointcloud/point.npy
```shell 
python build_2D_map.py [--color_path path/to/color0255.npy --point_path path/to/point.npy]
```

The map will be saved in `../Map/map.png`

- result
![](https://i.imgur.com/sAD0xHA.png)

### Run `navigation.py` in `Codes/`
arguments : 
- map : the location of `mpa.png`, default : `../Map/map.png`
- optimal : whether show the better RRT path result, default : True

```shell 
python build_2D_map.py [--map path/to/map.png --optimal True]
```

#### 1. Input the object to search 
- refrigerator : 0
- rack : 1
- cushion : 2
- lamp : 3
- cooktop : 4

The command line output : 
```shell 
Please input target (refrigerator -> 0, rack -> 1, cushion -> 2, lamp -> 3, cooktop -> 4) :
```

#### 2. Click on a position on map.png as the starting point, and then press a random key to start RRT

A sample command line output : 
```shell 
[251, 437]
```

#### 3. Show result images

I will show two results, the first one is original path(left), the second one is better path(right)
![](https://i.imgur.com/84txJrF.png)

#### 4. Show result in habitat-sim

Show the video for the navigation, and then save video in `../Map/observation.avi`

- video example : https://drive.google.com/file/d/17C1OgpVTx1Qi34himszZbz5L3CTD7YFX/view?usp=sharing


