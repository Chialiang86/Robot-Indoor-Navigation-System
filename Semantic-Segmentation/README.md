# 3d Reconstruction using habitat 

## How to run my program

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
    1. change CURRENT_DIR to 1 # 1 : first, 2 : second
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
    1. change CURRENT_DIR to 2 # 1 : first, 2 : second
    2. execution : python reconstruction.py

### about reconstruction.py 
- local_icp_algorithm is the function written by myself, it can align two point cloud by point-to-point ICP local_icp_algorithm
- depth_image_to_point_cloud is the function that can create point cloud by given rgb/depth image and the camera intrinsic