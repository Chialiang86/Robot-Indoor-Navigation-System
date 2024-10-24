
import numpy as np
import math
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import argparse

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.05) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



def navigateAndSee(action="", action_names=None, sim=None, id_to_label=None):
    if action in action_names:
        observations = sim.step(action)

        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        #cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        # cv2.imshow("semantic", transform_semantic(id_to_label[observations["semantic_sensor"]]))
        return transform_rgb_bgr(observations["color_sensor"])

    return None

def get_unit_xz_by_quaternion(q):

    # base = [0, 0, -1]

    [rw, rx, ry, rz] = q
    # m00 = 1 - 2 * (ry * ry + rz * rz)
    # m01 = 2 * (rx * ry - rw * rz)
    m02 = 2 * (rx * rz + rw * ry)

    # m10 = 2 * (rx * ry + rw * rz)
    # m11 = 1 - 2 * (rx * rx + rz * rz)
    # m12 = 2 * (ry * rz - rw * rx)

    # m20 = 2 * (rx * rz - rw * ry)
    # m21 = 2 * (ry * rz + rw * rx)
    m22 = 1 - 2 * (rx * rx + ry * ry)
    return [-m02, -m22]

def get_sensor_state(agent):
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']

    xz_pos = [sensor_state.position[0], sensor_state.position[2]]
    
    q = [sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z]
    xz_dir = get_unit_xz_by_quaternion(q)
    
    return xz_pos, xz_dir

def judge_angle(dir):
    if dir[0] != 0 :
        theta = math.atan(dir[1] / dir[0]) * 180. / math.pi
        if theta >= 0:
            theta = 180 + theta if dir[0] < 0 else theta
        else :
            theta = 180 + theta if dir[0] < 0 else theta + 360
    elif dir[1] == 1:
        theta = 90
    elif dir[1] == -1:
        theta = 270
    
    return theta

def get_rotate_dir(xz_dir, diff_dir):
    
    theta_diff = judge_angle(diff_dir)
    theta_xz = judge_angle(xz_dir)

    theta_relative = (theta_diff - theta_xz + 360) % 360
    
    rotate_dir = 1 if theta_relative <= 180 else -1
    rotate_amount = 360 - theta_relative if theta_relative > 180 else theta_relative
    
    return rotate_dir, rotate_amount

def get_distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def main(args):

    # This is the scene we are going to load.
    # support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
    ### put your scene path ###
    test_scene = "mesh_semantic.ply"
    path = "info_semantic.json"

    #global test_pic
    #### instance id to semantic id 
    with open(path, "r") as f:
        annotations = json.load(f)

    id_to_label = []
    instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
    for i in instance_id_to_semantic_label_id:
        if i < 0:
            id_to_label.append(0)
        else:
            id_to_label.append(i)
    id_to_label = np.asarray(id_to_label)

    ######

    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 0,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
    }

    # This function generates a config for the simulator.
    # It contains two parts:
    # one for the simulator backend
    # one for the agent, where you can attach a bunch of sensors

    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)


    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    ####################### Implement By Myself #######################

    # load .txt file which contains route xz position
    f_path = open(args.txt, 'r')
    paths = f_path.readlines()
    xz_nodes = []

    for p in paths:
        x = float(p.split(' ')[0])
        z = float(p.split(' ')[1])
        xz_nodes.append([x, z])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([xz_nodes[0][0], 0.0, xz_nodes[0][1]])  # agent in world space
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)


    # action = "move_forward"
    # navigateAndSee(action, action_names, sim, id_to_label)

    key = input("Ready to start, please press a key to continue...")
    
    img_frames = []

    for xz_node in xz_nodes[1:]:

        # rotate to accurate direction
        xz_pos, xz_dir = get_sensor_state(agent)

        diff_dir = [xz_node[0] - xz_pos[0], xz_node[1] - xz_pos[1]]
        rotate_dir, rotate_amount = get_rotate_dir(xz_dir, diff_dir)

        print('{} -> {}'.format(xz_pos, xz_node))
        
        while rotate_amount > 5.0:
            cv2.waitKey(10)

            action = "turn_left" if rotate_dir == -1 else "turn_right"
            frame = navigateAndSee(action, action_names, sim, id_to_label)
            img_frames.append(frame)

            xz_pos, xz_dir = get_sensor_state(agent)
            diff_dir = [xz_node[0] - xz_pos[0], xz_node[1] - xz_pos[1]]
            rotate_dir, rotate_amount = get_rotate_dir(xz_dir, diff_dir)
        
        while get_distance(xz_pos, xz_node) > 0.5:
            cv2.waitKey(10)
            
            action = "move_forward"
            frame = navigateAndSee(action, action_names, sim, id_to_label)
            img_frames.append(frame)

            xz_pos, xz_dir = get_sensor_state(agent)
    
    print('saving video...')

    h, w, _ = img_frames[0].shape
    size = (w, h)

    out = cv2.VideoWriter('../Map/observation.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, size)

    for frame in img_frames:
        out.write(frame)
    out.release()

    print('process completed.')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', '-t', default='route.txt')
    args = parser.parse_args()

    main(args)