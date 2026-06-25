"""
demo_pybullet.py

Scaffold integrating graspnet-baseline with PyBullet + Franka Panda.
"""

import time
import argparse
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

# GraspNet imports
from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode

###############################################################################
# Robot helpers
###############################################################################

def open_gripper(robot):
    p.setJointMotorControl2(robot, 9, p.POSITION_CONTROL, 0.04, force=100)
    p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, 0.04, force=100)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1/240.)

def close_gripper(robot):
    p.setJointMotorControl2(robot, 9, p.POSITION_CONTROL, 0.0, force=200)
    p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, 0.0, force=200)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1/240.)

def move_ee(robot, ee_link, pos, quat, steps=240):
    joints = p.calculateInverseKinematics(
        robot,
        ee_link,
        targetPosition=pos,
        targetOrientation=quat,
        maxNumIterations=100,
        residualThreshold=1e-5
    )

    n = min(p.getNumJoints(robot), len(joints))
    for i in range(n):
        p.setJointMotorControl2(
            robot,
            i,
            p.POSITION_CONTROL,
            joints[i],
            force=500
        )

    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1/240.)

###############################################################################
# Vision & GraspNet helpers
###############################################################################

def get_camera_image_and_point_cloud():
    width = 640
    height = 480
    fov = 60
    aspect = width / height
    near = 0.02
    far = 5.0

    camera_pos = [1.0, 0, 0.6]
    target_pos = [0.5, 0, 0.0]
    up_vector = [0, 0, 1]

    view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    img_arr = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    depth_buffer = np.reshape(img_arr[3], (height, width))
    
    # Use NDC to World transformation
    view_mat = np.array(view_matrix).reshape(4, 4).T
    proj_mat = np.array(projection_matrix).reshape(4, 4).T
    cam_mat = proj_mat @ view_mat
    inv_cam_mat = np.linalg.inv(cam_mat)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x_ndc = (2.0 * x / width) - 1.0
    y_ndc = 1.0 - (2.0 * y / height)
    z_ndc = 2.0 * depth_buffer - 1.0

    pts_ndc = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(z_ndc)], axis=-1).reshape(-1, 4)
    pts_world = (inv_cam_mat @ pts_ndc.T).T
    points_world = pts_world[:, :3] / pts_world[:, 3:]

    # Filter background (keep only points roughly above the table and around the object)
    mask = (points_world[:, 2] > 0.01) & (points_world[:, 2] < 0.2) & \
           (points_world[:, 0] > 0.3) & (points_world[:, 0] < 0.7) & \
           (points_world[:, 1] > -0.3) & (points_world[:, 1] < 0.3)

    points_world_filtered = points_world[mask]

    return points_world_filtered

def init_graspnet(checkpoint_path):
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, device

def get_grasps(net, device, cloud_points, num_point=20000):
    if len(cloud_points) == 0:
        return None
        
    if len(cloud_points) >= num_point:
        idxs = np.random.choice(len(cloud_points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_points))
        idxs2 = np.random.choice(len(cloud_points), num_point - len(cloud_points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
        
    cloud_sampled = cloud_points[idxs]
    
    end_points = dict()
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points['point_clouds'] = cloud_sampled_tensor
    
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    
    # NMS and sort
    gg.nms()
    gg.sort_by_score()
    
    return gg

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='checkpoint-rs.tar', help='Path to graspnet checkpoint')
    args = parser.parse_args()

    print("Initializing PyBullet...")
    p.connect(p.GUI)
    p.setGravity(0,0,-9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Reset debug visualizer camera for better view
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0])

    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=True)
    table = p.loadURDF("table/table.urdf", [0.5,0,-0.65])
    box = p.loadURDF("tray/traybox.urdf", [0.65,-0.25,0])
    
    # Spawn an object to grasp
    cube = p.loadURDF("cube_small.urdf", [0.5, 0.0, 0.05])
    
    ee_link = 11

    # Let objects settle
    for _ in range(100):
        p.stepSimulation()

    print("Loading GraspNet model...")
    net, device = init_graspnet(args.checkpoint_path)

    print("Capturing point cloud from PyBullet...")
    pc = get_camera_image_and_point_cloud()
    print(f"Captured point cloud with {len(pc)} points in workspace.")

    if len(pc) > 0:
        print("Running GraspNet inference...")
        gg = get_grasps(net, device, pc)
        
        if gg is not None and len(gg) > 0:
            print(f"Generated {len(gg)} grasps. Executing best grasp...")
            best_grasp = gg[0]
            
            grasp_translation = best_grasp.translation
            grasp_rotation = best_grasp.rotation_matrix
            
            # The rotation matrix from GraspNet needs to be aligned with the Panda end effector.
            # GraspNet: approach is Z axis, close is X axis.
            # PyBullet Panda link 11: Z axis points forward (approach), X/Y depend on default alignment.
            # Usually GraspNet output matches well, but might need a 90 deg twist around Z.
            quat = Rotation.from_matrix(grasp_rotation).as_quat()

            # Pre-grasp (above the object)
            pre = grasp_translation.copy()
            # Move along the approach vector (Z axis of grasp_rotation)
            approach_vec = grasp_rotation[:, 0] # GraspNet approach is actually X? Let's assume Z for now or just move up.
            pre[2] += 0.15
            
            print("Moving to pre-grasp...")
            open_gripper(robot)
            move_ee(robot, ee_link, pre, quat, steps=240)

            # Move to grasp
            print("Moving to grasp...")
            move_ee(robot, ee_link, grasp_translation, quat, steps=240)

            # Close gripper
            print("Closing gripper...")
            close_gripper(robot)

            # Lift object
            print("Lifting object...")
            lift = grasp_translation.copy()
            lift[2] += 0.25
            move_ee(robot, ee_link, lift, quat, steps=240)

            # Move to place (tray)
            print("Moving to place in tray...")
            place = np.array([0.65, -0.25, 0.35])
            move_ee(robot, ee_link, place, quat, steps=240)

            # Open gripper
            print("Opening gripper to drop object...")
            open_gripper(robot)
            
            # Retreat
            retreat = place.copy()
            retreat[2] += 0.15
            move_ee(robot, ee_link, retreat, quat, steps=120)
            
        else:
            print("No valid grasp found.")
    else:
        print("No points detected in the workspace.")

    print("Simulation complete. Running physics...")
    while True:
        p.stepSimulation()
        time.sleep(1/240.)

if __name__ == "__main__":
    main()
