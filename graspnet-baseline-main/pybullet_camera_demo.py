
import time
import cv2
import numpy as np
import pybullet as p
import pybullet_data

WIDTH, HEIGHT = 640, 480

def depth_buffer_to_meters(depth, near, far):
    return far * near / (far - (far - near) * depth)

def projection_to_intrinsics(fov_deg, width, height):
    fov = np.deg2rad(fov_deg)
    fy = height / (2 * np.tan(fov / 2))
    fx = fy
    cx = width / 2
    cy = height / 2
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

plane = p.loadURDF("plane.urdf")
table = p.loadURDF("table/table.urdf",[0.55,0,-0.65])
robot = p.loadURDF("franka_panda/panda.urdf",[0,0,0],useFixedBase=True)
tray = p.loadURDF("tray/traybox.urdf",[0.8,-0.25,0])

# Spawn objects
for i in range(4):
    p.loadURDF("cube_small.urdf",
               [0.45+0.05*i, (-1)**i*0.05, 0.03])

cam_eye=[0.55,0.0,0.8]
cam_target=[0.55,0.0,0.0]
cam_up=[0,1,0]

fov=60
near=0.02
far=2.0

K=projection_to_intrinsics(fov,WIDTH,HEIGHT)
print("Camera Intrinsics K:\n",K)

cv2.namedWindow("RGB",cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth",cv2.WINDOW_NORMAL)

while True:

    view = p.computeViewMatrix(
        cam_eye,
        cam_target,
        cam_up
    )

    proj = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=WIDTH/HEIGHT,
        nearVal=near,
        farVal=far
    )

    _,_,rgba,depth,seg = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.reshape(rgba, (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    depth=np.reshape(depth,(HEIGHT,WIDTH))
    depth_m=depth_buffer_to_meters(depth,near,far)
    depth_vis=cv2.normalize(depth_m,None,0,255,cv2.NORM_MINMAX)
    depth_vis=depth_vis.astype(np.uint8)
    depth_vis=cv2.applyColorMap(depth_vis,cv2.COLORMAP_JET)

    cv2.imshow("RGB",rgb)
    cv2.imshow("Depth",depth_vis)

    key=cv2.waitKey(1)&0xff
    if key==27 or key==ord('q'):
        break
    elif key==ord('s'):
        cv2.imwrite("rgb.png",rgb)
        np.save("depth.npy",depth_m)
        np.save("camK.npy",K)
        print("Saved rgb.png depth.npy camK.npy")

    p.stepSimulation()
    time.sleep(1/240)

cv2.destroyAllWindows()
p.disconnect()
