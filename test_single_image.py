"""
Test a single image through the full GraspNet paper pipeline.

Pipeline theo bài báo GraspNet-1Billion (CVPR 2020):
  1. Input Processing  : RGB-D → Point Cloud → Sample N points
  2. Backbone (PointNet++): Point cloud → Seed features (SA1→SA2→SA3→SA4→FP1→FP2)
  3. Approach Net       : Seed features → Objectness + Approach vectors (view scores)
  4. Cloud Crop         : Cylinder grouping at multiple depths
  5. Operation Net      : Grasp score, angle class, width prediction
  6. Tolerance Net      : Grasp tolerance prediction
  7. Grasp Decode       : Combine predictions → GraspGroup
  8. NMS + Sort         : Non-Maximum Suppression → Sort by score
  9. Collision Detection: Model-free collision check → Final grasps
  10. Visualization      : Open3D rendering

Usage:
  # Test with example data (default)
  python test_single_image.py --checkpoint_path logs/log_realsense/checkpoint-rs.tar

  # Test with GraspNet dataset scene/frame
  python test_single_image.py --checkpoint_path logs/log_realsense/checkpoint-rs.tar --dataset_root /path/to/graspnet --scene_id 100 --ann_id 0 --camera realsense

  # Test with custom RGB-D folder
  python test_single_image.py --checkpoint_path logs/log_realsense/checkpoint-rs.tar --data_dir doc/example_data

  # Skip visualization (console output only)
  python test_single_image.py --checkpoint_path logs/log_realsense/checkpoint-rs.tar --no_vis

  # Save results to file
  python test_single_image.py --checkpoint_path logs/log_realsense/checkpoint-rs.tar --save_dir results/
"""

import os
import sys
import time
import argparse
import numpy as np
import open3d as o3d
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


# ======================== ARGPARSE ========================
parser = argparse.ArgumentParser(description='Test single image through GraspNet pipeline')
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path (.tar)')

# Input source (choose one)
parser.add_argument('--data_dir', default=None, help='Custom RGB-D folder (color.png, depth.png, workspace_mask.png, meta.mat)')
parser.add_argument('--dataset_root', default=None, help='GraspNet dataset root path')
parser.add_argument('--scene_id', type=int, default=100, help='Scene ID in dataset [default: 100]')
parser.add_argument('--ann_id', type=int, default=0, help='Annotation/frame ID [default: 0]')
parser.add_argument('--camera', default='realsense', choices=['realsense', 'kinect'], help='Camera type [default: realsense]')

# Model params
parser.add_argument('--num_point', type=int, default=20000, help='Number of sampled points [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='Number of views [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision threshold [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size for collision detection [default: 0.01]')

# Output controls
parser.add_argument('--top_k', type=int, default=50, help='Number of top grasps to visualize [default: 50]')
parser.add_argument('--no_vis', action='store_true', help='Skip all visualization')
parser.add_argument('--save_dir', default=None, help='Directory to save grasp results (.npy)')

cfgs = parser.parse_args()


# ======================== HELPERS ========================
def print_header(stage_num, title):
    print(f'\n{"="*60}')
    print(f'  STAGE {stage_num}: {title}')
    print(f'{"="*60}')


def print_tensor_info(name, t):
    if isinstance(t, torch.Tensor):
        print(f'  {name:30s} shape={str(list(t.shape)):20s} dtype={t.dtype}  device={t.device}')
    elif isinstance(t, np.ndarray):
        print(f'  {name:30s} shape={str(list(t.shape)):20s} dtype={t.dtype}')
    else:
        print(f'  {name:30s} type={type(t).__name__}')


def score_to_rgb(scores):
    """Blue(low) → Red(high) colormap."""
    s = scores.astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    colors = np.zeros((len(s), 3), dtype=np.float32)
    colors[:, 0] = s          # red channel
    colors[:, 2] = 1.0 - s    # blue channel
    return colors


def make_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float32), 0, 1))
    return pcd


def open3d_show(geometries, title):
    if cfgs.no_vis:
        print(f'  [VIS SKIP] {title}')
        return
    print(f'  [VIS] {title}  (close window to continue)')
    o3d.visualization.draw_geometries(geometries, window_name=title, width=1280, height=720)


# ======================== DATA LOADING ========================
def load_from_custom_dir(data_dir):
    """Load RGB-D data from a custom folder (like doc/example_data)."""
    print(f'  Source: custom folder -> {data_dir}')

    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))

    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1],
                        intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    return cloud_masked, color_masked


def load_from_dataset(dataset_root, scene_id, ann_id, camera):
    """Load RGB-D data from GraspNet dataset structure."""
    scene_name = f'scene_{scene_id:04d}'
    print(f'  Source: GraspNet dataset')
    print(f'  Scene: {scene_name}  Frame: {ann_id:04d}  Camera: {camera}')

    scene_dir = os.path.join(dataset_root, 'scenes', scene_name, camera)
    color_path = os.path.join(scene_dir, 'rgb', f'{ann_id:04d}.png')
    depth_path = os.path.join(scene_dir, 'depth', f'{ann_id:04d}.png')
    meta_path = os.path.join(scene_dir, 'meta', f'{ann_id:04d}.mat')

    if not os.path.exists(color_path):
        raise FileNotFoundError(f'Color image not found: {color_path}')

    color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    meta = scio.loadmat(meta_path)

    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # Camera resolution
    h, w = depth.shape[:2]
    camera_info = CameraInfo(float(w), float(h), intrinsic[0][0], intrinsic[1][1],
                             intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    # Workspace mask: use segmentation or depth > 0
    seg_path = os.path.join(scene_dir, 'label', f'{ann_id:04d}.png')
    if os.path.exists(seg_path):
        seg = np.array(Image.open(seg_path))
        # Keep object regions + some background context
        workspace_mask = (depth > 0)
        # Use segmentation to focus on objects
        from data_utils import get_workspace_mask
        workspace_mask = get_workspace_mask(cloud, seg, organized=True, outlier=0.02)
        mask = workspace_mask & (depth > 0)
    else:
        mask = (depth > 0)

    cloud_masked = cloud[mask]
    color_masked = color[mask]

    return cloud_masked, color_masked


# ======================== PIPELINE ========================
def run_pipeline():
    total_start = time.time()

    # -------- STAGE 1: Load & Process Input --------
    print_header(1, 'INPUT PROCESSING')
    t0 = time.time()

    # Determine data source
    if cfgs.data_dir is not None:
        cloud_masked, color_masked = load_from_custom_dir(cfgs.data_dir)
    elif cfgs.dataset_root is not None:
        cloud_masked, color_masked = load_from_dataset(
            cfgs.dataset_root, cfgs.scene_id, cfgs.ann_id, cfgs.camera)
    else:
        # Default: use example data
        default_dir = os.path.join(ROOT_DIR, 'doc', 'example_data')
        print(f'  No --data_dir or --dataset_root specified, using default: {default_dir}')
        cloud_masked, color_masked = load_from_custom_dir(default_dir)

    print(f'  Masked points: {len(cloud_masked)}')

    # Sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # Build Open3D cloud for visualization
    cloud_o3d = make_pcd(cloud_masked, color_masked)

    # Build network input
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': cloud_tensor, 'cloud_colors': color_sampled}

    print(f'  Sampled points: {cfgs.num_point}')
    print(f'  Device: {device}')
    print(f'  Time: {time.time()-t0:.3f}s')

    # Vis: input point cloud
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    open3d_show([cloud_o3d, frame], 'Stage 1 - Input Point Cloud')

    # -------- STAGE 2: Load Model --------
    print_header(2, 'LOAD MODEL')
    t0 = time.time()

    net = GraspNet(
        input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    net.to(device)

    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f'  Checkpoint: {cfgs.checkpoint_path}')
    print(f'  Epoch: {checkpoint["epoch"]}')
    net.eval()
    print(f'  Time: {time.time()-t0:.3f}s')

    # -------- STAGE 3: PointNet++ Backbone (Feature Extraction) --------
    print_header(3, 'POINTNET++ BACKBONE')
    t0 = time.time()

    with torch.no_grad():
        # Run Stage 1 of GraspNet (backbone + approach net)
        end_points = net.view_estimator(end_points)

    print('  Backbone outputs (Set Abstraction + Feature Propagation):')
    for key in ['sa1_xyz', 'sa2_xyz', 'sa3_xyz', 'sa4_xyz', 'fp2_xyz', 'fp2_features']:
        if key in end_points:
            print_tensor_info(key, end_points[key])

    seed_xyz = end_points['fp2_xyz'][0].cpu().numpy()
    print(f'  Seed points (fp2): {seed_xyz.shape[0]}')
    print(f'  Time: {time.time()-t0:.3f}s')

    # Vis: seed points
    seed_pcd = make_pcd(seed_xyz)
    seed_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    open3d_show([cloud_o3d, seed_pcd, frame], 'Stage 3 - Seed Points (fp2_xyz, red)')

    # -------- STAGE 4: Objectness & Approach Vectors --------
    print_header(4, 'OBJECTNESS & APPROACH VECTORS')

    # Objectness
    obj_logits = end_points['objectness_score'][0]  # (2, M)
    obj_prob = torch.softmax(obj_logits, dim=0)[1].cpu().numpy()  # P(object)
    obj_pred = (obj_prob > 0.5).astype(int)

    print(f'  Objectness prob  min={obj_prob.min():.4f}  max={obj_prob.max():.4f}  mean={obj_prob.mean():.4f}')
    print(f'  Object seeds: {obj_pred.sum()} / {len(obj_pred)}')

    # Approach vectors
    view_scores = end_points['grasp_top_view_score'][0].cpu().numpy()
    print(f'  View scores  min={view_scores.min():.4f}  max={view_scores.max():.4f}')

    # Vis: objectness - chỉ seed points, 2 màu: đỏ = object, xanh = non-object
    obj_binary_colors = np.zeros((len(obj_pred), 3), dtype=np.float32)
    obj_binary_colors[obj_pred == 1] = [1.0, 0.0, 0.0]   # đỏ = object
    obj_binary_colors[obj_pred == 0] = [0.0, 0.0, 1.0]    # xanh = non-object
    obj_pcd = make_pcd(seed_xyz, obj_binary_colors)
    open3d_show([obj_pcd, frame],
                f'Stage 4 - Objectness (Red={obj_pred.sum()} object | Blue={len(obj_pred)-obj_pred.sum()} non-object)')

    # Vis: approach vectors
    if not cfgs.no_vis:
        top_view_xyz = end_points['grasp_top_view_xyz'][0].cpu().numpy()
        top_view_rot = end_points['grasp_top_view_rot'][0].cpu().numpy()
        n_arrows = min(200, len(seed_xyz))
        top_idxs = np.argsort(-view_scores)[:n_arrows]

        line_pts, line_idx, line_cls = [], [], []
        for li, i in enumerate(top_idxs):
            center = seed_xyz[i]
            direction = top_view_rot[i][:, 0]
            end_pt = center + direction * 0.03
            line_pts += [center.tolist(), end_pt.tolist()]
            line_idx.append([2*li, 2*li+1])
            line_cls.append(score_to_rgb(view_scores[top_idxs])[li].tolist())

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(line_pts)
        lines.lines = o3d.utility.Vector2iVector(line_idx)
        lines.colors = o3d.utility.Vector3dVector(line_cls)
        open3d_show([cloud_o3d, lines, frame],
                    'Stage 4 - Approach Vectors (top 200)')

    # -------- STAGE 5: Grasp Generation (CloudCrop + OperationNet + ToleranceNet) --------
    print_header(5, 'GRASP GENERATION (CloudCrop + OperationNet + ToleranceNet)')
    t0 = time.time()

    with torch.no_grad():
        end_points = net.grasp_generator(end_points)

    print('  Grasp generation outputs:')
    for key in ['grasp_score_pred', 'grasp_angle_cls_pred', 'grasp_width_pred', 'grasp_tolerance_pred']:
        if key in end_points:
            print_tensor_info(key, end_points[key])

    grasp_score = end_points['grasp_score_pred'][0].cpu().numpy()
    grasp_width = end_points['grasp_width_pred'][0].cpu().numpy()
    grasp_tol = end_points['grasp_tolerance_pred'][0].cpu().numpy()

    print(f'  Grasp score    min={grasp_score.min():.4f}  max={grasp_score.max():.4f}')
    print(f'  Grasp width    min={grasp_width.min():.4f}  max={grasp_width.max():.4f}')
    print(f'  Grasp tolerance min={grasp_tol.min():.4f}  max={grasp_tol.max():.4f}')
    print(f'  Time: {time.time()-t0:.3f}s')

    # -------- STAGE 6: Grasp Decode --------
    print_header(6, 'GRASP DECODE')
    t0 = time.time()

    with torch.no_grad():
        grasp_preds = pred_decode(end_points)

    gg_array = grasp_preds[0].cpu().numpy()
    gg_raw = GraspGroup(gg_array)

    print(f'  Raw decoded grasps: {len(gg_raw)}')
    if len(gg_raw) > 0:
        raw_scores = gg_raw.scores
        print(f'  Score  min={raw_scores.min():.4f}  max={raw_scores.max():.4f}  mean={raw_scores.mean():.4f}')
        raw_widths = gg_raw.widths
        print(f'  Width  min={raw_widths.min():.4f}  max={raw_widths.max():.4f}')
    print(f'  Time: {time.time()-t0:.3f}s')

    # Vis: raw grasps (top-k)
    if len(gg_raw) > 0:
        gg_raw_vis = gg_raw.sort_by_score()[:min(cfgs.top_k, len(gg_raw))]
        grippers = gg_raw_vis.to_open3d_geometry_list()
        open3d_show([cloud_o3d, frame, *grippers],
                    f'Stage 6 - Raw Decoded Grasps (top {len(gg_raw_vis)})')

    # -------- STAGE 7: NMS --------
    print_header(7, 'NON-MAXIMUM SUPPRESSION (NMS)')
    t0 = time.time()

    gg_nms = gg_raw.nms()
    print(f'  Before NMS: {len(gg_raw)}')
    print(f'  After NMS:  {len(gg_nms)}')
    print(f'  Removed:    {len(gg_raw) - len(gg_nms)}')
    print(f'  Time: {time.time()-t0:.3f}s')

    # -------- STAGE 8: Sort by Score --------
    print_header(8, 'SORT BY SCORE')

    gg_sorted = gg_nms.sort_by_score()
    if len(gg_sorted) > 0:
        print(f'  Top 10 scores: {gg_sorted.scores[:10]}')

    # Vis: after NMS + sort
    if len(gg_sorted) > 0:
        gg_nms_vis = gg_sorted[:min(cfgs.top_k, len(gg_sorted))]
        grippers = gg_nms_vis.to_open3d_geometry_list()
        open3d_show([cloud_o3d, frame, *grippers],
                    f'Stage 8 - After NMS + Sort (top {len(gg_nms_vis)})')

    # -------- STAGE 9: Collision Detection --------
    print_header(9, 'COLLISION DETECTION')
    t0 = time.time()

    if cfgs.collision_thresh > 0:
        cloud_points = np.asarray(cloud_o3d.points)
        mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=cfgs.voxel_size)
        collision_mask = mfcdetector.detect(
            gg_sorted, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg_final = gg_sorted[~collision_mask]
        print(f'  Before collision: {len(gg_sorted)}')
        print(f'  Colliding grasps: {int(collision_mask.sum())}')
        print(f'  After collision:  {len(gg_final)}')
    else:
        gg_final = gg_sorted
        print('  Collision detection SKIPPED (threshold <= 0)')
    print(f'  Time: {time.time()-t0:.3f}s')

    # -------- STAGE 10: Final Visualization --------
    print_header(10, 'FINAL RESULTS')

    total_time = time.time() - total_start
    print(f'  Final grasps: {len(gg_final)}')
    if len(gg_final) > 0:
        print(f'  Best score:   {gg_final.scores[0]:.4f}')
        print(f'  Worst score:  {gg_final.scores[-1]:.4f}')
    print(f'  Total time:   {total_time:.3f}s')

    # Save results
    if cfgs.save_dir is not None:
        os.makedirs(cfgs.save_dir, exist_ok=True)
        save_path = os.path.join(cfgs.save_dir, 'grasp_results.npy')
        gg_final.save_npy(save_path)
        print(f'  Saved to: {save_path}')

    # Vis: final grasps
    if len(gg_final) > 0:
        gg_final_vis = gg_final[:min(cfgs.top_k, len(gg_final))]
        grippers = gg_final_vis.to_open3d_geometry_list()
        open3d_show([cloud_o3d, frame, *grippers],
                    f'Stage 10 - FINAL Grasps (top {len(gg_final_vis)})')
    else:
        print('  WARNING: No valid grasps after collision detection!')
        open3d_show([cloud_o3d, frame], 'Stage 10 - No Valid Grasps')

    # -------- SUMMARY --------
    print(f'\n{"="*60}')
    print('  PIPELINE SUMMARY')
    print(f'{"="*60}')
    print(f'  Input points:      {len(cloud_masked)}')
    print(f'  Sampled points:    {cfgs.num_point}')
    print(f'  Seed points:       {seed_xyz.shape[0]}')
    print(f'  Object seeds:      {obj_pred.sum()}')
    print(f'  Raw grasps:        {len(gg_raw)}')
    print(f'  After NMS:         {len(gg_nms)}')
    print(f'  After collision:   {len(gg_final)}')
    print(f'  Total time:        {total_time:.3f}s')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    run_pipeline()
