"""
Visualize GraspNet pipeline internals as 2D charts/heatmaps (matplotlib).
Complements test_single_image.py (Open3D 3D vis) with non-3D data.

Usage:
  python visualize_charts.py --checkpoint_path logs/log_realsense/checkpoint-rs.tar
  python visualize_charts.py --checkpoint_path logs/log_realsense/checkpoint-rs.tar --dataset_root "path/to/dataset" --scene_id 100 --ann_id 0
  python visualize_charts.py --checkpoint_path ... --save_dir charts_output
"""

import os, sys, argparse, time
import numpy as np
import scipy.io as scio
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True)
parser.add_argument('--data_dir', default=None)
parser.add_argument('--dataset_root', default=None)
parser.add_argument('--scene_id', type=int, default=100)
parser.add_argument('--ann_id', type=int, default=0)
parser.add_argument('--camera', default='realsense', choices=['realsense', 'kinect'])
parser.add_argument('--num_point', type=int, default=20000)
parser.add_argument('--num_view', type=int, default=300)
parser.add_argument('--collision_thresh', type=float, default=0.01)
parser.add_argument('--voxel_size', type=float, default=0.01)
parser.add_argument('--save_dir', default=None, help='Save charts as PNG instead of showing')
parser.add_argument('--seed_idx', type=int, default=-1, help='Seed index for per-seed heatmap (-1=best)')
cfgs = parser.parse_args()

fig_count = [0]

def save_or_show(fig, name):
    fig_count[0] += 1
    if cfgs.save_dir:
        os.makedirs(cfgs.save_dir, exist_ok=True)
        path = os.path.join(cfgs.save_dir, f'{fig_count[0]:02d}_{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  [SAVED] {path}')
        plt.close(fig)
    else:
        plt.show()


# ============ CHART 1: Input RGB + Depth + Mask ============
def chart_input_images(color, depth, workspace_mask):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CHART 1: Input Images', fontsize=14, fontweight='bold')
    axes[0].imshow(color); axes[0].set_title('RGB'); axes[0].axis('off')
    im = axes[1].imshow(depth, cmap='plasma'); axes[1].set_title('Depth'); axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(workspace_mask, cmap='gray'); axes[2].set_title('Workspace Mask'); axes[2].axis('off')
    fig.tight_layout()
    save_or_show(fig, 'input_images')


# ============ CHART 2: Point count through backbone ============
def chart_backbone_reduction(end_points):
    stages = ['Input\n(N)', 'SA1\n(2048)', 'SA2\n(1024)', 'SA3\n(512)', 'SA4\n(256)', 'FP2 Seeds\n(1024)']
    counts = [
        end_points['point_clouds'].shape[1],
        end_points['sa1_xyz'].shape[1], end_points['sa2_xyz'].shape[1],
        end_points['sa3_xyz'].shape[1], end_points['sa4_xyz'].shape[1],
        end_points['fp2_xyz'].shape[1]
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('CHART 2: PointNet++ Backbone - Point Count per Layer', fontsize=14, fontweight='bold')
    colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0','#00BCD4']
    bars = ax.bar(stages, counts, color=colors, edgecolor='black', linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, str(c),
                ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel('Number of Points')
    ax.set_ylim(0, max(counts)*1.15)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, 'backbone_reduction')


# ============ CHART 3: Objectness distribution ============
def chart_objectness(end_points):
    logits = end_points['objectness_score'][0]
    prob = torch.softmax(logits, dim=0)[1].cpu().numpy()
    pred = (prob > 0.5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('CHART 3: Objectness Score Analysis', fontsize=14, fontweight='bold')

    axes[0].hist(prob, bins=50, color='#2196F3', edgecolor='black', alpha=0.8)
    axes[0].axvline(0.5, color='red', linestyle='--', label='Threshold=0.5')
    axes[0].set_title('Objectness Probability Distribution')
    axes[0].set_xlabel('P(object)'); axes[0].set_ylabel('Count'); axes[0].legend()

    labels = ['Non-Object', 'Object']
    sizes = [len(pred)-pred.sum(), pred.sum()]
    axes[1].pie(sizes, labels=labels, colors=['#2196F3','#F44336'], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize':12})
    axes[1].set_title(f'Object vs Non-Object Seeds')

    sorted_prob = np.sort(prob)[::-1]
    axes[2].plot(sorted_prob, color='#F44336', linewidth=2)
    axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].fill_between(range(len(sorted_prob)), sorted_prob, alpha=0.2, color='#F44336')
    axes[2].set_title('Objectness Scores (sorted)')
    axes[2].set_xlabel('Seed Index (ranked)'); axes[2].set_ylabel('Score')

    fig.tight_layout()
    save_or_show(fig, 'objectness')


# ============ CHART 4: View scores ============
def chart_view_scores(end_points):
    view_score = end_points['grasp_top_view_score'][0].cpu().numpy()
    view_inds = end_points['grasp_top_view_inds'][0].cpu().numpy()
    all_view_scores = end_points['view_score'][0].cpu().numpy()  # (M, 300)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CHART 4: Approach View Score Analysis', fontsize=14, fontweight='bold')

    axes[0].hist(view_score, bins=50, color='#4CAF50', edgecolor='black', alpha=0.8)
    axes[0].set_title('Top View Score Distribution')
    axes[0].set_xlabel('Score'); axes[0].set_ylabel('Count')

    axes[1].hist(view_inds, bins=100, color='#FF9800', edgecolor='black', alpha=0.8)
    axes[1].set_title('Selected View Bin Histogram')
    axes[1].set_xlabel('View Bin ID (0..299)'); axes[1].set_ylabel('Count')

    mean_per_view = all_view_scores.mean(axis=0)
    axes[2].plot(mean_per_view, color='#9C27B0', linewidth=1)
    axes[2].set_title('Mean Score per View Direction')
    axes[2].set_xlabel('View Bin ID'); axes[2].set_ylabel('Mean Score')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    save_or_show(fig, 'view_scores')


# ============ CHART 5: Feature PCA ============
def chart_feature_pca(end_points):
    features = end_points['fp2_features'][0].cpu().numpy().T  # (1024, 256)
    x = features - features.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    explained = (s**2) / (s**2).sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CHART 5: Seed Feature Analysis (fp2_features, 256-dim)', fontsize=14, fontweight='bold')

    axes[0].bar(range(min(20, len(explained))), explained[:20]*100, color='#00BCD4', edgecolor='black')
    axes[0].set_title('PCA Explained Variance (top 20)')
    axes[0].set_xlabel('Component'); axes[0].set_ylabel('Variance (%)')

    proj = x @ vt[:2].T
    axes[1].scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.6, c='#2196F3')
    axes[1].set_title('Seed Points in PCA Space (PC1 vs PC2)')
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].grid(alpha=0.3)

    corr = np.corrcoef(features.T)
    im = axes[2].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_title('Feature Correlation Matrix (256x256)')
    axes[2].set_xlabel('Feature Dim'); axes[2].set_ylabel('Feature Dim')
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    save_or_show(fig, 'feature_pca')


# ============ CHART 6: Grasp Score/Width/Tolerance Heatmaps (angle x depth) ============
def chart_grasp_heatmaps(end_points, seed_idx=-1):
    num_seed = end_points['grasp_score_pred'].shape[2]
    if seed_idx < 0:
        seed_idx = torch.argmax(end_points['grasp_top_view_score'][0]).item()
    seed_idx = min(seed_idx, num_seed - 1)

    score = end_points['grasp_score_pred'][0, :, seed_idx, :].cpu().numpy()    # (12, 4)
    angle_cls = end_points['grasp_angle_cls_pred'][0, :, seed_idx, :].cpu().numpy()
    width = end_points['grasp_width_pred'][0, :, seed_idx, :].cpu().numpy()
    tol = end_points['grasp_tolerance_pred'][0, :, seed_idx, :].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f'CHART 6: Per-Seed Prediction Heatmaps (seed={seed_idx}, best view score)',
                 fontsize=14, fontweight='bold')

    data = [(score, 'Grasp Score', 'viridis'), (angle_cls, 'Angle Class Logits', 'coolwarm'),
            (width, 'Grasp Width', 'YlOrRd'), (tol, 'Tolerance', 'PuBuGn')]
    depth_labels = ['0.01m', '0.02m', '0.03m', '0.04m']
    angle_labels = [f'{i*15}' for i in range(12)]

    for ax, (d, title, cmap) in zip(axes, data):
        im = ax.imshow(d, cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Depth Bin'); ax.set_ylabel('Angle Bin (deg)')
        ax.set_xticks(range(4)); ax.set_xticklabels(depth_labels)
        ax.set_yticks(range(12)); ax.set_yticklabels(angle_labels)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    save_or_show(fig, f'grasp_heatmaps_seed{seed_idx}')


# ============ CHART 7: Global grasp score statistics ============
def chart_grasp_global_stats(end_points):
    score = end_points['grasp_score_pred'][0].cpu().numpy()   # (12, 1024, 4)
    width = end_points['grasp_width_pred'][0].cpu().numpy()
    tol = end_points['grasp_tolerance_pred'][0].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CHART 7: Global Grasp Prediction Statistics (all seeds)', fontsize=14, fontweight='bold')

    axes[0,0].hist(score.flatten(), bins=80, color='#2196F3', edgecolor='black', alpha=0.8)
    axes[0,0].set_title('Grasp Score Distribution'); axes[0,0].set_xlabel('Score')

    axes[0,1].hist(width.flatten(), bins=80, color='#FF9800', edgecolor='black', alpha=0.8)
    axes[0,1].set_title('Grasp Width Distribution'); axes[0,1].set_xlabel('Width (m)')

    axes[0,2].hist(tol.flatten(), bins=80, color='#4CAF50', edgecolor='black', alpha=0.8)
    axes[0,2].set_title('Tolerance Distribution'); axes[0,2].set_xlabel('Tolerance')

    mean_score = score.mean(axis=1)  # (12, 4)
    im1 = axes[1,0].imshow(mean_score, cmap='viridis', aspect='auto')
    axes[1,0].set_title('Mean Score (angle x depth)'); axes[1,0].set_xlabel('Depth'); axes[1,0].set_ylabel('Angle')
    fig.colorbar(im1, ax=axes[1,0], fraction=0.046)

    max_score = score.max(axis=1)
    im2 = axes[1,1].imshow(max_score, cmap='hot', aspect='auto')
    axes[1,1].set_title('Max Score (angle x depth)'); axes[1,1].set_xlabel('Depth'); axes[1,1].set_ylabel('Angle')
    fig.colorbar(im2, ax=axes[1,1], fraction=0.046)

    max_per_seed = score.max(axis=(0, 2))
    axes[1,2].plot(np.sort(max_per_seed)[::-1], color='#F44336', linewidth=2)
    axes[1,2].set_title('Max Score per Seed (ranked)')
    axes[1,2].set_xlabel('Seed Rank'); axes[1,2].set_ylabel('Max Score')
    axes[1,2].grid(alpha=0.3)

    fig.tight_layout()
    save_or_show(fig, 'grasp_global_stats')


# ============ CHART 8: Decoded grasp analysis ============
def chart_decoded_grasps(gg_raw, gg_nms, gg_final):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CHART 8: Decoded Grasp Analysis (Raw → NMS → Collision)', fontsize=14, fontweight='bold')

    stages = [('Raw', gg_raw, '#2196F3'), ('After NMS', gg_nms, '#FF9800'), ('After Collision', gg_final, '#4CAF50')]
    for ax, (label, gg, c) in zip(axes[0], stages):
        if len(gg) > 0:
            ax.hist(gg.scores, bins=40, color=c, edgecolor='black', alpha=0.8)
        ax.set_title(f'{label}: {len(gg)} grasps')
        ax.set_xlabel('Score'); ax.set_ylabel('Count')

    for ax, (label, gg, c) in zip(axes[1][:2], stages[:2]):
        if len(gg) > 0:
            ax.scatter(gg.scores, gg.widths, s=10, alpha=0.5, c=c)
        ax.set_title(f'{label}: Score vs Width')
        ax.set_xlabel('Score'); ax.set_ylabel('Width (m)')

    # Pipeline funnel
    labels = ['Raw', 'NMS', 'Collision']
    counts = [len(gg_raw), len(gg_nms), len(gg_final)]
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = axes[1,2].barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor='black')
    for bar, c in zip(bars, counts[::-1]):
        axes[1,2].text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(c),
                       va='center', fontweight='bold')
    axes[1,2].set_title('Pipeline Funnel')
    axes[1,2].set_xlabel('Number of Grasps')

    fig.tight_layout()
    save_or_show(fig, 'decoded_grasps')


# ============ CHART 9: Full pipeline summary ============
def chart_pipeline_summary(counts_dict, total_time):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('CHART 9: Pipeline Summary', fontsize=14, fontweight='bold')

    gs = GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    colors = ['#2196F3','#03A9F4','#00BCD4','#4CAF50','#8BC34A','#FF9800','#F44336']
    colors = colors[:len(labels)]

    bars = ax1.bar(labels, values, color=colors, edgecolor='black')
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values)*0.02,
                 str(v), ha='center', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Count')
    ax1.set_title('Point/Grasp Count at Each Stage')
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(axis='y', alpha=0.3)

    cell_text = [[str(v)] for v in values]
    cell_text.append([f'{total_time:.1f}s'])
    row_labels = labels + ['Total Time']
    table = ax2.table(cellText=cell_text, rowLabels=row_labels, colLabels=['Value'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.5)
    ax2.axis('off'); ax2.set_title('Summary Table')

    fig.tight_layout()
    save_or_show(fig, 'pipeline_summary')


# ============ MAIN ============
def main():
    total_start = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- Load data ---
    if cfgs.data_dir:
        data_dir = cfgs.data_dir
    elif cfgs.dataset_root:
        data_dir = None
    else:
        data_dir = os.path.join(ROOT_DIR, 'doc', 'example_data')

    if data_dir:
        color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']; factor_depth = meta['factor_depth']
        cam = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, cam, organized=True)
        mask = (workspace_mask & (depth > 0))
    else:
        scene_dir = os.path.join(cfgs.dataset_root, 'scenes', f'scene_{cfgs.scene_id:04d}', cfgs.camera)
        color = np.array(Image.open(os.path.join(scene_dir, 'rgb', f'{cfgs.ann_id:04d}.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(scene_dir, 'depth', f'{cfgs.ann_id:04d}.png')))
        meta = scio.loadmat(os.path.join(scene_dir, 'meta', f'{cfgs.ann_id:04d}.mat'))
        intrinsic = meta['intrinsic_matrix']; factor_depth = meta['factor_depth']
        h, w = depth.shape[:2]
        cam = CameraInfo(float(w), float(h), intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, cam, organized=True)
        seg_path = os.path.join(scene_dir, 'label', f'{cfgs.ann_id:04d}.png')
        if os.path.exists(seg_path):
            seg = np.array(Image.open(seg_path))
            from data_utils import get_workspace_mask
            workspace_mask = get_workspace_mask(cloud, seg, organized=True, outlier=0.02)
            mask = workspace_mask & (depth > 0)
        else:
            mask = (depth > 0)
            workspace_mask = mask.astype(np.uint8) * 255

    cloud_masked = cloud[mask]; color_masked = color[mask]

    print(f'Input points: {len(cloud_masked)}')
    chart_input_images(color, depth, workspace_mask)

    # --- Sample ---
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs = np.concatenate([np.arange(len(cloud_masked)),
                               np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)])
    cloud_sampled = cloud_masked[idxs]
    end_points = {'point_clouds': torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device),
                  'cloud_colors': color_masked[idxs]}

    # --- Model ---
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    net.to(device)
    ckpt = torch.load(cfgs.checkpoint_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict']); net.eval()
    print(f'Model loaded (epoch {ckpt["epoch"]})')

    # --- Forward ---
    with torch.no_grad():
        end_points = net.view_estimator(end_points)
    chart_backbone_reduction(end_points)
    chart_objectness(end_points)
    chart_view_scores(end_points)
    chart_feature_pca(end_points)

    with torch.no_grad():
        end_points = net.grasp_generator(end_points)
    chart_grasp_heatmaps(end_points, seed_idx=cfgs.seed_idx)
    chart_grasp_global_stats(end_points)

    # --- Decode + NMS + Collision ---
    with torch.no_grad():
        grasp_preds = pred_decode(end_points)
    gg_raw = GraspGroup(grasp_preds[0].cpu().numpy())
    gg_nms = gg_raw.nms().sort_by_score()

    if cfgs.collision_thresh > 0:
        import open3d as o3d
        det = ModelFreeCollisionDetector(cloud_masked.astype(np.float32), voxel_size=cfgs.voxel_size)
        cmask = det.detect(gg_nms, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg_final = gg_nms[~cmask]
    else:
        gg_final = gg_nms

    chart_decoded_grasps(gg_raw, gg_nms, gg_final)

    total_time = time.time() - total_start
    counts = {'Input': len(cloud_masked), 'Sampled': cfgs.num_point, 'Seeds': end_points['fp2_xyz'].shape[1],
              'Object Seeds': int((torch.softmax(end_points['objectness_score'][0], dim=0)[1] > 0.5).sum().item()),
              'Raw Grasps': len(gg_raw), 'After NMS': len(gg_nms), 'Final': len(gg_final)}
    chart_pipeline_summary(counts, total_time)
    print(f'\nDone! Total: {total_time:.1f}s, Generated {fig_count[0]} charts.')

if __name__ == '__main__':
    main()
