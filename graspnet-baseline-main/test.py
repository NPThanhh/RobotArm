""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset, collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=30, help='Number of workers used in evaluation [default: 30]')
parser.add_argument('--split', default='test', choices=['test', 'test_seen', 'test_similar', 'test_novel'], help='Dataset split to evaluate')
parser.add_argument('--skip_inference', action='store_true', help='Skip inference and only run evaluation')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split=cfgs.split, camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False)

print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TEST_DATALOADER))
# Init the model
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                     cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def inference():
    batch_interval = 100
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[data_idx], cfgs.camera)
            frame_id = TEST_DATASET.frameid[data_idx]
            save_path = os.path.join(save_dir, str(frame_id).zfill(4)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs'%(batch_idx, (toc-tic)/batch_interval))
            tic = time.time()

def evaluate():
    import json
    
    print(f'\n================= GENERATING JSON EVALUATION =================')
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    
    json_path = os.path.join(cfgs.dump_dir, 'eval_results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            eval_data = json.load(f)
    else:
        eval_data = {
            "camera": cfgs.camera,
            "checkpoint": cfgs.checkpoint_path,
            "scenes": {}
        }
    
    frictions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    
    if cfgs.split == 'test':
        scene_ids = list(range(100, 190))
    elif cfgs.split == 'test_seen':
        scene_ids = list(range(100, 130))
    elif cfgs.split == 'test_similar':
        scene_ids = list(range(130, 160))
    elif cfgs.split == 'test_novel':
        scene_ids = list(range(160, 190))
    else:
        scene_ids = []

    for scene_id in scene_ids:
        scene_name = f'scene_{scene_id:04d}'
        print(f'\nEvaluating {scene_name}...')
        
        if scene_name in eval_data["scenes"]:
            print(f'  -> Already evaluated, skipping.')
            continue
            
        try:
            scene_acc_list = ge.eval_scene(scene_id, cfgs.dump_dir, return_list=False)
            
            scene_info = {
                "annotations": {},
                "scene_ap": float(np.mean(scene_acc_list)),
                "ap_per_friction": {}
            }
            
            for f_idx, f_val in enumerate(frictions):
                scene_info["ap_per_friction"][str(f_val)] = float(np.mean([acc[:, f_idx] for acc in scene_acc_list]))
                
            for ann_id, acc in enumerate(scene_acc_list):
                ann_info = {
                    "ap": float(np.mean(acc)),
                    "ap_per_friction": {
                        str(f_val): float(np.mean(acc[:, f_idx])) for f_idx, f_val in enumerate(frictions)
                    }
                }
                scene_info["annotations"][str(ann_id)] = ann_info
                
            eval_data["scenes"][scene_name] = scene_info
            
            with open(json_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
                
            print(f'\n  -> Saved {scene_name}. AP: {scene_info["scene_ap"]:.4f}')
            
        except Exception as e:
            print(f'\nError evaluating {scene_name}: {e}')
            
    print(f'\nJSON Evaluation completed and saved to {json_path}.')

if __name__=='__main__':
    if not cfgs.skip_inference:
        inference()
    evaluate()