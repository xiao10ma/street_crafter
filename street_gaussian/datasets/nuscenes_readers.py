from easyvolcap.utils.console_utils import *
import pyquaternion
from PIL import Image
import os
import numpy as np
import cv2
import sys
import shutil
from nuscenes.nuscenes import NuScenes
import pickle
sys.path.append(os.getcwd())

CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def readNuscenesInfo(args):
    nusc = NuScenes(version=args.data.version, dataroot=args.data.nusc_dir, verbose=True)

    selected_frames = args.data.selected_frames
    if args.debug:
        selected_frames = [0, 0]

    # TODO: 暂时注释下面这段，尚未看，似乎是用于 train 的
    # if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
    #     load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
    #     save_dir = os.path.join(cfg.model_path, 'input_ply')
    #     os.system(f'rm -rf {save_dir}')
    #     shutil.copytree(load_dir, save_dir)

    #     colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
    #     save_dir = os.path.join(cfg.model_path, 'colmap')
    #     os.system(f'rm -rf {save_dir}')
    #     shutil.copytree(colmap_dir, save_dir)

    # dynamic mask, TODO: 尚不清楚其作用
    # dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    # load_dynamic_mask = True

    # sky mask, TODO: 尚不清楚其作用
    # sky_mask_dir = os.path.join(path, 'sky_mask')
    # load_sky_mask = True

    with open('/HDD_DISK/users/mazipei/street_crafter/SparseDrive/data/infos/mini/nuscenes_infos_train.pkl', 'rb') as f:
        data = pickle.load(f)

    # load intrinsics and extrinsics
    intrinsics = []
    extrinsics = []


    for i, cam in enumerate(CAM_LIST):
        intrinsic = data['infos'][0]['cams'][cam]['cam_intrinsic']
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = data['infos'][0]['cams'][cam]['sensor2ego_translation']
        quat = data['infos'][0]['cams'][cam]['sensor2ego_rotation']
        extrinsic[:3, :3] = pyquaternion.Quaternion(quat).rotation_matrix
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)

    scene_token = data['infos'][0]['scene_token']
    cur_scene = scene_token

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(6)]

    scene_len = len(data['infos'])
    for i in tqdm(range(scene_len), desc='Loading Nuscenes Info'):
        cur_scene = data['infos'][i]['scene_token']
        if cur_scene != scene_token:
            break
        # ego frame pose: per frame ego -> world
        ego_frame_pose = np.eye(4)
        ego_frame_pose[:3, 3] = data['infos'][i]['ego2global_translation']
        ego_frame_pose[:3, :3] = pyquaternion.Quaternion(data['infos'][0]['ego2global_rotation']).rotation_matrix
        ego_frame_poses.append(ego_frame_pose)
        for i, cam in enumerate(CAM_LIST):
            ego_cam_pose = np.eye(4)
            ego_cam_pose[:3, 3] = data['infos'][0]['cams'][cam]['ego2global_translation']
            ego_cam_pose[:3, :3] = pyquaternion.Quaternion(data['infos'][0]['cams'][cam]['ego2global_rotation']).rotation_matrix
            ego_cam_poses[i].append(ego_cam_pose)

        

    ego_frame_poses = np.array(ego_frame_poses)
    ego_frame_poses[:, :3, 3] -= np.mean(ego_frame_poses[:, :3, 3], axis=0)
    