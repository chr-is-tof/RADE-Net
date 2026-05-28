import os
import os.path as osp
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_matrices_from_dict_calib(dict_calib=None):
    img_size = (dict_calib['img_size_w'], dict_calib['img_size_h'])
    intrinsics = np.array([
        [dict_calib['fx'], 0.0, dict_calib['px']],
        [0.0, dict_calib['fy'], dict_calib['py']],
        [0.0, 0.0, 1.0]
    ])
    distortion = np.array([
        dict_calib['k1'], dict_calib['k2'], dict_calib['k3'], \
        dict_calib['k4'], dict_calib['k5']
    ]).reshape((-1,1))

    # L to C
    yaw_ldr2cam = dict_calib['yaw_ldr2cam']
    pitch_ldr2cam = dict_calib['pitch_ldr2cam']
    roll_ldr2cam = dict_calib['roll_ldr2cam']
    r_ldr2cam = (R.from_euler('zyx', [yaw_ldr2cam, pitch_ldr2cam, roll_ldr2cam], degrees=True)).as_matrix()

    x_ldr2cam = dict_calib['x_ldr2cam']
    y_ldr2cam = dict_calib['y_ldr2cam']
    z_ldr2cam = dict_calib['z_ldr2cam']
    T_ldr2cam = np.concatenate([r_ldr2cam, np.array([x_ldr2cam,y_ldr2cam,z_ldr2cam]).reshape(-1,1)], axis=1)
    # L to C

    return img_size, intrinsics, distortion, T_ldr2cam


def get_dict_cam_calib_from_yml(cam_calib_path):
    dict_cam_calib = dict()
    dir_cam_calib = cam_calib_path
    list_yml = os.listdir(dir_cam_calib)
    for yml_file_name in list_yml:
        key_name = yml_file_name.split('.')[0].split('_')[1]
        with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
            dict_temp = yaml.safe_load(yml_file)
        dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam
    return dict_cam_calib
