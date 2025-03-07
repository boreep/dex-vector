# -*- coding: utf-8 -*-
import math
import time
import threading
import signal
from pathlib import Path
from queue import Empty

import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from VR_hand_detector import VRHandDetector


stop_flag = threading.Event()

def start_retargeting(robot_dir: Path, left_config_path: Path, right_config_path: Path ):
    """ 启动机器人重定向进程 """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Starting retargeting with config: {left_config_path}and {right_config_path}")
    left_retargeting = RetargetingConfig.load_from_file(left_config_path).build()
    right_retargeting = RetargetingConfig.load_from_file(right_config_path).build()

    detector = VRHandDetector(address="tcp://172.17.129.39:12346")

    # 初始化仿真环境
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    left_config = RetargetingConfig.load_from_file(left_config_path)
    right_config= RetargetingConfig.load_from_file(right_config_path)
    scene = sapien.Scene()

    # 配置渲染材质
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # 添加光照
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # 设置相机
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    # 初始化Viewer
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # 加载左机器人
    loader = scene.create_urdf_loader()
    left_filepath = Path(left_config.urdf_path)
    robot_name = left_filepath.stem

    right_filepath = Path(right_config.urdf_path)
    loader.load_multiple_collisions_from_file = True

    left_robot = loader.load(str(left_filepath))
    right_robot = loader.load(str(right_filepath))
    if "inspire"  in robot_name:
        left_robot.set_pose(sapien.Pose([0, -0.25, -0.1]))
        right_robot.set_pose(sapien.Pose([0, 0.25, -0.1]))

    if "roh"  in robot_name:
        left_robot.set_pose(sapien.Pose([0, -0.25, -0.1]))
        right_robot.set_pose(sapien.Pose([0, 0.25, -0.1]))

    # 关节映射
    left_sapien_joint_names = [joint.get_name() for joint in left_robot.get_active_joints()]
    left_retargeting_joint_names = left_retargeting.joint_names
    #['if_proximal_joint', 'if_distal_joint', 'lf_proximal_joint', 'lf_distal_joint', 'mf_proximal_joint', 'mf_distal_joint', 'rf_proximal_joint', 'rf_distal_joint', 'th_root_joint', 'th_proximal_joint', 'th_distal_joint']
    #[0,4,6,2,8,9]
    left_retargeting_to_sapien = np.array([left_retargeting_joint_names.index(name) for name in left_sapien_joint_names], dtype=int)

    right_sapien_joint_names = [joint.get_name() for joint in right_robot.get_active_joints()]
    right_retargeting_joint_names = right_retargeting.joint_names
    right_retargeting_to_sapien = np.array([right_retargeting_joint_names.index(name) for name in right_sapien_joint_names], dtype=int)

    try:
        while not stop_flag.is_set():
            try:
                array_data = detector.recv_origin_array()
                if array_data is None:
                    continue
            except Empty:
                logger.error(f"未接收到来自 {detector.address} 的数据")
                continue

            right_hand_data = array_data[44:70, :]
            left_hand_data = array_data[18:44, :]

            joint_pos, right_joint_pos, keypoint_2d = detector.detect(detector,left_hand_data, right_hand_data)

            if joint_pos is None:
                logger.warning(f"left or right hand is not detected.")
                continue

            indices = left_retargeting.optimizer.target_link_human_indices
            right_indices = right_retargeting.optimizer.target_link_human_indices

            origin_indices, task_indices = indices[0, :], indices[1, :]
            right_origin_indices, right_task_indices = right_indices[0, :], right_indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            right_ref_value = right_joint_pos[right_task_indices, :] - right_joint_pos[right_origin_indices, :]

            left_qpos = left_retargeting.retarget(ref_value) 
            #[0,4,6,2,8,9]
            #[if,mf,rf,lf,th_root,th_proximal]
            right_qpos = right_retargeting.retarget(right_ref_value)


            OffsetAngleForURDF = [0.4243102998823798, 2.8069436083614603, 2.7765308802396014, 2.7806045952072487, 2.7543096279771366]
            selected_left_qpos = [left_qpos[i] for i in [9, 0, 4, 6, 2, 8]] # 注意这里的索引应该是正确的
            #[th_proximal,if,mf,rf,lf,th_root]
            selected_right_qpos = [right_qpos[i] for i in [9, 0, 4, 6, 2, 8]]
            # 确保我们不会访问超出范围的索引
            left_real_qpos = [
                selected_left_qpos[0] + OffsetAngleForURDF[0], 
                selected_left_qpos[1] + OffsetAngleForURDF[1],
                selected_left_qpos[2] + OffsetAngleForURDF[2], 
                selected_left_qpos[3] + OffsetAngleForURDF[3],
                selected_left_qpos[4] + OffsetAngleForURDF[4], 
                selected_left_qpos[5]   # 对于th_root，不需要额外的偏移
            ]

            right_real_qpos = [
                selected_right_qpos[0] + OffsetAngleForURDF[0],
                selected_right_qpos[1] + OffsetAngleForURDF[1],
                selected_right_qpos[2] + OffsetAngleForURDF[2], 
                selected_right_qpos[3] + OffsetAngleForURDF[3],
                selected_right_qpos[4] + OffsetAngleForURDF[4], 
                selected_right_qpos[5]   # 对于th_root，不需要额外的偏移
            ]

            # 将弧度转换为角度
            left_real_angles = [int(angle * 180 / math.pi) for angle in left_real_qpos] # 直接转换为整数
            right_real_angles =[int(angle * 180 / math.pi) for angle in right_real_qpos]
            print(left_real_angles)
 
 
            left_robot.set_qpos(left_qpos[left_retargeting_to_sapien])
            right_robot.set_qpos(right_qpos[right_retargeting_to_sapien])

            viewer.render()

    except KeyboardInterrupt:
        logger.info("Retargeting interrupted by user.")
    

def signal_handler(sig, frame):
    """ 处理终止信号 """
    logger.info("Received termination signal. Stopping...")
    stop_flag.set()

def main():
    """ 运行主程序 """
    left_config_path = get_default_config_path(RobotName.roh, RetargetingType.vector, HandType.left)
    #left_config_path = "config/inspire_vector_left.yaml"
    right_config_path = get_default_config_path(RobotName.roh, RetargetingType.vector, HandType.right)

    robot_dir = Path(__file__).resolve().parents[2] / "assets" / "robots" / "hands"
    #robot_dir = "/home/user/project/assets/robots/hands"
    signal.signal(signal.SIGINT, signal_handler)  # 捕获 Ctrl+C 信号


    threading.Thread(target=start_retargeting, args=(robot_dir, left_config_path, right_config_path), daemon=True).start()

    try:
        while not stop_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Main thread interrupted.")
    finally:
        logger.info("Shutting down.")

if __name__ == "__main__":
    tyro.cli(main)
