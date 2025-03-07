# -*- coding: utf-8 -*-
import math
import time
import threading
import signal
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import tyro
from loguru import logger

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from VR_hand_detector import VRHandDetector

from Robotic_Arm.rm_robot_interface import *
import ctypes

stop_flag = threading.Event()

def start_retargeting(robot_dir: Path, left_config_path: Path, left_arm: rm_robot_handle):
    """ 启动机器人重定向进程 """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Starting retargeting with config: {left_config_path}")
    left_retargeting = RetargetingConfig.load_from_file(left_config_path).build()

    detector = VRHandDetector(address="tcp://172.17.129.39:12346")

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

            retargeting_type = left_retargeting.optimizer.retargeting_type
            indices = left_retargeting.optimizer.target_link_human_indices

            origin_indices, task_indices = indices[0, :], indices[1, :]
            ref_value = right_joint_pos[task_indices, :] - right_joint_pos[origin_indices, :]

            left_qpos = left_retargeting.retarget(ref_value)

            #[0,4,6,2,8,9]
            #[if,mf,rf,lf,th_root,th_proximal]

            OffsetAngleForURDF = [0.4243102998823798, 2.8069436083614603, 2.7765308802396014, 2.7806045952072487, 2.7543096279771366]
            selected_left_qpos = [left_qpos[i] for i in [9, 0, 4, 6, 2, 8]] # 注意这里的索引应该是正确的
            #[th_proximal,if,mf,rf,lf,th_root]


            # 确保我们不会访问超出范围的索引
            left_real_qpos = [
                selected_left_qpos[0] + OffsetAngleForURDF[0], 
                selected_left_qpos[1] + OffsetAngleForURDF[1],
                selected_left_qpos[2] + OffsetAngleForURDF[2], 
                selected_left_qpos[3] + OffsetAngleForURDF[3],
                selected_left_qpos[4] + OffsetAngleForURDF[4], 
                selected_left_qpos[5]   # 对于th_root，不需要额外的偏移
            ]

            # 关节角度范围 (单位: 0.01°)
            left_joint_min = np.array([226, 10022, 9781, 10138, 9884 , 0])
            left_joint_max = np.array([3676, 17837, 17606, 17654, 17486, 9000])

            # 将弧度转换为角度（单位 0.01°）
            left_real_angles = np.array([int(angle * 18000 / math.pi) for angle in left_real_qpos])
            # 使用 np.clip 限制角度范围
            left_real_angles = np.clip(left_real_angles, left_joint_min, left_joint_max)

            print(left_arm.rm_set_hand_follow_angle( left_real_angles, 0))


    except KeyboardInterrupt:
        logger.info("Retargeting interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        left_arm.rm_delete_robot_arm()

def signal_handler(sig, frame):
    """ 处理终止信号 """
    logger.info("Received termination signal. Stopping...")
    stop_flag.set()

def main():
    """ 运行主程序 """
    left_config_path = get_default_config_path(RobotName.roh, RetargetingType.vector, HandType.right)

    robot_dir = Path(__file__).resolve().parents[2] / "assets" / "robots" / "hands"
    #robot_dir = "/home/user/project/assets/robots/hands"
    signal.signal(signal.SIGINT, signal_handler)  # 捕获 Ctrl+C 信号

    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建左机械臂连接，打印连接id
    left_handle = left_arm.rm_create_robot_arm("192.168.12.19", 8080)

    threading.Thread(target=start_retargeting, args=(robot_dir, left_config_path, left_arm), daemon=True).start()

    try:
        while not stop_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Main thread interrupted.")
    finally:
        logger.info("Shutting down.")
        left_arm.rm_delete_robot_arm()

if __name__ == "__main__":
    tyro.cli(main)
