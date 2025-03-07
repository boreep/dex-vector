# -*- coding: utf-8 -*-
import math
import time
import threading
import signal
from pathlib import Path
from queue import Empty

import cv2
import tyro
from loguru import logger


from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from VR_hand_detector import VRHandDetector

from Robotic_Arm.rm_robot_interface import *

stop_flag = threading.Event()

def start_retargeting(robot_dir: Path, left_config_path: Path, right_config_path: Path ):
    """ 启动机器人重定向进程 """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Starting retargeting with config: {left_config_path}and {right_config_path}")
    left_retargeting = RetargetingConfig.load_from_file(left_config_path).build()
    right_retargeting = RetargetingConfig.load_from_file(right_config_path).build()

    detector = VRHandDetector(address="tcp://172.17.129.39:12346")

    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    left_handle = left_arm.rm_create_robot_arm("192.168.12.20", 8080)
    right_handle = right_arm.rm_create_robot_arm("192.168.12.19", 8080)


    print(f"Left Arm ID: {left_handle.id}")
    print(f"Right Arm ID: {right_handle.id}")


        

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
            left_real_angles = [int(angle * 18000 / math.pi) for angle in left_real_qpos] # 直接转换为整数
            print(left_arm.rm_set_hand_follow_angle(left_real_angles, 0))

            right_real_angles = [int(angle * 18000 / math.pi) for angle in right_real_qpos]
            print(right_arm.rm_set_hand_follow_angle(right_real_angles, 0))




            # selected_left_qpos = [left_qpos[i] for i in [9, 0, 4, 6, 2, 8]] # 注意这里的索引应该是正确的
            # # #[th_proximal,if,mf,rf,lf,th_root]

            # joint_limit_low=[-0.385, -1.058, -1.069, -1.011, -1.0292, -0.035]
            # joint_limit_high=[0.217, 0.3062, 0.2963, 0.3006, 0.2976, 1.605]



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
    left_config_path = get_default_config_path(RobotName.roh, RetargetingType.vector, HandType.left)
    #left_config_path = "config/inspire_vector_left.yaml"
    right_config_path = get_default_config_path(RobotName.roh, RetargetingType.vector, HandType.right)

    robot_dir = Path(__file__).resolve().parents[2] / "assets" / "robots" / "hands"
    #robot_dir = "/home/user/project/assets/robots/hands"
    signal.signal(signal.SIGINT, signal_handler)  # 捕获 Ctrl+C 信号


    threading.Thread(target=start_retargeting, args=(robot_dir, left_config_path, right_config_path), daemon=True).start()


    try:
        while not stop_flag.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Main thread interrupted.")
    finally:
        logger.info("Shutting down.")


if __name__ == "__main__":
    tyro.cli(main)
