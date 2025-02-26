# -*- coding: utf-8 -*-
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

stop_flag = threading.Event()

def start_retargeting(robot_dir: Path, left_config_path: Path, right_config_path: Path):
    """ 启动机器人重定向进程 """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Starting retargeting with config: {left_config_path}and {right_config_path}")
    left_retargeting = RetargetingConfig.load_from_file(left_config_path).build()
    right_retargeting = RetargetingConfig.load_from_file(right_config_path).build()

    detector = VRHandDetector(address="tcp://172.17.129.39:12346")



    # 关节映射

    left_retargeting_joint_names = left_retargeting.joint_names

    right_retargeting_joint_names = right_retargeting.joint_names

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
            right_indices = right_retargeting.optimizer.target_link_human_indices

            origin_indices, task_indices = indices[0, :], indices[1, :]
            right_origin_indices, right_task_indices = right_indices[0, :], right_indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            right_ref_value = right_joint_pos[right_task_indices, :] - right_joint_pos[right_origin_indices, :]

            left_qpos = left_retargeting.retarget(ref_value)
            right_qpos = right_retargeting.retarget(right_ref_value)

            print(left_qpos)

            # left_robot.set_qpos(left_qpos[left_retargeting_to_sapien])
            # right_robot.set_qpos(right_qpos[right_retargeting_to_sapien])


    except KeyboardInterrupt:
        logger.info("Retargeting interrupted by user.")
    finally:
        cv2.destroyAllWindows()

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
