import time
from pathlib import Path
import threading
import signal
from queue import Empty

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import tyro
from loguru import logger

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig

from example.vector_retargeting.mytests.VR_hand_detector_test import VRHandDetector

stop_flag = threading.Event()

def start_retargeting(robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in str(config_path).lower() else "Left"
    detector = VRHandDetector(address="tcp://172.17.129.126:12346", hand_type="Right")

    config = RetargetingConfig.load_from_file(config_path)

    fig, ax = plt.subplots()
    sc = ax.scatter([], [])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    def update(frame):
        try:
            array_data = detector.recv_origin_array()
        except Empty:
            logger.error(f"未接收到消息来自： {detector.address}")
            return

        right_hand_data = array_data[44:69, :]
        joint_pos, keypoint_2d, _ = detector.VRHandDetector(right_hand_data)
        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        
        sc.set_offsets(keypoint_2d)
        return sc,

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()

def signal_handler(sig, frame):
    logger.info("Received termination signal. Stopping...")
    stop_flag.set()

def main(robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    signal.signal(signal.SIGINT, signal_handler)  # 只用这个处理 Ctrl+C 终止
    threading.Thread(target=start_retargeting, args=(robot_dir, config_path), daemon=True).start()

    while not stop_flag.is_set():
        time.sleep(0.1)

    logger.info("Shutting down.")

if __name__ == "__main__":
    tyro.cli(main)
