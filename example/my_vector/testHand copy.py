import time
from Robotic_Arm.rm_robot_interface import *

# 实例化左臂和右臂
left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接
left_handle = left_arm.rm_create_robot_arm("192.168.12.20", 8080)
right_handle = right_arm.rm_create_robot_arm("192.168.12.19", 8080)

print(f"Left Arm ID: {left_handle.id}")
print(f"Right Arm ID: {right_handle.id}")

# 控制右手（示例：角度参数需要根据右手调整）
print(left_arm.rm_set_hand_follow_pos([65535, 0, 00, 00, 00, 0], 0))
time.sleep(2)
print(left_arm.rm_set_hand_follow_pos([0, 0, 00, 00, 00, 0], 0))

