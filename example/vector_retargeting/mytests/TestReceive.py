import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from example.vector_retargeting.mytests.VR_hand_detector_test import VRHandDetector

# 创建 VRHandDetector 实例
detector = VRHandDetector(address="tcp://172.17.129.195:12346", hand_type="Left")

# 初始化 3D Matplotlib 图像
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], color="red", label="Hand Keypoints")
sc_origin = ax.scatter([], [], [], color="green", label="Origin Positions")
sc_joint1 = ax.scatter([], [], [], color="blue", label="Joint Positions1")
texts = []  # 用于显示关键点编号

# 设置图像参数
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
ax.set_title("Hand Keypoints (3D)")

ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.2, 0.2)

ax.legend()

# 更新函数
def update(frame):
    try:
        array_data = detector.recv_origin_array()
        right_hand_data = array_data[44:70, :]
        pos_right_data = right_hand_data[:, :3]
        left_hand_data = array_data[18:44, :]
        pos_left_data = left_hand_data[:, :3]

        joint_pos1, _, keypoint_3d = detector.VRHandDetector(detector, left_hand_data)

        if joint_pos1 is None or keypoint_3d is None:
            print("未检测到手部数据")
            return sc_joint1, sc, sc_origin, *texts

        sc._offsets3d = (keypoint_3d[:, 0], keypoint_3d[:, 1], keypoint_3d[:, 2])
        sc_origin._offsets3d = (pos_left_data[:, 0], pos_left_data[:, 1], pos_left_data[:, 2])
        sc_joint1._offsets3d = (joint_pos1[:, 0], joint_pos1[:, 1], joint_pos1[:, 2])

    except Exception as e:
        print(f"更新过程中出现错误: {e}")
        detector.stop()

    return sc, sc_joint1, sc_origin, *texts

# 启动动画
ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)

# 关闭窗口时释放资源
def on_close(event):
    print("关闭窗口，停止进程...")
    detector.stop()

fig.canvas.mpl_connect('close_event', on_close)

# 显示动画
try:
    plt.show()
except KeyboardInterrupt:
    print("检测到 Ctrl+C，正在清理资源...")
    detector.stop()
