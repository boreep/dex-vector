import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具
from VR_hand_detector_test import VRHandDetector

# 创建 VRHandDetector 实例
detector = VRHandDetector(address="tcp://172.17.129.195:12346", hand_type="Left")


# 初始化 3D Matplotlib 图像
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')  # 使用 3D 投影
sc = ax.scatter([], [], [], color="red", label="Keypoints")  # 关键点
texts = []  # 用于显示关键点编号

# 设置图像参数
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
ax.set_title("Hand Keypoints (3D)")

# 设置轴范围，根据需要调整
# ax.set_xlim(-0.2, 0.2)
# ax.set_ylim(-0.2, 0.2)
# ax.set_zlim(-0.2, 0.2)

ax.legend()

# 更新函数（每帧获取新数据）
def update(frame):
    try:
        # 获取 VR 传感器原始数据
        array_data = detector.recv_origin_array()

        if array_data is None:
            print("未检测到手部数据")
        pos_data=array_data[:,:3]
       
        # 更新散点图的数据（3D）
        sc._offsets3d = (pos_data[:, 0], pos_data[:, 1], pos_data[:, 2])
       
   
    except Exception as e:
        print(f"更新过程中出现错误: {e}")

    return sc, *texts

# 启动动画，每 20ms 更新一次
ani = animation.FuncAnimation(fig, update, interval=20, blit=False)

plt.show()
