import numpy as np
import matplotlib.pyplot as plt
from FingerMathURDF import HAND_FingerPosToAngle

angles_1 = []
angles_2 = []

# 设置手指ID和位置范围
finger_id = 0  # 选择食指（可以调整为1、2、3、4来选择其他手指）
if finger_id != 0:
    positions = np.linspace(-0.003, 0.016, 100)  
    for pos in positions:
        angle = HAND_FingerPosToAngle(finger_id, pos)  # 调用函数获取角度
        angles_1.append(np.degrees(angle[1]))  # 转换为度数并保存第一个角度
        angles_2.append(np.degrees(angle[2]))  # 转换为度数并保存第二个角度

else:
    positions=np.linspace(-0.0035,0.008,100)
    for pos in positions:
        angle = HAND_FingerPosToAngle(finger_id, pos)
        angles_1.append(np.degrees(angle[0]))
        angles_2.append(np.degrees(angle[2]))

# 选择拟合的阶数
poly_degree = 1

# 对angles_1和angles_2进行拟合
coeffs = np.polyfit(angles_1, angles_2, poly_degree)
polynomial_fit = np.poly1d(coeffs)

# 生成用于绘制拟合曲线的数据点
angles_1_fit = np.linspace(min(angles_1), max(angles_1), 100)
angles_2_fit = polynomial_fit(angles_1_fit)

# 绘制原始数据和拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(angles_1, angles_2, color='blue', label=f"Original Data (Angle 1 vs Angle 2)")
plt.plot(angles_1_fit, angles_2_fit, color='red', label=f"Fit: {str(polynomial_fit)}", linewidth=2)
plt.xlabel("Angle 1 (degrees)")
plt.ylabel("Angle 2 (degrees)")
plt.title(f"Angle 1 vs Angle 2 for Finger {finger_id} with Fit")
plt.legend()
plt.grid(True)
plt.show()