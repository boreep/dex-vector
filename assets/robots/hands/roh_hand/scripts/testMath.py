import numpy as np
import matplotlib.pyplot as plt
from FingerMathURDF import HAND_FingerPosToAngle

angles_1 = []
angles_2 = []

# 设置手指ID和位置范围
finger_id = 4 # 选择手指ID，例如 0（拇指）或 1,2,3,4（其他手指）
if finger_id != 0:
    positions = np.linspace(-0.003, 0.016, 100)  
    for pos in positions:
        angle = HAND_FingerPosToAngle(finger_id, pos)
        angles_1.append(angle[1])
        angles_2.append(angle[2])
else:
    positions = np.linspace(-0.0035, 0.008, 100)
    for pos in positions:
        angle = HAND_FingerPosToAngle(finger_id, pos)
        angles_1.append(angle[0])
        angles_2.append(angle[2])

# 三阶多项式拟合
poly_degree = 3  
coeffs_1 = np.polyfit(positions, angles_1, poly_degree)
polynomial_1 = np.poly1d(coeffs_1)
angles_1_fit = polynomial_1(positions)

coeffs_2 = np.polyfit(positions, angles_2, poly_degree)
polynomial_2 = np.poly1d(coeffs_2)
angles_2_fit = polynomial_2(positions)

# 生成拟合方程的字符串
def poly_eq(coeffs):
    return f"y={coeffs[0]:.4f}x³{coeffs[1]:+.4f}x²{coeffs[2]:+.4f}x{coeffs[3]:+.4f}"

eq_1 = poly_eq(coeffs_1)
eq_2 = poly_eq(coeffs_2)

# 绘制图像
plt.figure(figsize=(10, 5))
plt.scatter(positions, angles_1, color='blue', label=f"Finger {finger_id} Angle 1 vs Position (Original)", s=10)
plt.plot(positions, angles_1_fit, color='red', label=f"Fit for Angle 1: {eq_1}", linewidth=2)

plt.scatter(positions, angles_2, color='green', label=f"Finger {finger_id} Angle 2 vs Position (Original)", s=10)
plt.plot(positions, angles_2_fit, color='magenta', label=f"Fit for Angle 2: {eq_2}", linewidth=2)

plt.xlabel("Position (mm)")
plt.ylabel("Angle (degrees)")
plt.title(f"Angle vs Position for Finger {finger_id}")
plt.legend()
plt.grid(True)
plt.show()
