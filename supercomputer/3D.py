import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图模块

# 读取纯数字 CSV 文件（无表头）
data = pd.read_csv("solution_800x1200_T1.csv", header=None).values

# 获取矩阵尺寸
ny, nx = data.shape
print(f"数据维度: {ny} × {nx}")

# 创建坐标网格 (假设 x, y 范围都是 0~3，可根据需要调整)
x = np.linspace(0, 3, nx)
y = np.linspace(0, 3, ny)
X, Y = np.meshgrid(x, y)
Z = data  # Z 为高度值

# 创建 3D 图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 添加颜色条与标签
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label="u(x,y)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
ax.set_title(f"3D Visualization for Grid ({ny} × {nx})")

# 保存图片（在当前目录）
plt.savefig("3D_result.png", dpi=300, bbox_inches='tight')

# 显示
plt.show()
