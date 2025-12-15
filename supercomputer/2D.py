import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取纯数字 CSV 文件（没有表头）
data = pd.read_csv("solution_800x1200_T1.csv", header=None).values  # 转成numpy二维数组

# 获取尺寸
ny, nx = data.shape
print(f"数据维度: {ny} × {nx}")

# 创建坐标轴 (假设坐标范围 0~3，可以根据需要改)
x = np.linspace(0, 3, nx)
y = np.linspace(0, 3, ny)
X, Y = np.meshgrid(x, y)

# 绘制平滑二维色图
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, data, shading='auto', cmap='viridis')
plt.colorbar(label="u(x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"2D Visualization for Grid ({ny} × {nx})")

# 保存或显示
plt.tight_layout()
plt.savefig("2D_result.png", dpi=300, bbox_inches='tight')
plt.show()
