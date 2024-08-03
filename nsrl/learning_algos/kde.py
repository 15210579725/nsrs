import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 生成二维正态分布数据
mean = [0, 0]
cov = [[1, 0], [0, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)

# 设定我们要估计密度的位置 x0
x0 = [0.5, 0.5]

# 创建核密度估计对象
kde = gaussian_kde(data.T)  #转置

# 估计在位置 x0 处的密度
density_x0 = kde.evaluate([2,2])

print(f"在位置 {x0} 处的密度估计为: {density_x0[0]}")

# 生成用于绘图的网格点
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

# 绘制数据点和核密度估计结果
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], s=5, label='Data Points')
plt.contour(X, Y, Z, colors='k', alpha=0.5)
plt.imshow(np.rot90(Z), cmap=plt.cm.viridis, extent=[-3, 3, -3, 3])
plt.colorbar(label='Density')
plt.scatter(x0[0], x0[1], color='red', marker='x', s=100, label=f'x0: {x0}, Density: {density_x0[0]:.4f}')
plt.title('2D Kernel Density Estimation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
