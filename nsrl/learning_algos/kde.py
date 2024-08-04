import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# import numpy as np

# # 假设你的数据形状是 (batch_size1, batch_size2, ..., batch_sizeN, dim1, dim2, ..., dimM)
# # 这里我们用一个例子来说明
# batch_sizes = [100, 100]
# dims = [1, 2]
# data = np.random.rand(*(batch_sizes + dims))
# print("data", data)

# # 计算dim的总大小
# dim_total_size = np.prod(dims)

# # 将数据 reshape 成 (batch_size1, batch_size2, ..., batch_sizeN, dim_total_size)
# flattened_data = data.reshape(*batch_sizes, dim_total_size) #*是把矩阵拆开参数传进去[2,3]->2,3

# # 将数据 reshape 成 (dim_total_size, batch_size1 * batch_size2 * ... * batch_sizeN)
# result = np.moveaxis(flattened_data, -1, 0).reshape(dim_total_size, -1)
# print("result", result)

# print("Original shape:", data.shape)
# print("Flattened and transposed shape:", result.shape)
# print(result)

import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

# 生成一些二维示例数据
data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=1000)

# 创建核密度估计对象
kde = gaussian_kde(data.T)

# 定义核密度函数
def kde_function(x, y):
    return kde([x, y])

# 对核密度函数进行积分
result, error = dblquad(kde_function, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

print(f"Integral result: {result}")
print(f"Integral error: {error}")

# 判断总概率是否为1
if np.isclose(result, 1, atol=1e-5):
    print("Total probability is approximately 1.")
else:
    print("Total probability is not 1.")













# 生成二维正态分布数据
mean = [0, 0]
cov = [[1, 0], [0, 1]]
# data = np.random.multivariate_normal(mean, cov, 1000)
data = np.array([[0,0], [1,1], [1,-1],[-1,1],[-1,-1], [0,0], [1,1], [1,-1],[-1,1],[-1,-1]])

# 设定我们要估计密度的位置 x0
x0 = [-0.5, -0.5]

# 创建核密度估计对象
# kde = gaussian_kde(result)  #转置
kde = gaussian_kde(data.T)  #转置

# 估计在位置 x0 处的密度
# density_x0 = kde.evaluate([1,1,1,1])

# for _ in range(20):
# x0 = np.random.rand(4)
density_x0 = kde.evaluate([[0,1], [0,1]])
print(f"在位置 {x0} 处的密度估计为: {density_x0[0]}")
print("x0", x0)
print("************")

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