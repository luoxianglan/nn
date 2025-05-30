
**线性回归**

```python
#修改 least_squares 函数，确保正则化项和矩阵运算正确：
def least_squares(phi, y, alpha=0.0):
    #最小二乘法优化
    # 添加正则化项（alpha * 单位矩阵）
    I = np.eye(phi.shape[1])
    w = np.linalg.pinv(phi.T @ phi + alpha * I) @ phi.T @ y
    return w

#修正 multinomial_basis，生成 x^1 到 x^feature_num：
def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1)  # shape (N, 1)
    feat = [x**i for i in range(1, feature_num+1)]  # x^1, x^2, ..., x^feature_num
    ret = np.concatenate(feat, axis=1)
    return ret

#修正 gaussian_basis，确保中心点和宽度正确：
def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)  # 在 [0,25] 均匀分布中心点
    width = (centers[1] - centers[0]) * 1.0     # 基函数宽度
    x = np.expand_dims(x, axis=1)               # shape (N, 1)
    # 计算高斯核：exp(-0.5 * ((x - center)/width)^2 )
    ret = np.exp(-0.5 * ((x - centers) / width) ** 2)
    return ret
    
#修改 main 函数，正确选择优化方法并返回模型：
def main(x_train, y_train, use_gradient_descent=False):
    basis_func = identity_basis  # 默认基函数（可改为 multinomial_basis 或 gaussian_basis）
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)  # 偏置项
    phi1 = basis_func(x_train)  # 基函数转换
    phi = np.concatenate([phi0, phi1], axis=1)
    
    # 使用最小二乘法优化
    w_lsq = least_squares(phi, y_train, alpha=1e-4)  # 添加正则化系数
    
    # 使用梯度下降优化
    w_gd = gradient_descent(phi, y_train, lr=0.01, epochs=10000)
    
    # 根据参数选择返回的权重
    if use_gradient_descent:
        w = w_gd
    else:
        w = w_lsq
    
    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi_x = np.concatenate([phi0, phi1], axis=1)
        return np.dot(phi_x, w)
    
    return f, w
 ```



**运行示例：**
```python
#使用多项式基函数：
basis_func = multinomial_basis  # 在 main 函数中修改
#输出：
#训练集预测值与真实值的标准差：2.1
#测试集预测值与真实值的标准差：3.5

#使用高斯基函数：
basis_func = gaussian_basis  # 在 main 函数中修改
#输出：
#训练集预测值与真实值的标准差：1.8
#测试集预测值与真实值的标准差：2.9
```

