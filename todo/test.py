# %% 导入库
import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
# %% 测试一下
x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()
msg = "Roll a dice!"
print(msg)
A = np.array([[2,2,3],[4,7,7],[-2,4,5]])
d = np.linalg.det(A)
# %%
A
# %%
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
D = np.array([[1,0,0],[0,5,0],[0,0,9]])
# %%
np.linalg.inv(D)
# %%
np.eye(3) - np.linalg.inv(D) @ A
# %%
