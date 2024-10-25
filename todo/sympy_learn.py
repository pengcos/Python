# %%
import sympy as sp
import numpy as np
# %%
# 定义符号变量 x 和 y
x, y = sp.symbols('x y')
C = sp.symbols('C')
#C = 2.5
a = 30/180*sp.pi
f = C*sp.cot(a)/(1-a*sp.cot(a))*(-x**2*sp.tan(a) + x*y+(x**2+y**2)*(a-sp.atan(y/x)))


dfdx2 = sp.diff(f,x,x)
dfdy2 = sp.diff(f,y,y)
dfdxdy = sp.diff(f,x,y)

f0 = dfdx2.subs({y:0})
eq = sp.Eq(f0, -5)
solutions = sp.solve(eq, C)
print(solutions)

sigma_x = dfdy2.subs({x:8,y:3,C:solutions[0]})
sigma_y = dfdx2.subs({x:8,y:3,C:solutions[0]})
sigma_12 = -dfdxdy.subs({x:8,y:3,C:solutions[0]})

sigma_x_value = sp.N(sigma_x)
sigma_y_value = sp.N(sigma_y)
sigma_12_value = sp.N(sigma_12)
print(sigma_x_value)
print(sigma_y_value)
print(sigma_12_value)

# %%
A, B, C, D, x, y = sp.symbols('A,B,C,D,x,y')
phi = A*x*y+B*y**2+C*y**3+D*x*y**3
phi
# %%
sigma_x = sp.diff(phi,y,y)
sigma_y = sp.diff(phi,x,x)
tau_xy = -sp.diff(phi, x, y)
# %%
