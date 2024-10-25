import numpy as np

epsilon = np.array([[10, 15 / 2, 0], [15 / 2, 20, 0], [0, 0, -10]]) * 10**-6

E = 200 * 10**9
nu = 0.25
delta = np.eye(3)

sigma = E / (1 + nu) * epsilon + nu * E * delta * np.trace(epsilon) / (1 + nu) / (
    1 - 2 * nu
)
print(sigma)


tzz, tzxl = np.linalg.eig(sigma)
print(tzz)
tau_8 = (
    np.sqrt((tzz[0] - tzz[1]) ** 2 + (tzz[1] - tzz[2]) ** 2 + (tzz[2] - tzz[0]) ** 2)
    / 3
)
print(tau_8)
