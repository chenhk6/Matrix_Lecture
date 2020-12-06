"""
Matrix_Lecture_Exercises_Chap_04-05_Solution

Version: 1.0
Author: chenhk6
"""
import numpy as np
from numpy import linalg as lg
from matplotlib import pyplot as plt


x1 = np.mat([1, 5]).T
x2 = np.mat([-5, 1]).T
ld = np.diag([10, 2])
x = np.hstack((x1, x2))
x_inv = lg.inv(x)
r1 = x@ld@x_inv
sampleNo = 400
mv = np.mat([[0, 0]])
s = np.dot(np.random.randn(sampleNo, 2), r1)
plt.subplot()
plt.plot(s[:, 1], s[:, 0], '+')
plt.show()
r2 = np.cov(s.T)

u, sigma, v = lg.svd(r2)
v1 = u[:, 0].T
v2 = u[:, 1].T
ax = plt.axes()
ax.arrow(0, 0, *v1, length_includes_head=True, head_width=0.02, head_length=0.1,
         shape="full", fc='red', ec='red', alpha=0.9, overhang=0.5)
ax.arrow(0, 0, *v2, length_includes_head=True, head_width=0.02, head_length=0.1,
         shape="full", fc='red', ec='blue', alpha=0.9, overhang=0.5)
plt.show()

