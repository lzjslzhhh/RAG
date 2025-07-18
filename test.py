# test_numpy.py
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import torch

print("✅ NumPy版本:", np.__version__)
print("✅ SciPy版本:", scipy.__version__)
print("✅ PyTorch版本:", torch.__version__)

# 测试核心功能
arr = np.random.rand(3,3)
print("NumPy数组运算:", arr @ arr.T)