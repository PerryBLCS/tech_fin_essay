"""项目配置和共享常量"""

import random
import numpy as np
import torch
import warnings

warnings.filterwarnings(
    'ignore',
    message='.*does not have many workers.*',
    category=UserWarning,
)

# 数据集
GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
TAIWAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


# 设置随机种子
def set_seed(seed, deterministic=True):
    """设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')