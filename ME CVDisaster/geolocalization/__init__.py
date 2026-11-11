import sys
from pathlib import Path


# 两个模型共用已经复现的数据处理与采样实现。
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "Author CVDisaster" / "geolocalization"))
