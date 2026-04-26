简单记录一下气象预测大赛
## 导入模块
我们要使用`numpy torch sklearn lightgbm`这些算法库，还有一些系统相关的工具`os pathlib `
```python
import torch
import numpy as np
import sklearn
import lightgbm as lgb
from pathlib import Path
import random
```
## 导入数据
上一节处理好了数据，在"output/xxx.pt"下
获取目录下所有的文件名字：
```python
dir_path = Path('/out/')
files = [str(file) for file in dir_path.rglob('*') if file.is_file()]


```
