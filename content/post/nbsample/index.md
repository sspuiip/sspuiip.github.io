---
title: Jupyter Notebooks example
# subtitle: Learn how to blog in Academic using Jupyter notebooks
# summary: Learn how to blog in Academic using Jupyter notebooks
authors:
- admin
tags: []
categories: [misc]
date: "2024-12-24T00:00:00Z"
lastMod: "2024-12-24T00:00:00Z"
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

### 观察Python演示代码的运行结果

&emsp;&emsp;用以下Python代码画一个正弦函数的图像。



```python
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-4,4,200)
y = np.sin(x)

plt.plot(x,y)
plt.show()
```


    
![png](./index_2_0.png)
    



