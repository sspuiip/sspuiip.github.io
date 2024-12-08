---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "集成学习结合方法"
#subtitle: "To measure the consistency of hypotheses"
summary: ""
authors: [admin]
tags: [ensemble]
categories: [misc]
date: 2024-12-09T15:56:30+08:00
lastmod: 2024-12-09T16:40:30+08:00
featured: false
draft: false
math: true
diagram: true


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

```markmap
- 结合方法
  - 均值法
    - 简单平均法
    - 加权平均法
```


&emsp;&emsp;本文探讨集成学习中的一些常见结合方法，即基学习器结合输出方式。基学习器结合方法带来的好处大致可表示为以下三个方面：
- 假设空间相对单个学习器大得多。
- 多个假设可降低单个假设陷入局部最优的风险。
- 结合多个假设可以拓展假设空间，相较于单个假设，更有可能近似真实未知假设。

### 均值法

&emsp;&emsp;对于数值型输出问题，均值法是最基本的的结合方法。均值法分为简单平均法和加权平均法。下面是相应的简单介绍。

- [x] **简单平均法**

&emsp;&emsp;给定基学习器$h_i:\mathcal{X}\rightarrow \mathcal{Y} (i\in [1..T])$和样本$(\pmb{x},y)$，简单平均法$H(\pmb{x})$为，
$$
H(\pmb{x})=\frac1T\sum_{i=1}^T h_i(\pmb{x}),\quad h_i(\pmb{x})=y+\epsilon_i(\pmb{x})
$$

易知，单一学习器误差为，
{{<math>}}
$$
\begin{split}
\bar{\textrm{err}}(h)&= \frac1T\sum_{i=1}^T\int \left[ h_i(\pmb{x})-y\right]^2 p(\pmb{x}) d\pmb{x}\\
&=\frac1T\sum_{i=1}^T\int \epsilon_i(\pmb{x})^2 p(\pmb{x}) d\pmb{x}
\end{split}
$$
{{</math>}}
平均法集成学习器的误差为，
{{<math>}}
$$
\begin{split}
\textrm{err}(H)&= \int \left[\frac1T\sum_{i=1}^T h_i(\pmb{x})-y\right]^2 p(\pmb{x}) d\pmb{x}\\
&=\int \left[\frac1T\sum_{i=1}^T \epsilon_i(\pmb{x})\right]^2 p(\pmb{x}) d\pmb{x}\le \bar{\textrm{err}}(h)\\
\end{split}
$$
{{</math>}}
这说明，平均法集成学习器误差不会大于单一学习器误差。

&emsp;&emsp;如果假设$\epsilon$均值为0且相互独立，即
$$
\int \epsilon_i(\pmb{x})p(\pmb{x})d\pmb{x}=0\quad\wedge\quad\int \epsilon_i(\pmb{x})\epsilon_j(\pmb{x})p(\pmb{x})d\pmb{x}=0
$$
则有，
$$
\textrm{err}(H)=\frac1T\bar{\textrm{err}}(h)
$$
即，集成误差仅为单一学习器平均误差的$1/T$。

- [x] **加权平均法**

$$
H(\pmb{x})=\frac1T\sum_{i=1}^T w_i h_i(\pmb{x}),\quad w_i\ge 0\wedge \sum_i^T w_i=1
$$

&emsp;&emsp;易知，$\textrm{err}(H)$为，
{{<math>}}
$$
\begin{split}
\textrm{err}(H)&= \int \left[\sum_{i=1}^T w_ih_i(\pmb{x})-y\right]^2 p(\pmb{x}) d\pmb{x}\\
&=\int \left[\sum_{i=1}^T w_ih_i(\pmb{x})-y\right]\left[\sum_{j=1}^T w_jh_j(\pmb{x})-y\right] p(\pmb{x}) d\pmb{x}\le \bar{\textrm{err}}(h)\\
&=\sum_{i=1}^T\sum_{j=1}^Tw_iw_j C_{ij}
\end{split}
$$
{{</math>}}
其中，
{{<math>}}
$$
C_{ij}=\int [h_i(\pmb{x})-y][h_j(\pmb{x})-y]p(\pmb{x}) d\pmb{x}
$$
{{</math>}}

使用拉格朗日乘子法，可以得到$\pmb{w}$的闭式解，
{{<math>}}
$$
\begin{split}
\hat{\pmb{w}}&=\mathop{\arg\min}\limits_{\pmb{w}}\quad\textrm{err}(H)\\
w_i&=\frac{\sum_{j=1}^TC_{ij}^{-1}}{\sum_{k=1}^T\sum_{j=1}^TC_{kj}^{-1}}
\end{split}
$$
{{</math>}}
前提是$C$矩阵可逆，但一般情况下不可行。简单平均法为加权平均法的特例，但也不意味着效果一定不比加权平均法差。



