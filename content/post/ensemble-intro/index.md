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
  - 投票法
    - 绝对多数投票法
    - 相对多数投票法
    - 加权投票法
    - 软投票法
  - 学习结合法
    - Stacking
    - 无限集成
  - 相关方法
    - 纠错输出编码法
    - 混合专家模型
```


&emsp;&emsp;本文探讨集成学习中的一些常见结合方法，即基学习器结合输出方式。基学习器结合方法带来的好处大致可表示为以下三个方面：
- 假设空间相对单个学习器大得多。
- 多个假设可降低单个假设陷入局部最优的风险。
- 结合多个假设可以拓展假设空间，相较于单个假设，更有可能近似真实未知假设。

### 1. 均值法

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

### 2. 投票法

&emsp;&emsp;多值问题（一般指分类），投票法是最基本的结合方法。假设有$T$个不同的分类器{{<math>}}$\{h_1,...,h_T\}${{</math>}}，投票法要从$l$个标记$(c_1,...,c_l)$中给出所有分类器结合后的输出类别。一般情况，分类器$h_i$的输出结果是一个$l$维向量$(h_i^1(\pmb{x},h_i^2(\pmb{x},...,h_i^l(\pmb{x}))$,其中
{{<math>}}
$$
h_i^j(\pmb{x})\in\left\{\begin{array}{cc}[0,1],& c_j为类别概率;\\ \{0,1\},&c_j为类别. \end{array} \right.
$$
{{</math>}}

- [x] 绝对多数投票法：绝对多数投票的输出为获票过半的类别标记。如果所有票都未过半，则拒绝。

{{<math>}}
$$
H(\pmb{x})=\left\{\begin{array}{cc}c_j,& \sum_{i=1}^T h_i^j(\pmb{x})>\frac12\sum_{k=1}^l\sum_{i=1}^Th_i^k(\pmb{x});\\ reject,&otherwise. \end{array} \right.
$$
{{</math>}}



- [x] 相对多数投票法：相对多数投票法的输出为获票最多的类别标记。

{{<math>}}
$$
H(\pmb{x})=c_{\mathop{\arg\max}\limits_{j}\sum_{i=1}^T h_i^j(\pmb{x})}
$$
{{</math>}}

- [x] 加权投票法
{{<math>}}
$$
H(\pmb{x})=c_{\mathop{\arg\max}\limits_{j}\sum_{i=1}^T w_ih_i^j(\pmb{x})},\quad\sum_i w_i=1 \wedge \pmb{w}\succeq 0
$$
{{</math>}}

- [x] 软投票法

&emsp;&emsp;对于类别概率输出的分类器，一般采用软投票法。如果所有分类器都且有相同权重，则可以对所有输出进行平均，即
{{<math>}}
$$
H^j(\pmb{x})=\frac1T\sum_{i=1}^Th_i^j(\pmb{x})
$$
{{</math>}}
  - [ ]分类器权重投票法
{{<math>}}
$$
H^j(\pmb{x})=\sum_{i=1}^Tw_ih_i^j(\pmb{x})
$$
{{</math>}}
- [ ]类别权重投票法
{{<math>}}
$$
H^j(\pmb{x})=\sum_{i=1}^Tw_i^jh_i^j(\pmb{x})
$$
{{</math>}}

### 3. Stacking

&emsp;&emsp;学习结合法：结合通过学习器训练得到集成。Stacking是通过训练的形式，学习得到个体学习器结合方式的集成学习方法。个体学习器一般称为一级学习器，学习结合的学习器称为二级学习器，也称为元学习器。具体学习过程大致如下：

```
#1. 训练一级学习器
for t=1..T:
  h_t = L_t(D)

D'= emptyset

#2. 生成新数据集
for i=1..m:
  for t=1..T:
    z_it = h_t(x_i)
  D' = D' U ((z_i1,...,z_iT),y_i)

#3. 训练二级学习器
h' = L(D')
```

最后的输出形式为，

$$
H(\pmb{x})=h'(h_1(\pmb{x}),...,h_T(\pmb{x}))
$$

### 4. 无限集成

&emsp;&emsp;思考：当$T\rightarrow\infty$时，该如何开展集成学习？引出了一个问题：无限多个假设集如何集成？

&emsp;&emsp;无限集成是指：集成无限多假设集的集成框架。无限集成可看成对所有假设集都进行结合的一种学习方法。结合核方法，可以将无限假设集嵌入核中，通过支持向量机学习，可获得无限假设集的集成方式。

&emsp;&emsp;假设{{<math>}}$\mathcal{H}=\{h_a:a\in\mathcal{C}\}${{</math>}}，其中$\mathcal{C}$是测度空间。嵌入$\mathcal{H}$的核定义为，
{{<math>}}
$$
K_{\mathcal{H},r}(\pmb{x}_i,\pmb{x}_j)=\int_{\mathcal{C}}\Phi_{\pmb{x}_i}(a)\Phi_{\pmb{x}_j}(a)da
$$
{{</math>}}
其中，$\Phi_{\pmb{x}}(a)=r(a)h_a(\pmb{x})$，该核是有效核函数[Scholkopf & Smola, 2002]。该框架可以定义出原问题如下，
{{<math>}}
$$
\begin{split}
\min\limits_{w\in\mathcal{L}_2(\mathcal{C}),b\in\mathbb{R},\epsilon\in\mathbb{R}^m}\quad &\frac12\int_{\mathcal{C}}w^2(a)da+C\sum_{i=1}^m\epsilon_i\\
\textrm{s.t.}\quad &y_i\left(\int_{\mathcal{C}}w(a)r(a)h_a(\pmb{x})da + b \right)\ge 1-\epsilon_i\\
&\epsilon_i\ge 0(\forall i=1,...,m)
\end{split}
$$
{{</math>}}
最终分类器为，
{{<math>}}
$$
g(\pmb{x})=\textrm{sign}\left( \int_{\mathcal{C}}w(a)r(a)h_a(\pmb{x})da + b \right)
$$
{{</math>}}
&emsp;&emsp;使用拉格朗日乘子法和核技巧，可以得到对偶问题，最终分类器可以通过核$K_{\mathcal{H},r}$表示为，
{{<math>}}
$$
g(\pmb{x})=\textrm{sign}\left(\sum_{i=1}^my_i\lambda_iK_{\mathcal{H},r}(\pmb{x}_i,\pmb{x}_j) + b \right)
$$
{{</math>}}

### 5. 纠错输出编码法

&emsp;&emsp;该方法由编码解码**两个阶段组**成：
- **编码阶段**： 首先构建一组$B$个不同的类别标记划分{$c_1,...,c_l$}，然后在每个划分上训练$B$个二元分类器$h_1,...,h_B$。
- **解码阶段**： 给定样本$\pmb{x}$，使用$B$个二元分类器的输出生成一个码字。然后该码字和每一个类别的码字进行对比，具有最相似码字的类别作为输出。

&emsp;&emsp;以下是$l(=4)$个类别$B(=5)$个分类器的二元纠错输出码示例：
{{< table path="code2.csv" header="true" caption="四类问题的二元纠错输出码示例." >}}

&emsp;&emsp;以下是$l(=4)$个类别$B(=7)$个分类器的三元纠错输出码示例：
{{< table path="code3.csv" header="true" caption="四类问题的三元纠错输出码示例.(注：0与-1，+1的距离为0.5)" >}}

- **常见二元解码器**
  - [x] 汉明解码器
  {{<math>}}
  $$
  \textrm{HD}(\pmb{v},\pmb{w})=\frac{\sum_j (1-\textrm{sign}(v_j-w_j))}{2}
  $$
  {{</math>}}
  - [x] 欧氏解码器
   {{<math>}}
  $$
  \textrm{ED}(\pmb{v},\pmb{w})=\sqrt{\sum_j(v_j-w_j)^2}
  $$
  {{</math>}}

- **常见三元解码器**
  - [x] 衰减欧氏解码器
   {{<math>}}
  $$
  \textrm{AED}(\pmb{v},\pmb{w})=\sqrt{\sum_j|w_j|(v_j-w_j)^2}
  $$
  {{</math>}}
  - [x] 基于损失解码器
  {{<math>}}
  $$
  \textrm{LB}(\pmb{x},\pmb{w})=\sum_j L[h_j(\pmb{x}),w_j]
  $$
  {{</math>}}

  ### 6. 混合专家模型

  &emsp;&emsp;混合专家模型采用**分而治之**策略，将一个复杂的任务拆成几个相对简单且更小的子任务，并对这些子任务分别训练个体学习器（专家），最后通过门控来结合这些专家。该模型的**关键问题**在于如何切分任务，并由子问题的解推导出最终解。

  &emsp;&emsp;假设一个混合专家模型由$T$个专家组成，每一个局部专家$h_i$都尝试得到局部输出$h_i(y|\pmb{x};\pmb{\theta}_i)$。门控函数为每个专家提供贡献度权重$\pi_i(\pmb{x};\pmb{\alpha})$。最终混合专家模型的输出可以表示为所有专家局部输出的加权求和，
  {{<math>}}
  $$
  H(y|\pmb{x};\pmb{\Phi})=\sum_{i=1}^T \pi_i(\pmb{x};\pmb{\alpha})\cdot h_i(y|\pmb{x};\pmb{\theta}_i)
  $$
  {{</math>}}
  门控函数的输出通常由softmax函数给出，即
  {{<math>}}
  $$
\pi_i(\pmb{x};\pmb{\alpha})=\frac{\exp(\pmb{v}_i^\top\pmb{x})}{\sum_{j=1}^T\exp(\pmb{v}_j^\top\pmb{x})}
  $$
  {{</math>}}
  通常情况下，训练过程期望达到两个目标：给定专家，找到最优门控函数；给定门控函数，在指定的分布上训练专家。参数一般使用EM算法估计。