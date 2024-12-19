---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "聚类集成"
#subtitle: "To measure the consistency of hypotheses"
summary: ""
authors: [admin]
tags: [ensemble]
categories: [misc]
date: 2024-12-19T10:20:30+08:00
lastmod: 2024-12-19T10:20:30+08:00
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



&emsp;&emsp;本文探讨聚类集成学习的一些方法。

### 1. 聚类

&emsp;&emsp;聚类的目标是将无监督数据划分为若干个簇，利用这些簇结构来探索数据的内在结构。簇划分的一些常见方法有：划分方法、层次方法、基于密度的方法、基于网格的方法以及基于模型的方法等。

- [x] **划分方法**

&emsp;&emsp;优化划分标准，将数据集$D$划分为$k$个簇。如$k$均值的优化目标是平方误差，即
$$
\textrm{err}=\sum_{i=1}^k\sum_{\pmb{x}\in\mathcal{C}_i} \textrm{dist} ( \pmb{x},\bar{C}_i)
$$


### 3. 聚类集成方法

&emsp;&emsp;聚类集成主要有以下三个方面的效果：
  * 提高聚类质量。聚类方法本身具有很多内在随机因素，生成多样性强的聚类结果相对容易。
  * 提高聚类鲁棒性。
  * 知识重用和分布式计算。

&emsp;&emsp;聚类集成主要有分二步实现：
- **聚类生成**。

&emsp;&emsp;基聚类器$\mathcal{L}^{(q)}$将$D$分为$k^{(q)}$个簇{{<math>}}$\{C_j^{(q)}|j=1,2,...,k^{(q)} \}${{</math>}}，聚类结果可以表示为标记向量$\lambda^{(q)}\in\mathbb{N}^m$，即
{{<math>}}
$$
\lambda_i^{(q)}\in\{1,2,...,k^{(q)} \}
$$
{{</math>}}表示为样本$\pmb{x}_i$在基聚类器$\mathcal{L}^{(q)}$下的类簇指派。

- **聚类结果聚合**。

&emsp;&emsp;给定$r$个聚类器指派结果{$\pmb{\lambda}^{(1)},\pmb{\lambda}^{(2)},...,\pmb{\lambda}^{(r)}$}，使用某种聚合函数$\Gamma()$将聚类结果合并成最终包含$k$个簇的聚类结果$\pmb{\lambda}=\Gamma(\pmb{\lambda}^{(1)},\pmb{\lambda}^{(2)},...,\pmb{\lambda}^{(r)})$。

&emsp;&emsp;集成的关键是如何表达和聚合每次个基聚类器的信息。大致可以分为4种类型：
1. 基于相似度的方法
2. 基于图的方法
3. 基于重标记的方法
4. 基于变换的方法

#### 3.1 基于相似度的方法
&emsp;&emsp;利用基聚类器形成一个$m\times m$的一致相似度矩阵，然后基于该矩阵生成最终聚类结果。相似度矩阵的元素一般为，
{{<math>}}
$$
M_{ij}=\textrm{sim}(\pmb{x}_i,\pmb{x}_j)
$$
{{</math>}}

&emsp;&emsp;具体步骤伪码：
```python
        for q in range(r):
          lambda_q = learn_q(D)  #基聚类器结果
          M_q = f(lambda_q)
        M = 1/r * [sum(M_i) for i in range(r)] 
        lambda_ = learn(M)       #常规聚类算法
```
&emsp;&emsp;$M^{(q)}$的两种获取方式：
- **硬聚类**。 每个样本只属于一个簇。如$k$均值聚类。$M^{(q)}$可按此规则设置：
{{<math>}}
$$
M^{(q)}(i,j)=\left\{\begin{array}{ll}1, & \lambda_i^{(q)}=\lambda_j^{(q)};\\ 0, & otherwise. \end{array} \right.
$$
{{</math>}}
- **软聚类**。 每个样本可属于多个簇，归属的程度可以介于$[0,1]$之间。例如可以用$P(l|i)$来表示样本$\pmb{x}_i$属于类簇$l$的程度。$M^{(q)}$可按此规则设置:
{{<math>}}
$$
M^{(q)}(i,j)=\sum_{l=1}^{k^{(q)}}P(l|i)\cdot P(l|j)
$$
{{</math>}}

&emsp;&emsp;优缺点：易于实现，概念简单。缺点在于效率低下，只适合中小规模问题。

#### 3.2 基于图的方法
&emsp;&emsp;构造一个图$G$来整合聚类信息，分解图获取最终聚类集成的结果。图的生成主要有三种策略：$V=D, V=C, V=D\cup C$。

1. $V=D$. HGPA是这一类方法的代表。HGPA将每个$c\in C$视为对集合中点进行连接的超边，并将其加入到边集$E$中。然后使用HMETIS等软件包分割得到集成结果。具体步骤如下：
```python
        V=D
        E=emptyset
        for i in range(k):
          E = E \cup {C_i}
        G=(V,E)
        lambda_=HMETIS(G)
```
2. $V=C$. 图中每个顶点$v$对应一个类簇$c\in C$。$E$中的边为连接不同基聚类器中的两个点的普通边。MCLA是这类算法的代表。具体步骤如下：
```python
        V=C
        E=emptyset
        for i in range(k):
          for j in range(k):
            if(C_i,C_j属于不同的基聚类类簇):
              E=E \cup {e_ij}
              w_ij = |C_i \cap C_j |/(|C_i|+|C_j|-| C_i\cap C_j|)
        
        C_1',C_2',...,C_k' = METIS(G)
        for p in range(k):
          for i in range(m):
            h_pi = sum([I(x_i \in c)/C_p])
        for i in range(m):
          lambda_i = argmax_p (h_pi)
```

3. $V=D\cup C$. $V$中的顶点对应一个样本$\pmb{x}_i$或一个类簇$c\in C$。$E$中的每条边都是$D$中的一点到$C$中一点的边。HBGF为代表算法。具体步骤如下：
```python
        V = D \cup C
        E = emptyset
        for i in range(m):
          for j in range(k):
            if (v_i \in v_j):     #v_i为样本，v_j为类簇。
              E = E \cup {e_ij}
              w_ij = 1
        lambda_ = learn(G)        #METIS, SPEC等
```

&emsp;&emsp;**缺陷**：集成的性能依赖于图分割算法。
