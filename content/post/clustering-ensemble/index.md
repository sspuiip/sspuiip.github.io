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

#### 3.3 基于重标记的方法
&emsp;&emsp;该方法的基本思想：校准或重标所有基聚类器的簇标记，使相同的标记指代基聚类器中相似的簇，然后再基于这些已校准的标记生成集成的结果。

- 硬标记对应. 该设置下，所有基聚类器生成相同数量的类簇。

| 硬标记算法 |
|:--- |
|1. 随机选择{{<math>}}$\lambda^{(b)}=\{C_l^{(b)}|l=1,2,...,k\},\lambda^{(b)}\in\Lambda${{</math>}}.<br/>
2. $\Lambda=\Lambda-${$\lambda^{(b)}$}<br/>
3. repeat{ <br/>
4. &emsp;&emsp;随机选择{{<math>}}$\lambda^{(q)}=\{C_l^{(q)} |l=1,2,...,k \}${{</math>}}与$\lambda^{(b)}$对齐。<br/>
5. &emsp;&emsp;利用{{<math>}}$O(u,v)=|C_u^{(b)}\cap C_v^{(q)}| $ {{</math>}}初使化矩阵$O$。<br/>
6. &emsp;&emsp;{{<math>}}$I=\{(u,v)|1\le u,v\le k${{</math>}}。<br/>
7. &emsp;&emsp;repeat{<br/>
8. &emsp;&emsp;&emsp;&emsp;{{<math>}}$(u',v')=\mathop{\arg\max}\limits_{(u,v)\in I} O(u,v)${{</math>}}<br/>
9. &emsp;&emsp;&emsp;&emsp;将{{<math>}}$C_{v'}^{(q)}${{</math>}}重新标记为{{<math>}}$C_{u'}^{(q)}${{</math>}}<br/>
10. &emsp;&emsp;&emsp;{{<math>}}$ I=I-\{(u',w)|(u',w)\in I\} \cup \{(w,v')|(w,v')\in I\} ${{</math>}}<br/>
11. &emsp;&emsp;}until $I=\emptyset$<br/>
12. &emsp;&emsp;$\Lambda=\Lambda-${$\lambda^{(q)}$}<br/>
13. }until $\Lambda=\emptyset$|
重新标记后，使用不同的结合策略获得最终聚类结果。

- 软标记对应




#### 3.4 基于变换的方法
&emsp;&emsp;该方法将每个样本的分类结果表示为一个$r$元组，元组的每元素表示某个基聚类器的类簇标记。通过变换后的元组，可以通过聚类分析得到集成结果。

&emsp;&emsp;例如：4个基聚类器，5个样本的一个聚类结果可表示为，
| 样本簇向量 | 基聚类器1 | 基聚类器2 | 基聚类器3 | 基聚类器4 |
|:---:|:---:|:---:|:---:|:---:|
|  $\lambda^{(1)}$   |  1   | 1    |  2   |  3   |
|  $\lambda^{(2)}$   |  1   | 2    |  2   | 1    |
|  $\lambda^{(3)}$   |  2   | 2    |  3   | 4    | 
|  $\lambda^{(4)}$   |  2   | 2    |  1   | 2    | 
|  $\lambda^{(5)}$   |  3   | 3    |  3   | 3    | 

&emsp;&emsp;第一步：变换的目标是将上述结果转换为$r$元组。即$\varphi :\mathcal{X}\rightarrow \mathbb{R}^r$，
|  样本   | 变换 $\varphi(\pmb{x})$    |
|:---:|:---:|
|  $\pmb{x}_1$   |  $(\varphi_1(\pmb{x}_1,...,\varphi_4(\pmb{x}_1))=(1,1,2,3)$   |
|  $\pmb{x}_2$   |  $(\varphi_1(\pmb{x}_1,...,\varphi_4(\pmb{x}_1))=(1,2,2,1)$   |
|  $\pmb{x}_3$   |  $(\varphi_1(\pmb{x}_1,...,\varphi_4(\pmb{x}_1))=(2,2,3,4)$   |
|  $\pmb{x}_4$   |  $(\varphi_1(\pmb{x}_1,...,\varphi_4(\pmb{x}_1))=(2,2,1,2)$   |
|  $\pmb{x}_5$   |  $(\varphi_1(\pmb{x}_1,...,\varphi_4(\pmb{x}_1))=(3,3,3,3)$   |

&emsp;&emsp;第二步：$r$元组通过聚类获得类簇。例如，可定义相似度如下，再使用$k$-means聚类。
{{<math>}}
$$
\textrm{sim}(\pmb{x}_i,\pmb{x}_j)=\sum_{q=1}^r\mathbb{I}(\varphi_q(\pmb{x}_i)=\varphi_q(\pmb{x}_i))
$$
{{</math>}}

- 第二步可选概率框架

&emsp;&emsp;令$\pmb{y}_i=\varphi(\pmb{x}_i)=(y_1^i,...,y^i_r)^\top$，随机向量$\pmb{y}$可使用一个混合多项式分布表示，

{{<math>}}
$$
P(\pmb{y}|\Theta)=\sum_{j=1}^k \alpha_j P_j(\pmb{y}|\theta_j)
$$
{{</math>}}
其中，$k$是混合成分，对应最终类簇的数量。假设$\pmb{y}$的成分是条件独立的，即
{{<math>}}
$$
P_j(\pmb{y}|\theta_j)=\prod_{q=1}^r P_j^{(q)}(y_q|\theta_j^{(q)})
$$
{{</math>}}
其中，$r$为基聚类器数量。并且，条件概率$P_j(\pmb{y}|\theta_j)$可视为一个多项分布的输出，即
{{<math>}}
$$
P_j(\pmb{y}|\theta_j)=\prod_{l=1}^{k^{(q)}}\vartheta_{qj}(l)^{\delta(y_q,l)}
$$
{{</math>}}
其中，$k^{(q)}$为第$q$个基聚类器的簇数量，$\vartheta_{qj}(l)$为类别$l$的概率。

&emsp;&emsp;基于上述假设，可最大化$m$个$r$元组{{<math>}}$Y=\{\pmb{y}_i|1\le i\le m\}${{</math>}}的似然来获取最优化参数$\Theta^*$，即
{{<math>}}
$$
\begin{split}
\Theta^* &= \mathop{\arg\max}\limits_{\Theta}\log L(Y|\Theta)=\mathop{\arg\max}\limits_{\Theta} \log\left( \prod_{i=1}^m P(\pmb{y}_i|\Theta)\right)\\
&= \mathop{\arg\max}\limits_{\Theta} \sum_{i=1}^m \log \left( \sum_{j=1}^k\alpha_jP_j(\pmb{y}_i|\theta_j) \right)
\end{split}
$$
{{</math>}}

上述概率模型可用EM算法求解。
