---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "多样性度量"
#subtitle: "To measure the consistency of hypotheses"
summary: ""
authors: [admin]
tags: [measure]
categories: [misc]
date: 2024-12-09T12:56:30+08:00
lastmod: 2024-12-09T13:56:30+08:00
featured: false
draft: false


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

## 1. 重要性

&emsp;&emsp;Tumer&Ghosh[1995]，引入$\theta$项来刻画不同学习器之间的联系，得到以下结论：
{{<math>}}
$$
\textrm{err}_{\textrm{add}}^{\textrm{ssv}}(H)=\frac{1+\theta(T-1)}{T}\bar{\textrm{err}}_{\text{add}}(h)
$$
{{</math>}}
等式左边为集成后的期望累加误差，$\bar{\textrm{err}}_{\text{add}}(h)$为不同个体学习器累加误差的期望。通过分析上式可以发现，
- $\theta=0$时，集成后误差降低$T$倍。
- $\theta=1$时，集成后不会有效果提升。

&emsp;&emsp;这说明个体学习器的多样性在集成学习中的重要性。集成学习的成功依赖于个体学习器的精度和多样性之间的平衡。虽然，多样性很重要，但难以准确刻画度量。将集成泛化误差分解为一个与多样性相关的项，是理解多样性的一种重要方式。

### 1.1 误差-分岐分解

&emsp;&emsp;Kroph&Vedelsby[1995]，提出误差-分岐分解，假设有加权平均的集成方式，即
$$
H(\pmb{x})=\sum_{i=1}^T w_ih_i(\pmb{x})
$$
并且，个体学习器$h_i$、集成后$H$的误差和个体学习器分岐分别为，
{{<math>}}
$$
\textrm{err}(h_i|\pmb{x})=(y-h_i(\pmb{x}))^2,\quad \textrm{err}(H|\pmb{x})=(y-H(\pmb{x}))^2,\quad \boxed{\textrm{ambi}(h_i|\pmb{x})}=(h_i(\pmb{x})-H(\pmb{x}))^2
$$
{{</math>}}
由上式三者定义不难得到，
{{<math>}}
$$
\begin{split}
\bar{\textrm{ambi}}(h|\pmb{x}) &\triangleq \sum_{i=1}^T w_i\cdot\textrm{ambi}(h_i|\pmb{x})\\
&= \sum_{i=1}^Tw_i\cdot\textrm{err}(h_i|\pmb{x})-\textrm{err}(H|\pmb{x}), \quad \textrm{i.e.},\\
\boxed{\textrm{err}(H|\pmb{x}) }&= \sum_{i=1}^Tw_i\cdot\textrm{err}(h_i|\pmb{x}) - \bar{\textrm{ambi}}(h|\pmb{x}) 
\end{split}
$$
{{</math>}}
由于上式对于任意$\pmb{x}$成立，因此$\bar{\textrm{ambi}}(h|\pmb{x})$可以在分布$p(\pmb{x})$上求得平均，即
{{<math>}}
$$
\begin{split}
\sum_{i=1}^{T}w_i\int\textrm{ambi}(h_i|\pmb{x})p(\pmb{x})d\pmb{x} &= \sum_{i=1}^{T}w_i\int\textrm{err}(h_i|\pmb{x})p(\pmb{x})d\pmb{x} - \int \textrm{err}(H|\pmb{x})p(\pmb{x})d\pmb{x},\quad \textrm{i.e.,}\\
\boxed{\int \textrm{err}(H|\pmb{x})p(\pmb{x})d\pmb{x}} &= \sum_{i=1}^{T}w_i\int\textrm{err}(h_i|\pmb{x})p(\pmb{x})d\pmb{x} - \sum_{i=1}^{T}w_i\int\textrm{ambi}(h_i|\pmb{x})p(\pmb{x})d\pmb{x}
\end{split}
$$
{{</math>}}

&emsp;&emsp;因此，误差-分岐分解指的是下式，
{{<math>}}
$$
\begin{split}
\textrm{err}(H)&=\int \textrm{err}(H|\pmb{x})p(\pmb{x})d\pmb{x}\\
&=\sum_{i=1}^{T}w_i\int\textrm{err}(h_i|\pmb{x})p(\pmb{x})d\pmb{x} - \sum_{i=1}^{T}w_i\int\textrm{ambi}(h_i|\pmb{x})p(\pmb{x})d\pmb{x}\\
&=\boxed{\bar{\textrm{err}}(h)-\bar{\textrm{ambi}}(h)}\\
\end{split}
$$
{{</math>}}

&emsp;&emsp;可以看出$\bar{\textrm{err}}(h)$是个体学习器的平均误差，由个体学习器的泛化能力决定；$\bar{\textrm{ambi}}(h)$为集成分岐，衡量个体学习器之间预测结果的差异性。因此可以得到以下结论：
- **分岐为正，因此集成后的误差不会比个体学习器的平均误差更高**。
- **个体学习器的精度越高，并且差异性更大，则集成的泛化性会更好**。

&emsp;&emsp;需要注意的是：以上结果是回归问题得到的结论。对于分类问题很难得到类似结果。估计$\bar{\textrm{ambi}}(h)$也会很困难，一般由$\textrm{err}(H)-\bar{\textrm{err}}(h)$而得到，并非真正的多样性。虽有启发，但不能对多样性提供统计的形式化描述。

### 1.2 偏差-方差-协方差分解

&emsp;&emsp;Geman et al. [1992] 将学习器的泛化误差分解为固有噪声、偏差和方差三部分。

{{<math>}}
$$
\begin{split}
\textrm{err}(h)&=\mathbb{E}\left[ (h-y)^2\right]\\
&=(\mathbb{E}[h]-y)^2 + \mathbb{E}\left[(h-\mathbb{E}[h])^2\right]\\
&=\textrm{bias}^2(h)+\textrm{variance}(h)
\end{split}
$$
{{</math>}}
由于固有噪声很难估计，一般被纳入偏差项。因此，泛化误差被分解为偏差项和方差项。
- **偏差项**一般表示学习器误差期望的偏离程度(预测均值与真实目标差的期望)；
- **方差项**表示学习器对不同数据集敏感程度(预测值与预测值均值差的期望)。

&emsp;&emsp;假设个体学习器以相同的权重结合[Ueda & Nakano, 1996]，则集成后的平方误差可以分解为，
{{<math>}}
$$
\textrm{err}(H)=\bar{\textrm{bias}}^2(H)+\frac1T\bar{\textrm{variance}}(H)+\left(1-\frac1T\right)\bar{\textrm{covariance}}(H)
$$
{{</math>}}
其中，
{{<math>}}
$$
\begin{split}
\bar{\textrm{bias}}^2(H) &= \frac1T\sum_{i=1}^T (\mathbb{E}(h_i)-y)\\
\bar{\textrm{variance}}(H) &= \frac1T\sum_{i=1}^T \mathbb{E}[h_i-\mathbb{E}(h_i)]\\
\bar{\textrm{covariance}}(H) &= \frac{1}{T(T-1)}\sum_{i=1}^T\sum_{j=1,j\neq i}^T\mathbb{E}(h_i-\mathbb{E}[h_i])\mathbb{E}(h_j-\mathbb{E}[h_j])
\end{split}
$$
{{</math>}}
上式表明，集成的平方误差依赖于协方差项，体现了不同学习器之间的关联。不同于其它两项，协方差项的系数可以为负。此外，该式由回归设定下推导得到，对分类任务难以取得类似结果，因此也难以成为集成多样性的形式化定义。

## 2. 多样性度量

### 2.1 成对度量

假设有数据集$D=\{(\pmb{x}_1,y_1),...,(\pmb{x}_m,y_m)\}$， 以及任意两个学习器($h_i,h_j$)，并且有以下统计观察结果，


|       |   $h_i=+1$ |  $h_i=-1$ |
| :---: |  :---:     | :---:     |
| $h_j=+1$| a  |  c |
| $h_j=-1$| b  |  d |

其中$a+b+c+d=m$。

- [x] **Divergency measure**

{{< math >}}
$$
\mathrm{dis}_{ij}=\frac{b+c}{m},\quad \in [0,1].
$$
{{< /math >}}

- [x] **Q-statistic**

$$
Q_{ij}=\frac{ad-bc}{ad+bc},\quad \in[-1,1].
$$

- [x] **Correlation**

$$
\rho_{ij}=\frac{ad-bc}{\sqrt{(a+b)(a+c)(c+d)(b+d)}}.
$$

- [x] **Kappa-statistic**

$$
\kappa_{ij}=\frac{\theta_1-\theta_2}{1-\theta_2}
$$
其中,
$$
\theta_1=\frac{a+d}{m},\quad\theta_2=\frac{(a+b)(a+c)+(c+d)(b+d)}{m^2}
$$
注意： 当$\kappa_{ij}=1$时，数据集$D$上的两个学习器完全一致；$\kappa_{ij}=0$时完全独立；$\kappa_{ij}<0$时，学习器达成一致的概率要小于随机预测的期望。

### 2.2 非成对度量

- [x] **Kohavi-Wolpert方差**

&emsp;&emsp;假设个体学习器在类目标$y$上预测的变化性为，
{{<math>}}
$$
\textrm{var}_x=\frac12\left( 1-\sum_{y\in\{+1,-1\}} P(y|\pmb{x})^2 \right)
$$
{{</math>}}
若考虑两个分类器的输出($\tilde{y}=+1$：正确、$\tilde{y}=-1$：错误)和估计$P(\tilde{y}=+1|\pmb{x})$和$P(\tilde{y}=+1|\pmb{x})$来度量多样性，即
{{<math>}}
$$
\hat{P}(\tilde{y}=1|\pmb{x})=\frac{\rho(\pmb{x})}{T},\quad \hat{P}(\tilde{y}=-1|\pmb{x})=1-\frac{\rho(\pmb{x})}{T}
$$
{{</math>}}
其中，$\rho(\pmb{x})$为对样本$\pmb{x}$分类正确的个体学习器数目。将上式代入变化性等式，则有单个样本的变化性度量：
{{<math>}}
$$
\begin{split}
\textrm{var}_x &= \frac12\left( 1-\left[\frac{\rho(\pmb{x})^2}{T^2} + \left(1-\frac{\rho(\pmb{x})^2}{T^2}\right) \right]\right)\\
&=\frac{1}{T^2}\rho(\pmb{x})(T-\rho(\pmb{x}))
\end{split}
$$
{{</math>}}


则有**kw度量**：
{{<math>}}
$$
\boxed{\textrm{kw}=\frac{1}{mT^2}\sum_{k=1}^m \rho(\pmb{x}_k)(T-\rho(\pmb{x}_k))}
$$
{{</math>}}

&emsp;&emsp;可以看出，kw度量的值越大，则差异性越大。

- [x] **评分者一致性**

{{<math>}}
$$
\boxed{\kappa=1-\frac{\frac1T\sum_{k=1}^m \rho(\pmb{x}_k)(T-\rho(\pmb{x}_k)) }  {m(T-1)\bar{p}(1-\bar{p})}}
$$
{{</math>}}
其中，
{{<math>}}
$$
\bar{p}=\frac{1}{mT}\sum_{i=1}^T\sum_{k=1}^m\mathbb{I}(h_i(\pmb{x}_k)=y_k)
$$
{{</math>}}
为个体学习器的平均精度。

- [x] **熵**：如果个体学习器分类结果打平，则不一致性最高。

{{<math>}}
$$
\textrm{Ent}_{cc}=\frac1m\sum_{k=1}^m\sum_{y\in\{+1,-1\}}-P(y|\pmb{x}_k)\log P(y|\pmb{x}_k)
$$
{{</math>}}
其中，
{{<math>}}
$$
P(y|\pmb{x}_k)=\frac1T\sum_{i=1}^T \mathbb{I}(h_i(\pmb{x}_k)\neq y)
$$
{{</math>}}


- [x] **通用多样性**：当一个分类器预测错误伴随着另一个分类器预测正确时，多样性最大。

{{<math>}}
$$
\textrm{gd}=1-\frac{p^{(2)}}{p^{(1)}}
$$
{{</math>}}
其中，
{{<math>}}
$$
p^{(1)}=\sum_{i=1}^T\frac{i}{T}p_i,\quad p^{(2)}=\sum_{i=1}^T \frac{i}{T}\frac{i-1}{T-1}p_i
$$
{{</math>}}
$p_i$代表随机样本$\pmb{x}$在随机挑选的分类器上预测失败的概率。

## 3. 局限性与增强

&emsp;&emsp;大量实验表明，现有的多样性度量效果不如人意，似乎多样性度量指标和集成预测之间并无明显关系[Kuncheva & Whitaker 2003]。但是有一些有效的启发式方法来增强集成多样性。常用作法是在学习过程中引入随机性，
- 扰动样本：采样方法，如AdaBoost。
- 扰动输入特征：不同子空间学习的个体学习器通常是不同的。如随机子空间方法[Ho, 1998]。在子空间训练学习器不仅提高精度，而且加快训练速度。需要提前过滤与目标最不相关的特征。
- 扰动学习参数:使用不同的学习参数来生成多样学习器。
- 扰动输出表示：使用不同的输出表示来生成多样化的学习器。如纠错输出码。

&emsp;&emsp;此外，不同的扰动方法还可以组合使用。如随机森林同时扰动样本和输入特征。

