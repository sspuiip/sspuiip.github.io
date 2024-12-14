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

### Pairwise measure

Given a sample dataset $D=\{(\pmb{x}_1,y_1),...,(\pmb{x}_m,y_m)\}$, and two hypotheses ($h_i,h_j$), We have the following statistic observations,


|       |   $h_i=+1$ |  $h_i=-1$ |
| :---: |  :---:     | :---:     |
| $h_j=+1$| a  |  c |
| $h_j=-1$| b  |  d |

where $a+b+c+d=m$.

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
where,
$$
\theta_1=\frac{a+d}{m},\quad\theta_2=\frac{(a+b)(a+c)+(c+d)(b+d)}{m^2}
$$

Note that, two hypotheses predicted on the dataset $D$ are completely consistent at $\kappa_{ij}=1$, and completely independent at $\kappa_{ij}=0$. In case of $\kappa_{ij}<0$, it is mean that the probability that hypotheses reach agreement is less than the expectation of random prediction.