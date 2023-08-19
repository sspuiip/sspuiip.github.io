---
title: 核方法
subtitle: 使用核方法增强机器学习算法
#summary: Applying kernel methods to machine learning
authors:
  - admin
tags: []
categories: []
projects: []
date: '2023-08-15T00:00:00Z'
lastMod: '2023-08-15T00:00:00Z'
image:
  caption: ''
  focal_point: ''
---

首先，从什么是核函数开始！


## 核函数

了解核函数，先从内积空间开始！

>**定义1(内积(inner product))**. 令 $\mathcal{H}$ 为一个向量空间，则一个函数$\langle\cdot,\cdot\rangle_\mathcal{H}:\mathcal{H}\times\mathcal{H}\rightarrow\mathbb{R}$定义为$\mathcal{H}$的内积，如果满足以下条件：
 1. $\langle\alpha_1 f_1+\alpha_2 f_2,g\rangle_{\mathcal{H}}=\alpha_1\langle f_1,g\rangle_\mathcal{H}+\alpha_2\langle f_2,g\rangle_{\mathcal{H}}$
 2. $\langle f,g\rangle_{\mathcal{H}}=\langle g,f\rangle_{\mathcal{H}}$
 3. $\langle f,f\rangle_{\mathcal{H}}\ge 0 and \langle f,f\rangle_{\mathcal{H}}=0$ if and only if $f=0$

&emsp;&emsp;希尔伯空间是定义了内积的一个空间，并附加了一个技术性条件(A Hilber space is a space on which an inner product is defined, along with an additional technical condition)。



>**定义2(核函数, Kernel)**.  令$\mathcal{X}$为一个非空集合。一个函数 $k:\mathcal{X}\times\mathcal{X}\rightarrow \mathbb{R}$称之为核函数，如果存在一个$\mathbb{R}$-Hilbert空间以及映射 $\phi:\mathcal{X}\rightarrow \mathcal{H}$且满足$\forall x,x'\in \mathcal{X}$，

$$
k(x,x')=\langle\phi(x),\phi(x')\rangle_\mathcal{H}
$$




>**Lemma(Sums of kernels are kernels)**. 给定$\alpha >0$以及 $k,k_1,k_2$为核函数定义在域$\mathcal{X}$, 则有$\alpha k$和$k_1+k_2$都是定义在$\mathcal{X}$的核函数.

>**Lemma(Mapping between spaces)**. 若$\mathcal{X}$ 和$\tilde{\mathcal{X}}$为非空集，且有一个映射$A:\mathcal{X}\rightarrow\tilde{\mathcal{X}}$. 若有$k$定义在域$\tilde{\mathcal{X}}$. 则$k(A(x),A(x'))$是一个定义在$\mathcal{X}$的核函数.

>**Lemma(Products of kernels are kernels)**. 给定$k_1$定义在域 $\mathcal{X}_1$ 以及$k_2$定义在域$\mathcal{X}_2$, 则 $k_1\times k_2$是一个定义在域$\mathcal{X}_1\times\mathcal{X}_2$的核函数。如果$\mathcal{X}_1=\mathcal{X}_2=\mathcal{X}$, 则 $k=k_1\times k_2$ 是一个定义在域$\mathcal{X}$的核函数。

---

