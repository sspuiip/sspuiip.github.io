---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Diversity Measure"
subtitle: "To measure the consistency of hypotheses"
summary: ""
authors: [admin]
tags: [measure]
categories: [misc]
date: 2024-12-09
lastmod: 2024-12-09
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

## Measuring two hypothesese consistency

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

