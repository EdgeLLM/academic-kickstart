---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: ""
summary: "使用神经网络求解偏微分方程"
authors: [Xiaolin Hu]
tags:
- Deep Learning 
categories: 
- PINN PDEs 
date: 2020-07-07T22:07:29+08:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: PINN for Solving Burger Equations
  focal_point: Smart
  preview_only: true

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---
# Physics-Informed Neural Network

物理启发式神经网络(Physics-informed neural networks, PINN) 指的是基于神经网络求解偏微分方程(partial differential equation, PDE)的技术。区别于传统的有限元方法，PINN直接将偏微分方程以及相对应的边界条件作为要学习的模型的约束，从而用数据驱动的方法使得模型逼近方程的解。PINN虽然属于监督学习，但是其训练过程不需要使用任何数值解，因为其标签值实际上是由PDE方程本身以及边界条件给定的。使用神经网络求解偏微分方程的相关探索至少可以追溯到20年前，近几年再次受到大家关注很大程度上得益于TensorFLow、Pytorch和Jax等深度学习框架的出现使得自动微分技术变得触手可及，以及GPU等硬件算力的大幅提升。

## Physics-Informed Neural Network for Solving PDEs
PDE的一般形式如下所示
$$

$$
## Solving the Burgers' Equation 

## Solving the Helmholtz Equation

