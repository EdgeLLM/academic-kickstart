---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Understanding Support Vector Machine"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2020-06-20T20:24:13+08:00
lastmod: 2020-06-20T20:24:13+08:00
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
首先用分类问题对SVM的原理进行解释。现有如下图所示的二分类问题，目的是找到一个合适的超平面有效地区分正号与负号，就可以以此为基准对未知种类的点进行预测。