---
marp: true
title: Object Detection API
description: Covers KerasCV's object detection API.
theme: uncover
paginate: true
_paginate: false
---

# Object Detection with KerasCV API

---

## ⚠️ API is still experimental ⚠️

---

![bg contain](assets/img/overview.png)

---

# Background

- 1.5~ years ago I wrote a few object detection pipelines
- user experience was not good
- many issues (format mismatch, NaN loss, etc)

---

# Key painpoints

- bounding box formats were hard to manage
- data augmentation
- image shape management
- inherent ragged-ness of bounding boxes
- metric evaluation

---

# Feature Highlights

- TPU compatibility
- Train time COCO metric evaluation
- Native support for ragged bounding box inputs
- bounding box enabled augmentations

---

# API Highlights

- explicit bounding box formats
- highly modular
- ragged native preprocessing and augmentation layers

---

```python
# What format should the bounding boxes be in?
shear = layers.RandomShear(
  factor=0.1,
)
```

---

```python
shear = layers.RandomShear(
  factor=0.1,
  # bounding box format is explicit
  bounding_box_format='xywh'
)
```

---

![bg left]()

```python
# images are ragged
# bounding box correctly augmented
augmenter = [
  layers.RandomFlip(bounding_box_format='xywh'),
  layers.RandomAspectRatio(factor=(0.9, 1.1)),
  layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.8, 1.35),
    bounding_box_format='xywh'
  ),
  layers.MixUp()
]
```

---

# [Demo Colab Notebook](https://colab.research.google.com/drive/1FXF4kT6WNymY5IvBBhkamdFF5NG5C0K3?usp=sharing)
