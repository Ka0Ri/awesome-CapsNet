# Awesome Capsule Network [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome Capsule Network works
<img src="assets\Routing-plan.png" />

## Table of Contents

- [Awesome Capsule Network ](#awesome-capsule-network-)
  - [Table of Contents](#table-of-contents)
  - [Survey](#survey)
  - [Papers](#papers)
  - [Tutorials](#tutorials)
  - [Brief comparision](#brief-comparision)
  - [Implementations](#implementations)
    - [Tensorflow](#tensorflow)
    - [PyTorch](#pytorch)
- [Awesome](#awesome)
- [License](#license)

## Survey

- [Capsule Networks – A survey](https://www.sciencedirect.com/science/article/pii/S1319157819309322), Menash et al., JKSU 2022 | [bibtex](assets\citations.txt#L5)
- [Capsule networks for image classification: A review](https://www.sciencedirect.com/science/article/pii/S0925231222010657), Pawan et al., Neurocomputing 2022 | [bibtex](assets\citations.txt#L12)

## Papers
<details open>
<summary>Papers by Hinton et al.</summary>

- [Matrix capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb), Hiton et al., ICLR 2018 | [bibtex](assets\citations.txt#L20)
- [Dynamic routing between capsules](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html), Sabour et al., NIPS 2017 | [bibtex](assets\citations.txt#L28)
- [Transforming Auto-encoders](https://www.cs.toronto.edu/~bonner/courses/2022s/csc2547/papers/capsules/transforming-autoencoders,-hinton,-icann-2011.pdf), Hinton et al., ICANN 2011 | [bibtex](assets\citations.txt#L35)
- [Learning to Parse Images](https://proceedings.neurips.cc/paper/1999/hash/5a142a55461d5fef016acfb927fee0bd-Abstract.html), Hinton et al., NIPS 1999 | [bibtex](assets\citations.txt#L52)
- [Stacked Capsule Autoencoders](https://proceedings.neurips.cc/paper/2019/hash/2e0d41e02c5be4668ec1b0730b3346a8-Abstract.html), Adam Kosiorek et al., NIPS 2019 | [bibtex](assets\citations.txt#L44)
- [Detecting and diagnosing adversarial images with class-conditional capsule reconstructions](https://arxiv.org/abs/1907.02957), Yao Qin et al., ICLR 2020 | [bibtex](assets\citations.txt#L60)
- [Canonical capsules: Self-supervised capsules in canonical pose](https://canonical-capsules.github.io), Weiwei Sun et al., NIPS 2021 | [bibtex](assets\citations.txt#L67)
- [Unsupervised part representation by flow capsules](https://proceedings.mlr.press/v139/sabour21a.html), Sabour et al., PMLR 2021 | [bibtex](assets\citations.txt#L76)
- [Darccc: Detecting adversaries by reconstruction from class conditional capsules](https://arxiv.org/abs/1811.06969), Nicholas Frosst et al., | [bibtex](assets\citations.txt#L76)
</details>


## Tutorials

<details open>
<summary>Blogs</summary>

- [Capsule neural network](https://en.wikipedia.org/wiki/Capsule_neural_network), Wikipedia. 
- [Understanding Hinton’s Capsule Networks series](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b), Max Pechyonkin, Medium 2017.
- [Uncovering the Intuition behind Capsule Networks and Inverse Graphic](https://hackernoon.com/uncovering-the-intuition-behind-capsule-networks-and-inverse-graphics-part-i-7412d121798d),  Tanay Kothari, Hackernoon 2017.
- [A Visual Representation of Capsule Connections in Dynamic Routing Between Capsules](https://medium.com/@mike_ross/a-visual-representation-of-capsule-network-computations-83767d79e737), Mike Ross, Medium 2017.
- [What is a CapsNet or Capsule Network?](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc) - Debarko De, Hackernoon 2017.
- [Capsule Networks Are Shaking up AI — Here’s How to Use Them](https://hackernoon.com/capsule-networks-are-shaking-up-ai-heres-how-to-use-them-c233a0971952), Nick Bourdakos, Hackernoon 2017.
- [Understanding Capsule Networks — AI’s Alluring New Architecture](https://medium.freecodecamp.org/understanding-capsule-networks-ais-alluring-new-architecture-bdb228173ddc) - Nick Bourdakos, Medium 2018.
- [Capsule Networks Explained](https://kndrck.co/posts/capsule_networks_explained/), Kendrick Tan.
- [Understanding Dynamic Routing between Capsules (Capsule Networks)](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/), Jonathan Hui, Blog 2017.
- [Matrix capsules with EM routing](https://blog.acolyer.org/2017/11/14/matrix-capsules-with-em-routing/), Adrian Colyer, Blog 2017.
- [Capsule Networks: A Quick Primer](https://blog.paperspace.com/capsule-networks/), Vihar Kurama, Paperspace 2020.
- [Capsule Networks: The New Deep Learning Network](https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8), Aryan Misra, Towardsdatascience 2019.

</details>

<details open>
<summary>Lectures</summary>

- [Capsule Networks (CapsNets) – Tutorial](https://www.youtube.com/watch?v=pPN8d0E3900), Aurélien Géron, Youtube 2017.
- [How to implement CapsNets using TensorFlow](https://www.youtube.com/watch?v=2Kawrd5szHE), Aurélien Géron, Youtube 2017.
- [Capsule network explained](https://www.youtube.com/watch?v=v0tgo3c_7Xs), Count From Zero, Youtube 2021
- [Geoffrey Hinton Capsule theory](https://www.youtube.com/watch?v=6S1_WqE55UQ), 2017.
- [Geoffrey Hinton – Capsule Networks](https://www.youtube.com/watch?v=x5Vxk9twXlE), 2018.
- [Capsule Networks for Computer Vision](https://www.crcv.ucf.edu/cvpr2019-tutorial/), CVPR 2019.
</details>

## Brief comparision

| Reseach   |      Main results      |  Innovation |
|----------|:---------------|:------:|
| [PathCapsNet]() |  Mnist: 99.65 | fan-in routing technique, deep parallel multi-path |
| [Fast Dynamic Routing]() |    smallNORB: 97.4 <br /> MNIST:  99.58 <br /> FMnist: 94 <br /> CIFAR10: 84.6 |   weighted kernel density estimation |
| [Spiking CapsNet]() | MNIST: 99.17 <br /> FMnist: 91.07 |    Spiking Neural Networks |
| [Self-attention CapsNet ]() | CIFAR-10: 92.14 <br /> SVHN: 96.88 <br /> SmallNORB: 92.38|    Self-attention routing |
| [CVAECapOSR]() | TinyImageNet: 71.5 <br /> CIFAR10: 83.5 <br /> Mnist: 99.2 <br /> SVHN: 95.6 |    Conditional Variational |
| [HitNet]() | Mnist: 99.68 <br /> FMnist: 92.3 <br /> CIFAR10: 73.3 <br /> SVHN: 94.5 <br /> affNIST: 83.03|    Hit-Miss layer, Ghost Capsule |
| [Max-Min routing]() | Mnist: 99.55 <br /> FMnist: 92.07 <br /> CIFAR10: 75.92|   replace Softmax with Max-Min Normalization |
| [Two-phase Dynamic Routing]() | SVHN: 90.19 <br /> FMnist: 90.96 <br /> CIFAR10: 75.82 | Micro and Macro-level routing |
| [Inverted dot-product ]() | CIFAR10: 85.17 <br /> CIFAR100: 57.32 |    inverted dot-product attention routing |
| [G-CapsNet]() | Mnist: 99.34 |    trainable routing procedure |
| [Efficient-CapsNet ]() | Mnist: 99.74 <br /> smallNORB: 97.66 <br /> MultiMNIST: 88.75|    Self-attention routing |
| [Res-CapsNet]() | Mnist: 99.4 <br /> FMnist: 89.2 <br /> SVHN: 92.4 <br /> SmallNORB: 90.3|    residual connections |
| [DCNet]() |Mnist: 99.75 <br />  SVHN: 96.90 <br />  CIFAR10: 89.32 <br />  SmallNORB: 95.27 |   Dense connections |
| [DeepCaps]() | CIFAR10: 92.74 <br /> SVHN: 97.56 <br /> FMnist: 94.73|    3D Capsule convolutions |
| [NASCaps]() | CIFAR10: 76.46 <br /> Mnist: 99.7 <br /> FMnist: 93.87 <br /> SVHN: 96.59|    Neural Architecture Search for CapsNet |
| [EncapNet]() | CIFAR10: 95.45 <br /> CIFAR100: 73.33 <br /> SVHN: 97.99 <br /> h-ImageNet: 59.95|    master branch for primary information and an aide branch for pattern |
| [PT-CapsNet]() | CIFAR10: 95.71 <br /> CIFAR100: 78.36 <br /> FMnist: 95.99 <br /> ISIC2018: 83.12 <br /> VOC: 78.2|    more difficult vision tasks |
| [DeeperCaps]() | CIFAR10: 81.29 <br /> smallNORB: 91.75 <br /> Mnist: 99.84|    capsule pool |
| [MT-CapsNet]() | CIFAR10: 92.96 <br /> FMNIST: 94.25|   The Multi-Lane Capsule Network |
| [MS-CapsNet]() | FMNist: 92.7 <br /> CIFAR10: 75.7|   multi-scale feature extraction |
| [DE-CapsNet]() | CIFAR10: 92.96 <br /> FMNIST: 94.25|    Spatial Group-wise Enhance mechanism |
| [VB-Caps]() | smallNORB: 98.4 <br /> FMNIST: 94.8 <br /> SVHN: 96.1 <br /> CIFAR10: 89.8|    Variational Bayes |
| [Capsule-VAE]() | smallNORB: 96.3 <br /> affNIST: 94.08 <br /> Mnist: 99.02 <br /> SVHN: 94.02|    Spatial Group-wise Enhance mechanism |
| [SparseCaps]() | affNist: 90.12 <br /> Mnist: 99|    Unsupervised sparsening of latent capsule layer |
| [GraCapsNets]() | Mnist: 99.50 <br /> FMnist: 93.1 <br /> CIFAR10: 82.21| Multi-head attention-based Graph Pooling approach incorporates built-in explanation |
| [STAR-Caps]() | MNIST: 99.49 <br /> SmallNORB: 95.72 <br /> CIFAR10: 91.23 <br /> CIFAR100: 67.66| straight-through attentive routing, differentiable binary routers |
| [Group-Caps]() | Mnist: 98.42 <br /> AffNist: 89.1| group equivariant capsule network |
| [Em routing]() | smallNORB: 98.2 <br /> Mnist: 99.54 <br /> CIFAR10: 88.1| EM-based routing mixture coefficients |
| [Dynamic routing]() | Mnist: 99.75 <br /> MultiMNIST: 94.8 <br /> CIFAR10: 89.4 <br /> smallNORB: 97.3 <br /> SVHN: 95.7| Cosine based routing coefficient |
| [shortcut routing]() | Mnist: 99.57 <br /> affNist: 89.02 <br /> smallNorb: 94.77 <br /> FNist: 92.18 <br /> | shortcut connection and fuzzy routing|




## Implementations

### Tensorflow

- [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)
- [bourdakos1/capsule-networks](https://github.com/bourdakos1/capsule-networks)
- [JunYeopLee/capsule-networks](https://github.com/JunYeopLee/capsule-networks)
- [jaesik817/adv_attack_capsnet](https://github.com/jaesik817/adv_attack_capsnet)
- [thibo73800/capsnet-traffic-sign-classifier](https://github.com/thibo73800/capsnet-traffic-sign-classifier)
- [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)
- [gusgad/capsule-GAN](https://github.com/gusgad/capsule-GAN)
- [gyang274/capsulesEM](https://github.com/gyang274/capsulesEM)
- [www0wwwjs1/Matrix-Capsules-EM-Tensorflow](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow)

### PyTorch

- [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
- [higgsfield/Capsule-Network-Tutorial](https://github.com/higgsfield/Capsule-Network-Tutorial)
- [danielhavir/capsule-network](https://github.com/danielhavir/capsule-network)
- [Ka0Ri/Capsule-Network](https://github.com/Ka0Ri/Capsule-Network)
- [shzygmyx/Matrix-Capsules-pytorch](https://github.com/shzygmyx/Matrix-Capsules-pytorch)

# Awesome
[Capsule Network](https://github.com/sekwiatkowski/awesome-capsule-networks)

[Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision/tree/master)

[NeRF](https://github.com/awesome-NeRF/awesome-NeRF)

# License 
MIT