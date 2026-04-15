# SPLITZ: Certifiable Robustness via Split Lipschitz Randomized Smoothing

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 8, 5

## Abstract
Certifiable robustness gives the guarantee that small perturbations around an input to a classifier will not change the prediction. There are two approaches to provide certifiable robustness to adversarial examples-- a) explicitly training classifiers with small Lipschitz constants, and b) Randomized smoothing, which adds random noise to the input to create a smooth classifier. We propose \textit{SPLITZ}, a practical and novel approach which leverages the synergistic benefits of both the above ideas into a single framework. Our main idea is to \textit{split} a classifier into two halves, constrain the Lipschitz constant of the first half, and smooth the second half via randomization. Motivation for \textit{SPLITZ} comes from the observation that many standard deep networks exhibit heterogeneity in Lipschitz constants across layers. \textit{SPLITZ} can exploit this heterogeneity while inheriting the scalability of randomized smoothing. 
We present a principled approach to train \textit{SPLITZ} and provide theoretical analysis to derive certified robustness guarantees. 
We present a comprehensive comparison of robustness-accuracy tradeoffs and show that \textit{SPLITZ} consistently improves upon existing state-of-the-art approaches on MNIST, CIFAR-10 and ImageNet datasets. For instance, with $\ell_2$ norm perturbation budget of $\epsilon=1$, \textit{SPLITZ} achieves $\textbf{61.7\%}$ top-1 test accuracy on CIFAR-10 dataset compared to state-of-art top-1 test accuracy $39.8\%$, a $55.0\%$ improvement in certified accuracy over various approaches (including, denoising based methods, ensemble methods, and adversarial smoothing).

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes SPLITZ, a new approach for training neural networks with certifiable robustness guarantees by exploiting both Lipschitz continuity and randomized smoothing (RS). The key idea is to split the neural network into two parts, the first part would have constrained Lipschitz constants, while the latter layers are smoothed by randomized smoothing. The authors claim that this new method has the advantages of both certified approaches (Lipschitz continuity and RS) in a single framework.  The authors provide a theoretical analysis to compute the robustness radius for SPLITZ classifiers and demonstrate the effectiveness of SPLITZ with experiments on MNIST, CIFAR, and ImageNet.

### Strengths
This idea is interesting, and given that Lipschitz continuity and randomized smoothing are two well-established methods for certifiable robustness, it is interesting to work toward a unification of the two approaches.

### Weaknesses
**Major weakness: I believe the paper to be flawed**

- The results on CIFAR10 show an increase of 21.9 points of certified robustness for eps = 1 with the L2 norm compared to the state of the art. This increase is extremely high and, in my opinion, suspicious. I took the time to check the code and it seems that the authors normalize the inputs of the model. The authors mention this in Appendix E.2 DETAILS OF DATASETS. 

After a review of the code, it seems that the authors use a function called `get_architecture` to initialize the model, this function is defined as:

```
def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
     """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = resnet50(pretrained=False).cuda()
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "mnist_lenet":
        model = lenet().cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
```
 and the `get_normalize_layer` function is:

```
def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]
```

If that's the case, the certified radius should be scaled accordingly, for example, if CIFAR10 images are scaled with 
```
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
```
then if I'm not mistaken, the certified radius should be divided by `1/min(_CIFAR10_STDDEV) ≈ 5.0`. This would explain the very high certified robustness of CIFAR10.

- Regarding the results on ImagetNet, the results are not better but on par with the state-of-the-art DDS approach (Carlini et al. 2023). This again is surprising given that the authors use a simple ResNet50 while Carlini et al. 2023 have used a combined model (diffusion + ViT classifier BEiT large model) with +800M parameters. Again, the normalization problem could explain this discrepancy. 

**Other comments:**
- The proposed approach has already been investigated in the preprint [1]. 
- The authors do not seem to be aware of a large body of work on Lipschitz networks. The sentence "Lipschitz constrained training is often only feasible for smaller (few layers) neural networks" is false. The work of Meunier et al. [2] successfully trained a 1000-layer Lipschitz network by constraining all layers to have a Lipschitz equal to 1. Furthermore, there are many papers that the authors did not acknowledge, the sentence: "The main challenge is that accurate estimation of Lipschitz constants becomes infeasible for larger networks, and upper bounds become loose, leading to empty bounds on the certified radius." is true, but 1-Lipshitz networks actually solve this problem see [2, 3, 4, 5, 6].
- Using MNIST to demonstrate an adversarial robustness approach is not convincing to me due to the nature of the dataset (toy dataset). 

[1] Zeng et al. Certified Defense via Latent Space Randomized Smoothing with Orthogonal Encoders  
[2] Meunier et al., A Dynamical System Perspective for Lipschitz Neural Networks ICML 2022  
[3] Araujo et al., A Unified Algebraic Perspective on Lipschitz Neural Networks, ICLR 2023  
[4] Trockman et al., Orthogonalizing Convolutional Layers with the Cayley Transform, ICLR 2021  
[5] Singla et al, Skew Orthogonal Convolutions, ICML 2021  
[6] Prach et al., Almost-Orthogonal Layers for Efficient General-Purpose Lipschitz Networks, ECCV 2022

### Questions
Can the authors comment on the normalization issue?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present a novel algorithm to obtain certified robustness with high probability (in the sense of randomized smoothing), named SPLITZ. By combining local Lipschitz regularization in the first layer(s) with randomized smoothing in the latent space, the authors improve on the state-of-the-art by a sizeable margin on MNIST and CIFAR-10.

### Strengths
The idea behind the proposed approach is novel and conceptually simple/intuitive: to the best of my knowledge, this is the first work combining Lipschitz-based certified training schemes with randomized smoothing. 
The paper is mostly well-written, with a clear presentation of the required technical background (sometimes in the appendix) and of the main technical building blocks of SPLITZ.
What stands out the most, though, is the experimental section, showing that SPLITZ outperforms previous approaches (even those using additional data) by a significant margin on both MNIST and CIFAR-10, with the performance improvement increasing with the perturbation radius.

### Weaknesses
To my mind, the main weakness of the work lies in the introduction of a fair number of hyper-parameters (train-time $\gamma$, $\theta$, $\lambda$), which will inevitably increase the overall runtime overhead of the proposed approach. Analogously, it would be nice to see a detailed analysis of the overhead incurred by the optimization over $\gamma$ (remark 1).

In addition, I think that the presentation itself could be somewhat improved in a couple of instances. For instance, the authors repeatedly state that certified training is either randomized smoothing or Lipschitz-based methods, somewhat ignoring the family of methods that train against network relaxations (for instance, IBP, CROWN-IBP, and more recent works such as SABR, TAPS, CC/MTL-IBP, etc.): these methods do not have an explicit Lipschitz estimation. Analogously, it is claimed that a large certified radius is equivalent to a small Lipschitz constant, but such is a sufficient rather than necessary condition for robustness. There is a formatting issue in page 8, where the body of the text and the caption become hard to visually separate.
Furthermore, some sections are not entirely self-contained: for instance, the paragraph before equation (9) does not explain how LB/UBs are contained, and defers to the appendix important details of the Lipschitz part of the training.

### Questions
Could the authors provide an indication of the overhead of running SPLITZ (including the hyper-parameter tuning process) with respect to the reported baselines?

The authors refer to $\theta$ as a "learnable" parameter: do you mean tunable hyper-parameter? If so, why does not $\lambda$ suffice? 

Judging from "dataset configuration", it would feel like SPLITZ is quite sensitive to the $\lambda$ schedule. How was this tuned?

Results on MNIST and CIFAR-10 are remarkable. However, how do the authors explain the fact that performance improvements (if any) upon the baselines are significantly smaller on ImageNet?

It would be interesting to hear the authors' opinion as to why the effect of the splitting location on performance increases witht $\sigma$.

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a novel technique for achieving certified adversarial robustness by combining the principles of Lipschitz bounded networks with randomized smoothing. The approach involves partitioning a neural network into two components, where the first is bounded by a local Lipschitz constraint, and the second is robustified through randomized smoothing. The authors present a training procedure designed to reinforce the respective parts—ensuring Lipschitz continuity in the former and noise resilience in the latter. This allows the model to outperform state-of-the-art L2 certificates for image classification. 

The proposed method has been evaluated on image datasets MNIST, CIFAR-10, and ImageNet. It consistently outperforms the state of the art on MNIST and CIFAR-10 datasets. However, on the ImageNet dataset, it only outperforms the state-of-the-art methods that do not use additional data for some of the certified radii. It does not quite outperform the method that uses additional data.

### Strengths
1. The ideas in the paper are articulated with clarity and are easy to follow.
2. The paper proposes a novel technique that combines two established methods with considerable efficacy.
3. Along with theoretical robustness guarantees, it proposes a training procedure to optimize the robustness criteria (Lipschitzness for the first part and robustness to random noise for the second) needed for this method.
4. It outperforms state-of-the-art techniques of certified robustness for MNIST and CIFAR-10 Image classification datasets.

### Weaknesses
1. The method does not consistently outperform existing approaches for ImageNet. Specifically, it does not perform as well as DDS, which leverages additional data to improve robustness.

Despite this, the improvement for smaller datasets is noteworthy. Keeping this in mind, I am leaning toward accepting this paper.

### Questions
Could this technique be adapted to incorporate additional training data (like DDS) to improve its performance on large-scale datasets like ImageNet?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper combines Lipschitz networks with randomized smoothing (RS) to develop the SPLITZ method which splits a classifier into two halves, constrain the Lipschitz constant of the first half, and smooth the second half via randomization. The motivation is that many standard deep networks exhibit heterogeneity in Lipschitz constants across layers, and the proposed method is capable of exploiting this heterogeneity while improving scalability of RS. Training methods and related robustness theory are developed. Numerical results are presented to show that the proposed method achieves good results on MNIST, CIFAR-10 and ImageNet datasets.

### Strengths
1. The proposed method is more scalable than RS.

2. The idea of exploiting heterogeneity in Lipschitz constants across layers is interesting.

3. Numerical study is quite comprehensive.

### Weaknesses
1. The idea of combining Lipschitz networks and RS may not be that original. In general, Lipschitz training is not the only way to restrict the Lipschitz constant of networks. One can enforce various network structures so a prescribed Lipschitz constant is ensured. The following paper which appeared in 2021 combines orthogonal Lipschitz layers with RS:

Huimin Zeng, Jiahao Su, and Furong Huang. Certified defense via latent space randomized smoothing with orthogonal encoders. arXiv2021.

Therefore, on the conceptual level, the paper may not be that novel. Btw, the above paper should be cited. 

2. The advantage of the proposed method over DDS is not that convincing. I mean, from Table 3, it seems that DDS is better for smaller perturbations?

3. When talking about constraining the Lipschitz constant of networks, the authors mainly focus on Lipschitz training and ignore a large body of works that use prescribed network structures to constrain the Lipschitz constants. The following list of papers is relevant and should be discussed:

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. ICLR, 2018.

Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger B Grosse, and Joern-Henrik Jacobsen. Preventing gradient attenuation in lipschitz constrained convolutional networks. NeurIPS, 2019.

Asher Trockman and J Zico Kolter. Orthogonalizing convolutional layers with the cayley transform. ICLR, 2021

Sahil Singla and Soheil Feizi. Skew orthogonal convolutions. ICML, 2021.

Tan Yu, Jun Li, Yunfeng Cai, and Ping Li. Constructing orthogonal convolutions in an explicit manner. ICLR 2022.

Laurent Meunier, Blaise Delattre, Alexandre Araujo, and Alexandre Allauzen. A dynamical system perspective for lipschitz neural networks. ICML 2022.

Bernd Prach and Christoph H Lampert. Almost-orthogonal layers for efficient general-purpose lipschitz networks. ECCV 2022.

Xiaojun Xu, Linyi Li, and Bo Li. Lot: Layer-wise orthogonal training on improving l2 certified robustness. NeurIPS 2022.

Alexandre Araujo, Aaron Havens, Blaise Delattre, Alexandre Allauzen, and Bin Hu. A unified algebraic perspective on lipschitz neural networks. ICLR 2023.

Ruigang Wang, and Ian Manchester. Direct parameterization of lipschitz-bounded deep networks. ICML 2023.

**A big question:Why do not use the above Lipschitz networks for the first half and then integrate RS with the second half? How to compare such a network parameterization approach with the proposed method?**

### Questions
1. The idea of combining Lipschitz networks and RS may not be that new. Can the authors be more specific about the unique conceptual novelty of this paper?

2.  From Table 3, it seems that DDS is better for smaller perturbations? Is it possible to make the proposed method achieve better results than DDS even for small perturbations?

3. As mentioned above, there is a large body of literature on parameterizing networks in certain ways to enforce Lipschitz constraints. Why do not use the above Lipschitz networks for the first half and then integrate RS with the second half? How to compare such a network parameterization approach with the proposed method?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
