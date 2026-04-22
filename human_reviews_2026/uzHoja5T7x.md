# Learning Equivariant Models by Discovering Symmetries with Learnable Augmentations

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Recently, a trend has emerged that favors shifting away from designing constrained equivariant architectures for data in geometric domains and instead (1) modifying the training protocol, e.g., with a specific loss and data augmentations (soft equivariance), or (2) ignoring equivariance and inferring it only implicitly.
However, both options have limitations, e.g., soft equivariance still requires a priori knowledge about the underlying symmetries, while implicitly learning equivariance from data lacks interpretability. To address these limitations, we propose SEMoLA, an end-to-end approach that jointly (1) discovers a priori unknown symmetries in the data via learnable data augmentations, and uses them to (2) encode the respective approximate equivariance into arbitrary unconstrained models. Hence, it enables learning equivariant models that do not need prior knowledge about symmetries, offer interpretability, and maintain robustness to distribution shifts. Empirically, we demonstrate the ability of SEMoLA to robustly discover relevant symmetries while achieving high prediction performance across various datasets, encompassing multiple data modalities and underlying symmetry groups.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose SEMoLA, a method for unsupervised symmetry discovery consisting of two components: 1). An augmenter which learns a Lie algebra basis and samples augmentations which are applied to input data; and 2). A secondary network which takes in both original and augmented data together to make predictions. Experiments are performed on several datasets.

### Strengths
To my knowledge the composition of the augmenter with an unconstrained model is novel, and the authors provide extensive ablations in the supplement to justify their approach.

### Weaknesses
Unfortunately, there are several problems with this paper.  

- First, the authors do not consider non-connected Lie groups and state that this is consistent with prior work. This is not true -- the authors   fail to cite or referece the following recent work: 

  1). Neural Fourier Transform: A General Approach to Equivariant Representation Learning (Koyama et al, ICLR 2024) https://openreview.net/ forum?id=eOCvA8iwXH

  2). Neural Isometries: Taming Transformations for Equivariant ML (Mitchel et al, NeurIPS 2024) https://arxiv.org/pdf/2405.19296

  in which both of these models are shown to discover symmetries corresponding to non-connected and non-compact Lie groups (e.g. SL(3, R) and SL(2, C)).  Both of these methods appear to handle much more challenging cases than the proposed approach and go further in considering a variety of different applications, including some with real world data.

- To this point, the experiments are unconvincing. Two out of the three experiments in the main paper consider very simple, synthetic datasets (Rotated MNIST, and N-body dynamics) and the proposed method is outperformed by EMLP + LieGAN on the remaining dataset. Furthermore, the symmetry groups considered are very simple (SO(2), SE(3)) and are already well-studied in the symmetry discovery literature. 

- Other outstanding problems include a poor description of the actual approach itself. Specifically, the mechanics of the proposed approach are unclear. For instance, how is there assumed to be a well defined action of $g$ on $\hat{y}$ as implied in Equation (2). Furthermore the loss in Equation (1) contains five terms which seem to require careful balancing, which calls in to question both  the robustness and soundness of the method.

Overall, the proposed method does not appear to move the field of symmetry discovery forward in a significant way, and so I do not recommend acceptance.

### Questions
How do you ensure that Lie algebra actually forms a basis, and is closed under the Lie bracket? If it is not closed, then it isn't really a true Lie algebra basis. I am aware that the LieGAN line of work also does not ensure this, but it seems like an important outstanding theoretical problem that needs to be addressed when learning generators.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new method that takes an unconstrained base model and finds the symmetry and its extent underlying a dataset by end-to-end joint learning of the base model and group representations. This is done by restricting the scope to connected Lie groups with surjective Lie exponential, and learning the basis of the Lie algebra, such that data augmentations are produced from uniformly distributed coefficients under the basis. While this approach is similar to Augerino, a main difference is that the method does not assume fixed basis directions. The authors demonstrate that the proposed method outperforms Augerino+ and LieGAN and is competitive with the ground truth data augmentation in MNIST under SO(2), 2-body dynamics under SO(2), and QM9 under SE(3), including the cases where the training set contains only a restricted set of augmentations.

### Strengths
S1. The paper tackles the challenging problem of jointly discovering symmetry and its extent from data in the form of data augmentation, which is jointly used with an unconstrained model to produce (approximately) equivariant predictions.

S2. The empirical results show that the proposed method outperforms Augerino and LieGAN in the three experimented setups.

### Weaknesses
W1. A main weakness of the work is that the experimented setups only consider Lie groups with low-dimensional Lie algebras and with numerically well-behaved generators (SO(2), SE(3), and small permutation groups), such that it is hard to verify whether the method can indeed discover symmetries in nontrivially hard problem instances, e.g., discovering affine transformations and/or homographies from transformed MNIST images [1], and/or discovering Lorentz transformations from jet tagging as in the LieGAN paper.

W2. It was unclear to me whether, for data dimension $d$ (e.g., $H\times W$ for monochrome images), the group action is assumed to be unknown linear maps on $\mathbb{R}^d$, or is assumed to act homogeneously on coordinates in $\mathbb{R}^2$ or $\mathbb{R}^3$. If it is the latter, then I believe the algorithm has access to a nontrivial amount of knowledge about data transformations prior to the learning (specifically, low-dimensionality and factorization structure of the action), and it is unclear whether we have similar amounts of information in practical setups when symmetry is unknown.

W2. Line 154-155: As far as I know, it is not true for every connected (matrix) Lie group that any element can be written as the output of Lie exponential. In case of noncompact groups the exponential map can be nonsurjective. A counterexample can be found in [2]. This can be a limiting factor of the applicability of the method.

[1] MacDonald et al., Enabling Equivariance for Arbitrary Lie Groups, CVPR 2022.

[2] https://math.stackexchange.com/questions/348699/showing-that-the-exponential-map-mathrmexp-mathfraksl2-mathbbr-to-ma

### Questions
I have no particular questions but would like to hear the authors' response to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method to discover symmetries from data by learning augmentations. Specifically, the first module, LieAugmenter, learns a Lie algebra basis from the data. Lie group elements are then sampled and applied to the original data. The second module then takes in the augmented inputs and learns a task-specific function. The method uses several regularization terms to learn the correct symmetry. In experiments on RotMNIST, N-body dynamics, and QM9, SEMoLA outperforms other baselines such as LieGAN or Augerino.

Overall, this paper provides a good contribution to the area of symmetry discovery, beating current baselines.

### Strengths
- The method doesn't seem to rely on the distribution of Lie algebra basis coefficients and can use a uniform distribution.
- SEMoLA forgoes adversarial training leading to more stability and can be used end-to-end with the task function
- There are extensive experiments on various datasets and scenarios, including the experiments in the appendix.

### Weaknesses
- One small weakness is that this method relies on connected Lie groups. However, this is a common assumption taken in many other symmetry discovery papers as well.
- There are numerous regularization terms to balance. How sensitive is this method w.r.t. to the regularization coefficients?
- The training times seem somewhat longer than other baselines. Is this because the model needs to train on K augmented samples? Is the method sensitive to the value of K?
- Doesn't Augerino also learn a Lie algebra basis as done in SEMoLA? Or is it only learning ranges of augmentations?
- I believe L-conv (Dehmamy et al. 2021) also learns a Lie algebra basis in the conv layer. Does decoupling the learning of the basis with the task as done here (LieAugmenter + unconstrained model) compared to learning them together as in L-conv have a big impact on performance?
- In order to learn discrete groups (as in Appendix A.8.1) was it necessary to modify the distribution of the Lie algebra coefficients?

### Questions
See questions.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a novel framework (SEMoLA) for discovering continuous symmetries directly from data and leveraging them as learned data augmentations during both training and testing. Instead of assuming a prior distribution over such group transformations, the proposed framework learns a Lie algebra basis whose exponential maps can be used to sample data augmentation applied to the model's input and output. The authors propose to jointly optimize for both the performance on the downstream tasks and the learned symmetry through a multi-task objective.  SEMoLA is empirically evaluated across multiple domains, showcasing how the proposed method can benefit the task's performance, while also providing interpretability regarding the symmetries of the given dataset and tasks.

### Strengths
- The proposed framework is agnostic to the choice of the network architecture and can be easily incorporated in a large range of possible tasks and models.
- The authors conduct a comprehensive evaluation of the different desing choices, including detailed ablation studies that assess the contribution of each component.
- The ability of the framework to provide interpretable learned augmentations can also be a useful tool in tasks where the main goal is to analyze or verify underlying symmetries, rather than solely optimize the performance of a downstream task.

### Weaknesses
- By fixing $\rho_y(g)=g$ or $\rho_y(g)=I$, the authors limit the framework to only learn the general group acting on the network's output rather than the representation acting on it. This assumption limits the applicability of the method in settings where the relevant representation is not obvious, which is common in many cases where this method could be impactful, since the appropriate transformation is unknown a priori
- The experiments focus mainly on datasets with useful and interpretable symmetries, where discovering the underlying transformation naturally benefits the task. However, the paper does not explore cases where such a transformation can be detrimental to the task, and the imposed regularizers that promote non-trivial symmetry discovery can harm the task.

### Questions
- Applicability without known representations: Is the method applicable in cases where the representation acting on the output or input is not known a priori? For example, in a typical image encoder, where we do not know if the output feature map should be interpreted as a scalar map or an equivariant vector field, how would SEMoLA handle such ambiguity?
- Behavior in the absence of true symmetries: What happens when no useful transformations are present in the data? Could the framework discover spurious symmetries that negatively affect generalization or stability?

### Soundness
3

### Presentation
3

### Contribution
2
