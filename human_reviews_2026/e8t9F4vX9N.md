# Symmetric Space Learning for Combinatorial Generalization

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6

## Abstract
Combinatorial generalization (CG)—generalizing to unseen combinations of known semantic factors—remains a grand challenge in machine learning. 
While symmetry-based methods are promising, they learn from observed data and thus fail at what we term $\textbf{symmetry generalization}$: extending learned symmetries to novel data. 
We tackle this by proposing a novel framework that endows the latent space with the structure of a $\textbf{symmetric space}$, a class of manifolds whose geometric properties provide a principled way to extend these symmetries. 
Our method operates in two steps: first, it imposes this structure by learning the underlying algebraic properties via the $\textbf{Cartan decomposition}$ of a learnable Lie algebra. 
Second, it uses $\textbf{geodesic symmetry}$ as a powerful self-supervisory signal to ensure this learned structure extrapolates from observed samples to unseen ones. 
A detailed analysis on a synthetic dataset validates our geometric claims, and experiments on standard CG benchmarks show our method significantly outperforms existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper explores a method for so-called combinatorial generalization. The idea is to assume that the data lies on a manifold $M$ which has the form of a symmetric space, i.e. has the structure $G/K$, where $K \subseteq G$ are Lie groups. $G$ and $K$ are thereby unknown. The authors utilize that such a structure induces a decomposition of the Lie algebra $\mathfrak{g}$ of $G$ into spaces $\mathfrak{l}$ and $\mathfrak{p}$ corresponding to the Lie algebra of the subgroup $K$ and the tangent space of the manifold $M$. Flipping a vector $P\in \mathfrak{p}$ thereby corresponds to changing direction of a geodesic through a chosen origin on $M$. 

The authors use facts about the structure of $\mathfrak{l}$ and $\mathfrak{p}$ ($[\mathfrak{l},\mathfrak{l}]\subseteq \mathfrak{l}, [\mathfrak{p},\mathfrak{l}]\subseteq \mathfrak{p}, [\mathfrak{p},\mathfrak{p}]\subseteq \mathfrak{l}]$), and the fact that flipping the $P$-vector corresponds to changing time-direction in geodesics to design loss functions for a learning framework. This is subsequently incorporated into a flow-matching model to produce state-of-the art results on several datasets.

### Strengths
The framework the authors propose is grounded in mathematical theory and is elegant.The models have been tested on multiple dataset, and fare well on all of them.

### Weaknesses
Let me start by saying that I have little experience on the problem of combinatorial generalization, so I am very open to the authors correcting me/ critiquing my assessment.

With this said, I think that the main weakness of this paper in that it suffers in clarity and reproducibility. This is the main reason for my rating.

1. The 3D sphere shape manifold reconstruction data is not properly described. It is not describe how the training and data set are generated. In fact, it is to me not clear whether the point cloud in (a) is the entire dataset or if (a) is a point in the training dataset -- same for (b). Given the description of the dataset in the appendix, I am inclined to believe that (a) is the entire data set, and that each point in the cloud is a training example, but how is then a PointNet used to process the data point?

It is not clear what the generation process of the points in (a) is, and how the ground truth (b) is fed into the models. If I understand e.g. the dSprites datasets correctly, the net is given a set of latent parameters it has not seen before (shape,scale, orientation, position-x,position-y) and its task is to generate an image corresponding to the latent parameters. What are the latent parameters here? 

It should also be noted that the authors have provided no code.

2. I do not understand entirely understand to which extent the GSC-loss is encouraging the decoder to be geodesic-symmetric. In fact, as formulated in 3.2, the loss 'only' encourages the encoder E and decoder D to satisfy $E\circ D\approx \mathrm{id}$ on the flipped versions of the latent vectors of the datasets. Can the authors provide any experimential insight to whether this is actually what happens in practice? This is particularly interesting given that the ablation studies point to the GSC-loss being important.

 In my opinion, providing some type of experimental evidence here is crucial. If the GSC-loss is not actually encouraging the decoder to be geodesic-symmetric, the theoretical considerations of the paper are not providing and explaination to the increased performance of the model.

### Questions
See weaknesses. 

Also, the symmetry techniques do not bring any increase in performance on the R2R task on the 3D Shapes dataset. Can the authors comment on why they think this is?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose a method for addressing the issue of symmetry generalization. The method operates under the assumption of symmetric spaces, using Cartan decomposition and learning a Lie algebra. The geodesic symmetry property of symmetric spaces is used to extend/extrapolate symmetries to unobserved samples in the complete data space--that is, the space in which the observed data lives. The authors also conduct experiments in which their method appears to outperform baselines.

### Strengths
1. This paper is built upon a strong mathematical foundation.

2. The experimental results demonstrate the advantages of the proposed methodology compared with the chosen baselines.

3. This paper demonstrates potential for acceptance, provided that the perceived weaknesses can be addressed.

### Weaknesses
1. *The issue of symmetry generalization is not properly motivated.* The authors cite a single paper from which they draw the conclusion that current symmetry-based machine learning methods are limited in that learned symmetries do not extend beyond observed samples. In light of previous work that is not considered in this paper, the statement on the symmetry generalization challenge is actually not true. The authors are encouraged to discuss their statement in light of additional recent work on symmetry discovery, particularly for the following sources: "Generative Adversarial Symmetry Discovery" (Yang et al., 2023), "Learning Infinitesimal Generators of Continuous Symmetries from Data" (Ko et al., 2024), and "Symmetry Discovery Beyond Affine Transformations" (Shaw et al., 2024). On the other hand, the authors may be able to justify the exclusion of certain sources. But as written, the symmetry generalization statement does not appear to be reflective of recent work.

2. *Additional methods to compare with.* From an experimental standpoint, it seems there are other geometry-inspired autoencoders that should be compared with. One such method is "Geometry Regularized Autoencoders" (Duque et al., 2023), and another is "Geometry-aware generative autoencoders for warped riemannian metric learning and generative modeling on data manifolds" (Sun et al., 2024).

### Questions
1. It seems that the perceived issue of symmetry generalization is being solved primarily by making restrictive assumptions about the structure of the symmetry group. Generally, restrictive assumptions can make problems more tractable, but I am concerned that the limitation to transitive group actions is too restrictive. After all, the action of rotations in the plane about the origin is not a transitive group action: this seems like one of the simplest possible symmetries to consider. Can the authors speak to this?

2. The notion that there are points in the complete space that are not mapped to via the observed group acting on the observed data seems to be a model assumption, which assumption seems equally valid to the assumption that training data defines the manifold and/or distribution of data. It seems that the notion that test data is not only out of distribution, but also that it aligns with the symmetric space model, is a model assumption that is not motivated by information obtainable from the training data: in fact, methods which rely on the training data are prone to inherit the generalization challenge spoken of. Are these correct statements? If so, can the authors speak to (or reiterate) justification for these model assumptions?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a key challenge in symmetry learning—how to generalize from training data to unseen data. It introduces the Cartan loss and the Geodesic Symmetry Consistency loss, which are designed to enforce fundamental properties of Lie algebras and enhance generalization capabilities, respectively. The learnable components include a Lie Algebra Encoder, which maps input data to coefficients of Lie algebra bases; learnable Lie algebra bases; and a conditional flow matching model, which generates a vector field based on the Lie algebra and data points. Across a series of combinatorial generalization benchmarks, the proposed method consistently outperforms existing baselines.

Given the theoretical rigor and innovation, I am inclined to accept it.

### Strengths
- To my knowledge, none of the existing symmetry discovery methods take into account the fundamental property of Lie algebras—the relationship defined by the Lie bracket. The proposed CartanFM in this paper rigorously incorporates this aspect, resulting in a more reasonable learned Lie algebra space, which I consider one of the highlights of this work.

- Most symmetry discovery approaches opt to directly learn either the basis of the Lie algebra or the vector field. The former is often limited to relatively simple linear symmetries, while the latter, due to the complexity of vector fields, generally lacks interpretability. CartanFM’s flow matching module innovatively addresses both limitations. Specifically, it establishes a mapping between the Lie algebra and the vector field, enabling us to explicitly solve for the Lie algebra basis while also obtaining its specific action form on data points.

- The generalization capability on unseen data provides potential for scientific discovery and the learning of structured representation spaces.

### Weaknesses
- Due to the involvement of learning Lie algebra bases, a brief discussion and comparison of the related symmetry discovery work (https://arxiv.org/abs/2310.00105, https://arxiv.org/abs/2410.21853, https://arxiv.org/abs/2403.01946, https://arxiv.org/abs/2510.01855) is necessary.

- It is better to list the contributions point by point in detail in Section 1.

### Questions
- I notice that Definition 2.2 is defined for a one-parameter group action ($t \in \mathbb{R}$). So how does it guide the design of the GSC loss in the multi-parameter case?

- Does this method have potential for application in high-dimensional data—such as in the embedding spaces of images and text?

### Soundness
3

### Presentation
3

### Contribution
3
