# Learning Unified Representation of 3D Gaussian Splatting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 8

## Abstract
A well-designed vectorized representation is crucial for the learning systems natively based on 3D Gaussian Splatting. While 3DGS enables efficient and explicit 3D reconstruction, its parameter-based representation remains hard to learn as features, especially for neural-network-based models. Directly feeding raw Gaussian parameters into learning frameworks fails to address the non-unique and heterogeneous nature of the Gaussian parameterization, yielding highly data-dependent models. This challenge motivates us to explore a more principled approach to represent 3D Gaussian Splatting in neural networks that preserves the underlying color and geometric structure while enforcing unique mapping and channel homogeneity. In this paper, we propose an embedding representation of 3DGS based on continuous submanifold fields that encapsulate the intrinsic information of Gaussian primitives, thereby benefiting the learning of 3DGS.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses challenges in 3D Gaussian Splatting (3DGS) representations, where individual Gaussians have multiple parameter sets. The authors note that some parameters are non-unique (e.g., different rotation representations can yield the same state) and that parameter groups are from different distributions or manifolds. The authors claim this non-uniqueness and heterogeneity make the standard 3DGS parameter space ill-suited for neural networks tasked with generating or estimating Gaussians. To resolve this, the paper introduces "submanifold field" representations, a novel parameter space for these Gaussian features. The authors demonstrate the uniqueness of this new representation and propose a corresponding manifold distance (M-Dist).
Through experiments, the paper demonstrated superior performance against baselines.

### Strengths
- The paper presents an interesting and timely idea, given the growing interest in large-scale reconstruction models which could benefit from improved 3D Gaussian Splatting representations.
- The proposed method demonstrates strong performance improvements over the baseline.
- The authors provide a broad analysis, including evaluations of robustness, clustering, and interpolation.

### Weaknesses
**Motivation**: The motivation presented in the Introduction could be strengthened. The importance of resolving the identified issues should be emphasized more clearly, particularly by highlighting specific downstream applications and detailing why and when existing representations fail.

**Experiments**: The current experiments focus on small MLP and VAE-style networks, which may not reflect real-world applications. Even if the goal is to isolate the effect of the new representation, it is unclear whether results from these small networks will generalize to the much larger, more complex models used in practice. Without testing or adapting existing large-scale reconstruction models, the broader applicability of the proposed representation remains uncertain.

### Questions
- To aid in understanding the concept, could the authors elaborate on the definitions and intentions behind the terms "sub" and "field" in "submanifold fields"?

- While the paper discusses the ill-posedness of standard 3DGS features, is there a specific reason to formalize the non-uniqueness and heterogeneity so strongly through definitions and propositions?

- In Section 4.2 (lines 362-364), it is stated that M-Dist has a higher correlation with the perceptual metric than L1. Could the authors also provide the correlation with L2? Furthermore, in Figure 4 (right table), the M-Dist for the SF embedding appears to increase significantly with noise. This seems to contradict the claim that M-Dist has a better correlation with perceived image quality than L1. Could the authors clarify this apparent discrepancy?

- Regarding Figure 4 (robustness to noise): Since the different representations have different scales and properties, is it certain that adding uniform noise affects them in a comparable way?

- For Figure 5 (interpolation): Would it be possible to include more straightforward or intuitive examples? From the current figure, it is unclear precisely how the SF-VAE interpolation is superior to the other methods.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses key challenges of using parameter-based Gaussian representations as network inputs: non-uniqueness, numerical heterogeneity, and distinct mathematical manifolds. To overcome these issues while preserving the underlying geometric structure, the authors propose representing each Gaussian as a continuous field defined on its iso-probability surface. Experimental results demonstrate that the proposed representation achieves superior reconstruction quality and exhibits strong robustness to domain shifts compared to traditional parametric representations.

### Strengths
- The paper clearly explains the problem formulation and the motivations behind introducing a new representation for Gaussian primitives.
- The idea of learning a discretized iso-probability surface as a geometric representation is interesting, and I believe it is also novel.
- The writing is clear and effectively conveys the core ideas; the main figure is particularly helpful in illustrating the concept.
- The proposed model can be trained using a randomly generated dataset, enabling domain-agnostic learning that can later be applied to embed optimized Gaussian primitives.

### Weaknesses
- While the paper aims to validate the idea of using iso-probability surfaces as an alternative representation of parameter-based Gaussian representations to make them more amenable to learning-based systems, it remains unclear how well this representation generalizes to various downstream tasks. From my understanding, one motivation for learning such representations is to enable applications such as compressing large sets of Gaussians or training generative models to sample new Gaussian primitives. However, the experimental results primarily demonstrate that the proposed representation can reconstruct Gaussian parameters more accurately in a controlled setting. This alone does not necessarily prove that the learned representation will be effective in downstream or scene-level applications.
- Each iso-probability surface is independently encoded and decoded by the network, meaning that each latent embedding is learned in isolation without awareness of global scene-level context. This independence could limit the model’s ability to generate spatially consistent or semantically coherent Gaussian primitives when generating full scenes, which is an important consideration for practical use cases (e.g., scene generation).

### Questions
- The decoding process involves batched PCA, eigen decomposition, and spherical harmonics fitting for every reconstructed Gaussian primitive. Could the authors clarify the computational cost of this step and whether it poses a bottleneck when decoding a large number of Gaussians?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Gaussian splats have been used for 3D scene reconstruction with a lot
of success. Recent works explore mapping the GS parameters to another
embedding space using neural networks to allow applications, such as
generation, editing, compression, and even predicting GS parameters
multi-view image sets directly. Authors here focus on such mappings
and note that the common GS parameterization is not optimal. The main
reasons are non-uniqueness, different parameterization lead to the
radiance fields, and numerical heterogeneity, different parameters
have different magnitudes and such magnitude heterogeneity makes
gradient-descent based learning challenging. They propose another
parameterization as well as an auto-encoder model that maps this
parameterization to an embedding space and back. The representation in
the embedding space is claimed to be more suitable for neural network
training. The model is trained using synthetically generated Gaussians
and applied then to the ShapeSplat and Mip-NeRF 360 datasets. Both
object-level and scene-level experiments are conducted. Comparisons
based on similarity of rasterized images and Gaussian parameters are
presented.

### Strengths
+ Authors identify an interesting and important problem.
+ The explanation of the problem is very well done.
+ GS is a very relevant topic and the paper is timely.
+ The proposed approach is sound and based on solid
  theory. Proposition 2 is well appreciated.

### Weaknesses
+ The experiments on domain-shifts are not well explained. It is not
  clear what is trained on the datasets and what is tested. I thought
  the proposed method was trained using synthetically generated
  Gaussians. This part needs to be explained better I believe.

### Questions
+ The issue of heterogeneity of parameter magnitude is a problem of
  optimization. I wonder whether optimizers that approximates second
  order optimization, such as Adam, would be less susceptible to this
  issue.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper argues that the standard parameter-based representation for 3D Gaussian splatting is not a great fit for neural networks. The limitations in regard to non-uniqueness, numerical inconsistency, and mismatches with learning spaces. This leads to instability, poor generalization, and noise sensitivity. To address this, they propose to model each Gaussian as a continuous field on its iso-probability surface and discretized colored point cloud. This preserves geometry and appearance while ensuring uniqueness and consistency, leading to better learning performance.

### Strengths
* The paper convincingly explains why Gaussian parameters are a poor representation for learning. The arguments about non-uniqueness and manifold mismatch are backed by valid theoretical analysis. The contents in sections 3.2 and 3.3 are the major strengths, which are generally convincing.

* The qualitative and quantitative results support the claims about robustness and generalization.

* The paper is generally well written and easy to follow. The graphical illustrations are particularly helpful. 

* The unsupervised graph clustering (shown in Figure 5) convincingly communicates how the submanifold field embeddings preserve detailed semantics.

### Weaknesses
The ablation study in regard to the proposed design (of the architecture of Figure 2) is missing. Currently, the Behavior/sensitivity studies of Figure 6 are presented as the ablation. The choice radar plot of Figure 6 (c) does not really help. The sensitivity comparison of Figure 4 is unfair because the embedding noise and parametric noise do not mean the same thing. It needs to be either justified in an interpretable way or otherwise must be removed to avoid misleading. The confusion on rotation space claimed with the example of Figure 3 is not very convincing, as the source of the artifacts can be anything. For example, among many arbitrarily oriented splats (from initialization), it could be due to not being able to optimize the opacity parameter. Since there could be many splats (which may effectively not contribute) with undesired rotations, their effect can be minimized by minimizing the opacity only.

### Questions
Please, conduct the ablation study, behavior study, and sensitivity study in a structured way. 
The ablation study must ablate (remove) the components of the design choices (like the network architecture). 
The behavior study must showcase how the method behaves over the variables (like the embedding size, training data).
The sensitivity study must showcase how the input fluctuations impact the output (like the noise study).
Authors are recommended to follow a more formal definition and comment on these modifications during the rebuttal. 
Please also address the weaknesses reported above.

### Soundness
3

### Presentation
3

### Contribution
3
