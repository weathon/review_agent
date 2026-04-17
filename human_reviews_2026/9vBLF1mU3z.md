# Inverting Data Transformations via Diffusion Sampling

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
We study the problem of transformation inversion on general Lie groups: a datum is transformed by an unknown group element, and the goal is to recover an inverse transformation that maps it back to the original data distribution. Such unknown transformations arise widely in machine learning and scientific modeling, where they can significantly distort observations. As a key application, we focus on test-time equivariance, where the objective is to improve the robustness of pretrained neural networks to input transformations at inference time and without any (re)training. We take a probabilistic view and model the posterior over transformations as a Boltzmann distribution defined by an energy function in data space. To sample from this posterior, we introduce a diffusion process on Lie groups that keeps all updates on-manifold and only requires computations in the associated Lie algebra. Our method, Transformation-Inverting Energy Diffusion (TIED), relies on a new trivialized target-score identity that enables efficient score-based sampling of the transformation posterior. TIED naturally handles curved group geometries and rugged, multimodal energy landscapes, and it applies to a broad class of Lie groups and nonlinear actions without assuming compactness or a bi-invariant metric. Experiments on image homographies and PDE symmetries demonstrate that TIED can restore transformed inputs to the training distribution at test time, showing improved performance over strong canonicalization and sampling baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose an approach that learns to invert Lie symmetries in data via the reverse process of a specialized diffusion process over the group. Specifically, inverse transformations are modeled as a Boltzmann distribution corresponding to an energy term over the data. Scores are recovered from the gradients of the energy functional. Notably, the proposed method can handle non-compact groups that act projectively. Instead of incorporating the proposed approach into existing equivariant models, the authors do the opposite: In applications, the inversion framework is used to robustify non-equivariant models in the presence of challenging perturbations.

### Strengths
The authors proposed approach is both novel and creative. The ability to successfully handle non-compact groups and non-linear actions is particularly impressive and important. The paper is also well written. 

In particular, I think this paper establishes an important new direction in symmetry discovery, successfully incorporating a modern probabilistic approach consisting of a well-formulated, group-specific version of diffusion and suggesting that meticulously hand-crafted approaches based on engineering equivariant architectures may not be necessary.

### Weaknesses
- My main concern involves baselines for comparison. While the main contribution of this paper is clearly techincal, I think it would be useful for the authors to include comparisons with existing equivariant approaches on some of the main experiments for context. 

- In addition, several recent works on symmetry discovery are overlooked in the related works, even though these methods handle some of the same tasks (e.g. Homography-perturbed MNIST). One of these methods (Neural Isometries) also handles non-compact and non-linear actions. While these models don't take probabilistic approaches, they are probably worth a short discussion as this paper also represents a contribution to symmetry discovery.

  Latent Space Symmetry Discovery (Yang et al, ICLR 2024) https://arxiv.org/pdf/2310.00105

  Neural Fourier Transform: A General Approach to Equivariant Representation Learning (Koyama et al, ICLR 2024) https://openreview.net/forum?id=eOCvA8iwXH

That said, I expect these concerns to be straight-forward for the authors to address and I recommend acceptance. 

  Neural Isometries: Taming Transformations for Equivariant ML (Mitchel et al, NeurIPS 2024) https://arxiv.org/pdf/2405.19296

### Questions
Can the authors comment on the scalability of the proposed method? How, if at all, is the complexity tied to the dimensionality of the data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces Transformation-Inverting Energy Diffusion (TIED) for inverting data transformations. Unlike traditional diffusion models, TIED leverages a novel trivialized target-score identity and enables efficient sampling from the Boltzmann posterior on the group.

TIED effectively addresses the challenges of curved group geometry and rough, multimodal energy landscapes. Specifically, it demonstrates superior performance compared to previous works in two key application scenarios: image homographies and Lie point symmetries for partial differential equations (PDEs).

### Strengths
1. The work proposes a novel framework for inverting unknown data transformations from general Lie groups, which breaks through the limitations of traditional diffusion models and provides a new research perspective for the field of transformation inversion.

2. Compared with traditional diffusion models, TIED infers the posterior distribution of transformations based on an energy prior, which significantly accelerates the reconstruction process. Experimental results further confirm that TIED achieves substantial performance improvements over existing methods in targeted tasks.

3. The paper provides a clear and detailed description of the construction process of TIED, accompanied by rigorous mathematical derivations and explanations. This enhances the interpretability and reproducibility of the proposed method.

4. The paper proves a key conclusion: **"Inversion of data transformations can be utilized to achieve test-time equivariance of pre-trained models, without requiring additional modules beyond an approximate energy model (which can even be the predictor itself)"**. This finding expands the application scope of a series of previous works and provides practical guidance for optimizing pre-trained model performance.

### Weaknesses
1. The experimental setup lacks sufficient clarity. In Section 5.2, the specific parameters of the affine matrix (a critical factor directly affecting experimental results) are not provided. This omission makes it difficult to fully verify the reliability of the experimental conclusions and hinders the reproducibility of the study.

2. In existing works on affine reconstruction, experimental results are typically presented as quantitative data distributed across the real number axis to reflect result variability. However, this work only reports specific single values, ignoring the importance of repeated experiments. This approach fails to demonstrate the stability and robustness of TIED, weakening the persuasiveness of the experimental results.

3. Some key statements in the paper lack corresponding data support. For example, claims about "reconstructing speed" or "model size" are not backed by specific comparative data , reducing the credibility of the arguments.

### Questions
1. Lines 100\~102 state: **"However, since they operate in data space instead of the group, they require large diffusion models and for non-linear actions do not recover transformations that lie on the manifold."** However, the paper does not mention the exact model size of TIED (e.g., number of parameters, computational complexity) or provide a direct comparison with existing methods in terms of model scale. Could the authors supplement this information to verify the advantage of TIED in reducing model size?

2. Lines 361\~363 discuss the difference in the number of steps between the proposed TIED and previous methods. However, the paper does not report the time consumption of both methods during actual operation. Since a smaller number of timesteps does not necessarily translate to higher reconstruction speed (as it may be affected by per-step computational complexity), could the authors provide specific time consumption data and analysis to confirm the efficiency advantage of TIED?

3. Section 5.2, TIED shows considerable performance improvement. For image inverse problems, indicators such as PSNR, SSIM, or FID are commonly used to directly evaluate reconstruction quality. However, this work only uses ResNet as a classifier to assess performance. Given that not all baseline methods are specifically designed for this classification-based evaluation, could the authors explain the rationale for choosing this evaluation metric and supplement it with direct reconstruction quality indicators?

4. What are the specific values of the affine matrix (the ITS achieved an accuracy of 0.89 in the case of pure rotation, and it is well-known that rotation is also a type of affine transformation) in experiment? Were tests conducted under multiple affine transformations? If so, what is the variance of the experiment?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper “Inverting Data Transformations via Diffusion Sampling” introduces TIED (Transformation-Inverting Energy Diffusion), a probabilistic framework for recovering unknown transformations—such as rotations, perspective warps, or coordinate shifts—that distort data. It models the inverse transformation as a Boltzmann distribution over a Lie group, where low-energy transformations correspond to more “natural” data. TIED performs diffusion sampling directly on the transformation manifold, combining precise gradient-based local optimization with stochastic exploration to efficiently find likely inverse transformations.

The experiments demonstrate that TIED performs favorably compared to gradient-based and optimization baselines across three settings:
(1) synthetic high-dimensional sampling tasks,
(2) image classification under random affine and homography distortions, and
(3) partial differential equation (PDE) problems with hidden symmetry transformations.

### Strengths
- Overall, I found the setup both novel and conceptually interesting. The method enables density estimation on Lie groups using purely data-driven constraints, bridging geometry and probabilistic modeling elegantly.
- The experiments show that applying TIED before inference improves neural network accuracy and consistency under unseen transformations, providing a convincing proof of concept for the proposed idea.

### Weaknesses
- Despite its technical novelty, the paper suffers from weak presentation and clarity. The writing is often dense, and key visual aids (e.g., Figure 1) are underexplained—either the caption needs more detail or the figure itself should be redesigned to convey the main idea clearly.
- Conceptually, the work frames “canonicalization” as a way to achieve equivariance through energy-based sampling on Lie groups. However, the paper lacks a clear discussion of how this approach fits within the broader spectrum of data augmentation versus architectural equivariance methods.
- The experimental evaluation, while promising, remains proof-of-concept rather than comprehensive. The authors convincingly demonstrate that the method functions as intended, but a more thorough benchmarking—especially against equivariant neural networks—would be essential to assess its broader impact.

### Questions
Overall, this appears like a creative and well-motivated idea with some good arguments with theoretical underpinnings and good initial results. However, the paper currently reads more as a conceptual demonstration than a fully mature study. I would not recommend it for acceptance in its current form, but with clearer exposition and stronger experimental validation, it could become a valuable contribution.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents TIED, which is a method that can de-transform an image from an arbitrary group action to invert its data transformation. This is a novel idea and in fact learning this inverse transformation process is valid and interpretable and robust. The authors also proposed a test-time equivariance algorithm which helps pretrained networks to be G-equivariant. The experiments are performed on image recognition and PDE data.

### Strengths
This paper presents a very interesting and working algorithm. 

The problem of group-equivariant learning has been studied for a long time. As an important property of physics (symmetry), the learning/prediction outcomes should be equivariant to group transformations. However, most of the prior methods (i) data augmentation (ii) learning a standard position or (iii) group convolutions can be (i) not robust (ii) hard to define (iii) expensive. This framework nicely combines the strength of the diffusion model to learn the inverse process, which seems to provide a good solution to this community.

### Weaknesses
The main weakness of this paper is the presentation. Some figures and experimental results are hard to understand easily (even after a couple of re-reads from the reviewer's side - who is familiar with the field). 

Please see the questions below.

### Questions
1. Could the authors explain Figure 2? The reviewer would request the authors to talk about t, sigma, what are the blue and green lines. Are they the same thing? (one is log density and the other is the original density)? What does the author mean by "Note the multimodal nature of the posterior, with modes centered around g = −20◦, 135◦, 180◦." Does this come from the available dataset? For an image network, which image network is this? 

2. Figure 3 requires a much better description, as this is a very interesting experiment. First, it is not immediately understandable what X_{1,1}^2 represents. And what is SO(10)? In fact SO(2), SO(3) also need to be defined. And what does it mean by dim G = 45? And how many samples kinetic Langevin algorithm was used? Why is mean of X_{1,1} important? The reviewer believes the readers will apprecaite this experiment more if the authors can write a better description. 

3. In Table 1, why ResNet18+ITS may cause n/a? Would the authors try to center the words like "energy: VAE evidence lower bound (+ adv. reg.)" or make these fonts bold (to better separate these lines)? 

4. Would the authors comment more on the computational requirement? Since it may implicitly require a lot of computation to solve the inverse process. With the same computational budget, one might already be able to get a pretty well generalized model (from pre-training with Lie-equivariant learning methods). In case the reviewer misses the context, would a trained Lie inverse transformation model be able to reverse both the PDE and images once it is trained? To be clear as well, the result might not be able to generalize to images that haven't been seen before right (since the energe function will be wrong)? 

The reviewer is very likely to improve the rating if one (or more) of the prior questions can answered by the authors.

### Soundness
3

### Presentation
2

### Contribution
4
