# Pseudo-Non-Linear Data Augmentation: A Constrained Energy Minimization Viewpoint

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 8

## Abstract
We propose a simple yet novel data augmentation method for general data modalities based on energy-based modeling and principles from information geometry. Unlike most existing learning-based data augmentation methods, which rely on learning latent representations with generative models, our proposed framework enables an intuitive construction of a geometrically aware latent space that represents the structure of the data itself, supporting efficient and explicit encoding and decoding procedures. We then present and discuss how to design latent spaces that will subsequently control the augmentation with the proposed algorithm. Empirical results demonstrate that our data augmentation method achieves competitive performance in downstream tasks compared to other baselines, while offering fine-grained controllability that is lacking in the existing literature.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a learning-free, geometry-aware framework for data augmentation based on energy-based modeling and information geometry. Unlike deep generative models that rely on trained latent representations, the proposed method constructs a statistical manifold via log-linear models on posets and performs data augmentation through forward and backward projections on low-dimensional submanifolds. The approach, termed Pseudo-Nonlinear Data Augmentation (PNL), combines the linearity of projections with nonlinear geometric curvature, offering both computational efficiency and controllability. Empirical results on images (MNIST, CIFAR-10), speech, and tabular datasets show that PNL performs comparably or better than autoencoder-based and standard augmentation methods, while being faster and interpretable.

### Strengths
1. The paper introduces a geometric viewpoint on data augmentation grounded in energy minimization and information geometry, distinct from the dominant deep learning paradigms. The formulation via dually-flat manifolds and log-linear models on posets is both elegant and principled.

2. The method requires no model training, avoiding the heavy computational cost and data dependence of generative augmentation. Projections are formulated as convex programs solvable with first-order methods.

3. Augmentation behavior can be explicitly controlled by the choice of submanifold and the poset structure, enabling fine-grained manipulation of data geometry.

4. The method demonstrates applicability to image, speech, and tabular data without domain-specific modifications.
Shows robustness and consistency, especially on small datasets with high variance.

### Weaknesses
1. The baselines are somewhat minimal (autoencoder and standard augmentation).
Comparison to modern generative augmentations (e.g., diffusion-based, mixup, manifold mixup) would strengthen the claims.

2. The method requires defining a partial order (poset) over features, which might be nontrivial or restrictive for unstructured or permutation-invariant data.

3. While the method is theoretically interpretable, more concrete visualization or case studies of controllable augmentation (beyond MNIST/CIFAR) would enhance clarity.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new learning-free data augmentation method that is based on energy-based modeling. The authors first assume a poset structure of the data. Using this poset structure, they map the data to a log linear model on the poset. They then associate dually-flat coordinates with the log linear model. This constitutes the embedding of the data which is then used to generate data augmentations as follows. Given a sub-manifold, the authors project the embedding and then backward project using some heuristic assumptions on the structure of the data. Overall, this results in a task-independent data augmentation method.

### Strengths
The paper is presented clearly. The authors manage to put interesting and novel ideas together. This is an interesting addition to the landscape of data augmentation. The fact that the method is learning-free is very appealing.

### Weaknesses
1. The poset structure does not fit many data domains well. This is acknowledged by the authors. I don't understand why they suggest that images have an inherent partial order. Directed graphs and time series might conform to such partial orderings.

2. It is unclear what the log linear model captures about the data. The authors mention energy-based geometric modeling but more discussion is needed. It seems that there is first a normalization is done (Example 4.1.) and then this is interpreted as a probability distribution. It is unclear what the meaning of this probability distribution is for many data domains.

3. Apart from the total energy in the features, $\phi(z_i)$ preserves all the information about the data. Similarly, $\theta$ coordinates should then do the same. It is unclear why the geometry here is better for data augmentation than the direct data space.

### Questions
1. Can $z_i$ in principle contain data labeling? As far as I understand, $z_i$ is just data without labels. This makes the data augmentation procedure task-agnostic. This results in a method that cannot incorporate any biases that are beneficial for the downstream task except the choice of poset structure. Can you explain how the poset structure introduces such biases in the case of images?

2. Why do you think the probability distribution that comes after applying $\phi$ is a good representation of the data? This should be dependent on the properties of data. For example, could you explain why it is a good representation for images? What is added by the dually flat coordinates? Why would the data be in a linear subspace of this manifold?

3. You cite generative-model-based data augmentation suffers from two challenges: efficiency and controllability. Is this the case for MNIST and CIFAR-10? I believe the challenges apply to harder datasets and I don't see why the proposed method scales to high dimensions.

The following paper is very relevant related work as it has an encoding/decoding scheme over a manifold that is learned.

[1] Yüksel, Oguz Kaan, et al. "Semantic perturbations with normalizing flows for improved generalization." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a projection-based data augmentation framework that maps data onto a statistical manifold and performs controlled perturbations through forward and backward projection. The main novelty focuses on achieving learning-free, efficient, and controllable augmentation with semantics preserved, by leveraging the geometric structure induced by posets. Experiments on image and speech datasets validate that the method can maintain core structural features while introducing meaningful variation.

### Strengths
The proposed method is learning-free and inverse-consistent. In addition, it is controllable with explicit selection of structural attributes.

### Weaknesses
1. Performance gains over simple AE-based augmentation are modest, especially in CIFAR, where improvements are marginal.
2. Efficiency is claimed relative to diffusion models, but no direct comparison or runtime analysis is provided.
3. As discussed in the paper, the method requires data to admit a meaningful poset structure, which cannot directly handle permutation-invariant data such as point clouds or general graphs.
4. The experiments are primarily on images, as argued in line 88, with more different modalities, like graph, video, might be more interesting.

### Questions
The backward projection relies on nearest neighbors in the latent space to reconstruct data, which means the augmented samples often stay very close to existing examples. As the dataset grows, will this approach become less scalable and may fail to produce diverse new samples?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a learning-free and geometry-aware data augmentation framework termed Pseudo-Non-Linear Data Augmentation (PNL-DA). The key idea is to view data augmentation as an energy minimization problem on a latent manifold derived from the intrinsic structure of the dataset.

Instead of using neural generators or search-based augmentation (e.g., AutoAugment), the authors construct an information-geometric latent space and perform constrained energy descent to generate perturbed samples that respect the data manifold while introducing non-linear diversity. This results in augmented data that remain semantically valid yet expand the effective data support.

The paper claims that PNL-DA:
1. requires no additional network training,
2. offers fine-grained controllability over augmentation strength via explicit constraints, and
3. achieves competitive or superior downstream performance across standard image classification benchmarks.

### Strengths
1. The paper reframes data augmentation as an energy minimization process under latent-space constraints. This is a refreshing deviation from purely heuristic or search-based augmentations. The geometric viewpoint is conceptually elegant and could inspire follow-up work linking augmentation and manifold regularization.

2. In contrast to most augmentation strategies that involve training auxiliary networks or policy search, this method is deterministic and learning-free once the latent geometry is built. This makes it lightweight and potentially appealing for resource-constrained environments.

3. The energy constraint allows explicit control over perturbation magnitude, which is a desirable property compared to black-box stochastic augmentation policies. The paper demonstrates that tuning this constraint can balance sample diversity and fidelity.

4. Experiments on several vision datasets (CIFAR-10/100, Tiny-ImageNet, SVHN, and others) show that PNL-DA matches or slightly surpasses traditional augmentations and performs competitively against AutoAugment and RandAugment, while using less computational cost.

5. The exposition of the method—particularly the geometric intuition and constrained optimization formulation—is well organized and mathematically sound. The visualizations of latent-space perturbations help the reader grasp the intuition.

### Weaknesses
1. The energy constraint is intuitive, but there is no formal proof that minimizing it preserves label semantics or improves generalization bounds. The paper could benefit from a more rigorous analysis linking the energy formulation to risk minimization.

2. The method introduces a constraint coefficient controlling perturbation scale. The results indicate non-trivial sensitivity, but the paper lacks systematic tuning guidelines or ablation to quantify robustness.

### Questions
1.The paper frames data augmentation as energy minimization on a latent manifold. Could you clarify the exact connection between your “energy” function and the traditional risk minimization objective? Is the energy equivalent to a potential function, or is it empirically constructed?

2. You define “pseudo-non-linear” transformations. In what precise sense are they pseudo non-linear? Are these transformations non-linear in the data space but linear in latent space, or vice versa?

### Soundness
3

### Presentation
4

### Contribution
3
