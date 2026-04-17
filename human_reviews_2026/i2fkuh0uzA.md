# Aligning Rotational and Hierarchical Geometry in Molecular Representation Learning with Product-Manifold Latent Spaces

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Learning effective molecular representations requires capturing two fundamental but largely disjoint aspects of the structure of molecules: rotational symmetries in 3D conformations and the hierarchical organization of chemical scaffolds. We introduce a new paradigm of product-manifold representation learning with product-manifold message passing on $\mathrm{SO}(3) \times \mathbb{H}^d$, which couples equivariant geometric features with hyperbolic embeddings of chemical hierarchy. Our construction preserves $\mathrm{SO}(3)$-equivariance in the geometric channel and uses an $\mathrm{E}(3)$‑invariant readout for scalar properties while enabling curvature-aware aggregation in the hyperbolic channel, with cross-coupling restricted to scalar invariants to maintain symmetry. Unlike prior approaches that fuse equivariant and hierarchical encoders via concatenation or stacking, our method defines message passing directly on the product manifold, yielding a unified representation. We outline how such models could be evaluated on molecular property prediction, scaffold-split generalization, and generative design, and discuss how embeddings in $\mathrm{SO}(3) \times \mathbb{H}^d$ provide a natural surrogate space for manifold Bayesian optimization, enabling more sample-efficient discovery of high-value molecules compared to Euclidean BO. Together, these results suggest a principled path toward unifying physical symmetries and chemical hierarchies within a single geometric learning framework.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel graph neural network architecture designed to unify two critical but typically separate concepts in molecular representation learning: the 3D rotational symmetries of molecular conformations and the hierarchical organization of chemical scaffolds.

The authors propose a message-passing framework that operates on a product manifold, $\mathcal{M} = SO(3) \times \mathbb{H}^{d}$. The key idea is to maintain two coupled channels:
* An SO(3)-equivariant channel that uses irreducible representations (irreps) to process 3D geometric information, preserving physical symmetries.
* A hyperbolic channel that embeds the molecule's chemical hierarchy (derived from a junction tree) using curvature-aware operations like Fréchet means.

The paper's main contribution is the "symmetry-safe" coupling mechanism: the two channels interact only through scalar ($l=0$) invariants (e.g., Euclidean distances, hyperbolic distances, feature norms). This allows hierarchical information to gate and influence the geometric representations without breaking the fundamental SO(3)-equivariance. The authors evaluate this product-manifold model on QM9 (property prediction), OGB-MolHIV (scaffold-split generalization), and Guacamol (Bayesian optimization), arguing that their unified latent space improves performance, particularly in generalization and sample-efficient search.

While I quite like the idea of the paper, there are some major concerns that need to be addressed.

### Strengths
* The intuition of the idea is good. Instead of the popular "late-fusion", the authors propose to use a "product-manifold message passing" mechanism for exchanging information between the geometric and hyperbolic channels. This is surely more advanced than the "late-fusion" approach in concept and should bring *some* performance gains.
* The proposed product-manifold latent space and network architecture are tested in various experimental settings -- QM9 (property prediction), OGB-MolHIV (scaffold-split generalization), and Guacamol (Bayesian optimization). This provides a comprehensive evaluation of the proposed method.

### Weaknesses
* The presentation of this paper is problematic, especially in the Methods section where the details of the model definition are described. The formulas are simply hard to follow due to undefined or exotic notations. For example, what does the $\times$ mean in $[M_i^{(R)}]_{\times}$ in Equation 6? For another example, the $dist_H$ symbol appears in Equation 4, but is only defined in the text around Equation 7. I therefore suggest a major revision for the Methods section to increase its readability.
* A preliminary section should be included in the main text or in the Appendix. The proposed method uses two kinds of advanced concepts (the SO(3) irreps and the hyperbolic geometry), and an average reader is only familiar with at most one of them.
* The performance of the proposed architecture falls behind SoTA. The authors has mainly shown the efficacy of the proposed product-manifold message passing through ablation study, but the performance is still not as good as the SoTA models.

### Questions
* Could you improve the presentation of this paper, especially that of the Methods section?
* What is $\phi_{ij}$?
* In addition to the current experiments, could you also compare your proposed approach with "late-fusion"?
* Given an existing model architecture that has SO(3) irreps representations for nodes (such as MACE, Nequip, etc.), how easy can one modify the architecture to adopt the proposed product-manifold latent space and message passing?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on enhancing the expressiveness of graph neural network. The authors introduce a product-manifold message passing to learn equivariant geometric features with hyperbolic embeddings of chemical hierarchy. Additionally, authors show the model could be evaluated on molecular property prediction, scaffold-split generalization, and discuss how embeddings provide a natural surrogate space for manifold Bayesian optimization.

### Strengths
- This paper proposes an interesting structure that has the potential to introduce some n-body priors into molecular systems, which may be beneficial for molecular systems.

### Weaknesses
1. $S(u)$ is an important variable introduced in this paper, used to provide information about the junction tree. However, I could not find its definition in the paper. How is this tree defined? If it is as shown in Figure 1, is this kind of tree only related to the cutoff of the number of edges? How should this tree be defined in 3D space without explicit bonds?

2. The core idea of this paper is somewhat similar to introducing richer n-body information in the message passing process. If so, I think the authors should refer to MACE [1], SLEM [2], which author mentioned in Introduction. In these methods, the model no longer processes the features of each node and its single neighbor, but instead processes the fused features of each node and multiple neighbors, or the fused features with the mean of multiple neighbor nodes.

3. Does the hyperbolic channel interact with the SO(3) channel at every layer of the model?

4. The experimental results still have a gap compared to the state-of-the-art (SOTA), not to mention the lack of the latest baselines such as Equiformer v2 and GotenNet. There is also a lack of more extensive experiments, such as on OC20 and Molecule3D. This makes it hard for me to consider the proposed tree structure in the paper as effective, although it is interesting. I suggest that the authors directly incorporate the product manifold into Equiformer and train for enough epochs to evaluate whether the method can improve performance with sufficient fitting. The results in Table 1 currently appear to be underfitted.

5. The completeness of this paper still needs to be improved. The model architecture, experimental settings, and baseline configurations all need to be detailed in the paper. The equations in Section 3 are somewhat confusing. The position of the table captions is incorrect. In the code section, I only saw a model class, and not even a training script.

[1] MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields

[2] Learning local equivariant representations for quantum operators

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

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
This paper presents a method that combines the 3D rotational equivariance of small molecule conformations with hyperbolic embeddings of their chemical hierarchy. Instead of combining these independent modes of describing molecules following their distinct symmetry-respecting embeddings, the authors show how to develop a joint embedding that exchanges only scalar invariant information within a layer. The proposed method shows some promise in early benchmark evaluations, which indicates that the idea might merit further investigation as a way to perform optimization in this product-manifold latent space.

### Strengths
This paper contributes an original idea that integrates two independent concepts at an architectural level in a coherent way, providing a symmetry-safe unification of geometric and hierarchical molecular representations.  The background and method descriptions are clear, and the early tests show that the fusion works better than the independent components, supporting further research.

### Weaknesses
Although the architecture preserves the required symmetries, the current implementation of this symmetry preservation appears to come at a cost in performance on standard benchmarks, possibly due to the constraint of having per-layer equivariance and the use of scalar-only couplings.  The demonstrations are limited, lacking directly reproducible code and comprehensive comparisons of the graph-only and coordinate-only components.  The validation performance in OGB-MOLHIV is noteworthy, however, even Gabriele Corso's PNA work from 2020 had validation above 0.85 while performing poorly on the test.  The guacamol results could have been the most interesting, however, that particular benchmark is no longer maintained and there is no clear leaderboard to compare against. The fact that GraphGA, an simple genetic algorithm, performed unexpectedly well in the rankings up to 3 years ago underscores the benchmark's uncertain relevance today.  More generally, it is unclear why the authors evaluated only a handful of GuacaMol task, especially as there appears to be some merit to their argument for creating a better joint embedding space.

### Questions
Does the current layer-wise equivariant message-passing architecture suffer from similar problems as those identified in general GNNs in the zero-one laws of graph neural networks (https://arxiv.org/abs/2301.13060)?  In particular, do the authors anticipate that the architecture might not support all-atom representations of large biomolecules, including proteins, due to the same type of collapse?

Can the authors provide detailed protocols/scripts for the training and evaluation and also describe (perhaps in a supplement) the computational performance and stability of the architecture with respect to hyperparameters?

Is it feasible to pretrain this model on large mixed datasets and then finetune it on datasets with only partial information (e.g. no coordinates or no graph)?  An benefit of certain late-fusion techniques, or of ways to align the embeddings of separate techniques, is that one could potentially use partial information in followup work.

### Soundness
2

### Presentation
2

### Contribution
3
