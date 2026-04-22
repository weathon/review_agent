# Learning Inter-Atomic Potentials without Explicit Equivariance

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Accurate and scalable machine-learned inter-atomic potentials (MLIPs) are essential for molecular simulations ranging from drug discovery to new material design. Current state-of-the-art models enforce roto-translational symmetries through equivariant neural network architectures,  a hard-wired inductive bias that can often lead to reduced flexibility, computational efficiency, and scalability. In this work, we introduce \textbf{TransIP}: \textbf{Trans}former-based \textbf{I}nter-Atomic \textbf{P}otentials, a novel training paradigm for interatomic potentials achieving symmetry compliance without explicit architectural constraints. Our approach guides a generic non-equivariant Transformer-based model to learn $\mathrm{SO}(3)$-equivariance by optimizing its representations in the embedding space. Trained on the recent Open Molecules (OMol25) collection, a large and diverse molecular dataset built specifically for MLIPs  and covering different types of molecules (including small organics, biomolecular fragments, and electrolyte-like species), TransIP attains comparable performance in machine-learning force fields versus state-of-the-art equivariant baselines. Further, compared to a data augmentation baseline, TransIP achieves 40\% to 60\% improvement in performance across varying OMol25 dataset sizes. More broadly, our work shows that learned equivariance can be a powerful and efficient alternative to equivariant or augmentation-based MLIP models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
While the proposed TransIP achieves faster inference and improved scalability, the experimental comparison to equivariant baselines is insufficient to substantiate the claimed “comparable accuracy.”

### Strengths
The idea of learning latent equivariance through a contrastive objective is conceptually interesting and practically promising.

### Weaknesses
While the proposed TransIP achieves faster inference and improved scalability, the experimental comparison to equivariant baselines is insufficient to substantiate the claimed “comparable accuracy.” Specifically, Table 1 reports results only against eSCN and GemNet-OC, but the configuration details (number of parameters, training epochs, and compute budgets) are not provided for these models.
Without matching model size or FLOPs, the efficiency-accuracy trade-off is difficult to interpret.

Although TransIPreaches ∼0.10 eV Å⁻¹ in force MAE, the best eSCN models achieve ∼0.01 eV Å⁻¹. This difference of nearly one order of magnitude cannot be considered “comparable.”

The study omits several strong and more recent baselines that represent the current frontier in equivariant MLIPs, such as SO3Krates, ViSNet, MACE.

Figure 4 qualitatively shows “faster inference,” but numerical throughput (atoms/s) is not tabulated, and wall-clock time per step or per molecule is not reported.

### Questions
See Section Weaknesses.

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
This paper presents TransIP, a Transformer-based framework for learning interatomic potentials without explicit SO(3)-equivariant architectures. The authors propose a training scheme that achieves learned equivariance in embedding space through a contrastive latent objective. Instead of using tensor products or spherical harmonics, TransIP encourages the model to learn rotational consistency directly from data, improving flexibility and scalability compared to conventional equivariant MLIPs.

The method shows comparable performance to state-of-the-art equivariant models and notably outperforms data augmentation baselines, particularly in low-data regimes. The study positions TransIP as a bridge between fully constrained equivariant networks and unconstrained Transformers.

Overall, this is a well-motivated and clearly scoped investigation into learned symmetry for molecular systems, though additional probing of the latent representation and model behavior under symmetry operations would substantially strengthen the work.

### Strengths
* Introduces a clear, architecture-agnostic approach for encouraging SO(3) equivariance in a Transformer backbone without explicit geometric layers.
* Demonstrates strong empirical performance relative to augmentation-based baselines, validating the utility of learned equivariance.
* Employs rotary position embeddings (RoPE) to inject coordinate dependence without fixed distance cutoffs
* Provides a scalable alternative to equivariant message-passing networks, highlighting potential for deployment on large molecular datasets.
* The experiments on scaling (dataset and model size) are thoughtfully structured and well contextualized.

### Weaknesses
* Enforcing equivariance is only explored on the final pooled molecular embedding, not at the per-atom or intermediate-layer level. As a result, local geometric information is only indirectly constrained and cannot be explicitly evaluated.
* The contrastive loss may over-regularize or under-constrain the latent space; a discussion or ablation on this sensitivity is missing.
* The current benchmark results are fine, but more investigation on the latent space would really enhance the paper's contributions.
* Evaluation tables should follow community standards (e.g., reporting energies in meV and forces in meV/Å) and include units on all axes (e.g., Fig. 3).
* The literature review could better acknowledge earlier works using spherical harmonics for convolutional equivariance prior to Gasteiger et al. (2020) and Klicpera et al. (2021), along the lines of 3D steerable CNNs, Cormorant, Tensor Field Networks, etc.

### Questions
1. **Latent structure and decomposition:**
   Can the authors measure how much of the latent representation becomes invariant vs. equivariant under the learned transformation? Since the final target (energy) is invariant and forces are derived via gradients, it would be insightful to quantify what fraction of latent dimensions encode rotational behavior.
2. **Layerwise analysis:**
  It would be interesting to evaluate how the latents per layer. This would help illustrate whether early layers learn geometry or whether equivariance only appears late. 
3. **Intermediate supervision:**
  Have you tested applying the contrastive loss on intermediate latent representations or per-atom embeddings? Does this improve or degrade performance, and how sensitive is it to the strength of the loss? (It would be especially interesting to see whether standard nonlinearities—most of which break equivariance—prevent stable intermediate supervision.)
4. **Per-atom vs. global equivariance:**
  Currently, the loss is applied only to the final aggregated molecular embedding. In principle, a per-atom latent equivariance loss could improve local geometric fidelity. Could you comment on whether this was attempted or ruled out due to stability or computational cost?
5. **Translation symmetry and RoPE:**
Can you elaborate more in the text on why you chose the RoPE embeddings. Currently one needs to read through RoPE to get this and it would be helpful for the read to not have to do that -- it breaks focus while reading. Additionally, please comment on how RoPE does / does not handles translation symmetry. Since the model centers coordinates by the center of mass, how does it behave for systems lacking a natural origin (e.g., extended periodic systems)?
5. **Metric conventions:**
For consistency and clarity, please provide units in Fig. 3 and report energies in meV and forces in meV/Å in Table 1.

### Soundness
4

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
4

### Summary
The paper introduces TransIP, which steers a standard Transformer toward SO(3) equivariance via a latent equivariance (contrastive) objective in feature space, avoiding hard-wired equivariant layers. The model adds a loss term that aligns the features of a molecule with its rotated embeddings via a learnable transform  T_τ. On OMol25 dataset, TransIP outperforms rotation-augmentation baselines and achieves performance comparable to equivariant MLIP baselines.

### Strengths
A friendly way to add equivalence. The core idea of this paper is clear. The latent equivariance loss is architecture-agnostic and straightforward to implement, allowing the backbone to remain unconstrained while satisfying equivariance by avoiding the use of spherical tensors.

### Weaknesses
Focus on molecules only. Experimental results only target OMol25. SPICE and its SOTA benchmarks are not discussed.
Ablations on  T_τ are limited. The core contribution is an implicit equivariance module(a learnable transform). However, the evaluation does not display the effect of the learned equivariance itself from other invariant MLIPs. The authors should add the implicit-equivariance module to other invariant backbones and compare with the invariant-only MLIPs.
The results are not good enough. Based on the results in Table 1, TransIP does not outperform the competing models. In fact, its accuracy is generally lower than that of eSEN and GemNet.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a Transformer-based interatomic potential prediction model called TransIP. The core innovation lies in learning SO(3) equivariance implicitly through a contrastive learning objective in the embedding space, without relying on explicit equivariant architectures. The authors validate the method on the large molecular dataset OMol25, showing that it outperforms traditional data augmentation methods both in limited data and large-scale training scenarios.

### Strengths
1.The MLIP training pipeline and architecture-agnostic contrastive loss function proposed in the paper are easy to follow.
2.TransIP shows a significant improvement in force prediction compared to traditional data augmentation methods.

### Weaknesses
1.TransIP does not perform well. The comparison with eSEN and GemNet-OC in Table 1 shows that it does not have an advantage in terms of performance.
2.The evaluation is only conducted on the OMol25 dataset and has not been extended to other datasets (e.g., SPICE, MPTrj, OMat24), failing to demonstrate the generalizability of the method.
3.The backbone uses only a Transformer-based architecture, and the effectiveness of the method on other invariant MLIPs has not been explored.

### Questions
1.Regarding the dataset: OMol25 is an inorganic molecular dataset, but it’s also important to verify whether TransIP delivers the same results on other types of datasets (e.g., Materials, Catalysis). Could you compare the performance of TranAug and TransIP on other datasets, such as MPTrj, OMat24, or OC20?
2.Regarding the backbone architecture: The current backbone is based on a Transformer, but it’s crucial to assess the method’s effectiveness with other architectures as well. Could you provide results for the architecture-agnostic contrastive loss function when applied to other invariant MLIPs?

### Soundness
2

### Presentation
3

### Contribution
2
