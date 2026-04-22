# Scalable Autoregressive 3D Molecule Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Generative models of 3D molecular structure play a rapidly growing role in the design and simulation of molecules. Diffusion models currently dominate the space of 3D molecule generation, while autoregressive models have trailed behind. In this work, we present Quetzal, a simple but scalable autoregressive model that builds molecules atom-by-atom in 3D. Treating each molecule as an ordered sequence of atoms, Quetzal combines a causal transformer that predicts the next atom's discrete type with a smaller Diffusion MLP that models the continuous next-position distribution. Compared to existing autoregressive baselines, Quetzal achieves substantial improvements in generation quality and is competitive with the performance of state-of-the-art diffusion models. In addition, by reducing the number of expensive forward passes through a dense transformer, Quetzal enables significantly faster generation speed, as well as exact divergence-based likelihood computation. Finally, without any architectural changes, Quetzal natively handles variable-size tasks like hydrogen decoration and scaffold completion. We hope that our work motivates a perspective on scalability and generality for generative modelling of 3D molecules. Code is available at https://anonymous.4open.science/r/quetzal-5BD3.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Quetzal, an autoregressive 3D molecule generative model. The key innovations are modelling the next atom’s position as continuous distribution via a diffusion model, and predicting prefix conditioning vectors with a causal transformer for a cheaper MLP enabling parallel training. Quetzal is benchmarked on QM9 and GEOM-Drugs, showing superior molecule validity with significant speedup.

### Strengths
The architecture and overall design makes a lot of sense to me. 
For a given atom ordering, Quetzal first computes the prefix conditioning vectors from the current atom positions, using a causal transformer. The prefix conditioning vectors are then fed to predict the next atom types. The use of the causal transformer enables the prediction of the atom types in one forward pass during training, which is more efficient than the existing GNN based methods. Then, the prefix conditioning vectors are fed into a MLP diffusion model, to predict the position of the next atom. The use of the (cheaper) MLP avoids multiple calls to the expensive transformer during diffusion sampling.
The speedups and overall model performance on GEOM-DRUGS and QM9 is quite impressive, showing performance similar to state-of-the-art diffusion models but with significantly faster inference time.

### Weaknesses
The reliance on a fixed atom ordering (based on the .xyz file) seems a little worrying (ie Table 4 in Appendix B.2 and the statement “If the bare molecule without hydrogens is reordered, QUETZAL’s performance degrades significantly, as the prefix becomes out-of-distribution.”) The authors are honest about this, which I do commend. But it does suggest a reliance on a canonicalization of the atom order, which is usually not possible in general.

### Questions
* In Symphony, they use a nearest-neighbor ordering for the atoms (which is well-defined in the absence of symmetric degeneracy). Perhaps this could help canonicalize your atom ordering instead of using the .xyz file order? I would be interested in seeing such an experiment.
* Symphony also predicts the position distribution in continuous space (without discretization) via spherical harmonics. However, you are correct that for sampling, the distribution is discretized, so I agree with your overall point.
* Some results on length generalization would be good to understand how long the model can go before falling off.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces QUETZAL, an autoregressive model for 3D molecular generation that predicts atom types and coordinates sequentially using a causal transformer paired with a per-token diffusion MLP. The approach is conceptually simple, highly scalable, and achieves strong sample quality on QM9 and GEOM, with significant generation-speed advantages over diffusion baselines. The authors highlight limitations around atom ordering and rotational/translation augmentation, and they provide candid analysis regarding where their method underperforms (e.g., NLL on QM9) and when comparisons may reflect undertrained models.

This is a technically solid and thoughtfully written paper with good empirical results and strong framing. The permutation-symmetry limitation is significant, but the authors acknowledge it directly. I view this as a worthwhile contribution that advances scalable autoregressive modeling for 3D molecules and provides a useful baseline for future work on order-robust and symmetry-aware variants. Thank you to the authors for the clarity, transparency, and thoroughness.

### Strengths
* Clear and well-motivated architecture combining autoregression with continuous coordinate diffusion.
* Competitive or superior sample quality to prior autoregressive methods and close to recent diffusion-based approaches.
* Very fast sampling relative to diffusion models, especially on small molecules.
* Nice demonstrations of variable-size tasks such as hydrogen decoration and scaffold completion without architectural changes.
* Honest and transparent discussion of failure modes and when improvements reflect training scale rather than architecture.
* Thoughtful positioning of autoregressive models as a scalable, flexible alternative rather than a replacement for equivariant diffusion.

### Weaknesses
* The core limitation is the lack of permutation symmetry. Performance degrades when atom orderings change, and the model does not generalize to unseen orderings or larger-than-training molecules. This restricts applicability for larger or more diverse chemistries, biomolecules, and materials systems.
* Reliance on fixed xyz ordering and positional inductive bias makes results sensitive to preprocessing and data convention choices.

### Questions
1. Did the authors try any strategies to reduce order sensitivity (e.g., randomized orderings during training, curriculum orderings, or masked prefix strategies)? Even negative results would be informative.
2. For hydrogen decoration and scaffold completion, are there systematic diagnostics for how ordering, centering, or orientation perturbations affect performance?
3. Could the authors comment on prospects for integrating partial symmetry handling (e.g., relative positional embeddings or ordering-inference models) while preserving scalability?

### Soundness
3

### Presentation
4

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
This paper proposes a hybrid approach that combines an autoregressive (AR) model and a diffusion model for generating the 3D structure of molecules. The method leverages a large, computationally intensive AR model to perform a single pass per atom, predicting atom types and associated latent vectors. A small, efficient diffusion model subsequently conditions on these latent vectors to generate the continuous 3D atomic positions.

By integrating the complementary strengths of these two paradigms, the method achieves substantial improvements in sampling efficiency over standard diffusion models, albeit with a larger total parameter count. A key limitation, however, is that the AR component's performance is dependent on the canonical atom ordering found in the source chemical files and is not permutation-invariant.

### Strengths
The core idea of hybridizing AR and diffusion models to leverage their respective strengths is compelling. The paper correctly identifies the complementary weaknesses of each paradigm: diffusion models, while powerful, suffer from high sampling costs (modeling all-atom interactions at each denoising step), whereas AR models often struggle to accurately generate continuous 3D coordinates.

The proposed method cleverly capitalizes on the Markovian atom-level decomposition native to AR models. It employs an efficient diffusion model only for the specific task of sampling the 3D position of the current atom, conditioned on a latent vector that aggregates past information. This division of labor is elegant.

The empirical results persuasively demonstrate that this approach achieves competitive generative performance while being substantially faster at sampling than comparable diffusion-based baselines.

### Weaknesses
The primary limitation of this work is its dependence on a canonical atom ordering, as found in the chemical data files. The AR component is not permutation-invariant, and its performance degrades significantly when this ordering is broken.

To their credit, the authors are fully transparent about this issue and explicitly report results on permuted data. However, this reliance on a specific data artifact raises significant questions about the method's practical utility and generalizability.

It remains unclear whether this canonical ordering is a common feature of most large-scale chemical databases or a peculiarity of the QM9 and GEOM datasets used for evaluation. While the intricacies of chemical data formats may be outside the typical scope of a core machine learning paper, a method that so fundamentally relies on this data structure warrants a more in-depth discussion of its prevalence and the implications for real-world application.

### Questions
1. The paper focuses on the significant gains in sampling efficiency. Could the authors provide a comparison of the training time (e.g., wall-clock time or total FLOPs) against the baselines? This information would provide a more complete picture of the method's overall computational profile.
2. Regarding the "with atom permutations" results reported in Table 4: For the training process, were the atoms permuted once per training instance (i.e., a fixed, new random order) or dynamically in each training epoch (i.e., a different permutation of the same molecule each time it is seen)? This detail is important for understanding the nature of the performance degradation.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes QUETZAL, an autoregressive model for 3D molecular generation that builds molecules atom by atom. It combines a causal transformer to predict the next atom’s discrete type with a small diffusion MLP that models its continuous 3D position. This design reduces the number of expensive transformer passes and improves sampling efficiency. QUETZAL achieves generation quality surpassing existing autoregressive baselines and comparable to diffusion models, while also handling variable-size tasks such as hydrogen decoration and scaffold completion without additional training.

### Strengths
- Simple and effective combination of autoregression and per-token diffusion.  
- Demonstrates strong generation quality and faster inference than diffusion models.  
- Naturally supports variable-size molecular generation and related tasks.

### Weaknesses
- The method section is not well organized, making it difficult to follow the overall model design and training flow. The paper mixes notation, derivations, and implementation details without clear separation, which obscures the core idea.  
- The presentation alternates between describing autoregressive factorization, diffusion objectives, and sampling procedures, but the hierarchy between these components is unclear. It is not obvious at first how the transformer-based atom-type predictor and the DiffMLP position predictor interact or are trained jointly.  
- Many implementation notes (e.g., FlashAttention, torch.compile, GPT-2 architecture) appear mid-section and interrupt the theoretical flow. These should be moved to an implementation or appendix section.  
- Overall, while the technical components are standard, the lack of structure and guiding intuition in the method section reduces readability and makes it harder to assess novelty and correctness.
- The proposed method appears relatively straightforward, and the paper does not clearly articulate why prior autoregressive approaches were considered unscalable or how QUETZAL specifically overcomes those limitations. The claimed innovation seems incremental without a strong theoretical or architectural justification for the reported improvements.
- Lots of recent papers are not included in the comparison, like GeoBFN (​​https://arxiv.org/abs/2403.15441), which seems to be better on validity metrics

### Questions
- Why is the model faster than EDM, since it requires diffusion for each atom, while EDM only requires one diffusion trajectory for sampling
- How is the noising scheme and denoising scheme defined for each atom decoding step

### Soundness
2

### Presentation
1

### Contribution
2
