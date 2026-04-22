# OXtal: An All-Atom Diffusion Model for Organic Crystal Structure Prediction

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Accurately predicting experimentally realizable 3D molecular crystal structures from their 2D chemical graphs is a long-standing open challenge in computational chemistry called crystal structure prediction (CSP). Efficiently solving this problem has implications ranging from pharmaceuticals to organic semiconductors, as crystal packing directly governs the physical and chemical properties of organic solids. In this paper, we introduce OXtal, a large-scale 100M parameter all-atom diffusion model that directly learns the conditional joint distribution over intramolecular conformations and periodic packing. To efficiently scale OXtal, we abandon explicit equivariant architectures imposing inductive bias arising from crystal symmetries in favor of data augmentation strategies. We further propose a novel crystallization-inspired lattice-free training scheme, Stoichiometric Stochastic Shell Sampling ($S^4$), that efficiently captures long-range interactions while sidestepping explicit lattice parametrization---thus enabling more scalable architectural choices at all-atom resolution.
By leveraging a large dataset of 600K experimentally validated crystal structures (including rigid and flexible molecules, co-crystals, and solvates), OXtal achieves orders-of-magnitude improvements over prior ab initio machine learning CSP methods, while remaining orders of magnitude cheaper than traditional quantum-chemical approaches. Specifically, OXtal recovers experimental structures with conformer $\mathrm{RMSD}_1<0.5$ Å and attains over 80% packing similarity rate, demonstrating its ability to model both thermodynamic and kinetic regularities of molecular crystallization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces OXTAL, an all-atom diffusion model for molecular crystal structure prediction (CSP).
Trained on ∼600k crystals from the Cambridge Structural Database (CSD), OXTAL learns the joint distribution of molecular conformations and periodic packings using a new S4 (Stoichiometric Stochastic Shell Sampling) scheme.
It achieves strong performance on CCDC blind tests with competitive lattice recovery compared to DFT-based CSP at vastly lower cost.

### Strengths
This is among the most technically compelling works to date on molecular crystal structure prediction, and, to the best of my knowledge, the first diffusion-based framework that addresses molecular crystals comprehensively.
- Innovative lattice-free diffusion formulation (S4).
- Extensive benchmarks and realistic evaluation metrics (RMSD₁, RMSD₁₅, Smooth-LDDT).
- Strong empirical results and clear scientific motivation.

### Weaknesses
- The training dataset (CSD) is under commercial license, and the authors will not release the training code. This prevents full replication of results.
- The reason given (“CSD redistribution restrictions”) does not strictly apply to code; code could be released with dummy data-loading interfaces.
- The paper does not evaluate OXTAL on public molecular crystal datasets (e.g., COD, MP-organic subsets), so it is unclear whether the method generalizes beyond CSD.
- The paper could include more quantitative analysis on the impact of shell radius or token size in S4 on model accuracy.
- Minor language and formatting issues:
  * “do we 2 rounds of recycling” → should be “we do 2 rounds of recycling.”
  * “orders-of-magnitude lower inference cost” → better phrased as “several orders of magnitude lower.”
  * Variables  L and  d in Eqs. (21–27) are undefined in the main text.

### Questions
- Could the authors clarify why the training code cannot be released if it only depends on the data interface, not the CSD files themselves?
- How does OXTAL perform on public datasets (e.g., COD) or inorganic datasets to demonstrate cross-domain generalization?
- What is the impact of using non-equivariant Transformer architecture compared to SE(3)-equivariant baselines such as GemNet-OC or EquiformerV2 in this setting?
- Could a smaller-scale variant of OXTAL (e.g., <100 M parameters) achieve comparable results with fewer GPU resources?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents OXtal, an all-atom diffusion model for molecular crystal structure prediction (CSP). The method directly learns the conditional joint distribution of intramolecular conformations and periodic crystal packing from input 2D chemical graphs. A key novelty is the introduction of the Stoichiometric Stochastic Shell Sampling ($S^4$) scheme, which avoids explicit lattice parameterization while capturing long-range interactions. Extensive experiments show that OXtal substantially improves over prior machine-learning CSP methods and even partially surpasses DFT-based methods on CCDC CSP blind tests.

### Strengths
1. To the best of the reviewer's knowledge, this is the first work to apply a full all-atom diffusion framework to molecular crystal structure prediction, which is a significant leap compared to previous approaches.

2. The empirical results are impressive, especially in realistic CCDC CSP blind tests, suggesting both effectiveness and efficiency of the proposed framework.

3. The $S^4$ representation is a novel idea for trading off explicit lattice parametrisation by sampling shells around stoichiometric units, which may offer a scalable solution for large-scale training of molecule CSP models.

### Weaknesses
While the proposed $S^4$ representation enables training on the proposed all-atom backbone model and the diffusion framework, there remain several issues and open questions that should be addressed or clarified, which are listed as follows:

1. The number of conformations inside each $S^4$ unit is not deterministic for one molecule crystal. In generation time, how many conformers one should generate per $S^4$ unit? Does this number affect downstream results?
2. Within each $S^4$ unit, the consistency of generated conformations is unclear. If conformations vary slightly or even significantly, how does one pick a representative conformer to compute RMSD or other evaluation metrics?
3. Although the selection of the lattice matrix is not unique, is it possible to recover one $S^4$ unit into a valid $(L,B)$ pair, i.e. an arbitrary unit cell? If so, can the detailed algorithm be provided?

### Questions
In addition to the above weaknesses specific to $S^4$, the reviewer has the following additional questions:

1. In the case where $n_S>1$, how are the evaluation metrics chosen? For instance, are they the average over all samples, or the best among them, or some other criterion?
2. In the CSP Blind Test 5 benchmark, the OXtal method outperforms prior DFT-based methods on most metrics, yet on the Sol_C subset there remains a noticeable gap. What are the suspected causes of the lower performance on Sol_C? Can the authors identify any corner-cases that the model struggles with and that would guide future improvements?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents OXTAL, a large-scale diffusion model for organic crystal structure prediction (CSP). The model uniquely abandons explicit equivariant architectures, instead training a non-equivariant transformer on a massive 600K-structure dataset. It introduces "Stoichiometric Stochastic Shell Sampling" (S⁴) to train on local, non-periodic atomic blocks, aiming to learn both intramolecular conformation and intermolecular packing. The method reports state-of-the-art results among machine learning baselines and achieves performance competitive with DFT methods at orders-of-magnitude lower inference cost.

### Strengths
1. Performance and Efficiency: The model demonstrates impressive results, significantly outperforming existing ML baselines and achieving a massive reduction in computational cost compared to traditional DFT-based CSP workflows.

2. Scalability and Data: The work successfully scales a generative model to an exceptionally large and diverse dataset (600K CSD structures), a non-trivial data engineering and modeling achievement.

3. Clarity and Presentation: The figures and tables are clear, informative, and visually compelling.

### Weaknesses
1. Input Conformer Dependency: The model relies on a pre-optimized GFN2-xTB 3D conformer as input. This introduces a significant confounding variable and undermines the claim of learning a "joint distribution" ab initio; it appears to be refining a pre-calculated conformer rather than generating one from scratch.

2. Weak Theoretical Justification (Prop. 1): Proposition 1 merely describes a standard geometric surface-to-volume scaling law $O(T^{-1/3})$. It provides no evidence that the model learns true long-range periodicity or order from the local S⁴ crops.

3. Lack of Ablation Studies: The paper is missing crucial ablation studies. The individual contributions of the S⁴ sampling scheme, the non-equivariant architecture, and the large dataset are not decoupled, making it difficult to assess the true sources of the performance gain.

### Questions
1. Learned Representations (Fig. 4): In Figure 4, the generated molecules appear to have an identical orientation, suggesting the model may have only learned a packing arrangement rather than diverse, atomic-resolution geometries. Is this a common failure mode, and what does it imply about the model's learned representations?

2. Robustness to Input: How robust is OXTAL's performance to the quality of the initial GFN2-xTB conformer? What happens if the input is a high-energy or physically unrealistic conformer?

3. Choice of Architecture: Given that the S⁴ method trains on local atomic blocks, why was an explicitly SE(3)-equivariant denoiser (e.g., based on Equiformer or E(3)-GNNs) not used?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces OXTAL, a large-scale, all-atom diffusion model designed to tackle the problem of organic crystal structure prediction (CSP). The model aims to predict the complete 3D crystal structure—encompassing both the molecule's internal conformation and its periodic packing arrangement—conditioned only on the 2D molecular graphs.

### Strengths
1. The idea of the lattice-free approach is interesting. Instead of "generate a valid unit cell," it becomes "generate a locally consistent, periodic atomic environment". This is a fundamental and elegant simplification that is physically motivated by the local-to-global nature of crystallization.

2. The decision to abandon explicit equivariance in favor of a massive model, massive data, and simple data augmentation is a bold and counterintuitive claim compared to most current research literature in geometric ML for this application.

3. The cost-performance plot in Figure 7 is an amazing result. It demonstrates that the proposed OXTAL provides a solution that is orders of magnitude cheaper than the baseline models.

### Weaknesses
1. The paper's most important claim is that abandoning explicit equivariance is an interesting idea. This is a central, non-obvious claim. However, there is no ablation study to back this up. The ablation study regarding other components in the $S^4$ is also missing.


2. Some baseline models are missing in the experiments, including MatterGen[1] and ADiT[2]. It is important to compare the proposed method with the latest methods in this evolving field.


3. Algorithm 1 introduces several new hyperparameters, including the shell radius, which is particularly critical. The paper provides no sensitivity analysis for these choices.



[1] Claudio Zeni, Robert Pinsler, Daniel Zugner, Andrew Fowler, Matthew Horton, Xiang Fu, Zilong Wang, Aliaksandra Shysheya, Jonathan Crabbe, Shoko Ueda, et al. A generative model for inorganic materials design. Nature, 639(8055):624–632, 2025. 


[2] Chaitanya K. Joshi and Xiang Fu and Yi-Lun Liao and Vahe Gharakhanyan and Benjamin Kurt Miller and Anuroop Sriram and Zachary W. Ulissi. All-atom Diffusion Transformers: Unified generative modelling of molecules and materials. International Conference on Machine Learning, 2025.

### Questions
1. Could you provide an ablation study comparing OXTAL's non-equivariant Pairformer trunk against a state-of-the-art equivariant trunk (e.g., an equivariant transformer or GNN) trained with the same $S^4$ cropping method?


2. Can you provide a baseline comparison showing the benefit of the $S^4$ "lattice-free" scheme?

3. How sensitive is the model's performance to the choice of the shell radius?

4. Could you please elaborate on how the model handles co-crystals and solvates with specific stoichiometries? How is the model conditioned at inference time to produce, given that the input is just the SMILES strings?

### Soundness
3

### Presentation
3

### Contribution
3
