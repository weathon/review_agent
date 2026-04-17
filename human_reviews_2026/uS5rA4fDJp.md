# RankFlow: Property-aware Transport for Protein Optimization

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 4

## Abstract
A key step in protein optimization is modeling the fitness landscape, which maps proteins to functional assay readouts. Existing methods typically either use property-agnostic likelihoods/embeddings from pretrained protein language models (PLMs) for fitness prediction, or assume independent mutational effects, limiting their ability to capture higher-order interactions. In this work, we introduce RankFlow, a conditional flow framework that refines PLM representations to be a property-aligned distribution via a tailored energy function and captures multi-mutation interactions through learnable embeddings. To align optimization with evaluation protocols, we propose the Rank-Consistent Conditional Flow Loss (RC$^2$), a differentiable ranking objective that enforces the correct order of mutants rather than absolute values, which improves out-of-distribution generalization. Finally, we introduce a Property-guided Steering Gate (PSG) that concentrates learning on positions carrying signals for the target property while suppressing unrelated evolutionary biases. Across the ProteinGym, PEER, and FLIP benchmarks, RankFlow obtains state-of-the-art ranking accuracy and superior generalization performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents RankFlow, a method for improving protein fitness prediction using flow matching. The main contribution is reformulating fitness prediction as a generative task rather than a regression problem. While the proposed approach is potentially interesting, the paper requires major revisions, particularly to improve clarity in the overall motivation and experimental setup.

### Strengths
- The idea of applying flow matching to protein fitness prediction appears novel.
- The proposed components, such as the Rank-Consistent Conditional Flow Loss and the Property-Guided Steering Gate, are original and well-motivated.

### Weaknesses
Major comments

- Although the method’s technical details are described, the overall intuition and practical usage remain unclear. The motivation behind reformulating fitness prediction as a generative problem (rather than regression) is not well explained. A conceptual figure illustrating this idea would be helpful (see, for example, Fig. 2 in [1]). In addition, several typos make the method difficult to follow (see minor comments).
- The experimental setup is insufficiently described:
  - Evaluation metrics are not mentioned in the main text.
  - The values reported in Table 1 and Table 2 do not match those on the official ProteinGym website [2], and no explanation is provided. For example, ProteinNPT achieves a “Stability” score of 0.904 in the paper, while [2] reports 0.776.
- The evaluation mixes unsupervised models (which do not use any fitness data, e.g., ESM-2, SaProt) with supervised models (e.g., ProteinNPT) without justification or explanation. It is unclear what training data RankFlow uses. The parameter count comparison in the table appears inconsistent. How can RankFlow have fewer parameters than ESM-2 if it uses ESM-2 as its backbone?

Minor comments

- [Line 152] x^{mt} is likely a typo; it should be x^{wt}.
- [Line 257] R_{\tau} is mentioned but never explained.
- [Line 273 and Eq. 13] The notation for S^+ and S^- is inconsistent.
- [Eq. 16] g_i is introduced here but its role is not clarified in the rest of the technical explanation of the method.
- [Line 372] The text claims: “As shown in Fig. 3, variants predicted to have high fitness are at solvent-exposed positions and are located away from the active site, which is in line with established biological knowledge (Notin et al., 2022a).” However, Fig. 3 does not demonstrate this.

[1] Corso et al., 2023. DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking.https://arxiv.org/abs/2210.01776

[2] https://proteingym.org/

### Questions
1. Is performance difference between RankFlow and DePLM statistically significant?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a flow-matching-based framework for protein fitness prediction. For that, they design a new energy function that is combined with representations from pLMs. They also add a gate mechanism to leverage mutational information from a dataset to bias mutations in the framework. Results show that the proposed method achieves state-of-the-art performance for the ProteinGym benchmark and three more tasks. Ablation studies show that the proposed components, especially the proposed energy function, are crucial to achieve good performance.

### Strengths
1. The authors propose a conditional flow framework that learns property-aligned embeddings that improve fitness prediction, especially for cases in which mutants have multiple mutations.
2. A new energy function (Eq. 7) and loss are proposed for protein fitness prediction.
3. The proposed RankFlow method achieves state-of-the-art results on important benchmarks such as ProteinGym.

### Weaknesses
1. Hyperparameters that should be tuned to achieve state-of-the-art performance are discussed only in the Appendix.
2. The implementation of the flow-based method is not clear when following the information in Section 3.2.
3. Additional analysis of the method for proteins without structure available (without the multimodal fusion encoder) seems needed.
4. Code is not available.

### Questions
Overall, the paper contributes a new flow-based framework for protein fitness prediction that could be a contribution to epistasis modeling. My initial recommendation is borderline acceptance, but I would like to discuss with the authors the following comments.

Comments:

1. The proposed method uses structure as input. For the datasets experimented, are all structures experimental or some are predicted? A clarification regarding which benchmark methods use sequence, structure, and sequence-structure seems needed.
2. Many hyperparameters are included in Section 3, and additional information is given only in the Appendix. As these seem to directly influence the state-of-the-art performance of the proposed method, more discussion about how to choose these hyperparameters is needed.
3. For the gate, how much assay data is needed for this term to be effective? Does the nature of assay data influence this term?
4. (lines 315-316) Can the authors elaborate on what it means that the learnable embeddings are applied only at the mutated positions?
5. Additional clarification regarding the modeling of the energy function in Eq. 7 seems needed. The ablations in the Appendix show that this term is crucial for performance.

Minor Comments (that did not impact the score):

1. The use of \citet and \citep is wrong throughout the manuscript. Almost all references are missing parentheses.
2. The acronym of pLMs is defined twice in text.
3. Need consistency in the use of hyphens, e.g., “deep-mutational-scanning”; and when using a comma for numbers.
4. Figure 4 should be improved as it is very hard to read the caption in the printed version.
5. Typo: “reproducibility”

### Soundness
3

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
The goal of this work is to develop a **fitness predictor** to assess the effects of mutations on a given protein sequence (wild type). This is a central problem in computational biology, with applications in both **genomics** and **computational protein design**.  

The authors propose a new computational procedure called **RankFlow**, which employs **flow matching** to learn a distribution in the latent space of a protein language model (PLM). This distribution is *tilted* toward mutants that are both high in fitness and anomalous relative to local substitution trends. Samples from this learned distribution are then passed through a PLM head and subsequently used for fitness prediction.  

Results on relevant benchmarks show that RankFlow performs well compared to multiple existing methods.

### Strengths
- Good performance on multiple relevant benchmarks.  
- The method can be pre-trained on multiple DMS assays, thereby benefiting from previously observed cross-protein transfer effects [1]. I believe this is the main strength of the proposed approach.

### Weaknesses
- The evaluation is missing two important baselines: [2] and [3].  
- The evaluation does not include **out-of-distribution (OOD)** splits from **ProteinGym**.  
- The authors do not propose any means for **uncertainty quantification (UQ)**, which limits the applicability of this predictor for **Bayesian Optimization**.  
- Additional **ablations** on architectural choices would strengthen the work. For example:
  - What happens if the flow-matching module is replaced with a simple MLP layer?  
  - What happens if the ranking loss is replaced with other commonly used losses?  
  - How does this method compare with an approach similar to **CPT-1** [1]?

### Questions
1. Can diffusion be used to provide **uncertainty quantification**? It would be interesting to compare this form of UQ with other approaches [3, 4].  
2. The authors claim that previous methods do not capture **epistasis**, but PLM embeddings can, in principle, model epistasis. Can the authors provide a comparative analysis of their model on multiple mutants versus other methods?  
3. The notation is confusing in places. For example:
   - In **Figure 2**, the unified representation is denoted as *F*, but *F* never appears in the flow-matching equations—can the authors clarify this?
   - In **Equation (3)**, what are \(\dot{y}\) and \(\dot{\sigma}\)?  
4. Can the authors clearly explain how the **datasets are split**? Do they use the random split for ProteinGym?  
5. Can the authors clearly define how **fitness predictions** are generated according to their method in the main text?  
6. Is the method sensitive to any **flow-matching hyperparameters**? How does randomness in the flow-matching process affect predictions?

---

[1] Jagota, M., Ye, C., Albors, C., Rastogi, R., Koehl, A., Ioannidis, N., & Song, Y. S. (2023). *Cross-protein transfer learning substantially improves disease variant prediction.* **Genome Biology**, 24(1), 182.  

[2] Groth, P. M., Kerrn, M., Olsen, L., Salomon, J., & Boomsma, W. (2024). *Kermut: Composite kernel regression for protein variant effects.* **Advances in Neural Information Processing Systems**, 37, 29514–29565.  

[3] Ronen, O., Zhao, A. Y., Boger, R., Ye, C., & Yu, B. (2025). *Stabilizing protein fitness predictors via the PCS framework.* In *The Exploration in AI Today Workshop at ICML 2025.*  

[4] Greenman, K. P., Amini, A. P., & Yang, K. K. (2025). *Benchmarking uncertainty quantification for protein engineering.* **PLOS Computational Biology**, 21(1): e1012639.

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
3

### Summary
This paper presents RankFlow, a property-aware conditional flow model for protein optimization. It refines PLM embeddings into property-aligned representations. It captures complex multi-mutation interactions through an energy-guided flow and a Property-Guided Steering Gate (PSG). The Rank-Consistent Conditional Flow Loss (RC² Loss) enforces correct mutant ranking and ensures out-of-distribution generalization. The model achieves superior performance on various benchmarks.

### Strengths
Unlike existing methods that simply add the mutational effect of an individual site, RankFlow uses energy-guided conditional transport to capture complex multi-mutation interactions. In addition, optimizing the relative ordering of mutants provides better generalization to unknown protein families.

### Weaknesses
The model achieves better performance than previous models. However, there could be more detailed experimental setup descriptions, especially as the setup is not identical to the standard ProteinGym benchmark. The author could also discuss more on other supervised models, such as ProteinNPT and Metalic.

### Questions
1. The ProteinGym benchmark has a standard evaluation pipeline for unsupervised and supervised fitness prediction tasks. What is the difference between the evaluation procedure this paper follows and the standard ProteinGym evaluation pipeline?

2. Are the unsupervised methods and supervised methods evaluated on the same set of protein sequences in Tables 1 and 2?

### Soundness
2

### Presentation
2

### Contribution
3
