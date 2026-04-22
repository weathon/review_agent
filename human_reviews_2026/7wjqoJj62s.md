# Soft Non-Diagonality Penalty Enables Latent Space-Level Interpretability of pLM at No Performance Cost

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Emergence of large scale protein language models (pLMs) has led to significant performance gains in predictive protein modeling. However, it comes at a high price of interpretability, and efforts to push representation learning towards explainable feature spaces remain scarce. The prevailing use of domain-agnostic and sparse encodings in such models fosters a perception that developing both parameter-efficient and generalizable models in a low-data regime is not feasible. In this work, we explore an alternative approach to develop compact models with interpretable embeddings while maintaining competitive performance. With the Bidirectional Long Short-Term Memory Autoencoder (BiLSTM-AE) model as an example trained on positional property matrices, we introduce a soft weight matrix non-diagonality penalty. Through Jacobian analysis, we show that this penalty aligns embeddings with the initial feature space while leading to a marginal increase in performance on a suite of four common peptide biological activity classification benchmarks. Moreover, it was demonstrated that the use of one-hot encoded sequence clustering-based contrastive loss to produce semantically meaningful latent space allows to further improve benchmarking performance. The use of amino acid physicochemical properties and density functional theory (DFT) derived cofactor interaction energies as input features provides a foundation for intrinsic interpretability, which we demonstrate on fundamental peptide properties. The resulting model is over 33,000 times more compact than the state-of-the-art pLM ProtT5. It demonstrates performance stability across diverse benchmarks without task-specific fine-tuning, showcasing that domain-tailored architectural design can yield highly parameter-efficient models with fast inference and preserved generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Soft Non-Diagonality Penalty (SNDP), a regularization method for Vision Transformers (ViTs). The idea is to encourage moderate decorrelation among feature dimensions rather than strict orthogonality by introducing a learnable penalty on the off-diagonal elements of the covariance matrix. The authors argue that complete decorrelation may harm representational capacity, and that “soft” non-diagonality achieves a better trade-off between feature independence and expressiveness. Experimental results on ImageNet and downstream tasks show small but consistent performance improvements.

### Strengths
The motivation is intuitively appealing: enforcing complete feature independence may be suboptimal, and a learnable “soft” constraint could allow ViTs to retain richer dependencies.

The method is lightweight, general, and easily pluggable into existing architectures without architecture modification.

The paper is clearly written and well-organized, with ablation studies and visual analyses that demonstrate stable training behavior.

### Weaknesses
Limited empirical evidence for the central claim (“soft > hard”).
While the experiments show that SNDP enhances interpretability and sometimes accuracy, the evidence remains indirect and confined to a single setup.

The paper provides positive signals (ablation, Jacobian diagonality, correlation with chemical attributes), showing SNDP helps or at least does not harm performance.

However, there is no direct comparison with hard orthogonality or whitening baselines, so it remains unclear whether soft non-diagonality is indeed superior to full decorrelation.

The absence of a λ-sweep experiment leaves open whether the observed improvements simply reflect a moderate regularization effect rather than an optimal “soft” balance.

Missing quantitative trade-off analysis.
The hypothesis of a performance–diagonality Pareto balance is central, yet there is no curve or quantitative exploration showing where SNDP achieves the best equilibrium between independence and expressiveness.

Limited scope and external validity.
All experiments are conducted on a narrow set of ViT backbones and datasets. Without testing on broader architectures or modalities, it is difficult to judge whether the claimed benefits generalize beyond the current configuration.

Lack of theoretical grounding.
The argument for “soft correlation is beneficial” remains empirical. There is no formal justification or analytical insight into why or when partial correlation improves representation learning.

Incremental methodological novelty.
The method extends existing decorrelation and orthogonality regularizers by adding learnable weights, which feels like an evolutionary, rather than revolutionary, step. The conceptual novelty might be insufficient for ICLR unless supported by deeper analysis.

### Questions
Could you provide a λ–performance trade-off curve (e.g., accuracy vs. off-diagonal norm) to substantiate the claimed balance between correlation and independence?

How does SNDP compare against hard decorrelation or whitening methods (e.g., Barlow Twins, VICReg, or orthogonal regularization)?

Can you show feature correlation heatmaps or singular-value spectra to illustrate the “soft” structure SNDP encourages?

Have you tested SNDP on larger backbones or different tasks (e.g., detection, segmentation, or multimodal learning) to support generalization?

Could you include a conditional theoretical discussion or hypothesis on when and why partial correlation yields better generalization?

### Soundness
3

### Presentation
2

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
The authors propose a parameter-efficient protein language model focused on peptides, combining physicochemical features with a BiLSTM autoencoder and a soft non-diagonality penalty for interpretability. While the goal of developing compact, interpretable models is laudable, the work suffers from a limited scope, insufficient engagement with literature, and some overclaiming. The paper's focus on peptides under 20 residues fundamentally limits its utility, and the writing quality is iffy.

The most significant limitation is the exclusive focus on short peptides. The claim of developing a "protein language model" when restricting to peptides feels misleading. PLMs typically excel on longer sequences where complex dependencies and evolutionary patterns emerge. For sequences under 20 residues, it's unsurprising that simple encodings (one-hot, physicochemical features) remain competitive or superior. The settings in which PLMs outperform 1-hot encodings for supervised property prediction (as in this paper) has been an active area of research (e.g., Hsu et al., Nature Biotech 2022). 

The soft non-diagonality penalty, presented as a major contribution, amounts to penalizing off-diagonal elements of weight matrices and computing a simple diagonality metric from the Jacobian. The extensive build-up to this relatively minor regularization technique gets to my earlier point about overclaiming. 

Specific Issues:

- Missing baselines: What about ESM-2 8M or other small PLMs at 35M capacity? These models work surprisingly well and would be more appropriate comparisons than ProtBERT or ProtT5.
- Literature gaps: The paper ignores substantial work on PLM interpretability, including recent advances using sparse autoencoders (e.g. InterPLM). 
- Writing quality: The manuscript is repetitive, weird mix of active and passive voice, and the tense etc are inconsistent. Sections 3-5 could be condensed by half without losing meaningful content.
- Architectural choices: The number of variations tried in Appendix B is very large, verging on p-hacking? It's not clear to me if there's a nice separation between train, val and test. 
- What about other physicochemical feature sets for amino acids, such as VHSE?

### Strengths
see above

### Weaknesses
see above

### Questions
see above

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on learning interpretable representations of peptides and evaluates a number of autoencoder-based architectures that solely rely on molecular descriptors. While underperforming compared to state-of-the-art protein language models, the tested models are significantly smaller and are also shown to be more interpretable, in the sense that some of their embedding dimensions highly correlate with certain physicochemical properties of the peptides.

### Strengths
- Issues of scaling and interpretability in protein/peptide foundation models are critical issues, and I appreciate the authors' attempt to address these issues using simple model architectures.
- The presented correlation results in Table 3 are interesting and shed light on the interpretability of the embedding space.

### Weaknesses
- The novelty and the contribution of the paper to the machine learning community are marginal in my opinion. Several standard architectures were trained and evaluated on peptide data, and the soft penalty term does not involve a significant contribution in and of itself.
- In lines 95-96, it is mentioned that PLMs' performance on peptides is on par with one-hot encoding methods. It is unclear whether this is due to the underrepresentation of short peptide sequences in the training datasets of state-of-the-art PLMs; there is no discussion on whether removing such a bias could significantly improve PLMs' performance on short amino acid sequences.
- The writing of the manuscript needs improvement. Several acronyms were either undefined (to the best of my knowledge) or only defined after several occurrences. As an example, DFT was used 5 times before being defined on page 4. Additionally, the second paragraph of the introduction is too lengthy and should be broken down.
- Following the above point, the related work can be significantly shortened due to the similarities between different PLM architectures. Instead, I suggest highlighting prior work on the interpretability of protein language models.

### Questions
- The statement on line 23 ("while leading to a marginal decrease in performance on a suite of four common peptide biological activity classification benchmarks") directly contradicts the one on line 58 ("while leading to even marginal increase in performance on a suite of four peptide biological activity classification benchmarks").
- On line 246, $M$ is not necessarily a square matrix (when $n \neq k$). How is the identity matrix $I$ and the Hadamard product then defined?

### Soundness
2

### Presentation
2

### Contribution
2
