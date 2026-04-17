# SymSpectra: Symmetric Information Bottleneck Framework for Molecular Structure Recognition under Imbalanced Settings

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Identifying molecular structures from spectral data is essential for early-stage chemical analysis, yet it remains a difficult task due to the imbalance in functional group distributions. Current methods often overfit to prevalent groups while neglecting underrepresented ones, failing to capture key dependencies between functional groups. This highlights the need for a unified approach that addresses both data imbalance and structural constraints. In this work, we present \textbf{SymSpectra}, a \textbf{Sym}metric Conditional Information Bottleneck (SCIB) framework designed to seamlessly integrate multi-modal \textbf{Spectra} features. Our model employs the SCIB framework to fuse multi-modal spectroscopic data into a unified representation, effectively preserving discriminative signals while mitigating redundancy. To enhance robustness against data imbalance, we incorporate conditional mutual information into the training objective, increasing the model’s sensitivity to rare functional groups and challenging molecular cases. Additionally, a specialized module captures the dependencies among functional groups, improving both prediction accuracy and chemically meaningful interpretability. Experiments on multimodal spectral datasets demonstrate that SymSpectra significantly outperforms state-of-the-art methods, achieving an F1-score of 0.970 in substructure classification. More importantly, SymSpectra consistently outperforms baselines under various imbalanced scenarios, exhibiting superior robustness and generalizability, which may help advance the automation of chemical discovery. Our code can be found at \href{https://anonymous.4open.science/r/SymSpectra-0017}{https://anonymous.4open.science.}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SymSpectra, a technique for mitigating class imbalance in predicting the presence of functional groups from spectral data such as IR and NMR. The method is grounded in an information-theoretic formulation and is shown to outperform existing methods in the domain.

### Strengths
- The theoretical framework is innovative. To the best of my knowledge, a similar information-theoretic approach has not been explored for spectrometric or spectroscopic modalities before.
- The proposed method outperforms recently published studies.
- The work is comprehensive, with attention to detail from data preparation to ablations.
- The paper is accompanied by clean source code.

### Weaknesses
- My major concern is that the paper does not compare SymSpectra against standard class-imbalance techniques such as under/oversampling, inverse class-frequency loss weighting, focal loss, etc.
- Quite many details are unclear:
  - How were the baselines retrained?
  - How is the graph in Figure 1c constructed? Is it based on the first section of Appendix L.3?
  - Figure 2 is difficult to understand without a detailed caption; some terms in the figure do not seem to match the main text.
  - Equation 1: please specify what the modalities are and define I (mutual information).
  - Line 169: what are B, L, C?
  - Line 186: what exactly is H_j^I?
  - Equation 8: why does the training objective not involve labels Y?
  - Lines 318–320: the definitions of micro- and macro-averages appear to be swapped.
  - Line 347: please define “sample-level”.
  - Line 365: what are “importance distributions”?
  - Line 377: it is not clear what “shown” refers to
  - Line 398: how are the other splits constructed (e.g., those used in Figure 3a)?
  - Table 2: what do 8, 9, 13, 16, and 22 denote? Please name functional groups
- While it’s a minor point, it would be valuable to evaluate SymSpectra on a standard benchmark for spectra, such as MassSpecGym in the uni-modal setting (https://arxiv.org/abs/2410.23326).

### Questions
- Can the authors demonstrate that SymSpectra outperforms standard imbalance-mitigation techniques (e.g., class weighting, focal loss, under/oversampling) under the same training and evaluation protocol?

### Soundness
2

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
This paper proposes a multimodal spectra fusion framework grounded in information theory, aiming to improve rare class functional group classification. The method fuses multiple spectroscopic modalities, including NMR, IR, MS, and MS/MS, and claims that the proposed model enhances cross-modal feature fusion by suppressing redundant information while retaining discriminative signals.
Experiments are conducted on two datasets, the NeurIPS 2024 benchmark dataset and the SDBS database. The authors report performance across various modality combinations.

### Strengths
The authors make an effort to ground the multi-modal fusion in information theory, attempting to provide a conceptual understanding of how IB may contribute to effective multi-modal feature selection.

The paper demonstrates reasonably strong performance across multiple modality combinations and datasets.

### Weaknesses
## Limited competing methods
The baseline selection appears weak and mostly borrowed from the NeurIPS 2024 benchmark models. The paper would benefit from including stronger and more recent competing methods in spectrum-based structure elucidation. If the goal is to demonstrate the advantage of multimodal fusion, it is essential to compare against the best single-modality SOTA models rather than relatively simple baselines. Recent relevant works such as DiffMS (ICML 2025), DiffSpectra (arXiv 2025), and LLM-based molecular reasoning benchmarks (NeurIPS 2024) should be considered or at least discussed.

## Gap between theory and implementation.
While the paper presents extensive information theory (IB, CMI), the implementation diverges from the stated formulation. In practice, the modality “information bottleneck” seems to be merely cross–attention–based modality weighting combined with element-wise feature summation and a shallow 1D CNN fusion. 

CA weighting + summation + 1D CNN

Also, the proposed total loss function is closer to a focal loss style weighting, rather than a principled information-theoretic objective. 

## Justification for improvements on rare functional groups.
The paper emphasizes improvements for rare groups, but the underlying cause of this improvement is not clearly attributed to a specific component of the method. Beyond empirical performance, a deeper analysis or ablation that isolates which module (SCIB, dynamic reweighting, sequential FG decoding, etc.) contributes to rare-class gains would strengthen the claim. At present, the explanation lacks theoretical or mechanistic grounding.

### Questions
See weakness

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
3

### Summary
The paper introduces SymSpectra, a model designed to predict molecular substructures from multi-modal spectral data. While the proposed approach appears novel, its practical significance and methodological contribution are not fully clear.

### Strengths
- The paper addresses the problem of class imbalance, which is an important and underexplored challenge in machine learning for small molecules.
- The analyses of model performance in Section 3 and the Appendix are comprehensive.

### Weaknesses
Major comments

- Lack of clarity. The Abstract and Introduction emphasize class imbalance as the central motivation, yet Section 2 (Methodology) primarily focuses on multi-modality. These two issues appear to be intertwined without clear justification or structure. This weakens the overall narrative of the paper. Please also refer to the minor comments below.
- Practical relevance is unclear. It appears that Alberts et al. is the only non-trivial baseline (beyond simple 1D-CNN or Transformer) comparable to the proposed method (Wu et al. seems to be proposed for infrared spectra rather than multi-modal data). Furthermore, Section 3.1 notes that no publicly available multi-modal datasets beyond approximately 12K compounds exist. This raises concerns about the practical relevance and scalability of multi-modal spectral prediction. The paper should better justify why predicting molecular classes from multi-modal spectra is meaningful in real-world applications, supported by appropriate references.
- Unclear methodological novelty. The paper does not clearly explain how SymSpectra improves upon or differs from the approach of Alberts et al. A more explicit comparison of architectural components and methodological innovations is needed.
- Missing baselines for class imbalance. Although class imbalance is presented as the central challenge, the paper omits standard baseline techniques commonly used to address this issue, such as sample reweighting, undersampling/oversampling strategies [1], or focal loss for classification [2].

Minor Comments

- Figure 1:
  - Panel (a): Please specify the underlying dataset and clarify what the plotted values are, and their scale.
  - Panel (b): The comparison of significance across methods is unclear—shouldn’t significance be calculated for frequencies which is the main message of the panel?
  - Panel (c): The purpose and interpretation of this panel are unclear; please elaborate on its relevance.
- Figure 3d: The caption mentions “correlation,” but no correlation is analyzed in the figure itself. Please clarify or revise.

[1] He et al., 2009, Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering

[2] Lin et al., 2017, Focal Loss for Dense Object Detection. arXiv:1708.02002

### Questions
1. What would be the performance (e.g., in Figure 3d) of the method proposed by Alberts et al. if combined with standard class imbalance mitigation techniques, such as class reweighting? (Please refer to the corresponding discussion in the Weaknesses section above.)

### Soundness
2

### Presentation
2

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
This paper presents SymSpectra, a framework designed to identify molecular functional groups from multi-modal spectroscopic data (IR, NMR, MS) while addressing class imbalance challenges. The core contribution is a Symmetric Conditional Information Bottleneck (SCIB) approach that fuses spectral modalities, combined with conditional mutual information (CMI)-based dynamic weighting to prioritize rare functional groups and an LSTM decoder to model functional group dependencies. The authors report achieving an F1-score of 0.970 on simulated data and demonstrate robustness under various imbalanced scenarios.

### Strengths
Well-motivated problem: The class imbalance issue in functional group prediction is clearly articulated with supporting evidence (Figure 1), showing a 38% F1-score gap between high and low-frequency groups.
Comprehensive experimental validation: The paper includes extensive experiments across multiple dimensions such as multiple baselines (1D-CNN, Transformer, recent SOTA methods), both simulated (794K molecules) and experimental (12K molecules from SDBS) datasets and multiple imbalance scenarios(label imbalance, structural imbalance)

Thoughtful architectural design: The SCIB framework provides a principled approach to multi-modal fusion by explicitly modeling what information each modality uniquely contributes (Equation 1), moving beyond simple concatenation or attention-based fusion.

Interpretability efforts: The paper provides visualization of importance weights (Figure 7) and demonstrates that the model adaptively allocates attention based on sample complexity (Figure 8, Appendix M).

### Weaknesses
1. Limited theoretical justification for SCIB design. The paper drops the I(Y; T^¬i) term from Equation 15 citing "optimization instability" (Section E.1.2), but provides no empirical evidence of this instability. This is a significant theoretical compromise that weakens the SCIB framework's foundational claim.

2. The Gaussian assumptions in Equations 18-19 and 24-25 are strong but not validated. Why should spectral representations follow Gaussian distributions? The paper acknowledges exploring other priors (Appendix G, Table 4) but doesn't explain why Gaussian is fundamentally appropriate for spectral data.

3. The connection between minimizing I(T^i; X^i | X^¬i) and the specific attention-weighted formulation (Equations 26, 30) involves several non-trivial steps that are relegated to the appendix without sufficient intuition in the main text.

4. CMI-based weighting lacks rigorous justification. The paper claims CMI "quantifies and amplifies the informational significance" of rare groups (lines 96-99), but the actual implementation uses auxiliary predictors with modality dropout as a proxy (Section 2.2.3). The connection between this dropout-based approximation and true CMI is not formally established.

5. Equations 11-12 introduce normalization with batch statistics (μ_s, σ_s, μ_f, σ_f) and scaling factors (s1, s2), which appear ad-hoc. Why is this specific normalization scheme information-theoretically justified? The sensitivity analysis (Appendix Q, Figure 11) shows performance is sensitive to these hyperparameters, suggesting they're tuning knobs rather than principled quantities.
6. The claim that this approach is superior to "traditional class reweighting or resampling heuristics" (line 98) is not empirically validated, there is no comparison with focal loss, class-balanced loss, or re-sampling methods is provided.

7. The 1D-CNN and Transformer baselines seem weak (Table 1). For example, the Transformer achieves only 0.881 F1 on IR spectra while SymSpectra gets 0.924. This gap is suspiciously large and raises questions about whether baselines were properly tuned. The paper provides no details on baseline hyperparameters or architecture specifics.

8. Functional group dependency modeling seems oversimplified. The LSTM decoder uses a "predefined order" based on IUPAC nomenclature (Section 3.4, Table 2), which is essentially a chemical convention, not a learned dependency structure. While the paper explores MI-based and GNN-based ordering (Appendix L), these are not used in the final model.

9. The sequential prediction assumes a linear chain dependency structure, but functional group co-occurrence patterns (Figure 1c) suggest a more complex graphical structure. Why not use a conditional random field (CRF) or graph-based structured prediction?

9. The ~12K SDBS experimental dataset is much smaller and shows lower absolute performance across all models (0.919 vs 0.970 F1). This gap raises concerns about sim-to-real transfer, yet the paper doesn't deeply investigate why this gap exists or propose methods to close it.

10. Table 7 shows SymSpectra uses 7GB memory and 2.7 hours training, but what hardware? What batch size? The comparison with Transformer (1.7GB, 35 hours) seems inconsistent, why would the simpler model use less memory but far more time?

### Questions
1. Can you provide empirical evidence of the "optimization instability" claimed for minimizing I(Y; T^¬i)? Have you tried alternative formulations or regularization techniques to include this term?
2. How accurate is the modality dropout-based approximation of CMI? Can you provide theoretical analysis or empirical validation comparing the estimated CMI values with ground truth (if computable on synthetic data)?
3. Comparison with class imbalance methods: Can you compare SymSpectra with established techniques for handling class imbalance (focal loss, cost-sensitive learning, SMOTE, etc.) to validate that the CMI-based approach is indeed superior?
4. Have you considered more sophisticated dependency structures beyond sequential LSTM? Can you compare with CRF or graph neural network-based structured prediction?
5. What causes the significant performance drop from simulated (0.970 F1) to experimental (0.919 F1) data? Can domain adaptation or transfer learning techniques help?
6. What happens if you remove the information bottleneck term entirely (β=0) and just use the prediction loss with CMI weighting? This would help isolate the contribution of SCIB vs. CMI.
7. Table 2 shows different orderings yield substantially different results (e.g., Imine: 0.756 for GNN order vs 0.885 for IUPAC). How sensitive is the final model to ordering choice, and does this suggest the LSTM is exploiting spurious sequential patterns rather than true dependencies?

### Soundness
3

### Presentation
2

### Contribution
3
