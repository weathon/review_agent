# EXPLOR: Extrapolatory Pseudo-Label Matching for OOD Uncertainty Based Rejection

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
EXPLOR is a novel framework that utilizes support-expanding, extrapolatory pseudo-labeling to improve prediction and uncertainty-based rejection on out-of-distribution (OOD) points. EXPLOR utilizes a diverse set of pseudo-labelers on an expansive augmented dataset to improve OOD performance through multiple MLP heads (one per pseudo-labeler) with shared embedding trained with a novel per-head matching loss. Unlike prior methods that rely on modality-specific augmentations or assume access to OOD data, EXPLOR introduces extrapolatory pseudo-labeling on latent-space augmentations, enabling robust OOD generalization with any real-valued vector data. In contrast to prior modality-agnostic methods with neural backbones, EXPLOR is model-agnostic, working effectively with methods from simple tree-based models to complex OOD generalization models. We demonstrate that EXPLOR achieves superior performance compared to state-of-the-art methods on diverse datasets in single-source domain generalization settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EXPLOR, a framework for single-source out-of-distribution (OOD) generalization and uncertainty-based rejection. The method integrates three elements: (1) latent-space extrapolation to synthetically expand training support, (2) supervision from a diverse set of pseudo-labelers, and (3) a multi-head student trained with per-head and mean-matching losses. A bias–variance decomposition is provided as an intuitive justification. Experiments on seven chemical and several tabular datasets demonstrate consistent improvements in AUPRC and AUPRC@R<τ over relevant baselines.

### Strengths
- **Relevance and clarity**
The paper addresses an important problem in OOD generalization under a single-source assumption. The motivation is clear, the method is concise, and the presentation is easy to follow.

- **Methodological consistency**
The combination of per-head matching and latent extrapolation provides a coherent multi-task formulation that plausibly enhances robustness.

- **Reproducibility and efficiency**
The implementation is simple, cost-efficient, and reproducible with standard hardware.

### Weaknesses
- **Redundancy at inference**
EXPLOR retains all pseudo-labelers at test time (Eq. 8), forming a hybrid ensemble of teachers and student heads. This design questions whether the student truly distills ensemble knowledge or merely supplements it. The claimed variance-reduction interpretation (Eq. 9–10) remains heuristic without quantitative analysis.

- **Limited conceptual novelty**
Each component of EXPLOR—ensemble pseudo-labeling, latent-space augmentation, and mean-matching regularization—has clear precedents in prior work (e.g., Hydra 2021, PixMix 2022, ACET 2019). The paper’s main contribution lies in system-level integration within a single-source OOD setup rather than in a new theoretical insight.

- **Dataset and representation scope**
The use of ChEMBL, TDC, and DrugOOD benchmarks is reasonable, but all experiments employ fixed ECFP4 fingerprints. This limits the generality of the “modality-agnostic” claim. Results on learned embeddings, such as graph neural networks or visual features, would better demonstrate transferability.

- **Ablation and statistical support**
Per-head matching and bottleneck ablations show small absolute gains (around 1 percent) and lack significance testing. The uncertainty claims would be stronger with calibration or variance metrics (for example expected calibration error).

- **Heuristic theoretical argument**
The bias–variance decomposition provides intuition but lacks empirical verification. No analysis is presented to confirm that predictive variance actually decreases as claimed.

### Questions
**Inference overhead**
What is the actual computational overhead in terms of time and memory when retaining up to 1024 pseudo-labelers during inference compared with using the student alone? A quantitative comparison would clarify the method’s practicality.

**Dependence on pseudo-labelers**
How would EXPLOR perform if pseudo-labelers were unavailable at test time? Would the student alone preserve similar accuracy and uncertainty behavior?

**Generality across learned embeddings**
Does the proposed framework extend beyond fixed handcrafted features to learned embeddings such as those obtained from graph neural networks or visual backbones? Showing this would strengthen the claim of modality-agnostic generalization.

**Notation clarification for Equation 8**
Equation 8 appears to omit parentheses in the summation term.

### Soundness
4

### Presentation
4

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
This paper proposes a new method for single-source domain generalization. The method trains multiple pseudo-labelers on different data subsets. It expands training data through latent-space augmentations. Then it trains a multi-headed network where each head matches a different pseudo-labeler. The approach works with any vector data and different base models. Experiments show strong performance on high-confidence predictions.

### Strengths
1. Single-source domain generalization is realistic for drug discovery applications. Many real scenarios only have one labeled dataset. The paper targets this practical setting.

2. Works with tree models and neural networks. Works with any vector data. Unlike image-specific methods, this is general purpose. Can be applied to many domains.

3. Tests on diverse datasets including chemical and tabular data. Compares with multiple baselines. Shows consistent improvements over pseudo-labelers across datasets.

### Weaknesses
1.  The expansion pushes points away from origin by z' = (1 + |ε|)z. But no validation that expanded points are realistic. They might just be random noise. No check if they represent plausible OOD samples.

2. The method relies heavily on pseudo-labels from diverse labelers. But no check if pseudo-labels are reliable on OOD data. If all pseudo-labelers are wrong, student learns bad supervision. No quality control mechanism.

3. Diverse ensembles are standard ensemble practice. Pseudo-labeling is well-known in semi-supervised learning. Main contribution is combining them with latent expansion. The individual components are not new.

### Questions
1. How do you verify expanded points are realistic? Can you show they resemble real OOD samples?

2. How do you ensure pseudo-labels are reliable? What if all pseudo-labelers are wrong?

3. Why 1024 models? How did you choose this number? What happens with 64 or 256?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces EXPLOR, a method for single-source out-of-distribution (OOD) generalization using pseudo-label matching across multiple heads.
The core idea is to expand latent representations beyond the in-distribution manifold via simple scaling, generate pseudo-labels from diverse models (e.g., XGBoost, D-BAT), and train a multi-head network to match each pseudo-labeler’s predictions.
The approach aims to improve high-confidence OOD predictions and rejection accuracy.
Experiments on chemical (ChEMBL, DrugOOD) and tabular (Tableshift) datasets show consistent improvements over baseline and semi-supervised methods.

### Strengths
1. Clear problem formulation. The paper targets the single-source OOD setting, which is realistic but rarely addressed. The motivation from drug screening and risk prediction is well justified.

2. Simple yet effective design. EXPLOR combines latent-space expansion and per-head pseudo-label matching into a lightweight framework that requires no unlabeled OOD data or domain annotations.

3. Strong empirical performance. Across more than ten datasets, EXPLOR achieves stable gains, particularly in the high-confidence regime (AUPRC@R < 0.2).

4. New evaluation metric. The introduction of AUPRC@R < τ provides a practical way to assess selective prediction quality, relevant to safety-critical tasks.

### Weaknesses
1. Limited novelty. The method essentially combines self-training, ensemble averaging, and multi-task learning in a new context.
No fundamentally new theoretical idea or architecture is introduced.

2. Empirical generality overstated. Although the paper claims to be modality-agnostic, all experiments are confined to tabular data.
There is no evidence that the approach extends to images, text, or graphs.

3. Dependence on pseudo-label quality. The performance gain scales with the accuracy of pseudo-labelers. Poor labelers could degrade the overall results, and this dependency is only briefly mentioned in the appendix.

4. Weak theoretical justification. The variance-reduction derivation (Eq. 9–10) is heuristic; there is no formal analysis showing that latent scaling approximates the true OOD distribution.

5. Overstated terminology. Terms like “extrapolatory” or “modality-agnostic” may mislead readers, given the limited scope of experiments.

### Questions
Please carefully read Weakness and answer all the five concerns.

### Soundness
3

### Presentation
3

### Contribution
3
