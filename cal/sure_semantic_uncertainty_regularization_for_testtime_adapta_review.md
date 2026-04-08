=== CALIBRATION EXAMPLE 22 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "SURE: Semantic Uncertainty REgularization for Test-Time Adaptation in Vision-Language Models" accurately reflects the contribution. The abstract is readable and correctly identifies the core problem: existing TTA methods treat pseudo-labels independently, ignoring temporal reliability and inter-class semantic structure. However, the claim that SURE "consistently outperforms prior methods" deserves closer scrutiny — on several individual benchmarks (e.g., ImageNet-A under ViT-B, EuroSAT under ViT-B), SURE is actually *worse* than some competitors (DPE achieves 59.63%, ZERO achieves 62.75% on ImageNet-A ViT-B vs. SURE's 61.45%). The abstract's language should be more precise.

---

### Introduction & Motivation

The motivation is clear and well-structured. The authors correctly identify that (1) entropy minimization leads to overconfidence in wrong predictions, and (2) early prototype updates propagate noise. These are genuine, well-documented failure modes in TTA.

The contributions are enumerated cleanly. However, **Contribution 3** ("four domain-shift benchmarks and two VLM backbones") is modest scope — only CLIP RN50 and ViT-B/16 are tested. The absence of ViT-L/14 evaluations, which are now the de facto baseline in VLM TTA work, is a notable gap.

A more substantive concern: the introduction does not acknowledge or distinguish from PROGRAM (Sun et al., 2024), which shares the core concept of a prototype graph for TTA. The distinction is only developed in Related Work, and PROGRAM is never used as an experimental baseline — an important omission since it is the most directly comparable prior work.

---

### Related Work

The related work coverage is comprehensive and well-organized. The distinction from PROGRAM (Sun et al., 2024) is articulated clearly: SURE uses reliability-driven, dynamic edge modulation and is VLM-specific, whereas PROGRAM is a static, general-purpose graph over unimodal classifiers. This distinction is important to the paper's originality claim. However, this differentiation must be validated experimentally, and the absence of PROGRAM as a baseline is a critical weakness — readers cannot assess whether the gains come from graph-based reasoning in general or from SURE's specific reliability-weighted design.

---

### Method (Section 4)

**Reliability Score (Eq. 4):** The formulation $R_j = \mu_j \cdot (1 - \sigma_j/\sigma_\text{max})$ is a reasonable heuristic, but several concerns arise:

1. **Initialization bias**: All classes are initialized with $\mu_j = 1.0$ and $\sigma_j = 0.0$, giving every class $R_j = 1.0$ at the start. This means the initial graph is simply the cosine similarity matrix, equivalent to the baseline without any reliability weighting. Early predictions, before any buffer builds up, thus receive no benefit from the reliability mechanism, potentially making SURE indistinguishable from the "Graph w/o Rel" ablation for the first $L$ samples. This is not discussed.

2. **Class imbalance under long-tailed or imbalanced test streams**: Classes that rarely appear in the test stream will have small (or empty) buffers $Q_i$, making $\mu_i$ and $\sigma_i$ estimates unreliable or uninformative. The paper does not address this.

3. **$\sigma_\text{max}$ is a fixed scalar hyperparameter** (set to 0.5), but this value normalizes variance across all classes equally. Classes with inherently high within-class prediction variance (e.g., fine-grained categories) are systematically penalized regardless of actual reliability. No justification or ablation for this choice is given.

**Adjacency Construction (Eq. 7):** The top-$k$ sparsification with $k = 3 \cdot \log(C)$ is a somewhat arbitrary formula. For ImageNet ($C=1000$), this gives $k \approx 20$ neighbors. No ablation justifies the $\log(C)$ scaling — is this principled or empirical? Fig. 3 only varies $k$ for a single fixed $C$, not across datasets with different $C$.

**Logit Combination (Eq. 10):** The final prediction $\hat{p}(y_i|\mathbf{x})$ is a uniform average of local and graph-based scores. There is no learned or adaptive weighting. An obvious question is whether the balance should depend on the current reliability level — if the graph is highly uncertain early in adaptation, the uniform average may hurt rather than help. No ablation on the weighting coefficient is provided.

**One-step propagation:** The method uses a single step of belief propagation (Eq. 9). While computational efficiency is a stated goal, there is no analysis of whether additional propagation steps would help or hurt. For a method that frames itself as principled graph inference, this is surprising.

**Prototype initialization with $N_i^\text{proto} = 30000$**: This is stated in Implementation Details ("each class prototype is initialized with $N_i^\text{proto} = 30000$ confident samples"). This implies initializing the counter to a large number, effectively making early test samples contribute very little to prototype updates. This seems to require access to labeled source training data (or at least a warm-start from source-domain statistics). If so, this is a significant assumption that **several baselines do not share**, potentially making comparisons unfair. The paper does not clarify where these 30,000 samples come from or whether competing methods are given the same initialization.

---

### Experiments & Results

**Table 1 (Natural Distribution Shifts):**
- SURE's improvements on ViT-B are very small in absolute terms: +1.22% over TDA, +0.13% over ZERO on average accuracy. Under RN50, SURE gets 29.57% on ImageNet-A vs. DPE's 30.15% and BCA's 30.35% — i.e., SURE is *worse* than two baselines on the hardest OOD split under the weaker backbone.
- No variance across seeds is reported anywhere in the paper. Given the small margins (e.g., 66.23% vs. 66.10%), statistical significance is entirely unclear.

**Table 2 (Cross-Dataset Generalization):**
- There is a **duplicate entry for ZERO** (Farina et al., 2024): one row labeled "Zero" and another labeled "ZERO" with different numbers (e.g., SUN397: 66.90 vs. 67.63). This is either an error or undisclosed experimental variation that needs clarification.
- SURE is below DPE on several important datasets (e.g., EuroSAT: 53.60 vs. DPE's 55.79; Pets: 89.81 vs. DPE's 91.14). The claim of "best average" is accurate, but performance on individual datasets is non-uniform.

**Table 3 (Efficiency):**
- There is a clear numerical error: SURE achieves 66.23% accuracy and CLIP-ViT-B achieves 61.20%, giving a gain of +5.03%. However, the table reports "+7.12%" in the ∆ Gain column. This is factually incorrect and must be fixed.
- Additionally, MTA's accuracy in Table 3 is listed as 63.16%, but Table 1 reports MTA at 64.06%. This unexplained discrepancy raises concerns about consistency across experimental conditions.

**Missing baseline:** PROGRAM (Sun et al., 2024) — the most directly comparable graph-based TTA method — is discussed in related work but never evaluated. This omission makes it impossible to verify whether SURE's graph-based reliability mechanism provides actual improvements over simpler graph-based TTA.

**Missing corruption benchmarks:** Standard TTA evaluation protocols include ImageNet-C (corruption shifts), which is absent here. The paper only tests "natural" distribution shifts. Corruption robustness is a standard expectation for TTA submissions at ICLR.

**No analysis of class-frequency effects on adaptation**: SURE's reliability mechanism is inherently tied to how many times each class appears. For long-tailed distributions, rare classes will have near-random reliability scores throughout adaptation. This failure mode is unacknowledged.

---

### Ablation Study (Section 5.3)

The ablation in Table 4 is structured, with four progressive variants (ProtoOnly → +Graph w/o Rel → +Graph+Rel → +LogitProp). This is a reasonable design. However:

- The ablation only covers the ViT-B backbone and natural distribution shifts. There is no ablation for cross-dataset generalization, where the performance profile might differ substantially.
- The "+Graph w/o Rel" variant *hurts* performance on ImageNet-A (-0.24% relative to ProtoOnly). This confirms the authors' claim that graph smoothing without reliability control is risky, but it also raises the question: under what test-time conditions might the full SURE model similarly degrade? No analysis of failure modes is provided.
- The hyperparameter study (Fig. 3) reports $\theta = 0.4$ as optimal, but the Implementation Details section states $\theta = 0.3$. This discrepancy is unexplained — which value was actually used for the main results?

---

### Visualization (Section 5.4)

The graph evolution in Fig. 4 is evaluated on only **5 hand-selected classes** under **simulated** distribution shift. This is too narrow to constitute validation of the core mechanism. The example is constructed to show an obvious qualitative result (Television is spuriously connected to Tabby in CLIP). A more rigorous validation would involve quantitative analysis of adjacency matrix evolution on the full ImageNet class set, showing that reliability scores correlate with actual prediction accuracy across classes.

---

### Limitations & Broader Impact

The paper lacks a Limitations section entirely (it goes directly from Conclusion to References). There is no discussion of:
- Scenarios where SURE might degrade performance (e.g., test streams with severe class imbalance, extremely short streams where reliability buffers are uninformative).
- The dependence on a warm-start prototype initialization ($N_i^\text{proto} = 30000$) and whether SURE degrades gracefully when this is unavailable.
- Whether the closed-loop system can diverge — no convergence analysis is given for the feedback loop between prototype updates, reliability estimation, and graph reconstruction.
- Applicability beyond image classification (e.g., retrieval, VQA).

---

### Writing & Clarity

**Algorithm 1 is fragmented**: Lines 4–12 of the algorithm appear discontinuously at lines 576–604 in the parsed file, separated from lines 1–3 and 13–17 by several tables and figures. While this may be a PDF parsing artifact, the algorithm presentation in the paper appears to be split across non-contiguous regions, which hurts reproducibility.

The distinction between the smoothed adjacency $\bar{A}^{(\ell)}$ (Eq. 8) and the per-step adjacency $A^{(\ell)}$ (Eq. 7) is not clearly conveyed in the body text — the reader must infer that graph-based smoothing in Eq. 9 uses $\bar{A}$ and not $A$.

---

### Overall Assessment

SURE presents a coherent, principled approach to test-time adaptation for VLMs by dynamically constructing a reliability-weighted graph over class prototypes. The core idea of combining semantic affinity with temporal confidence stability to gate pseudo-label propagation is well-motivated and addresses a genuine weakness in prior work. However, several concerns diminish the submission's readiness for ICLR. First, the performance margins on the primary ViT-B benchmark are very thin (+0.13% over ZERO, +1.22% over TDA) and are reported without any statistical testing, making it unclear whether the improvements are meaningful. Second, a factual error in the efficiency table (∆ Gain of +7.12% instead of +5.03%), an unexplained discrepancy between Table 1 and Table 3 for MTA, and a hyperparameter inconsistency ($\theta = 0.3$ vs. $\theta = 0.4$) undermine credibility. Third, the most directly comparable baseline — PROGRAM (Sun et al., 2024), explicitly discussed in related work — is omitted from all experiments, making the claimed advantages of SURE's reliability-driven graph design unverifiable. Fourth, the prototype warm-start initialization ($N_i^\text{proto} = 30000$) may give SURE an unfair advantage over some baselines, and its data source is not disclosed. These issues collectively make the paper's empirical claims insufficiently substantiated for acceptance; the method's conceptual contribution is genuine but requires a substantially stronger and more honest experimental evaluation.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces SURE, a gradient-free test-time adaptation (TTA) framework for Vision-Language Models that regularizes predictions via a dynamically evolving Prototype-Reliability Graph (PRG). The PRG fuses semantic affinity from textual prototypes with temporal prediction stability to propagate reliable signals while suppressing noisy pseudo-labels. Through a closed-loop update of prototypes, reliability scores, and graph topology, SURE mitigates semantic drift and error amplification under distribution shifts.

### Strengths
1. **Clear Motivation & Problem Formulation:** The paper accurately identifies the core limitation of existing VLM TTA methods (overconfidence and error propagation from noisy, independently treated pseudo-labels) and proposes a principled, structure-aware alternative grounded in reliability-gated message passing (Sec. 1 & 3).
2. **Strong & Comprehensive Empirical Validation:** SURE delivers consistent state-of-the-art or highly competitive results across 15 benchmarks, including natural shifts (ImageNet-A/V2/R/Sketch) and cross-domain datasets, evaluated on both RN50 and ViT-B backbones (Tab. 1 & 2). The gains are particularly notable on challenging domains like ImageNet-Sketch and fine-grained datasets, demonstrating robustness to severe visual distortion and semantic ambiguity.
3. **Efficiency & Deployment-Friendly Design:** SURE operates without gradient computation or multiple augmented views, achieving ~10× faster inference than TPT while matching/exceeding other gradient-free baselines like ZERO and BCA (Tab. 3). The use of sliding windows and cached statistics makes it well-suited for real-time streaming deployment.
4. **Thorough Ablations & Auxiliary Analyses:** The component-wise ablation (Tab. 4) clearly validates the synergy between graph structure, reliability gating, and logit propagation. Additional analyses on calibration (Tab. 10), prompt sensitivity (Tab. 9), hyperparameter stability (Fig. 3), and cross-seed variance demonstrate methodological maturity and align well with ICLR's emphasis on reproducibility and robustness.

### Weaknesses
1. **Scalability & Computational Complexity Claims:** Section 4.1 constructs a `C × C` similarity matrix and joint reliability matrix before top-`k` sparsification. For ImageNet-1K, this requires ~1M pairwise computations per adaptation step, which contradicts the claim in Sec. 5.2 that "graph updates scale linearly with class count C." The quadratic bottleneck is unaddressed and could hinder adoption on larger taxonomies.
2. **Heuristic Reliability Formulation Lacks Theoretical Grounding:** The reliability score `R_j = μ_j · (1 - σ_j / σ_max)` (Eq. 4) is presented as an information-theoretic proxy but relies on ad-hoc clipping and a fixed `σ_max` = 0.5. There is no formal analysis of how variance relates to predictive uncertainty or why this linear scaling outperforms standard entropy, Dirichlet variance, or conformal prediction in this setting.
3. **Limited Evaluation under Non-i.i.d. & Rapid Shifts:** TTA in practice faces correlated, non-stationary streams or sudden domain jumps. The paper assumes a relatively stable statistical stream where sliding-window reliability converges (Sec. 4.2). No experiments evaluate performance under concept drift, class-prior shifts mid-stream, or varying sample arrival rates (e.g., batched vs. online), leaving the temporal assumptions underexplored.
4. **Visualization & Qualitative Analysis Scope:** Figure 4 illustrates graph evolution on a handcrafted 5-class "micro-universe," which does not reflect the complexity or inter-class confusion of ImageNet or cross-domain datasets. Missing qualitative error analysis or failure cases makes it unclear where the reliability gating fails (e.g., semantically adjacent but visually distinct classes).

### Novelty & Significance
**Novelty:** Moderate to High. Graph-based label propagation and prototype updating exist in TTA (e.g., PROGRAM), but SURE's specific integration of temporal confidence stability into edge weighting for VLMs is a well-motivated and distinct contribution. The closed-loop co-evolution of semantic structure and prediction certainty advances beyond static or purely distance-based topologies.
**Significance:** High. Reliable, efficient TTA for VLMs is critical for real-world deployment. By decoupling adaptation from gradient descent and mitigating pseudo-label confirmation bias through structured regularization, SURE offers a practical, drop-in improvement with strong empirical backing.
**Clarity & Reproducibility:** High. The methodology is clearly structured, equations and algorithmic steps are explicit, and hyperparameters/datasets are fully specified. Despite minor parser artifacts, the technical narrative is easy to follow.

### Suggestions for Improvement
1. **Clarify & Optimize Graph Complexity:** Provide a precise complexity analysis and propose an efficient approximation for large `C` (e.g., ANN-based neighbor retrieval, class-hierarchy pruning, or sparse updates only for activated classes) to reconcile the quadratic similarity computation with the stated linear scaling.
2. **Benchmark Alternative Uncertainty Metrics:** Add an ablation replacing `R_j` with standard predictive entropy, Monte Carlo dropout variance, or conformal prediction intervals. This would strengthen the justification for the mean/variance heuristic and clarify whether gains stem from the metric itself or the graph propagation mechanism.
3. **Test Non-i.i.d. & Sequential Shift Scenarios:** Evaluate SURE under simulated domain-switching streams (e.g., alternating ImageNet-C → ImageNet-Sketch sequences or varying class priors) to assess adaptation latency and recovery. Report how window size `L` and threshold `θ` behave under drift.
4. **Expand Qualitative Analysis & Failure Modes:** Include confusion matrices, attention/activation visualizations, or retrieval neighbors for misclassified samples on full-scale datasets. Explicitly discuss scenarios where reliability gating under-corrects (e.g., novel classes absent in source prototypes) to provide practical deployment guidance.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Cold-Start Sensitivity Analysis:** The reliability initialization ($\mu=1, \sigma=0$) assumes perfect initial confidence. Vary this initialization to prove the method does not lock into early pseudo-label noise, as early errors would otherwise be treated as highly reliable.
2. **Sequential Domain Shift Benchmark:** Evaluate on a sequential shift protocol (e.g., ImageNet $\to$ Sketch $\to$ Real) to verify the sliding window ($L$) does not cause hysteresis or lag when the distribution changes abruptly.
3. **Fine-Grained Text Ambiguity Test:** Test on a dataset with high semantic overlap (e.g., CUB-200) where text prototypes are nearly identical. This validates whether the graph propagates errors between semantically similar classes when visual features are ambiguous.
4. **Memory Overhead Quantification:** Table 3 reports latency but ignores VRAM usage. Quantify the memory cost of storing reliability buffers ($C \times L$) for large vocabularies to verify the claimed deployment feasibility.

### Deeper Analysis Needed (top 3-5 only)
1. **Label Correction Quantification:** Report the percentage of pseudo-labels flipped by the graph and the accuracy of those flips. Without this, the core claim of "error suppression" versus "confidence smoothing" is unsubstantiated.
2. **Reliability-Accuracy Correlation:** Plot per-class reliability scores ($R_j$) against actual per-class accuracy. If $R_j$ remains high for low-accuracy classes, the uncertainty estimation is invalid and undermines the regularization logic.
3. **Text Encoder Dependency:** Ablate the text encoder quality (e.g., use random text embeddings or a weaker encoder). If performance drops significantly, the method relies too heavily on pre-trained semantic priors rather than robust test-time adaptation.

### Visualizations & Case Studies
1. **Full-Scale Adjacency Heatmap:** Figure 4 uses only 5 classes. Provide a heatmap for the full 1000-class ImageNet adjacency to demonstrate that sparsity and meaningful structure hold at scale.
2. **Failure Case Gallery:** Show specific samples where graph propagation *incorrectly* overrides a correct CLIP prediction. This exposes the failure mode of semantic over-smoothing which is currently hidden.
3. **Reliability Score Trajectory:** Plot the evolution of $R_j$ for a specific class over the test stream. This reveals whether the reliability metric reacts dynamically to shifts or remains stagnant due to the buffer.

### Obvious Next Steps
1. **Theoretical Convergence Bound:** Provide a theoretical analysis of how the graph propagation affects the entropy minimization objective to justify the regularization principle mathematically.
2. **Continuous Corruption Benchmark:** Evaluate on ImageNet-C (sequential corruptions) instead of static shifts. This better tests the "dynamic" adaptation claim of the PRG under continuous distribution drift.
3. **Buffer Size vs. Adaptation Speed Trade-off:** Explicitly analyze the trade-off between window size $L$ and adaptation speed to sudden distribution changes to guide hyperparameter selection for real-world deployment.

# Final Consolidated Review
## Summary

SURE introduces a Prototype-Reliability Graph (PRG) for test-time adaptation of vision-language models. The PRG dynamically combines semantic affinity (from textual prototypes) with temporal reliability (from pseudo-label confidence statistics) to regularize predictions via graph-based propagation. This structured approach aims to suppress error amplification from noisy pseudo-labels—a known failure mode in entropy-based and prototype-based TTA methods.

## Strengths

- **Principled motivation and clean formulation:** The paper correctly identifies a genuine weakness in prior TTA methods: treating pseudo-labels independently ignores both inter-class semantic structure and temporal reliability. The proposed solution—gating graph edges by joint reliability scores derived from sliding-window confidence statistics—is well-motivated and coherently integrated into a closed-loop adaptation mechanism.

- **Strong empirical results across diverse benchmarks:** SURE achieves state-of-the-art average accuracy on natural distribution shifts (66.23% on ViT-B, +5.03% over CLIP baseline) and cross-dataset generalization (70.04% average). The gains are consistent across 15 datasets with two backbones, with particularly notable improvements on ImageNet-Sketch (+4.64% over CLIP-ViT-B) and fine-grained domains like Flowers102.

- **Comprehensive ablation with clear component contributions:** Table 4 cleanly isolates the contribution of each component. The "+Graph w/o Rel" variant actually hurts on ImageNet-A (-0.24%), validating the authors' claim that unweighted graph smoothing is risky; the reliability mechanism (+1.24%) and logit propagation (+1.05%) then recover and exceed baseline performance. This demonstrates genuine synergy rather than additive improvements.

- **Efficiency competitive with gradient-free methods:** SURE operates at 0.067s/sample, ~10× faster than TPT (0.706s) and comparable to BCA (0.023s), while achieving higher accuracy than both. This is achieved without multiple augmented views or gradient computation.

- **Calibration analysis and stability verification (Appendix):** Tables 7–8 report standard deviations across three random seeds (all ≤0.28%), and Table 10 shows ECE improvements over naive prototype updating (7.48 vs. 11.23), demonstrating that PRG reduces overconfidence.

## Weaknesses

- **Missing the most directly comparable baseline:** PROGRAM (Sun et al., 2024) is discussed in Related Work as "sharing the graph-based spirit" but explicitly differentiated by its static topology versus SURE's reliability-driven dynamic edges. Yet PROGRAM is never evaluated as a baseline. This is a critical omission—without comparing to PROGRAM, readers cannot verify whether SURE's reliability-weighted edges provide actual improvements over simpler graph-based TTA. The paper must include this comparison to substantiate its novelty claim.

- **Numerical errors and inconsistencies in experimental tables:** Table 3 reports ∆ Gain = +7.12% for SURE, but the actual improvement over CLIP-ViT-B (66.23% – 61.20%) is +5.03%. This is a factual error. Additionally, MTA's accuracy differs between Table 1 (64.06%) and Table 3 (63.16%) with no explanation, and Table 2 contains duplicate "Zero" vs. "ZERO" rows with different values (e.g., SUN397: 66.90 vs. 67.63). These inconsistencies undermine confidence in the reported results.

- **Hyperparameter discrepancy between main experiments and ablation:** The Implementation Details state θ = 0.3, but Figure 3 shows θ = 0.4 as optimal with no clarification. The paper must state which value was used for main results and justify the choice.

- **Scalability concern unaddressed:** Section 4.1 constructs a C × C similarity matrix before top-k sparsification, requiring ~10⁶ pairwise computations for ImageNet-1K. Yet Section 5.2 claims "graph updates scale linearly with class count C." This contradiction is not resolved—either the complexity claim is incorrect, or an unstated approximation is used.

- **Prototype initialization requires clarification:** The paper states "each class prototype is initialized with N_i^proto = 30000 confident samples" following Zhou et al. (2025). While BCA uses this initialization, it is unclear whether all baselines (TPT, TDA, ZERO, etc.) receive the same warm-start. If SURE benefits from initialization that competitors lack, comparisons would be unfair. The paper must clarify what data/assumptions this requires and ensure parity across methods.

- **Limited evaluation under realistic streaming conditions:** TTA methods face correlated, non-i.i.d. test streams and sudden domain shifts. The paper evaluates only static benchmarks, not sequential domain-switching protocols (e.g., ImageNet → Sketch → corruption) or varying class-prior streams. The sliding-window reliability mechanism (L = 5) could exhibit hysteresis or lag under such conditions—this is untested.

- **Theoretical grounding of reliability formulation:** The reliability score R_j = μ_j · (1 – σ_j/σ_max) uses σ_max = 0.5 as a fixed constant. No ablation justifies this value, and no analysis connects variance to predictive uncertainty in a principled way (entropy, conformal intervals, or Dirichlet variance are common alternatives). The formulation works empirically but lacks formal motivation.

## Nice-to-Haves

- **Reliability-accuracy correlation analysis:** Plot per-class reliability scores R_j against actual per-class accuracy. If R_j remains high for low-accuracy classes, the regularization premise would be undermined.

- **Label correction quantification:** Report what percentage of pseudo-labels are flipped by graph propagation and the accuracy of those flips—this would directly validate the "error suppression" mechanism.

- **Sequential shift evaluation:** Test on continuous corruption benchmarks (ImageNet-C) or alternating domain sequences to verify adaptation dynamics under realistic streaming conditions.

- **Analysis of rare-class behavior:** Examine how SURE performs on classes that appear rarely in the test stream, where reliability buffers Q_i remain sparse or uninformative.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No variance reported across seeds"** — False. Tables 7 and 8 in the appendix report standard deviations across three random seeds, all ≤0.28% for SURE.

- **"Algorithm 1 is fragmented across pages"** — This is a PDF parsing artifact visible in the extracted text, not a paper presentation issue.

- **"SURE is worse than DPE on ImageNet-A (ViT-B)"** — Incorrect. Table 1 shows SURE (61.45%) > DPE (59.63%) on ImageNet-A with ViT-B. The valid comparison is that ZERO (62.75%) outperforms SURE on this specific benchmark.

- **"Contribution 3 scope is modest (only RN50 and ViT-B)"** — While ViT-L/14 would strengthen the evaluation, testing two architectures across 15 benchmarks is reasonable scope for an empirical contribution. This becomes a nice-to-have rather than a core weakness.

## Novel Insights

The ablation in Table 4 reveals a subtle but important finding: graph-based propagation *without* reliability weighting (row "+Graph w/o Rel") slightly hurts on ImageNet-A compared to prototype-only updating. This confirms that naive semantic smoothing can amplify noise when class relationships are distorted by distribution shift. The reliability mechanism is not merely an incremental improvement—it is essential for making graph-based regularization safe under shift. This contrasts with assumptions in prior graph-based TTA (e.g., PROGRAM) that static semantic similarity suffices for propagation.

## Suggestions

1. **Include PROGRAM (Sun et al., 2024) as a baseline** with re-implementation or official code. This is essential for novelty claims.

2. **Correct Table 3's ∆ Gain column** (+5.03% not +7.12%) and resolve the MTA inconsistency between tables. Clarify the duplicate "Zero" entries in Table 2.

3. **Clarify prototype initialization parity:** State explicitly which baselines use N_i^proto = 30000 initialization and whether this requires source data access. If some baselines do not receive this warm-start, re-run experiments with parity.

4. **Add ImageNet-C or a sequential shift protocol** to evaluate adaptation dynamics under continuous distribution drift.

5. **Provide reliability-accuracy scatter plots** (per-class R_j vs. accuracy) to validate that the reliability metric correlates with actual prediction quality across classes.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0, 4.0]
Average score: 4.4
Binary outcome: Reject
