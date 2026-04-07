=== CALIBRATION EXAMPLE 32 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title ("DenseFace: Bias Mitigation in Face Recognition via Density-Aware Probabilistic Matching") adequately describes the technical approach. The abstract's core claim — that DenseFace "reduces demographic biases...without compromising accuracy" — is an unusually strong claim that requires careful scrutiny, and the paper does not always deliver sufficient evidence for it in the main text.

The abstract also implies generality across "network architectures, training datasets and loss functions," but the experiments are limited to AdaFace and CosFace variants. The mismatch between this claimed breadth and the narrow experimental scope is a recurring issue throughout the paper.

---

### Introduction & Motivation

The motivation is sound: post-hoc bias mitigation that avoids retraining is practically valuable. The framing is reasonable, and the connection to biometric systems where a single threshold is applied across groups is well-articulated.

However, the introduction conflates several distinct contributions without clearly demarcating them:
1. A new matching procedure (DenseFace proper)
2. A critique of existing bias metrics (Std on RFW)
3. An argument that large-scale unbalanced datasets can match balanced-dataset models

This makes the reader uncertain about what the primary contribution actually is. The contributions list at the end of the introduction is helpful but still somewhat vague on what is genuinely novel versus adopted from prior work (e.g., the MLS matching score of Li et al., 2021 is taken wholesale).

---

### Related Work (Section 2)

The related work is thorough. However, the positioning of DenseFace against Conti et al. (2022) — who also explicitly use vMF distributions for face recognition debiasing — is inadequate. Conti et al. use vMF mixtures to project embeddings into a fairer latent space; DenseFace uses vMF concentration as a density signal for post-hoc matching adjustment. These are different, but the distinction deserves a sharper technical elaboration than the single sentence provided. Crucially, Conti et al. (2022) is never compared against quantitatively using the paper's proposed NIST FPR metric.

Similarly, Linghu et al. (2024) on score normalization for demographic fairness is the closest post-hoc competing method, yet it appears only in the related work and not in the experimental comparison tables.

---

### Method / Approach (Section 3)

**Section 3.1 — Probabilistic Embedding Representation:**
The use of vMF for unit-sphere embeddings is well-motivated and consistent with prior work (Banerjee et al. 2005; Conti et al. 2022; Li et al. 2021). The decision to use the mean embedding per identity as anchor set representatives is reasonable.

The anchor set requirement is a significant practical limitation that is understated: the method requires a large, demographically balanced, labeled reference set (54,000 identities, 8 balanced subgroups defined by race × gender). This requires (a) access to demographic classifiers, (b) a large-scale face dataset like Glint360K, and (c) demographic annotation of that dataset. The paper does not acknowledge this as a limitation.

**Section 3.2 — Local Distortion of Embedding Space:**
This is the most technically problematic section. The motivation is correct: in high dimensions, face embeddings from different identities are nearly orthogonal, causing inter-class cosine similarities to cluster near zero, which makes the vMF concentration κ insensitive across identities. The proposed fix — inserting an angular margin *m* into the cosine computation via Equation (7) — is a heuristic. After applying this margin, κ^(m) is no longer the MLE of a genuine vMF distribution over the anchor neighbors. The statistical validity of the density interpretation is lost. The paper calls this a "local distortion of embedding space," but this is a metaphor rather than a mathematically rigorous concept.

The margin *m* is a hyperparameter, but the paper does not include any sensitivity analysis for *m* in the main text. What is the actual value used? How does bias reduction and accuracy change with *m*?

**Section 3.3 — Density-Aware Probabilistic Matching:**
The matching score in Equation (9) is directly adopted from Li et al. (2021) (SCF / Mutual Likelihood Score). This is explicitly stated, but the degree of methodological borrowing from SCF raises questions about the novelty of this component. The paper's contribution here is substituting κ_i with κ_i^(m) — the density-adjusted concentration parameter. The novelty is real but incremental.

---

### Performance Metrics (Section 4.3)

The critique of the RFW Std metric is interesting and partially compelling. The example in Figure 5 — where CosFace-R50-Glint360K has lower Std (suggesting less bias) but higher Δ in cosine similarities across groups (suggesting more bias) — is a concrete motivating example. However, one example is not a systematic analysis. There could be cases where the two metrics align.

There is a more fundamental issue with the proposed NIST FPR metric: it is explicitly Caucasian-centric. The threshold is calibrated so that the Caucasian FPR = 10^{-3}, and then FPR is measured for other groups at that fixed threshold. The paper does not justify why Caucasian performance should be the reference point. In a fair system, one might instead calibrate to a global threshold or to the lowest-FPR group. This design choice has non-trivial ethical implications that are not discussed.

Furthermore, the paper introduces a new metric and argues for its superiority, but then Table 1 still uses the old Std metric for comparison with prior methods — precisely because prior methods don't report the NIST FPR metric. This creates an unfair comparison: prior debiasing methods are evaluated on a metric the paper itself argues is insufficient, while DenseFace is evaluated on a new metric where no comparison exists.

---

### Experiments & Results (Section 4)

**Table 2 and Table 3 (Main Results):**
The FPR reductions are large and consistently positive across groups and models. This is the paper's strongest evidence.

However, a critical gap: **there is no comparison to any competing post-hoc debiasing method under the proposed NIST FPR metric**. The paper compares only to the cosine similarity baseline. Methods like Terhorst et al. (2020a, 2020b), Dhar et al. (2021), Conti et al. (2022), Kotwal & Marcel (2024), and Linghu et al. (2024) are all post-hoc debiasing methods that could in principle be evaluated under the same protocol. Without these comparisons, it is impossible to assess whether DenseFace represents a meaningful advance over the state of the art in post-hoc debiasing.

**Table 4 (Accuracy Preservation):**
The parser has heavily garbled this table, so it is difficult to evaluate the accuracy-preservation claim precisely. The paper claims "preserving or improving its verification performance" — but there is no statistical significance analysis. Given that DenseFace is adjusting similarity scores, it is possible that some accuracy degradation occurs at thresholds other than the one calibrated for Caucasians.

**Gender Bias:**
Figure 3 shows density distributions for gender groups, motivating that the method should also reduce gender bias. However, the experimental tables (Tables 2, 3, 5) only report racial bias results. Gender bias mitigation is mentioned qualitatively but never demonstrated quantitatively.

**Section 4.5 (Learning-Based Approach):**
DenseFace† (the learning-based variant) consistently outperforms the anchor-based DenseFace (Table 5). This is presented as a "surprising" finding, but no analysis of why this occurs is provided. If a small regression network trained on Glint360K can predict better density estimates, is it simply learning to classify race implicitly and applying a calibration accordingly? This would undermine the theoretical framing of the entire paper. This warrants serious investigation.

**Computational Cost (Section 4.8):**
The analysis is honest — 1.5× slower matching with lookup tables. For production-scale face recognition systems processing millions of pairs, this could be a real concern. The learning-based variant largely solves this, which is a practical advantage.

---

### Writing & Clarity

The paper's structure is disrupted by the PDF extraction artifacts, making some sections (especially the algorithm box and parts of Section 3.2) very hard to follow. Section 3 lacks a coherent narrative; the derivation jumps between equations without adequate prose connecting them. In particular, the transition from Equations (5) to (6) to (7) — the core density computation — is presented mechanically without explaining the geometric intuition clearly. The observation that anchor-set embeddings are nearly orthogonal (all cosines near zero → the sum collapses to √K) deserves a more explicit derivation.

Section 4.6 and 4.7 appear to be out of order relative to Section 4.4 and 4.5. Section 4.8 (Limitations) is placed after the conclusion-adjacent Section 4.7.

---

### Limitations & Broader Impact

The paper's Section 4.8 focuses on computational cost but largely ignores:

1. **Anchor set dependency**: The method requires a large, demographically labeled and balanced anchor set. This is not free — it requires demographic classifiers and curated data. This limits deployability in settings without such resources.

2. **Caucasian-centric calibration**: As noted above, normalizing to Caucasian FPR as the reference group is an ethical choice that deserves explicit discussion.

3. **FNMR impact**: Reducing FPR (false match rate) for non-Caucasian groups while preserving accuracy at a Caucasian-anchored threshold may be shifting burden. Does FNMR (false non-match rate) increase for any group? This is not examined.

4. **Beyond race and binary gender**: The method is evaluated only on race (4 groups) and binary gender. No discussion of intersectionality or other demographic attributes.

5. **Implicit race encoding**: The finding in Figure 6 that nearest neighbors are predominantly same-race effectively confirms that the anchor-based density is implicitly encoding race. The ethical implications of building a debiasing system around implicit racial clustering are not discussed.

---

### Overall Assessment

DenseFace presents a technically coherent idea: exploiting the correlation between vMF inter-class embedding density and demographic bias to build a density-aware matching score. The empirical results (Tables 2, 3) show substantial FPR reductions relative to cosine matching across multiple models, which is encouraging. The observation that large-scale unbalanced training datasets can match or exceed balanced-dataset models on bias metrics is a useful finding.

However, the paper has several significant weaknesses that fall short of ICLR's acceptance bar. Most critically: **(1)** there is no comparison to competing post-hoc debiasing methods (Conti et al. 2022, Linghu et al. 2024, Kotwal & Marcel 2024) under the paper's own proposed metric — so the claimed advancement over the state of the art is unverified; **(2)** the "local distortion" of embedding space via angular margin m is a heuristic with no statistical justification, and its sensitivity is not analyzed; **(3)** the learning-based DenseFace† outperforms the theoretically-motivated anchor-based version without explanation, raising the possibility that the regression network is simply learning an implicit race calibration that would undermine the method's theoretical framing; **(4)** the proposed NIST FPR metric is Caucasian-centric by design, a choice with ethical implications that goes unexamined; and **(5)** gender bias mitigation is motivated but never experimentally demonstrated. The paper should be substantially revised to address the comparison gap, analyze the margin hyperparameter, investigate the learning-based variant more deeply, and provide a more honest discussion of the ethical assumptions embedded in the evaluation protocol.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **DenseFace**, a post-training calibration method that mitigates demographic bias in face recognition without retraining pretrained models. It models face embeddings using von Mises-Fisher distributions and adjusts matching scores based on the local density of the embedding space estimated via a balanced anchor set. Extensive experiments demonstrate that DenseFace reduces racial bias metrics (specifically False Positive Rates) across various SOTA architectures while preserving verification accuracy.

### Strengths
1.  **Practical Post-Training Calibration:** The method operates without retraining the backbone face recognition model, making it deployable on existing systems. The text explicitly states, "DenseFace requires no retraining of existing face recognition models," offering significant logistical advantages over loss-function modifications or fine-tuning (Section 1 & 2.3).
2.  **Insightful Use of Embedding Density:** The core technical contribution involves modeling embeddings probabilistically and linking local density (estimated via K-nearest neighbors in an anchor set) to demographic bias. The observation that non-Caucasian groups often occupy higher-density regions in the embedding space is supported by data in Figure 3 and Table 2.
3.  **Robust Evaluation Metrics:** The authors critique the standard "Standard Deviation of accuracy" used in RFW and advocate for the NIST protocol (FPR at fixed thresholds), arguing it better reflects production constraints where a single threshold is enforced across groups. This is supported by Figure 5, showing inconsistencies in RFW Std metrics.
4.  **Comprehensive Empirical Validation:** The method is tested across multiple architectures (ArcFace, AdaFace, CosFace), large-scale datasets (WebFace4M/12M, MS1MV2), and benchmarks (RFW, RB-WebFace). Consistent bias reduction is reported in Table 2, 3, and 5, often improving FPR ratios compared to the cosine baseline.

### Weaknesses
1.  **Inference Computational Cost and Dependency:** The non-learning-based version requires searching a large anchor set ($K=128$ neighbors from $\sim$54k identities) for every verification query. Section 4.8 admits a matching latency increase of 1.5x to 2.5x compared to cosine similarity. This may hinder real-time applications compared to the marginal gains in specific bias metrics.
2.  **Anchor Set Requirements:** The method relies on an external, balanced "image anchor set" to estimate population density (Section 3.1 & 4.2). Constructing this requires access to 54,000 balanced identities. In many deployment scenarios (e.g., forensic or specific customer bases), such a general population anchor set may not be available or representative of the specific demographic distribution, potentially limiting generalizability.
3.  **Learning-Based Variant Complexity:** While Section 4.5 proposes a learning-based regression network to replace anchor search, this effectively introduces a new training phase (minimizing MSE loss, 100 epochs). Although the *backbone* is not retrained, the dependency on training an auxiliary network contradicts the fully "zero-shot" appeal suggested in the abstract.
4.  **Limited Ablation on Key Hyperparameters:** The paper discusses the importance of the angular margin $m$ (Equation 7) but lacks detailed ablation studies on how sensitive the bias reduction is to $\kappa$ (density), $K$ (neighbor count), or the specific composition of the anchor set. Table 4 and 5 show best results but do not explain the optimization landscape.
5.  **Artifacted Presentation:** The provided text contains significant OCR artifacts (e.g., broken equations, garbled text around Equation 1-9). While IRL must treat these as parser issues, this severely impacts the reproducibility and clarity of the mathematical derivation for potential readers.

### Novelty & Significance
*   **Novelty:** The application of von Mises-Fisher distributions for density-aware probabilistic matching in the context of post-hoc bias mitigation is a notable contribution. While probabilistic embeddings exist (e.g., SCF), using them specifically for demographic calibration via local density correction is distinct from existing score normalization techniques (e.g., Terhorst et al.).
*   **Significance:** Demographic bias in biometrics is a high-stakes societal issue. A method that can significantly reduce bias (up to >50% FPR reduction on specific groups in Table 3) without retraining valuable backbone models holds high practical significance for the industry. The advocacy for NIST-based metrics also adds value to the community's benchmarking standards.
*   **ICLR Fit:** The paper fits ICLR standards regarding technical depth in probabilistic modeling and societal impact. However, the computational overhead and anchor dependencies must be carefully weighed against the novelty of the calibration strategy.

### Suggestions for Improvement
1.  **Optimize Inference Latency:** The authors should provide more details on how the $O(N)$ neighbor search scales or suggest approximate nearest neighbor (ANN) techniques to mitigate the 1.5x latency penalty mentioned in Section 4.8.
2.  **Clarify Anchor Dependency:** A sensitivity analysis should be included to show how performance changes if the anchor set composition varies (e.g., different balances of races). This will help practitioners understand the risk of deploying without a perfectly balanced public anchor set.
3.  **Expand Metric Discussion:** While the critique of RFW Std is valid, a more rigorous theoretical justification in Section 4.3 for why FPR at a fixed threshold is the *only* appropriate metric is needed to fully sway the community towards this evaluation change.
4.  **Improve Mathematical Presentation:** Ensure clean equations in the final submission. The current derivation (especially Eq 1-9) is difficult to parse due to formatting, which hurts the scientific reproducibility.
5.  **Ablation Studies:** Include ablation experiments varying the margin $m$ and the anchor set size $K$ to demonstrate robustness. Currently, the choice of $K=128$ and specific margin values appears fixed without justification of why other values were rejected.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Cross-racial matching evaluation is absent** — The paper criticizes RFW for not accounting for cross-racial matching but never provides cross-racial verification results. Without this, the claim that DenseFace addresses real-world scenarios is unsupported.

2. **No direct comparison to post-training calibration baselines** — Methods like Terhörst et al. (2020a;b), Conti et al. (2022), and Linghu et al. (2024) are mentioned but not compared under identical experimental conditions. This makes it impossible to verify if DenseFace actually outperforms existing calibration approaches.

3. **Gender bias results are incomplete** — Despite claiming to mitigate multiple demographic attributes, gender bias evaluation is minimal and not prominently reported in main tables. The core claim about broad demographic fairness lacks supporting evidence.

4. **Anchor set composition ablation missing** — The method relies heavily on a balanced anchor set, but there's no analysis of sensitivity to anchor set size, balance ratio, or source dataset. This is critical for understanding practical deployment requirements.

5. **No evaluation across multiple operational thresholds** — Results are reported at a single FPR threshold (10^-3), but real systems operate at various points. Without showing performance across the full ROC curve, practical utility remains unclear.

### Deeper Analysis Needed (top 3-5 only)
1. **Mechanism explaining density-demographic correlation is unexplored** — The paper observes that embedding density correlates with demographic attributes but provides no theoretical or empirical explanation for why this relationship exists. This undermines confidence in the method's generalizability.

2. **Failure mode analysis is absent** — There's no examination of cases where DenseFace fails or potentially worsens bias for specific subgroups. Understanding limitations is essential for trustworthy deployment.

3. **NIST protocol superiority claim is underjustified** — The argument that NIST protocol is better than RFW/accuracy-std metrics needs more rigorous justification beyond assertion, especially since the paper still reports Std in Table 1.

4. **No intersectional bias analysis** — Race and gender intersections (e.g., Black women) are known to have compounded bias, but this isn't examined despite gender being mentioned as a factor.

5. **Sensitivity to anchor set quality unanalyzed** — How does performance degrade if the anchor set itself contains bias or is imperfectly balanced? This directly impacts real-world applicability.

### Visualizations & Case Studies
1. **Embedding space visualization before/after DenseFace** — Show how DenseFace actually transforms the embedding geometry (e.g., t-SNE plots), not just final metrics. This would reveal whether the method genuinely expands/contracts space as claimed.

2. **Failure case examples with matched pairs** — Display specific image pairs where DenseFace corrects errors vs. where it introduces new errors. This would expose whether the method actually works vs. fails on hard cases.

3. **Intersectional density distributions** — Visualize density across race-gender combinations, not just race alone (Figure 3 only shows race and gender separately). This would reveal whether bias mitigation transfers across intersections.

### Obvious Next Steps
1. **Include gender bias results prominently in main tables** — This should be in primary results alongside race, not briefly mentioned. ICLR expects comprehensive fairness evaluation.

2. **Test on additional benchmarks beyond RFW and RB-WebFace** — Evaluate on datasets like IJB-C or MORPH for age bias and more diverse scenarios to support generalizability claims.

3. **Provide deployment-ready computational analysis** — More thorough latency and memory profiling across different hardware configurations, since the paper claims practical applicability but matching is 1.5× slower.

# Final Consolidated Review
## Summary

DenseFace proposes a post-training calibration method to mitigate demographic bias in face recognition without retraining pretrained models. The method models face embeddings as von Mises-Fisher distributions, estimates local embedding density using a balanced anchor set, and adjusts matching scores by decreasing similarity in dense regions and increasing it in sparse regions. Experiments on RFW and RB-WebFace benchmarks demonstrate reduced false positive rate disparities across racial groups while maintaining verification accuracy.

## Strengths

- **Practical post-hoc calibration approach**: DenseFace operates on pre-trained models without requiring retraining, making it deployable on existing systems. This addresses a real practical need given the cost of training large face recognition models.

- **Novel use of embedding density for bias correction**: The paper establishes a correlation between embedding density and demographic attributes (Figure 3 shows non-Caucasian groups occupy denser regions), then leverages this observation to build a density-aware matching score that demonstrably reduces FPR disparities (Tables 2-3 show consistent bias reduction across multiple models and datasets).

- **Critical re-evaluation of bias metrics**: The paper provides a concrete example (Figure 5, Table 1) where CosFace-R50-Glint360K has lower standard deviation than AdaFace-R50-WebFace4M, yet higher cross-group similarity differences—showing that Std can be misleading. The adoption of the NIST FPR protocol better reflects production scenarios where a single threshold must be applied globally.

- **Learning-based variant offers practical efficiency**: DenseFace† uses a lightweight regression network that achieves even better bias reduction than the anchor-based version with only marginal computational overhead (+0.2% parameters, +1.75% latency per Section 4.8), addressing real-world deployment concerns.

- **Consistent results across architectures and datasets**: The method is validated on AdaFace and CosFace trained on MS1MV2, Glint360K, WebFace4M, and WebFace12M, demonstrating generalizability across different training regimes and model capacities.

## Weaknesses

- **No direct comparison to competing post-hoc debiasing methods**: The paper mentions several related approaches (Terhörst et al. 2020a,b; Dhar et al. 2021; Conti et al. 2022; Kotwal & Marcel 2024; Linghu et al. 2024) in related work but provides no experimental comparison under the proposed NIST protocol. Without benchmarking against these methods, it is impossible to assess whether DenseFace represents an actual advance over the state of the art in post-hoc debiasing. This gap is substantial given the paper's positioning against this specific line of work.

- **Gender bias evaluation is incomplete**: Figure 3 motivates that gender also correlates with embedding density, but the experimental results (Tables 2, 3, 5) exclusively report racial subgroups. The claim that DenseFace mitigates bias across "gender, race and other attributes" (Introduction) lacks empirical verification for gender in the main results.

- **Learning-based variant's success lacks explanation**: DenseFace† (learning-based) consistently outperforms the theoretically-motivated anchor-based version (Table 5), which the paper describes as "surprising." This raises an important question: if a simple regression network can predict better density estimates, is it implicitly learning demographic calibration that undermines the theoretical framing? The paper should investigate this to validate that the learning-based variant is learning density rather than shortcut demographic features.

- **Margin hyperparameter lacks sensitivity analysis**: The angular margin m introduced in Equation (7) to address near-orthogonal embeddings is a core design choice, yet the paper provides no analysis of how bias reduction varies with different m values, nor justification for the specific value used.

- **Caucasian-centric calibration choice is not discussed**: The proposed NIST protocol calibrates thresholds so that Caucasian FPR = 10⁻³ and measures other groups relative to this. While this follows the NIST FRVT report, the paper does not discuss why Caucasian performance should be the reference point, or the ethical implications of this design choice. An alternative could be calibrating to the lowest-FPR group or a global threshold.

- **Anchor set dependency requires substantial infrastructure**: The method requires a demographically balanced anchor set of 54,000 identities with race and gender labels. While the paper discusses this in Section 4.2, the practical difficulty of obtaining such a labeled, balanced dataset in deployment scenarios deserves more prominent acknowledgment as a limitation.

## Nice-to-Haves

- Ablation studies varying the anchor set composition (size, balance ratio, source dataset) to understand sensitivity to these design choices

- Evaluation at multiple FPR thresholds beyond the single 10⁻³ point to show performance across the full operating range

- Intersectional analysis examining bias for race-gender combinations (e.g., Black women) known to experience compounded disparities

- Failure case analysis showing where DenseFace does not correct or potentially worsens predictions

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Learning-based variant contradicts the 'no retraining' claim"**: This criticism misreads the paper. The abstract and introduction clearly state "no retraining of existing face recognition models"—the backbone is frozen. The small regression network for κ prediction is a separate auxiliary module, and the paper is transparent about this training requirement.

- **"No cross-racial matching evaluation"**: The paper explicitly states it considers "all possible pairs of images in RFW" (approximately 14k positive and 50M negative pairs per group), which inherently includes cross-racial pairs. The NIST protocol evaluation accounts for this.

- **"Mechanism explaining density-demographic correlation is unexplored"**: While deeper theoretical analysis would strengthen the paper, this is an empirical contribution. The correlation is observed and exploited—explaining its origins is beyond the stated scope.

- **"OCR artifacts in equations"**: The review instructions explicitly note these are parser issues, not paper problems. The actual submission would have properly formatted equations.

- **"Generic strengths about well-written paper"**: Removed as instructed—these would apply to any paper and are not specific contributions.

## Novel Insights

The paper makes a valuable empirical observation that challenges conventional wisdom: models trained on large-scale unbalanced datasets (WebFace4M/12M) can exhibit lower or comparable bias than models trained on balanced datasets (BUPT-Balancedface), as shown in Table 1. This finding aligns with Gwilliam et al. (2021) and suggests that the community's focus on balanced training data may be partially misplaced—scale may compensate for imbalance. Additionally, the density-demographic correlation (Figure 3) provides a new lens for understanding why certain demographic groups experience higher false positive rates: the embedding space geometry itself encodes bias, not just the decision boundary.

## Suggestions

1. **Add comparison to post-hoc calibration baselines**: Implement and compare DenseFace against at least one competing post-training method (e.g., Terhörst et al.'s fair score normalization or Conti et al.'s vMF projection) under the NIST protocol to validate the claimed advancement.

2. **Include gender bias results in main tables**: Either add gender-specific FPR results or explicitly scope the contribution to racial bias if gender mitigation is not substantively evaluated.

3. **Provide sensitivity analysis for margin m**: Report how bias reduction and accuracy change across a range of m values to establish robustness of the design choice.

4. **Investigate DenseFace†'s learning behavior**: Analyze whether the regression network learns meaningful density estimates versus implicit demographic shortcuts by examining learned representations or testing on out-of-distribution demographics.

# Actual Human Scores
Individual reviewer scores: [2.0, 6.0, 2.0]
Average score: 3.3
Binary outcome: Reject
