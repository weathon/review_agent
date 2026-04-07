=== CALIBRATION EXAMPLE 40 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Evil in the Pairing Assumption" is provocative but somewhat disconnected from the paper's actual contribution — the subtitle "Multimodal Attribution via Adaptive Information Bottleneck" carries the real content. The framing of "evil" is never given a precise technical meaning in the text. The abstract claims "extensive experiments on large-scale image-text datasets," which is somewhat overstated given that only 2,000 samples are drawn per run from CC3M and LAION-400M (Appendix A.2), a small fraction even of the training-size splits often used in evaluation.

---

### Introduction & Motivation

The motivating observation — that existing IB-based attribution methods (M2IB, NIB) produce forced, misleading explanations on misaligned image-text pairs — is genuine and well-illustrated by Figure 1. However, the paper repeatedly uses the term "overfitting" to describe what happens when the IB objective is applied to mismatched pairs. This is conceptually imprecise. Overfitting in the standard ML sense (training vs. generalization gap) does not apply here; the actual phenomenon is closer to "hallucinated alignment" or "spurious compression." The paper would benefit from a more careful terminology.

The three stated contributions are reasonable, but Contribution 2 ("formal analysis of functional properties") oversells what are largely straightforward asymptotic and algebraic results (detailed below).

---

### Related Work

The related work is adequately broad. One citation concern: "(Hossain et al.)" in Section 2.2 appears without a year or venue, suggesting an incomplete reference — if this work is used to substantiate limitations of prior approaches, the missing citation undermines the argument. The NIB baseline (Zhu et al., 2025) is only an arXiv preprint; building the paper's primary comparison around a concurrent, non-peer-reviewed work warrants acknowledgment.

---

### Method (Section 4) — Core Technical Concerns

**The f(X,Y) inconsistency — potentially fundamental:** This is the most serious issue in the paper. The theoretical framework (Definition 1, Theorems 1–3) requires f(X,Y) to be a *relevance function* where **large f means high semantic similarity** between X and Y. Section 4.1 explicitly states: "Large f emphasizes sufficiency by scaling up I(Z;Y), while small f emphasizes minimality via g(f)" — i.e., strongly aligned pairs should have large f and drive toward prediction, while mismatched pairs should have small f and drive toward compression.

However, Section 5.2 states: *"For the function f(X,Y), we choose the L2 distance by default."* L2 distance is **large when vectors are distant/dissimilar** and small when similar. Under this choice, mismatched pairs would have *large* f — which drives toward sufficiency (more prediction), which is precisely the wrong behavior. Well-matched pairs would have *small* f — driving toward compression, again the wrong direction.

The appendix (Section B.5, Step (i)) defines an "inverse-distance-based" f = 1/(d(x,y)+ε), which has the correct directionality. But Figure 4 (Appendix D) states that "matched image-caption pairs are predominantly associated with higher f values" and "mismatched pairs are more concentrated in the lower range of f values" — this is consistent with using inverse-distance, but inconsistent with using raw L2 as stated in Section 5.2. This inconsistency is never resolved. If the empirical f is indeed raw L2 distance (large for dissimilar pairs), the adaptive mechanism is inverted relative to the stated theory, and the entire argument collapses.

**Theorems 1–3 — limited novelty and depth:** 
- Theorem 1 (sufficiency at high relevance) establishes that as f → ∞, L_AdaIB ∼ f·I(Z;Y). This is a direct asymptotic calculation, essentially following because g(f)/f → 0 for any bounded g. It is a trivial consequence of the definition.
- Theorem 2 (minimality at low relevance) is more substantive but relies on lim_{u→0+} g(u)/u = +∞, which is a strong assumption not discussed in the context of the practical choices for g.
- Theorem 3 (adaptive trade-off) is explicitly just an algebraic rearrangement combined with Lemma B.1. It is not a theorem in any meaningful sense.
- Theorem 4 (existence of stationary points via Brouwer's fixed-point theorem) invokes the assumption that parameter space Ω is compact and convex (A2, A3), which is unrealistic for neural network training in practice. This result would apply to virtually any smooth bounded loss function and is not specific to AdaIB.

**Learning rate of 1.0 (Section 5.2):** Adam with a learning rate of 1.0 is extremely unusual and orders of magnitude higher than standard practice. The paper provides no justification. This raises reproducibility concerns.

**Variational objective (Eq. 3/4):** The derivation is standard and correct, following directly from IBA (Schulz et al., 2020). The paper could more explicitly acknowledge how much of the training procedure is inherited from IBA vs. novel.

---

### Experiments & Results

**Table 1 is incomplete.** The main results table only shows three rows of data (cc3M-I Drop, cc3M-I Incr., cc3M-T Drop) — Flickr8k and LAION results, as well as the image-side text attribution rows, appear to be missing or the table is truncated. This makes it impossible to assess overall performance from the main paper body.

**AdaIB underperforms M2IB on cc3M-T Drop:** Table 1 shows M2IB at 0.90±0.08 vs. AdaIB at 1.07±0.11 for text Drop on CC3M. The paper does not comment on this. The claimed performance advantage is not uniform, and the differences on many metrics appear marginal relative to reported standard deviations.

**The central claim is tested only in appendix, and that table is unreadable.** The paper's core contribution is robustness under misalignment. Table 5 (Appendix C), which partitions data into noisy/borderline/clean subsets, is the key experimental result supporting this claim. In the parsed document, this table is entirely garbled (row headers and values are missing). Even setting aside parsing artifacts, the main paper (Section 6) simply says "our method maintains leading performance... see Appendix C" without presenting actual numbers in the main body. This is inadequate: the primary empirical claim should be substantiated in the main text.

**ROAR metric adaptation is not adequately justified.** The paper "adapts" the ROAR benchmark (Hooker et al., 2019) by skipping the retraining step and instead measuring zero-shot retrieval degradation. The original ROAR specifically requires retraining to avoid confounds from distribution shift; by skipping retraining, the authors measure something different. The validity of this adaptation for the intended purpose (faithfulness of attribution) is not rigorously argued, and the name "ROAR" is potentially misleading.

**Computational cost (Table 4):** AdaIB achieves 2.27 FPS vs. NIB's 12.5 FPS — a 5.5× slowdown compared to the strongest information-bottleneck baseline. The paper describes this as "no significant increase in computational load" by comparing only to M2IB (2.47 FPS), ignoring the NIB comparison. Given NIB is a primary baseline, this framing is misleading.

**Misalignment experiment design:** The misalignment experiments in Appendix C create mismatched pairs by randomly shuffling captions. This is a very coarse, artificial manipulation. Real-world noisy web data (CC3M, LAION) contains *partially* aligned pairs, subtle mismatches, and domain shifts — not purely random shuffles. The gap between artificial and natural misalignment is not analyzed.

---

### Ablation Studies (Appendix E)

The ablation studies on f and g are appreciated. However, the choice of Flickr8k as the validation set for architecture ablation (Table 7) is odd, given that Flickr8k is described as a "clean" curated dataset and the method's advantage is claimed on noisy data. Ablation performance on the noisy settings (CC3M, LAION) would be more informative.

The finding in Appendix D — that the same f value can correspond to very different g values depending on data — is highlighted as a strength ("AdaIB learns more nuanced, context-dependent characteristics"). However, if g is not monotonically determined by f, then the theoretical properties in Sections 4.1–4.3 (which assume g is non-increasing in f) may not hold in practice, since g is freely parameterized. This potential gap between theory and practice is not discussed.

---

### Writing & Clarity

Section 6 ("Analysis of AdaIB") functions more as an extended abstract of the appendix than as genuine analysis — almost every statement ends with "see Appendix X." This structural choice pushes the most important empirical evidence (robustness to misalignment) entirely to supplementary material, making the main paper incomplete on its own. The paper should be restructured to surface the misalignment analysis results in the main body.

---

### Limitations & Broader Impact

The acknowledged limitation — that AdaIB is not evaluated on subtle mismatches like sarcasm or metaphor — is appropriate. Several important limitations are not discussed:

1. The method inherits CLIP's biases as the source of f(X,Y): if CLIP incorrectly judges a misaligned pair as aligned, the adaptive mechanism provides no correction.
2. The choice of CLIP embedding space for f(X,Y) means the method is not model-agnostic and cannot easily extend to VLMs with different embedding geometries.
3. The paper does not discuss whether the "explain by suppression" behavior on mismatched pairs (Figure 1, last column) is actually useful or could cause silent failures in real attribution pipelines.

---

### Overall Assessment

AdaIB addresses a genuine and underexplored problem — attribution under cross-modal misalignment — and the core intuition (adaptively weighting the IB objective per sample) is sound. However, the paper has a potentially fundamental flaw: the empirical choice of raw L2 distance as f(X,Y) appears inverted relative to the theoretical requirement that f be large for similar pairs, and the paper never cleanly reconciles this. The main results table is incomplete, and the central empirical claim about misalignment robustness is relegated to an appendix with a garbled table. The theoretical contributions are largely asymptotic/algebraic rather than deep, and Theorem 4's Brouwer-based stationary point guarantee relies on unrealistic compactness assumptions. The adapted ROAR metric departs from the established benchmark without sufficient justification, and the significant FPS disadvantage over NIB is understated. In its current state, the paper requires substantial revision — at minimum, a clear and consistent definition of f(X,Y) throughout theory and practice, and a restructuring that places the primary misalignment experiment in the main body with complete, readable results.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes AdaIB, an adaptive information bottleneck framework for multimodal attribution in vision-language models that relaxes the strict semantic alignment assumption required by existing methods like M2IB. By introducing a learnable relevance function and an adaptive compression term, AdaIB dynamically balances the sufficiency-minimality trade-off based on sample-specific alignment. Extensive experiments across large-scale datasets demonstrate improved robustness on noisy/misaligned pairs compared to state-of-the-art baselines.

### Strengths
1.  **Clear Motivation and Real-World Relevance:** The identification of the "pairing assumption" flaw in existing attribution methods for open-world settings is a critical and well-motivated problem. Figure 1 effectively illustrates how baselines like M2IB and GradCAM produce forced explanations on mismatched pairs, whereas AdaIB suppresses responses, validating the practical need for the method.
2.  **Strong Theoretical Justification:** The paper provides a robust theoretical analysis of the proposed framework. Proposition 1 establishes AdaIB as a generalization of classical IB, while Theorems 1 and 2 formally prove the sufficiency and minimality behaviors in high and low relevance regimes, respectively. This theoretical grounding adds significant value beyond empirical results.
3.  **Comprehensive Empirical Validation:** The evaluation covers diverse datasets (CC3M, LAION-400M, RefCOCO) and multiple metrics (Confidence Drop/Incr, Pointing-game IoU, ROAR). The inclusion of large-scale web-crawled data (LAION) is a strength, as it tests the claimed robustness against noise more effectively than smaller, curated datasets like Flickr8k alone.

### Weaknesses
1.  **Marginal Quantitative Gains:** In several key metrics, the performance improvement over baselines is minimal. For instance, in Table 1 (cc3M-I Drop), AdaIB scores 1.01 ± 0.08 vs. M2IB at 1.11 ± 0.10 (improved), but on cc3M-T Drop, AdaIB is 1.07 vs. M2IB 0.90 (worse). On cc3M-I Incr., AdaIB is 40.70 vs. NIB 38.80 (modest improvement). These gains sometimes lack statistical significance given the standard deviations, raising questions about the magnitude of the benefit in clean settings.
2.  **Limited Evaluation Rigor on Key Metrics:** The ROAR experiment (Appendix F) uses a "Remove and Corrupt" approximation rather than full retraining ("Remove and Retrain") due to computational cost. This is explicitly noted ("We adapt... Remove and Retrain (ROAR)... computationally efficient"), but it weakens the reliability of the "importance" claim compared to the original benchmark.
3.  **Dependence on External Similarity for Adaptation:** The analysis in Section 4.4 and Appendix E suggests the relevance function $f(X, Y)$ is often implemented as simple L2 or Cosine distance. If the adaptation mechanism relies chiefly on standard CLIP similarity scores, it risks being circular (using the model's own alignment metric to adjust interpretability of that same model). The learnable $g$ function helps, but the coupling is not fully decoupled from the alignment bias.
4.  **Absence of Human Evaluation:** Multimodal attribution is fundamentally about human understanding. The paper relies entirely on quantitative proxies (confidence drop, IoU). Without human studies to verify if the generated maps are actually more "faithful" or easier to interpret for users, the qualitative claims (Fig 1, Fig 7) remain subjective.

### Novelty & Significance
*   **Novelty:** The core idea of an *adaptive* Information Bottleneck for interpretability is novel in the multimodal context. While IB-based interpretability exists (M2IB, NIB), the introduction of a sample-specific, learnable trade-off parameter conditioned on modality alignment distinguishes this work. The theoretical extension to "No Gratuitous Leakage" is a nice addition.
*   **Significance:** Addressing robustness to misaligned data is highly significant for deploying VLMs in uncurated environments. If validated, this approach could make attribution trustworthy in real-world applications where clean, aligned data is assumed but not guaranteed. However, the incremental nature of the change to the IB objective keeps the breakthrough level at "solid but not groundbreaking."

### Suggestions for Improvement
1.  **Address Title Typos:** The title "Evil in the Pairing Assumption" appears to be an OCR/processing error (likely meant "A" or similar). This must be corrected immediately to maintain professional credibility, as the current text is distracting.
2.  **Clarify the "Adaptive" Mechanism:** Provide a deeper ablation analysis showing how much performance relies on the learnable $g$ function versus the predefined $f$ (similarity). If $f$ is standard cosine similarity, explicitly state this trade-off. Consider visualizing how different $f$ metrics change the explanation space.
3.  **Include Human Evaluation:** To satisfy ICLR's standards for interpretability papers, include at least a small-scale user study or a more rigorous qualitative comparison where participants judge explanation faithfulness on mismatched pairs.
4.  **Statistical Significance Testing:** Ensure the reported improvements (e.g., Table 1) are backed by statistical significance tests, especially where standard deviations overlap or margins are thin (e.g., 1.01 vs 0.90).
5.  **Formatting Cleanup:** While parser artifacts should be ignored, the provided text contains garbled tables (e.g., Table 5, Table 7) that are difficult to read even for a reviewer. The final submission must ensure all LaTeX/Markdown tables render correctly to avoid obscuring data.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Decouple relevance function $f$ from CLIP embeddings:** Currently, $f(X, Y)$ relies on the same model's embedding space to detect mismatch, creating circular reasoning where AdaIB simply thresholds CLIP's own confidence. Validate using an external alignment scorer (e.g., a separate classifier) to prove the adaptive bottleneck adds value beyond simple similarity filtering.
2. **Resolve computational efficiency contradictions:** The text claims "no significant increase in computational load," yet Table 4 shows AdaIB (2.27 FPS) is ~5x slower than NIB (12.5 FPS). Provide a detailed breakdown of the overhead introduced by the learnable $g$ and per-sample optimization to justify the efficiency claims.
3. **Complete missing ROAR benchmark data:** Table 3 headers are present but the result rows are missing/garbled in the manuscript, yet specific numerical claims are made in the text. Reproduce and fully report these zero-shot retrieval degradation scores to verify the robustness claims.
4. **Define ground truth for mismatched pairs:** The paper claims "trustworthy explanations" on mismatched data but lacks a metric for correct abstention versus forced hallucination. Introduce a metric or protocol (e.g., human annotation) to quantify when suppressing attribution is the correct behavior versus when it is a failure.

### Deeper Analysis Needed (top 3-5 only)
1. **Disentangle adaptive IB from simple weighting:** Verify if the performance gain comes from the Information Bottleneck formulation or merely from weighting gradients by $f(X, Y)$. Compare against a baseline that applies $f(X, Y)$ as a post-hoc mask on standard GradCAM to isolate the contribution of the adaptive objective.
2. **Analyze false negative rates on hard valid samples:** Since low relevance $f$ triggers strong compression, analyze if AdaIB incorrectly suppresses attributions for valid but semantically complex pairs (e.g., abstract art, occlusion). Quantify the trade-off between noise suppression and loss of explanatory power on difficult but aligned data.
3. **Verify learned $g$ adheres to theoretical constraints:** The proofs rely on $g$ being non-increasing relative to $f$, but $g$ is learned via an MLP. Visualize the learned function shape to confirm it respects the monotonicity required for the theoretical guarantees (Theorems 1-3) to hold in practice.

### Visualizations & Case Studies
1. **Distribution of effective $\beta$ across datasets:** Plot the histogram of the effective trade-off parameter $\beta_{eff} = g/f$ for clean versus noisy datasets. This would directly reveal whether the method dynamically adjusts constraints as claimed or collapses to a static regime.
2. **Failure cases of over-suppression:** Show examples where AdaIB suppresses attribution on correctly paired but low-similarity inputs (false negatives). Contrasting these with successful suppressions on mismatched data is necessary to assess the risk of discarding valid explanations.
3. **Comparison of $f$ metric influence on spatial attention:** Figure 5 shows different $f$ metrics change attention regions, but does not explain why. Provide a case study showing how choosing L2 vs. Cosine for $f$ alters the semantic meaning of the attribution map to justify the default L2 choice.

### Obvious Next Steps
1. **Replace internal similarity with external relevance:** Future work must define $f(X, Y)$ using signals external to the explained model (e.g., metadata, external verifier) to avoid the circularity of using the model's own confidence to explain its reliability.
2. **Conduct human evaluation of explanation quality:** Quantitative proxy metrics (Drop/Increase) are insufficient for "trustworthiness." A user study comparing AdaIB against baselines on noisy data is required to validate the claim of improved interpretability.
3. **Evaluate downstream impact of attribution-guided pruning:** Demonstrate that using AdaIB masks to remove features actually improves model robustness or calibration in downstream tasks, rather than just improving attribution metrics themselves.

# Final Consolidated Review
## Summary

The paper proposes AdaIB (Adaptive Information Bottleneck), a framework for multimodal attribution in vision-language models that dynamically adjusts the compression-prediction trade-off based on the semantic alignment between image-text pairs. Unlike prior IB-based attribution methods (M2IB, NIB) that assume aligned modalities, AdaIB introduces a learnable relevance function to weight the IB objective sample-wise, enabling robust attributions even for mismatched or noisy pairs.

## Strengths

- **Well-motivated problem formulation:** The identification of the "pairing assumption" as a vulnerability in existing attribution methods is genuine. Figure 1 effectively demonstrates that baselines produce forced explanations on mismatched pairs while AdaIB suppresses spurious attributions—a practically important capability for real-world deployment.

- **Theoretical grounding with meaningful guarantees:** Proposition 1 establishes AdaIB as a strict generalization of classical IB, and the bounded leakage property (Proposition 2, Corollary B.1) ensures that high relevance does not incentivize unnecessary information retention. The formal analysis of sufficiency/minimality regimes (Theorems 1-2) provides principled bounds on behavior.

- **Comprehensive empirical evaluation:** Experiments span multiple datasets (CC3M, LAION-400M, Flickr8k, RefCOCO) and metrics (Confidence Drop/Increase, Pointing-game IoU, ROAR). The inclusion of naturally noisy web-crawled data appropriately tests robustness claims.

## Weaknesses

- **Inconsistent description of the relevance function:** Section 5.2 states "For the function f(X, Y), we choose the L2 distance by default," but L2 distance is *large* for dissimilar pairs—the opposite of what the theory requires (f should be large for high relevance). The appendix correctly defines inverse-distance formulations (f = 1/(d + ε)), and Figure 4 shows matched pairs having higher f values. This inconsistency between the main text and implementation creates confusion about whether the theory correctly guides practice.

- **Misleading computational cost framing:** Table 4 shows AdaIB achieves 2.27 FPS versus NIB's 12.5 FPS—a ~5× slowdown. Yet the paper claims "no significant increase in computational load" by comparing only to M2IB (2.47 FPS), ignoring the faster baseline. This selective comparison understates the efficiency cost.

- **Key misalignment results relegated to appendix:** The central empirical claim—that AdaIB handles misaligned pairs better—is substantiated primarily in Appendix C (Table 5). The main paper body should include these results for readers to evaluate the core contribution without consulting supplementary material.

- **Adapted ROAR metric lacks justification:** The paper modifies ROAR by skipping retraining and measuring zero-shot retrieval degradation instead. The original ROAR requires retraining specifically to control for distribution shift; this adaptation changes what the metric measures without sufficient discussion of validity implications.

- **Dependence on model's own similarity for adaptation:** The relevance function f(X,Y) is computed from CLIP embeddings—the same model being interpreted. This creates potential circularity: if CLIP itself produces spurious alignments, f inherits those biases rather than correcting them. The learnable g function helps, but the dependence remains.

- **Marginal or inconsistent improvements on some metrics:** On cc3M-T Drop, AdaIB scores 1.07±0.11 versus M2IB's 0.90±0.08—worse performance that the paper does not acknowledge. Several improvements over baselines are modest relative to standard deviations, raising questions about effect size in clean settings.

## Nice-to-Haves

- **Human evaluation of explanation quality:** The paper claims improved "trustworthiness" but relies entirely on proxy metrics. User studies evaluating whether suppressed attributions on mismatched pairs are perceived as appropriate would strengthen interpretability claims.

- **External relevance function validation:** Using a separate alignment model (not the explained model itself) to compute f(X,Y) would decouple the adaptation from the explained model's potential biases.

## Removed Points

- The harsh critic claims Theorems 1-3 are "trivial" and Theorem 4's Brouwer-based guarantee is unrealistic. While some theorems are indeed straightforward consequences of the formulation, the overall theoretical framework (Proposition 1, bounded leakage property) provides meaningful guarantees. The stationary point existence, while technical, ensures optimization is well-founded.

- The claim that the title "Evil in the Pairing Assumption" is an OCR error appears unfounded—the phrasing seems intentional as a provocative framing of the problem.

- The criticism of "learning rate 1.0 is extremely unusual" is hyperparameter nitpicking; the method works with this setting, which is what matters for reproducibility.

- The demand for human evaluation, while valuable, is not standard for this venue's interpretability papers and belongs in nice-to-have rather than weaknesses.

## Novel Insights

The observation that adaptive IB creates natural "explanation suppression" behavior on mismatched pairs—rather than forcing hallucinated alignments—represents a genuine insight. The paper's analysis of how different samples require different optimal β values (Figure 2) directly motivates the sample-specific adaptation, which prior IB-based methods assumed fixed. Additionally, Appendix D's finding that identical f values can yield different learned g values suggests the network captures contextual nuance beyond simple similarity scoring, though this also raises questions about whether learned g respects the monotonicity assumptions underlying the theory.

## Suggestions

- Clarify in the main text whether f is raw L2 distance or inverse distance, and ensure consistency throughout. If inverse distance is used (as the results suggest), correct Section 5.2.

- Report complete ROAR results in the main body and discuss the methodological adaptation from the original benchmark.

- Add a sentence acknowledging computational overhead relative to NIB and explain the trade-off (adaptivity versus speed).

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
