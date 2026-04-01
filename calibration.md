=== CALIBRATION EXAMPLE 1 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

### Title & Abstract

The title accurately reflects the core idea of the work. However, the abstract contains a significant factual inconsistency with the body: it claims experiments on "three widely used benchmark datasets," while Section 4.1 opens by announcing "four ZSL datasets." Only three datasets ultimately appear in results tables (aPY, AwA1, AwA2), so the paper is internally inconsistent regardless of which count is correct. The abstract provides almost no quantitative summary of results (no numbers are given), which is a problem for ICLR where readers expect to know the magnitude of improvement at a glance. The claim of "fine-grained visual-semantic relationships at the image level" is stated as the key idea but is never operationally defined in the abstract.

---

### Introduction & Motivation

**Weak dismissal of generative methods.** The paper asserts that generative approaches "do not fundamentally solve the zero-shot problem" and are "inconsistent with the original intention of the zero-shot problem." This is an editorial opinion, not a scientific argument. The generative paradigm has a principled motivation (bridging the train/test class distribution gap) and is not dismissed anywhere in the literature on these grounds. This weakens the paper's framing from the outset.

**Missing the central challenge: GZSL.** Since at least 2018 (Xian et al., 2018), Generalized Zero-Shot Learning (GZSL) — where at test time both seen and unseen classes are candidates — has been the primary evaluation setting in the ZSL community. The introduction makes no mention of GZSL and the paper does not evaluate under it at all. For an ICLR submission in 2025, this is a critical omission; the paper is implicitly evaluating a largely superseded protocol.

**Contributions are vague.** The first bullet says the method "breaks new ground by discerning fine-grained visual-semantic relationships at the image level," but this is a re-statement of the motivation, not a technically precise claim. The second contribution — "a dual-layer embedding method" — is so generic it does not differentiate from prior coupled dictionary learning work (e.g., CDL, Jiang et al. 2018). The third contribution ("a large number of experiments") is not a scientific contribution at all.

**Duplicated paragraph.** The second paragraph of Section 1 (beginning "In recent years, the development of general artificial intelligence...") is reproduced verbatim at the end of the same section. This is either a manuscript error or a parser artifact; either way it needs to be flagged as it suggests insufficient proofreading.

---

### Related Work

**Closest prior work underexplored.** The most directly related work — CDL (Jiang et al., 2018) and HCDDL (Li et al., 2023) — uses coupled dictionaries for ZSL. Both are listed in the comparison table, but the related work section does not adequately explain how CF-ZIR differs from them conceptually or technically. HCDDL in particular ("Hierarchical Coupled Discriminative Dictionary Learning") sounds extremely similar to CF-ZIR's "dual-layer embedding"; the paper needs to clearly articulate the architectural and conceptual distinction.

**CLIP and large vision-language models.** The paper cites CLIP (Radford et al., 2021) in its related work section, explicitly identifying it as a method to position against, yet CLIP is not included in the comparison table and no CLIP-based ZSL methods are cited (e.g., CoOP, CALIP, CaFo, or CLIP-ZSL variants that are widely used as baselines). Given that CLIP-based ZSL achieves very high accuracy on exactly these datasets, omitting it from comparison is a significant gap.

**Missing influential methods.** Widely cited ZSL works that are absent include: AREN (Xie et al., 2019), APN (Xu et al., 2020), GEM-ZSL (Liu et al., 2021), SGMA (Zhu et al., 2019), and any work on CUB/SUN which are the two most commonly used fine-grained benchmarks. The reliance solely on aPY/AwA1/AwA2 — which are considered "easier" datasets — and the omission of CUB-200 and SUN are notable.

---

### Method / Approach

**Sparsity is promised but never enforced.** The text repeatedly refers to the "sparsity coefficient" obtained from reconstructing image features over the dictionary: "the sparsity coefficient obtained from the reconstruction of the common visual feature dictionary atom for a seen class image can describe the degree to which each attribute is included." However, **none of the loss functions (Eq. 1–6) contain an ℓ₁ norm or any sparsity-inducing penalty.** Without a sparsity constraint, X^s is simply a least-squares solution and has no reason to be sparse or interpretable as attribute scores. This is a fundamental mismatch between the motivation and the actual formulation.

**Dimensional inconsistency in Eq. (2).** D_A is defined in the text as "the attribute dictionary of the visual-attribute coupled dictionary" with dimension R^{M_v × K} (line 216, where M_v is the *visual* feature dimension), but in Eq. (2) it multiplies X_r to reconstruct **A** ∈ R^{M_a × K}. The dimensions are inconsistent: D_A should be R^{M_a × K} (attribute space), not R^{M_v × K}. This appears to be an error in the problem formulation.

**Undefined "image-level" dictionaries.** In Section 3.4, the recognition procedure relies on D^{image}_s and D^{image}_v, yet these are never defined in Section 3.3. The paper says in 3.4, "using the image-level semantic dictionary D^{image}_s" without ever distinguishing image-level from class-level dictionaries in the training section. The claimed novelty of image-level versus class-level relationships is thus not technically grounded anywhere in the method description.

**Algorithm 1 contains an apparent error.** Line 10 states "Update X_r via minimizing Eq. (6)," but Eq. (6) is L_vs = L_d + βL_q, which involves X_e, D_y, D_x, Q — not X_r. X_r is specific to the first stage. The second stage's embedding variable should be X_e (updated via Eq. 4). This suggests a copy-paste error but undermines confidence in the reproducibility of the algorithm.

**Eq. (7) is missing.** Equation 7 is rendered as a blank picture placeholder in the OCR output but is referenced explicitly in the text of Section 3.4 ("shown as Eq. 7"). Even discounting parser artifacts, the recognition equations depend on this missing formula, making the recognition procedure incompletely specified.

**No closed-form or convergence analysis.** The optimization alternates between multiple non-convex objectives without any discussion of convergence guarantees, computational complexity, or whether the alternating updates are guaranteed to decrease the joint objective.

---

### Experiments & Results

**No Generalized ZSL evaluation.** As noted above, the paper evaluates only on the conventional ZSL (cZSL) protocol, where at test time only unseen classes are candidates. GZSL — which reports the harmonic mean H of per-class accuracy on seen (S) and unseen (U) classes — is the standard evaluation for ICLR-level papers. Every method in Table 2 that was published after 2018 also reports GZSL results in its original paper. Omitting GZSL entirely in 2025 is a fundamental deficiency that would be flagged by any area chair familiar with the field.

**No results on CUB or SUN.** The three "widely-used benchmarks" chosen (aPY, AwA1, AwA2) are among the simplest ZSL benchmarks. AwA1 and AwA2 share the same 50 classes; reporting both without CUB-200-2011 (fine-grained birds) or SUN (fine-grained scene understanding) substantially narrows the scope of the claimed contribution. The paper's claim of "fine-grained" relationships is especially unconvincing without evaluation on fine-grained datasets.

**Marginal improvements.** The headline result is 72.0% on AwA2 vs. 71.8% for ERPCNet (Li et al., 2022) — a 0.2% difference — with no error bars or statistical significance test. On aPY, CF-ZIR achieves 48.0% vs. HCDDL's 50.6% (second place, 2.6% lower); on AwA1, 71.5% vs. HCDDL's 71.8% (second place, 0.3% lower). Given the small margins and the absence of statistical testing, the claims of competitive performance cannot be assessed with confidence.

**Ablation is incomplete.** Table 3 only considers two variants: removing discrimination loss (DL) and removing visual-semantic alignment (VSA). Missing ablations that would materially affect conclusions include:
- Effect of dictionary size K and L (the primary capacity hyperparameter)
- Contribution of the image-level semantic vector versus a simple class-level baseline (the central claimed novelty)
- Sensitivity to the attribute feature extraction network choice
- Effect of the backbone (only ResNet-101 is tested)
- Comparison between the three recognition spaces (visual, embedding, semantic) — mentioned in the text but no results reported

**Hyperparameter tuning opacity.** Six hyperparameters (λ, α, β, γ, µ, η) are searched over a coarse grid of 5 values each. It is not stated whether γ (not appearing in any equation) is a typo, what the winning values are on each dataset, or whether the same hyperparameters work across datasets. With six hyperparameters and small datasets (aPY has only 5,932 seen images), overfitting to the validation split is a real concern.

**t-SNE analysis (Section 4.4) is qualitative and self-referential.** Fig. 2 shows that image semantic vectors cluster better than raw image features, which is expected by construction (the loss function explicitly penalizes deviation from class semantic vectors). Comparing only against the model's own raw features rather than against the semantic vectors produced by a baseline method (e.g., CDL or HCDDL) does not demonstrate that CF-ZIR's image semantic vectors are better than alternatives.

---

### Writing & Clarity

**"Four datasets" vs. "three datasets" inconsistency.** Section 4 header states "four benchmarks" and Section 4.1 says "four ZSL datasets," but Table 1 lists only three, Table 2 has three columns, and the abstract says three. This discrepancy is never resolved.

**Section 3.4 is not self-contained.** The recognition in embedding space (Eq. 9–10) and semantic space (Eq. 11) reference D^{image}_v and D^{image}_s without prior definition. A reader trying to reproduce the method cannot do so from the paper as written.

**Framework figure (Fig. 1) is incomprehensible from the paper text.** The ASCII description of the figure rendered by the parser shows a complex diagram, but the textual walkthrough in Section 3.2 does not map clearly to specific components and their information flow. Given that Fig. 1 is the primary architectural illustration, it needs to be fully interpretable.

---

### Limitations & Broader Impact

**No limitations section.** The conclusion section makes no mention of limitations. The following are substantive omissions:

1. **Attribute dependency.** The method requires class attribute vectors for *all* classes including unseen classes at test time. This is a strong assumption that many practical applications cannot satisfy. No discussion is provided.

2. **Scalability.** Dictionary learning scales poorly with dataset size and number of classes. The paper does not discuss computational cost or how the method would behave on datasets with hundreds of classes.

3. **No GZSL capability demonstrated.** The paper's recognition formulas all assume the test set contains only unseen classes, which cannot be relaxed to GZSL without architectural changes. This is an unacknowledged fundamental limitation.

4. **Dependence on pre-extracted ResNet-101 features.** The method takes fixed CNN features as input and does not perform end-to-end learning. No justification is given for why this design choice is appropriate, and no comparison to end-to-end methods is made.

---

### Overall Assessment

CF-ZIR proposes a two-stage coupled dictionary learning framework for zero-shot image recognition that generates image-level semantic vectors as an intermediate representation. The core idea — guiding visual dictionary atoms by attribute semantics to obtain per-image attribute scores — has an intuitive motivation. However, the paper has critical weaknesses that prevent acceptance at ICLR in its current form. Most importantly: (1) the paper does not evaluate on Generalized ZSL, which has been the required evaluation protocol at top venues since 2018; (2) the formulation contains internal inconsistencies — sparsity is motivated but never enforced, a key dimensional mismatch exists in Eq. (2), and Algorithm 1 has a likely copy-paste error — that undermine reproducibility; (3) the central novelty claim of "image-level" over "class-level" relationships is stated qualitatively but never ablated directly; (4) experiments are limited to three easy benchmarks (no CUB, no SUN), omit CLIP-based baselines, and show improvements of 0.2% without significance testing; and (5) the ablation study does not examine the primary design choices. The contribution, while coherent, reads more as an incremental extension of coupled dictionary learning methods (particularly CDL and HCDDL) than a conceptually novel advance. Addressing the GZSL evaluation, fixing the formulation inconsistencies, adding fine-grained benchmark experiments, and strengthening ablations would be the minimum revisions needed to meet ICLR standards.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper proposes CF-ZIR (Common Feature Learning for Zero-shot Image Recognition), a method that learns fine-grained visual-semantic relationships at the image level through a dual-layer embedding approach. The method first establishes visual-attribute coupled dictionaries to extract common visual features and generate image semantic vectors, then builds visual-semantic coupled dictionaries for cross-domain alignment. Experiments on three benchmark datasets (aPY, AwA1, AwA2) demonstrate competitive performance compared to existing embedding-based and generative methods.

### Strengths
1. **Clear motivation and problem identification**: The paper correctly identifies that existing ZSL methods typically establish class-level, coarse-grained relationships between visual and semantic spaces, ignoring intra-class variations. This observation is valid and the proposed image-level approach addresses a real limitation.

2. **Well-structured methodology**: The two-stage framework (visual-attribute embedding followed by visual-semantic embedding) is logically organized. The mathematical formulations in Equations 1-6 are clearly presented with proper notation definitions.

3. **Solid experimental results**: CF-ZIR achieves 48.0%, 71.5%, and 72.0% accuracy on aPY, AwA1, and AwA2 respectively, outperforming or matching most compared methods. Notably, it achieves the best result on AwA2 among embedding-based methods.

4. **Meaningful ablation studies**: Table 3 demonstrates the contribution of the discrimination loss (improving 0.3-1.6%) and visual-semantic alignment (improving 2.0-6.0%), confirming each component's importance.

5. **Qualitative analysis**: The t-SNE visualization in Figure 2 effectively shows that generated image semantic vectors exhibit better intra-class clustering and inter-class dispersion compared to raw visual features.

### Weaknesses
1. **Limited novelty**: Dictionary learning for ZSL has been extensively explored (CDL Jiang et al. 2018, HCDDL Li et al. 2023). The paper does not clearly articulate what distinguishes CF-ZIR from these prior works beyond the specific formulation. The "dual-layer embedding" concept, while framed as novel, shares significant similarities with hierarchical dictionary learning approaches.

2. **Missing GZSL evaluation**: The paper only evaluates conventional ZSL (CZSL) but omits Generalized Zero-Shot Learning (GZSL) evaluation, where both seen and unseen classes are considered during testing. GZSL is the more challenging and practically relevant setting, and its omission weakens the contribution's significance.

3. **Inconsistent mathematical notation**: In Equation 2, A ∈ R^{Ma×K} (attribute feature matrix) and D_A ∈ R^{Mv×K} (attribute dictionary) have different dimensionalities, but both are multiplied with X_r ∈ R^{K×K}, creating dimensional inconsistency that the paper does not explain.

4. **Insufficient hyperparameter discussion**: The paper mentions selecting λ, α, β, γ, µ, η from {0.001, 0.01, 0.1, 1, 10} but does not report the actual values used for each dataset. The hyperparameter γ appears in the text but is not used in any equation, suggesting a notation error.

5. **Limited comparison depth**: While Table 2 shows competitive results, the paper lacks analysis of computational complexity, training time, or memory requirements compared to other methods—important factors for practical deployment.

6. **Redundant writing**: The introduction contains significant text repetition (the first paragraph and its subsequent restatement are nearly identical), and some contribution statements are repeated, suggesting insufficient editing.

### Novelty & Significance
The novelty of CF-ZIR is incremental rather than substantial. The core idea of using dictionary learning to bridge visual and semantic spaces follows established research directions (CDL, HCDDL). The image-level semantic vector generation is a reasonable extension but does not represent a fundamental breakthrough. The significance is moderate—competitive results on standard benchmarks are achieved, but without GZSL evaluation and with limited analysis of what the method enables beyond existing approaches, the impact may be limited.

### Suggestions for Improvement
1. **Include GZSL evaluation**: Add the generalized zero-shot learning setting with both seen and unseen classes during testing. Report harmonious mean (H) alongside seen class accuracy (S) and unseen class accuracy (U).

2. **Clarify the novelty over existing dictionary-based ZSL methods**: Explicitly compare and contrast with CDL and HCDDL, explaining the specific differences in formulation and why CF-ZIR should be preferred.

3. **Fix mathematical inconsistencies**: Correct the dimensionality mismatch in Equation 2, clarify or remove the unused γ parameter, and ensure all notation is consistent throughout.

4. **Provide complete implementation details**: Report exact hyperparameter values used for each dataset, number of training iterations, convergence criteria, random seeds, and computational resources required.

5. **Add computational analysis**: Include training time, inference speed, and memory footprint comparisons with baseline methods to help readers assess practical applicability.

6. **Improve writing quality**: Remove the duplicated text in the introduction and abstract sections, and ensure contributions are stated concisely without repetition.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments

1. **Generalized ZSL (GZSL) evaluation is entirely absent.** ICLR reviewers will immediately flag this — GZSL (where test classes include both seen and unseen classes) has been the dominant evaluation protocol since Xian et al. (2019). Reporting Harmonic Mean (H) of seen-class accuracy (S) and unseen-class accuracy (U) on all three datasets would make the paper significantly more credible and complete.

2. **Missing standard fine-grained benchmarks: CUB-200-2011 and SUN Attribute.** These two datasets are universally included in ZSL papers. CUB is especially critical because the paper's core claim is about learning *fine-grained* visual-semantic relationships — a fine-grained bird dataset is the most natural place to validate this. Showing strong performance on CUB would powerfully support the paper's narrative. Table 1 mentions "four benchmarks" while the paper only reports results on three — this inconsistency should be resolved by actually adding the fourth.

3. **No comparison with CLIP-based or vision-language model baselines.** The paper mentions CLIP briefly in the related work but does not include it as a comparison point in Table 2. Given that CLIP and its derivatives (e.g., CoCoOp, CALIP, CaFo) are now strong ZSL baselines, including them — even to show where CF-ZIR is complementary — would substantially strengthen the empirical section for ICLR 2025 reviewers.

4. **Sensitivity analysis for the four hyperparameters (λ, α, β, µ, η).** The paper states values are selected from {0.001, 0.01, 0.1, 1, 10} but provides no analysis of how sensitive results are to these choices. A heatmap or line plot showing performance vs. key hyperparameter values (e.g., λ vs. α) would greatly increase trust in the robustness of the method.

5. **Ablation of the dictionary size K and L.** The number of dictionary atoms is a core architectural choice that directly affects the representation capacity of both stages. An experiment varying K and L would reveal important trade-offs and help practitioners configure the model.

6. **Comparison under the same visual backbone.** It is unclear whether all baselines in Table 2 use the same ResNet-101 features. A controlled comparison where all embedding-based methods use identical features (same split, same backbone) would make the comparison fairer and more convincing.

---

### Deeper Analysis Needed

1. **Convergence analysis for the alternating optimization.** The training algorithm (Algorithm 1) uses alternating updates without any guarantee of convergence. A brief theoretical argument (or empirical convergence curve showing loss vs. iteration) would significantly strengthen the paper's rigor — especially important for ICLR, which values theoretical grounding.

2. **Theoretical justification for why image-level dictionaries generalize to unseen classes better than class-level ones.** The paper argues intuitively but does not provide a formal argument. A bound or analysis showing that image-level semantic vectors reduce intra-class variance in a way that provably improves transfer would strengthen the theoretical contribution.

3. **Analysis of the hubness problem.** ZSL embedding methods are well-known to suffer from hubness (a few unseen class prototypes dominate nearest-neighbor retrieval). The paper does not discuss or address this. Showing that CF-ZIR's image-level semantic vectors reduce hubness (e.g., measuring the Robin Hood index) would add meaningful insight.

4. **Ablation of the three recognition spaces (visual, embedding, semantic).** Section 3.4 describes three prediction strategies but Table 3 only ablates discriminative loss and visual-semantic alignment. A systematic comparison of recognition in visual space vs. embedding space vs. semantic space (as described in Eqs. 8–11) would tell a much richer story about *where* the method's benefit comes from.

5. **Complexity comparison with generative methods.** The paper argues that generative methods are "computationally intensive" and the proposed approach avoids this, but provides no quantitative evidence. Reporting training time, number of parameters, or FLOPs relative to GAZSL, CE-GZSL, and other generatives would make this claim concrete.

---

### Untapped Applications

1. **Zero-shot action recognition in video.** The dictionary-learning framework for mapping visual to semantic space is architecture-agnostic and could naturally extend to temporal visual features. Testing on UCF-101 or ActivityNet with action attribute annotations would broaden impact considerably.

2. **Zero-shot medical image classification.** Clinical attributes (e.g., texture, shape, color of lesions) are a natural analogue to visual attributes in ZSL. Applying CF-ZIR to a dataset like CheXpert or a dermatology benchmark (where class imbalance makes zero-shot transfer practically valuable) would demonstrate real-world utility.

3. **Few-shot extension.** The common visual feature dictionary **F** learned from seen classes could be adapted to the few-shot setting by fine-tuning on a handful of labeled examples from new classes. Exploring this would significantly broaden the paper's scope and appeal.

4. **Cross-dataset zero-shot transfer.** Training on AwA1 and evaluating on AwA2 (or vice versa), or training on ImageNet and evaluating on aPY, would test whether the learned visual-attribute dictionary truly captures class-independent common features — a direct empirical test of the paper's central claim.

---

### Visualizations & Case Studies

1. **Attribute activation maps.** Showing which spatial regions of an image activate specific dictionary atoms (attributes) — e.g., using Grad-CAM-style visualizations — would make it vivid and intuitive that CF-ZIR is truly learning *where* attributes appear, not just fitting a global feature vector. This would be a compelling Figure 3.

2. **Qualitative failure case analysis.** Adding a small panel showing images that CF-ZIR misclassifies and explaining *why* (e.g., visually ambiguous classes, attribute overlap) would demonstrate intellectual honesty and sharpen the paper's self-understanding, which ICLR reviewers appreciate.

3. **Image semantic vector quality: quantitative metric alongside t-SNE.** The t-SNE in Figure 2 is visually suggestive but hard to evaluate rigorously. Adding a quantitative measure of cluster quality (e.g., intra-class compactness and inter-class separation using silhouette score or class-level cosine similarity matrices) would make Section 4.4 much stronger.

4. **Dictionary atom visualization.** Visualizing the learned dictionary atoms in **F** — e.g., showing which training images have the highest activation for each atom and correlating with attribute names — would provide intuitive evidence that the dictionary has learned semantically meaningful visual primitives.

5. **Nearest-neighbor retrieval examples in semantic space.** Showing unseen class images alongside their top-5 nearest class prototypes in semantic space (correct and incorrect predictions) would give readers a concrete sense of when and why the method succeeds or fails on specific unseen classes.

---

### Natural Next Steps

1. **End-to-end training with learned visual features.** The current method uses frozen ResNet-101 features. A natural extension would be to train the dictionary learning objective jointly with the CNN backbone (or a ViT), allowing the visual features themselves to be optimized for attribute alignment. This could substantially boost performance.

2. **Replacing hand-annotated attributes with LLM-generated descriptions.** The method relies on manually curated attribute vectors, which limits scalability. A follow-up could use GPT-4 or LLaMA to generate attribute descriptions automatically and verify that CF-ZIR still benefits from image-level alignment — bridging dictionary learning with modern language models.

3. **Extending the dual-layer dictionary to a hierarchical semantic ontology.** Rather than a flat attribute space, future work could organize attributes hierarchically (e.g., body parts → textures → colors) and learn dictionaries at each level, potentially enabling compositional generalization to truly novel categories.

4. **Transductive and inductive ZSL unification.** The current framework is inductive. A promising extension would incorporate transductive inference — using the distribution of unlocked unseen class images at test time to re-estimate the semantic dictionary — which could close the gap with generative methods without the computational cost of generation.

5. **Federated or privacy-preserving ZSL.** Since CF-ZIR's dictionary learning separates visual feature extraction from semantic space learning, it could be adapted to federated settings where raw images cannot be shared but dictionary atoms can — a practically motivated and timely direction.

# Report: Potentially Missed Related Work
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

1. **ConSE (Convex Combination of Semantic Embeddings)** — Norouzi et al. (2014, ICLR).
   Why potentially missed: This is a seminal embedding-based ZSL method that established core principles for visual-semantic space alignment through convex combinations of class label embedding vectors. The paper extensively discusses embedding-based approaches and their foundations, but does not cite this foundational work that pioneered probability-based embedding strategies for zero-shot recognition.

2. **HEX (Hierarchy and Exclusion)** — (2015-2016, relevant ZSL venue).
   Why potentially missed: This work explicitly models relationships between objects and attributes (hierarchical and exclusion relations) to map visual features to label scores for unseen categories. Since the paper's CF-ZIR method relies heavily on attribute-guided feature extraction and inter-class association information, HEX's attribute relationship modeling framework is directly relevant to the paper's theoretical foundation.

3. **DSRL (Deep Sparse Representation Learning)** — Authors and venue unspecified.
   Why potentially missed: Employs non-negative sparse matrix factorization to align visual feature representations with semantic label representation vectors. This approach is technically related to the paper's dictionary learning methodology, as both leverage sparse representation techniques for visual-semantic alignment. The paper could benefit from discussing this alternative sparse coding approach.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [3.0, 3.0, 5.0, 1.0]
Average score: 3.0
Decision: Reject
Binary: Reject

=== CALIBRATION EXAMPLE 2 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

### Title & Abstract

The title, "A Clustering Baseline for Object-Centric Representations," is accurate and appropriately modest. The abstract clearly states the problem (object-centric learning overlooks representation quality), method (multi-scale k-means on SSL features), and claims (outperforms fine-tuned methods on downstream tasks, competitive on segmentation). One concern: the abstract states the method "surpasses the performance of fine-tuned object-centric learning methods," but this claim conflates performance on downstream tasks (where k-means wins) with segmentation quality (where Table 3 shows k-means falls behind SPOT under the standard Locatello protocol). The abstract should be more precise about which evaluation regime drives this conclusion.

---

### Introduction & Motivation

The motivation is well-constructed. The argument that object-centric learning has drifted toward optimizing segmentation metrics at the expense of representation quality is both genuine and important for the field. The identification that slot-based auto-encoders entangle semantic and positional information (leading to degraded embeddings) is a substantive critique grounded in the results.

However, the introduction slightly over-claims: "we demonstrate how to discover objects and parts with a simple multi-scale application of k-means" presents multi-scale k-means as if it were itself a discovery, whereas the bulk of the novelty is in the *evaluation framework* that reveals the quality gap between slot-based methods and simple clustering. The actual algorithmic contribution (Eq. 1) is modest — a geometric progression of K values plus the global CLS token — and this should be acknowledged rather than soft-pedaled.

A concrete gap: the introduction does not identify *why* prior methods entangle semantic and positional information specifically. This is stated as a property of the auto-encoding objective but is not derived or cited to ablative evidence in prior work; Section 3.2 provides the reasoning, but the connection should be made clearer.

---

### Related Work

The related work is thorough across SSL, object discovery, and object-centric learning. The distinction between object discovery (focused on localization) and object-centric learning (focused on representation) is useful and correctly applied throughout the paper. References to LOST, TokenCut, MOST, and spectral clustering methods are appropriate.

One substantive omission: **STEGO (Hamilton et al., 2021)** is cited only in passing despite being a direct competitor in the unsupervised semantic segmentation literature. More importantly, methods that also use k-means or clustering on SSL features for segmentation — such as work using DINO attention maps for unsupervised segmentation — should be more carefully positioned against, since the paper's core algorithmic element (clustering dense SSL features) is not new in isolation. The paper is honest about this ("from a methodological perspective, this work is also based on processing dense features…"), but ICLR reviewers will want a sharper account of what is *new* relative to prior clustering-based approaches to segmentation.

The claim that "most object discovery methods do not output object embeddings" is correct and important for positioning, but the paper could be more explicit about which methods do output embeddings (e.g., ODIN, DINOSAUR) to sharpen the comparison.

---

### Method / Approach

**Clarity:** The method is described concisely. Equation 1 defines the multi-scale representation clearly. The distinction between the native-resolution mode (for representation evaluation) and the 4× upsampled mode with hierarchical k-means (for segmentation evaluation) is a critical implementation detail that is buried in Section 3.3 rather than prominently flagged. This creates ambiguity when reading the results.

**Key assumptions not fully justified:**

1. **Hierarchical k-means (K=256 then target K):** The paper states this "reduces the bias towards equally-sized clusters," citing Vo et al. (2024). However, no ablation is provided to show this matters. Does performance degrade meaningfully without the hierarchical step? This is especially important because the hierarchical k-means runs only for the segmentation evaluation, not the downstream classification tasks — and readers may not catch this split.

2. **Geometric progression of K:** Equation 1 fixes the structure as powers of 2. This is never ablated. Are other progressions (e.g., linear, arithmetic) worse? For classification tasks, the paper compares S4, S8, S16, S32 in Figure 2, but the *structure* of the progression is not varied.

3. **K-means non-determinism:** K-means has random initialization. The paper reports no variance estimates, no averaging over multiple runs, and no comparison of initialization strategies (e.g., k-means++ vs. random). For a paper arguing that its method is a reliable baseline, this is a material omission.

**Slot attention as "soft k-means":** The paper states that slot-based methods are a "soft and parametric approximation of k-means clustering," citing Locatello et al. (2020) and Chang et al. (2022). This is correct but incomplete: the slot attention cross-attention mechanism is iteratively refined over multiple rounds, which introduces competition between slots. Standard k-means also has iterative assignment and update steps. The key differences (soft vs. hard assignment, learned initialization vs. random centroids, gradient-based training vs. Lloyd iterations) deserve a more careful treatment to support the claim that k-means is strictly superior.

---

### Experiments & Results

**The central fairness problem — backbone mismatch:** Table 1, the paper's most important result, compares SPOT trained on **DINOv1 ViT-B/16** against k-means applied to **DINOv2 ViT-B/14 reg**. DINOv2 (with register tokens, trained on larger curated data) is substantially stronger than DINOv1. The DINO CLS token alone is 78.2% on ImageNet; DINOv2 CLS alone is 83.9%. This 5.7-point gap is larger than many of the gains being claimed for k-means over SPOT. The authors attempted to train SPOT with DINOv2 but report it "failed to converge." This is stated in a single sentence with no investigation.

This is the paper's most serious weakness for ICLR reviewers. The proper control experiment — SPOT trained on DINOv2 — is absent precisely because it failed. The paper cannot claim k-means outperforms slot-based methods without controlling for backbone quality. While Table 1 also shows DINO+k-means outperforming DINO+SPOT, the gap is more modest and primarily reflects that slot attention degrades the backbone features, which is already known. The DINOv2 column only has k-means, so readers cannot tell whether the DINOv2 backbone improvement alone would close the gap.

The correct experiment would be to either: (a) fix DINOv2 as backbone and investigate *why* SPOT fails to converge with it (is slot attention incompatible with DINOv2's feature geometry? is it a training instability?), or (b) compare k-means on DINOv1 vs. SPOT on DINOv1 as the primary result and clearly bound DINOv2 gains to backbone quality. Appendix A.3 provides some SPOT+DINOv2 results (Table 7), showing severe degradation of mBO metrics with DINOv2. This is a fascinating and important finding — DINOv2 features appear to resist slot-attention training — but it is treated as a failure rather than an investigation.

**No statistical significance / error bars anywhere.** Classification accuracy values are reported to one decimal place with no confidence intervals across any table or figure. Given that k-means results may vary across runs (random initialization), and that the gains in some cells are small (e.g., +0.4 on ImageNet, +0.5 on SUN397 for DINOv2+k-means), this is a problem for an ICLR submission.

**Segmentation results (Tables 2 and 3) — cherry-picking evaluation protocol:** Table 2 uses the Hénaff et al. (2022) high-recall protocol (255 predicted masks), where k-means wins. Table 3 uses the Locatello et al. (2020) standard protocol (fixed non-overlapping slots), where k-means loses. The paper acknowledges this but does not give equal prominence to both results. The paper presents Table 2 as the main segmentation result and Table 3 as a secondary caveat. For ICLR, which values contribution over cherry-picked numbers, this asymmetric framing is likely to draw criticism.

More concretely: under the Locatello protocol (Table 3), k-means with DINOv2 on COCO scores 30.4 mBO_i (equal to LSD, the weakest baseline) and 38.8 mBO_c (equal to DINOSAUR), falling substantially short of SPOT at 35.0 and 44.7. The paper's narrative of "comparable quality to specialized methods" does not hold under this protocol.

**CLEVR results:** The k-means method achieves dramatic gains on CLEVR (78.2 → 89.7 mAP vs. SPOT at 81.5 for single-model SPOT-16). CLEVR is a *synthetic* dataset with simple textureless colored shapes where k-means will trivially succeed. These gains should be interpreted cautiously and separated from gains on real-world datasets (COCO, ImageNet). The paper does not flag CLEVR's special status.

**Video classification (Section 4.1.2):** Figure 3 compares k-means to CLS and all-patches but does *not* compare to SPOT or any other object-centric method in the video setting. This omission is significant: the video experiment only demonstrates that k-means tokens are better than CLS or random/grid pooling under a token budget, not that they outperform trained object-centric methods in this setting.

**Missing ablation — effect of the attention pooling classifier:** The downstream evaluation uses an attention-pooling classifier (with learnable W_key, W_value, W_out, and a query vector q). This is not a linear probe — it has quadratic expressivity in the number of tokens. The comparison between k-means and SPOT in Table 1 trains *the same classifier architecture* on both token sets, which is fair. However, the classifier is trained with full gradient descent, and it is not clear whether the performance differences reflect representation quality or the classifier's ability to compensate for representation deficiencies (especially relevant for SPOT, where the slots have lower dimensionality or different geometry). An ablation using strictly a linear probe (mean-pooled slots vs. mean-pooled k-means centroids) would make the representation quality comparison cleaner.

---

### Writing & Clarity

The paper is generally well-written. The flow from problem motivation (Section 1) through method (Section 3) to experiments (Section 4) is logical. Figure 1 is informative and shows the intuition clearly. Figure 2's scaling trends are presented cleanly.

One significant clarity issue: the paper conflates two different evaluation setups for k-means — (1) native backbone resolution for classification, and (2) 4× upsampled with hierarchical k-means for segmentation — without a clear schematic or explicit summary. A reader going from Table 1 to Table 2 may not realize the method is materially different (different resolution, different hierarchical procedure). This needs to be flagged more explicitly.

Another clarity issue: Equation 1 defines S_N as including C_1 (a single centroid covering the whole image), which is redundant with the global representation g. The distinction between C_1 and g should be clarified — are they the same vector? If C_1 is the mean of all patches and g is the CLS token, these are different, but both are paired with "dummy masks that cover the whole image." This is confusing.

---

### Limitations & Broader Impact

The authors acknowledge several limitations: noisy mask boundaries, suboptimal performance under the fixed-partition protocol, and no dataset-specific tuning. However, several important limitations are not discussed:

1. **k-means at high resolution is not truly "negligible":** For the segmentation evaluation, the paper upsamples to 4× resolution (56×56 feature maps) and runs hierarchical k-means with K_init=256 then K_target. At batch scale, k-means on 3,136-dimensional feature vectors (56×56) is not trivially fast. No concrete timing numbers are given.

2. **Failure modes of k-means:** k-means is sensitive to (a) feature space geometry (isotropy of clusters) and (b) number of clusters K. The paper shows in Table 4 that DINOv2 ViT-L/14 actually performs *worse* on segmentation than DINOv2 ViT-B/14 ("likely due to clustering issues when the embeddings grow larger"). This is a significant and unresolved failure mode. If larger backbones produce worse clusters, the method may not scale with future SSL progress as the conclusion claims.

3. **The SPOT+DINOv2 convergence failure** is never explained. This is a critical missing analysis: does DINOv2 produce features that are too semantically structured for slot attention to decompose? Too isotropically distributed? Understanding this would be a genuine contribution to the field.

4. **Evaluation on videos uses no temporal modeling beyond positional embeddings:** The video classifier adds temporal embeddings and uses a 2-block transformer. The paper claims "object-centric representations are particularly valuable in scenarios where large spatial and temporal resolutions are crucial." However, the method clusters each frame independently, with no cross-frame object consistency. For long videos or tracking tasks, this is a fundamental limitation not discussed.

5. **The approach is not object-centric in the strict sense:** k-means clusters are not guaranteed to correspond to semantic objects. The paper argues they do empirically (via segmentation metrics), but the method has no mechanism to enforce this. In cluttered scenes with fine-grained textures, clusters may capture textures or backgrounds rather than objects. The paper's own results (Table 3, lower segmentation quality under strict protocols) support this concern.

---

### Overall Assessment

This paper makes a genuine and important empirical contribution by demonstrating that multi-scale k-means clustering on frozen DINOv2 features outperforms trained slot-based object-centric methods (SPOT, DINOSAUR) on downstream classification tasks, and is competitive on unsupervised segmentation under high-recall protocols. The core message — that the field has over-indexed on segmentation metrics and under-valued representation quality — is well-motivated and the experiments largely support it. However, the paper has three problems that are serious by ICLR standards: (1) the primary comparison in Table 1 mixes backbones (DINOv2 for k-means vs. DINOv1 for SPOT), and the paper's own Appendix A.3 shows that SPOT trained with DINOv2 catastrophically degrades — this unexplained failure is the paper's most interesting finding and the least investigated; (2) no variance estimates, confidence intervals, or statistical tests are reported anywhere, despite some margins being small; and (3) the segmentation results under the standard Locatello protocol (Table 3) clearly show k-means *loses* to SPOT, which contradicts the abstract's claim of "comparable quality to specialized methods." The paper would be strengthened substantially by investigating *why* SPOT fails with DINOv2, providing fair same-backbone comparisons as the primary result, reporting variance across k-means random seeds, and giving more balanced prominence to Table 3. As submitted, the contribution stands but is over-claimed, and the experimental design does not fully support the strongest version of the argument.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a simple yet effective baseline for object-centric learning by applying multi-scale k-means clustering to dense features from pre-trained SSL backbones (e.g., DINOv2). The authors demonstrate that this training-free approach produces high-quality object embeddings that outperform fine-tuned object-centric methods like SPOT on downstream tasks (scene classification, video action recognition) while achieving competitive unsupervised segmentation results. The work argues for re-focusing the object-centric learning community on representation quality, not just mask accuracy.

### Strengths
1. **Strong empirical results with a simple method**: The k-means approach outperforms SPOT on representation quality benchmarks by substantial margins (e.g., 89.7 vs 84.4 mAP on CLEVR, 72.6 vs 70.8 mAP on COCO) while requiring no training. This is a compelling demonstration that SSL features already encode object-centric structure.

2. **Comprehensive dual-axis evaluation**: Unlike prior work that focuses primarily on segmentation metrics, this paper evaluates both representation quality (Section 4.1) and mask quality (Section 4.2), providing a more complete picture of object-centric capabilities.

3. **Practical advantages**: The method is training-free, fast, flexible across backbones and number of objects, produces overlapping masks for part-whole hierarchies, and preserves backbone embedding quality—all addressing documented limitations of slot-based methods.

4. **Thorough experimental analysis**: The paper includes scaling studies (Figure 2), backbone ablations (Table 4), video experiments (Figure 3), and comparisons under two evaluation protocols (Tables 2 and 3).

5. **Important conceptual contribution**: The paper makes a strong case that the community's focus on mask quality has come at the expense of representation quality—a valuable re-orientation for future research.

### Weaknesses
1. **Limited methodological novelty**: K-means is an extremely basic algorithm. While the application and multi-scale formulation are novel in this context, the core technique offers little technical innovation. Some readers may question whether this constitutes sufficient contribution for a learning conference.

2. **Noisier mask boundaries**: Qualitative results (Figure 4) and quantitative results under the total partitioning protocol (Table 3) show k-means produces less precise masks than specialized methods, which may limit applicability where precise localization matters.

3. **Hyperparameter sensitivity**: The method requires choosing K values for clustering. While the geometric progression (1, 2, 4, 8, ...) works well, the paper provides limited guidance on adapting these choices to new domains or datasets.

4. **Incomplete comparisons**: The authors note DINOSAUR checkpoints are unavailable and SPOT with DINOv2 backbone failed to converge. While not the authors' fault, this limits the comparison depth to the strongest baselines.

5. **Limited theoretical grounding**: The paper observes that k-means works well empirically but provides little theoretical analysis of why SSL features are so amenable to clustering-based object discovery.

### Novelty & Significance
The novelty lies primarily in reframing object-centric learning around a simple clustering baseline and demonstrating its effectiveness, rather than in the technical contribution of k-means itself. The significance is substantial: this work challenges the prevailing approach of training complex slot-attention architectures and provides a strong baseline that future methods must beat. The finding that SSL features already encode object-centric structure is valuable for the community. However, the simplicity of the approach may divide opinions on novelty grounds.

### Suggestions for Improvement
1. **Add theoretical or analytical insight**: Investigate and explain why DINOv2 features cluster so well into semantic objects—is it the iBOT loss, architectural inductive biases, or data scale? This would strengthen the contribution beyond empirical demonstration.

2. **Provide guidance on K selection**: Develop heuristics or automatic methods for selecting appropriate K values for new domains, perhaps based on feature variance or cluster quality metrics.

3. **Compare with more recent methods**: Include comparisons with other unsupervised segmentation methods like CutLER, TokenCut, or FreeSOLO to strengthen the segmentation baseline.

4. **Explore mask refinement**: Investigate whether simple post-processing (CRF, morphological operations) can improve mask quality while maintaining the training-free nature of the approach.

5. **Add failure case analysis**: Discuss scenarios where k-means fails to capture objects appropriately to help readers understand limitations.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments

1. **Compositional reasoning benchmarks**: The paper's central claim is that object-centric representations improve compositional understanding, yet evaluation stops at classification and segmentation. Testing on CLEVR-Hans, GQA, or Visual Genome relation detection would directly validate the compositionality hypothesis and be far more compelling to ICLR reviewers than mAP on COCO multi-label.

2. **Comparison with SAM and open-vocabulary segmenters for mask quality**: Table 2 and 3 compare only against slot-based methods. Including SAM (zero-shot), DINO-based segmenters, and spectral clustering methods like STEGO or Deep Spectral Methods would establish a much more complete picture of where k-means sits in the segmentation landscape. Reviewers will ask why SAM is not a baseline.

3. **Downstream object-level retrieval evaluation**: Figure 1 teases qualitative nearest-neighbor retrieval using centroids as queries, but the paper never quantitatively evaluates this. Running a formal object-level retrieval benchmark (e.g., Oxford/Paris, or ROxford instance retrieval) would demonstrate that centroid embeddings maintain semantic fidelity in a measurable way.

4. **Few-shot and cross-domain transfer**: The paper argues that k-means avoids dataset-specific overfitting, but only tests in-domain (trained on COCO/ImageNet, tested on same). Evaluating representation transfer on few-shot benchmarks (e.g., Meta-Dataset, EuroSAT, or medical imaging like ChestX-ray14) would make this claim concrete and strongly differentiate the method from SPOT/DINOSAUR which require dataset-specific fine-tuning.

5. **Clustering stability across random seeds**: K-means is non-deterministic. Adding a brief experiment quantifying variance across multiple runs (e.g., reporting std over 5 seeds for key metrics) is necessary for reproducibility and would help reviewers trust the results—especially for the segmentation metrics where differences between methods can be small.

6. **Temporal extension for video**: The video experiments (Section 4.1.2) apply per-frame k-means independently. Adding a variant that enforces temporal consistency (e.g., re-seeding k-means from previous frame centroids) would be a natural and inexpensive extension, and results on SSv2—where object tracking matters—would likely show meaningful gains.

7. **Evaluation on object-centric VQA tasks**: CLEVR-VQA, CLEVR-Hans-3, or PTR are natural evaluation settings where object-centric representations should provide measurable advantages over CLS tokens. This would directly bridge the gap between the representation quality claims and practical usefulness.

---

### Deeper Analysis Needed

1. **Why does DINOv2 cluster so well? A geometric analysis of the feature space**: The paper attributes its success to "rich semantic features" but this is qualitative. Computing feature-space metrics like inter-cluster distance, silhouette scores, or anisotropy (as functions of backbone family and size) would explain *why* DINOv2 ≫ MAE ≫ CLIP in Table 4, and would turn an empirical observation into a principled insight about what SSL objectives produce clusterable features.

2. **Formal connection between k-means centroids and slot attention**: The paper notes that slot attention is "a soft and parametric approximation of k-means" but never formalizes this. A brief theoretical treatment showing that in the infinite-samples limit (or under certain distributional assumptions), slot attention converges to k-means would strengthen the motivation for why centroids should preserve backbone embedding quality while slots do not.

3. **Ablation of the aggregation function (attention pooling)**: The attention pooling classifier introduces learnable parameters and could itself be contributing to the performance delta vs. SPOT. An ablation comparing: (a) mean pooling over centroids, (b) max pooling, (c) the CLS token alone, and (d) the proposed attention pooling would isolate how much comes from the representation vs. the downstream classifier architecture.

4. **Sensitivity analysis of K-progression and optimal K selection**: The paper uses a fixed geometric progression K ∈ {1, 2, 4, 8, …} but never justifies why geometric vs. arithmetic vs. adaptive. An ablation on Places205 or COCO (the tasks where multi-scale matters most) varying the progression type and range would sharpen the methodological claim and give practitioners actionable guidance.

5. **Layer-wise ablation for which transformer block features to cluster**: Appendix A.1 compares Q/K/V/Output at the *last* layer but doesn't explore whether intermediate layers produce better clusters for certain tasks. For segmentation, intermediate layers (as in DINO attention maps) might produce spatially sharper boundaries, while output features favor semantic coherence. This would yield insight analogous to classic CNN layer analysis.

6. **Analysis of failure modes and boundary conditions**: When does k-means produce semantically incoherent clusters? A systematic analysis on images with fine-grained texture patterns (e.g., DTD), images with very many objects (crowded pedestrian scenes), or heavily occluded scenes would define the envelope of the method and guide future work.

---

### Untapped Applications

1. **Visual token compression for vision-language models (VLMs)**: This is perhaps the most impactful near-term application. Using k-means centroids as the visual token sequence fed to an LLM (instead of hundreds of patch tokens) would reduce compute by 10–100×. A proof-of-concept experiment fine-tuning LLaVA or Flamingo with k-means tokens vs. all patches on VQAv2 or MMMU would be both highly relevant to the ICLR community and practically significant.

2. **Reinforcement learning and embodied AI**: The paper mentions world models in the conclusion but does not experiment. Testing whether k-means object tokens accelerate policy learning on object manipulation tasks (e.g., in MuJoCo or RoboSuite environments with ViT-based perception) would validate the broader narrative about structured world understanding.

3. **Point cloud and 3D scene understanding**: K-means applied to 3D SSL backbones (e.g., Point-MAE, I-JEPA on multi-view) could extract 3D object-centric representations. Even a preliminary result on ShapeNet part segmentation or ScanNet would show that the clustering principle generalizes beyond 2D images.

4. **Medical image analysis**: Histopathology images and radiology scans have natural part-whole hierarchies (cell → nucleus → slide; organ → lesion → subregion). Testing on CAMELYON17 or CheXpert using pre-trained med-DINO backbones would demonstrate domain generalization without any retraining—a compelling argument for the method's flexibility.

5. **Continual/lifelong learning**: Object-centric representations could be particularly useful in continual learning settings where new object categories appear over time. Using k-means centroids as a rehearsal buffer representation (storing centroids instead of raw images) could be an elegant low-memory approach worth exploring.

---

### Visualizations & Case Studies

1. **Systematic failure gallery with diagnostic analysis**: A figure showing 10–15 failure cases organized by failure type (texture-dominated scenes, highly occluded objects, fine-grained categories, very small objects) would help readers understand the practical limitations and guide future improvements. This is more useful than additional quantitative tables.

2. **Centroid semantic drift visualization across K**: Showing how the semantic content of centroids changes as K increases from 2→4→8→16 on the same image—using both the mask overlays and nearest-neighbor retrievals from ImageNet—would make the multi-scale argument tangible and visually compelling. The part-whole hierarchy point (laptop → screen → keyboard buttons) is currently illustrated only for one K level in Figure 1.

3. **t-SNE/UMAP of centroid embeddings vs. CLS vs. patch features**: Plotting the embedding geometry for a representative subset of images on COCO or Places would show that centroids occupy semantically meaningful and well-separated regions of the embedding space, supporting the "preserved quality" claim with direct visual evidence.

4. **Temporal centroid tracking in video**: A video frame sequence (from SSv2 or Kinetics) with centroids colored by identity across frames would visually demonstrate whether k-means centroids implicitly track objects over time—and would directly motivate the temporal consistency extension suggested above.

5. **Centroid-based image editing / compositing**: A qualitative demonstration of swapping centroids between images (e.g., replacing the "sky" centroid from one image with another) as a form of structured image editing would illustrate the interpretability claim in a vivid and memorable way, and would resonate strongly with ICLR's representation learning audience.

---

### Natural Next Steps

1. **Learned refinement stage on top of k-means**: Since k-means produces noisy boundaries (acknowledged in Table 3), a lightweight boundary refinement module (e.g., a thin CNN border predictor trained on SAM-generated pseudo-labels) could close the gap to specialized methods while retaining the no-training-on-task-data property for the upstream features. This hybrid approach would be the direct architectural successor to this paper.

2. **K-means as warm initialization for slot attention**: Rather than framing k-means as an *alternative* to slot attention, using k-means centroids to initialize slot attention's query vectors could accelerate convergence and improve quality. Demonstrating that the DINOv2 + k-means initialization fixes the convergence failure documented in Appendix A.3 (Table 7) would be an immediately impactful follow-up that also resolves the paper's own open question.

3. **Standardized object-centric representation benchmark**: The paper calls for "standardized models and extended benchmarks" in its conclusion—actually building this would be a major community contribution. A benchmark covering classification, segmentation, retrieval, and VQA with standardized evaluation protocols, splits, and metrics would resolve the evaluation fragmentation that the paper correctly diagnoses as a problem in the field.

4. **Hierarchical and compositional structure learning**: K-means currently captures multi-scale but not *hierarchical* structure (wheel is part of car, but this relationship is implicit). Adding a lightweight tree-structured aggregation step—grouping fine-grained centroids into coarser ones in a learnable way—would produce proper part-whole hierarchies and connect to the cognitive science motivation cited in the introduction (Spelke & Kinzler, Whitehead).

5. **Adaptive K selection per image**: The paper uses a fixed geometric progression of K regardless of scene complexity. A learned or heuristic method for selecting K adaptively (e.g., based on the silhouette score or the within-cluster variance relative to the total variance) would make the method more practical in deployment settings and addresses the "how many objects?" question that the paper currently sidesteps by using high-recall protocols.

# Report: Potentially Missed Related Work
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

1. **Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning** — Kucuksozen et al. (2025, CVPR).
   Why potentially missed: This work proposes a hierarchical attention-based clustering module that extracts object-centric representations from backbone features without additional training—directly paralleling the paper's multi-scale k-means approach on off-the-shelf features. Both methods share the core insight of avoiding fine-tuning while achieving hierarchical object decomposition, making this a highly relevant contemporary baseline.

2. **Eigen Aggregation Learning for Object-Centric Unsupervised Semantic Segmentation (EAGLE)** — Authors not specified (2024, arXiv).
   Why potentially missed: This method applies k-means clustering on feature eigenvectors from pretrained backbones for object-centric representations and unsupervised segmentation. It emphasizes preserving semantic features for downstream tasks, aligning closely with the paper's motivation of maintaining backbone embedding quality while extracting structured object representations.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [3.0, 5.0, 3.0, 3.0]
Average score: 3.5
Decision: Reject
Binary: Reject

=== CALIBRATION EXAMPLE 3 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

### Title & Abstract

The title "EDM2+: Exploring Efficient Diffusion Model Architectures for Visual Generation" is accurate and honest about the paper's scope. The abstract correctly identifies the problem (underexplored architecture design, computational burden of large diffusion models), the approach (systematic redesign of EDM2's U-Net), and the headline result (2× compute reduction without quality loss, SOTA FID on ImageNet 64×64).

One concern: the abstract claims "state-of-the-art FID on the hallmark ImageNet benchmark." However, inspection of Table 3 reveals that EDM2+-S matches EDM2-S at FID=1.58, not strictly surpasses it. The SOTA claim rests on the combination with Autoguidance (FID=1.00 vs. 1.01), a difference of 0.01 that falls within plausible evaluation noise—especially given the authors' own admission that EDM2+ computes FID only once, whereas EDM2 reports the minimum of three runs. The abstract should be more precise about what "SOTA" means here.

---

### Introduction & Motivation

The motivation is coherent: U-Net architectures for diffusion models have received less rigorous treatment than training/sampling, and large models are computationally expensive. These are real gaps.

The contributions are clearly enumerated. However, the third bullet—"sets new standard to the generative modeling field"—is self-aggrandizing. The paper matches EDM2's FID with half the FLOPs; it does not clearly surpass it on quality. The introduction does not over-claim dramatically, but the FID=1.00 result depends on Autoguidance (a separate, concurrent method), which should be more explicitly flagged upfront as a combination result rather than a standalone architectural contribution.

The claim that "the underlying network architecture design remains on a shaky empirical footing" is too strong—EDM2 itself performed extensive architecture ablations. A more accurate framing is that the micro-architecture design space (block-level design, embedding placement) remains underexplored relative to macro decisions.

---

### Related Work

The coverage of diffusion training and sampling is competent and appropriately scoped. The architecture engineering subsection correctly positions the work relative to U-Net variants, DiT, and DiffuSSM. The distinction from post-hoc compression methods (pruning, quantization, caching) is important and clearly made.

**Potential gap:** There is no mention of concurrent work on efficient generative architectures beyond DiffuSSM. While this may be a legitimate double-blind limitation, works on structured/efficient blocks in generative models (e.g., from the GAN literature) could be more thoroughly engaged. More importantly, Mamba/SSM-based diffusion models and hybrid CNN-attention models specifically designed for efficiency (not just classification networks like MobileNet) are only lightly discussed.

The positioning against DiT is fair, and the claim that EDM2 outperforms DiT-class models on ImageNet is supported by Table 3.

---

### Method / Approach

**Strengths:** The step-by-step ablation narrative (CONFIGS A→G*) is the paper's strongest asset. Each design decision is presented with before-and-after FID, Mparams, and GFLOPs, enabling a clear causal reading of the contribution of each component. The final block (Eq. 9 / CONFIG G*) is well-motivated.

**Concerns:**

1. **Information bottleneck invocation (§3.4) is informal and under-justified.** The authors claim that injecting the condition embedding into a narrow (bottleneck) feature map is more effective "from the viewpoint of information bottleneck theory." However, they do not formally use or cite information bottleneck (Tishby et al.) — the argument is purely intuitive. This is fine as engineering intuition, but calling it "information bottleneck theory" overstates the theoretical grounding. A more honest framing would be that narrower injection reduces parameter count while providing more targeted modulation — which is what the experiments actually show.

2. **Statistical significance of ablation FID differences.** Many FID differences in Tables 1–2 are in the range 0.01–0.08 (e.g., CONFIG G at 1.66 vs. CONFIG G* at 1.58). Given that FID is a stochastic metric and the authors compute it only once, these differences may not be statistically meaningful. No confidence intervals are provided. The progression from 1.66 → 1.58 across the embedding bottleneck step is the most convincing delta, but it still merits uncertainty quantification.

3. **The "fade in / fade out" spatial mixing analysis (§3.3) draws a fairly strong conclusion from limited evidence.** The conclusion that "exactly one depthwise convolution is optimal" is derived from comparing a small number of configurations on a single benchmark. The mechanism (interaction with self-attention) is plausible but not rigorously verified (e.g., by ablating self-attention jointly).

4. **Self-attention is dismissed prematurely.** The authors state "since self-attention does not occupy a majority of the computation in our context, we omit to discuss it in the sequel." However, the interaction between the self-attention modules and the redesigned convolutional blocks is non-trivial. The neutral effect of 7×7 kernels is attributed to self-attention capturing long-range dependencies (CONFIG E⋄), implying a meaningful interplay — yet the attention module design itself is never ablated.

5. **Training dynamics not discussed.** EDM2 places significant emphasis on magnitude-preserving operations, weight standardization, and careful initialization. The substitution of regular convolutions with depthwise separable blocks and MBConv-style structures meaningfully changes the per-layer weight dimensions and gradient flow. The paper does not discuss whether these changes affect training stability or require re-tuning of the magnitude-preserving framework.

---

### Experiments & Results

**Strengths:** The main result—same FID as EDM2 at ~50% of FLOPs and ~55% of parameters—is consistent across S, L, and XL model sizes (Table 3), which is a compelling demonstration of the architectural contribution.

**Concerns:**

1. **Single benchmark, single resolution.** All ablations and all main results are on ImageNet 64×64 pixel-space diffusion. There are no experiments at 128×128, 256×256, or in latent space. This is a significant limitation at ICLR, where generality of findings is expected. The authors acknowledge latent-space diffusion as future work, but the design conclusions (channel mixing preference, embedding bottleneck) may not transfer to higher resolutions where spatial mixing could matter more.

2. **Missing M-sized model in Table 3.** EDM2 reports S, M, L, XL variants. EDM2+ reports only S, L, XL. The omission of the M-sized variant is unexplained and potentially suspicious — it should be included or the omission explicitly justified.

3. **FID evaluation protocol asymmetry.** The authors compute FID once; EDM2 reports the minimum of three runs. This is acknowledged, but it creates a fundamental confound: EDM2+-S at FID=1.58 "matches" the published EDM2-S at 1.58, but the authors' own reproduction of EDM2-S gives 1.63 (CONFIG A). This 0.05 gap between reproduction and original is larger than several of the ablation deltas being discussed, and it suggests meaningful evaluation variance that undermines the claimed equivalence.

4. **Autoguidance-based SOTA claim.** The headline "FID=1.00, SOTA" result comes from combining EDM2+ with Autoguidance (Karras et al., 2024a), where the marginal improvement over EDM2+Autoguidance is just 0.01 FID (1.00 vs. 1.01). This difference is smaller than evaluation noise given single-run FID. Framing this as "SOTA" is questionable.

5. **No training efficiency reported.** Table 4 profiles inference time and memory. However, for researchers choosing an architecture to train from scratch, training time per iteration and total training cost are equally important. The doubled channel width in some intermediate configs (e.g., CONFIG B) and the changes in pointwise convolution counts affect training FLOPs, which are not reported.

6. **Qualitative results are minimal.** Figure 4 shows uncurated samples from EDM2+-XL without guidance at 64×64. At this resolution, differences between models are difficult to assess qualitatively. No failure mode analysis or per-class quality inspection is provided.

7. **Error bars / statistical significance are entirely absent** from both ablation tables and the main comparison table. For a paper whose key contribution rests on FID comparisons with deltas of 0.01–0.08, this is a significant methodological gap.

---

### Writing & Clarity

The paper is generally well-written and the ablation narrative is easy to follow. The block architecture figures (Figures 2 and 3) are useful for understanding the structural changes, though the OCR-extracted text suggests the figures themselves carry significant information that may not fully transfer in text form.

One genuine clarity issue: The paper uses "embed bottleneck" to describe two related but distinct ideas — (a) reducing the output dimensionality of the embedding network, and (b) repositioning where the embedding is injected into the denoising network. These are conflated in §3.4, making it harder to isolate which effect is responsible for the observed gains. A cleaner decomposition would strengthen the analysis.

The description of CONFIG G* in Eq. (9) should be more explicitly contrasted with CONFIG G (Eq. 8) in prose — the reader must track which equation corresponds to which config through the narrative.

---

### Limitations & Broader Impact

The authors briefly acknowledge: (1) practical runtime needs further optimization for real-time deployment, (2) per-block design choices are unexplored, and (3) latent-space diffusion is future work.

**Fundamental limitations not discussed:**

1. **Resolution generality.** All claims are validated solely at 64×64. The preference for channel mixing over spatial mixing may reverse at higher resolutions where spatial structure matters more. This is perhaps the most important missing discussion given that modern applications operate at 512×512 or higher.

2. **Task generality.** All experiments are class-conditional on ImageNet. Whether the architectural insights transfer to text-to-image, video, or 3D generation — the applications motivation in the introduction — is entirely untested.

3. **Interaction with training dynamics.** As noted above, the interplay between the MBConv-style redesign and EDM2's magnitude-preserving training framework is not examined. It is unclear whether the architecture would be robust without this specific training regime.

4. **No broader impact section.** ICLR expects some discussion of broader societal implications, even for an architecture paper. This is entirely absent.

---

### Overall Assessment

This paper makes a genuine and practically useful contribution: by systematically applying depthwise-separable convolutions with MBConv-style blocks and repositioning condition embedding injection to bottleneck layers, the authors achieve approximately 2× reduction in FLOPs and parameters relative to EDM2 while matching its FID on ImageNet 64×64. The step-by-step ablation methodology is admirably transparent and the consistency across model scales (S, L, XL) lends credibility to the findings. However, the paper has several important weaknesses that limit its ICLR impact: all results are restricted to a single benchmark (ImageNet 64×64 pixel-space), making generalization claims tenuous; the FID evaluation protocol is asymmetric with EDM2, with single-run measurements for deltas as small as 0.01; statistical uncertainty is entirely unaddressed; the M-sized model is absent without explanation; and the most prominent SOTA claim (FID=1.00) depends on a 0.01 difference attributable to Autoguidance rather than the architectural contribution. The theoretical framing (information bottleneck, channel-vs-spatial mixing hypothesis) is plausible but informal. As an engineering contribution with solid ablations, this paper clears a bar for publication, but the narrow experimental scope and missing uncertainty quantification would need to be substantially addressed to meet ICLR's standards for generality and rigor. Specifically, experiments at higher resolution and/or in latent space, error bars on FID comparisons, inclusion of the M-sized variant, and a more careful treatment of the evaluation protocol asymmetry are essential revisions.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper presents EDM2+, an efficient U-Net architecture for diffusion models that achieves comparable generation quality to EDM2 while reducing computational complexity by 2×. The authors systematically explore design choices along two axes—layer placement and module interconnection—finding that depthwise separable convolutions with linear bottlenecks and repositioning the condition embedding to narrow feature maps (the "embed bottleneck") yield significant efficiency gains. The proposed architecture achieves state-of-the-art FID of 1.00 on ImageNet 64×64 when combined with autoguidance.

### Strengths
1. **Systematic ablation study**: The paper provides a thorough step-by-step exploration of architectural modifications (Table 1 and 2), with clear intermediate results showing the impact of each design choice. This allows readers to understand which changes matter and why.

2. **Strong empirical results**: EDM2+ matches EDM2's FID (1.58 vs 1.58 for S-sized models) with approximately 50% of the FLOPs (50 vs 102 GFLOPs) and parameters (154M vs 280M). With autoguidance, achieves SOTA FID of 1.00.

3. **Practical efficiency gains**: Table 4 demonstrates real-world speedups—52% faster CPU latency, 30% higher GPU throughput, and 13% less GPU memory—making the improvements meaningful for deployment.

4. **Clear architectural visualizations**: Figures 2 and 3 clearly illustrate the progressive architectural changes, making it easy to understand the modifications at each step.

5. **Reproducibility considerations**: The authors use the identical training recipe and data processing strategy as EDM2 to isolate architectural effects, and promise code release upon acceptance.

### Weaknesses
1. **Limited evaluation scope**: The paper only evaluates on ImageNet 64×64, which is a relatively low resolution benchmark. No experiments on higher resolutions (e.g., 256×256, 512×512), text-conditional generation, or latent diffusion models are provided, limiting the broader applicability claims.

2. **Modest architectural novelty**: The core techniques—depthwise separable convolutions, linear bottlenecks—are borrowed from well-established efficient network literature (MobileNet, Xception). While the "embed bottleneck" insight is interesting, the overall contribution is more engineering-driven than conceptually novel.

3. **No comparison to Transformer-based architectures**: The paper acknowledges DiT-based models (Sora, Stable Diffusion 3) but provides no empirical comparison to Diffusion Transformer or other attention-based architectures for fairness.

4. **Limited evaluation metrics**: Only FID is reported. Additional metrics such as precision/recall, IS, or human evaluation would strengthen the quality assessment.

5. **Single FID measurement**: The authors note they compute FID only once due to computational constraints, while EDM2 reports the minimum of three runs. This introduces measurement variance concerns and potentially puts the method at a disadvantage in fair comparison.

6. **Lack of theoretical grounding**: The design choices are justified empirically but lack deeper theoretical analysis of why channel mixing should be prioritized over spatial mixing for diffusion models specifically.

### Novelty & Significance
The novelty lies primarily in the systematic application and adaptation of efficient CNN techniques to the diffusion model architecture domain, with the "embed bottleneck" being the most original insight. The significance is strong from a practical standpoint—the 2× efficiency gain without quality degradation is valuable for real-world deployment. However, the work would benefit from broader validation across different settings and resolutions to fully establish its significance.

### Suggestions for Improvement
1. **Expand evaluation to higher resolutions and latent diffusion**: Validate the architecture on ImageNet 256×256 or apply it to latent diffusion frameworks (Stable Diffusion) to demonstrate broader applicability.

2. **Add comparison with Transformer-based diffusion models**: Include DiT or related architectures in the comparison table to help readers understand the efficiency-quality trade-off between CNN and Transformer backbones.

3. **Include additional evaluation metrics**: Report precision/recall, Inception Score, and potentially conduct human evaluation to provide a more comprehensive quality assessment.

4. **Provide theoretical or empirical analysis of the spatial vs. channel mixing trade-off**: Visualization of learned features or ablation on the attention modules would help explain why channel mixing becomes more important under constrained compute budgets.

5. **Address FID measurement variance**: Either report multiple runs with standard deviation, or explain how the single-run evaluation affects confidence in the results.

6. **Discuss failure cases or limitations more explicitly**: The current limitation section is brief; discussing scenarios where the efficiency gains might come at the cost of other desirable properties would strengthen the paper's credibility.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments
1. **Higher-resolution benchmarks (ImageNet 128×128 and 256×256):** The paper exclusively demonstrates results at 64×64 pixels. ICLR reviewers will almost certainly ask whether the 2× efficiency gain persists at higher resolutions where the spatial-vs-channel mixing tradeoff may shift materially — larger feature maps make spatial operations proportionally more expensive, so the findings could be even stronger or could break down.

2. **Latent diffusion setting (LDM/Stable Diffusion style):** The paper acknowledges this as future work, but given that virtually all production-scale systems now operate in latent space (SD3, SDXL, etc.), even a preliminary experiment on ImageNet 256×256 in latent space would dramatically broaden the paper's impact and demonstrate that the depthwise-separable + embed-bottleneck insight transfers to the dominant practical regime.

3. **EDM2+-M size point:** Table 3 shows S, L, and XL variants but skips the Medium (M) size that appears in the EDM2 comparison. Including M would complete the scaling curve and demonstrate that the Pareto efficiency gain holds uniformly across all sizes — a minor but cleanly fixable gap.

4. **Additional evaluation metrics beyond FID-50K:** Precision/Recall (Kynkäänniemi et al., 2019), IS, and FD-DINOv2 (Stein et al., 2023) alongside FID would paint a fuller picture. Notably, the paper itself cites Stein et al. to argue that Inception-based FID favors GANs; using the DINOv2-based metric would both address this bias and show EDM2+'s advantage under a fairer measurement.

5. **Training-compute Pareto curve:** All efficiency results focus on *inference* FLOPs. A plot of FID vs. total training GPU-hours for EDM2-S vs. EDM2+-S would show whether the smaller model also trains faster to a given quality level, which is equally important for practitioners.

6. **Ablation on the self-attention component:** The paper explicitly sets aside attention ("we omit to discuss it in the sequel") but this is a non-trivial chunk of compute at low resolutions. A brief ablation showing the sensitivity of the final design to attention placement/count would seal the story on computational allocation.

7. **Stochastic sampling results for EDM2+:** Table 3 leaves the stochastic columns blank for EDM2+. Since EDM2 does not report stochastic results either but references EDM1 stochastic as a meaningful baseline, showing EDM2+ in stochastic mode (or explaining why it is not applicable) would close a comparison gap.

8. **Combination with distillation methods:** The paper claims orthogonality with quantization, pruning, and distillation, but a single experiment combining EDM2+ with consistency distillation or progressive distillation (e.g., achieving 1–4 step generation) would make the claim concrete and show multiplicative efficiency gains.

---

### Deeper Analysis Needed
1. **Rigorous theoretical grounding for the embed-bottleneck insight:** The paper invokes "information bottleneck theory" informally. A more precise analysis — e.g., measuring the mutual information between condition embedding and denoising network activations across CONFIGs, or framing it in terms of the Fisher information of condition signals — would elevate this empirical observation into a principled design rule transferable beyond this specific architecture.

2. **Training dynamics comparison:** Plotting gradient norms, loss curves, and EMA-adjusted validation loss as a function of training images for EDM2 vs. EDM2+ would reveal whether the efficiency gain is purely architectural or also accompanied by better optimization trajectories. EDM2's core thesis is about training dynamics, so comparing these is naturally expected by reviewers familiar with the baseline.

3. **Theoretical justification of spatial vs. channel mixing hypothesis:** The paper argues that "channel mixing outweighs spatial mixing under a limited compute budget" for generative modeling, but this is supported only by ablation results. An analysis connecting this to the spectral structure of natural images, or to the denoising objective's dependence on noise level (where high-frequency spatial correlations matter more at low noise), would make this a transferable principle rather than an empirical curiosity.

4. **Sensitivity analysis across expansion ratios and kernel sizes:** The paper tests e=4 and e=6, and k=3 and k=7. A broader sweep (e ∈ {2,3,4,6,8}, k ∈ {3,5,7,9}) would produce a cleaner Pareto-optimal curve and substantiate the chosen hyperparameters more definitively.

5. **Per-noise-level analysis:** Diffusion models denoise across a wide range of noise levels (σ), and the optimal spatial-vs-channel mixing balance may differ between coarse (high σ, semantic) and fine (low σ, textural) denoising steps. Analyzing which blocks are most active at different noise levels (e.g., via activation magnitudes) would deepen the architectural understanding considerably.

6. **Encoder vs. decoder asymmetry:** The conclusion mentions this as future work, but even a preliminary experiment distinguishing whether the efficiency gains primarily come from encoder blocks, decoder blocks, or both equally would be a valuable in-paper finding rather than a future-work note.

---

### Untapped Applications
1. **Video diffusion models:** The depthwise-separable approach transfers naturally to 3D (spatiotemporal) convolutions, where computational savings would be even larger (3D depthwise conv is far cheaper relative to 3D regular conv). Demonstrating even a toy result on UCF-101 or similar would position this as a framework contribution, not just an ImageNet result.

2. **Text-conditional generation:** Applying EDM2+ to a text-conditioned setup (e.g., using a frozen CLIP encoder for conditioning) would show that the embed-bottleneck principle generalizes beyond the simple scalar (class label + noise level) conditioning of the current work, which matters for practical relevance at ICLR 2025.

3. **Medical and scientific imaging:** Domains like radiology (chest X-rays, MRI) or remote sensing often demand high-quality generation with strict compute budgets (edge deployment, privacy constraints). EDM2+'s efficiency profile makes it particularly attractive here, and a single experiment on a medical dataset would broaden the community impact.

4. **Diffusion-based super-resolution or inpainting:** These downstream tasks are natural testbeds for architecture efficiency — they often run at higher resolution where the spatial-vs-channel tradeoff is more pronounced — and would demonstrate generalization beyond unconditional class-conditional generation.

---

### Visualizations & Case Studies
1. **Feature map activation visualizations across CONFIGs:** Plotting the channel-wise activation statistics (e.g., mean/std of feature maps) for EDM2 and EDM2+ at each layer would visually demonstrate why the embed-bottleneck improves information flow — readers could literally *see* the condition signal being more cleanly routed.

2. **FID vs. FLOPs training curves (not just final numbers):** Showing how FID improves over training steps for each model size in both EDM2 and EDM2+ families, plotted against cumulative training FLOPs, would provide a compelling "efficiency during training" story to complement the inference-time results.

3. **Failure mode and class-conditional generation diversity analysis:** Uncurated samples are shown for XL without guidance, but a per-class breakdown (e.g., showing the 10 worst-performing ImageNet classes by FID or FID-per-class proxy) would help readers understand *when and why* the architecture succeeds or struggles — particularly useful for practitioners.

4. **Side-by-side qualitative comparison EDM2 vs. EDM2+:** Currently, Figure 4 shows only EDM2+ samples. Placing EDM2 and EDM2+ samples side-by-side (same class, same seed) would make the "on-par quality at half the compute" claim immediately visually verifiable.

5. **Architectural diagram of the full U-Net with block-type annotations:** The paper shows individual block diagrams (Figures 2–3) but not the full macro architecture with block placements, resolution levels, and attention positions. A single annotated U-Net diagram for EDM2+ would make the architecture fully reproducible from the paper alone, which ICLR reviewers value highly for reproducibility.

---

### Natural Next Steps
1. **Neural Architecture Search (NAS) over the identified design axes:** The paper manually explores layer placement and module interconnection. Given the promising design space defined (expansion ratio, kernel size, embedding injection position, depthwise count), a differentiable NAS procedure could find a globally optimal per-block configuration, potentially yielding another 10–20% efficiency gain on top of EDM2+.

2. **Extending the embed-bottleneck principle to DiT architectures:** The core insight — that condition signals injected at a lower-dimensional bottleneck are more effective — should translate to the adaLN modulation in DiT, where the condition embedding is currently injected into the full-width feature dimension. A brief experiment on DiT with "bottleneck adaLN" would establish whether this is a universal design principle.

3. **Combining with flow matching / rectified flow:** The architecture is currently paired with EDM training; applying EDM2+ blocks within a rectified flow (SD3/Flux style) or consistency model framework would test robustness of the design choices and position the architecture for the trajectory that appears to be dominating production systems in 2025.

4. **Per-block heterogeneous design:** Currently all blocks share the same CONFIG G* design. The natural next step — also noted in the conclusion — is to allow encoder blocks, decoder blocks, and blocks at different resolutions to use different configurations. This could be studied systematically by measuring per-block sensitivity (e.g., via structured pruning signals) before committing to a fixed design.

5. **Efficient attention integration:** The paper deliberately excludes attention from its analysis. A follow-up study specifically on attention efficiency (linear attention, local attention, FlashAttention variants) combined with the EDM2+ convolutional blocks would complete the architectural story and address the remaining compute bottleneck for high-resolution generation.

# Report: Potentially Missed Related Work
Looking at the paper's references and citations carefully:

## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

1. **AT-EDM: Attention-Driven Training-Free Efficiency Enhancement of Diffusion Models** — Anonymous (2024, arXiv:2405.05252).
   Why potentially missed: This work proposes a training-free token pruning method for diffusion models, achieving significant FLOPs reduction (~38.8%) while maintaining generation quality. While the paper mentions that its architectural redesign is orthogonal to post-hoc methods like pruning, quantization, and caching, AT-EDM specifically addresses efficiency in attention-heavy diffusion architectures through a complementary approach. The authors discuss these complementary techniques but do not cite this recent training-free efficiency method.

---

**Removed from consideration:**

- **EDM2** — Already cited extensively as the direct baseline (Karras et al., 2024b throughout the paper, including Reference 37).

- **Efficient Diffusion Models without Attention (DiffuSSM)** — Already cited in Section 2.3 and Reference 87: "Subsequently, DiffuSSM (Yan et al., 2024) supplants the attention mechanism of DiT with the State Space Model (SSM) blocks to promote the efficiency."

- **EDT** — Tangentially related; focuses on DiT architectures while this paper explicitly focuses on U-Net/CNN-based architectures. The paper already discusses DiT as an alternative paradigm.

- **EMDM** — Tangentially related; addresses motion/video generation rather than static image synthesis, representing a different problem domain.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [5.0, 6.0, 6.0, 3.0]
Average score: 5.0
Decision: Reject
Binary: Reject

=== CALIBRATION EXAMPLE 4 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Self-Supervised Grid Cells Without Path Integration" is largely accurate but carries a subtle misnomer: the feedforward model is trained using explicit Euclidean distances between Cartesian coordinates as learning targets. In standard ML usage, "self-supervised" implies pseudo-labels derived from the data itself without external ground-truth supervision. Here the distances *are* the external labels. This is arguably a supervised objective applied to spatial data, not self-supervised learning in the conventional sense. The authors should either defend this terminology or adjust it.

The abstract accurately summarizes the two-model comparison, the emergence of grid-like representations, and the pruning result implicating band cells in path integration. However, the claim that "distance preservation and high population capacity is a more likely candidate task for learning grid cells in artificial neural networks" is somewhat undercut by the fact that the model is specifically constructed *to* do distance preservation — this is more circular than the abstract implies. The abstract does not acknowledge the critical limitation that all main findings appear to rest on single training runs.

---

### Introduction & Motivation

The motivation is strong and well-situated. The gap — that prior path-integration models are too complex to isolate which cell type actually drives path integration — is genuine and clearly articulated. The citation of Nayebi et al. (2021) and Schøyen et al. (2023) to motivate re-examination of grid cells as path integrators is appropriate and fair.

The contributions are stated with reasonable accuracy: (1) a minimal two-term loss that produces grid-like representations, (2) grid emergence without path integration in FF networks, (3) pruning evidence implicating band cells in path integration. The paper does not obviously over-claim at the introduction stage, though phrases like "questions the role of grid cells as neural path integrators" will need the experiments to bear significant weight.

One gap: the introduction does not explain why the L1 capacity term (rather than, say, L2) is the principled choice. Given that L1 vs. L2 turns out to be crucial (FF networks fail under L2), this design choice warrants motivation in the introduction, not just in an appendix.

---

### Related Work

Coverage of the immediately relevant normative modeling literature (Schaeffer et al. 2023, Xu et al. 2022, Sorscher et al. 2022, Dorrell et al. 2022, Dordek et al. 2016) is thorough. Table 1 provides a useful structured comparison. The authors fairly note where band-like cells appear in other models and discuss why those architectures may suppress or obscure them.

One meaningful omission: attractor network models (Burak & Fiete 2009 is cited for path integration, but the broader continuous attractor network literature, including models like Fuhs & Touretzky 2006, is not engaged). Since a central claim involves band cells driving path integration via excitation-inhibition shifts — a mechanism consistent with attractor dynamics — a brief discussion of how the paper's connectivity findings (Fig. 3b) relate to this prior mechanistic literature would strengthen the positioning.

The paper also does not engage with Redman et al. (2024) in the main text until Appendix A.7. Since Redman et al. directly examines what RNNs actually path integrate using, this citation deserves mention in the main Related Work discussion, not just an appendix.

---

### Method / Approach

**Loss function (Eq. 1):** The two-term objective is clearly stated. The Gaussian envelope over spatial distance is a natural formulation for local distance preservation, and the connection to conformal isometry (Xu et al. 2022) is acknowledged with a pointer to Appendix A.9. However, the sampling procedure for pairs (t, t′) is underspecified in the main text. Are pairs sampled uniformly over the batch regardless of temporal proximity? Or are they trajectory-adjacent pairs? Since the Gaussian envelope effectively downweights distant pairs anyway, this may not matter much — but it should be stated explicitly for reproducibility.

**L1 capacity term (Eq. 2):** The paper argues that L1 promotes maximally distributed, co-active representations, and provides a geometric illustration (Fig. 1b). This is plausible but the theoretical argument is informal. Why does L1 specifically favor hexagonal rather than square or other periodic tilings? An argument connecting the capacity term's geometry to hexagonal packing (even informally) is missing. The empirical demonstration in Appendix A.8 that L2 fails for FF networks is important — it implies the main finding is heavily dependent on this specific regularization choice — yet the theoretical justification for *why* L1 works and L2 doesn't is left almost entirely to the empirical observation.

**Velocity pruning (Section 3.3):** The pruning approach is conceptually clean: silence velocity input to select units and measure deviation from unpruned trajectories (Eq. 4). However, there is a significant confound that the paper partially acknowledges but does not resolve: Fig. 3b shows that velocity input weights (*W_in*) *decrease with grid score*. This means that by construction, band cells receive stronger velocity projections and grid cells receive weaker ones. Pruning velocity input to band cells therefore removes a stronger signal, which trivially predicts larger deviations. The pruning analysis in Fig. 2 thus conflates "band cells are functionally responsible for path integration" with "band cells happen to have larger velocity weights, so removing their velocity input has a larger effect." The appropriate control would be to equate the magnitude of pruned velocity weights across high- and low-GS subpopulations, or to use a metric that is not directly confounded with weight magnitude.

**Grid score threshold (0.15):** The cutoff used to classify cells as "band-like" (grid score < 0.15) vs. "grid-like" is central to the pruning analysis — it determines which n=29 cells are labeled "band cells" and drives the main conclusion. This threshold appears to be chosen post-hoc based on visual inspection of the score distribution (Fig. 2b) and is not justified quantitatively. The conclusions should be shown to be robust across a range of cutoffs.

**Architecture choice and FF advantage:** The FF network receives direct Cartesian coordinates as input, while the RNN must infer position from cumulative velocity. This architectural asymmetry provides the FF network a large representational advantage for the spatial distance preservation task. The finding that both achieve comparable grid scores does not therefore demonstrate that path integration is irrelevant to grid emergence — it could simply show that two different inputs (position vs. velocity integral) can support the same objective when that objective is loss-driven. A fairer test would hold the information content of inputs more constant.

---

### Experiments & Results

**Multiple runs and statistical reliability:** This is the most serious methodological weakness. The main results (ratemaps in Fig. 1, grid score distributions, the count of exactly 29 band cells, and the pruning analysis in Fig. 2) all appear to come from *single* trained networks. ICLR reviewers will rightly demand that key quantitative claims be replicated across multiple random seeds with confidence intervals or standard errors. If the band cell count varies substantially across runs, or the Pearson correlations in Fig. 2d shift considerably, the conclusions are fragile. The paper provides only one demonstration that "band structures appear linked to path integration" and "grid cells are less important" — without variance estimates across runs.

**Pruning result interpretation (Fig. 2c–f):** The central pruning finding is that pruning velocity inputs to the 29 low-GS (band) cells causes markedly higher path integration error than pruning equal-sized subsets of high-GS cells. This is a well-designed experiment with a reasonable null condition (random subsets from the full population). The Pearson correlations between mean grid score and PI error at each timestep (r ≈ −0.78 to −0.83, Fig. 2d) are reasonably consistent and provide a quantitative summary. However, as noted above, the confound with velocity weight magnitudes weakens the causal claim. The ISD metric (Eq. 5, Fig. 2f) is an interesting complement but the interpretation ("the neural path integrator is close to being turned off") is slightly overreached — a flat ISD when pruning band cells could also reflect that the representation collapses to a fixed point rather than genuinely "turning off" path integration.

**Generalization results (Fig. 3c):** The finding that the RNN generalizes outside the training domain (when started inside) while the FF network does not is well-demonstrated and provides a meaningful functional distinction between the models that the paper uses appropriately.

**Loss ablation:** The appendix (Fig. A1b, A6) demonstrates that both loss terms are necessary, which is a required sanity check. However, the ablation is shown only for the RNN. Showing that the FF network also requires both terms is necessary to make the minimal-model claim credible.

**Toroidal topology (Fig. 1e):** The persistence homology analysis showing 2 persistent 1D and 1 persistent 2D cocycle (consistent with a torus) is a nice corroboration. However, for the FF network, the authors restrict analysis to units with orientation in [0.4, 0.8], effectively selecting a subset. The choice of this range and its effect on the topological result should be made explicit — it's possible that the full FF population does *not* form a clean torus, which would be an important qualification.

**Hyperparameter selection:** σ=1.2 and α=0.54 are chosen from a grid search based on maximizing grid score (Fig. A1a). This risks overfitting the hyperparameters to the grid score metric itself, which is then used to evaluate the model. The paper should discuss whether qualitative conclusions (grid emergence, band cell pruning effects) hold for a range of hyperparameter values, not just the maximum-grid-score configuration.

---

### Writing & Clarity

The decision to present Results before Methods is a legitimate stylistic choice for a neuroscience-oriented paper. However, Section 2 (Results & Discussion) refers to many methodological details (normReLU, trajectory sampling, pruning masks) before they are defined in Section 3, creating significant forward-reference burden for readers.

The caption for Fig. 2 is dense and requires cross-referencing across panels c), d), e), f) to reconstruct the full pruning story. A brief narrative sentence tying the panels together would help. The figure itself is visually crowded with redundant decorative elements from the OCR (these are parser artifacts).

The geometric illustration of L1 capacity (Fig. 1b) is described well enough in text, but the connection between the geometric argument ("near the diagonal vector, with all units coactive") and why this favors hexagonal firing patterns is never made explicit — there is a conceptual gap between the capacity geometry and the emergent spatial structure.

Section 2.2 (Pattern Formation, Connectivity, and Generalizability) contains some of the paper's most interesting mechanistic content (band cells as a shifted excitation-inhibition pattern for path integration; velocity weights forming a hexagonal connectivity pattern) but this section is too brief. The mechanistic hypothesis in the last paragraph of 2.2 — that band cells integrate velocity via an excitation-inhibition shift — is a significant claim that deserves its own figure or quantitative support rather than a single sentence.

---

### Limitations & Broader Impact

The authors acknowledge the main limitations: biologically implausible Cartesian inputs, Euclidean distance metric, single environment geometry, and the lack of labeled distance signals in biology. These are significant and appropriate to note.

**Unacknowledged limitations:**

1. **Single-seed results**: As noted above, the absence of multi-seed reliability estimates is a significant methodological limitation that is not discussed.

2. **The velocity-weight confound in pruning**: Partially acknowledged in the connectivity discussion but not framed as a limitation of the pruning analysis itself.

3. **Model-to-biology inference**: The paper makes strong claims about what grid cells "do" in biological systems based on a simplified ANN model with implausible inputs. This epistemological limitation — that conclusions from a normative ANN model may not transfer to biology — is underacknowledged in the main text, though the limitations section does note the implausibility of Cartesian inputs.

4. **Single-module constraint**: Biological grid cells show multi-module organization with multiple spatial scales. The paper's model produces only one module (bimodal orientation in the FF model hints at two, but this is not resolved). Multi-scale distance preservation, which the authors argue underlies biological grid spacing variation, is not demonstrated.

5. **The "self-supervised" framing**: As noted in the title/abstract section, the learning objective uses explicit distance labels derived from known spatial coordinates, which is not self-supervised in the conventional sense. This could mislead readers familiar with contrastive or masked self-supervised learning.

---

### Overall Assessment

This paper addresses a genuine and important question in computational neuroscience — whether grid cells are primarily path integrators or spatial metric encoders — and proposes a pleasingly minimal model to investigate it. The key findings (grid-like representations emerging from distance preservation + L1 capacity alone, without path integration; band cells appearing more important than grid cells for path integration in the RNN) are scientifically interesting and connect to a live debate in the field. The paper is well-motivated, engages seriously with the prior literature, and provides meaningful comparisons (FF vs. RNN, speed ablations, out-of-bounds generalization, toroidal topology). However, the work has several weaknesses that are significant for an ICLR submission: (1) nearly all quantitative conclusions appear to rest on single training runs without variance estimates across random seeds, making it unclear how robust the specific ratio of band-to-grid cells or the pruning correlations are; (2) the pruning analysis has a structural confound — band cells by definition receive stronger velocity projections (visible in Fig. 3b), so silencing their velocity input has a larger mechanical effect, partially confounding the functional interpretation; (3) the band/grid threshold (0.15) is not justified or shown to be robust; and (4) the FF model's advantage from direct Cartesian inputs makes the FF-vs-RNN comparison less informative about path integration than it appears. These issues are collectively serious enough that the paper, in its current form, would likely require major revision to meet ICLR's standards of rigor. With multi-seed replication, better control for velocity-weight magnitude in the pruning analysis, and more careful calibration of causal claims, the core contribution could be convincing.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper investigates whether grid cells emerge due to path integration or spatial encoding requirements by proposing a minimal self-supervised model with two loss terms: local distance preservation and L1 capacity constraint. The authors compare feedforward networks (purely spatial encoding) with recurrent networks (implicit path integration) and find that both produce grid-like representations, but only the RNN develops band-like cells. Through velocity pruning experiments, they demonstrate that band cells—not grid cells—are critical for path integration, challenging the conventional view that grid cells serve as neural path integrators.

### Strengths
1. **Minimal, interpretable model**: The paper successfully demonstrates that grid-like representations emerge from just two simple ingredients (distance preservation + L1 capacity), avoiding the complexity of prior models that include multiple regularization terms, place cell decoding, or path invariance constraints. This is a valuable contribution toward understanding what is truly necessary for grid cell emergence.

2. **Clean experimental design for isolating path integration**: The FF vs RNN comparison provides a principled way to dissociate spatial encoding from path integration, and the velocity pruning methodology is a clever intervention that minimizes off-target effects compared to direct unit ablation.

3. **Comprehensive quantitative analysis**: The paper includes multiple analyses—grid scores, phase distributions, orientation histograms, persistence homology showing toroidal manifolds (consistent with Gardner et al. 2022), connectivity profiles showing short-range excitation/long-range inhibition, and out-of-domain generalization tests.

4. **Important scientific contribution**: Finding that grid cells emerge without path integration and that band cells are more important for path integration directly challenges a widely-held hypothesis in the field, making this a potentially influential contribution.

5. **Analysis of L1 vs L2 capacity constraints**: The comparison in Appendix A.8 provides useful insight into why L1 capacity promotes more grid-like representations, adding mechanistic understanding.

### Weaknesses
1. **Biological implausibility of supervision signal**: The loss function requires knowing true Euclidean distances between positions, which assumes access to labeled spatial information. The authors acknowledge this limitation but it reduces the model's relevance as a normative theory for biological grid cells, which must develop without explicit distance labels.

2. **Limited to single-module grid cells**: While the paper briefly mentions extending to multiple modules, it only demonstrates single-scale grid patterns. Biological grid cells show discrete modules with distinct spatial scales, and the paper doesn't address how this modular organization might emerge.

3. **Narrow environment testing**: All experiments use a simple square arena. Given recent work showing grid pattern deformation in complex geometries (Ginosar et al. 2023, cited by authors), testing in more naturalistic environments would strengthen claims about distance preservation.

4. **Limited comparison of pruning effects across training stages**: While Appendix A.7 shows path integration correlates with band scores across training, the main pruning experiments only examine trained networks. Understanding how the relationship between cell type and path integration develops could provide additional insight.

5. **Arbitrary grid score cutoff**: The classification of band-like vs grid-like cells uses a grid score cutoff of 0.15 without justification. Sensitivity analysis around this threshold would strengthen the conclusions.

### Novelty & Significance
The paper makes a significant novel contribution by providing a minimal sufficient condition for grid cell emergence and directly testing whether path integration is necessary. The finding that band cells, not grid cells, are critical for path integration is an important scientific claim that advances understanding of entorhinal cortex function. The work fits well with ICLR's interests in understanding neural representations and connecting neuroscience with deep learning. However, the biological relevance is limited by the use of labeled distance information during training.

### Suggestions for Improvement
1. **Address biological plausibility of the loss function**: Consider discussing how distance-preserving objectives might be implemented through local, biologically plausible mechanisms (e.g., through place cell projections or local inhibitory circuits), or frame the model more explicitly as a computational-level analysis rather than a mechanistic model.

2. **Extend to multiple modules**: Implement a multi-module version by partitioning the network or using different σ values for subpopulations, demonstrating whether the model can produce the modular organization observed biologically.

3. **Test in complex geometries**: Evaluate whether the model produces realistic grid pattern deformations in environments with barriers, curved walls, or other features that have been shown to distort biological grid cells.

4. **Add sensitivity analysis for the grid score cutoff**: Show that conclusions about band cells' importance for path integration are robust to the exact threshold used for classifying cells.

5. **Include comparison to biological data**: Compare the learned representations (grid scores, spacing distributions, phase distributions) to published biological grid cell data to strengthen claims about the model's biological relevance.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments

1. **Comparison on standard grid cell benchmarks with held-out biological data**: The paper would benefit from quantitatively comparing the learned representations to electrophysiological recordings from mEC (e.g., Hafting et al. 2005 or Gardner et al. 2022 datasets) — measuring statistics like grid score distributions, spacing ratios across modules, and orientation alignment — to directly assess biological plausibility beyond qualitative ratemap inspection.

2. **Ablation of the pruning methodology itself**: The velocity-pruning approach is novel, but the paper would be stronger with a complementary activation-knockout experiment (zeroing unit *states*, not just velocity inputs) and a comparison of whether conclusions hold under both methods. This would help rule out that residual recurrent dynamics compensate for pruned velocity inputs in ways that mask grid cell contributions.

3. **Scaling experiments on network size and cell count**: The paper uses *ng* = 256 units throughout. Experiments varying *ng* (e.g., 64, 128, 512, 1024) would clarify whether the grid/band cell ratio, the grid score quality, and the path integration dominance of band cells are robust properties or artifacts of a specific capacity regime — directly relevant to the paper's core capacity-constraint thesis.

4. **Testing the FF model with learned distance metrics (not Euclidean)**: The paper acknowledges that Euclidean distance may be a poor approximation in non-open environments, but does not test alternatives. Including a geodesic or graph-distance variant (especially in an environment with obstacles) would substantially strengthen the claim that distance preservation — not any specific distance function — is the key ingredient.

5. **Multi-module emergence via trainable ρ**: The paper mentions in Appendix A.9 that making ρ a unit-specific trainable parameter could yield multiple modules, but does not demonstrate this. A concrete experiment showing multi-scale, multi-module grid representations emerging automatically (rather than by manual partitioning) would significantly advance the paper's normative contribution.

6. **Cross-architecture replication**: Testing the core claims (grid cells without path integration; band cells dominate path integration) in at least one additional architecture — e.g., a GRU, an LSTM, or a transformer operating over positional sequences — would substantially increase confidence that the findings generalize beyond the specific vanilla RNN + normReLU setup.

7. **Biological trajectory statistics**: The paper uses Rayleigh-distributed step sizes and von Mises head directions. Running experiments with trajectories recorded from real rodents (or using standard synthetic rodent-like trajectories as in Sorscher et al.) would help establish that the emergence of grid-like representations is not sensitive to this particular motion model choice.

---

### Deeper Analysis Needed

1. **Theoretical analysis of why L1 vs. L2 capacity yields qualitatively different representations**: The paper shows empirically that L1 produces cleaner grids than L2, but offers only a geometric intuition (Fig. 1b). A more formal analysis — e.g., showing that the L1 constraint on the simplex under L2-norm normalization maximally promotes a specific angular distribution of population vectors — would give the finding real theoretical weight and make it a standalone contribution.

2. **Formal characterization of the loss landscape and convergence**: The paper trains with Adam and reports that grids emerge, but offers no analysis of convergence guarantees, the number of seeds tested, or sensitivity to initialization beyond the identity vs. random comparison in Appendix A.3. Reporting mean and variance of grid scores across multiple random seeds (at least 5–10) would strengthen confidence that results are reproducible rather than lucky optima.

3. **Why do band cells emerge specifically in the RNN and not the FF model?** The paper observes and partially explains this (path integration necessity), but a more rigorous mechanistic account is missing. For instance: does the band structure correspond to a specific attractor manifold in RNN state space? Is there a phase-space analysis (e.g., fixed-point or limit-cycle characterization) showing why the RNN settles into a mixed grid/band solution?

4. **Information-theoretic analysis of representational capacity**: The paper argues that the L1 constraint promotes "high capacity" representations, but never quantifies capacity formally. Adding a mutual information or Fisher information analysis (e.g., how much spatial information is encoded per unit, or across the population) would ground the capacity argument quantitatively and connect it to classical efficient coding theory.

5. **Sensitivity analysis for the grid score cutoff threshold (0.15)**: The band/grid classification is central to the pruning analysis, but the cutoff of 0.15 is chosen without formal justification. A robustness check showing that conclusions hold for cutoffs ranging from 0.05 to 0.30 (or using a data-driven threshold like k-means on the bimodal distribution) would substantially strengthen the interpretability of the cell-type dichotomy.

6. **Connection to continuous attractor network theory**: The RNN's learned recurrent weights show short-range excitation / long-range inhibition (Fig. 3b), a hallmark of continuous attractor networks (CANs). The paper would benefit from a formal analysis of whether the learned weights implement a CAN, what the attractor manifold looks like, and how this connects to the well-established CAN theory of grid cells (Burak & Fiete, 2009) — this would situate the work more precisely in the theoretical landscape.

---

### Untapped Applications

1. **3D spatial navigation**: Grid cells in bats fire in 3D volumetric patterns. Testing whether the same distance-preservation + L1-capacity objective in 3D input space produces face-centered cubic or other volumetric periodic patterns would be a natural and impactful extension, and would make a strong prediction for neuroscience.

2. **Non-Euclidean or graph-structured spaces**: The framework is defined for Cartesian coordinates, but the distance-preservation objective could naturally be extended to arbitrary metric spaces (e.g., graphs, manifolds, or task-defined similarity structures). Testing whether grid-like representations emerge on a torus, a sphere, or a discrete graph would open an exciting new direction connecting the work to topological data analysis and relational reasoning.

3. **Navigation in environments with walls and barriers**: Real grid cells deform their firing fields in environments with walls (O'Keefe & Burgess, 1996; Ginosar et al., 2023 are cited). Testing the model in a maze or an environment with internal obstacles — using geodesic rather than Euclidean distances — would directly test the model's ability to explain these known biological phenomena.

4. **Application to robot localization / SLAM**: The learned representations are compact, distance-preserving, and support path integration. Testing them as an internal state representation in a robot localization task (e.g., a simulated mobile robot) would demonstrate practical utility and connect the computational neuroscience work to robotics / embodied AI communities relevant to ICLR.

5. **Relational and non-spatial domains**: There is growing interest in whether grid-like codes emerge for non-spatial cognitive variables (e.g., Bellmund et al., 2018 for conceptual spaces; Whittington et al.'s TEM). Testing whether the same objective applied to abstract relational distances (e.g., in a semantic embedding space) produces periodic tile-like representations would directly probe the generality of the distance-preservation hypothesis beyond spatial navigation.

---

### Visualizations & Case Studies

1. **Phase space trajectories during path integration with and without pruning**: A visualization of RNN state-space trajectories (e.g., projected via PCA or UMAP) as the network integrates a path — with and without velocity inputs pruned from band vs. grid cells — would make the "turning off the path integrator" finding viscerally clear and more convincing than aggregate error metrics alone.

2. **Side-by-side ratemap comparison with real biological grid cells**: Showing a panel of model ratemaps directly adjacent to electrophysiologically recorded grid cells (with matched grid scores, spacings, and orientations) would strengthen the biological relevance claim and help readers unfamiliar with the neuroscience literature calibrate the quality of the learned representations.

3. **Connectivity matrix visualization as a function of phase offset**: Fig. 3b shows connectivity as a function of phase, but a full 2D heat map of recurrent weights indexed by the phases of both the source and target unit would more cleanly reveal the translational invariance of the connectivity and its relationship to the grid lattice — directly analogous to the canonical CAN connectivity kernel.

4. **Animation or time-series ratemap evolution during training**: A visualization showing how individual unit ratemaps evolve from random initialization to structured grid/band patterns over training would compellingly illustrate the self-organization process and help readers understand when and how the representations crystallize.

5. **Failure case analysis**: Showing examples of training runs that failed to produce grid-like representations (e.g., at extreme α values, or with very small/large σ) alongside an analysis of what went wrong (e.g., collapsed representations, uniform activity) would help readers understand the boundaries of the method and its robustness, making the success cases more interpretable.

6. **Decoder trajectory visualization across cell-type pruning conditions**: A panel showing example decoded trajectories (as in Fig. A7c) for the no-pruning, grid-pruned, and band-pruned conditions would provide an intuitive, qualitative counterpart to the quantitative error curves in Fig. 2c, making the band-cell-as-path-integrator finding immediately visually compelling.

---

### Natural Next Steps

1. **Incorporating head direction cells as biologically plausible velocity inputs**: As the authors note in Section 2.4, replacing Cartesian velocity with simulated head direction cell input would move the model substantially closer to biological plausibility. This is a well-defined next step — implementing a ring attractor head direction model as the velocity source — that would directly address the current limitation and strengthen the claim that the mechanism generalizes beyond clean Cartesian inputs.

2. **Learning the distance metric jointly with the representation**: The current model uses fixed Euclidean distance as the target. A natural extension would jointly learn a task-relevant metric (e.g., from reward structure or navigational experience) and the spatial representation — this would connect the work to metric learning and could explain how grid cell properties adapt to environmental geometry.

3. **Investigating the emergence of place cells as a downstream readout**: If grid cells encode a distance-preserving spatial metric, place cells might naturally emerge as sparse decoders of this representation. Building a second-stage network that learns place-cell-like readouts from the learned grid representation (without supervised place-cell targets) would create a complete, self-supervised model of the entorhinal-hippocampal circuit.

4. **Extending the pruning analysis to biological data using optogenetics-inspired virtual experiments**: The paper's core finding — that band cells, not grid cells, drive path integration — makes a sharp experimental prediction. The next step would be to design a virtual lesion protocol for existing large-scale neural recordings (e.g., Neuropixels data from mEC) that identifies band-like vs. grid-like cells and tests their differential contribution to navigational behavior, bringing the computational prediction into direct contact with biology.

5. **Multi-agent or shared representation extension**: A compelling direction would be to train multiple agents sharing a common spatial encoding objective (e.g., via a shared encoder or contrastive objective between agents at known relative positions). This would probe whether the grid code is an emergent property of *any* system needing to communicate distances — connecting the work to multi-agent coordination and collective navigation.

# Report: Potentially Missed Related Work
Looking at the paper's references and the suggested related works:

## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

**No significant potentially missed related work identified.**

The paper comprehensively cites the relevant literature:

- **Already cited foundational works**: Hafting et al. (2005) and Burak & Fiete (2009) are both in the references as the seminal grid cell discovery and foundational path integration model.

- **Already cited closely related works**: Schaeffer et al. (2023) ("Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells") is extensively cited and directly compared throughout the paper.

- **Already cited works on grid cell emergence without path integration**: Dordek et al. (2016) and Dorrell et al. (2022) are cited for demonstrating grid cell emergence from non-path-integration objectives.

- **Already cited works questioning grid cells' role in path integration**: Nayebi et al. (2021) and Schøyen et al. (2023) are cited for findings that grid units are not more important than random units for path integration, with band cells being more important.

- **Already cited Xu et al. papers**: Multiple Xu et al. papers (2022, 2023, 2024) on conformal isometry are cited and directly compared.

The remaining suggested works either duplicate already-cited papers or have titles that don't correspond to clearly identifiable peer-reviewed publications in this domain.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [6.0, 3.0, 6.0, 5.0]
Average score: 5.0
Decision: Reject
Binary: Reject

=== CALIBRATION EXAMPLE 5 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

### Title & Abstract

- The title accurately reflects the core contribution: guiding visual prompt generation via textual knowledge in VLMs.
- The abstract identifies the problem (visual prompt over-fitting to base classes, poor generalization), states the method (cross-attention-based Text-Knowledge Guidance Module), and claims improvement on three task settings. This is clear and appropriately scoped.
- One concern: the abstract claims the method provides "substantial improvement" — an overstatement for most datasets where gains are marginal (e.g., +0.11% Novel on ImageNet, +0.15 HM on Caltech101). The EuroSAT result (+8 HM) and cross-dataset average (+1.53%) are the genuinely substantial gains. The abstract over-generalizes this.

---

### Introduction & Motivation

- The core motivation — that textual prompts generalize better than visual prompts (Figure 1) — is well-grounded intuitively and empirically, and the pivot to "let textual knowledge guide visual prompts" follows logically.
- The critique of MaPLe (fixed text prompts, symmetric projection) is fair and substantiated.
- **Key under-specified claim**: The introduction states that text embeddings encode "high-level semantic information" that generalizes better. But the critical question of *how* novel class text embeddings are accessed at test time is never explicitly stated in the introduction. Does TGVP use the text encoder at inference to embed novel class names (which are textually observable even for unseen classes)? This is the mechanism that explains generalization and should be foregrounded — it is implicit in the method but not articulated as a conceptual contribution.
- Contributions are stated clearly. However, the third contribution ("extensive experiments") is not a scientific contribution; it is a validation strategy and should not be listed as a main contribution.
- The motivation experiment (Figure 1, EuroSAT and DTD only) is presented as representative evidence but these two are among the most extreme datasets in the benchmark. The appendix (Table 10) provides broader evidence, but this should be acknowledged upfront rather than deferred.

---

### Related Work

- Coverage of the prompt tuning landscape is adequate: CoOp, CoCoOp, KgCoOp, ProGrad, MaPLe, PromptSRC are all cited.
- **Positioning against MaPLe is the most critical differentiation** and it is done, but only in the introduction; the Related Work section does not explicitly distinguish TGVP from MaPLe.
- CLIP-Adapter is mentioned but not evaluated as a baseline in experiments, which is a minor inconsistency.
- LLM-based methods (CoPrompt, LLaMP, CGP, ArGue) are discussed only in an ablation table (Table 6) rather than in Related Work. Given the thematic relevance (using richer textual knowledge), at least a paragraph in Related Work should situate TGVP relative to these.
- **Duplicate references**: PromptSRC appears twice (Khattak et al., 2023b and 2023c) citing the same ICCV paper. KgCoOp also appears twice (Yao et al., 2023a and 2023b). CoCoOp appears twice (Zhou et al., 2022a and 2022b). These are the same papers. This is not a formatting artifact — it reflects a real citation management error that inflates the reference list.

---

### Method / Approach

**Core mechanism (TKG Module):**

1. **Test-time class access is never made explicit.** The most important design question for the base-to-novel task is: when evaluating on novel classes, what does `W_text` contain? If the text encoder encodes novel class names at inference (which CLIP allows since class names are textually specified), then TGVP's generalization comes from the visual encoder dynamically routing toward those novel text embeddings via the cross-attention. This is plausible and actually elegant — but it is *never stated explicitly*. Equation (11) and its surrounding text do not clarify this. A reader cannot reproduce the inference procedure for novel classes without this information.

2. **Misleading "EMA" terminology** in Equation (10). The formulation $P^{tg}_j = \lambda T^{guide}_j + (1-\lambda)P_j$ is a static weighted linear combination at each forward pass, not an Exponential Moving Average in the standard sense (which refers to temporal averaging across training iterations). Using "EMA" here is technically incorrect and will confuse readers familiar with EMA in optimization. This should be called a "weighted fusion" or "linear interpolation."

3. **Projector architecture is underspecified.** The projector maps $W_{text} \in \mathbb{R}^{N_c \times D_t}$ to $W_{guide} \in \mathbb{R}^{N_c \times D_v \times L_{dvp}}$ via down-project ($D_t \times D_{mid}$) and up-project ($D_{mid} \times D'$ where $D' = D_v \times L_{dvp}$). The value of $D_{mid}$ is never specified in the paper or the appendix. This is a required detail for reproducibility.

4. **Top-K selection rationale is weak.** Equations (6–9) implement a sparse top-K attention over text categories. The paper does not explain *why* top-K rather than standard softmax over all classes. Possible motivations (focusing on the most relevant categories, avoiding noise from unrelated categories, computational efficiency) are not discussed. Table 7 ablates $K$ but does not compare against full softmax ($K = N_c$). This baseline is needed to validate the top-K design choice.

5. **CLS token as a query**: The text-guided CLS token $C^{tg}$ is computed by applying the same cross-attention process to the CLS token. Unlike the visual prompt tokens (task-level), the CLS token is instance-specific and changes per image. Fusing a per-instance query with task-level text category guidance raises a question: does this introduce label leakage if the CLS token already encodes class-discriminative information in deeper layers? The paper does not discuss this. The ablation (Table 5) shows that adding CLS guidance gives +0.72% on Base but only +1.71% on Novel — understanding *why* it helps novel classes more than base classes would be valuable.

6. **Interaction with deep prompting**: The appendix states that TKG is applied in the first 9 layers for visual prompts and in the 9th layer only for the CLS token. Why only the 9th layer for CLS? This asymmetry is unexplained.

7. **No analysis of computational overhead.** Adding a cross-attention module applied at each of 9 transformer layers introduces non-trivial computation. No parameter count comparison or inference time analysis is provided.

---

### Experiments & Results

**Base-to-Novel Generalization (Table 1):**
- Improvements on most individual datasets are marginal and their statistical significance is questionable without reported confidence intervals or standard deviations (despite having 3 runs).
- The headline improvement is on Novel average (+1.63% over PromptSRC). This is the most meaningful claim and is reasonably convincing.
- **EuroSAT is a major outlier**: +10.69% Novel and +8.00 HM improvement. This is dramatically larger than all other datasets. No analysis is provided for why TGVP disproportionately benefits EuroSAT (a satellite imagery dataset very different from natural images). This could indicate a method-specific bias rather than general superiority.
- UCF101 Novel is -1.10% vs. PSRC, making HM -0.28%. TGVP does not beat the SoTA on UCF101 but the paper does not discuss this.
- **Missing statistical reporting**: Table 1 reports mean performance across 3 runs without standard deviations. Given that some improvements are as small as +0.11%, this is insufficient for claims of statistical significance.

**Cross-Dataset Transfer (Table 2):**
- Claim that TGVP surpasses MaPLe by 1.53% in target average is accurately supported by the table.
- Missing baseline: TCP achieves 66.29% average (essentially tied with MaPLe at 66.30%) while TGVP achieves 67.83%. The paper should compare against TCP more directly, which is a 2024 method.

**Few-Shot Classification (Table 3):**
- Only the 4-shot setting is shown. Standard practice in this literature (e.g., CoCoOp, MaPLe papers) is to show K = {1, 2, 4, 8, 16}-shot curves. Reporting only 4-shot is incomplete and makes it impossible to assess whether TGVP's advantage holds across different data regimes. This is a notable gap for an ICLR submission.

**Domain Generalization (Table 4):**
- TGVP achieves the best average (61.07%) but does not beat WiSE-FT on ImageNet-V2 (65.19% vs. 65.12%). The paper ignores this. More importantly, WiSE-FT is a simple model ensembling technique with no prompt tuning — narrower wins over it deserve acknowledgment.

**Table 6 (LLM-based methods):**
- This is a critical transparency issue. The "Ours" row in Table 6 shows 85.57/78.42/82.35, which differs from TGVP's 85.10/77.73/81.24 in Tables 1 and 5. The paper says "by simply replicating the process used by ArGue to generate more accurate and comprehensive textual knowledge through LLMs." This means "Ours" in Table 6 is **TGVP + LLM-generated class descriptions**, a different system than reported elsewhere. This must be explicitly labeled (e.g., "TGVP+LLM") to avoid misrepresenting the main TGVP result as superior when the comparison group also uses LLMs.

**Ablation (Table 5):**
- The baseline is IVLP (independent visual-language prompts). MaPLe, which is the most architecturally similar prior work, should also serve as a baseline here to isolate the specific contribution of TKG.
- Table 5 shows VP-tg gives +4.43% Novel over IVLP but Table 1 shows only +1.63% over PromptSRC. The gap between these two comparisons is not explained — what makes IVLP a weaker baseline relative to PromptSRC?

---

### Writing & Clarity

- The method description in Section 3.2 is generally followable, but the inference procedure for novel classes (as noted above) is a critical clarity gap that affects understanding of the core contribution.
- Figure 2 is described as "an overview of TGVP alongside a brief comparison with existing multi-modal prompt techniques," but as rendered (text-based parser), it is nearly unreadable. The paper would benefit from a clearer textual description of the architectural differences in the caption.
- The paper refers to "Table 9" for the K ablation in the main text but labels it as "Table 7" in the actual table header. This cross-reference inconsistency (Table 9 vs. Table 7) is present in the paper itself, not a parser artifact.
- The abstract appears duplicated — a second copy of roughly two paragraphs appears early in the paper body. This is unusual and should be corrected before submission.

---

### Limitations & Broader Impact

- The paper includes no Limitations section. For an ICLR submission in 2025, this is an unusual omission.
- **Key unacknowledged limitations:**
  1. The method requires enumerating all class text embeddings at test time. For very large-scale datasets (e.g., ImageNet-21K with 21,000 classes), the computational cost of the cross-attention over all N_c classes in the TKG module becomes non-trivial and is not discussed.
  2. The EuroSAT outlier suggests the method may not uniformly improve all domains — some task types may be better suited to text-guided visual tuning than others, but no characterization of when the method works vs. fails is offered.
  3. The method's dependence on PromptSRC's regularization loss (mentioned in the appendix) means it is not fully self-contained — its strong performance may partly rely on PromptSRC's regularization rather than TGVP-specific design choices. This should be ablated.
  4. No broader impact statement is included. While this is a methodology paper with limited direct societal risk, the omission is notable.

---

### Overall Assessment

TGVP presents a conceptually reasonable idea — using high-level text embeddings (rather than learnable text prompts) as dynamic guidance for visual prompt generation via cross-attention — and demonstrates consistent improvements over strong baselines across multiple evaluation protocols. The most convincing result is the +1.63% Novel improvement on the 11-dataset average in base-to-novel generalization, and the cross-dataset transfer gain. However, the paper has several issues that an ICLR reviewer would likely flag as significant: (1) the inference procedure for novel classes is never explicitly described, which is the single most important detail for understanding *why* the method generalizes; (2) the "EMA" formulation is a misnomer for what is a fixed weighted interpolation; (3) the projector's D_mid is unspecified, impeding reproducibility; (4) Table 6 conflates a TGVP+LLM variant with the base TGVP without clear labeling; (5) few-shot results are shown only for K=4 rather than the full curve; and (6) no error bars are reported despite marginal improvements on many individual datasets. The EuroSAT outlier (+10.69% Novel) is not explained and raises questions about whether the improvement is specific to certain domain types. The paper is close to the ICLR acceptance bar — the idea is sound and broadly validated — but the above gaps in methodology transparency, ablation completeness, and statistical rigor would need to be addressed in a revision.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Text-Guided Visual Prompt Tuning (TGVP), which leverages textual knowledge to guide the generation of visual prompts for vision-language models. The key insight is that textual prompts exhibit better generalization to novel classes than visual prompts. The authors introduce a Text-Knowledge Guidance Module that dynamically selects task-relevant textual knowledge via cross-attention to enhance visual prompts with semantic awareness.

### Strengths
1. **Strong motivational insight**: The paper provides a clear empirical observation (Figure 1) that textual prompts generalize better to novel classes than visual prompts, offering a principled motivation for using text to guide visual prompt learning.

2. **Comprehensive experimental evaluation**: The method is evaluated across four challenging settings (base-to-novel generalization, cross-dataset transfer, domain generalization, few-shot classification) on 11 datasets, demonstrating consistent improvements over strong baselines.

3. **Consistent performance gains**: TGVP achieves notable improvements over prior state-of-the-art methods, particularly on novel class generalization (1.63% improvement over PromptSRC on average) and in domain generalization (61.07% average accuracy).

4. **Well-structured ablation studies**: The paper provides thorough ablations on key components (CLS-tg vs VP-tg), hyperparameters (K, λ), and the layer from which textual knowledge is extracted (Table 8).

5. **Compatibility with LLM-enhanced methods**: Table 6 demonstrates that TGVP can benefit from LLM-generated textual knowledge, showing practical extensibility.

### Weaknesses
1. **Limited architectural novelty**: The core mechanism—a cross-attention module for selecting relevant textual knowledge—is well-established in the literature. While the application to prompt tuning is new, the technical contribution is incremental.

2. **Missing recent baselines**: Several recent prompt tuning methods (e.g., IVLP, PromptAlign, PLOT) are not included in comparisons, which could provide a more complete picture of the method's relative performance.

3. **Computational overhead not discussed**: The cross-attention mechanism adds computational cost, particularly when computing similarity between visual prompts and all class text embeddings. The paper lacks analysis of inference time and memory footprint.

4. **EMA terminology appears misused**: Equation 10 describes a simple linear interpolation (λ·T_guide + (1-λ)·P), not an Exponential Moving Average. This mislabeling could confuse readers.

5. **Unclear handling of novel classes at test time**: The method uses text embeddings from all N_c classes for guidance, but at test time, novel classes may not have been seen during training. The paper should clarify how text guidance is constructed for completely novel classes.

### Novelty & Significance
The paper offers moderate novelty. The insight about textual prompts generalizing better than visual prompts is valuable, and the proposed cross-modal knowledge transfer mechanism is sensible. However, the technical contribution (cross-attention for text-guided visual prompts) is not particularly novel. The significance lies in the consistent empirical improvements and the clear demonstration that asymmetric cross-modal guidance (text→vision) is more effective than symmetric multi-modal prompting.

### Suggestions for Improvement
1. **Clarify the EMA terminology**: Either correct the terminology to reflect the actual operation (linear interpolation/fusion) or explain why EMA is the appropriate term.

2. **Add computational analysis**: Include inference time, memory usage, and parameter count comparisons with baseline methods to help readers assess practical tradeoffs.

3. **Expand baseline comparisons**: Include more recent methods such as PromptAlign, PLOT, or other recent ICLR/CVPR papers on prompt tuning to strengthen the empirical validation.

4. **Elaborate on test-time behavior**: Provide more details on how the method handles novel classes during inference, particularly regarding the construction of text embeddings for unseen categories.

5. **Provide theoretical insight**: Beyond empirical results, offer some theoretical or intuitive explanation for why this asymmetric text→vision guidance works better than symmetric multi-modal prompting.

6. **Consider adaptive K selection**: Rather than using a fixed K=5 for all datasets, an adaptive mechanism could potentially improve performance further and would demonstrate methodological sophistication.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments

1. **Multi-backbone evaluation**: Results are reported exclusively on ViT-B/16. Testing on ViT-L/14 (and possibly RN50/RN101) would strengthen the generalizability claims of TGVP, since it's unclear whether the TKG module's gains are architecture-specific or genuinely backbone-agnostic.

2. **Full K-shot learning curves**: The few-shot section only shows 4-shot results. Plotting the full K-shot curve (K = 1, 2, 4, 8, 16) would reveal whether TGVP's advantage grows or shrinks with more labeled data — a critical signal for understanding *when* text guidance matters most.

3. **Comparison with adapter-based and cache-based methods**: CLIP-Adapter, Tip-Adapter, and CaFo are natural baselines missing from the comparison tables. Since these methods also perform efficient adaptation of CLIP's visual representations, their inclusion would sharpen the narrative around what TGVP uniquely offers.

4. **Ablation of full cross-attention vs. top-k selection**: The paper selects top-k classes, but there is no experiment showing full-attention (k = all classes) as a baseline, which would directly validate the claim that selective text guidance is better than attending to all class embeddings uniformly.

5. **Training efficiency and parameter count comparison**: The paper would benefit from a table reporting trainable parameters, FLOPs added per forward pass, and wall-clock training time versus baselines — especially MaPLe and PromptSRC — to properly contextualize efficiency vs. accuracy tradeoffs.

6. **Sensitivity of cross-dataset transfer to source training shots**: The cross-dataset transfer experiment trains on all 1,000 ImageNet classes without varying the number of training examples. Showing how TGVP performs under restricted source-domain data would reveal its real-world applicability.

7. **Evaluation on retrieval benchmarks**: Adding zero-shot image-text retrieval (e.g., Flickr30K, MS-COCO retrieval) would expand the scope of "generalizability" claims beyond purely classification-centric settings.

---

### Deeper Analysis Needed

1. **Formal analysis of text embeddings vs. text prompt tokens as guidance**: The paper observes empirically that text embeddings generalize better than text prompts (Figure 1), but a theoretical or representational analysis (e.g., comparing the alignment geometry of both in the shared CLIP space) would make this a deeper, more principled contribution.

2. **Layer-wise analysis of where text guidance matters most**: The paper applies TKG to the first nine layers but reports only the final-layer (J=12) ablation for the text source. A layer-by-layer breakdown of what information is being injected — and which visual layers benefit most from the guidance — would provide mechanistic insight into the method.

3. **Analysis of which text classes are selected and why**: An interpretability analysis of the top-k selection patterns (e.g., which classes are most frequently retrieved for given visual inputs, whether selections are semantically coherent with the input image, whether novel class concepts emerge in selections for base-trained queries) would make the TKG module's behavior much more transparent.

4. **Gradient flow and optimization dynamics**: An analysis of whether the cross-attention module and EMA combination lead to stable training dynamics, particularly for fine-grained datasets like FGVCAircraft where gains are smaller, would deepen understanding of the failure modes.

5. **Statistical significance testing**: Improvements over the second-best method are sometimes marginal (e.g., +0.11 on ImageNet HM). Reporting confidence intervals or statistical significance tests across the 3 runs would strengthen trust in the empirical claims.

6. **Convergence speed comparison**: Showing training loss curves or validation accuracy trajectories compared to MaPLe/PromptSRC would reveal whether TGVP converges faster, which is relevant for practitioners.

---

### Untapped Applications

1. **Medical image classification**: Datasets like CheXpert, HAM10000, or PathVQA would test whether text-guided visual prompts can transfer structured biomedical textual knowledge into the visual branch — a high-value domain where few-shot generalization is especially critical.

2. **Remote sensing beyond EuroSAT**: Extending to RESISC45 or more diverse satellite image benchmarks would confirm EuroSAT's strong gains (+10.69% novel) are systematic rather than dataset-specific.

3. **Video understanding**: UCF101 is included but treated as a static image task. Testing TGVP on video-level VLMs (e.g., Video-CLIP, X-CLIP) would open an entirely new application frontier for text-guided visual prompt tuning.

4. **Dense prediction tasks**: Applying TGVP to open-vocabulary segmentation (e.g., using CLIP-based frameworks like MaskCLIP or FC-CLIP) would test whether semantically-aware visual prompts improve spatial understanding, not just global classification.

5. **Multi-lingual and cross-lingual scenarios**: Testing TGVP when the text encoder operates in non-English languages (using multilingual CLIP variants) would reveal whether the text-guided visual generalization is language-agnostic — an underexplored but practically important axis.

6. **Continual/incremental learning settings**: Evaluating TGVP when new classes arrive incrementally would test whether dynamic text-guided visual prompts naturally mitigate catastrophic forgetting, a property already suggested by the strong novel-class performance.

---

### Visualizations & Case Studies

1. **Cross-attention weight maps over text categories**: Visualizing, for a given test image, which text class embeddings receive high attention weights during the TKG module's selection — and whether these are semantically related to the visual content — would powerfully illustrate the method's mechanism to readers.

2. **t-SNE/UMAP of visual features before and after TGVP**: Comparing the feature space geometry of CLIP zero-shot, PromptSRC, and TGVP (especially on a dataset with both base and novel classes) would make the "enhanced discriminability and generalizability" claims visually concrete.

3. **Qualitative failure case analysis**: Showing specific images where TGVP fails (e.g., the -1.10 novel drop on UCF101, or the -0.94 drop on Flowers novel) and diagnosing *why* — perhaps because top-k text guidance pulls toward wrong semantic categories — would provide honest, actionable insight.

4. **Attention visualization across transformer layers**: Showing how the text-guided visual prompt evolves across the 9 injection layers (e.g., GradCAM or attention rollout maps) would reveal whether early layers capture coarse semantic cues and later layers fine-grained discriminative ones.

5. **Side-by-side comparison of selected top-k classes across domains**: A qualitative table showing what top-5 text classes are dynamically selected for the same visual input across EuroSAT, DTD, and OxfordPets would build intuition for why cross-dataset transfer works.

---

### Natural Next Steps

1. **Integration with LLM-enriched class descriptions**: Table 6 shows that LLM-generated text descriptions boost TGVP further (+0.47 HM with ArGue-style descriptions). The paper would benefit significantly from a systematic study of what makes LLM descriptions more effective as textual guidance and whether structured descriptions (attributes, visual cues) outperform verbose ones.

2. **Extending TGVP to generative VLMs**: Applying the TKG module concept to generative VLMs (LLaVA, InstructBLIP) — where the visual encoder must guide generation rather than classification — would position this work at the frontier of VLM adaptation research and open new citation pathways.

3. **Dynamic K selection**: The current method uses a fixed K=5. A learned or input-adaptive K — where the model decides how many text classes to attend to based on visual ambiguity — could further improve performance on datasets with overlapping semantic categories.

4. **Multi-task prompt tuning**: Extending TGVP to simultaneously adapt to multiple datasets with a single prompt (using task-specific text guidance as a routing mechanism) would address the practical limitation that current prompt tuning methods require per-dataset training.

5. **Combining TGVP with test-time adaptation**: Since the TKG module naturally adapts to test-class distributions via the top-k selection from text embeddings of *all* test classes, exploring test-time optimization of the λ or temperature parameter could yield a training-free adaptation variant — a compelling follow-up contribution.

# Report: Potentially Missed Related Work
I'll carefully check the paper's references and citations against the search results.

**Checking citations in the paper:**

Looking through the references section and in-text citations:
- **KgCoOp** (Yao et al., 2023a/b) — CITED multiple times (references section and Table 1)
- **ArGue** (Tian et al., 2024) — CITED (in Table 6 and references)
- **CLIP** (Radford et al., 2021a/b) — CITED extensively throughout the paper
- **CoCoOp** (Zhou et al., 2022a/b) — CITED multiple times

**Evaluating remaining works:**

1. **Enhancing Visual-Language Prompt Tuning Through Sparse Knowledge-guided Context Optimization** — Anonymous (2024, PMC): This is an anonymous submission (likely under double-blind review like this paper). The authors could not have known about or cited a concurrent anonymous submission. Additionally, this work focuses on knowledge graphs for textual prompt optimization, which differs from the paper's approach of using text embeddings to guide visual prompts via cross-attention.

2. **Position-Guided Text Prompt for Vision-Language Pre-Training** — Wang et al. (2023, CVPR): This work addresses pre-training with position-guided text prompts to boost visual grounding. The connection to this paper's adaptation-focused text-guided visual prompt tuning is tangential — it requires multiple leaps (pre-training vs. adaptation, position-guided templates vs. semantic knowledge transfer). The problem settings and methodologies are substantially different.

---

## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

No significant potentially missed related work identified.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [5.0, 5.0, 5.0, 5.0]
Average score: 5.0
Decision: Reject
Binary: Reject

=== CALIBRATION EXAMPLE 6 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately captures the contribution. The abstract clearly identifies the problem (single-LLM critiques are flawed), the method (multi-agent feedback for SFT + RL), and the key result (7B model approaches 70B and GPT-4). One claim deserves scrutiny: "approaching the performance of advanced 70B LLMs and GPT-4" is supported on CRITICBENCH (75.66% vs 78.75%), but Table 1 suggests performance gaps persist on CRITICEVAL's objective revision metrics (R_obj). The abstract does not distinguish between benchmarks, which overstates breadth of the result. Additionally, the abstract states the pipeline "improves the preference accuracy of critique quality through multi-agent feedback" but the specific mechanism (MARS filtering) is not mentioned, making it harder to assess the novelty claim at first read.

---

### Introduction & Motivation

The motivation is well-constructed: the limitation of behavior cloning on single-model critiques is a real and recognized problem (Verga et al., 2024; Lan et al., 2024). The identification of two distinct failure modes—noisy SFT data and noisy preference data—as separate problems to solve is intellectually clear. The contributions are stated specifically enough to be verifiable. However, the introduction slightly over-claims that "flaws in critiques are propagated into the learned model during the SFT stage, leading to the **potential amplification** of these issues." This amplification claim is presented as established fact, but no evidence—empirical or theoretical—is provided for amplification versus mere propagation. This is a minor over-claim but worth noting given ICLR's standards for precision.

---

### Related Work

Coverage is generally solid across the three pillars (critique ability, preference-based RL, multi-agent frameworks). Two notable gaps:

1. **Constitutional AI (Bai et al., 2022)** is directly relevant to automated iterative critique-based training but is absent. This is not a trivial omission given the thematic overlap.
2. **Self-Taught Evaluators (Wang et al., 2024b)** and **Meta-Rewarding (Wu et al., 2024)** are cited, and the positioning against Meta-Rewarding is the most important comparison. The paper argues its MARS filtering is more robust than using an LLM as a meta-judge. This claim is directionally supported by Table 6 but the comparison is ablative rather than a head-to-head on the same pipeline. The claim deserves more careful qualification.

The positioning against Prometheus (Kim et al., 2024) in Appendix B is fair and detailed. The distinction drawn between inference-stage multi-agent frameworks (ChatEval, PoLL) and training-stage improvement (this work) is accurate.

---

### Method / Approach

This is the most substantive section, and several concerns arise:

**1. GPT-4 as both agent and meta-critic (circular evaluation risk).** GPT-4 is one of the four critique agents in MultiCritique-SFT, *and* GPT-4 is the meta-critic that classifies and filters all four agents' ACUs. This creates a structural bias: when GPT-4 disagrees with Claude/Qwen/InternLM, GPT-4 as meta-critic is positioned to favor its own perspective. The paper provides no analysis of how often GPT-4's critiques are retained vs. modified in the summarization step, nor whether GPT-4-generated ACUs disproportionately receive L0 (correct) labels. This is a significant methodological concern that should be analyzed empirically—even a sample-level agreement analysis would help.

**2. Inference-time overhead not made explicit in the main paper.** The trained model generates task description, two-tier criteria, and reference response *in addition to* the critique (Section 3.2, Appendix C.2, Appendix I). This means the model produces substantially more tokens than any baseline at inference time. Appendix A.4 acknowledges this but frames it as a "limitation," whereas it fundamentally changes the comparison: the model has more compute time and more intermediate reasoning steps. The authors should either (a) include a wall-clock and token-count comparison against baselines, or (b) more prominently acknowledge this as an architectural design choice—not just a limitation—that contributes to performance gains.

**3. Severity score threshold (= 5) for preference pairing is unjustified.** The threshold determining whether a chosen/rejected pair has a "significant performance gap" (accumulated severity score difference > 5) is set without ablation or sensitivity analysis. Given that the quality of the resulting preference dataset directly affects RL stability (shown dramatically in Table 6), the choice of this hyperparameter warrants at least a brief sensitivity analysis.

**4. MARS filtering has a potential model overlap issue.** The four 7B LLMs used for revision scoring (InternLM2.5-7B-Chat, Llama-3.1-8B-Instruct, Qwen2-7B-Chat, Mistral-7B-Instruct) and the InternLM2-20B-reward model used to score revisions are separate from the fine-tuned model (InternLM2-7B-Chat). This appears methodologically sound. However, the same InternLM2-20B-reward is also used in Step 1 to select high/medium/low quality responses. If this reward model has systematic biases (acknowledged in Appendix A.3), those biases propagate into both the SFT and RL datasets in correlated ways.

**5. No justification for exactly 4 critique agents.** The choice of 4 agents (GPT-4, Claude-1-instant, Qwen-1.5-72B, InternLM2-20B) is not ablated. Appendix A.2 acknowledges budget constraints but does not provide any empirical signal about the marginal value of the 4th agent over 3, or 3 over 2. For a paper whose core claim is that "multi-agent" feedback is key, the number of agents should be an ablated variable.

---

### Experiments & Results

**1. Table 1 (main results) is incomplete in the submitted paper.** The table header appears in the extracted text but the body data is missing or severely mangled, showing only partial entries for GPT-3.5-Turbo and GPT-4-Turbo. While this appears to be a PDF parsing artifact per the instructions, ICLR reviewers reading the PDF should verify that the main results table is complete and readable. The paper makes strong quantitative claims (e.g., "21.48% average performance gain on CRITICEVAL") that require Table 1 to be legible.

**2. No statistical significance testing anywhere.** Every quantitative comparison in the paper—Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Figure 2—reports single point estimates without error bars, confidence intervals, or significance tests. CRITICEVAL's subjective metrics (F_sub, R_sub) are particularly susceptible to variance because they depend on GPT-4 judgments. The RL improvement on CRITICBENCH (0.51% absolute) is almost certainly within noise but is presented as a meaningful gain.

**3. GPT-4 as both training signal and evaluator creates evaluation bias.** The subjective metrics F_sub and R_sub use GPT-4 as judge. Since the training data is heavily shaped by GPT-4 (meta-critique classification, summarization, crucial information generation), fine-tuned models may learn to produce critiques stylistically aligned with GPT-4's preferences, leading to inflated subjective scores. The objective metrics (F_obj, R_obj) are less susceptible to this bias. Notably, the RL stage gains are much larger on F_sub (6.3%) than on F_obj (~5%), and the RL stage actually *decreases* R_obj (19.26 < 19.33). The pattern suggests RL may be optimizing for GPT-4 stylistic preferences rather than genuine critique quality improvements.

**4. The RL benefit is inconsistent and modest.** On CRITICBENCH, the RL improvement is 0.51% (absolute), which is negligible. On CRITICEVAL, R_obj decreases while F_sub improves. The paper attributes R_obj regression to "inherent instability in evaluating revisions," but this is speculative. An alternative explanation—that RL is mode-collapsing toward GPT-4 stylistic preferences at the cost of actionable critique content—is not explored.

**5. Missing ablation on the number of agents.** As noted above, the paper does not ablate 2 vs. 3 vs. 4 agents. Since the core contribution is multi-agent aggregation, this is an important missing experiment.

**6. Generalization experiment (Table 8) is promising but limited.** The zero-shot transfer to unseen math/code tasks is a strong result (88.44% vs 59.46% on math). However, the experiment uses InternLM2.5-7B-Chat as the base, not InternLM2-7B-Chat-SFT (the main model). This inconsistency makes direct comparison to Table 1 results difficult and should be noted.

**7. Cost comparison is misleading.** The paper states that "Our MultiCritique framework achieves superior performance with merely 3K training samples (~$890 cost)." However, Table D (Appendix D) shows the *total* construction cost is $9,180 + $125.6 ≈ $9,300. The $890 cost refers to 3K samples in a scaling experiment, not the full dataset. The claim that this "demonstrates a substantial improvement in data efficiency, with a factor of 2.15-4.22× over existing methods" compares 3K samples of MultiCritiqueDataset against full 100K-257K baseline datasets. This is an unfair comparison: it conflates per-sample quality (which is higher) with total cost (which is also much higher). The true comparison should normalize by total dataset cost.

**8. Evaluation on only two benchmarks.** CRITICEVAL and CRITICBENCH are both relatively new benchmarks and may not fully capture critique ability across diverse domains. The decision to exclude pairwise comparison benchmarks (RewardBench, PandaLM) due to "unfairness" (Appendix E.2) is defensible but leaves open whether the gains generalize beyond single-response scoring tasks.

---

### Writing & Clarity

The paper is generally well-organized and the pipeline description is detailed enough to follow. Specific clarity concerns:

- **Section 5.1**: The discussion of R_obj vs. F_sub trade-offs in the RL stage is compressed to two sentences and attributes a real performance regression to "inherent instability" without analysis. This deserves more space.
- **Crucial information at inference time**: The paper never clearly states in the main text whether the model generates task description, criteria, and reference response at inference time, or whether these are provided externally. This is buried in appendices and is fundamental to understanding the model's behavior.
- **ACU severity scores**: The seven quality categories and their severity scores (Table 19) are only defined in Appendix J.5. Since the accumulated severity score is the primary quality signal driving the entire RL pipeline, at least a brief summary should appear in the main paper.
- **Table 5 (ablation on crucial information)**: The "w/o Criteria" row appears twice with different values (57.28 vs. 57.28 for F_obj but different other columns), likely a formatting artifact. The reported finding that removing criteria *improves* R_sub (6.17 > 5.78) while degrading other metrics is interesting and is acknowledged as unexplained ("we will explore the reasons...in our future work"). For ICLR, this deserves at least a hypothesis, not deferral.

---

### Limitations & Broader Impact

Appendix A is unusually thorough—five distinct limitation sections covering protocol scope, pipeline model choices, MARS limitations, inference efficiency, and 70B model results. This level of honesty is commendable. However, three substantive limitations are not discussed:

1. **Benchmark overfitting risk**: Both training queries and test queries come from overlapping task distributions (alignment, math, code). While the paper explicitly states no test samples are in training, the *task types* heavily overlap, and the pipeline's customized criteria generation is tailored to these task types. Performance on truly out-of-distribution critique tasks (e.g., scientific peer review, legal document analysis) is unknown.

2. **The GPT-4 ceiling problem**: The MultiCritique-SFT pipeline is explicitly bounded by GPT-4's meta-critique ability. The paper shows that Qwen2.5-72B and Claude-3.5-Sonnet achieve >70% agreement with GPT-4 (Table 15), but this also means any systematic errors in GPT-4's meta-critique (e.g., position bias, verbosity bias) are inherited by the entire dataset. This is not discussed.

3. **Evaluation self-consistency**: Models are evaluated on CRITICEVAL and CRITICBENCH, which use GPT-4 for subjective evaluation. Since the pipeline generates training data via GPT-4, there is a stylistic alignment loop that cannot be disentangled from genuine quality improvements using only these benchmarks.

---

### Overall Assessment

MultiCritique presents a technically coherent and practically motivated approach to improving LLM critique ability through multi-agent data aggregation and MARS-filtered preference learning. The empirical results are strong: a 7B model trained on 32.1K samples significantly outperforms larger baselines on two dedicated benchmarks, and the ablations confirm the value of the individual pipeline components. The paper is relatively well-written and unusually transparent about limitations.

However, three concerns are serious enough to warrant revision before acceptance at ICLR: (1) The circular evaluation problem—GPT-4 as simultaneously an agent, the meta-critic, and the evaluator—is not empirically characterized, creating an unresolvable confound between genuine quality improvement and GPT-4 stylistic alignment; (2) The absence of statistical significance testing is especially problematic for the modest RL gains (0.51% on CRITICBENCH, negative on R_obj), which may not be reliable; and (3) The per-sample vs. total-cost framing of data efficiency is misleading. The core multi-agent aggregation idea is sound and the MARS filtering is a genuinely useful contribution, but the paper needs to more clearly disentangle "multi-agent critique quality" from "GPT-4 preference alignment" before the empirical claims can be taken at face value.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper proposes MultiCritique, a novel data generation pipeline for improving LLM critique ability through multi-agent feedback in both SFT and RL stages. The key innovation lies in: (1) aggregating critiques from multiple LLMs via GPT-4 meta-critique classification to filter flawed ACUs (Analytical Critique Units), and (2) introducing Multi-Agent-Revision-Scoring (MARS) to construct high-quality preference pairs for RL training. Experiments demonstrate that a 7B model trained on MultiCritiqueDataset outperforms other 7B-13B models and approaches GPT-4 performance on CRITICEVAL and CRITICBENCH benchmarks.

### Strengths
1. **Strong empirical performance**: The fine-tuned InternLM2-7B achieves 75.66% F1 on CRITICBENCH, approaching GPT-4's 78.75%, with 21.48% and 22.50% average gains over existing critique datasets on CRITICEVAL and CRITICBENCH respectively (Table 3).

2. **Novel pipeline design**: The MultiCritique-SFT pipeline introduces structured ACUs (Analytical Critique Units) and meta-critique classification to aggregate high-quality critiques from multiple models while discarding flawed ones—this addresses a genuine limitation in prior single-model critique generation.

3. **Comprehensive ablations**: The paper provides thorough ablation studies on crucial information components (Table 5), MARS filtering effectiveness (Table 6), and multi-agent contribution analysis (Table 4), demonstrating each component's necessity.

4. **Data efficiency**: Figure 2 shows MultiCritique achieves superior performance with only 3K samples (~$890 cost) versus baselines requiring 100K-257K samples ($1,915-$3,758), demonstrating a 2.15-4.22× improvement in data efficiency.

5. **Generalization experiments**: Table 8 shows the model fine-tuned without math/code data still achieves 88.44% and 77.63% on these unseen tasks, demonstrating strong generalization.

### Weaknesses
1. **Heavy dependency on GPT-4**: The entire pipeline relies on GPT-4 for task description generation, criteria generation, reference response generation, meta-critique classification, and final critique summarization. While Appendix G shows other LLMs can conduct meta-critique, the main results still depend on GPT-4, limiting reproducibility and accessibility.

2. **High computational cost for RL stage**: The MARS filtering requires 4 LLMs × 8 revisions = 32 revisions per sample for preference pair construction, making the RL data generation expensive despite the claimed data efficiency in SFT.

3. **Limited results on larger models**: The paper acknowledges insufficient investigation on 70B+ models. The preliminary results suggest limited improvements (Section A.5), raising questions about scalability to more capable models where critique ability may already be saturated.

4. **Reliance on reward model for revision quality**: MARS filtering uses InternLM2-20B-reward to evaluate revision quality. While correlated with human judgment (95.3% consistency claimed), this may not accurately reflect true revision quality across all task domains, especially for subjective tasks.

5. **No human evaluation of generated critiques**: All subjective evaluations (F_sub, R_sub) rely on GPT-4 as evaluator. This introduces potential bias, as GPT-4 is also used in the data generation pipeline.

6. **Lack of comparison with concurrent RLHF critique methods**: While CriticGPT and Themis are mentioned, direct comparison with these concurrent works on identical benchmarks is limited.

### Novelty & Significance
The work presents a meaningful contribution to improving LLM critique ability—a meta-cognitive capability important for self-improvement and evaluation. The multi-agent aggregation with structured meta-critique classification is novel. The MARS filtering for preference pair construction is a practical innovation. The results are significant, showing strong performance from a 7B model. However, the heavy reliance on GPT-4 limits the practical accessibility of the approach, and the scalability to larger models remains uncertain.

### Suggestions for Improvement
1. **Include human evaluation**: Add human evaluation on a subset of generated critiques to validate that GPT-4 meta-critique classification correlates with human judgment of critique quality.

2. **Reduce GPT-4 dependency**: Explore using open-source models (e.g., Qwen2.5-72B as shown in Appendix G) for meta-critique in the main pipeline and report comparative results.

3. **Analyze 70B model limitations**: Provide deeper analysis on why improvements diminish for larger models—is it ceiling effects, data distribution mismatch, or optimization issues?

4. **Cost-performance tradeoff analysis**: Provide explicit comparison of API costs versus performance gains to help practitioners decide whether the approach is worth the expense.

5. **Release intermediate data**: Consider releasing not just final datasets but also the intermediate multi-agent critiques, enabling research on alternative aggregation strategies.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments
1. **Human evaluation of critique quality**: The paper relies heavily on GPT-4 as evaluator for subjective metrics (F_sub, R_sub). A targeted human study on a subset (e.g., 200–300 critiques) comparing MultiCritique outputs vs. GPT-4-only baselines would provide a ground-truth anchor that's independent of the judge, strengthening the central claim of quality improvement.

2. **Ablation on the number of agents**: The paper fixes 4 agents but never ablates this. An experiment sweeping 1→2→3→4→5 agents on a held-out critique quality metric would directly show the marginal benefit of each additional agent and establish whether 4 is the optimal cost/quality trade-off point.

3. **RL improvement under-analyzed**: The RL gain is 6.3% on CRITICEVAL but only 0.51% on CRITICBENCH. Adding a breakdown by task category within CRITICBENCH (math, code, symbolic, algorithmic) would reveal *where* RL helps, and whether the minimal overall gain is masking category-level wins and losses.

4. **MARS threshold sensitivity**: The severity threshold of 5 for preference pairing is set without a sweep. Reporting results for thresholds {2, 3, 5, 7, 10} would show robustness and guide practitioners in applying this pipeline to new domains.

5. **Iterative/self-play training rounds**: The paper trains one round of SFT+RL. Adding even one more iteration—using the fine-tuned 7B model as one of the critique agents in the next data-generation round—would show whether the pipeline supports self-improving loops, which is a compelling ICLR narrative.

6. **Broader model coverage beyond InternLM2**: The main results (Table 1) primarily feature InternLM2-7B. Although Appendix H extends to Qwen2.5/Llama3, these RL results are absent there. Including RL-stage results for all backbone models would substantially strengthen the universality claim.

7. **Comparison against direct preference optimization (DPO)**: The paper uses PPO, but many critiquing works now compare against DPO or SimPO. Adding a DPO baseline using the same MultiCritiqueDataset-RL would help practitioners understand which RL approach benefits most from this pipeline.

8. **Out-of-distribution task generalization**: The generalization study (Table 8) removes math/code and tests the same benchmarks. A stronger test would evaluate on a genuinely new benchmark not seen during data collection (e.g., FeedbackBench or FollowBench) to show the pipeline produces transferable critique skills.

---

### Deeper Analysis Needed
1. **Why does w/o Criteria improve R_sub?** The paper notes this anomaly but leaves it to "future work." Even a qualitative analysis of 20–30 sampled critiques comparing the "w/ Criteria" and "w/o Criteria" conditions would help readers understand the trade-off between criteria-guided precision and free-form revision diversity—a theoretically interesting finding.

2. **Inter-agent disagreement analysis**: The paper shows that multi-agent aggregation outperforms any single model, but there is no analysis of *how often* the four agents disagree, *which agent pairs* most commonly conflict, and whether disagreement rate correlates with the difficulty of the query. This would help explain why the aggregation works.

3. **ACU category distribution across models**: The seven ACU quality categories (L0–L6) are defined but their empirical distribution is never reported. A breakdown of how often each model falls into each category (false negatives, wrong severity, etc.) would reveal complementary strengths/weaknesses—directly motivating the multi-agent design.

4. **MARS filtering efficiency analysis**: The paper mentions that preference pairs are dropped if chosen critique's average reward is not higher, but the retention rate is never stated. Reporting what fraction of pairs survive MARS filtering per task type would indicate which domains are hardest to produce reliable preference signal for.

5. **Reward model calibration**: The RL stage's reward model uses focal ranking loss but there is no analysis of its classification accuracy on a held-out preference test set. Reporting this would validate that the reward model is not the bottleneck limiting RL gains.

6. **Correlation between ACU count and critique quality**: An analysis of whether critique quality (as measured by downstream revision reward) correlates with the number of identified ACUs would reveal whether flaw quantity or flaw accuracy is the more important dimension of critique skill.

---

### Untapped Applications
1. **Scientific paper review**: The structured ACU format (location → description → criteria → severity → suggestion) maps naturally onto academic peer review. Demonstrating MultiCritique on a domain like arXiv abstract/introduction review would show real-world applicability beyond chat-style alignment.

2. **Code review for software engineering**: While coding tasks are included, the paper treats code as a reasoning task. Applying MultiCritique to pull-request-style code review (e.g., CodeReviewer benchmark) where the evaluated "response" is a code diff would extend the framework to a high-value industrial application.

3. **Medical and legal text quality assessment**: These domains require nuanced, criteria-grounded critiques where errors carry high stakes. Adapting the two-tier criteria generation to structured clinical or legal rubrics would demonstrate the pipeline's generality and yield strong domain-specific baselines.

4. **Reward model improvement via critique-based data**: The paper mentions in Section 2 that textual critiques contribute to robust reward modeling. A direct experiment using MultiCritiqueDataset to improve a reward model (and evaluating on RewardBench) would close this loop and demonstrate a practical downstream use case.

5. **Multilingual critique**: All experiments are in English. Extending to Chinese or other languages—especially given that the base model (InternLM2) is strong in Chinese—would broaden the impact and test whether the structured ACU format transfers across languages.

---

### Visualizations & Case Studies
1. **Agent agreement/disagreement heatmap**: A matrix showing pairwise agreement rates between GPT-4, Claude, Qwen, and InternLM on ACU correctness (L0 vs. non-L0) across different task types would visually motivate why all four agents are needed and which pairs are most complementary.

2. **Qualitative SFT vs. RL critique comparison**: Side-by-side examples of the same query critiqued by the SFT model vs. the RL model, annotated with which specific ACUs changed, would give readers intuition for what the RL stage concretely improves—especially important given the modest objective metric gains.

3. **Failure mode gallery**: A set of 4–5 cases where MultiCritique still produces flawed critiques (false negatives, severity errors) would help practitioners understand the pipeline's limitations and guide future work more concretely than the current textual limitations section.

4. **Data efficiency curve with cost overlay**: Figure 2 shows performance vs. data scale, but adding API cost as a secondary axis (or annotation) would make the "2.15–4.22× data efficiency" claim visually compelling and directly comparable to baselines.

5. **ACU severity distribution per task category**: A stacked bar chart showing the distribution of ACU severity labels (Negligible/Minor/Moderate/Severe) per task type (alignment, math, code) would reveal whether certain task types systematically produce harder-to-critique responses, motivating the different treatment of math/code.

---

### Natural Next Steps
1. **Pairwise response comparison extension**: Already acknowledged as future work, but the paper would be significantly stronger at submission time if even preliminary pairwise results on RewardBench or PandaLM were included, since these are the most practically relevant evaluation protocols for RLHF.

2. **Automated pipeline with open-source meta-critique**: Since Appendix G shows Claude-3.5-Sonnet and Qwen2.5-72B can substitute for GPT-4 in meta-critique (≥70% agreement), the next version of MultiCritique could operate entirely with open-source models, making the pipeline truly cost-free and reproducible—a significant contribution.

3. **Process-level critique for chain-of-thought reasoning**: Extending ACU-level critique from final responses to intermediate reasoning steps (similar to process reward models) would connect this work to the increasingly important area of step-wise verification, with natural applications to math and coding.

4. **Critique-guided self-refinement loop**: Using the fine-tuned critique model within a closed-loop iterative refinement system—where a generator model repeatedly improves responses based on the critique model's feedback—would demonstrate end-to-end value beyond the benchmark metrics currently measured.

5. **Dataset versioning via the fine-tuned model**: Using the trained MultiCritique model as one of the four agents in the next generation round (replacing a weaker base model) would create a data flywheel. Showing even one iteration of this loop would demonstrate that the pipeline can scale over time without increasing GPT-4 API costs.

# Report: Potentially Missed Related Work
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Wei et al. (2022, NeurIPS).
   Why potentially missed: The paper repeatedly frames critique ability as a "meta-cognitive capability" and references meta-cognition works (Toy et al., 2024; Wang & Zhao, 2024), but does not cite the foundational CoT paper that established chain-of-thought reasoning—a core meta-cognitive technique in LLMs. Additionally, the paper explicitly mentions evaluating "chain-of-thought (CoT)" approaches in CRITICEVAL (Section 4.2), making this foundational reference relevant.

2. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — Rafailov et al. (2023, NeurIPS).
   Why potentially missed: The paper's RL stage uses PPO with a trained reward model for preference-based learning. DPO has become a prominent alternative to PPO-based RLHF that directly optimizes from preference pairs without requiring a separate reward model. Given the paper's focus on preference-based critique refinement, discussing or comparing to DPO would strengthen the preference learning methodology discussion in Section 2 (lines 141-156).

3. **Deep Reinforcement Learning from Human Preferences** — Christiano et al. (2017, NeurIPS).
   Why potentially missed: The paper cites later RLHF works (Stiennon et al., 2022; Ouyang et al., 2022) but omits the seminal Christiano et al. paper that established the foundational RLHF methodology. While the practical implementation is covered by later citations, the foundational work would complete the methodological lineage for the RL stage discussion.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [5.0, 5.0, 6.0, 6.0, 6.0, 8.0]
Average score: 6.0
Decision: Reject
Binary: Reject

=== CALIBRATION EXAMPLE 7 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

### Title & Abstract

- The title "NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer" accurately reflects the core contribution: repurposing a pre-trained video diffusion model (SVD) for NVS without training.
- The abstract correctly identifies the three pillars: training-free operation, adaptive score modulation, and multi-scenario applicability. However, "significant superiority" is stated without any quantitative anchor, making the abstract weaker than it could be.
- The claim that the method enables "visually pleasing results from single or multiple views of static scenes *or* monocular videos of dynamic scenes" is accurate and supported. The phrase "without the need for training" should be clarified — the base model SVD was obviously trained; the novelty is zero-shot adaptation at test time.

---

### Introduction & Motivation

- The motivation is solid. The gap is clearly identified: existing NVS methods either require per-scene optimization (NeRF, 3DGS) or task-specific fine-tuning, while powerful video diffusion priors are underexploited for geometry-guided view synthesis.
- The three stated contributions are honest and specific: training-free paradigm, adaptive score modulation with theoretical justification, multi-scenario performance.
- The introduction does not over-claim; it accurately scopes to a specific base model (SVD) and specific guidance mechanism (score modulation via warping). This is appropriate.
- One gap: the introduction does not clearly distinguish this work from SV3D (Voleti et al., 2024), which also leverages latent video diffusion for multi-view synthesis. The conceptual contrast (SV3D fine-tunes on 3D data; this work is truly zero-shot) is buried rather than foregrounded.

---

### Related Work

- Coverage of diffusion sampling algorithms is thorough and relevant (DDIM, DPM-Solver, posterior sampling, null-space methods).
- NVS coverage is reasonable but misses several directly relevant contemporaries:
  - **SV3D** (Voleti et al., 2024) is cited once in passing but deserves explicit discussion since it uses the same SVD base and targets multi-view synthesis — the key distinction (supervised fine-tuning vs. zero-shot) should be stated here.
  - **ZeroNVS** (Sargent et al., 2024) and **Photoconsistent-NVS** (Yu et al., 2023) appear only in the appendix despite being close neighbors; they belong in related work.
  - **ViewCrafter** and **DUSt3R/MASt3R** are absent despite being concurrent zero-shot or reconstruction-based NVS methods relevant to this comparison space.
  - **ReconFusion** (Wu et al., 2024b) is cited but not compared against.
- The positioning against "traditional guided sampling algorithms" that use image degradation models is valid but could be stated with more precision: the key novelty is the warping-driven guidance rather than degradation-driven guidance.

---

### Method / Approach

**Score modulation framework (Sec. 4.1):**
- The core idea — modulating the score function with warped views and then deriving a closed-form blending in Eq. (12) — is clean and well-motivated.
- The Taylor expansion in Eq. (8) relating I(p_0) to I(p_i) via Δp is only valid for small pose changes. The paper applies this to scenes with large viewpoint changes (up to 360°) without discussing when the linearization breaks down. This is a genuine theoretical gap.
- The "intensity function" formulation via (McMillan & Bishop, 1995) is a reasonable bridge, but the full pipeline also uses depth-driven warping W(·), estimated depth D̃, and depth errors ΔD. The residual term E_T in Eq. (10) bundles both the Taylor truncation error and depth estimation error, but these are treated as a single combined error in the bound analysis when they have very different statistical properties and dependencies.

**Adaptive λ derivation (Sec. 4.2):**
- The optimization in Eq. (16) is set up to minimize an upper bound of the estimation error — a reasonable strategy. However, two critical issues arise:
  1. **The empirical parameterizations** E_D = v_2·σ(t) and E_P = v_3·||Δp||_2 are derived from visual inspection of Fig. 2(a) and 2(b) on unspecified experimental data. Linear fits to log-scale plots can hide large residuals. The resulting λ̃(t,p) inherits all of this approximation but is presented as a theoretically justified quantity.
  2. **The hyperparameters (v_1, v_2, v_3) differ per scenario**: (1e-6, 9e-1, 5e-2) for single view, (1e-6, 7e-1, 1e-2) for sparse, (1e-6, 1.75, 3e-2) for dynamic scenes. This is a major practical limitation: the "adaptive" method still requires scenario-specific tuning. The paper does not discuss how these are selected or provide sensitivity analysis beyond Table 4 (which only ablates κ and normalization).
- Appendix A.1 proves E_M ≲ E_P (modulated score has lower error than naive inpainting-based guidance). The derivation is mostly correct but relies on the approximation |Q|≫2v_1 to simplify Taylor series of √(Q²+4v_1Q). This approximation is unstated as a condition and may not hold when σ(t) is large and ||Δp|| is small (early diffusion steps, small view changes).
- Appendix A.2 proves the DGS update stays near the data manifold by invoking the Jacobian of the clean-image mapping and a small step-size argument. The proof is valid under the assumption that the update rate κ=2e-2·σ(t) is small relative to the noised latent magnitude, which is plausible but not formally bounded.

**Implementation vs. theory mismatch:**
- Section 5 states: "since applying directly weighted sum usually results in blurry, we ordered the feature pixels by λ̃(t,p)||µ_t,p - X̃_{0,p}||_2 and take the ratio of smaller pixels from X̃_{0,p} and others from µ_t,p." This pixel selection strategy is not present in the theoretical derivation of Eq. (12). It is a heuristic deviation that is introduced without justification. The gap between theory (weighted average) and implementation (pixel-level selection) is non-trivial and could affect the validity of the theoretical guarantees.

---

### Experiments & Results

**Datasets and scale:**
- 9 scenes for single-view, 9 scenes (3 Tanks+6 DTU) for multi-view, 9 YouTube videos for dynamic NVS — these are extremely small evaluation sets by ICLR standards. A single outlier scene can substantially move aggregate metrics, as evidenced by the DepthAnythingV2 asterisk removal in App. F.
- The YouTube videos have no ground truth, preventing pixel-aligned evaluation for the dynamic setting.

**Metrics:**
- The paper uses FID (distribution-level) + SfM-estimated pose metrics (ATE, RPE-T, RPE-R via Particle-SFM) as primary metrics. There are two serious concerns:
  1. **FID on 9–25 images per scene** is statistically unreliable; FID requires thousands of samples for valid estimates. The paper does not clarify whether FID is computed per-scene or pooled, nor what the reference distribution is.
  2. **The pose metrics measure consistency of generated views with each other** (via SfM on synthetic images), not accuracy versus ground-truth camera trajectories. A method that generates internally consistent but geometrically wrong views would still score well. This is a proxy for geometric consistency, not geometric accuracy.
- LPIPS (App. I) and PSNR/SSIM (App. K, Dycheck only) are relegated to appendices. The authors justify avoiding paired metrics for single-view NVS due to scale ambiguity in depth estimation — this is legitimate for monocular relative-depth methods, but could be addressed with scale-invariant metrics or by registering predicted scale to GT scale. Notably, Appendix K does report PSNR/SSIM/LPIPS for Dycheck, suggesting the infrastructure exists; it should be the primary metric for the DTU dataset (which has calibrated cameras and ground-truth depth).

**Baseline selection:**
- Several training-based methods (Sparse Gaussian, SparseNeRF, Text2Nerf) require per-scene overfitting, which is a fundamentally different setup. Including them is reasonable for completeness, but the comparison in Table 1 (where they often show "– –") makes the table confusing.
- **SV3D** (the most directly comparable method: uses SVD, targets NVS) is not compared against in the main paper at all.
- **ZeroNVS** appears only in App. M with a qualitative comparison on a single scene, insufficient for a claim of "significant superiority."
- **ViewCrafter** and **ReconFusion** are absent despite overlapping scope.
- MotionCtrl (Wang et al., 2024) was not designed for NVS — it controls camera motion in video generation — so it is a somewhat unfair comparison point (weak baseline for NVS accuracy).

**Table 1 layout:**
- The multi-view column for Ours (DGS) shows ATE=22.00 vs. Ours (Post) ATE=4.052, a 5× difference in trajectory error. This large gap is acknowledged but not fully analyzed. For the DTU scan scenes (Table C-2), DGS shows ATE up to 32.65 while Post shows 2.42 — why does the multi-view setting degrade DGS so severely? The ablation does not address this regime.

**Ablation coverage:**
- The ablation on inference steps (Table 3) is useful and shows pose accuracy depends heavily on step count.
- The ablation on κ and normalization (Table 4) is appropriate.
- **Missing ablations**: (1) ablation on the base video model (what if SVD is replaced with another video diffusion model?), (2) sensitivity to v_2 and v_3 independently, (3) impact of warping quality on final NVS quality (partial results in App. F cover depth estimator choice but not warp strategy choices), (4) failure cases — the paper shows no examples where the method degrades.

**Computational cost:**
- Ours (Post) requires 1 hour per 25-view sequence on an A6000. This is impractical for most use cases. While the paper acknowledges this, it does not provide comparisons to methods with comparable computation budgets, which would change the competitive landscape.

---

### Writing & Clarity

- The method section is dense but largely navigable. The notation (X_t, p_i, µ̃, etc.) is heavy but self-consistent.
- A critical clarity issue: the implementation diverges from the theory (pixel-level selection heuristic instead of weighted sum from Eq. 12) but is described in a single compressed sentence in the implementation section rather than as an explicit deviation from the theoretical framework.
- There is a broken citation: "Depth Anything ( **?** )" in Section 5 — this appears to be a LaTeX citation resolution failure.
- Algorithm 1 references Eq. (17) for λ̃, but Eq. (17) is defined in Sec. 4.2 which comes after the algorithm is introduced; this requires re-reading.
- Table 1 and Table 2 overlap awkwardly — Table 2's caption and content appear to be duplicated mid-paragraph in Section 5.2, likely a layout artifact.
- Fig. 2 is described as showing the correlation of E_D with σ(t) and E_P with ||Δp||_2, but without knowing the axes scale or fitted r² values, the reader cannot assess how well the linear approximations hold.

---

### Limitations & Broader Impact

- The authors acknowledge the primary limitation: computation time (1 hour for Ours Post).
- **Unacknowledged limitations**:
  1. **Large pose change failure mode**: The Taylor expansion linearization becomes increasingly inaccurate for large Δp, but 360° NVS is demonstrated via iterative chaining (App. H). The error accumulation over iterative chaining is not analyzed.
  2. **Scenario-specific hyperparameter tuning**: The three sets of (v_1, v_2, v_3) are presented as given constants without discussing how a practitioner would select them for a new scenario type.
  3. **Scale ambiguity propagation**: Monocular depth estimation produces relative (up-to-scale) depth maps. The warped views X̃_{0,p_i} are thus geometrically distorted. The method implicitly "corrects" this via the diffusion prior, but the paper doesn't characterize when this correction fails.
  4. **Temporal consistency for dynamic scenes**: The monocular video NVS assumes the depth map from Depth Anything can be used to warp frames with complex motion. Dynamic objects (moving people, animals) will produce wrong warp positions, and the paper does not discuss how the score modulation handles this failure mode.
  5. The broader impact section is formulaic and does not engage with misuse cases (synthetic scene generation for misleading content, etc.).

---

### Overall Assessment

NVS-Solver presents a genuine and interesting contribution: a training-free framework for novel view synthesis that modulates a pre-trained video diffusion model's score function using geometry-guided view warping, with an adaptive blending weight derived from error bound analysis. The theoretical framework is creative and the multi-scenario scope (single view, multi-view, dynamic video) is impressive for a zero-shot method. However, several concerns materially affect confidence in the reported results. First, the primary evaluation metrics (FID on tiny sets, SfM-estimated trajectory errors on synthetic images) are proxies that do not directly measure reconstruction accuracy, and pixel-aligned metrics like PSNR/SSIM are nearly absent despite infrastructure for them on DTU. Second, the most directly comparable baseline, SV3D, is completely absent from quantitative comparisons, and ZeroNVS appears only in a qualitative appendix. Third, the claimed adaptive λ requires scenario-specific hyperparameter tuning (different v_2, v_3 per scenario type), which is a practical limitation inconsistent with the "principled adaptive" framing. Fourth, the implementation diverges from theory via an unexplained pixel-selection heuristic. The contribution is real and the paper was accepted at ICLR 2025, which suggests the reviewers found sufficient merit — but a more rigorous experimental evaluation against SV3D/ZeroNVS with calibrated pixel-aligned metrics on standard benchmarks would substantially strengthen confidence in the claimed superiority.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper proposes NVS-Solver, a training-free novel view synthesis method that leverages pre-trained video diffusion models (SVD) by adaptively modulating the diffusion sampling process with warped input views. The key innovation lies in the theoretical formulation of NVS as guided diffusion sampling, where the score function is modulated with scene priors through an adaptive weighting parameter λ derived from error bound analysis. The method handles single-view, multi-view, and dynamic scene scenarios without additional training.

### Strengths
1. **Novel problem formulation**: The paper provides a principled theoretical framework connecting diffusion sampling with NVS through score modulation, deriving how to optimally balance diffusion estimation error (ED) and warping error (EP) via adaptive weighting (Section 4.2).

2. **Training-free and versatile**: Unlike existing NVS methods requiring per-scene optimization or training, this approach works zero-shot across diverse scenarios (single/multiple views, static/dynamic scenes), making it highly practical.

3. **Strong quantitative results**: Tables 1-2 demonstrate consistent improvements over SOTA methods. For single-view NVS, the method achieves FID of 165.12 vs. 179.24 (MotionCtrl) with significantly lower pose errors (ATE 0.767 vs. 3.851).

4. **Comprehensive evaluation**: The paper evaluates on multiple datasets (Tanks and Temples, DTU, YouTube videos) with appropriate metrics (FID, ATE, RPE) and provides extensive ablation studies on inference steps, sampling strategies, and weight functions.

5. **Theoretical grounding**: Appendix A provides proofs that the modulated score is more accurate than inpainting-based guidance and that posterior sampling preserves the data manifold.

6. **Reproducibility**: Code is publicly available at the provided GitHub link, implementation details are specified, and the method uses standard pre-trained models.

### Weaknesses
1. **Computational inefficiency**: The superior posterior sampling variant takes ~1 hour to render 25 views (Section 5.3), which is 10x slower than the directly guided sampling (6 minutes) and significantly limits practical applicability.

2. **Dependency on external depth estimation**: The method relies on off-the-shelf depth estimation (Depth Anything), and as shown in Table F-4, performance varies with depth estimation quality, introducing an external dependency not fully analyzed.

3. **Limited comparison with diffusion-based NVS methods**: The paper lacks comparison with SV3D (Voleti et al., 2024), a highly relevant concurrent work that also uses video diffusion for novel view synthesis. ZeroNVS comparison is relegated to Appendix M.

4. **Empirical approximations in error formulations**: The error formulations ED = v2σ(t) and EP = v3||Δp||2 (Section 4.2) are based on empirical observations rather than rigorous theoretical derivation, weakening the theoretical contribution.

5. **Missing ground-truth comparison for dynamic scenes**: Dynamic scene evaluation relies solely on pose metrics without LPIPS/PSNR against ground truth, making quality assessment incomplete for this scenario.

### Novelty & Significance
The work presents genuine novelty in formulating NVS as score-modulated diffusion sampling with adaptive weighting based on error analysis. The training-free paradigm using pre-trained video diffusion models is innovative and practically significant. However, the connection to existing diffusion-based NVS works could be better contextualized. The theoretical framework, while interesting, contains empirical approximations that partially undermine its rigor.

### Suggestions for Improvement
1. **Add comparison with SV3D**: This is the most directly related work using video diffusion for multi-view synthesis and should be included in main comparisons.

2. **Improve computational efficiency**: Investigate accelerated sampling techniques (e.g., fewer steps, distilled models) to make posterior sampling more practical.

3. **Strengthen theoretical justification**: Provide more rigorous derivation for the error formulations or acknowledge the empirical nature more clearly in the main text.

4. **Add ground-truth metrics for dynamic scenes**: Include LPIPS/PSNR comparisons where ground truth is available (e.g., on DyCheck dataset, partially shown in Appendix K).

5. **Analyze failure cases**: The paper could benefit from discussing limitations and failure modes more thoroughly, particularly regarding depth estimation errors or extreme pose changes.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments
1. **Comparison with SV3D (Voleti et al., 2024) on a shared benchmark**: SV3D is the most direct competitor — it also uses a latent video diffusion model for NVS but requires training. A head-to-head comparison on, e.g., Google Scanned Objects or CO3D would directly quantify the cost of zero-shot versus supervised adaptation and is conspicuously absent from the main tables.
2. **Standard object-centric benchmarks (CO3D, ShapeNet-NVS, NeRF-Synthetic)**: All current evaluations use Tanks & Temples and DTU, which are large outdoor/indoor scenes. Including object-centric datasets would test whether the score-modulation strategy generalizes across scene scales and enable comparison with a far wider set of published baselines (e.g., PixelSplat, MVSplat, ZeroNVS on their home turf).
3. **Pixel-aligned reconstruction metrics (PSNR, SSIM) as primary metrics on held-out views**: The paper confines PSNR/SSIM/LPIPS to Appendix I for a subset of scenes, relying primarily on FID and camera-pose errors. For ICLR, reviewers will expect paired metrics on ground-truth held-out views as the main table; the pose-estimation pipeline used to compute ATE/RPE introduces its own error sources. A dedicated evaluation split (fixing the exact ground-truth target frames) would make the quantitative story far more compelling.
4. **Scaling from 1 → N input views**: The paper shows single-view and two-view inputs but provides no systematic study of how quality improves (or plateaus) as the number of conditioning views grows to 3, 4, 6, or 12. Such a curve would directly demonstrate the framework's data efficiency.
5. **Runtime–quality Pareto curve**: Ours (DGS) runs in 6 minutes and Ours (Post) takes 1 hour for 25 views. A table comparing wall-clock time versus FID/LPIPS for the proposed method and all baselines would help practitioners calibrate trade-offs and is standard practice in diffusion-speed papers.
6. **Robustness to large angular baselines**: The current experiments cover limited viewpoint changes. Evaluating on scenes with 90°, 180°, and 360° view changes — and showing per-angle quality curves — would strengthen claims about the method's generality for large-baseline NVS.
7. **Evaluation on RealEstate10K for indoor scenes**: This widely used benchmark for room-scale NVS would open comparison to geometry-free and feed-forward baselines (e.g., Watson et al. 2023, ReconFusion) and would validate the method on the type of sequences SVD was trained on.

### Deeper Analysis Needed
1. **Sensitivity analysis of v₁, v₂, v₃**: The adaptive λ formula relies on three empirically fixed constants. The paper would benefit from a grid search or ablation showing how robust the method is to their values, since a practitioner applying the method to a new depth estimator or scene type would need to re-tune them.
2. **Formal convergence analysis of the posterior sampling loop**: The paper proves the DGS update stays on the data manifold (Appendix A.2) but does not analyze whether repeated posterior sampling steps converge, at what rate, or whether an optimal stopping criterion exists. Even an empirical loss curve over iterations would help.
3. **Failure mode taxonomy**: A structured analysis identifying the conditions under which the method degrades — e.g., textureless regions, specular surfaces (non-Lambertian), very large depth estimation errors, or thin structures — would deepen understanding. The paper briefly mentions non-Lambertian effects but does not characterize failure systematically.
4. **Tighter error bound for non-linear depth error ∆D**: The current bound on E_T is derived under a linearized Taylor approximation and assumes ||E_P||₂ ≈ v₃||Δp||₂. A discussion of when this linear approximation breaks down (e.g., for curved surfaces, high-curvature objects, or very large pose changes) and how the bound degrades would clarify the theoretical regime of validity.
5. **Connection to the posterior sampling literature (DPS, DDNM, DDRM)**: The modulated score can be viewed as an implicit likelihood term in a diffusion posterior. A more explicit derivation relating Eq. (14) to the DPS framework (Chung et al., 2023) would situate the contribution more precisely in the guided-diffusion landscape and clarify what is genuinely novel versus shared with existing work.
6. **Ablation on the choice of video diffusion backbone**: The entire method is built on SVD. Testing with a second video diffusion model (e.g., CogVideoX, or an open image-to-video model) would establish how much of the performance comes from the proposed algorithm versus the SVD backbone's inherent scene understanding.

### Untapped Applications
1. **Autonomous driving simulation**: Given the monocular video input, applying the method to dashcam footage (KITTI, nuScenes) to synthesize novel driver viewpoints would be a high-impact application with existing evaluation protocols and baselines.
2. **Medical volumetric imaging from sparse projections**: The theoretical framework — using warped views as score-modulation priors — maps naturally onto sparse-view CT/MRI reconstruction, where few X-ray projections guide a diffusion model. This cross-domain transfer would significantly broaden the paper's impact.
3. **Free-viewpoint video for sports and events**: Extending the monocular dynamic NVS to broadcast-quality sports content (e.g., converting a single-camera soccer feed into multi-angle replay) is a commercially valuable and technically demanding test of temporal consistency.
4. **Downstream 3D Gaussian Splatting initialization**: The paper shows a mesh reconstruction result (Appendix J) but does not use the synthesized multi-view set as input for 3D Gaussian Splatting to produce a persistent radiance field. This would turn the method into a full reconstruction pipeline and enable quantitative comparison on 3D metrics (Chamfer distance, F-score).
5. **Satellite and aerial imagery**: NVS from satellite imagery (e.g., DFC2019 dataset) involves highly non-standard depth profiles and camera geometries — testing robustness in this domain would reveal generalization beyond the common photographic regime.

### Visualizations & Case Studies
1. **Diffusion trajectory visualization**: Showing the intermediate denoised estimates µ̃_{t,p_i} at several steps (t = T, 0.75T, 0.5T, 0.25T, 0) would illustrate *how* the warped prior gradually releases control to the generative model and give readers intuitive insight into the adaptive λ schedule.
2. **Side-by-side error maps with varying pose angle**: Plotting per-pixel L1 or LPIPS error as a function of the angular deviation from the input view (e.g., 10°, 30°, 60°, 90°) would precisely characterize where and how errors accumulate, directly connecting the theoretical E_T ≈ v₃||Δp||₂ bound to observed behavior.
3. **Depth-error impact case study**: A controlled experiment where the depth map is intentionally degraded (adding Gaussian noise of increasing variance) and the resulting NVS quality is tracked would concretize the depth-warping error analysis and guide practitioners on minimum depth quality requirements.
4. **Cross-view consistency heatmap**: For a synthesized video, visualizing optical-flow consistency between adjacent synthesized frames (or re-projected epipolar error) would demonstrate temporal coherence beyond pose metrics and address a natural reviewer question about inter-frame consistency.
5. **Attention map visualization from SVD**: Visualizing which spatial-temporal tokens SVD attends to when the warped guidance is injected would clarify whether the model uses the prior as expected (e.g., attending to the warped regions for texture, and to learned priors for occluded areas), building trust in the theoretical motivation.

### Natural Next Steps
1. **Pose-controllable video diffusion distillation**: The conclusion hints at this, and it is the most natural follow-on — using the high-quality posed outputs of NVS-Solver as pseudo-labeled training data to fine-tune SVD with explicit camera conditioning, producing a fast, single-pass NVS model without the 1-hour inference cost.
2. **Joint depth estimation and view synthesis**: Currently, depth estimation and NVS are decoupled. A feedback loop where synthesized novel views from multiple angles refine the monocular depth estimate (similar to MVS depth completion) could break the depth-quality ceiling that currently limits the method.
3. **Extension to 4D (video NVS + temporal synthesis)**: The current dynamic scene method synthesizes novel views at the same timestamp as input frames. Extending to arbitrary target timestamps — i.e., simultaneously controlling camera pose and time — would enable full 4D scene exploration from monocular video.
4. **Replacing point-based warping with learned flow**: The method uses classical depth-based warping which struggles with specularities and thin structures. Substituting a learned image-based rendering module (e.g., a neural warping network trained on synthetic data) for the geometric warping step while keeping the diffusion-guided refinement could address the non-Lambertian failure cases identified in the ablation.
5. **Amortized inference via cached guidance**: The posterior sampling loop currently reruns the full U-Net backward pass at every step, making it 10× slower than DGS. Caching or amortizing the gradient computation (e.g., via a lightweight guidance network trained on NVS-Solver outputs) could bring the quality of posterior sampling at near-DGS speed, making the method practical for real-world deployment.

# Report: Potentially Missed Related Work
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

1. **Zero-1-to-3: Zero-shot One Image to 3D Views** — Liu et al. (2023, ICCV).
   Why potentially missed: This is a seminal work in zero-shot novel view synthesis using diffusion models. While the paper cites ZeroNVS (Sargent et al., 2024), which is a successor to Zero-1-to-3, it does not appear to cite this foundational paper that established the paradigm of leveraging pre-trained diffusion models for geometry-aware view synthesis without training, which is the core contribution of the NVS-Solver paper.

2. **DynIBaR: Neural Dynamic Image-Based Rendering** — Li et al. (2023, CVPR).
   Why potentially missed: The paper presents extensive results for "NVS from Monocular Videos of Dynamic Scenes" (Section 5.2) and compares against Gaussian Splatting methods (Deformable-Gaussian, 4D-Gaussian). DynIBaR is a standard, state-of-the-art baseline for neural rendering of dynamic scenes from monocular video. Its omission from the related work or experimental comparison is significant given the problem scope.

3. **Wonder3D: Single Image to 3D using Cross-Domain Diffusion** — Long et al. (2024, CVPR).
   Why potentially missed: This work addresses the same problem of generating multi-view images from a single image using diffusion models. It is a relevant contemporaneous approach that employs cross-domain diffusion, offering a different methodology to the score modulation proposed in NVS-Solver, yet it is not cited in the related work section.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Decision: Accept (Poster)
Binary: Accept

=== CALIBRATION EXAMPLE 8 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

### Title & Abstract

- The title accurately reflects the paper's contribution: extending the visual context window of LMMs for long video understanding.
- The abstract clearly states the problem (LMMs struggle on long video), the method (visual context window extension via adapted YaRN + progressive pooling), and key results (MLVU beats GPT-4o; 45% memory reduction at 256 frames).
- **Concern**: The abstract's claim that the method "outperforms GPT-4o" is stated without the crucial caveat that GPT-4o uses 0.5 fps (extremely sparse sampling) while the authors use 256–512 frames. This omission makes the claim misleading. The abstract should contextualize this comparison honestly.
- **Concern**: The abstract says the method works "without retraining on long video datasets," but Section 4.2 discloses that a fine-tuned variant (10K image-text pairs, LoRA) is also used. The distinction between tuning-free and fine-tuned results is not surfaced in the abstract.

---

### Introduction & Motivation

- The central observation — that visual and language tokens occupy different effective context windows within the same LLM decoder — is a genuine and insightful framing. The Figure 1a evidence (performance degrades for visual but not language as sequence grows) is compelling.
- The t-SNE visualization (Figure 1b) showing visual and language embeddings cluster separately is a well-known fact about cross-modal models (e.g., documented in LLaVA and related work). Its use here as *motivating evidence* is fine, but the causal link claimed — that modality gap in embedding space causes context window mismatch — is asserted, not rigorously established. The two phenomena (embedding cluster separation, context window mismatch) could be entirely independent; the paper treats them as related but does not provide mechanistic justification.
- **Concern**: The introduction frames the problem as if the visual context window has never been separately considered in LMM design. However, works like LongVA (Zhang et al., 2024) already discuss the visual vs. language sequence length gap. The novelty of the *framing* should be more carefully bounded.
- The contributions are clearly listed and mostly accurate. The third bullet ("mitigating reducing memory consumption") contains a grammatical error suggesting hasty editing, but more importantly, the description undersells the fact that progressive pooling is the *only* component addressing memory, not a secondary contribution.

---

### Related Work

- The key baselines (LongVA, LongVILA, MovieChat, LLaMA-VID) are appropriately cited and positioned.
- The RoPE extension literature (PI, NTK, YaRN) is accurately summarized.
- **Missing reference**: Flash-Attention variants (e.g., ring attention, streaming attention) that address the quadratic cost of attention over long sequences are not discussed, even though the paper explicitly invokes attention complexity as a motivation. These are directly relevant alternative solutions to the memory problem.
- **Missing reference**: Gemini's approach to long context video (Gemini 1.5) and InternVL2's handling of long sequences are absent from related work, despite being strong contemporary baselines.
- **Missing reference**: Works on visual token compression (e.g., TokenPacker, LLaVA-PruMerge) are not cited in the context of progressive pooling, making the novelty of the pooling approach harder to evaluate.
- The comparison with LongVA deserves more nuance. LongVA extends context windows by continuing LLM pre-training on long *text*, then transfers to vision — a different but related philosophy to this paper. The positioning ("inevitably introduces high computational costs") applies equally to this paper's fine-tuning variant.

---

### Method / Approach

**Visual Context Window Extension:**
- The core idea — replacing YaRN's scaling factor `s = L_lang_test / L_lang_train` with `s = L_v_test / L_v_train` — is simple and reasonable. However, the paper **never explicitly states the value of L_v_train** for the backbone model (LLaVA-OneVision). A reader must infer that 32 frames × 196 tokens/frame = 6,272 visual tokens, while the LLM's language context window is 128K. This is a critical implementation detail omitted from the main paper.
- **Concern**: The scaling factor is defined at the token level, but visual tokens and language tokens have fundamentally different semantic roles. A sentence of 6,272 word tokens represents semantically rich sequential dependencies across a long document; 6,272 visual tokens represent spatial patch features across 32 video frames. The claim that RoPE frequency scaling — designed for sequential position in text — transfers meaningfully to video frame tokens with 2D spatial structure requires more justification. The paper does not address how 2D spatial positions within a frame interact with the 1D sequential position used for RoPE.
- **Concern**: Equation (5) and the surrounding discussion simply reproduce YaRN's formulation. The only change is the definition of `s`. The paper should clearly isolate this single-variable change rather than re-deriving the entire YaRN framework, which occupies considerable space without adding insight.
- The hyperparameters α=1, β=32 are inherited directly from YaRN without justification for whether they are appropriate for visual tokens. An ablation on these values is absent.

**Progressive Pooling:**
- The strategy is intuitive: first frame per group retains high resolution, remaining frames are pooled aggressively.
- **Key unvalidated assumption**: The paper assumes "the first frame preserves rich spatial, fine-grained information compared to the other frames within the group." This is asserted without empirical justification. In many video scenarios (e.g., sports highlights, action recognition), later frames within a temporal segment carry equally or more important spatial detail. No experiment probes whether using the *last* frame or a *random* frame at high resolution within each group would perform similarly.
- **Concern**: The grouping assumption (each group of K frames = one "event") is not validated. Real-world videos do not cleanly segment into K-frame events, and no analysis of how misalignment between the grouping stride K and actual scene boundaries affects performance is provided.
- The progressive pooling is described as reducing memory by 45%, but the paper only reports *peak* GPU memory. There is no analysis of inference throughput (tokens/second, latency) or how progressive pooling affects prefill time vs. decode time, which matters for practical deployment.

---

### Experiments & Results

**Baselines and Fairness:**
- **Major concern**: The comparison with GPT-4o on MLVU (Table 2) is the headline result ("our method significantly outperforms all comparison models, even surpassing GPT-4o"), but GPT-4o uses 0.5 fps, meaning it processes only ~2–4 frames for a 3–8 minute video, while the proposed method uses 256–512 frames. This is not a fair comparison and the paper does not adequately flag this. The result is presented as a capability comparison, but it's actually a frame-budget comparison. Surpassing GPT-4o with 256 frames while GPT-4o uses ~4 frames is not a surprising outcome and does not demonstrate superior architectural design.
- **Concern**: In Table 1 (VideoMME), the baseline is LLaVA-OneVision at 32 frames (58.2 overall). The proposed method at 256 frames achieves 61.3. But Table 4 shows that even LLaVA-OneVision with direct extrapolation to 256 frames achieves 56.2. The improvement from 56.2 (direct extrapolation) to 61.0 (proposed method, no progressive pooling) is 4.8 points; the improvement from 32-frame baseline to proposed is 3.1 points. These are real but modest gains.
- **Inconsistency**: In Table 1, the 512-frame setting (60.6 overall) underperforms the 256-frame setting (61.3). If the method "consistently improves performance as the number of video frames increases" (abstract claim), this contradiction demands explanation. The paper partially addresses it in Section 4.3, attributing it to attention distraction in shorter videos, but this undermines the central thesis.

**Ablations:**
- Table 4's ablation (direct ext. → YaRN → visual context window YaRN → + fine-tuning) is the most important ablation and is well-structured.
- **Missing ablation**: What is the effect of visual context window extension on *short* video tasks or image understanding? If modifying RoPE frequencies for visual tokens hurts image understanding performance, this is a significant limitation not reported.
- **Missing ablation**: The paper fine-tunes on image-text pairs (not video) after RoPE extension. Why does fine-tuning on images — which have static spatial structure — help with video temporal understanding? This is unexplained and unintuitive.
- Table 5's ablation on progressive pooling parameters is thorough for the hyperparameters explored, though the optimal configuration (sh=2, sl=8, K=4) differs from the default used in the main tables (K=4 but different grouping). The mapping between Table 5's experiments and the main result tables should be explicit.
- **Missing ablation**: The paper never tests whether progressive pooling alone (without visual context window extension) provides meaningful benefits, making it impossible to disentangle the two contributions.

**Statistical Reporting:**
- No variance, confidence intervals, or statistical significance tests are reported across any table. For benchmarks like VideoMME and LongVideoBench with thousands of questions, this is feasible and expected at an ICLR-level venue.

**V-NIAH (Appendix A.1):**
- The needle-in-a-haystack test (Figure 5) provides useful additional validation. However, the results are presented as binary heatmaps without a quantitative summary score, making it difficult to compare different configurations at a glance.

---

### Writing & Clarity

- The paper has significant clarity issues in several key places:
  - The exact value of L_v_train for the backbone model is never stated, making Section 3.1 impossible to verify without reverse-engineering.
  - Figure 1 caption conflates two subfigures with different y-axes and measurement scales; the negative perplexity as a proxy for language understanding performance requires more explanation for readers unfamiliar with this convention.
  - Figure 3 (progressive pooling pipeline) is largely unreadable due to OCR/rendering artifacts in the parsed version (though this may be a PDF extraction issue).
  - The paper's central claim — that the *ratio* s = L_v_test / L_v_train (rather than L_lang_test / L_lang_train) is the key change — is buried in Section 3.1 without sufficient emphasis. This is the entire technical contribution of Section 3.1, and it should be stated more prominently.
- Figure 7's visualization (Appendix A.5) shows that even after context window extension, visual and language embeddings still cluster separately. The paper correctly interprets this as expected, but this also means the theoretical motivation (modality gap → context window mismatch) remains unresolved by the proposed method. The paper implicitly acknowledges this disconnect but does not follow through on its implications.

---

### Limitations & Broader Impact

- The paper identifies that dense frame sampling can hurt performance on short videos (Table 3, 512-frame results on (8,15] and (15,60] intervals), suggesting the need for adaptive frame sampling strategies. This is acknowledged.
- **Missing limitation**: The method is only validated on one backbone (LLaVA-OneVision 7B). Whether the visual context window definition and YaRN adaptation generalize to models with different visual encoders (e.g., CLIP vs. SigLIP), different modality projection modules (cross-attention vs. MLP), or different positional encoding schemes (absolute vs. relative) is unknown.
- **Missing limitation**: The progressive pooling assumes fixed grouping (K frames per group), which does not adapt to content. A video with rapidly changing scenes will suffer more from low-resolution frames than a static-background video. No failure analysis addresses content-dependent performance degradation.
- **Missing limitation**: The fine-tuning variant uses only 10K image-text pairs and achieves meaningful gains. The paper does not discuss whether scaling this data budget further would continue improving performance, which matters for understanding whether the method is truly "training-free" in spirit.
- The paper does not discuss broader impact or societal considerations, though this is less critical for a systems/methods paper at ICLR.

---

### Overall Assessment

This paper makes a reasonable and clearly presented contribution: recognizing that visual and language tokens occupy different effective context windows within an LMM's decoder, and adapting YaRN's positional interpolation scheme accordingly. The core technical change — using L_v_test/L_v_train as the scaling factor rather than the language context window ratio — is simple but empirically effective, yielding consistent improvements on multiple benchmarks without retraining on long video data. The progressive pooling strategy is a practical engineering complement.

However, several concerns weigh against acceptance at ICLR. First, the technical novelty is thin: the entire method is one substitution in YaRN's scaling factor, which the paper arguably overinflates by reprinting YaRN's derivation. Second, the flagship result — "outperforming GPT-4o" on MLVU — is misleading given that GPT-4o uses ~0.5 fps (far fewer frames), and the paper does not adequately foreground this. Third, the theoretical motivation (modality gap → context window extension) is suggestive but not rigorously established; the t-SNE evidence shows clusters exist, not that YaRN addresses the right underlying problem. Fourth, critical details (L_v_train value, justification for α and β choices for visual tokens, why image fine-tuning helps video tasks) are omitted or left unjustified. Fifth, a generalization experiment beyond LLaVA-OneVision is entirely absent. The method is practical, low-cost, and beneficial for the community, but the paper as written overstates its theoretical contributions and cherry-presents its empirical comparisons in ways that require substantial revision to meet ICLR's bar for intellectual rigor.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper addresses long video understanding in Large Multimodal Models (LMMs) by proposing a visual context window extension method that adapts YaRN to extend the effective context window for visual tokens without retraining on long video-text pairs. The authors identify that visual and language embeddings occupy distinct clusters in the latent space, leading to different effective context windows, and further introduce a progressive pooling strategy to reduce memory consumption by ~45% while maintaining performance.

### Strengths
1. **Novel insight on modality-specific context windows**: The empirical observation that visual and language tokens have different effective context windows (visualized via t-SNE in Figure 1b) is a valuable contribution that explains why directly extending visual tokens fails and provides a clear motivation for the proposed approach.

2. **Training-free solution with strong empirical results**: The method achieves impressive results on multiple benchmarks (VideoMME, MLVU, LongVideoBench), outperforming GPT-4o on MLVU and LongVILA-8B on VideoMME without requiring expensive long video-text pretraining—addressing a key practical bottleneck in the field.

3. **Comprehensive ablation studies**: Tables 4 and 5 provide thorough analysis of both the context window extension (comparing direct extrapolation, YaRN, and the proposed method) and progressive pooling parameters, demonstrating the contribution of each component.

4. **Needle-in-a-Haystack validation**: The V-NIAH experiment (Figure 5) demonstrates that the model can effectively extend from 32 frames to 1024 frames without fine-tuning, providing concrete evidence of successful context extension.

5. **Memory efficiency with maintained performance**: The progressive pooling strategy reduces memory by ~45% (73GB to 40GB) while actually improving performance (61.0 → 61.3 on VideoMME), addressing a critical practical deployment concern.

### Weaknesses
1. **Limited architectural novelty**: The core technical contribution—applying YaRN to visual tokens—is relatively straightforward once the context window distinction is identified. The scaling factor redefinition is simple and may not constitute a substantial algorithmic contribution.

2. **Heuristic progressive pooling assumptions**: The progressive pooling strategy assumes "the first frame preserves rich spatial, fine-grained information" and that consecutive frames share static backgrounds. These assumptions may fail for videos with rapid motion, scene changes within groups, or where key details appear in non-first frames—no analysis of such failure modes is provided.

3. **Performance degradation on shorter videos with dense sampling**: Table 3 shows performance drops on shorter duration intervals (8,15] and (15,60] when using 512 frames compared to baseline. The attributed cause ("attention distraction") is underexplored, and the suggested remedy (different sampling strategies) is not implemented or validated.

4. **Single backbone evaluation**: All experiments use LLaVA-OneVision 7B with Qwen2 decoder. Generalizability to other architectures (e.g., InternVL, different position encoding schemes) is not demonstrated, limiting claims of broad applicability.

5. **Fine-tuning dependency for optimal results**: Table 4 shows fine-tuning with 10K image-text pairs improves results (61.0 → 61.8), suggesting the "training-free" claim is somewhat qualified—the best results still benefit from additional training.

### Novelty & Significance
The paper offers moderate novelty with good practical significance. The insight about separate visual/language context windows is valuable and underexplored. However, the technical solution (adapting YaRN) is incremental. The training-free aspect and strong empirical results make this a practically relevant contribution for ICLR's audience, particularly given the computational costs of long video pretraining.

### Suggestions for Improvement
1. **Expand architectural diversity**: Evaluate the method on at least one additional LMM architecture to demonstrate generalizability beyond LLaVA-OneVision.

2. **Analyze progressive pooling failure modes**: Provide quantitative analysis on video types where progressive pooling underperforms (e.g., fast-action videos, videos with frequent scene changes) and discuss limitations explicitly.

3. **Implement adaptive sampling**: Develop and validate the suggested remedy for performance drops on shorter videos rather than leaving it as future work.

4. **Deeper analysis of the modality gap**: The visual-language embedding separation is a key observation—additional analysis on whether context extension affects this gap (beyond the brief Appendix A.5) would strengthen the theoretical contribution.

5. **Provide code and model checkpoints**: For reproducibility and to facilitate adoption by the community, especially important for a training-free method that should be easily applicable to existing models.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments

1. **Generalizability across backbone models**: The method is only validated on LLaVA-OneVision-7B. Testing on at least 2–3 other architectures (e.g., InternVL2, Qwen2-VL, MiniCPM-V) at equivalent frame counts would demonstrate that visual context window extension is a general-purpose technique rather than a LLaVA-specific artifact — a key concern for ICLR reviewers.

2. **Fair comparison with GPT-4o on MLVU**: The paper compares its 256-frame method against GPT-4o at only 0.5fps. The paper would be stronger with either (a) a GPT-4o comparison at a matched number of frames/tokens, or (b) a clear theoretical argument for why this comparison is appropriate, so the MLVU headline result is bulletproof.

3. **Fine-tuning with video-text pairs vs. image-text pairs**: The fine-tuning step uses 10K *image*-text pairs (ALLaVA dataset), which is unconventional for video understanding. An ablation comparing image-text fine-tuning vs. an equivalent number of video-text pairs would clarify whether temporal content is needed in the adaptation data, potentially simplifying practitioners' workflows.

4. **Scalability study beyond 512 frames**: The V-NIAH appendix experiment shows 1024-frame capability, but no benchmark performance at 1024 frames is reported. Even one benchmark column at 1024 frames would support the "scales as frames increase" claim more convincingly and differentiate from methods capped at 256.

5. **Hyperparameter sensitivity for α and β**: The YaRN hyperparameters α=1, β=32 are taken from the language-domain defaults without ablation for the visual domain. A 3×3 grid search on these (α ∈ {0.5, 1, 2}, β ∈ {16, 32, 64}) at 256 frames would validate the choice and show robustness.

6. **Throughput / latency benchmarking**: The paper reports GPU memory savings but not inference time. A wall-clock time comparison (progressive pooling vs. baseline at matched memory budgets) would help practitioners understand the full efficiency profile.

7. **Short-video task regression test**: All benchmarks focus on long video. A small experiment on MSVD-QA or ActivityNet-QA would confirm the method does not harm short-video performance — an important sanity check for adoption.

8. **Adaptive frame sampling strategy experiment**: The paper acknowledges in Section 4.3 that dense sampling hurts shorter videos and suggests adaptive sampling as a remedy, but never tests it. Even a simple rule-based heuristic (e.g., cap frames at video_duration × N fps) would close this acknowledged gap.

---

### Deeper Analysis Needed

1. **How is the visual context window L_v_train determined?** The scaling factor *s* is defined as L_v_test / L_v_train, but L_v_train is never precisely defined or empirically justified. Is it the number of visual tokens per pre-training batch? The maximum sequence length seen by the visual encoder? A rigorous definition — and sensitivity analysis — would strengthen the theoretical grounding considerably.

2. **Attention map analysis before vs. after visual context window extension**: The t-SNE cluster visualization (Figure 1b / Figure 7) shows modality separation but does not reveal *how* attention patterns change with longer visual sequences. Attention heatmaps over frame positions would reveal whether extended context causes the model to genuinely integrate distant frames or still primarily rely on local context.

3. **Convergence analysis for the RoPE interpolation in the visual domain**: YaRN has theoretical and empirical support in the language domain; an analogous analysis for visual tokens is absent. Even a brief argument about why RoPE frequency dimensions in visual tokens exhibit analogous high-/low-frequency structure (or differ) would deepen the theoretical contribution.

4. **Layer-wise analysis of modality gap**: Figure 7 shows the modality gap persists after visual context window extension, but only at three specific layers. A full layer-by-layer plot of intra-cluster vs. inter-cluster distance would reveal whether the gap narrows at any depth and inform future work on closing it.

5. **Ablation on progressive pooling's assumption of temporal locality**: Progressive pooling assumes frames within a group share a background (same event). For highly dynamic content (action sports, multi-scene documentaries), this assumption likely breaks down. A per-genre breakdown of benchmark scores (e.g., movie vs. surveillance vs. sports subsets within MLVU) would quantify when this assumption helps and when it hurts.

6. **Analysis of the grouping stride K effect**: The paper states "larger grouping strides lead to greater intra-group scene variation" but does not quantify scene-change rate vs. performance drop. Connecting this to optical flow or scene-transition frequency would provide principled guidance for choosing K.

---

### Untapped Applications

1. **Temporal grounding and dense video captioning**: The paper focuses on QA benchmarks. Applying the method to temporal grounding (e.g., ActivityNet Captions, QVHighlights) would show whether improved long-context handling translates to precise timestamp prediction — a practically important and distinct capability.

2. **Embodied AI and robotics trajectories**: Long robot demonstration videos require temporal coherence over hundreds of frames. The progressive pooling approach, which preserves the "first frame" of each event at high resolution, maps naturally onto keyframe-based robot learning and could be evaluated on datasets like EPIC-Kitchens or BridgeData.

3. **Multi-document visual understanding**: The core idea — extending positional embeddings to accommodate sequences outside the pre-training distribution — applies equally to multi-page document understanding (sequences of image patches from PDFs or slides). Testing on DocVQA or SlideVQA multi-page settings would demonstrate generality beyond video.

4. **Streaming / online video inference**: The progressive pooling structure (group-based processing) is architecturally compatible with online/causal inference where frames arrive sequentially. Demonstrating a streaming-compatible variant would expand the practical scope substantially.

5. **Audio-visual long video understanding**: Audio cues (dialogue, music, sound events) are often complementary to visual frames for long video tasks. An extension incorporating audio tokens alongside visual tokens — with separate audio context windows — would be a natural and differentiated contribution.

---

### Visualizations & Case Studies

1. **Frame-attention weight distribution heatmap**: A visualization plotting attention weight (aggregated across heads and layers) as a function of frame index would reveal whether the model attends uniformly to the extended sequence or still concentrates on early/late frames — directly supporting or challenging the claim that visual context is genuinely extended.

2. **Progressive pooling information retention vs. scene-change rate**: A side-by-side visualization of a high-motion video and a static-background video, annotating which frames are pooled aggressively, would make the pooling strategy intuitive and expose when the heuristic may discard salient information.

3. **Failure case gallery**: Adding 2–3 failure examples (e.g., complex multi-event narratives where the model still confuses temporal order, or fast-cut videos where grouping assumptions fail) would give readers an honest calibration of when to apply the method and would make the contribution more credible.

4. **Scaling curve**: A plot of benchmark score vs. number of frames (e.g., 32 → 64 → 128 → 256 → 512 → 1024) for the proposed method vs. the baseline would visualize the headline claim — "consistently improves performance as frame count increases" — more compellingly than the current discrete comparisons.

5. **Memory-accuracy Pareto frontier**: A 2D scatter plot of (peak GPU memory, benchmark accuracy) for all progressive pooling configurations in Table 5 would let practitioners immediately identify the Pareto-optimal operating point for their hardware budget.

6. **Positional embedding frequency visualization before and after scaling**: Plotting the rotational frequencies θ_i^new vs. θ_i for the visual dimensions (analogous to Figure 2) specifically for the trained model would confirm that the rescaling is applied as intended and that different frequency bands behave as theorized.

---

### Natural Next Steps

1. **Learned visual context window size**: The scaling factor *s* is currently a fixed hyperparameter set heuristically. A meta-learning or calibration approach that automatically estimates L_v_train from the model's own attention entropy on held-out videos would remove the one remaining hand-tuned element of the method.

2. **Combining visual context window extension with sparse/linear attention**: The current method still incurs O(N²) attention over the full extended sequence. Integrating with sparse attention patterns (e.g., local + global tokens, Longformer-style) or linear attention would directly address the quadratic bottleneck and enable 2K+ frame inference on single GPUs.

3. **Extending to non-RoPE positional embeddings**: Several competitive LMMs (e.g., those using ALiBi or learned absolute positions) cannot directly use the YaRN-based approach. A generalization of the visual context window concept to these schemes would broaden applicability across the LMM ecosystem.

4. **Closing the modality gap as a complementary objective**: The paper explicitly acknowledges that visual context window extension does not reduce the visual-language modality gap (Figures 1b, 7). A follow-up that combines context window extension with modality alignment regularization (e.g., a contrastive loss between visual and language embeddings during the lightweight LoRA fine-tuning step) could achieve both goals simultaneously.

5. **Dynamic progressive pooling driven by content**: The current pooling strategy is static (fixed K, sh, sl). A content-aware variant — using motion magnitude or feature similarity to decide pooling aggressiveness per group — would adapt to video dynamics and likely improve performance on high-motion content while preserving memory savings on static content.

6. **Long video generation and retrieval**: The same positional mismatch problem exists in video generation (diffusion-based) and video retrieval models. Applying the visual context window extension idea to these settings would extend the paper's impact and open a new research thread.

# Report: Potentially Missed Related Work
Looking at the paper and the related works, I need to check which are already cited and which are genuinely relevant.

**Already Cited Works (REMOVE):**

1. **LongVILA** (Xue et al., 2024) - Explicitly cited in Section 2.2: "LongVILA (Xue et al., 2024) attempted to introduce long video-text pairs into the training of LMMs to expand the context window size." Also appears in references.

2. **LongVideoBench** (Wu et al., 2024) - Explicitly cited throughout the paper: "LongVideoBench (Wu et al., 2024)" is used as an evaluation benchmark and appears in the references section.

**Remaining Works Analysis:**

3. **From Trial to Triumph** (Suo et al., 2025, ICCV) - Addresses visual frame sampling for long video understanding. While this relates to the paper's progressive pooling strategy for frame handling, the 2025 ICCV date indicates this is likely concurrent work published around or after the submission timeframe. For a paper under double-blind review, this is more accurately characterized as concurrent work rather than a "missed" reference.

4. **Video Panels for Long Video Understanding** (2025, arXiv) - Uses visual prompting with video panels, which is a fundamentally different approach (prompting strategy vs. RoPE extension). The connection requires multiple leaps from prompting to positional embedding modification. TANGENTIALLY RELATED.

5. **Video-XL** and **LongVU** - The search results indicate these are "referenced in [3]" without complete bibliographic information, making it difficult to verify their relevance or status as independent works.

Given that the two most directly relevant works (LongVILA and LongVideoBench) are already properly cited, and the remaining works are either concurrent 2025 publications or have insufficient bibliographic details for proper verification:

## Potentially Missed Related Work

No significant potentially missed related work identified.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [6.0, 6.0, 5.0, 5.0]
Average score: 5.5
Decision: Reject
Binary: Reject

=== CALIBRATION EXAMPLE 9 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

### Title & Abstract

- The title accurately captures the dual focus: geometry-awareness and the performance-vs-guarantees tradeoff.
- The abstract clearly states the problem (gap between empirical and theoretical performance of LinTS/Greedy), the method (geometric tracking of the uncertainty ellipsoid), and the key result (minimax optimal Õ(d√T) regret via course-correction).
- One mild concern: the abstract claims the proposed bound is "efficiently computable using data observed from previous decision epochs," but this efficiency claim rests on rank-one SVD updates described only briefly in Remark 3. The efficiency claim is plausible but not fully substantiated in the main text.
- The term "coursecorrect" is used (hyphenated inconsistently as "course-correct" elsewhere) — not a serious issue, but worth noting for clarity.

---

### Introduction & Motivation

- The motivation is well-articulated and important: the gap between the empirical success of LinTS/Greedy and their pessimistic (or absent) frequentist guarantees is a genuine, long-standing problem in the bandits literature.
- The three contributions are clearly delineated and accurately represent what the paper delivers.
- **Concern:** The paper states "Hamidi & Bayati (2020a) confirms this inflation is necessary and LinTS's frequentist regret cannot be improved." This is a strong claim. What Hamidi & Bayati (2020a) shows is that *for the specific LinTS algorithm without the geometry-aware correction*, the inflation is necessary. But the current paper's contribution is precisely to argue that course-correction sidesteps this lower bound. The framing in the intro is accurate only if this distinction is clear — and it is, in context — but the sentence could mislead on first reading.
- The claim that OFUL is "generally NP-hard" is accurate but deserves nuance: OFUL is efficient for specific action sets (e.g., ellipsoids, polytopes with polynomial-time projection), and many practical instances are tractable. Overstating the computational hardness weakens the argument for TS-MR/Greedy-MR's practical advantage.

---

### Related Work

- The three research streams identified (methodological foundations of LB, spectral bandits, data-driven exploration) are appropriate and well-justified.
- The contrast with Hamidi & Bayati (2020a) — scalar thinness coefficient vs. full ellipsoid geometry — is the clearest differentiator and is correctly highlighted.
- The contrast with spectral bandits (Valko et al., 2014; Kocák et al., 2014, 2020) is appropriate; these works exploit graph-signal smoothness, while this paper exploits the covariance geometry.
- **Missing reference:** Kaufmann et al. (2016) and related works on "optimal" exploration-exploitation strategies (e.g., IMED, KL-UCB variants in parametric bandits) could be mentioned, though they are less directly related.
- **Missing reference:** The connection to model-selection bandits (e.g., Pacchiano et al., 2020 is cited, but Foster et al.'s corral and related adaptive algorithm-selection literature could strengthen the positioning of the meta-algorithm idea in Section 6).
- The positioning against Bastani et al. (2021) is fair: they use the minimum eigenvalue (a scalar summary), whereas this paper uses the full spectral profile.

---

### Method / Approach

**POFUL Framework (Section 3):**
- The generalization is elegant. Encoding OFUL (ιt=0, τt=1), LinTS (τt=0, ιt=ι_t^{TS}), TS-Freq (ιt=O(√d)), and Greedy (ιt=τt=0) as special cases of a single framework is a genuine conceptual contribution.
- Definition 2 (feasible pivots) is clear. The probability guarantee P[θ̃_t ∈ E_t^{PVT} | F_t] ≥ 1-δ' is required for each t individually, not uniformly — this is appropriate given the union bound structure.
- **Concern:** Algorithm 1 contains what appears to be a typographic artifact ("θt+1 ← V_{t+1}^{-1} Σ_{s=1} x_s r_s" with a malformed subscript), but since this is a parser artifact from the PDF, I will not penalize it.

**Uncertainty Ratio and Regret Proxy (Section 4.1):**
- The definition of αt = ‖x*_t‖_{V_t^{-1}} / ‖x̃_t‖_{V_t^{-1}} is clever. It quantifies the relative uncertainty of the optimal vs. chosen action.
- **Key logical concern:** αt depends on x*_t, which is the oracle optimal action and is unknown to the algorithm. The paper acknowledges this in Remark 1, and the entire strategy of Section 5 is to find a computable upper bound α̃_t. This is sound in principle. However, the claim in Remark 1 that "Theorem 1 instantly turns into a data-driven regret bound" is only true once α̃_t is established. The reader must be careful not to use Theorem 1 directly as if it were computable.
- The regret proxy µt = αt(1 + ιt − τt) + 1 + ιt + τt: for OFUL (ιt=0, τt=1), µt = αt·0 + 1+0+1 = 2, which matches the known OFUL instantaneous regret bound of 2‖x̃_t‖_{V_t^{-1}} β. This cross-check is reassuring.
- **Concern:** Theorem 1 states R(T) ≤ (formula in omitted picture). Due to PDF parsing, the exact bound cannot be verified. However, based on the surrounding text (Proposition 2 and the elliptical potential lemma), the structure should be Σ_t µ_t ‖x̃_t‖_{V_t^{-1}} β, bounded via Cauchy-Schwarz and the elliptical potential lemma. This structure is standard and likely correct, but the reviewer cannot independently verify the exact constants without the equation.

**Geometric Bound on αt (Section 5, Theorem 2):**
- The strategy of bounding αt using the ellipsoid geometry (projecting the confidence ellipsoid onto S^{d-1} to get the set of potentially optimal actions Ct) is novel and geometrically appealing.
- For Xt = S^{d-1}: x*_t(θ) = θ/‖θ‖ and x̃_t = θ̃_t/‖θ̃_t‖. The bound on αt then reduces to bounding ‖θ*/‖θ*‖‖_{V_t^{-1}} / ‖θ̃_t/‖θ̃_t‖‖_{V_t^{-1}}, which depends on the relative alignment of θ* and θ̃_t with the eigenvectors of V_t. The ellipsoid geometry enters through the spectral structure of V_t.
- **Concern:** Theorem 2 introduces quantities mt, Mt, and an integer k satisfying some conditions entirely contained in an omitted picture block. Without the defining conditions for k, the theorem cannot be evaluated for correctness or tightness. This is a significant gap in the review, though it is a parser artifact. The reader of the actual paper would have this information.
- **Concern about scope:** The main body only addresses the continuous (spherical) action space. For practical applications (clinical trials, recommendation systems), discrete action sets are more common. The discrete case is entirely deferred to the companion arXiv preprint. This is a notable limitation for the conference paper itself.
- **Edge case not discussed:** When ‖θ̃_t‖ is very small (near zero), the formula x̃_t = θ̃_t/‖θ̃_t‖ is numerically unstable. The paper does not discuss this case.
- **Edge case not discussed:** The bound involves λd(Vt), the smallest eigenvalue of Vt. Early in the horizon when Vt ≈ λreg·Id, all eigenvalues are equal and the geometric information is vacuous. The paper should explicitly discuss the warm-up phase and when the bound becomes nontrivial.

**Proofs:**
- All proofs are deferred to Luo & Bayati (2023), the arXiv companion. For an ICLR conference paper, this is an unusual practice. The paper contains essentially no proof sketches beyond high-level intuitions. This makes independent verification impossible from the conference submission alone and is a structural weakness, even if the arXiv preprint is publicly available.

---

### Experiments & Results

**Synthetic Experiments:**
- Three representative examples (standard LB, contextual bandit embedding, prior mean mismatch) cover important failure modes of LinTS/Greedy.
- 100 independent runs with ±2 SE is appropriate.
- The choice of µ̄=8 for Example 1, µ̄=12 for Examples 2 and 3 is ad hoc. The paper defers sensitivity analysis to the companion paper. This is a significant gap: practitioners need guidance on µ̄ selection, and varying µ̄ across examples without a principled rule raises concerns about post-hoc tuning.

**Critical claim in Figure 3, Example 2:** "Greedy-MR outperforms both Greedy and OFUL." If true, this is a strong result. But since OFUL achieves minimax-optimal regret, TS-MR/Greedy-MR can outperform OFUL only in terms of realized (not worst-case) regret. The claim is about empirical performance, which is valid, but should be more carefully qualified.

**Real-World Experiments:**
- **Significant factual concern (Section 7.2):** The paper states "OFUL and TS-Freq perform poorly due to their conservative exploration." This is misleading or incorrect. OFUL is an *optimistic* algorithm that explores *aggressively*. TS-Freq inflates the posterior, which also increases exploration. If they perform poorly empirically, it is because they *over-explore* (too many suboptimal exploratory pulls), not because they are conservative. The correct statement would be "due to their aggressive/costly exploration." This is a factual error in the experimental narrative.
- The three real-world datasets (Cardiotocography, JapaneseVowels, Segment) are converted from classification to contextual bandit problems. The conversion methodology follows Bietti et al. (2021) and Bastani et al. (2021), which is appropriate. However, it would strengthen the paper to note the dimensions d and number of arms K for each dataset.
- Figures 4(d)-(f) show the fraction of OFUL actions over time. The key observation (OFUL actions concentrated early) supports the paper's claim about computational efficiency. This is a useful and informative visualization.

**Missing ablations:**
1. No experiment varying d (problem dimension) to validate the O(d√T) scaling.
2. No experiment directly comparing TS-MR vs. the simpler approach of Hamidi & Bayati (2020a) using the thinness coefficient. The paper's main claim over Hamidi & Bayati (2020a) is "richer geometric information," but this advantage is never quantified empirically.
3. No experiment on the impact of µ̄ on the fraction of OFUL calls and cumulative regret (deferred to companion paper, but important for ICLR readers).

---

### Writing & Clarity

- The POFUL framework is well-explained. Figures 1a and 1b clearly illustrate the relationship between POFUL and its special cases.
- Figure 2 (illustration of Ct on S^{d-1}) is helpful but described only for R^2; a higher-dimensional example or more formal description would help.
- **Clarity issue:** Remark 1 discusses using µ̃_t^2 ≤ 2α̃_t^2(1+ιt-τt)^2 + 2(1+ιt+τt)^2 as an upper bound. The origin of this inequality (likely Jensen/AM-QM) is not stated and would benefit from a brief explanation.
- **Clarity issue:** The term "regret proxy" µt is introduced in Definition 5 but the motivation for its specific form is only explained informally. The derivation linking Proposition 4 to the form of µt is implicit.
- The meta-algorithm (Section 6) is described in prose without pseudocode in the main body (pseudocode deferred to companion paper). Given that TS-MR and Greedy-MR are the main algorithmic contributions, including the pseudocode in the main paper is important.
- The real-data experimental section references Figure 4 sub-panels (a)-(f) but the figure descriptions are truncated due to PDF parsing issues; this is a parser artifact, not a paper issue.

---

### Limitations & Broader Impact

- **Acknowledged limitation:** The paper is explicit that the threshold µ̄ requires tuning and that the course-correction introduces OFUL's computational cost for a fraction of steps.
- **Limitation not acknowledged:** The Corollary 1 bound is Õ(max{µ̄, 2}·d√T). If µ̄ is set to avoid too many OFUL calls, the constant factor inflates the regret bound. The paper never discusses the explicit tradeoff: a small µ̄ gives a tight regret bound but many OFUL calls; a large µ̄ gives fewer OFUL calls but a looser bound. The practical sweet spot is unclear.
- **Limitation not acknowledged:** For Xt = S^{d-1}, computing the OFUL action requires maximizing ⟨x, θ̃_t⟩ + τt‖x‖_{V_t^{-1}}β over S^{d-1}, which has a closed-form solution. But for general convex action sets, this is a second-order cone program, and for discrete action sets with exponentially many arms, it may be NP-hard. The computational discussion in Remark 3 only addresses updating V_t's SVD, not solving the OFUL optimization itself.
- **Limitation not acknowledged:** The analysis assumes Xt = S^{d-1} for the geometric bound (Theorem 2). Real applications rarely have this structure. The discrete case is deferred without providing the key results in the main paper.
- **No broader impact discussion.** For a bandits paper, the downstream applications (clinical trials, recommendation, pricing) carry meaningful ethical implications. While ICLR did not always mandate this section, mentioning it would be appropriate.

---

### Overall Assessment

This paper tackles a well-motivated and technically substantive problem: bridging the gap between the empirical success of LinTS/Greedy and their poor (or absent) frequentist regret guarantees. The POFUL framework is a clean unification, and the idea of using geometric properties of the confidence ellipsoid to construct a data-driven, computable regret proxy is genuinely novel and more sophisticated than prior approaches using scalar summaries (e.g., the thinness coefficient). The course-corrected algorithms (TS-MR, Greedy-MR) achieve minimax-optimal frequentist regret while preserving the empirical efficiency of the base algorithms, and simulations across synthetic and real-world settings are supportive.

However, the paper has several important weaknesses. First, and most fundamentally for peer review, **all proofs are deferred to an arXiv companion paper**, making independent verification of the main theorems impossible from the conference submission alone — this is unusual for ICLR and weakens the reviewing process significantly. Second, the main body addresses only the spherical continuous action space; the practically important discrete action space case is entirely absent. Third, there is a **factual error** in the experimental section claiming that OFUL performs poorly due to "conservative exploration," when OFUL is an aggressively optimistic algorithm. Fourth, the threshold µ̄ is tuned differently across experiments with no principled selection rule discussed in the main paper. Fifth, the key empirical ablation comparing TS-MR against the simpler thinness-coefficient approach of Hamidi & Bayati (2020a) is missing, leaving the practical benefit of full ellipsoid geometry unquantified. These concerns do not undermine the paper's core theoretical contribution, which appears sound, but they collectively reduce the paper's completeness and verifiability at publication. The contribution is meaningful and the paper is appropriate for ICLR, but revisions addressing the missing ablation, the experimental narrative error, and at minimum proof sketches in the main body are warranted.

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the gap between empirical performance and theoretical guarantees in linear bandits, where algorithms like Thompson sampling (LinTS) and Greedy perform well empirically but have pessimistic or non-existent frequentist regret bounds. The authors propose POFUL, a unified framework encompassing OFUL, LinTS, TS-Freq, and Greedy, and develop a data-driven regret bound based on geometric analysis of confidence ellipsoids. Using this, they create course-corrected variants (TS-MR and Greedy-MR) that achieve minimax optimal Õ(d√T) frequentist regret while preserving empirical performance.

### Strengths
1. **Addresses an important and well-known problem**: The gap between empirical performance and theoretical guarantees for LinTS and Greedy in linear bandits is a recognized issue in the literature, making this work highly relevant.

2. **Novel geometric analysis technique**: The paper introduces a sophisticated approach that tracks full geometric properties of the confidence ellipsoid, advancing beyond methods that use single scalar summaries (e.g., minimum eigenvalue in Bastani et al. (2021), thinness coefficient in Hamidi & Bayati (2020a)).

3. **Clean unification framework**: POFUL elegantly encompasses multiple existing algorithms (OFUL, LinTS, TS-Freq, Greedy) as special cases through the inflation parameter ι_t and optimism parameter τ_t, enabling unified regret analysis.

4. **Strong theoretical results**: The course-corrected algorithms achieve the minimax optimal Õ(d√T) frequentist regret (Corollary 1), resolving the suboptimal Õ(d√(dT)) bound for TS-Freq and the absence of guarantees for Greedy.

5. **Comprehensive empirical validation**: Experiments cover three synthetic scenarios (standard linear bandit, contextual bandit embedding, prior mismatch) and three real-world OpenML datasets, demonstrating that course-corrected algorithms rarely need OFUL actions while maintaining robustness.

### Weaknesses
1. **Incomplete theoretical coverage**: Theorem 2 for bounding the uncertainty ratio α_t is only provided for continuous action spaces (X_t = S^{d-1}). The paper states that discrete-action bounds appear in a longer version but does not include them, limiting completeness for a venue like ICLR where self-contained contributions are expected.

2. **Hyperparameter selection unclear**: The threshold μ is set to different values across experiments (μ = 8, 12) without principled guidance for practitioners. While the paper claims moderate μ values work robustly, an adaptive or principled selection method would strengthen practical applicability.

3. **Missing computational analysis**: While the paper claims SVD updates are efficient, there is no empirical comparison of computational overhead between the proposed algorithms and baselines. This is important since OFUL is known to be computationally expensive.

4. **Deferred proofs reduce verifiability**: Key proofs and algorithm pseudocode (TS-MR, Greedy-MR) are deferred to a longer version, making it harder for reviewers to fully assess correctness and completeness.

### Novelty & Significance
The geometric analysis of confidence ellipsoids is genuinely novel—it represents a significant advance over prior scalar-summary approaches. The POFUL framework provides useful conceptual unification. The course-correction mechanism offers a practical solution to the theory-practice gap that could influence both algorithm design and theoretical analysis in sequential decision-making. The work is well-positioned for ICLR's machine learning theory track, though the incomplete treatment of discrete action spaces somewhat limits its scope.

### Suggestions for Improvement
1. Include at least a theorem statement and proof sketch for discrete-action bounds in the main paper or supplement to ensure completeness.

2. Provide guidance on threshold μ selection: either an adaptive method based on data or theoretical analysis of sensitivity.

3. Add computational time comparisons in experiments to quantify the overhead of computing uncertainty ratio bounds.

4. Include the pseudocode for TS-MR and Greedy-MR in the main paper rather than deferring to the longer version.

5. Discuss more explicitly when the geometric analysis might be loose (upper bound quality) and potential tighter alternatives.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments
1. **Ablation on geometric richness vs. scalar summaries**: The paper argues that using the full *d*-dimensional ellipsoid geometry is superior to scalar summaries like the thinness coefficient (Hamidi & Bayati, 2020a) or minimum eigenvalue (Bastani et al., 2021). A direct controlled ablation — replacing the full geometric analysis with these simpler summaries within the same TS-MR/Greedy-MR framework — would quantify exactly how much the richer geometry contributes to tighter regret bounds and better empirical switching behavior.

2. **Sensitivity analysis of threshold µ on a wider range of problems**: The paper notes µ = 8 and µ = 12 are chosen for different examples, and claims robustness across "moderate µ values." A systematic sweep of µ across all synthetic and real-world examples with reporting of regret *and* fraction of OFUL actions would make this claim rigorous and help practitioners choose µ in the field.

3. **Comparison with model-selection bandit baselines**: Pacchiano et al. (2020) (cited in the paper) addresses regret bound balancing and elimination in a related spirit. A direct empirical comparison of TS-MR/Greedy-MR against Pacchiano et al.'s approach would clarify the paper's unique contribution relative to that competing paradigm.

4. **High-dimensional scaling experiments**: All synthetic experiments use *d* = 10 or *d* = 50. Experiments scaling *d* ∈ {10, 50, 200, 500} would show how the geometric analysis and the OFUL-switching fraction behave as the ambient dimension grows, which is particularly relevant for modern high-dimensional applications and ICLR audiences.

5. **Non-i.i.d. or adversarially perturbed action sets**: The current experiments use i.i.d. random or fixed action sets. Testing on structured or adversarially chosen action sets (e.g., nearly collinear actions) would stress-test the geometric analysis and validate whether the confidence ellipsoid tracking degrades gracefully.

6. **Discrete action space experiments**: The paper defers the discrete-action analysis and bounds to the companion arXiv version. Including even one illustrative simulation for discrete action spaces (e.g., a finite-armed bandit embedded in a linear bandit) would make the conference paper self-contained and strengthen claims about generality.

### Deeper Analysis Needed
1. **Tightness characterization of the data-driven regret bound**: Theorem 1 provides an oracle bound and Theorem 2 converts it to a computable one. It would significantly strengthen the theoretical contribution to quantify how much slack is introduced by the ellipsoid-based approximation of α̃_t — e.g., does the data-driven bound track the oracle bound tightly in the examples studied? Plotting the ratio of data-driven bound to realized regret over time would be illuminating.

2. **Distribution of the uncertainty ratio α_t under typical vs. adversarial regimes**: The paper's core insight is that α_t is small in "typical" problem instances (explaining empirical LinTS success) but large in adversarial ones. A formal characterization — even an informal probabilistic argument — of what makes α_t small (e.g., when θ⋆ is well-separated from the ellipsoid boundary) would provide theoretical grounding for the empirical success story and answer the open question posed by Abeille et al. (2017) more completely.

3. **Computational complexity of the full TS-MR/Greedy-MR pipeline**: Remark 3 mentions that SVD can be updated via rank-one perturbations. A formal complexity analysis per time step (including the argmax over the action set) compared with LinTS, Greedy, and OFUL would make the claim of "preserved computational efficiency" precise rather than qualitative.

4. **Gap between minimax optimality and instance-dependent bounds**: The paper achieves Õ(d√T) minimax regret. It would deepen the theoretical contribution to investigate whether instance-dependent (gap-dependent) regret bounds of order Õ(d²/Δ) are achievable for TS-MR and Greedy-MR, analogous to results available for OFUL. This would clarify whether course-correction preserves not just minimax but also gap-dependent optimality.

5. **Impact of misspecified regularization parameter λ_reg**: All theoretical results assume a known and well-chosen regularization parameter. An analysis of robustness to misspecification of λ_reg, or a data-driven method for selecting it, would enhance practical applicability.

### Untapped Applications
1. **Linear mixture MDPs in reinforcement learning**: The POFUL framework and geometric analysis could extend to linear mixture MDPs (where the transition kernel is linear), where a similar tension between Thompson sampling and conservative UCB-based methods exists. Testing whether geometry-aware course correction translates to RL would substantially broaden the paper's impact.

2. **Federated/distributed linear bandits**: Modern deployments often require multiple agents sharing bandit feedback. The geometric analysis of a shared (or aggregated) confidence ellipsoid under communication constraints is a natural extension, and the course-correction idea applies directly: agents could trigger OFUL actions locally when the shared ellipsoid is poorly conditioned.

3. **Active learning with linear models**: The ellipsoid tracking technique applies directly to active learning (adaptive design of experiments), where one wants to adaptively select informative measurements of a linear model. The uncertainty ratio α_t maps naturally to D-optimal experimental design criteria.

4. **Clinical trial arm selection**: The paper cites MAB applications in clinical trials (Villar et al., 2015). TS-MR's ability to provide theoretical guarantees while preserving the low-regret behavior of LinTS is particularly compelling in this high-stakes setting; a domain-specific case study would strengthen the paper's appeal to practitioners.

5. **Robustness to model misspecification**: Extending the analysis to misspecified linear bandits (where the true reward is nearly but not exactly linear) would be valuable for real-world deployment, since all three real-world datasets use an approximation of linear reward structure.

### Visualizations & Case Studies
1. **Evolution of the confidence ellipsoid and its geometry over time**: A sequence of snapshots (e.g., in 2D or via PCA-projected 3D) showing how the ellipsoid shrinks and reorients as data accumulates, annotated with timestamps when OFUL actions are triggered, would make the geometric intuition of the paper viscerally clear to readers.

2. **Time series of µ̃_t alongside the switching signal**: Plotting µ̃_t over time for a problematic instance (e.g., Example 3, prior mean mismatch) alongside vertical markers for OFUL-action switches would visually demonstrate how the data-driven bound detects deteriorating performance and triggers course correction before regret accumulates.

3. **Scatter plot of α̃_t vs. actual instantaneous regret**: A scatter plot across time steps of the computable upper bound α̃_t on the x-axis versus the realized instantaneous regret on the y-axis would empirically validate the tightness of the bound and show whether the geometric analysis is conservative or near-tight.

4. **Qualitative failure mode analysis for LinTS and Greedy**: A case study dissecting exactly *when* and *why* LinTS fails in Example 3 (prior mean mismatch) — tracing which direction of the ellipsoid is misaligned with θ⋆ — would provide an intuitive explanation grounded in the paper's geometric framework and make the technical narrative more accessible.

5. **Heatmap of OFUL-action fraction as a function of (d, T)**: A 2D heatmap showing the fraction of time steps where OFUL is triggered as a function of dimension d and horizon T would characterize the overhead of course correction and help practitioners assess computational trade-offs before deployment.

### Natural Next Steps
1. **Adaptive threshold selection for µ**: Currently µ is a fixed hyperparameter. A natural follow-on is a fully adaptive algorithm that adjusts µ dynamically based on the observed trajectory of µ̃_t, potentially achieving oracle-like performance without any hyperparameter tuning — analogous to adaptive step-size methods in optimization.

2. **Extension to kernelized and neural linear bandits**: The geometric analysis of confidence ellipsoids extends naturally to kernelized linear bandits (where the feature map is implicit) and neural linear bandits (where the last-layer representation changes over time). Exploring whether the uncertainty ratio α_t remains computable and well-behaved in these settings would substantially expand the method's scope for deep learning practitioners.

3. **Lower bounds for data-driven regret bounds**: The paper provides upper bounds. A complementary lower bound showing that no data-driven method can achieve a tighter instance-dependent regret bound without access to θ⋆ would establish the fundamental limits of the geometric approach and position it optimally in the landscape.

4. **Theoretical characterization of when course correction is triggered**: A formal result quantifying the *frequency* of OFUL actions as a function of problem geometry (e.g., condition number of the covariance matrix, alignment of θ⋆ with principal directions) would make precise the informal claim that course correction is "rare in typical instances," converting an empirical observation into a theorem.

5. **Connections to information-directed sampling (IDS)**: The paper's geometric analysis of uncertainty ratios has a conceptual connection to IDS (Russo & Van Roy, 2016), which also tracks information gain relative to regret. Formalizing this connection could lead to a geometry-aware IDS variant with improved theoretical guarantees and open a rich new research direction at the intersection of information theory and bandit geometry.

# Report: Potentially Missed Related Work
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

No significant potentially missed related work identified.

**Reasoning:** 
- The works listed under "Same Problem, Different Approach" (OFUL, Thompson Sampling for Contextual Bandits) are already extensively cited in the paper (Abbasi-Yadkori et al., 2011; Agrawal & Goyal, 2013).
- The "Foundational Work" entries reference Langford & Zhang (2008), Li et al. (2010), Tewari & Murphy (2017), Villar et al. (2015), Bastani & Bayati (2020), and Cohen et al. (2020)—all already present in the references.
- "Geometry-Aware Reinforcement Learning for Manipulation" addresses a different problem domain (robotic manipulation) with only superficial terminology overlap; the geometric analysis in the current paper specifically concerns confidence ellipsoids in bandit parameter space.
- "Generalized Linear Bandits: Almost Optimal Regret with One-Pass Submissions" addresses a different problem class (generalized linear bandits with non-linear link functions) and focuses on computational efficiency rather than geometry-aware regret calibration. The current paper restricts itself to linear bandits, making this tangentially related at best.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [8.0, 8.0, 8.0, 6.0, 5.0]
Average score: 7.0
Decision: Accept (Poster)
Binary: Accept

=== CALIBRATION EXAMPLE 10 ===

# Review 1: Harsh Critic
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Counterfactual Realizability" is precise and accurately reflects the contribution. The abstract correctly identifies the problem (which L3 distributions can be physically sampled), the method (a formal definition plus the CTF-REALIZE algorithm), and the key results (completeness theorem, applications to fairness and RL). One minor concern: the abstract claims the algorithm is "complete" — a strong word — but the proof of this claim (Thm. 3.5) is deferred entirely to a technical report (Raghavan & Bareinboim, 2025), meaning a reader of only this paper cannot verify it. This tension between claim strength and verification accessibility will recur throughout the review.

---

### Introduction & Motivation

**Strengths.** The paper is addressing a genuine and important open question. The setup — that L3 quantities are routinely treated as inaccessible yet Bareinboim et al. (2015) already showed one exception — creates a crisp intellectual hook. The contributions are clearly enumerated and reasonably accurate. The footnote quoting Shpitser & Pearl (2007) and Dawid (2000) as representatives of the "L3 is inaccessible" consensus is effective.

**Concerns.**
- The Introduction conflates two distinct prior works as setting the stage: (a) counterfactual identification (when can L3 be computed from L1/L2 data?) and (b) counterfactual realizability (when can L3 data be gathered?). These are related but distinct problems. The paper would benefit from sharper demarcation of why identification failure is insufficient motivation for realizability — this is implicit but not made crisp.
- The contribution in Sec. 4.2 (CRL application) is partially credited to a companion technical report (Thm. 4.2, Cor. 4.3 in Raghavan & Bareinboim, 2025), not to this paper. The introduction advertises a theorem that does not appear in the paper being reviewed. This is a structural problem for a standalone conference submission.

---

### Related Work

There is no dedicated Related Work section; relevant literature is distributed through the Introduction and Discussion. While space constraints may justify this, several gaps are notable:

1. **Shpitser & Pearl (2007), "What Counterfactuals Can Be Tested"** is cited only once (Bibliography), but this is arguably the most directly related prior work. That paper asks which counterfactual *statements* are empirically testable from L1/L2 data, which overlaps substantially with the realizability question. The paper never explicitly distinguishes its contribution from this work, nor does it explain why that paper's results are subsumed or orthogonal. This is a significant omission that a reviewer will almost certainly flag.

2. **SWIGs (Richardson & Robins, 2013)** are cited only in passing in the Discussion. Since SWIGs provide an alternative graphical framework for reasoning about potential responses and their compatibility, their relationship to the present formalism warrants at least a sentence of comparison.

3. The literature on **surrogate experiments and transportability** (e.g., Correa & Bareinboim, 2020) is relevant to the question of which distributions can be accessed via which experiments, yet is referenced only in a footnote without discussion of its relationship to realizability.

---

### Method / Approach

**Strengths.** The physical action primitives (Def. 2.1) are cleanly specified. The Fundamental Constraint of Experimentation (Assumption 3.1) is precisely stated and well-motivated by the temporal asymmetry of causation. The graphical characterization in Cor. 3.7 is elegant and genuinely useful. Recovering the FPCI (Holland, 1986) as a corollary (Cor. 3.8) is a satisfying result that grounds the new framework in a well-known landmark.

**Concerns.**

1. **Def. 2.2 (Counterfactual mediator) is left informal.** This is a central concept — Example 1 hinges on it, and CTF-RAND in Def. 2.3 relies on it — yet the formal definition is deferred to the technical report (App. E). For a theory paper at ICLR, leaving a core definition informal in the main text is problematic. The reader cannot verify whether the operations performed in Example 1 (Eq. 1–5) are formally justified by Def. 2.2, since the rigorous version of Def. 2.2 is unavailable.

2. **Algorithm 2 (COMPATIBLE) is essentially uninterpretable in isolation.** The sub-routine is presented in App. B with 37 lines of pseudocode and no accompanying explanation (the walkthrough is in the technical report's App. C.2). Algorithm 1 (CTF-REALIZE) calls COMPATIBLE in Line 10 but its interaction logic is opaque without the walkthrough. For a paper whose central claim is an algorithm, making the algorithm unverifiable within the submission is a serious issue.

3. **When can CTF-RAND actually be performed?** Def. 2.3 defines CTF-RAND(X→C) as an action, but the paper states it "can only be enacted under certain structural conditions" without providing a complete characterization. The two cases given (environments permitting simultaneous randomization/observation, and counterfactual mediators) are illustrative, but it is unclear whether these two cases are exhaustive. The realizability framework is circular if the input action set A is left underspecified.

4. **The completeness proof of Thm. 3.5** — the paper's main theoretical claim — is entirely absent from the submission, deferred to the technical report. For ICLR, where theoretical claims are reviewed as part of the paper, this is a critical gap. The paper cannot be accepted on the strength of a theorem whose proof is not included or even sketched.

5. **Realizability w.r.t. incorrect or partial graph knowledge.** The framework assumes access to an accurate causal diagram G. The Discussion acknowledges this limitation, but does not analyze the consequences: if G is misspecified, CTF-REALIZE may declare a distribution realizable when it is not (or vice versa). No discussion of robustness or sensitivity to graph misspecification is provided.

6. **Cor. 3.7 vs. Thm. 3.5 relationship.** Cor. 3.7 gives a clean graphical criterion *only under the maximal action set* A†(G). But in practice, an agent may not be able to perform all CTF-RAND actions. The relationship between what the algorithm computes for an arbitrary A and the graphical criterion in Cor. 3.7 could be made more explicit — in particular, does Cor. 3.7 degrade gracefully as A shrinks?

---

### Experiments & Results

**Strengths.** The three examples (mediation analysis, causal fairness, causal RL) are well-chosen and cover distinct application domains. The simulation in Example 3 with cumulative regret and optimal arm probability over 2000 iterations × 200 epochs is the most convincing empirical result. The finding that a contextual bandit ignoring the counterfactual structure (the "naive L2" green curve) performs far worse than the optimal L3 strategy is compelling.

**Concerns.**

1. **All three experiments use hand-crafted SCMs.** The SCMs are fully specified by the authors (details in the technical report). There is no demonstration on real-world data or on a benchmark where the true SCM is not known. This limits the empirical contribution to existence proofs that the framework *can* produce improvements, rather than evidence that it *does* in practice.

2. **Example 2 (Causal fairness) lacks clarity on implementation.** The paper states that µctf is the L3 fairness measure used as a training penalty, but does not explain how µctf is estimated during training. If CTF-RAND(X→Y) and CTF-RAND(X→Z) are performed as described (separately randomizing the name input to each model), this constitutes a specific data-collection procedure that requires careful implementation details — which are deferred. The histogram in Fig. 5(c) compares 1000 classifiers but does not report the L3 outcome distribution explicitly (only that classifiers "nearly always meet the fairness requirement").

3. **No statistical comparison between L3 and L2 approaches in Example 2.** The paper reports that ~50% of L2 classifiers show µctf > 5%. What is the corresponding rate for L3 classifiers? The paper says "nearly always" but provides no number. For a simulation paper, this omission of a key result is surprising.

4. **Example 3's optimality claim relies on Thm. 4.2, which is not in this paper.** The claim that E[Yx | x′, dx′′] is optimal — i.e., that no other realizable L3 distribution yields a better strategy — is asserted without proof in this submission. This is the most important claim in Sec. 4.2 and it is unverifiable here.

5. **The bandit setup in Example 3 is a single highly specific SCM.** The claim that the L3 strategy dominates is shown for one SCM with one particular structure of latent confounding. No sensitivity analysis across different confounding strengths, graph structures, or domain sizes is presented.

6. **Error bars are reported (CI=95%) for Example 3, which is appropriate.** This is a genuine strength.

7. **No comparison to non-Thompson Sampling baselines.** Only Thompson Sampling variants are tested. Given that the paper claims superiority of the L3 strategy for multi-armed bandits in general, testing only one algorithm family is limiting.

---

### Writing & Clarity

The main text is generally well-written and the high-level ideas are accessible. The Pearl Causal Hierarchy framing is used effectively throughout.

**Clarity concerns.**
- Figure 3 is described twice in succession (the text is duplicated verbatim: "Figure 3: Testing realizability of P(Zx, Wtx, Wt, Wtt) for G1 (left) and G2 (right). G1 yields conflicting requirements. yields conflicting requirements."). This is either a parser artifact or a genuine copy-paste error in the manuscript.
- The inline description of Algorithm 1 in Sec. 3 refers to inner loops and "edge-coloring intuition" without the visual/formal support needed to follow the algorithm. The two-column pseudocode layout for Algo. 1 is difficult to parse without the technical report's walkthrough.
- The description of Example 2's simulation (Sec. 4.1) has clear OCR/formatting artifacts in the figure caption ("CTF-RAND(-RAND(AND(( X → Y )" and "µctfctf"), which obscure the exposition, though these appear to be parser issues.

---

### Limitations & Broader Impact

**Acknowledged limitations:**
- Requires a known causal graph.
- CTF-RAND may not always be physically feasible.

**Missing or underdeveloped limitations:**

1. **Graph misspecification is acknowledged but not analyzed.** If the input graph G is wrong, the realizability decision could be incorrect in either direction. The practical consequence — an agent believing a distribution is realizable when it is not, and hence collecting corrupted data — is non-trivial and deserves at least a brief discussion.

2. **The FCE assumption (Assumption 3.1) may fail in ways not discussed.** The assumption that a unit undergoes each mechanism at most once is natural for biological units, but may be strained in settings with persistent digital entities (e.g., the social media example, Example 3, where the user interacts with the platform over 2000 days). What if a user's preferences evolve over time? The paper notes that repeating an experiment on the same unit "is tantamount to testing a new unit with unknown latent features," but this is asserted rather than carefully argued for the sequential setting.

3. **Ethical and dual-use concerns with ctf-randomization.** The paper discusses applications in automated HR pipelines and audit settings, where interventions involve randomizing perceived demographic identity (names, pronouns). While the paper cites the Bertrand-Mullainathan experiment as a precedent, the ethics of performing such interventions in production systems (rather than controlled field experiments) are not discussed.

4. **Scalability.** The time and space complexity of CTF-REALIZE is mentioned as being in App. C.3 of the technical report but not discussed in the paper. For ICLR readers interested in practical deployment, this is relevant.

---

### Overall Assessment

This paper addresses a well-defined, important open question in causal inference: which Layer 3 distributions can be physically sampled via experimentation? The formalization of realizability, the CTF-REALIZE algorithm, and the graphical criterion in Cor. 3.7 are genuine contributions, and grounding the Fundamental Problem of Causal Inference as a corollary of a more general framework is a satisfying theoretical result. The application examples are well-chosen and the bandit simulation provides modest empirical support.

However, the paper has structural problems that are significant for a standalone ICLR submission. **The proof of the completeness theorem (Thm. 3.5) — the paper's central theoretical claim — is entirely absent**, deferred to a technical report. Algorithm 2 (COMPATIBLE) is uninterpretable without the technical report's walkthrough, and the key concept of counterfactual mediator (Def. 2.2) is left formally undefined in the paper. The most important applied claim (optimality of the proposed bandit strategy, Thm. 4.2) also lives only in the technical report. The empirical validation is limited to hand-crafted SCMs and a single specific bandit setting. Finally, the relationship to the most directly related prior work — Shpitser & Pearl (2007) "What Counterfactuals Can Be Tested" — is never explicitly analyzed. Despite these concerns, the contribution is meaningful and the paper was ultimately accepted at ICLR 2025, suggesting the community found the core ideas valuable; the technical report presumably provides the missing proofs. Revisions should prioritize including at least a proof sketch for Thm. 3.5, formalizing Def. 2.2, and clearly situating the work relative to Shpitser & Pearl (2007).

# Review 2: Neutral Reviewer
## Balanced Review

### Summary
This paper addresses a fundamental question in causal inference: which counterfactual (Layer 3) distributions can be directly sampled from via physical experimentation, challenging the prevailing belief that only observational (L1) and interventional (L2) distributions are accessible. The authors formalize "realizability" as the ability to draw samples from a distribution, introduce "counterfactual randomization" (CTF-RAND) as a physical procedure extending Fisherian randomization, and provide a complete algorithm (CTF-REALIZE) with a graphical criterion to determine whether an arbitrary counterfactual distribution is realizable given physical constraints like the inability to time-travel.

### Strengths
1. **Highly novel conceptual contribution**: The paper challenges a deeply entrenched assumption in causal inference—that counterfactuals are inherently unobservable—and provides a rigorous framework for determining when they can actually be sampled (e.g., citations on p.1 stating counterfactuals "cannot be observed by definition").

2. **Complete theoretical solution**: Theorem 3.5 proves CTF-REALIZE is correct and complete, and Corollary 3.7 provides an elegant graphical criterion for realizability under maximal experimental capabilities—showing the distribution is realizable iff its ancestor set doesn't contain the same variable under different regimes.

3. **Generalizes foundational results**: Corollary 3.8 demonstrates that the well-known Fundamental Problem of Causal Inference (Holland, 1986) emerges as a specific consequence of their more general framework, providing deeper theoretical understanding.

4. **Practical relevance with concrete applications**: Three well-motivated examples demonstrate applicability—mediation analysis (Example 1), causal fairness (Example 2, with simulation showing L2 classifiers show statistically significant discrimination ~50% of the time while L3 classifiers nearly always meet fairness requirements), and causal RL (Example 3, with Table 1 showing the optimal L3 strategy achieves E[Y]=0.80 vs. 0.75 for ETT baseline).

5. **Clear distinction from identification**: The paper carefully distinguishes realizability (ability to physically gather samples) from identifiability (ability to compute from available data), clarifying an important conceptual point.

### Weaknesses
1. **Strong assumptions about environment structure**: The framework requires knowledge of the causal diagram and assumes counterfactual mediators exist (Def. 2.2) that can be manipulated. In many real-world scenarios, such variables may not exist or be practically manipulable—the fairness example requires "the structural assumption of race being revealed at the screening stage only by candidate name."

2. **Limited scope**: The treatment is restricted to recursive SCMs with finite discrete domains, which limits the generality of the results for continuous or cyclic systems.

3. **Key details deferred to technical report**: Proofs, algorithm complexity analysis, full simulation details, and formal definitions (e.g., Def. E.2 of counterfactual mediator) are in the referenced technical report rather than the main paper, making it harder for readers to fully evaluate the work.

4. **Limited empirical validation**: While the simulations demonstrate the concepts, they are relatively simple. More extensive empirical evaluation across diverse settings would strengthen claims about practical applicability.

5. **Practical feasibility concerns**: The feasibility of CTF-RAND depends heavily on the specific environment—e.g., the paper notes CTF-RAND "can only be enacted under certain structural conditions." The paper could better address how often such conditions arise in practice.

### Novelty & Significance
This is a highly novel and significant contribution. It fundamentally reconceptualizes the boundary between observable and unobservable quantities in causal inference. The theoretical results (complete algorithm + graphical criterion) are substantial. The work opens new research directions in experiment design, causal RL, and fairness. For ICLR, this work sits at the intersection of causal inference and learning, demonstrating how counterfactual-based strategies can provably dominate interventional approaches in RL and fairness settings.

### Suggestions for Improvement
1. **Include more key technical details in the main paper**: At minimum, include the algorithm's time/space complexity and a formal definition of counterfactual mediators in the main text.

2. **Add discussion of robustness**: Discuss how the framework behaves under partial causal knowledge or model misspecification—what happens if the assumed graph is incorrect?

3. **Expand empirical evaluation**: Consider additional experiments or a more thorough analysis of when counterfactual mediators exist in practice.

4. **Clarify practical prerequisites**: Provide clearer guidance on how to determine if CTF-RAND is feasible in a given environment before running the algorithm.

5. **Address continuous domains**: Discuss whether and how the results might extend to continuous variables, even if as future work.

# Review 3: Spark Finder
## Strengthening Opportunities

### Missing Experiments

1. **Real-world dataset validation**: All three examples (Examples 1–3) rely exclusively on synthetic SCMs. Testing the counterfactual fairness approach (Example 2) on real-world hiring or admissions datasets—such as the original Bertrand & Mullainathan resume audit data or publicly available COMPAS recidivism data—would substantially strengthen the empirical claims about L2 vs. L3 fairness constraints.

2. **Scalability benchmarks for CTF-REALIZE**: The paper claims polynomial complexity (deferred to the technical report), but no empirical runtime experiments are provided. Benchmarking CTF-REALIZE on causal graphs of increasing size (e.g., 10, 50, 200 nodes) and variable domain cardinality would validate practical applicability and expose any bottlenecks.

3. **Comparison with broader causal bandit baselines**: Example 3 compares only Thompson Sampling variants. The paper would benefit from comparisons with causal UCB (Lu et al., 2020), causal EXP3, or more recent causal bandit algorithms that exploit graph structure, to place the L3 strategy more precisely in the literature landscape.

4. **Sensitivity/robustness experiments under graph misspecification**: The algorithm requires a known causal graph, but graphs are often partially misspecified in practice. An experiment where the input graph deviates from the true data-generating graph—measuring how realizability errors and estimation bias degrade—would be practically essential for applied readers.

5. **Sample complexity comparison**: Given the same total sample budget, how many samples does a counterfactual strategy require to achieve comparable estimation accuracy to an interventional strategy? An empirical sample-efficiency curve (L2 vs. L3) across the three application domains would directly quantify the practical benefit of switching to counterfactual data collection.

6. **Mediation analysis experiment (Example 1)**: Example 1 (traffic camera audit) is developed theoretically but has no accompanying simulation. A controlled simulation—showing that the counterfactual mediator approach recovers the NDE under confounding where identification fails—would complete this application and make the section self-consistent.

---

### Deeper Analysis Needed

1. **Sample complexity theory**: The paper establishes when sampling is *possible* but not how *many* samples are needed. A formal sample complexity analysis—how the variance of counterfactual estimators compares to their interventional counterparts—would address the gap between realizability (existence) and statistical efficiency (practicality).

2. **Partial realizability and bounding**: When a query is not realizable (e.g., P(Y_x, Y_{x′})), the paper does not address what *partial* information is still extractable. Connecting non-realizable queries to tight partial-identification bounds (Zhang et al., 2022) using whatever L3-data *is* available would turn a negative result into an actionable one.

3. **Formal relationship between realizability and identifiability**: Sec. 5 flags this as future work, but even a preliminary theorem—characterizing the class of nonidentifiable L3-quantities that *become* identifiable once some ctf-randomization is admitted—would be a significant theoretical contribution and directly motivate the experiment-design implications.

4. **Convergence guarantees for CTF-based bandit strategies**: Thm. 4.2 (in the technical report) proves optimality in a static sense, but the dynamic (online) convergence rates—regret bounds—for the L3 Thompson Sampling variant are not derived. Formal regret bounds analogous to those for standard Thompson Sampling would make the RL contribution complete at the conference-paper level.

5. **Formal treatment of noisy/imperfect CTF-RAND**: The current framework assumes CTF-RAND can be performed perfectly. Analyzing how measurement noise or imperfect perception manipulation (e.g., OCR errors in video editing) propagates into bias in counterfactual estimates would make the framework more realistic and connect to the error-in-variables literature.

6. **Connection to Single World Intervention Graphs (SWIGs)**: The paper cites Richardson & Robins (2013) in passing but does not formally compare the realizability characterization to the SWIG framework. A brief formal comparison could clarify whether the graphical criterion in Cor. 3.7 could alternatively be read off from a SWIG directly.

---

### Untapped Applications

1. **Large language model (LLM) red-teaming and auditing**: The paper briefly mentions editing text inputs to simulate perceived demographic identity. This could be developed into a full application: using CTF-RAND on token-level features of prompts to audit LLM outputs for demographic bias, directly estimating counterfactual fairness measures without identifiability assumptions.

2. **Clinical trial design for personalized medicine**: The realized L3-data framework naturally maps to adaptive trial designs where a patient's natural treatment preference is recorded before randomization (e.g., preference-based randomized controlled trials). Making this connection explicit—and showing which NDE/probability-of-benefit quantities become estimable—would attract a healthcare-ML audience.

3. **Counterfactual off-policy evaluation in recommendation systems**: Online platforms routinely log user interactions. The CTF-RAND framework could be applied to logged-data settings where a user's organic preference signal is observable before the recommendation is imposed, enabling direct L3 off-policy evaluation that outperforms standard importance-sampling estimators.

4. **Algorithmic recourse**: Generating actionable counterfactual explanations ("what would have to change for a different decision?") is a major ML fairness topic. Realizability analysis directly characterizes which recourse interventions are physically enactable, providing a principled filter on the space of recourse suggestions.

5. **Causal data augmentation for low-resource ML**: Realizable counterfactual distributions can be used to generate additional training data (counterfactual augmentation). An experiment showing that counterfactually augmented training sets outperform interventionally augmented ones—especially in small-data regimes—would broaden the paper's appeal to the mainstream ML community.

---

### Visualizations & Case Studies

1. **Step-by-step visual walkthrough of CTF-REALIZE**: The algorithm's edge-coloring intuition (Fig. 3) is the most pedagogically important element of Sec. 3, but only two graphs are shown. A panel figure tracing the algorithm's execution—color-coding each node's INT_V set as it is built up in topological order—across 3–4 progressively complex graphs would make the algorithm far more accessible.

2. **Realizability map over the PCH**: Fig. 4 provides a high-level sketch, but a more detailed diagram—concretely cataloguing which families of L3 distributions (ETT, PNS, NDE, Probability of Sufficiency, etc.) fall into each realizability class, and under which graph conditions—would serve as a community reference table.

3. **Visualization of fairness measure distributions across classifiers**: Fig. 5(c) shows a histogram of 1000 classifiers, but a 2D scatter plot of µ_int vs. µ_ctf for each classifier would more clearly reveal whether the L2-trained classifiers are systematically biased or just high-variance, and whether the L3-trained classifiers cluster near the fairness boundary.

4. **Cumulative regret curves with confidence intervals across more graph structures**: Fig. 6(c–d) shows results for one specific SCM. Showing regret curves for multiple graph structures (with and without confounders, with varying numbers of arms) would demonstrate that the L3 advantage is structural rather than an artifact of the specific example.

5. **Illustrative "failure mode" case study**: A concrete case where an agent *incorrectly assumes* a distribution is realizable (e.g., uses a wrong graph), collects biased "counterfactual" data, and reaches incorrect conclusions would powerfully motivate the need for formal realizability analysis and help practitioners understand the stakes of graph misspecification.

---

### Natural Next Steps

1. **Extension to sequential decision-making (MDPs/POMDPs)**: The paper explicitly flags this in Sec. 5. Generalizing the CTF-REALIZE framework to multi-step environments—where the realizability of a trajectory-level counterfactual depends on the temporal structure—is the most direct and high-impact follow-on, particularly for the causal RL community.

2. **Updating identification algorithms to accept L3 data**: As noted in Sec. 5, current ID algorithms (e.g., the ID algorithm of Shpitser & Pearl) treat L1 and L2 as the only data sources. A "ctf-ID" algorithm that takes realizable L3-distributions as additional inputs and determines which previously non-identifiable quantities now become identifiable would be a foundational contribution to the identification literature.

3. **Causal discovery of counterfactual mediators**: The practicality of CTF-RAND depends on identifying counterfactual mediators (Def. 2.2) in the environment. Developing principled methods—perhaps combining causal discovery with domain knowledge—to detect and validate these mediators would make the framework self-contained and deployable without expert graph specification.

4. **Tight partial-identification bounds under partial realizability**: For distributions that are not fully realizable, quantifying how much the available L3-data (from realizable sub-distributions) tightens bounds on target quantities (e.g., probability of necessity/sufficiency) is a concrete and actionable theoretical program.

5. **Extension beyond recursive/acyclic SCMs**: The paper restricts to acyclic diagrams. Many real systems—biological feedback loops, dynamic economic equilibria—involve cycles. Extending the realizability framework to cyclic causal models (e.g., via equilibrium or interference-based SCMs) would significantly broaden applicability.

6. **Counterfactual realizability under partial graph knowledge**: The current framework requires a fully specified graph. A natural generalization would characterize realizability under a *set* of compatible graphs (i.e., a Markov equivalence class or a partial ancestral graph), providing robust guarantees when causal structure is only partially known from data.

# Report: Potentially Missed Related Work
Looking at the paper and related works:

**Already Cited Works:**
- Bareinboim, Forney & Pearl (2015) — cited in abstract, introduction, and references as "(Bareinboim et al., 2015; Forney et al., 2017)"
- Pearl & Mackenzie (2018) — cited as "(Pearl & Mackenzie, 2018)" 
- Bareinboim et al. (2022) — cited as "(Bareinboim et al., 2022)"

**Tangentially Related Works:**
- "Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models" addresses counterfactual prediction in time-varying settings using neural state-space models (Mamba). This is a different problem (estimation/prediction) with different methods, not addressing which counterfactual distributions are physically realizable. The connection is weak.

- "Gumbel Counterfactual Generation From Language Models" focuses on generating counterfactual strings from language models by modeling them as SCMs with Gumbel-max mechanisms. This is in the NLP domain with entirely different objectives, tangentially related at best.

## Potentially Missed Related Work

No significant potentially missed related work identified.

# ACTUAL HUMAN SCORES AND DECISION
Individual reviewer scores: [6.0, 10.0, 6.0, 8.0]
Average score: 7.5
Decision: Accept (Spotlight)
Binary: Accept
