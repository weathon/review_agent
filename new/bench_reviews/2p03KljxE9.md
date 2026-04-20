## Summary
This paper introduces LAFT (Language-Assisted Feature Transformation), a training-free method that transforms CLIP image features by projecting them onto concept subspaces derived from text embeddings, enabling users to guide or suppress specific attributes for anomaly detection. The method combines with kNN for semantic anomaly detection and with WinCLIP for industrial defect detection, showing strong empirical gains on multiple datasets without requiring any additional training.

## Strengths
- **Clear, training-free formulation with practical utility:** The method avoids gradient updates entirely, making it computationally cheap and agnostic to downstream tasks. The linear projection mechanism (Eqs. 5-6) is mathematically straightforward and interpretable, directly leveraging CLIP's image-text alignment without architectural changes.

- **Strong empirical results in "Guide" mode across diverse datasets:** LAFT AD achieves 98.5 AUROC on Colored MNIST, 95.6 AUROC on Waterbirds, and 98.1 AUROC on CelebA Eyeglasses (Tables 1-2), substantially outperforming the best baselines (WinCLIP+, InCTRL, ZOE). WinCLIP+LAFT further matches or exceeds pre-trained adapters on MVTec AD and VisA across few-shot settings (Table 3), validating the training-free advantage.

- **Transparent reporting and ablation on prompt quality:** The authors honestly report the failure of "Ignore" mode on Waterbirds (Section 5.1) and provide a systematic ablation (Table 4) varying Seen/Unseen/Auxiliary concept values, showing robustness to partial prompt specifications (e.g., "Partial anomalies" achieves 98.4 AUROC vs. 98.8 for "Exact anomalies" on Colored MNIST).

- **Effective integration with existing methods:** Applying LAFT to WinCLIP's window, image, and text embeddings demonstrates modularity—consistent gains across K=0 to K=8 without modifying WinCLIP's architecture.

## Weaknesses

### Major
- **The unvalidated linear subspace assumption is central to the method:** The entire LAFT mechanism depends on the claim that pairwise differences between CLIP text embeddings define subspaces that meaningfully separate visual attributes when projected onto image features. The paper motivates this by citing Mikolov (2013) word2vec arithmetic (Section 4.2) and provides a single scatter plot (Figure 3) on Colored MNIST. However, there is no quantitative analysis of whether concept axes are actually aligned with image feature directions—no cosine similarity scores between text-derived axes and image features, no variance-explained metrics, no orthogonality checks between axes for different attributes. For complex, entangled attributes like bird/background on Waterbirds, this assumption is particularly questionable (as evidenced by the "Ignore" mode failure). The paper treats the subspace hypothesis as given rather than validated empirically, which undermines confidence in the mechanism's generality. Without this validation, it remains plausible that gains on simpler datasets stem from noise filtering or implicit dimensionality reduction rather than semantic alignment.

- **The "Ignore" mode's key claim is unsupported on the most relevant dataset:** The abstract and contributions explicitly claim LAFT allows users to "ignore specific image attributes." Yet on Waterbirds—the benchmark specifically designed for disentangling entangled bird/background attributes—the "Ignore" variant achieves only 84.8 AUROC, barely above the unguided kNN baseline (82.3%) and far below "Guide" mode (95.6%) or even "kNN + All normal images" (83.0 AUROC, Table 1). The authors acknowledge this ("ignoring one attribute... does not directly improve performance on the other attribute," Section 5.1) but still claim the method "effectively guides one attribute while ignoring the others" (Section 5). The contribution of attribute suppression is therefore only demonstrated on synthetic/simple datasets (Colored MNIST: 97.4 AUROC, Table 1) where attributes are cleanly separable. For real-world entangled attributes—the target use case—the method does not deliver on this claim.

- **WinCLIP+LAFT integration is underspecified for reproducibility:** The industrial anomaly detection extension is described in a single sentence: "We apply LAFT to WinCLIP's window, image, and text embeddings" (Section 5.2). Critical details are missing: (1) how concept axes are computed per image category (each MVTec category has different defect types); (2) how multi-scale local features are projected without washing out fine-grained defect information; (3) whether WinCLIP's patch-level embeddings are transformed identically to image-level embeddings; (4) whether re-normalization is applied before WinCLIP's anomaly scoring. Without these details, reproducing Table 3's results is not feasible, and the claimed generalization of the method to industrial settings rests on black-box evaluation rather than transparent methodology.

### Minor
- **Post-projection normalization is undefined:** CLIP embeddings are unit-normalized for cosine similarity. The projection operations (Eqs. 5-6) fundamentally change vector magnitudes—projected vectors will generally have arbitrary norms. Equation 7 then applies cosine similarity to these unnormalized features without stating whether re-normalization occurs. While cosine similarity is scale-invariant for the direction of individual vectors, projecting onto a subspace changes their distribution in ways that could trivially alter nearest-neighbor geometry independent of semantic alignment. This affects interpretability of whether gains come from semantic directionality or implicit rescaling.

- **Theoretical desiderata (Eqs. 1-2) are never operationalized:** Section 3 presents two formal desiderata—invariance to irrelevant attributes (Eq. 1) and preservation of mutual information (Eq. 2)—but they are never measured or referenced in the experiments. While Section 3 mentions empirical evaluation "are provided in the Experiments," no corresponding analysis appears in Section 5. These equations serve as rhetorical framing rather than testable constraints, making it unclear whether LAFT actually achieves the invariance and informativeness it promises.

### Trivial
- **Prompt ablation conflates coverage with noise robustness:** Table 4 tests whether adding auxiliary concept values not in the dataset degrades performance, but using completely irrelevant arbitrary words (e.g., yellow/purple for colored MNIST) does not test robustness to noisy or semantically ambiguous guidance—the realistic failure mode for natural language prompts from non-expert users.

## Nice-to-Have
- Visualize original vs. LAFT-transformed features with t-SNE/UMAP colored by relevant/ignored attributes to provide intuitive geometric evidence beyond Figure 3's synthetic scatter plot.
- Add a random-projection or unsupervised PCA baseline on image features to isolate whether gains come from language-guided alignment or simply from dimensionality reduction/regularization effects.
- Explore whether the concept axis directions are stable across different CLIP variants (ViT-B/16 vs. ViT-L/14) to assess robustness to backbone choice.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- The harsh critic's concern that the "kNN + All normal images" baseline is a "strawman comparison" (Section-by-Section Notes): This reflects reviewer knowledge gaps. The paper explicitly justifies this baseline as "simulating image-only methods with attribute-specific image processing" where augmentation is straightforward. It is not an unfair comparison favoring the author's method—it sets a ceiling for what a supervised image-only baseline could achieve with perfect attribute coverage. Per hard rules, asymmetric comparisons favoring baselines should be retained as strength evidence.

- The harsh critic's claim that "prompt construction leaks anomaly knowledge, undermining the one-class AD premise" (Critical Issue 3): While Table 4 shows the "Unseen" set includes anomaly concept values, the paper also demonstrates in the "Only normals" row that LAFT achieves 94.3 AUROC (Colored MNIST) and 94.3 AUROC (Waterbirds) with zero anomaly concept knowledge. The method is explicitly framed as leveraging user prior knowledge (Section 4: "Note that our method is not intended to be used in situations where the user has no knowledge"). Providing more concept values improves performance but is not required. The criticism misreads the paper's positioning.

- The harsh critic's demand to "quantify invariance by training a linear probe" (Missing Experiments 2) is noted but weakened: while this would strengthen the paper, the lack of such an analysis does not invalidate the core contribution. The empirical performance improvements on the "Ignore" axis where it works (Colored MNIST: 97.4 AUROC) serve as proxy evidence.

- The harsh critic's demand for "orthogonality tests, attribute alignment metrics" under Critical Issue 1 is partially addressed by the paper's empirical validation (Figure 3 and strong results), though the lack of quantitative geometric analysis remains a valid concern. The claim that "the entire mechanism is operating on a flawed premise" is overstated given the demonstrated effectiveness on guide mode.

## Novel Insights
One genuinely novel aspect of this work is the insight that pairwise differences between CLIP text embeddings—rather than the embeddings themselves—can be used to construct interpretable "concept axes" via PCA, and that orthogonal projection in this derived subspace provides a principled mechanism for attribute suppression. This goes beyond prior CLIP-based AD methods that use text only for similarity scoring (MCM, ZOE, WinCLIP) or that require training adapter layers (InCTRL, APRIL-GAN). The dual guide/ignore formulation within a single projection framework is a meaningful contribution, and the training-free property makes it uniquely applicable to data-scarce scenarios.

None beyond the paper's own contributions.

## Suggestions
- Add a geometric validation section quantifying the alignment between text-derived concept axes and image features: compute cosine similarities between concept axes and image PCA directions, measure variance explained by each axis, and test orthogonality between axes for different attributes (e.g., number vs. color on Colored MNIST, bird vs. background on Waterbirds).
- Specify the WinCLIP+LAFT pipeline in full detail: for each MVTec category, describe which prompts are used, which embedding levels are transformed, and whether post-projection re-normalization is applied. Release code for reproducibility.
- Clarify whether projected vectors are re-normalized before cosine similarity scoring in Eq. 7, and if not, discuss whether the reported gains could be partially attributable to changes in the implicit temperature/scaling of the similarity metric.
- If retaining Eqs. 1-2 as desiderata, add at least a qualitative discussion in the experiments section of how LAFT relates to these formal criteria. If they are purely motivational, move them to the introduction.

## Score and Decision
Comparing against calibration anchors: CLIP-LAD (scores 3, 5, 1; withdrawn) is a weaker training-free CLIP method with limited novelty and unfair comparisons. The CEER concept-based projection paper (scores 3, 3, 3, 3; rejected) had fundamental misunderstandings about its reliance on CLIP's supervision. AnomalyCLIP (scores 8, 5, 6, 5, 8, 5; accepted poster) had stronger prompt-learning contributions but similar empirical validation. Deep Orthogonal Hypersphere (scores 8, 8, 8, 8; spotlight) had both theoretical grounding and extensive experiments.

This paper sits between CLIP-LAD and AnomalyCLIP in quality. It has stronger empirical results than CLIP-LAD across more datasets and is more honest about limitations. However, compared to accepted papers like AnomalyCLIP, it lacks theoretical depth, has underspecified extensions (WinCLIP+LAFT), and overclaims the "Ignore" capability. The linear subspace assumption is central but unvalidated beyond a single figure. On balance, the paper makes a real contribution with clear empirical value, but the weaknesses are substantive enough to prevent strong endorsement.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>