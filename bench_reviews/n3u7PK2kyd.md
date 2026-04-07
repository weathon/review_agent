## Summary

The paper proposes two complementary contributions: (1) Symmetric Representation Topology Divergence (SRTD), which unifies RTD and Max-RTD through a mathematically principled framework based on mapping cones, and (2) Normalized Topological Similarity (NTS), a scale-invariant similarity measure that compares merge-order rankings from Minimum Spanning Trees using Spearman's correlation. The authors demonstrate that NTS uniquely combines CKA's interpretability (graded similarity patterns) with topological methods' sensitivity to structural discontinuities (detecting functional shifts at pooling layers), while SRTD achieves comparable performance to RTD variants with theoretical elegance and moderate efficiency gains.

## Strengths

- **Theoretical elegance of SRTD framework**: The paper provides a clean mathematical unification of RTD, Max-RTD, and SRTD. Theorem 3.3 and Corollaries 3.4/3.5 establish that Max-RTD-lite + RTD-lite = SRTD-lite, explaining the empirical complementarity between directional RTD and Max-RTD (their asymmetries have opposite signs, Table 2f). The mapping cone construction (comparing union to intersection filtrations) is a principled resolution of the ad-hoc symmetrization issue.

- **NTS's unique capability**: Figure 4 convincingly demonstrates that NTS captures both graded similarity patterns (like CKA) and functional shifts at pooling layers (like RTD-lite), while neither baseline alone achieves both. The TinyCNN experiment shows CKA misses the pooling discontinuity while RTD-lite fails to show graded patterns—NTS succeeds at both.

- **Empirical evidence of CKA saturation in LLMs**: Section 5.4 identifies a real limitation of CKA for LLM representation analysis: CKA scores cluster near 0.8-0.9 for most model pairs, reducing discriminative power. NTS shows better separation (Figure 6). This finding has practical significance for the community.

- **Computational efficiency of NTS-E**: Section 6 shows NTS-E operates in O(n²(α(n) + d)) time with O(2n²) memory, avoiding the triple MST computation and quantile normalization of RTD/SRTD variants. Runtime benchmarks (Figure 7) confirm NTS-E is fastest among the tested methods.

## Weaknesses

- **Contradiction between scale-invariance claim and Z-score dependency**: The abstract claims NTS is "scale-invariant," yet Section 5.4 states "we recommend applying Z-score normalization across the feature dimension" and Appendix K.1 shows NTS scores collapse without it (Figure 21, especially for Llama). While rank correlation is invariant to monotonic scaling of individual distance matrices, the necessity of Z-scoring reveals sensitivity to the *distribution of activation magnitudes across features*. This tension must be resolved: either revise the formal definition of NTS to include Z-score normalization and update its theoretical properties, or acknowledge that NTS is not fully scale-invariant and explain what invariance properties it does possess.

- **LLM experiments rely entirely on qualitative visual inspection**: The claim that NTS is "more discriminative" than CKA for LLMs rests on heatmap inspection (Figure 6) and the DeepSeek-R1-Ds lineage argument. No quantitative metric (e.g., silhouette score of model family clustering, rank correlation with known lineage distance) is provided. The "layer 6 empirically yielded the most discriminative results" (line 246-247) is post-hoc selection without systematic justification or correction for multiple comparisons.

- **DeepSeek lineage argument is overstated**: The paper asserts CKA makes a "critical, counter-intuitive error" by showing low similarity between DeepSeek-R1-Ds and its parent Qwen2.5. However, DeepSeek-R1-Ds undergoes extensive reinforcement learning after distillation—it is entirely plausible that its representations genuinely diverge from the parent. Presenting NTS's high score as "correct" and CKA's low score as an "error" assumes without justification that lineage implies representation similarity.

- **Core pairs selection is under-explained**: NTS compares merge ranks only on E_core = MST(w) ∪ MST(w̃), which contains O(n) edges from O(n²) available. The union construction could create an unbalanced set when MSTs differ substantially, and the paper provides no analysis of edge overlap or its impact on NTS scores. The relationship between this design choice and the method's effectiveness is not investigated.

- **0-dimensional restriction limits topological coverage**: NTS captures only hierarchical clustering structure (H₀ persistence) and is blind to higher-dimensional features (cycles, voids). The paper acknowledges this briefly but does not assess what information is lost for representations with non-trivial H₁/H₂ topology, or why this limitation is acceptable for LLM representations specifically.

## Nice-to-Haves

- **Quantitative evaluation metric for LLM discriminativity**: A silhouette score or cluster accuracy based on model family labels would provide statistical validation beyond visual inspection of heatmaps.

- **Mechanistic explanation of functional shift detection**: The paper observes that NTS detects pooling layer discontinuities but does not explain *which* topological features change or *why* this detection works.

- **Ablation of MST-edge-based RSA**: Comparing NTS-E against Spearman's RSA restricted to the same E_core would isolate whether benefits come from topological structure or simply from sparse edge selection.

- **Differentiable approximation of NTS**: The paper acknowledges NTS is non-differentiable and thus analysis-only. A discussion of potential differentiable approximations would strengthen the paper's utility claims.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **PWCCA/SVCCA baseline comparison**: The paper explicitly addresses this in footnote 1, citing prior work (Kornblith et al., 2019; Barannikov et al., 2021a) showing these methods are less effective for layer analysis. Demanding additional baselines beyond CKA is scope creep.

- **Cross-architecture validation on RNNs/SSMs**: The paper focuses on CNNs and Transformers, the dominant architectures in representation analysis. Testing on additional architectures is beyond the stated scope.

- **"Brute-force" symmetrization criticism**: The paper's characterization of average-based symmetrization as "brute-force" is a minor framing choice, not a substantive weakness. Arithmetic averaging is a valid symmetrization technique; SRTD offers a theoretically richer interpretation.

- **Metric properties of SRTD**: Whether SRTD satisfies triangle inequality is not claimed or required. The RTD family was never positioned as metrics.

- **Non-differentiability of NTS**: This is explicitly acknowledged in the conclusion as a limitation and suggested direction for future work. It is not a novel criticism.

## Novel Insights

The most significant insight is the identification that CKA's "score saturation" for LLM representations—where most model pairs receive scores clustered near 0.8-0.9—may be a fundamental limitation of kernel-based similarity measures for high-dimensional representations, while topological methods operating on local connectivity structure (NTS) can maintain discriminative power. This suggests that as representations become increasingly high-dimensional (LLM hidden states with d ≈ 4096), comparing local merge-order structure may be more informative than global kernel-based geometric comparisons. The mathematical insight that SRTD = RTD + Max-RTD for the lite case (Corollary 3.4) also clarifies that the empirically observed complementarity between directional RTD and Max-RTD is not coincidental but reflects a decomposition into shared symmetric and private asymmetric components.

## Suggestions

- **Revise scale-invariance claims**: Either formally incorporate Z-score normalization into NTS's definition (updating Theorem 4.1/4.2 accordingly) or explicitly state the conditions under which NTS is scale-invariant and when preprocessing is required.

- **Add quantitative LLM evaluation**: Report a clustering metric (e.g., normalized mutual information between predicted clusters and known model families) to statistically validate discriminativity claims.

- **Analyze core pair overlap**: Report the average edge overlap between MST(w) and MST(w̃) across experiments, and discuss how overlap relates to NTS score variance.

- **Tone down DeepSeek argument**: Rephrase the DeepSeek lineage discussion to acknowledge that the "ground truth" of representation similarity is not well-defined, and that NTS and CKA may be capturing different aspects of similarity.

- **Add efficiency clarification**: Explicitly state that O(n²) distance matrix construction is still required, and discuss whether this dominates computational cost for high-dimensional LLM representations.