## Summary

This paper introduces a circuit tracing method for transformers based on the observation that attention scores are *sparsely decomposable* in the basis given by the SVD of the QK-derived matrix Ω. By projecting residual streams onto the small set of singular-vector subspaces that dominate an attention head's score, the method identifies low-dimensional "signals" mediating inter-head communication and constructs a communication graph. The approach is demonstrated on GPT-2 small performing the Indirect Object Identification (IOI) task, recovering known circuit components and identifying new ones, with causal validation via targeted interventions.

## Strengths

- **Novel and elegant core observation**: The finding that attention scores are sparsely constructed in the SVD basis of Ω (rather than being due to low-rank structure of Ω itself) is a genuinely new insight with clear implications for interpretability. Figure 1 compellingly illustrates this sparsity, and the mathematical framework (bilinear form, orthogonal slices, projection-based denoising) is clean and well-presented.

- **Practical advantages over patching**: The method operates on a single forward pass without requiring counterfactual datasets, avoiding known pathologies of patching-based methods (self-repair, dataset dependency). This is a concrete methodological benefit acknowledged in Section 2.

- **Thoughtful causal validation**: The paper validates traced edges with both ablation and boosting interventions in identified subspaces (Fig 6), demonstrating that these have significantly larger effects than random subspace interventions. The distinction between local and global interventions (and their differing effects) provides genuine mechanistic insight, including evidence for redundant parallel pathways (Fig 7).

- **Recovers and extends known results**: The trace recovers heads from Wang et al. (2023) with precision 0.52 and recall 0.69 while identifying new functionally active heads (e.g., (2,8), (4,3)) and elucidating redundant lattice structures at layers 7-8-9. The method goes beyond prior work by identifying the actual subspaces used for inter-head communication, not just which heads communicate.

## Weaknesses

### Major:

- **The claim that sparse decomposability is a general property of attention heads is insufficiently supported**. The main quantitative evidence is the cardinality of S_{ij} for a handful of IOI-relevant heads on 256 IOI prompts plus 256 Pile snippets. No statistics are reported over all heads, no alternative tasks are tested, and no comparison to alternative bases (e.g., random orthogonal, PCA on activations) is provided to establish that the SVD of Ω is *uniquely* effective rather than just convenient. The heuristic for defining S_{ij} ("largest set of terms whose sum ≤ 0, then take remaining positive terms") is presented without theoretical justification and without sensitivity analysis. Alternative definitions (e.g., top-k by magnitude) might yield different circuits. This substantially weakens the conceptual contribution claimed in the abstract and Section 6.

- **Overclaiming the interpretability of identified signals**: The abstract claims the method identifies "features used to communicate between attention heads," and Section 5.2 states "signals show interpretability." The actual evidence for this is extremely thin: a single anecdote showing that the V-subspace of head (9,9) separates names from non-names. There is no systematic analysis of what the identified subspaces encode across multiple heads and edges, no comparison to simpler probes (e.g., linear classifiers on raw residuals), and no demonstration that the same subspace serves as a reusable feature across edges. The paper moves from "this low-dimensional subspace influences downstream attention" to "these are semantically grounded features" without adequate evidence for the latter.

- **No systematic comparison to existing circuit tracing methods**: The paper claims practical advantages over patching-based methods (speed, no counterfactuals, no self-repair) but provides no quantitative comparison. There is no benchmarking against ACDC, EAP, path patching, or the single-forward-pass method of Ferrando & Voita (2024) in terms of circuit recovery quality, completeness, or computational cost. The moderate precision (0.52) and recall (0.69) against Wang et al. is mentioned only in the appendix and not contextualized against what other methods achieve on the same benchmark.

### Minor:

- **Contribution metric (Eq. 7) has an ad hoc √σ weighting**: The choice to split σ equally between source and destination tokens is a design choice without derivation. Alternative weightings could change the ranking of upstream heads and the resulting graph structure. The impact of this choice is not examined.

- **Narrow evaluation scope**: Only GPT-2 small on the IOI task is studied. Whether the approach generalizes to other models, larger transformers, or other tasks (e.g., docstring, greater-than) remains unknown. The paper acknowledges this limitation but does not provide even preliminary evidence beyond the non-specific dataset comparison in Fig 3(b), which uses only 256 short Pile snippets.

- **MLP contributions are excluded**: The paper explicitly states that "extending our framework to include the contributions of MLPs is an important direction for future work." Since MLPs are known to participate functionally in IOI circuits, the resulting trace is an attention-head-only subgraph rather than a complete circuit.

- **Firing definition restricts scope**: Only heads placing >50% attention on a single source token are traced. This excludes diffuse but potentially important multi-source attention patterns. The paper acknowledges this but the scope limitation should be more prominent in claims about "tracing circuits."

## Trivial

- The 70% cumulative-contribution threshold for selecting upstream heads (Section 5.3) and the 65-occurrence threshold for edge filtering (Figure 5) are both arbitrary without sensitivity analysis, though the qualitative structure of the graph is unlikely to change dramatically under modest variations.

## Nice-to-Haves

- Evaluate sparse decomposition on at least one additional model and task to assess generalizability.
- Compare against patching-based methods (ACDC, EAP) on the same IOI benchmark, reporting precision/recall and computational cost.
- Provide systematic interpretability analysis of identified signal subspaces across multiple heads (e.g., probing classifiers, correlation with linguistic features).
- Extend the framework to incorporate MLP contributions, even approximately.
- Discuss how the identifiability concerns raised by recent work (multiple valid decompositions of the same circuit) apply to the SVD-based approach.

## Removed Points

- **Criticisms about SVD basis non-uniqueness with degenerate singular values**: While the SVD of Ω is unique only up to unitary transformations within equal singular values, the paper's definition of S_{ij} is based on which terms make large positive contributions to the actual attention *score* on specific inputs, not purely on the SVD structure. This partially addresses the concern since the empirical sparsity pattern depends on input-dependent projections, not just the weight matrix. Still, sensitivity of S_{ij} to small perturbations deserves future attention.
- **Criticisms that the paper does not establish causal effect on logit difference (only on attention scores)**: The paper does validate with logit difference interventions (Section 5.4, using F(X) following Wang et al.), so this critique is partially addressed. However, the trace graph G is constructed from attention-score contributions, and the mapping from score changes to logit changes is not always tight.
- **Formatting nitpicks (e.g., "nowt" instead of "not" in Section 1, bias dimension handling in Appendix)**: Removed as formatting/style nitpicks.
- **Reproducibility concerns about undisclosed hyperparameters**: The key algorithmic parameters (S_{ij} definition, 70% threshold, 50% firing) are all stated in the paper. Detailed implementation (e.g., layernorm folding, zero-centering) is described in the Appendix. Reproducibility of the method is not a major concern.
- **Risk of interpretability illusions (dormant pathways)**: The paper's local vs. global intervention comparison (Section 5.4) actually provides empirical evidence relevant to this concern, showing cases where local > global and vice versa. While a formal discussion of dormant pathways would strengthen the paper, the experiments are not oblivious to this issue.

## Novel Insights

The paper's most distinctive contribution is the recognition that attention scores, when expressed in the SVD basis of the QK bilinear form Ω, are typically *sparsely constructed* — a small number of orthogonal slices dominate the score computation. This is conceptually different from observing that Ω is low-rank (it generally is not for these heads) or that specific feature directions are important. The sparsity arises from the *input representations* aligning with specific singular vectors, suggesting a structured relationship between learned feature directions and the attention mechanism's internal basis. The theoretical discussion in Section 6 (Lemma 1, near-orthogonal feature sets) provides a plausible mechanistic account, though it remains speculative without direct measurements. The demonstration that local interventions sometimes outperform global ones (implying signals are modified or amplified downstream) is a concrete empirical finding that future circuit analysis methods should confront.

## Suggestions

- **Test alternative S_{ij} definitions and report sensitivity**: Try top-k by magnitude, thresholding on proportion of total score, and random subset selections. Report how the traced graph and intervention effects change. This is the single most impactful improvement.
- **Compare against at least one existing circuit tracing method on IOI**: Run ACDC or EAP on the same dataset and report side-by-side precision/recall against the Wang et al. gold standard, plus computational cost comparison.
- **Report the S_{ij} definition's reconstruction quality**: Show average |A'_{ij} - \sum_{k∈S_{ij}} x_i^T D_k x_j| across heads and prompts, as a function of |S_{ij}|. This quantifies how well the sparse decomposition actually approximates the full score, directly testing the "approximate equality" claimed in Eq. 5.
- **Provide head-by-head interpretability analysis**: For each traced edge, project the signal subspace onto token dimensions and check whether it encodes identifiable features (name detection, duplicate detection, positional information). This would substantiate the claim that these subspaces carry "features used for communication."

## Score and Decision

**Calibration**: I compared against Sparse Feature Circuits (score 8, oral — strongly validated, interpretable features, downstream applications), CD-T (avg ~6.3, poster — efficient method with clear baselines, quantitative benchmarks), Subspace Activation Patching/Interpretability Illusions (avg ~6.3, poster — identifies important problem, formal analysis), Hierarchical Tracing (avg ~3.4, withdrawn/reject — weak validation, lack of faithfulness evidence, ad-hoc method), and Is MI Identifiable (avg ~7, poster — conceptual but rigorous). This paper is more novel and better validated than Hierarchical Tracing, but significantly less rigorously evaluated than CD-T or Sparse Feature Circuits. It lacks quantitative baselines, has moderate (0.52/0.69) recovery against known circuits, and overclaims interpretability. Its core observation is genuinely novel and the intervention experiments are thoughtful, but the execution gaps are substantial.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>