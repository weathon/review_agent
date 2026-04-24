## Summary

This paper presents a systematic empirical study investigating the impact of normalization type, position, and pairwise combination on Mamba block performance across sequence modeling (Breakfast) and image classification (ImageNet-100) tasks. The authors conduct an exhaustive sweep across five normalization methods placed before, after, or both before-and-after the SSM module, derive practical recommendations (IN→LN for sequence, RMSN→BN for vision), and interpret findings through weight L2-norm analysis. They validate their top configurations against original Mamba baselines on ImageNet-1k and LRA ListOps.

## Strengths

- **Systematic grid sweep across a neglected design space**: The exhaustive 5×5 pairwise combination study (Table 4) across sequence and vision modalities fills a genuine gap. Prior Mamba variants apply normalization positions ad-hoc without empirical justification (Section 2 catalogs the diversity); this paper establishes the first comprehensive reference grid for these choices.

- **Focused experimental design isolating a single variable**: Unlike many architecture papers that stack multiple untested modifications, this work cleanly isolates normalization type and position while holding everything else constant. Sections 3.3–3.4 (Equations 7–9) formalize the N1/N2 framing clearly, enabling exact replication.

- **Task-specific actionable recommendations**: The paper translates findings into concrete guidance — IN→SSM→LN (72.5%) for sequence and RMSN→SSM→BN (87.3%) for vision — validated externally on ImageNet-1k and ListOps (Table 5). These provide immediately usable design rules for practitioners building Mamba variants.

## Weaknesses

### Fatal
None

### Major

- **Undefined evaluation metric for Breakfast dataset**: The paper reports "Sequence Accuracy (%)" for the Breakfast action segmentation benchmark (Section 4.1, Tables 1–4) without defining how sequence accuracy is computed. Breakfast is typically evaluated via frame-wise accuracy, mean-over-class, edit distance, or F1 scores. Without the metric definition, results cannot be compared to the literature or reproduced. Given that this is a core experimental dataset, this ambiguity undermines the paper's empirical validity for the sequence modality.

- **No computational efficiency analysis despite normalization being a critical cost factor**: Different normalization methods (BN, GN, LN, IN, RMSN) have substantially different memory, compute, and batch-statistics overheads that directly affect Mamba's claimed efficiency advantage. The paper reports only accuracy (Tables 1–5) with zero mention of FLOPs, training time, inference latency, or memory footprint. For a paper making recommendations about Mamba architectures — whose primary selling point is efficiency — omitting the efficiency trade-off of its recommended normalization configurations is a significant gap. A practitioner cannot determine whether the 1–2% accuracy gains from RMSN→SSM→BN over LN→SSM→LN come at an acceptable computational cost.

- **Validation experiments lack essential experimental details**: Table 5 compares recommended configurations against "original" baselines on ImageNet-1k and ListOps, but the paper omits training epochs, batch size, optimizer, learning rate schedule, and network depth for these validation runs. Without these details, the reader cannot assess whether the observed improvements (71.1% vs. 70.8% on ImageNet-1k; 72.5% vs. 56.9% on ListOps) are attributable to normalization changes or differences in training configurations.

### Minor

- **Mechanistic explanation is correlational and the paper itself disclaims it**: Section 4.6 attributes performance gains to "stabilized L2 norms of weight matrices" and a "harmonic structure," but the paper explicitly states "this is not intended as an essential explanation" (line 290). Normalization layers operate on activations, not weights; the observed correlation between weight norms and accuracy does not establish causation. The authors acknowledge this is an "intuitive inference" rather than a rigorous mechanism, which weakens the claim in the abstract/Introduction that normalization "mitigates large variations in weight norms."

- **No error bars or multiple-run reporting**: All results in Tables 1–5 report single-run accuracy values without standard deviations or error bars. For an empirical study whose contribution is the reliability of its numbers, the absence of any variance reporting makes it difficult to judge whether small differences (e.g., 86.7% vs. 86.8% for LN after SSM vs. GN after SSM in Table 3) are meaningful or noise.

### Trivial

- **Figure 3 bar chart labeling is unclear**: The 3D bar chart caption labels both sub-charts as "(a)" and mixes "Type," "Position," "Combination," and "Top3 Result" categories in a way that does not clearly map to the table rows, making the figure difficult to interpret without cross-referencing Table 4 extensively.

## Nice-to-Haves

- Reporting training loss curves for the various normalization configurations would help distinguish cases where the "None" baseline fails to converge versus converges to a poor local minimum.
- Confusion matrices or qualitative segmentation samples for the Breakfast dataset would demonstrate that the model genuinely learns temporal structure rather than exploiting dataset shortcuts.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Broken baseline" / "catastrophic baseline performance indicates a fundamentally broken training pipeline"**: The harsh critic claims a Mamba block without normalization should achieve >40–50% accuracy. This is an unsupported assertion by the critic — not a paper error. The paper correctly shows that without normalization, performance collapses, which is a well-known phenomenon for deep architectures. The 7.0% / 10.7% baselines serve to demonstrate why normalization is necessary; they are not claimed to be competitive results.

- **"Cherry-picking suboptimal baselines in Table 5"**: The critic argues comparing IN→SSM→LN against RMSN→SSM→RMSN (56.9%) is cherry-picked because GN→GN achieves 68.8%. However, RMSN→SSM→RMSN is the original Mamba configuration — comparing against it is the correct baseline for demonstrating improvement over standard practice. The fact that GN→GN also exceeds it does not invalidate the comparison.

- **"Overgeneralization contradicted by own tables"**: The critic claims the "after-SSM is generally better" conclusion contradicts the data because IN benefits from before-SSM in sequence modeling. The paper uses the qualifier "generally" and the data supports it: for 3/5 normalizations in sequence and 4/5 in vision, after-SSM outperforms before-SSM. The paper also acknowledges task-specific differences in its Recommendations section.

- **"Weight L2 norms cannot be a mechanistic explanation because normalization operates on activations"**: While technically correct that normalization acts on activations, the paper itself disclaims the weight-norm analysis as "an intuitive inference" and "not intended as an essential explanation" (Section 4.6, line 290). The critic attacks this as if it were a central claim, but the authors have already softened it to a hypothesis.

- **"Related work reads as a citation list and does not synthesize why prior works chose specific positions"**: Acceptable for an empirical study whose scope is to *determine* the best positions rather than explain historical choices. The paper's purpose is to fill that gap empirically.

- **Requests for undisclosed hyperparameters (epochs, batch size, optimizer) in the primary experiments (Sections 4.2–4.4)**: While missing, these are reproducibility concerns about implementation details rather than flaws in the paper's core contribution. The primary claim — that normalization type and position matter, and specific combinations are best — is supported by the relative differences observed, which are robust to hyperparameter choice as long as they are consistent across configurations.

## Novel Insights

One paragraph synthesizing genuinely novel observations.

None beyond the paper's own contributions. The empirical findings (normalization placement matters, GN excels after SSM, specific combinations outperform single-type setups) are useful but largely consistent with what practitioners would expect from the broader normalization literature. The weight-norm "harmonic structure" observation is tentative and explicitly disclaimed by the authors as intuition rather than mechanism.

## Suggestions

- Define the "Sequence Accuracy" metric for Breakfast and, if possible, also report a standard action-segmentation metric (e.g., frame-wise accuracy, edit score, or F1) to enable cross-paper comparison.
- Add a single paragraph or small table reporting training time, peak GPU memory, and approximate FLOPs overhead for the top-3 normalization configurations versus the baseline, so readers can judge the efficiency trade-offs.
- Include standard deviations (≥3 runs) for the primary results in Tables 1–4, or at minimum for the top configurations and the largest gaps, to demonstrate result stability.
- Add hyperparameter details (epochs, batch size, optimizer, learning rate, weight decay, network depth/width) for the ImageNet-1k and ListOps validation experiments in Section 4.5.

## Score and Decision

**Calibration anchors consulted:**

| Anchor | Avg Score | Comparison |
|---|---|---|
| SSM frequency bias paper (wkHcXDv7cv) | 7.50 (Spotlight) | Much stronger: combines theory with empirical mechanism + tunable solution |
| Hymba hybrid SSM architecture (A1ztozypga) | 7.50 (Spotlight) | Novel architecture with scaling results; above this paper |
| ConvBN blocks transfer learning (lHZm9vNm5H) | 7.50 (Spotlight) | Proposes a new method (Tune mode), not just ablation; above this paper |
| SSM parameterization HOPE (RZwtbg3qYD) | 6.60 (Poster) | Novel parameterization with empirical comparison; above this paper |
| Mix-LN position study (BChpQU64RG) | 6.20 (Poster) | Similar "position study" theme but proposes a new combination method; borderline above |
| Subspace Grid-sweep defense eval (8S7eGD15b6) | 5.25 (Reject) | Similar "brute-force empirical sweep" paper; also rejected. Comparable. |
| Transformer composition study (tHHzfZSP6T) | 5.00 (Reject) | Systematic empirical study on Transformer capabilities; rejected. Comparable. |
| Hierarchical search study (eqVu9eaVAB) | 5.50 (Reject) | Systematic component analysis; rejected. Comparable. |
| Position sensitivity in embeddings (4GD7a9Bo9A) | 4.50 (Reject) | Empirical study on position effects; rejected. Comparable. |
| VLM design choices (5wmAfwDBoi) | 4.25 (Reject) | Comprehensive empirical study; rejected. Comparable. |
| ReAct-style prompting critique (85Ik12q2hP) | 4.00 (Reject) | Empirical critique; rejected but with deeper analysis than this paper. |

The paper under review is most similar to the "systematic empirical study" cluster scoring 4.50–5.50 — competent, thorough component sweeps that are useful but ultimately rejected at ICLR. The high-scoring anchors (6.20–7.50) either propose novel methods/techniques backed by empirical validation, or combine theoretical analysis with empirical findings. This paper does neither: it conducts an exhaustive grid search (valuable but methodologically straightforward) and offers a correlational weight-norm intuition explicitly disclaimed as non-essential. The undefined metric, missing efficiency analysis, and absent details in validation experiments are real weaknesses that further limit its standing.

Positioned against anchors: it is clearly below the 6.20+ papers (no novel method, no theory, no new mechanism). It is comparable to the 4.50–5.50 rejected systematic studies. The quality of execution and the usefulness of the reference grid place it slightly above the 4.00–4.50 anchors, but the significant reproducibility gaps (undefined metric, missing hyperparameters, no error bars, no efficiency data) keep it below a 5.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>