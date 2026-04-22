Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

The paper proposes TAK (Task Arithmetic with KFAC regularization), a dataless approach to weight disentanglement in task arithmetic. Under linearized fine-tuning, the authors show that representation drift regularization reduces to a quadratic form of the GGN matrix (Eq. 3), which can be approximated via KFAC without requiring external task data at fine-tuning time. They further introduce a Kronecker-factor merging heuristic (Eq. 8) that aggregates per-task curvature factors into a single surrogate, achieving O(1) complexity in the number of tasks.

## Strengths

- **Principled theoretical derivation under linearization**: The connection between representation drift regularization and the GGN/Jacobian Gram matrix (Sections 3.1–3.2) is a clean, non-obvious insight that eliminates data dependency entirely during fine-tuning. The derivation is rigorous under the linearization assumption and bridges the task arithmetic and second-order optimization literatures (Eq. 3–5).

- **Strong task negation results**: Table 2 shows TAK achieves 3.4–3.5% target forgetting on ViT models while maintaining the best control-task accuracy (62.4%/66.4%/72.6%), clearly surpassing τJp (6.7%/4.7%/3.7% target, 60.8%/66.0%/73.0% control). This is a clear and meaningful win.

- **Competitive task addition in the linearized regime**: On 8 Vision, TAK matches or exceeds τJp while being dataless: e.g., 91.6% vs 91.1% absolute accuracy on ViT-L/14 at α=1 (Table 1).

- **α-robustness eliminates held-out tuning**: Figure 4a demonstrates TAK maintains high accuracy across the full α ∈ [0, 2] range in the linearized regime, while competing methods show sharp peaks. This is a significant practical advantage when validation data is unavailable.

- **Task localization provides OOD detection mechanism**: Figure 5 shows that under KFAC regularization, the squared Jacobian norm concentrates near zero for out-of-distribution inputs while remaining high for in-distribution inputs, providing a principled byproduct for OOD detection.

- **Practical efficiency**: Figure 6a shows TAK's training time is ~1/3 of τJp's in the linearized regime. KFAC estimation with MC=1 takes only 3.9 minutes for all 8 Vision tasks (Fig. 6b). The method introduces no inference overhead.

## Weaknesses

### Fatal
None.

### Major

- **The merging heuristic (Eq. 8) is the backbone of the O(1) complexity claim but lacks theoretical justification and is validated on only 8 tasks**: The approximation \(\sum_{t \neq t'} \lambda_t B_t^l \otimes A_t^l \approx (\sum_{t \neq t'} B_t^l) \otimes (\sum_{t \neq t'} \lambda_t A_t^l)\) is mathematically incorrect in general — a sum of Kronecker products is not a Kronecker product of sums. The paper acknowledges this is a "heuristic" (Section 3.4) and notes that Table 3 shows marginal gaps, but provides no analysis of approximation error bounds or conditions under which the heuristic might degrade. Since constant complexity in the number of tasks is a headline contribution, this gap matters more as T grows. Table 3 shows a 0.5–0.6 point gap on ViT-B/32 (86.5→85.8 at α=1), and the paper notes "smaller architectures tend to be more sensitive to curvature regularization and hence to the quality of the approximation." A scaling study with more tasks (e.g., 16–32) would directly test whether the approximation degrades, but none is provided.

- **The "state-of-the-art" claim in the abstract is overbroad**: The abstract claims "state-of-the-art results in task addition and negation" without qualification. On language task addition (Fig. 3a), TAK achieves 78.7% absolute accuracy vs τJp's 81.3% — a 2.6-point gap. The paper acknowledges this in the body ("leveraging data from other tasks (τJp) yields additional gains"), but the abstract's unqualified SOTA claim is misleading for a setting that constitutes half the evaluation scope. The negation results are genuinely SOTA, and vision addition results are competitive, but the language addition gap should be reflected in the abstract framing.

### Minor

- **Non-linear extension lacks theoretical grounding**: The paper transparently states that "our regularization is not theoretically exact in the non-linear regime" (Section 4) and pairs TAK with Attention-Only FT to provide implicit kernel-like behavior. However, Table 1 shows that at α=1, "Attn. Only FT + TAK" achieves only 60.3%/59.0%/82.1% absolute accuracy — significantly below the linearized regime. The non-linear regime is presented as evidence of broader applicability, but the α=1 results suggest the method's benefits in this regime are heavily dependent on coefficient tuning. The paper could more explicitly acknowledge this limitation.

- **The asymmetric λ_t weighting in Eq. 8 is a design choice without justification**: Since \(\lambda_t B_t \otimes A_t = B_t \otimes \lambda_t A_t = \sqrt{\lambda_t}B_t \otimes \sqrt{\lambda_t}A_t\), how λ_t is distributed across the Kronecker factors affects approximation quality. The paper puts all of λ_t on A^l and none on B^l without discussion of alternatives.

- **MC samples beyond 1–2 hurt performance (Fig. 7a) without explanation**: The paper notes "variance across seeds increasing as the number of MC samples grows" but does not investigate why. This is a surprising and potentially informative finding about how MC sampling interacts with the KFAC approximation in this context.

### Trivial
None.

## Nice-to-Haves

- A scaling study with 16–32 tasks would directly test the merging heuristic's robustness and strengthen the O(1) complexity claim.
- An ablation on the asymmetric λ weighting (e.g., comparing \(\sqrt{\lambda_t}\) on both factors vs. λ on A only) would clarify whether the design choice matters.
- Analysis of per-task accuracy breakdown for language results would clarify where the 2.6-point gap to τJp originates.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Dataless" framing misleading**: The harsh critic argues the abstract's "dataless approach" could imply no data dependency whatsoever. However, Algorithm 1 and Section 3.1 make clear that KFAC factors must be pre-computed from task data; the method is dataless at *fine-tuning time*. The abstract's phrasing, while concise, is standard in the literature for methods that eliminate data access during the core procedure. This is a minor presentation preference, not a substantive weakness.

- **Stale curvature at θ₀**: The harsh critic notes KFAC factors are computed at θ₀ and never updated. This is not a limitation in the linearized regime (where the Jacobian is fixed at θ₀ by definition), which is the paper's primary setting. In the non-linear regime, updating curvature would deviate from the theoretical framework. This is more of a theoretical discussion point than a weakness.

- **Squared loss vs. actual training loss for GGN**: The harsh critic suggests using the actual training loss (e.g., cross-entropy) Hessian in the GGN could improve results. However, this misreads the paper's derivation: Eq. 3 shows that the representation drift objective *requires* the Jacobian Gram matrix, which corresponds to the GGN under squared loss. Using cross-entropy GGN would regularize a different (and incorrect for the stated objective) quantity. This is by design, not an oversight.

- **TaLoS numbers from original paper (†)**: Using results from the original paper for comparison is standard practice and not a weakness.

- **Compression results show 4+ point drop**: The harsh critic claims the best accuracy-memory trade-off (Block-8) drops accuracy from 88.3% to ~84%. However, the paper text explicitly states the drop is from 88.3 to 87.1 (~1 point). The harsh critic appears to have misread the figure. The paper's stated numbers are consistent with the ~1-point drop claim.

- **Request for missing related works**: Per instructions, removed.

- **Missing appendix/proofs**: Per instructions, removed.

## Novel Insights

The paper reveals an underappreciated connection between the task arithmetic and second-order optimization communities: representation drift regularization under linearization reduces exactly to a quadratic form of the GGN with squared loss, making the entire toolkit of KFAC approximations available for dataless disentanglement. This bridges two literatures that have largely developed independently. The empirical finding that increasing MC samples beyond 1–2 *hurts* performance in this context is counterintuitive and suggests that the KFAC approximation's interaction with MC sampling in regularization (as opposed to optimization) may behave differently than expected.

## Suggestions

- Qualify the abstract's "state-of-the-art" claim to specify that it holds for negation and vision addition, while acknowledging the gap on language addition relative to data-requiring methods.
- Add at least an informal argument for why the Kronecker-product-of-sums approximation (Eq. 8) might work — for instance, if per-task A and B factors are similar across tasks (expected since they're all computed at the same θ₀), the approximation should be more accurate. Verify this by analyzing factor similarity across tasks.
- Run a scaling experiment with 16–32 tasks to validate the O(1) heuristic under the conditions where it matters most.

## Score and Decision

**Calibration anchors:**

| Paper | Score | Decision | Relation to TAK |
|-------|-------|----------|-----------------|
| vRvVVb0NAz (theoretical TV generalization) | 7.50 | Accept (Oral) | Same topic, much stronger theory |
| g8sGBSQjYk (KFAC/Shampoo parameterization) | 7.33 | Accept (Poster) | KFAC-related, stronger theory |
| dj0TktJcVI (Attn-only FT for TA) | 6.25 | Accept (Poster) | Most topically similar, simpler contribution |
| W2Wkp9MQsF (data-free compression, overclaimed) | 5.75 | Accept (Poster) | Similar overclaim pattern, accepted |
| q3ztjJRQuJ (TA in Trust Region) | 5.75 | Reject | Same topic, weaker contribution |
| rKMz6cDE7W (streaming attention, no experiments) | 2.33 | Reject | Low anchor, no real experiments |

TAK is clearly above the low-scoring papers and the rejected medium papers. Compared to dj0TktJcVI (6.25, the most topically similar), TAK has a deeper theoretical foundation (connecting representation drift → GGN → KFAC), broader experiments (vision + language, addition + negation), and a clear practical advantage (dataless + O(1)). TAK is not at the level of vRvVVb0NAz (7.50) which provides rigorous theoretical guarantees. The merging heuristic's limited validation and the overbroad SOTA claim are real but not fatal weaknesses. The paper is honest about most limitations and the core contribution — making representation drift regularization dataless via curvature approximation — is sound and well-supported.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>