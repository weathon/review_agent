Now I have enough information to write the final review. Let me consolidate everything.

## Summary

The paper proposes TAK (Task Arithmetic with KFAC regularization), a dataless regularizer for task arithmetic that eliminates the need for external task data by connecting representation drift regularization to the generalized Gauss-Newton (GGN) matrix and approximating it via KFAC. The key insight is that the Jacobian Gram matrix underlying representation drift is an instance of the GGN with squared loss, enabling well-established curvature approximation techniques to be repurposed for data-free regularization. The paper also proposes a merging heuristic (Eq. 8) that aggregates per-task KFAC factors into a single surrogate, achieving O(1) complexity in the number of tasks.

## Strengths

- **The theoretical connection between representation drift regularization and the GGN matrix** (Sections 3.1–3.2) is a genuine and novel contribution. The derivation showing that the Jacobian Gram matrix equals the GGN with squared loss (∇²c = I) transforms a data-dependent regularizer into a pre-computable curvature problem. This insight opens a clear, principled path to data-free regularization and is the paper's strongest contribution.

- **TAK achieves strong performance as a dataless method**: In the linearized regime, TAK matches or exceeds the data-dependent competitor τ_Jp across ViT scales (Tab. 1: 85.8 vs. 85.0 on ViT-B/32 at α=1), and strictly outperforms it in task negation (Tab. 2: 3.4 vs. 6.7 target accuracy on ViT-B/32). The dataless property is a significant practical advantage for privacy and modularity.

- **Robustness to task vector rescaling** (Fig. 4) is practically significant: TAK maintains stable accuracy across a wide range of α values, eliminating the need for validation-set tuning — a concrete advantage in settings where α cannot be tuned.

- **Extensive experimental evaluation**: Three ViT scales (B/32, B/16, L/14), T5-base, both linearized and non-linear regimes, task addition, task negation, comparison with post-hoc merging methods, KFAC estimation ablations, compression analysis, and memory profiling. This is above-average experimental coverage.

- **Practical efficiency**: MC KFAC estimation takes only ~4 minutes for 8 tasks (Fig. 6b), and during training TAK requires roughly one-third the time of τ_Jp since it avoids a second forward-backward pass through the linearized model.

## Weaknesses

### Fatal
None.

### Major

- **The Kronecker merging heuristic (Eq. 8) is a stated contribution but lacks any error analysis or scalability validation beyond 8 tasks.** The approximation Σ_t λ_t(B_t^l ⊗ A_t^l) ≈ (Σ_t λ_t B_t^l) ⊗ (Σ_t λ_t A_t^l) is the linchpin of the "constant complexity" claim in the abstract and contributions. This is distinct from the standard KFAC assumption (which factors within a single task and has Gaussian-justification); here, factoring *across* tasks asserts that the sum of Kronecker products equals the Kronecker product of sums. Table 3 shows the gap is small (0.7 absolute on ViT-B/32; negligible or reversed on ViT-B/16 and T5-base), but this is tested only at 8 tasks — the same scale as prior work. The paper provides no analysis of when or why this approximation degrades, no error bounds, and no experiments with more tasks. For a method whose advertised advantage is scalability to many tasks, this is a significant evidential gap. The constant-complexity claim rests on an unanalyzed heuristic validated at a single data point.

- **The non-linear extension conflates TAK with Attention-Only FT, preventing attribution of the improvement.** Section 4 pairs TAK with Attention-Only FT in the non-linear regime, justified by the claim (from Jin et al., 2025) that attention-only fine-tuning "induces approximately linear fine-tuning dynamics." This is a qualitative, citation-based argument — the paper does not measure *how approximately* linear the dynamics are. More importantly, Table 1's comparison (Attn Only FT: 60.3 → Attn Only FT + TAK: 83.1 on ViT-B/32) confounds the effect of TAK's curvature regularization with whatever implicit linearization Attention-Only FT provides. Without an ablation applying TAK to standard non-linear FT (which Table 1 shows performs terribly at α=1: 32.0 on ViT-B/32), the reader cannot assess whether the non-linear results are primarily due to TAK or to the attention-only constraint. The paper acknowledges the theory is not exact in the non-linear regime (line 261), but does not disentangle the two interventions.

### Minor

- **No variance or standard deviation is reported**, making it difficult to assess the statistical significance of small margins (e.g., ViT-B/16: 88.3 vs. 88.2 between TAK and τ_Jp). The paper mentions "variance across seeds" in the KFAC estimation section (Fig. 7a discussion), indicating multiple seeds were at least partially run, but these are not reported in the main tables. While single-run reporting is common in this community, the smallest margins warrant uncertainty estimates.

- **The abstract's "state-of-the-art" claim should be qualified.** The abstract says "state-of-the-art results in task addition and negation" without specifying "among dataless methods." While the abstract's opening sentence frames the contribution as a "dataless approach," on ViT-B/16 with best α, τ_Jp slightly outperforms TAK (88.6 vs. 88.3 absolute). The SOTA claim is clearly valid for the dataless category and for task negation, but should be more precisely stated for task addition.

- **The MC sample degradation finding (Fig. 7a) is counterintuitive and unexplained.** Performance deteriorates with more MC samples beyond 1–2, and variance across seeds increases. The paper notes this but does not investigate whether this reflects optimization instability, a bias-variance tradeoff in the regularizer, or something else. This finding is unusual and deserves discussion since it could inform practitioners.

- **The paper does not explicitly confirm whether KFAC factors B^l are computed with ∇²c = I** (squared loss) in practice, consistent with the theoretical derivation in Section 3.2, or with the cross-entropy Hessian. The theory derives the equivalence between the Jacobian Gram matrix and the GGN specifically under squared loss, but Section 3.3's description of Exact and MC variants references ∇²c_n generically. If cross-entropy KFAC is used instead, there is a mismatch with the theory.

### Trivial
- The term "representation drift" is described verbally as "the change in the last-layer activations" (line 72) but the formal definition z_t(x) = f_lin(x, θ_0 + α_tτ_t) uses the full model output (logits). This terminological inconsistency could confuse readers about what exactly is being preserved, though it does not affect the correctness of the method.

## Nice-to-Haves

- Experiments evaluating the merging heuristic (Eq. 8) with more tasks (16–32+) would directly validate the O(1) scalability claim, which is a stated contribution.
- A per-task accuracy breakdown for the merging comparison (Table 3) would reveal whether the approximation error is concentrated on specific tasks or diffuse.
- Discussion of privacy implications of releasing KFAC factors alongside pre-trained weights, given that KFAC factors encode input/gradient covariances and the paper motivates the method partly through privacy (line 86).

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Best α equals α=1 for TAK on ViT-B/16 and ViT-L/14, raising question about search range"**: This is not a weakness — it directly demonstrates the α-robustness that the paper claims as a strength. If α=1 is the best value, this confirms that no tuning is needed, which is the paper's point.
- **"Figure 5 OOD detection is speculative without proper OOD evaluation"**: The paper says the normalcy score "suggests a natural use" for OOD detection — this is appropriately hedged language indicating a potential direction, not a claim. Criticizing it as if it were a claim is a strawman.
- **"Missing related works"**: Removed per hard rules — cannot verify existence of uncited works.
- **"Reproducibility concerns about undisclosed hyperparameters"**: The paper states complete source code with all hyperparameters will be released (line 319). Removed per hard rules on reproducibility nitpicks.
- **"Privacy implications of releasing KFAC factors" as a Major weakness**: The paper motivates the method through data-access privacy (not needing other tasks' data for regularization), not through information-theoretic privacy of the KFAC factors themselves. Whether KFAC factors leak training data is a separate research question. Moved to Nice-to-Have.

## Novel Insights

The paper reveals an underexplored connection: that the data-dependency bottleneck in representation drift regularization — a seemingly fundamental constraint — can be circumvented entirely by recognizing that the Jacobian Gram matrix is an instance of the GGN. This reframing shifts the problem from "how to avoid needing task data" to "how to approximate a well-studied curvature matrix without data," for which decades of second-order optimization research provides answers. The MC-sample degradation finding (Fig. 7a) is also intriguing and under-discussed — it suggests that more accurate curvature estimates can paradoxically harm regularization, possibly because a noisier curvature approximation acts as an implicit regularizer of the regularizer itself, a phenomenon worth investigating further.

## Suggestions

- Add a basic quantitative analysis of the merging approximation error (e.g., Frobenius norm ratio between merged and unmerged curvature matrices across layers) and test with more than 8 tasks, even if synthetic. This would directly strengthen the scalability claim.
- Include an ablation of TAK applied to standard non-linear FT (not just attention-only) to disentangle the contributions of each component in the non-linear regime.
- Report standard deviations for at least the main comparison tables (Tab. 1, Tab. 3), given that the paper already mentions running multiple seeds for some experiments.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| Narrow Finetuning / Activation Difference Lens (qyVzZsrsnS) | 7.50 | Accept Poster | Stronger novelty and cleaner evaluation; TAK has comparable conceptual novelty but weaker validation of merging heuristic |
| WSM: Learning rate decay ↔ model merging (HhThhjKyfw) | 7.00 | Accept Oral | Cleaner theoretical contribution with stronger proofs; TAK's GGN insight is comparable in depth |
| OTA Merging / Curvature-informed (ULxerRB2DF) | 6.00 | Reject | Similar topic (curvature for merging); TAK has stronger theoretical grounding (full KFAC vs diagonal) and broader evaluation (3 ViT + T5 vs Llama-only), but shares the weakness of unanalyzed approximations |
| Gram-weighted Mahalanobis merging (y0gom847Oy) | 5.33 | Reject | Dataless merging with limited validation; TAK is clearly stronger due to principled GGN connection |
| Gompertz approximation (1ybp5rRz32) | 1.33 | Reject | No error bounds, single example; TAK is far above this — it has genuine theoretical grounding and extensive experiments |

TAK sits above the rejected curvature-informed merging papers (6.0) due to its stronger GGN theoretical connection and broader experimental coverage, but below the clearly-accepted papers (7+) due to the unvalidated merging heuristic and the conflation in the non-linear regime. The core contribution (GGN ↔ dataless regularization) is genuine and impactful, but the paper's second contribution (O(1) merging) is inadequately validated for the role it plays in the claims.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>