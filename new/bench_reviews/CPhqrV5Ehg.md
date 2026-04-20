## Summary

The paper analyzes Reward Augmented Decoding (RAD) through a matrix completion lens, showing empirically that RAD learns low-rank reward matrices despite its high-rank capacity. Building on this observation, the authors propose ARM (Autoregressive Reward Model), a low-rank parametrization that scores all next-token candidates in a single forward pass. Experiments on detoxification and sentiment control show ARM distils closely to RAD in quality while delivering constant-time decoding regardless of top-k, addressing RAD's linear scaling bottleneck.

## Strengths

1. **Clean matrix completion framing of RAD's training objective.** Section 3.1.1 reformulates RAD's weighted MSE loss as approximating an incomplete reward matrix $P_\Omega(R)$, and Figure 1 empirically demonstrates that $\hat{R}_{\text{RAD}}$ has rank $\sim 10^2$, far below both $|V| = 50257$ and $d = 768$. This analytical lens justifies the entire low-rank design.

2. **ARM achieves constant-time decoding with near-constant quality trade-off.** Table 1 and Figure 6 show ARM processes only $O(L)$ input tokens versus RAD's $O(Lk)$, with measured per-token latency of ~0.001s independent of $k$ (vs. RAD's linear increase from 0.001s to 0.010s at $k=80$). Figures 3 and 4 show the distilled ARM student closely tracks (and on sentiment slightly exceeds) the RAD teacher on the toxicity/fluency and sentiment/fluency Pareto frontier.

3. **Rigorous evaluation methodology.** Rather than cherry-picking single control-strength values, the paper reports continuous trade-off curves across $\beta$ for both tasks (§5.4), which is the correct standard for controlled generation benchmarks. The ablation study (Figure 5) cleanly isolates the effect of the baseline component and regularization on both matrix rank and downstream performance.

4. **Practical distillation pathway from existing RAD checkpoints.** The distillation loss (Eq. 10) enables converting an already-trained RAD model into an efficient ARM without retraining from data, empirically yielding better quality than training from responses alone (§5.4 summary), which is practically valuable given RAD checkpoints already exist.

## Weaknesses

### Fatal
None.

### Major

- **ARM trained from scratch (without distillation) shows consistent fluency degradation vs. RAD.** The paper's central claim — that ARM "performs on par with the more flexible RAD parametrization" (Abstract) — relies almost entirely on the distilled ARM-RAD-teacher setup. The directly trained ARM variant ("ARM resp. only") underperforms on fluency across both tasks (Figure 3 top plot, Figure 4 top-left), which the paper itself concedes in §5.4: "ARM trained on responses only shows slightly worse fluency w.r.t. average perplexity for lower levels of toxicity." This indicates the low-rank constraint *is* binding in practice when not bootstrapped by the high-rank teacher, and the paper overstates the strength of the parity claim. The contribution partially conflates architectural efficiency with distillation-dependent performance.

- **ARM's low-rank behavior is actively engineered by regularization, not discovered as a natural property of the reward task.** Section 3.3 introduces $\mathcal{L}_{\text{reg}}$ (Eq. 11) to push marginal rewards toward the baseline for random tokens. Figure 5a confirms that removing regularization substantially increases ARM's output rank (from ~15 to ~50). Yet the paper frames the low-rank design as motivated by an analysis of *RAD's* learned matrix structure (§3.1), obscuring that ARM's rank behavior is also driven by an explicit architectural inductive bias. The regularization is a sound design choice, but the paper should more clearly separate what is empirically observed in RAD versus what ARM's own training procedure enforces.

### Minor

- **No direct comparison to GeDi/DExperts trained under identical direct-training regimes.** Table 1 lists all four models (GeDi=1 call, DExperts=2 calls, RAD=$k$ calls, ARM=1 call), but the paper's main comparisons (Figures 3-4) show GeDi and DExperts as distant single-point baselines from prior work. Without GeDi/DExperts retrained on the same 2M Jigsaw or Amazon Polarity datasets, it is unclear whether ARM's low-rank factorization offers any real advantage over established single-pass efficient baselines that also score all candidates in one forward pass.

- **The regularization strength is varied only as an on/off ablation.** Figure 5 compares "with regularization" vs. "without regularization," masking whether an intermediate regularization coefficient could better balance the rank-fluency trade-off. The paper itself notes in §5.5 that "a very strong regularization would result in the model always predicting the baseline score," suggesting a non-trivial trade-off curve worth mapping.

- **Efficiency numbers are per-token and do not account for end-to-end pipeline overhead.** Figure 6 measures raw forward-pass time per generated token on a single RTX A6000 GPU. In real deployments, the absolute latency difference ($\sim$2ms at $k=20$) must be weighed against KV-cache management, batched inference, and the base LM's own decoding cost. The paper does not report full-generation wall-clock latency for a fixed-length sequence under realistic batch sizes, making it difficult to assess the practical significance of the speedup.

### Trivial
None that remain after filtering parser artifacts and style nitpicks.

## Nice-to-Haves
- An error analysis identifying specific prompt categories or contexts where directly trained ARM fails to match RAD would strengthen the paper and help practitioners understand the boundary conditions of the low-rank assumption.
- Reporting confidence intervals or multiple runs for the toxicity/fluency trade-off curves would provide a sense of statistical stability, particularly given the closed-source Perspective API's known volatility (which the authors already acknowledge in §5.2).
- A side-by-side qualitative examination of continuations where ARM abstains (predicts baseline) versus where RAD applies strong token-specific adjustments could illuminate whether the low-rank constraint leads to conservative control in complex contexts.

## Removed Points

These points are flagged to be removed; treat them with caution.

- *"The central premise is unproven because the observed low rank of RAD is an artifact of sparse observation matrices and implicit regularization."* — The paper already acknowledges in §3.1.3 that "the presence of a low-rank solution compatible with $\Omega$ does not imply that the true reward, if it could be fully observed, is necessarily low rank." The claim is too harsh given the paper's explicit caveats.

- *"Distillation from RAD is circular by construction and guarantees functional similarity."* — Distillation is a standard and legitimate training paradigm throughout the literature. It does not "guarantee" similarity; a radically underparameterized student can still fail to match a teacher. The concern about distillation dependency is valid (captured in the Major weakness), but the "circular by construction" framing is overstated.

- *"ARM is essentially GeDi/DExperts with a different output layer."* — ARM shares the single-pass paradigm but introduces a baseline + delta decomposition with principled regularization for abstention, which GeDi/DExperts lack. This is more than a notation change.

- *"Guided decoding with large $k$ is rarely used in practice, so the efficiency analysis is irrelevant."* — The $k$-scaling analysis is relevant for understanding algorithmic complexity and future use cases, even if current pipelines use modest $k$. This is scope creep (soft rule).

- *"Missing confidence intervals, theoretical guarantees, user studies, or full wall-clock benchmarking under batched inference."* — Single-run evaluation with well-established protocols (Deng & Raffel 2023; Liu et al. 2021) is the norm in this subfield. These are nice-to-haves moved to the appropriate section, not weaknesses.

- *"ARM is incapable of modeling non-linear token-context interactions needed for nuanced control."* — This follows by definition from the linear parametrization (Eq. 7) and does not constitute a flaw — every method sacrifices expressivity for some benefit.

## Novel Insights

The matrix completion framing (§3.1.1) is the paper's most valuable conceptual contribution: it converts the opaque relationship between autoregressive reward modeling and token-level control into a clean, testable hypothesis about rank. This reframing productively unifies the RAD and ARM lineages and clarifies *why* high-rank representational flexibility is unnecessary for the tasks studied. The distillation observation (§5.4 summary) — that a RAD teacher naturally compresses noisy prefix responses into deterministic per-context targets, yielding a better-trained student — is also insightful, as it explains *when* distillation offers genuine advantages over raw response training. However, the ARM architecture itself (single-pass scoring with a baseline-delta decomposition) is a straightforward combination of known efficient-generation ideas; the novelty lies in the analysis and the demonstration, not the method.

## Suggestions

1. **Reframe the abstract and introduction** to acknowledge that parity with RAD is primarily demonstrated under distillation, and that directly trained ARM trades some fluency for efficiency. This would make the claim more precise and less vulnerable to overstatement critiques.

2. **Add a regularization-strength sweep** (not just on/off) in the ablation study to map the rank-fluency trade-off curve for intermediate values of the regularization coefficient in Eq. 11.

3. **Include directly trained GeDi/DExperts** on the same datasets and response regimes as ARM, so that the single-pass efficiency paradigm can be properly attributed to ARM's design rather than inherited from the shared decoding-time guidance paradigm.

4. **Clarify in §3.2 and §3.3** that $\mathcal{L}_{\text{reg}}$ is an architectural inductive bias that actively suppresses rank, distinct from the empirical rank observation in RAD. This strengthens, rather than weakens, the paper's analytical honesty.

## Score and Decision

**Calibration.** I compared this paper against several anchors:
- **SASA** (jY5oml9fe9.md; scores 6,6,6,6,6, accepted as poster): A clean empirical controlled-decoding paper with solid experiments and modest novelty — scored uniformly at 6.
- **COLLAB** (7ohlQUbTpp.md; scores 5,6,6,8, accepted as poster): Inference-time controlled decoding with reasonable baselines — borderline to solid accept range.
- **MaRio** (t8eO0CiZJV.md; scores 6,6,8,6,6, accepted as poster): Multi-reward distillation with strong human evals but acknowledged limited algorithmic novelty — similar profile to ARM.
- **MiniPLM** (tJHDw8XfeC.md; scores 6,6,6,8,6, accepted): KD framework for LM pre-training with solid empirical results but some overclaim concerns.

The paper under review shares the profile of the accepted poster papers in the 6 range: clear empirical methodology, a sound analytical framing (matrix completion), and a method that works well but is incrementally built on known efficient-generation principles. The two major weaknesses (distillation-dependent parity and regularization-engineered low rank) prevent it from reaching the 7+ tier, but the solid empirical results, clean ablations, and rigorous trade-off analysis distinguish it from papers scoring 4-5. The efficiency contribution, while modest in absolute latency, is real and useful.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>