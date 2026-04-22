Now I have a thorough understanding of the paper and the calibration landscape. Let me write the final consolidated review.

## Summary

The paper proposes QZO (Quantized Zeroth-Order Optimization), which enables fine-tuning of quantized LLMs using zeroth-order optimization. The core idea is to perturb the continuous quantization scale (Δ) rather than discrete quantized weights for gradient estimation (Q-SPSA), combined with a directional derivative clipping (DDC) method for training stabilization. QZO is demonstrated on 4-bit LLMs using GPTQ and 2-bit LLMs using AQLM, achieving measured memory savings of 5–6× over full fine-tuning.

## Strengths

- **Elegant and well-motivated core idea**: Perturbing continuous quantization scales instead of discrete weights (Q-SPSA, Definition 3.3, Eq. 5) cleanly sidesteps the de-quantization/re-quantization problem that makes ZO optimization incompatible with quantized weights. This is a simple, principled insight that is orthogonal to the choice of quantization method.

- **Meaningful and real memory savings**: Table 1 demonstrates that QZO reduces peak GPU memory from 26–32GB (full fine-tuning) to 4.8–6.3GB across three model families, enabling 7B-model fine-tuning on a single 24GB consumer GPU. The measured 5–6× savings are practically significant even if not as large as the headline claim.

- **Generality across quantization paradigms**: QZO works with both scalar-based (GPTQ, 4-bit, Table 1) and codebook-based (AQLM, 2-bit, Table 3) quantization methods, demonstrating plug-and-play compatibility. The 2-bit results on Llama-2-13B (Table 3) show QZO's potential under extreme quantization.

- **Empirical effectiveness of DDC**: Figure 2 clearly shows training collapse without DDC, and Figure 3 shows robustness to the clipping threshold C over a reasonable range (75–150), providing practical guidance.

## Weaknesses

### Fatal
None

### Major

- **Theorem 1 (unbiasedness of clipped gradient estimate) is incorrect** — The theorem claims that the clipped gradient estimate $\hat{\nabla}'_\Delta\mathcal{L} = d' \cdot \mathbf{z}$ (where $d' = \text{clip}(d, C)$) is an unbiased estimate of the full gradient. This is false in general because $d$ is correlated with $\mathbf{z}$ (since $d \approx \mathbf{z}^\top \nabla_\Delta \mathcal{L}$), and clipping a quantity correlated with $\mathbf{z}$ introduces bias. A simple counterexample: in 1D with $\nabla L = 1$, the unclipped SPSA estimate $\mathbb{E}[d \cdot z] = 1$, but $\mathbb{E}[\text{clip}(d, C) \cdot z] < 1$ for any finite $C$ because clipping truncates the contribution of large perturbations. Since the variance reduction argument in Eq. (8) relies on Theorem 1 (specifically requiring $\mathbb{E}[\hat{\nabla}'_{\Delta_k}\mathcal{L}]^2 = \nabla_{\Delta_k}\mathcal{L}^2$ to cancel terms), the theoretical justification for DDC collapses. In practice, with the default $C = 100$, the bias is negligible, so the method works—but the theoretical contribution as stated is invalid. The paper should either provide a correct proof (which appears impossible in general), or honestly characterize DDC as a heuristic with empirical but not theoretical justification.

- **The headline 18× memory reduction claim is misleading** — The abstract states QZO "can reduce the total memory cost by more than 18× for 4-bit LLMs." However, Table 1 shows measured peak memory reductions of 5.6× (OPT-6.7B: 26.8→4.8GB), 5.2× (Llama-2-7B: 26.0→5.0GB), and 5.1× (Llama-3.1-8B: 31.9→6.3GB). The 18× figure appears to come from comparing a theoretical fine-tuning maximum (56GB from the introduction's breakdown) against QZO's theoretical minimum (int4 weights only, ~3.5GB), ignoring all runtime overhead. This is a factor-of-3 overstatement relative to measured results. The abstract presents this as the primary result without clarifying it is a theoretical best-case, which misrepresents the method's actual demonstrated achievement.

### Minor

- **Missing experimental comparison with concurrent methods** — The paper cites three concurrent works solving the same problem (Feng et al., 2024; Zhou et al., 2025; Bar & Giryes, 2025) and claims QZO is "inherently more efficient and flexible," but provides no experimental comparison. QuZO (Zhou et al., 2025) targets the exact same problem setting. Without head-to-head comparison, the magnitude of QZO's contribution relative to the state of the art remains uncalibrated.

- **The "on par with MeZO" claim overstates some results** — While QZO matches or exceeds MeZO on many tasks (e.g., Llama-2-7B SQuAD: 85.5 vs 80.7), there are substantial gaps on some: OPT-6.7B SST-2 (87.6 vs 93.0), and notably Llama-3.1-8B CB (69.6 vs 91.1). The claim is partially accurate but should be qualified.

- **FLOPs comparison in Table 2 needs clarification** — The paper claims QZO uses "about 1% of the FLOPs of MeZO," but for OPT-6.7B the ratio is 8.19×10¹³ / 9.91×10¹⁷ ≈ 0.008% (a 12,000× difference), while for Llama-3.1-8B it is 7.9×10¹⁶ / 1.13×10¹⁸ ≈ 7%. Both QZO and MeZO require two forward passes per step, so the forward pass FLOPs should dominate and be similar. The wildly varying ratios across models and the implausibly large gap for OPT-6.7B suggest the FLOPs may not be counting the same operations. The paper should clarify what is being counted.

- **No ablation isolating the core design choice** — The paper's key contribution is perturbing Δ instead of θ̄. An ablation comparing Q-SPSA (perturb scales) against a naive baseline that perturbs de-quantized weights and re-quantizes would directly quantify the benefit of this design choice, which is currently only argued qualitatively.

### Trivial
None

## Nice-to-Haves

- An analysis of what functional changes scale-only updates can express. With group size 128, QZO trains ~1% of parameters and can only uniformly rescale groups of weights. Discussing what task adaptations this parameterization can and cannot capture would help predict failure modes.

- Training loss curves beyond the first 1,000 steps shown in Figure 2, to reveal whether DDC affects convergence speed and final performance beyond preventing collapse.

- A QLoRA baseline to position QZO on the quality-memory Pareto frontier for quantized fine-tuning.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic: "No comparison with QLoRA"** — QLoRA uses first-order optimization with LoRA adapters, operating in a fundamentally different regime (it stores gradients and optimizer states for the adapters). Including it as a baseline would be informative but is not strictly necessary since QZO's contribution is specifically about combining ZO with quantization, and QLoRA doesn't do ZO.

- **Harsh critic: "Projection step max(Δi − ηt * d' * z, 0) not discussed"** — The non-negative constraint on scales is a natural property of quantization scales (they cannot be negative). While not discussed in the methodology section, this is an implementation detail that naturally follows from the problem structure and doesn't require separate analysis.

- **Strength Finder: "The overall memory reduction from fine-tuning (e.g., 26.8GB) to QZO (4.8GB) delivers the claimed 18× reduction"** — This is factually incorrect. 26.8/4.8 = 5.6×, not 18×. This strength conflicts with a verified Major weakness and is dropped.

- **Strength Finder: "Theoretical justification for directional derivative clipping — Theorem 1 proves the clipped gradient estimate remains unbiased"** — This strength conflicts with the verified weakness that Theorem 1 is incorrect. Dropped.

- **Harsh critic: "Notational problem in Eq. (8) from line 2 to line 3"** — This is a minor presentation nitpick that doesn't affect the substance of the argument.

## Novel Insights

The paper reveals an interesting asymmetry in its empirical results: QZO outperforms MeZO on Llama-2-7B (winning 4/5 tasks) but underperforms on Llama-3.1-8B (losing 3/5 tasks, with a dramatic 21.5-point CB gap). This suggests that the effectiveness of scale-only ZO updates may depend significantly on the base model's architecture and how well its pre-trained weights are calibrated for quantization—a dimension the paper does not explore but that could explain when and why QZO succeeds or fails.

## Suggestions

- Fix or retract Theorem 1 and the variance reduction argument. Replace with an honest characterization: DDC introduces small bias but substantially reduces variance, and for large enough C the bias is negligible. A bound on the bias magnitude as a function of C and gradient norm would be a more useful and correct theoretical contribution.

- Replace the 18× claim in the abstract with the measured 5–6× memory reduction, or clearly distinguish theoretical maximum from measured results (e.g., "up to 18× in theory, 5–6× measured in practice").

- Add at least one experimental comparison with a concurrent ZO-quantization method (QuZO) to calibrate QZO's contribution against the current state of the art.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LOZO | /home/wg25r/review_agent/human_reviews/9BiVepgmWW.md | 7.0 | Same topic (ZO for LLMs); stronger theoretical foundation and cleaner experiments. QZO has a more novel combination (quantization+ZO) but weaker theory. |
| Sparse MeZO | /home/wg25r/review_agent/human_reviews/4Kw4KAoVnx.md | 5.5 | Same topic; also had weak theoretical justification for an empirical method. QZO has a cleaner core idea but more serious theoretical flaws (incorrect theorem). |
| Overclaimed theorems paper | /home/wg25r/review_agent/human_reviews/2ErS9Bkc3O.md | 4.5 | Similar pattern: theorems derived under strong assumptions yet presented as general. QZO's Theorem 1 is actually incorrect (worse than overly strong assumptions). |
| Variational Inference (wrong claims) | /home/wg25r/review_agent/human_reviews/pu7a7JHW20.md | 3.0 | Multiple incorrect technical claims. Much weaker overall contribution than QZO. |
| Biased estimator / unbiased claim | /home/wg25r/review_agent/human_reviews/wUaOVNv94O.md | 4.0 | Similar issue: questionable unbiasedness claim. QZO has a stronger empirical contribution. |
| H-QLoRA | /home/wg25r/review_agent/human_reviews/B4S1GAMBLG.md | 2.0 | Quantization+fine-tuning space; fundamentally flawed contribution (numerically equivalent to baseline). QZO is much stronger. |

QZO falls below Sparse MeZO (5.5) because of the incorrect theorem, but above the biased estimator paper (4.0) because of its genuinely novel and impactful core idea with real practical results. The 18× overclaim pushes it below 5.0, but the method's practical value keeps it from going lower.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>