Now I have all the information needed. Let me compose the final consolidated review.

## Summary

QZO proposes fine-tuning quantized LLMs via zeroth-order (ZO) optimization by perturbing continuous quantization scales (Q-SPSA) instead of discrete weights, combined with a Directional Derivative Clipping (DDC) technique for training stabilization. The method achieves ~3× memory reduction over MeZO on 4-bit models and works with both scalar-based (GPTQ) and codebook-based (AQLM) quantization, including 2-bit extreme quantization.

## Strengths

- **Novel and clean core idea (Q-SPSA):** The key insight of perturbing continuous quantization scales ∆ while keeping discrete weights θ̄ fixed (Definition 3.3, Eq. 5) elegantly sidesteps the de-quantization/re-quantization problem that makes ZO optimization incompatible with quantized weights. This is the paper's strongest contribution — simple, well-motivated, and effective.

- **Plug-and-play compatibility with diverse PTQ methods:** QZO works with both scalar-based quantization (GPTQ, 4-bit, on three model families in Table 1) and codebook-based quantization (AQLM, 2-bit, on Llama-2-13B in Table 3), confirming the claimed orthogonality to PTQ methods.

- **Meaningful measured memory savings:** Table 1 shows QZO on 4-bit models uses 4.8–6.3GB versus 14.8–20.5GB for MeZO (a genuine ~3× reduction), with careful peak-memory profiling at batch size 1 during the first 100 steps.

- **Empirical demonstration that DDC prevents training collapse:** Figure 2 shows that without DDC, training produces NaN losses from abnormal directional derivatives, while DDC stabilizes training. The ablation (Figure 3) shows a reasonable sensitivity profile for clipping threshold C.

- **Extreme quantization viability:** Table 3 shows QZO on 2-bit Llama-2-13B achieves 80.5 on SST-2 (vs. 57.6 zero-shot) using only 5.78GB, enabling fine-tuning of a 13B model on a single consumer GPU.

## Weaknesses

### Fatal

None.

### Major

- **Theorem 1 is incorrect — clipping introduces bias, not unbiasedness:** Theorem 1 claims the clipped gradient estimate ∇̂'ΔL is an unbiased estimate of the full gradient ∇ΔL. This is false. The clipped estimate is d'·z where d' = clip(d, −C, C) and d is the directional derivative. Since d is a nonlinear function of the random vector z, and clipping is a nonlinear operation, E[clip(d, −C, C)·z_k] ≠ E[d·z_k] = ∇Δ_kL in general. A simple counterexample: if d ≈ a·z₁ + b, then E[clip(a·z₁+b, −C, C)·z₁] ≠ a whenever Pr[|d| > C] > 0. This error propagates into Eq. 8: the variance reduction argument requires E[∇̂'Δ_kL] = ∇Δ_kL (unbiasedness) to cancel the terms ∇Δ_kL² − E[∇̂'Δ_kL]². Without unbiasedness, the sign of this residual is unknown, and the variance reduction claim does not follow. Additionally, the statement that "Var[∇̂'Δ_kL'] ≤ Var[∇̂Δ_kL] holds almost surely" misuses "almost surely" — variance is a population quantity, not a random event. Since DDC with its theoretical justification is presented as a core technical contribution, this incorrect theorem significantly undermines the paper's claimed theoretical contribution. DDC works as a practical heuristic (the ablation demonstrates this), but the theoretical framing is wrong.

- **FLOPs comparison in Table 2 is internally inconsistent and potentially misleading:** The reported QZO FLOPs vary by 2–3 orders of magnitude across models for no clear reason: OPT-6.7B shows 8.19×10¹³ (a ratio of ~12,100× vs. MeZO), Llama-2-7B shows 2.26×10¹⁶ (~50× vs. MeZO), and Llama-3.1-8B shows 7.9×10¹⁶ (~14× vs. MeZO). The paper claims "~1% of FLOPs" but the actual ratios range from 0.008% to 7%. Both QZO and MeZO require two forward passes per step, so the FLOPs per step should be comparable (QZO's quantized model may be slightly cheaper per forward pass, but not by 100×–12,000×). The OPT-6.7B number (8.19×10¹³) appears to count only the parameter update cost (~50M scales), excluding forward passes, while the other models' numbers are higher. The paper does not define what "Total FLOPs" measures, making the comparison unreliable.

- **No experimental comparison with directly competing concurrent methods:** The paper discusses three concurrent works combining ZO with quantization — QuZO (Zhou et al., 2025), Feng et al. (NeurIPS 2024), and Bar & Giryes (ICASSP 2025) — and claims QZO is "inherently more efficient and flexible," yet provides no experimental comparison with any of them. These address the same problem (ZO fine-tuning of quantized models) and were available before submission. Without this comparison, the claimed practical advantages over the most directly comparable baselines are unsubstantiated.

### Minor

- **"On par with MeZO" claim is overstated:** The results are mixed, not uniformly comparable. On Llama-3.1-8B, QZO substantially underperforms MeZO on CB (69.6 vs. 91.1, a 21.5-point gap), RTE (66.8 vs. 70.0), and BoolQ (78.2 vs. 83.4). On OPT-6.7B, QZO lags on SST-2 (87.6 vs. 93.0). The claim should be qualified — QZO is broadly competitive on some models/tasks but shows significant degradation on others.

- **The 18× memory reduction claim conflates contributions and doesn't match measured values:** The abstract claims "more than 18×" reduction vs. full fine-tuning in 16 bits, but the measured ratios in Table 1 are only ~5× (e.g., 31.9GB/6.3GB for Llama-2-7B). The 18× figure likely combines the contributions of quantization (not novel to this paper) and ZO (adapted from MeZO) against a theoretical maximum. The actual measured savings over MeZO alone are ~3×, which is the more honest and relevant comparison.

- **No variance/std reported across runs:** ZO methods are known for high variance, making standard deviations critical for assessing result reliability. No error bars or standard deviations are reported for any result in the paper.

- **Negative quantization scale during perturbation not discussed:** Algorithm 1 includes `max(Δ_i − η_t * d' * z, 0)` to ensure non-negative scales during updates, but the perturbation step `Δ_i + εz` has no such safeguard. If Δ_i + εz becomes negative during gradient estimation, it would invert the sign of the corresponding quantized weight, potentially causing catastrophic outputs. The paper does not address this edge case.

- **DDC ablation limited to one model and one dataset:** The ablation in Section 4.3 uses only Llama-2-7B on SST-2. Given that performance varies significantly across models (Table 1), more comprehensive ablation would strengthen the analysis.

### Trivial

- None.

## Nice-to-Haves

- Comparison with LoRA/QLoRA at similar parameter budgets (~50M) would establish whether quantization-scale parameterization offers advantages over low-rank parameterization, though these are somewhat different use cases (no-backprop vs. backprop).
- Analysis of how quantization scales change during training (magnitude, distribution, correlation with task performance) would reveal whether QZO is genuinely adapting the model or primarily recalibrating quantization error.
- Variance bars on the C-sensitivity analysis (Figure 3) would be informative.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Strength: "Theoretical justification for DDC reducing gradient variance while maintaining unbiasedness" (from Strength Finder):** Removed because Theorem 1 is incorrect — the clipped gradient estimate is biased, which invalidates the variance reduction proof (Eq. 8). A strength cannot conflict with a verified Major weakness.

- **Strength: "Computation efficiency alongside memory efficiency" (from Strength Finder, citing Table 2 FLOPs):** Removed because the FLOPs comparison is internally inconsistent and potentially misleading (verified Major weakness). The computation efficiency claim cannot stand on unreliable numbers.

- **Weakness: "The paper doesn't discuss what happens when the perturbed scale Δ_i + εz becomes negative" (from Harsh Critic):** Moved to Minor tier rather than Major because in practice, quantization scales are positive and the perturbation ε is small (10⁻³), so negative perturbations are likely rare. The algorithm's update safeguard (`max(..., 0)`) suggests awareness, but the perturbation step's lack of safeguard should still be acknowledged.

- **Weakness: "No comparison with parameter-efficient fine-tuning methods (LoRA, QLoRA)" (from Harsh Critic):** Moved to Nice-to-Have. LoRA requires backpropagation while QZO does not — they address different use cases. The comparison would be informative but is not a core flaw.

- **Weakness: "Training on only 1,000 examples with 5 simple benchmarks provides a limited evaluation" (from Harsh Critic):** Removed — this follows MeZO's evaluation protocol and is standard practice in the ZO fine-tuning literature.

- **Weakness: "Missing related works" (from Harsh Critic):** Removed per instructions — cannot verify existence of uncited works.

- **Weakness: "Reproducibility concerns" (from Harsh Critic, implicit):** Removed — the paper provides a public GitHub repository and reproducibility statement (Section 7).

- **Formatting/typo complaints (e.g., "quantization sclaes" in Theorem 1):** Removed as parser artifacts.

## Novel Insights

The key tension in this paper is between its genuinely clever core idea (Q-SPSA) and its attempt to dress up a practical heuristic (DDC) as a theoretically grounded contribution. The paper would be stronger if it presented DDC honestly as a bias-variance tradeoff: clipping outlier directional derivatives introduces some bias but dramatically reduces variance, which stabilizes training. This is a well-understood principle in robust statistics, and acknowledging the bias would not diminish the method's practical value — the ablation clearly shows DDC is necessary for training to converge at all. The incorrect unbiasedness claim, unfortunately, does more damage to the paper's credibility than the bias itself would.

## Suggestions

- Correct or retract Theorem 1. Re-frame DDC as a bias-variance tradeoff: clipping introduces bias proportional to Pr[|d| > C] but reduces variance. This honest framing still supports the method's practical utility and would be more scientifically sound.
- Clarify what "Total FLOPs" measures in Table 2. If it includes forward passes, explain the implausible ratios. If it excludes forward passes, label it as "parameter update FLOPs" and add a separate row for forward-pass FLOPs to enable fair comparison.
- Add experimental comparison with at least one concurrent ZO-quantization method (QuZO being the most directly comparable) to substantiate the claimed efficiency/flexibility advantages.
- Qualify the "on par with MeZO" claim to acknowledge the significant gaps on certain model/task combinations, particularly CB on Llama-3.1-8B.

## Score and Decision

**Calibration anchors compared against:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| ZeroQAT | LUopdQeiz1.md | 2.50 | Similar ZO+quantization topic but had more fundamental understanding gaps; QZO is stronger with a cleaner core idea |
| Fréchet Mean (invalid unbiasedness) | eQD8d9DeZc.md | 1.33 | Invalid unbiasedness proof that invalidated entire theoretical framework; much worse — QZO's core idea (Q-SPSA) is still valid |
| GUM | k3qeF8tBfm.md | 4.50 | Closest match: overstated unbiasedness claim, memory-efficient LLM training. QZO has a cleaner novel idea but also more empirical issues |
| SparQ | FF3o9flavI.md | 4.00 | ZO+quantization for model parallelism; had novelty issues. QZO has more genuine novelty |
| SVD-0 | nlgQsugmGw.md | 4.00 | ZO with theorem issues and overlap concerns; similar quality tier |
| MobiEdit | fb7yTBOV3p.md | 5.00 | ZO+quantization for mobile; stronger practical demo, no incorrect theorem. QZO is weaker |
| FZOO | NMlF3YjS8E.md | 5.00 | ZO optimization improvement; stronger empirical results, no incorrect theorem. QZO is weaker |
| CWD | Gwe6gbGng5.md | 7.00 | Theorem issues but very strong empirical results at billion-parameter scale. QZO's empirics are not as compelling |

QZO sits below FZOO/MobiEdit (5.0, no incorrect theorems, stronger empirics) and above ZeroQAT (2.5, more fundamental issues). It is comparable to GUM (4.5, overstated unbiasedness claim) and SparQ/SVD-0 (4.0, novelty/theorem issues). The incorrect Theorem 1 is more severe than GUM's overstated claim (GUM's theorem may be technically correct but practically vacuous; QZO's theorem is simply wrong). However, QZO has a genuinely novel and clean core idea (Q-SPSA) that works empirically, which is more than SparQ or SVD-0 offered.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>