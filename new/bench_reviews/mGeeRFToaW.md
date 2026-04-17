Now I have enough information to write a comprehensive final review. Let me carefully synthesize all the points.

## Summary

This paper proposes Quantized Zeroth-Order Optimization (QZO), a method for fine-tuning quantized LLMs using zeroth-order (ZO) optimization. The key insight is that since ZO gradient estimation requires perturbing continuous parameters but quantized weights are discrete, QZO perturbs the continuous quantization scales (Δ) rather than the discrete weights (θ̄), thereby combining the memory savings of quantization (low-precision weights) with ZO optimization (no gradient or optimizer state memory). A Directional Derivative Clipping (DDC) mechanism is also proposed to stabilize training. Experiments on 4-bit and 2-bit LLMs demonstrate memory savings up to 18× vs. full 16-bit fine-tuning, with performance competitive to MeZO on quantized models.

## Strengths

- **Clean and practical core idea.** Perturbing the continuous quantization scale instead of the discrete weights is a well-motivated, elegant solution to the fundamental incompatibility between ZO optimization and quantized models. It avoids the need for de-quantization/re-quantization cycles and is naturally compatible with both scalar-based (GPTQ) and codebook-based (AQLM) PTQ methods.

- **Compelling memory reduction achieved.** The paper demonstrates that QZO can fine-tune Llama-2-13B on a single 24GB RTX 4090 (Table 3: 5.78GB for 2-bit), which is a concrete and meaningful practical result. The 18× memory reduction over 16-bit full fine-tuning is verified with profiling on real hardware.

- **Empirical effectiveness under extreme quantization.** QZO consistently outperforms zero-shot quantized baselines, including in the challenging 2-bit regime (Llama-2-13B with AQLM). Some results even surpass MeZO (e.g., SQuAD on Llama-2-7B: 85.5 vs. 80.7), which is notable given the much lower precision of QZO.

- **DDC ablation is informative.** The training collapse without DDC (Figure 2) and the sensitivity analysis of C (Figure 3) clearly demonstrate the practical necessity of directional derivative clipping for stable training.

## Weaknesses

### Major:

- **Theorem 1 (unbiasedness of clipped gradient estimator) is almost certainly incorrect as stated.** DDC clips the directional derivative d (which is a function of the perturbation vector z) before multiplying by z, yielding the estimator clip(d, -C, C) · z. The claim that this is an unbiased estimate of ∇_Δ L does not hold in general: (1) clipping is a nonlinear operation and introduces bias when the true directional derivative is nonzero, and (2) even if d were unbiased for the directional derivative, the clipped version d' and z are correlated, so E[d' · z] ≠ E[d'] · E[z]. The proof is relegated to the appendix, and the main text provides no conditions under which unbiasedness would hold. This matters because the subsequent variance-reduction argument in Eq. (8) critically depends on E[∇̂'_Δk L'] = ∇_Δ_k L, which is used to cancel terms. The empirical effectiveness of DDC is not in question—only the theoretical justification. The paper should simply present DDC as a variance-reduction heuristic with strong empirical support, rather than claiming it produces an unbiased gradient estimator.

- **Misleading FLOPs comparison.** Table 2 claims QZO uses "about 1% of the FLOPs of MeZO," but the numbers are inconsistent across models (8.19×10¹³ vs. 9.91×10¹⁷ for OPT-6.7B is a ~12,000× ratio, while 7.9×10¹⁶ vs. 1.13×10¹⁸ for Llama-3.1-8B is only a ~14× ratio). Moreover, both QZO and MeZO require two full forward passes per step—FLOPs are dominated by these forward passes, not by the parameter update. The most likely explanation is that the FLOPs count only multiplies involving trainable parameters (the ~1% that are scales), ignoring the full-model forward passes. This makes "computation-efficiency" claims misleading, as the forward-pass cost dominates total training compute and is comparable between QZO and MeZO.

- **No direct empirical comparison with concurrent ZO + quantization methods.** The paper cites ZOQO (Bar & Giryes, 2025) and QuZO (Zhou et al., 2025), states that QZO is "inherently more efficient and flexible," but provides zero empirical head-to-head comparison. Since these methods target exactly the same problem (ZO fine-tuning of quantized LLMs), the absence of comparison makes it impossible to assess whether QZO's different approach (perturbing scales vs. quantizing perturbations) actually offers practical advantages. The novelty claim rests on this distinction, and it is empirically unsubstantiated.

### Minor:

- **Missing QLoRA/LoRA baseline.** Given that QZO's primary selling point is memory efficiency, QLoRA—which also fine-tunes quantized LLMs (with LoRA adapters via backprop)—is the most natural competing paradigm. While QZO likely uses less memory than QLoRA (no gradient or optimizer states needed), QLoRA may offer better performance per unit of memory. The tradeoff is not explored.

- **Limited evaluation scope.** All tasks use only 1,000 training examples, and only classification (SST-2, RTE, CB, BoolQ) and one generation task (SQuAD) are tested. On CB with Llama-2-7B, QZO achieves 69.6 vs. MeZO's 91.1—a 21.5 point gap the paper does not discuss. The narrative focuses on favorable comparisons without acknowledging where QZO significantly underperforms.

- **Non-negative scale projection is unanalyzed.** Algorithm 1 projects scale updates to ensure non-negative scales (line: `max(∆i − ηt ∗ d′ ∗ z, 0)`), which introduces additional bias in the parameter updates. This is not discussed or analyzed anywhere in the paper.

### Trivial:

- Minor typographical error in Section 4.3 where "QZO w/ DDC can be seen as setting C to an infinitely large value" should read "QZO w/o DDC."

## Nice-to-Haves

- Comparison on benchmarks requiring more complex reasoning (e.g., MMLU, GSM8k) would strengthen claims of practical utility.
- Analysis of how quantization scales evolve during training (magnitude/norm trajectories) would reveal whether QZO makes meaningful adaptations or minor adjustments.
- Results with multiple seeds and confidence intervals, given the known high variance of ZO methods.
- Evaluation on larger models (e.g., 70B-scale), which are the most relevant for the memory-constrained scenario the paper targets.

## Removed Points

- **"Not yet released" concerns about ZOQO/QuZO**: Removed per instructions—if the paper cites these works, they are assumed to exist.

- **Scale-only fine-tuning via backprop as a baseline**: While this would be informative (and is mentioned under "Missing QLoRA baseline"), a "scale-only SGD" baseline would be memory-comparable to QZO only with gradient checkpointing, which is a meaningfully different regime. The suggestion to compare with gradient-based methods at comparable memory budgets is reasonable but goes beyond the paper's stated scope of maximizing memory reduction through ZO + quantization.

- **Complaints about fairness of 18× memory comparison**: The paper is transparent that the comparison is against full 16-bit fine-tuning. This is a standard and informative comparison that highlights the target benefit. The comparison against MeZO (16-bit ZO) shows the 3× memory reduction from quantization alone.

- **Requests for theoretical convergence rate analysis for Q-SPSA**: This would strengthen the paper but is not standard for empirical ZO methods papers (MeZO itself was primarily empirical). Not a core flaw.

- **Concerns about 1000 training examples being "too easy"**: This is the standard MeZO evaluation protocol. While larger datasets would be valuable, the setup mirrors the established benchmark in this field.

- **Requests for evaluation on instruction-tuning/chat benchmarks**: Outside the paper's stated scope and adds scope creep.

## Novel Insights

The most interesting observation from the reviews is that QZO's practical effectiveness despite its theoretical flaw (Theorem 1) highlights a disconnect between the formal analysis and what actually matters: DDC works not because it preserves unbiasedness, but because it acts as a variance-reducing stabilizer (essentially gradient clipping adapted for ZO). This reframing—from an attempt at rigorous variance reduction via unbiased clipping to an effective heuristic—would actually strengthen the paper by focusing on what is genuinely novel and useful.

## Suggestions

1. **Remove or significantly revise Theorem 1.** Either provide correct assumptions under which unbiasedness holds, or reposition DDC as a practical variance-reduction technique with empirical (not theoretical) support. The current incorrect claim undermines otherwise solid work.

2. **Fix the FLOPs reporting.** Report total training FLOPs including forward passes (the dominant cost), or clearly state that the "1% FLOPs" figure counts only parameter-update operations and explain why this metric was chosen.

3. **Add at least one head-to-head comparison with a concurrent ZO+quantization method** (ZOQO or QuZO) to substantiate the efficiency/flexibility claims.

4. **Acknowledge performance gaps** where QZO significantly underperforms MeZO (e.g., CB on Llama-2-7B) and discuss what the limited capacity of scale-only updates imposes on adaptation.

## Score and Decision

**Calibration:** I compared against several ZO-LLM papers:
- SensZOQ (ZO+quantization+sparsity, scores 5-6, Accept Poster): Similar topic, modest novelty, sound experiments
- LOZO (low-rank ZO, scores 6-8, Accept Poster): Stronger theoretical contribution
- HiZOO (Hessian-informed ZO, scores 5-6, Accept Poster): Moderate novelty, some theoretical issues
- Sparse MeZO (scores 5-6, Reject): Weak justification, missing baselines
- SubZero (scores 3-6, Withdrawn): Incorrect/misleading claims

QZO has a genuine and practical contribution (enabling ZO fine-tuning on quantized models), clear engineering value (13B on 24GB GPU), and solid empirical results. However: (1) Theorem 1 is incorrect, which is a significant theoretical flaw; (2) the FLOPs claim is misleading; (3) no comparison with directly competing concurrent work. These are substantive but not fatal issues—the method works empirically and addresses a real need. Relative to SensZOQ (accepted with 5-6 scores), QZO has a similar practical contribution but a more serious theoretical error. Relative to Sparse MeZO (rejected with 5-6 scores), QZO has stronger empirical results but also lacks critical baselines.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>