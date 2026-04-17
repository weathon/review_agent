Now I have a thorough understanding of the paper. Let me write the final review.

## Summary

This paper challenges the prevailing multi-task LoRA paradigm that relies on multi-component architectures (multi-adapter, multi-head with routing) designed to isolate task-specific knowledge. Through a series of observations, the authors show that: (1) a simplified multi-head variant (M-LoRA) without routing outperforms complex baselines despite high inter-head similarity; (2) a standard single-adapter LoRA with increased rank matches multi-component architectures. Based on these findings, they propose Align-LoRA, which adds a KL-divergence (or MK-MMD) alignment loss on the down-projection representations across tasks, achieving competitive or superior performance with zero inference overhead.

## Strengths

- **Important and timely challenge to prevailing assumptions.** The paper directly questions the widely adopted multi-component LoRA paradigm with compelling counter-evidence. The M-LoRA paradox — that a simpler, no-router variant with high redundancy outperforms diversity-enforcing architectures — is a genuinely thought-provoking finding supported by clear evidence (Table 1, Section 3).

- **Zero inference overhead by design.** Unlike multi-component architectures with non-mergeable routers, Align-LoRA can be merged into the base model post-training. This practical advantage is meaningful and clearly articulated.

- **Consistent improvements across models and benchmarks.** A-LoRA-K outperforms all baselines on BBH (out-of-domain) across Qwen2.5 (3B, 7B, 14B) and LLaMA3-8B (Table 4), and on 8-task in-domain benchmarks (Table 5), with smaller parameter budgets than multi-component methods. These improvements are consistent and non-marginal (~1.5–3.5 points on BBH).

- **The empirical result that high-rank single LoRA matches multi-component designs is valuable in its own right.** It serves as a strong baseline that the community should consider when proposing new multi-task PEFT architectures.

## Weaknesses

### Major:

- **The causal interpretation overreaches the evidence.** The paper's central claim — that "learning task-shared representations provides a powerful alternative to architectural isolation" — is drawn from the observation that M-LoRA (no router + summation + multi-head dropout) outperforms R-LoRA and that high-rank LoRA matches multi-component methods. However, M-LoRA differs from R-LoRA on multiple axes simultaneously (no routing, summation vs. softmax aggregation, presence of multi-head dropout). The only ablation provided is "HydraLoRA w/o Router," which removes the router from a variant that *doesn't use dropout*, so it cannot isolate the effect of dropout vs. routing vs. aggregation. The paper attributes the success to "collaborative ensemble" from dropout+summation, but without controlled ablations (e.g., M-LoRA without dropout, R-LoRA with dropout but without routing), the claimed mechanism remains a plausible but unverified hypothesis. This is especially important because the conceptual claim is quite strong — that the community's focus on diversity/isolation is "fundamentally misguided."

- **No statistical significance analysis.** Across all tables, no standard deviations or confidence intervals from multiple runs are reported. Many performance differences are in the 1–3 point range (e.g., Table 1 averages: 75.45 vs 74.67 for M-LoRA vs R-LoRA). Given that PEFT methods can be sensitive to initialization and hyperparameters, the lack of variance reporting makes it difficult to assess whether some claimed improvements are robust or within noise. This is a meaningful gap given that the paper's narrative rests on fine-grained performance differences between architectural variants.

- **The theoretical section (5.3) is largely decorative.** The generalization bound presented is a standard domain-adaptation-style result: expected risk ≤ empirical risk + distribution discrepancy + complexity term. This follows the well-known Ben-David/Pan pattern. Crucially, no specific connection is drawn between the KL/MMD losses used in Align-LoRA and the ∆(D_i, D_j) term in the bound. The paper claims that "minimizing ∆ leads to a tighter bound," but this is trivially true of any alignment method and does not explain *why* Align-LoRA's specific mechanism (Gaussian approximation of A·X, symmetric KL) should be effective, nor does it differentiate Align-LoRA from any other representation-alignment approach. The theory adds no specific insight about LoRA structure, rank, or the observed empirical behavior.

- **Limited evaluation on highly conflicting/diverse task regimes.** All experiments use tasks (QNLI, PiQA, Winogrande, ARC, GSM8K, and Flan-v2 subsets) that are relatively harmonious — there is no evaluation on task pairs known to exhibit strong negative transfer. If aggressive alignment of representations hurts performance on genuinely conflicting tasks, the "shared is always better" thesis would be significantly weakened. The paper mentions robustness on "heterogeneous and complex task benchmarks" in Appendix H.2 (not in the main text), but does not analyze per-task win/loss patterns or explore the boundary conditions of the alignment approach.

### Minor:

- **Scaling with the number of tasks.** The alignment loss sums over all i<j pairs, yielding O(M²) complexity. With M=5 or M=8 tasks this is manageable, but the paper positions this as a general "multi-task paradigm" without discussing what happens when M is large (e.g., 50–100 tasks) or in continual task addition scenarios.

- **The Gaussian + diagonal covariance assumption for A·X is strong but unvalidated.** Modeling per-task representations as diagonal Gaussians is a simplification. The paper provides an MMD variant as a non-parametric alternative, but A-LoRA-M (MMD) often underperforms A-LoRA-K (KL) and in some cases even underperforms standard LoRA (Table 4, Qwen2.5-7B: A-LoRA-M scores 47.53 vs. LoRA's 48.36). This inconsistency is not analyzed.

- **A-LoRA-K uses rank 8 while several baselines use rank 4 in Table 4.** While the % parameter columns show A-LoRA-K uses fewer total trainable parameters (0.20% vs 0.25%), the raw rank is higher, and a single rank-8 adapter has different capacity characteristics than multiple rank-4 adapters. The paper could more explicitly discuss the role of rank structure vs. parameter count.

### Trivial:

- The conclusion uses somewhat absolute language ("calls their fundamental utility into question") where more calibrated claims would better reflect the evidence. The experiments show these architectures may not deliver performance gains *commensurate with their complexity* on the tested benchmarks, which is a meaningful but narrower finding.

## Nice-to-Haves

- Experiments with explicitly conflicting tasks (e.g., sentiment with opposing labels, or tasks from distant domains) to identify the limits of the alignment approach.
- A per-task win/loss analysis for Align-LoRA to reveal whether alignment uniformly helps or trades off task-specific performance on some tasks.
- Deeper ablations isolating the effect of multi-head dropout, router removal, and aggregation strategy separately, to establish the mechanistic claim more rigorously.
- Testing whether alignment on A vs. alignment on B vs. alignment on both matters, since the paper claims A captures task-general features but does not verify this empirically.
- Comparison against simpler regularization baselines (e.g., increased weight decay, standard dropout) to disentangle whether gains come specifically from distribution alignment or from generic regularization.

## Removed Points

- **"Some baselines' results may be taken from prior work, unclear which"** — This is a standard practice when comparing against published methods; the paper follows HydraLoRA's experimental setup and marks results from prior work. Removed as the paper provides sufficient transparency about this.

- **"Missing comparison with MixLoRA, LoRAHub, C-Poly, etc."** — The paper *does* compare with LoRAHub, LoRAMoE, HydraLoRA, and R-LoRA across Tables 2–5. The specific complaint about C-Poly is a missing related work concern, which I should not flag per the rules. Removed.

- **"Parameter budget comparison unfair in rank-scaling experiments"** — The tables clearly show LoRA at rank 8–10 achieves *comparable parameter counts* (0.20–0.25%) to multi-component methods (0.25–0.34%). The comparison is actually parameter-favorable to the baselines (they use more parameters and still lose). This is not an unfair comparison favoring the authors' method — it strengthens the claim. Removed per the hard rule about unfair comparisons that favor baselines.

- **"Reproducibility concerns about undisclosed hyperparameters"** — The paper references detailed experimental settings in Appendix G and provides code. Removed as nitpick about reproducibility of minor details.

- **"Rank scaling only explored to r≈10"** — The paper shows rank scaling from 4 to 10, with multi-component baselines at rank 4. For the 7B and 14B models tested, this is a reasonable range. Removed as a generic weakness that doesn't harm the core claim.

- **"LoRA placement differences could confound results"** — This is speculative and not verified; the paper states it follows HydraLoRA's setup for consistency. Removed as unsubstantiated.

- **"The claim that multi-component architectures 'cannot be merged' is an oversimplification"** — The paper specifically discusses *routed* multi-component architectures, where the dynamic router prevents pre-computation of ΔW. This is factually correct for the methods discussed (R-LoRA, HydraLoRA with routing). Removed as factually inaccurate criticism.

## Novel Insights

The paper's most interesting contribution is the empirical demonstration that, in the multi-task LoRA setting, simple high-rank single adapters are competitive with or superior to elaborate multi-component architectures. This is a valuable "null result" that should restrain the field's tendency toward architectural complexity. However, the mechanistic interpretation — that this is specifically because shared representations dominate — remains under-validated. An alternative explanation consistent with the data is that multi-head/multi-adapter designs with routing introduce optimization difficulties (router learning, load imbalance) that a simple high-rank adapter avoids, rather than shared representations being inherently superior. The paper does not fully disentangle these explanations.

## Suggestions

1. **Add controlled ablations for M-LoRA's success.** The most convincing addition would be: M-LoRA variant without multi-head dropout, and R-LoRA without router but with softmax aggregation. This would isolate whether the performance gap is due to routing specifically, dropout, or aggregation strategy.

2. **Report mean ± std over at least 3 seeds** for the main comparisons, especially Table 1 and Tables 4–5. This would substantially strengthen the evidential basis for the claimed improvements.

3. **Tone down the conceptual claims.** Replace "fundamental contradiction" and "calls into question the fundamental utility" with more calibrated language like "challenges the assumption" and "raises questions about the necessity of." The evidence supports skepticism about multi-component complexity, not wholesale dismissal.

4. **Add per-task analysis.** Report which tasks benefit and which (if any) are hurt by alignment. This would both strengthen the paper and identify the boundary conditions of the approach.

## Score and Decision Calibration

Papers compared for calibration:
- **UnoLoRA** (scores 3,3,3,3 → Rejected): Similar topic (single shared LoRA for multi-task), but UnoLoRA had much weaker novelty, fewer baselines, only GLUE evaluation, and presentation issues.
- **MoRE** (scores 3,3,5,5 → Rejected): Similar topic area (multi-task LoRA), but MoRE lacked novelty, couldn't merge weights during inference, and had marginal improvements.
- **C-Poly** (scores 6,6,6,6 → Accept Poster): Better experimental depth, per-task analysis, clear ablations, and more calibrated claims.
- **ELREA** (scores 6,6,6,6,5 → Accept Poster): Similar topic, strong experiments but with some weaknesses in inference cost.

This paper is stronger than UnoLoRA and MoRE in terms of novelty (challenging a prevailing paradigm), empirical scope (multiple models + BBH generalization), and practical advantage (zero inference overhead). However, it is weaker than C-Poly and ELREA in ablation depth, statistical rigor, and calibration of claims. The overclaiming (especially around the causal mechanism and the theoretical section) is a notable weakness that partially undermines the contribution. The core empirical finding is genuinely valuable, but the paper goes further in its interpretation than the evidence warrants.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>