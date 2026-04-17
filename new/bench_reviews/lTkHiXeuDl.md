Now I have enough calibration data. Let me consolidate all the reviews and write my final meta-review.

## Summary

HMoRA proposes a hierarchical mixture-of-LoRA-experts fine-tuning method for LLMs that combines token-level and task-level routing in a layer-dependent manner (shallow layers → token routing, deep layers → task routing), introduces a Constrained Generalized Jensen-Shannon (CGJS) auxiliary loss to improve routing certainty and expert balance, and provides optional lightweight designs to reduce parameters and computation. The method is evaluated on Qwen2-1.5B fine-tuned on Flan v2, where it modestly outperforms full fine-tuning and several MoE-LoRA baselines on multiple NLP benchmarks while training 3.9% of parameters.

## Strengths

- **Well-motivated hierarchical design**: The hierarchical hybrid routing mechanism (token-level in shallow layers, task-level in deep layers) is grounded in the established observation that LLM layers encode information at different granularities (Geva et al., 2021). This is a sensible architectural choice. The ablation in Appendix E.5 supports that increasing α^(l) with depth helps.

- **CGJS auxiliary loss is a meaningful contribution**: The proposed CGJS loss (Eq. 11) addresses a real and documented problem—standard load balancing loss trades off routing certainty for balance. The entropy analysis in Figure 3 provides direct empirical evidence that CGJS achieves both higher certainty (lower per-sample entropy) and better balance (higher average entropy) compared to standard load balancing loss or no auxiliary loss, and Table 1 shows consistent accuracy gains.

- **Strong parameter efficiency**: HMoRA with lightweight designs achieves competitive or better-than-full-FT performance using only 3.9% trainable parameters (Table 2), which is a practically useful result.

- **Comprehensive routing analysis**: The per-layer entropy visualizations (Figure 3) and t-SNE routing analyses (Figure 4) provide useful diagnostic tools rarely seen in MoE+PEFT papers.

## Weaknesses

### Major:

- **The "clustering-like" and "unsupervised task differentiation / unseen-task generalization" claims are overstated relative to what the CGJS loss actually does.** The loss (Eq. 11-12) constrains per-batch statistics: it pushes the average gate distribution toward high entropy (balance) and each individual distribution toward low entropy (certainty). There is no explicit term encouraging same-task samples to cluster or different-task samples to separate. The claim in Sec. 3.3 that this "essentially performs a clustering-like effect" is not justified by the form of the objective—it depends on the task mixture within each batch, which is not analyzed. The t-SNE visualizations (Fig. 4) are suggestive but not quantitative, and the "42/57 tasks differentiated" metric (Appendix E.8) lacks specification of the threshold used. Additionally, many architectural changes are introduced simultaneously (task encoder, task embedding, hierarchical routing), making it impossible to isolate whether CGJS itself drives the task differentiation. Given that these claims are central to the paper's conceptual contribution, this is a significant gap.

- **Performance margins over baselines are modest and reported without variance.** HMoRA w/ LW averages 63.88 vs. Full FT's 63.15 (+0.73) and MoLoRA's 63.02 (+0.86). Individual benchmark gains are often <1 point (e.g., MMLU: +0.51, ARC-E: +0.77). Though the paper repeats each experiment 5 times, only means are reported—no standard deviations or confidence intervals. Without variance information, it is unclear whether these improvements are statistically meaningful. This is particularly concerning because the paper's headline claims ("outperforms full fine-tuning across multiple NLP benchmarks") rest on these small margins.

- **Key ablation missing for hierarchical routing's contribution.** The paper does not compare hierarchical hybrid routing (α varying by layer) against a flat hybrid routing (uniform α across all layers) or against task-level-only routing across all layers with the same auxiliary loss and task encoder. Without this, it is unclear whether the hierarchical scheduling of α is a meaningful ingredient or whether simply "adding a task router somewhere" accounts for the gains. The ablation in Appendix E.5 only varies ε and μ, not the fundamental architectural choice of hierarchy vs. flat.

### Minor:

- **Limited model scale and task diversity**: All main results use Qwen2-1.5B; LLaMA 3.2 1B results are relegated to Appendix E.7 without detailed discussion. All evaluation benchmarks are multiple-choice QA tasks. It remains unclear whether hierarchical routing insights and CGJS gains transfer to larger models (7B+) or to generative tasks.

- **Inference overhead not analyzed**: The TaskEncoder is a separate Transformer that processes the full input sequence, and the MoE routing with multiple LoRA experts cannot be merged back into the base weights (unlike standard LoRA). The paper does not analyze inference latency or FLOPs, which matters for assessing the practical efficiency of this "parameter-efficient" method.

- **Ad-hoc interpolation formula**: The α^(l) schedule (Eq. 8) uses a sigmoid with two hyperparameters (ε, μ) that is hand-designed. It would strengthen the paper to compare against simpler alternatives (e.g., linear α^(l) = l/L) to show the extra complexity is warranted.

### Trivial:

- The task embedding is initialized with a question mark symbol embedding—this is a minor detail unlikely to affect results but lacks justification.

## Nice-to-Haves

- Report standard deviations alongside means for all benchmark results so readers can assess statistical significance.
- Add a comparison against LoRA with a matched parameter budget (e.g., higher rank that matches HMoRA's 3.9-6.3% trainable params) to disentangle whether gains come from the MoE structure or simply from more parameters.
- Test on at least one larger model (7B+) to assess scalability.
- Evaluate on at least one generative benchmark beyond multiple-choice QA.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Multiple parallel LoRA experts may be equivalent to a single higher-rank LoRA"**: While this is a valid concern raised by the Human Finder from MoLoRA reviews, the paper *does* include LoRA r=64 (4.78% params) as a baseline in Table 2, and HMoRA w/ LW with 3.9% params outperforms it. HMoRA w/o LW at 6.31% has more params than LoRA r=64, but the comparison is still informative. The criticism about "unfair" parameter-count comparison is weakened because HMoRA w/ LW actually uses *fewer* parameters than LoRA r=64 and still wins.

- **"Hyperparameter sensitivity and practical tuning difficulty"**: This is a generic concern applicable to any MoE method. The paper provides ablation studies on the key hyperparameters and finds that performance is not highly sensitive to ε, μ, and γ_c. While more hyperparameter analysis is always better, this doesn't rise to the level of a weakness specific to this paper.

- **"Motivation for many experts during fine-tuning needs further justification"**: This is a generic philosophical concern about MoE-LoRA methods, not a specific weakness of this paper. The paper trains on Flan v2, which has 1,836 tasks—substantial heterogeneity that plausibly justifies 8 experts.

## Novel Insights

The paper's observation that standard load balancing loss and routing certainty are in tension—and that a constrained entropy-based objective can address both simultaneously—is genuinely insightful. However, the leap from "high certainty + balanced load" to "unsupervised task clustering" is not rigorously supported. The routing entropy analysis (Fig. 3) could become a more widely adopted diagnostic in MoE work: plotting per-layer certainty and balance metrics across training clearly reveals pathologies (e.g., near-uniform routing in shallow layers, expert collapse in deep layers) that are typically invisible.

## Suggestions

- **Tone down the clustering/generalization claims**: Replace "essentially performs a clustering-like effect" with "may encourage clustering-like behavior" and qualify the unseen-task generalization claims with acknowledgment that the evidence is based on MMLU tasks that may share distributional overlap with Flan v2.
- **Add a flat routing ablation**: Run HMoRA with uniform α (e.g., α=0.5 at all layers) under identical settings; this is the single most important missing comparison for validating the hierarchical design.
- **Report standard deviations** for Table 2 results to allow readers to assess significance of the modest margins.

## Score and Decision

Calibration against similar papers:
- **MoLoRA/MoV** (Accept poster, scores 6,5,8,8): Directly comparable MoE+LoRA method. Got accepted with comparable strengths (empirical gains over baselines) and similar weaknesses (no variance reporting, limited model scale, inference overhead unanalyzed). HMoRA adds more architectural complexity (hierarchical routing, CGJS loss, task encoder) but its gains are modest.
- **MoRE** (Reject, scores 3,3,5,5): MoE+LoRA method with marginal improvements, rejected for insufficient novelty, marginal gains, and lack of significance testing.
- **MoLE/MoLE** (Accept poster, scores 3,5,6,6): Marginal accept for LoRA fusion with MoE-style gating; had marginal improvements but was accepted partly due to simplicity.
- **MoLEx** (Accept poster, scores 5,6,8): Layer-expert PEFT with moderate scores.

HMoRA is comparable to MoLoRA/MoV in overall quality: it has a meaningful auxiliary loss contribution (CGJS), reasonable empirical results, but overclaims on clustering/generalization and has modest margins without variance. It is somewhat stronger than MoRE but its overclaiming on the theoretical/mechanistic side is a real issue. Given the calibration, I place it below MoLoRA (which had a cleaner story) but above MoRE.

**Score: 5.5**

HMoRA is a reasonable engineering contribution with a useful auxiliary loss and sensible hierarchical routing, but the paper overclaims on the clustering/generalization narrative without sufficient evidence, and the empirical gains are modest and reported without variance. The missing flat-routing ablation also undermines confidence in the hierarchical design's contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>