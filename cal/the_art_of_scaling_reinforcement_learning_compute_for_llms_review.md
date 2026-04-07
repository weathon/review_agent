=== CALIBRATION EXAMPLE 84 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is appropriately evocative. The abstract accurately reflects the core contributions: the sigmoidal framework, the ablation study, and SCALERL. The claim of "the first large-scale systematic study" is made but not fully defended — Vattikonda et al. (2026) and LitePPO (Liu et al., 2025c) are acknowledged later in related work as doing something similar, and the "first" claim is mostly valid by virtue of the compute scale rather than the conceptual framework being entirely new. The abstract also claims SCALERL achieves "near state-of-the-art," but this is only demonstrated against the authors' own reimplementations of baselines, not the original systems.

---

### Introduction & Motivation

**Motivation is strong.** The analogy to pre-training scaling laws is natural and well-articulated. The framing of "art" vs. "science" is a useful rhetorical device. The three principles (universal ceilings, bitter lesson, re-evaluating common wisdom) are clearly stated and genuinely useful.

**A concern with the "bitter lesson" framing.** The paper argues that methods superior at small compute can be worse at large compute (Figure 2), and that the sigmoidal framework helps identify this early. But the methodology for the small-scale forward ablations in Section 3.2 is only 3.5k–4k GPU-hours — exactly the regime that the paper warns can be misleading. There is a tension between "small-scale ablations can identify scalable methods" and "small-scale performance rankings can invert at scale." The paper does not explain precisely when one should trust vs. distrust the small-scale fits.

---

### Predictive Framework (§2.1 and Appendices A.4–A.8)

**Choice of sigmoidal curve.** The justification in Appendix A.4 is compelling: power-law fits overpredict (predicting A=1.0 for the 100k run), while the sigmoid gives A=0.645. The connection to the high-compute equivalence (sigmoid ≈ power-law for C >> C_mid) is a useful sanity check. The grid search procedure for A (0.45 to 0.80 in steps of 0.005) raises a practical question: what happens for methods with A < 0.45 or A > 0.80? Are any baselines clipped by this prior? The paper does not address this boundary case.

**Reliability of fits.** Three independent SCALERL runs yield ±0.015 variation in A, giving an error margin of ±0.02. This is reported as sufficient to distinguish methods. But looking at Figure 5, several LOO variants cluster within 0.01 of SCALERL's A — tighter than the stated margin. When the paper says "all LOO variants reach similar asymptotic reward," this is correct but it also means SCALERL's advantage over some ablations (e.g., w/o zero-variance filtering, w/o No-Positive-Resampling) may not be statistically distinguishable. The paper handles this gracefully by reweighting emphasis to efficiency (B) in LOO analysis, which is honest, but readers should note that many individual components have ambiguous effects on A.

**Predicting from early training.** The paper consistently fits on the first half of a run and extrapolates to the second half. However, it is not analyzed how *early* one can start relying on the sigmoidal fit. In the 100k run, fitting starts at 1.5k GPU-hours — but the paper fits over 1.5k–50k to predict 100k. That is already deep into training. For practical research use (identifying scalable algorithms cheaply), one would want to know: can you reliably predict A from 3k GPU-hours? 5k? 8k? Figure 1 validates one extrapolation point but doesn't systematically study the minimum fitting budget as a function of run length.

---

### Empirical Study of Design Choices (§3)

**Scope.** Most forward ablations are conducted at 3.5k–4k GPU-hours, with the justification that instabilities emerge beyond this scale for many variants. This is understandable but limits confidence: if a method is only stable for 3.5k hours, we are fitting a sigmoid to only a fraction of an epoch's signal. Figure 12 provides some reassurance that stable recipes remain predictable at 16k GPU-hours, but there is a selection bias: only stable variants get extended.

**FP32 precision fix.** This is arguably the most impactful single finding (A jumps from 0.52 to 0.61 on the baseline). The paper correctly credits MiniMax et al. (2025) for the fix, but the broader implication is underexplored: many published RL results (including the DeepSeek/GRPO and Qwen2.5/DAPO baselines in Figure 2) likely suffered from the same numerical issue. This means the comparison in Figure 2 may not reflect what those methods achieve with a fair implementation. The paper acknowledges this indirectly (Figure 4c), but does not discuss whether the community's existing results are systematically biased due to this bug.

**Loss type (CISPO > DAPO >> GRPO).** The finding is convincing and the robustness analysis in Appendix A.18 is thorough. The GSPO instability on larger models (Appendix A.18.4) — diverging mid-training even after checkpoint restarts on Scout — is an important practical warning that the paper reports honestly.

**Advantage normalization.** The result that all three normalization strategies perform similarly (Appendix A.9) is a useful negative result. The paper adopts batch-level normalization as "theoretically sound and marginally better," but "marginally" should be quantified — Figure 10b suggests differences well within noise margins.

**Zero-variance filtering vs. DAPO's dynamic sampling.** The paper conflates these in the main text ("zero-variance filtering differs from dynamic sampling in DAPO"), but the practical distinction is subtle and the paper does not empirically verify that they have genuinely different effects at scale. This should be clarified.

---

### SCALERL Recipe and LOO Experiments (§4)

**LOO design.** The LOO methodology is sound and an appropriate validation step. Crucially, the paper uses a fixed average A across LOO variants and refits B — this is a reasonable approach when A differences are within margin of error. The result that SCALERL consistently achieves higher B (efficiency) is meaningful even if A differences are small.

**Fairness of the LOO interpretation.** The paper states "most LOO variants reach similar asymptotic reward." But examine the table in Figure 5: SCALERL achieves A=0.61 while, e.g., loo-fp32 (without the FP32 fix) also achieves A=0.61. This is *because* LOO uses the full SCALERL stack minus one component — by this stage, CISPO (which already fixes many of the IS-ratio issues that FP32 addresses) may be compensating for the absent FP32 fix. The forward ablations (Section 3.2) show FP32 is critical without CISPO; the LOO shows it is redundant when CISPO is present. This interaction effect is addressed in Appendix A.10 but deserves more prominence in §4.

---

### Scaling Across Axes (§5)

**Model scale (Scout MoE).** The extension to 17B×16 Llama-4 Scout is the most compelling demonstration of transferability. The consistent predictability across model scales is a strong result. The observation that larger models achieve the 8B asymptote using only 1/6 of the RL compute is quantitatively striking.

**Batch size.** The finding that smaller batches stagnate on *downstream* benchmarks even while improving *in-distribution* validation is important and somewhat alarming. It suggests that in-distribution validation metrics (used throughout for scaling curves) can diverge from downstream performance — exactly the setting where the paper's framework is supposed to be predictive. This tension is briefly acknowledged in §7 but deserves more investigation, as it implies that choosing configurations based solely on sigmoidal fits of validation pass-rate may not optimize the quantities practitioners care about.

**Generations per prompt.** The finding of "essentially unchanged scaling curves" when sweeping 8–32 generations per fixed batch is a useful negative result. The claim that "clearer differences may emerge at much larger batches" is speculative and left for future work — this is an appropriate acknowledgment.

**Multi-task RL.** Relegated entirely to the appendix (Figure 16). The math+code result is promising, but the code curve (B=1.09) has dramatically lower efficiency than the math curve (B=2.05). The paper does not investigate why, or whether SCALERL's design choices are equally well-suited to code tasks.

---

### Comparison with Existing Recipes (Figure 2)

This figure is central to SCALERL's positioning yet methodologically fragile in several ways:

1. **Reimplementation fidelity.** The authors implement their own versions of DeepSeek/GRPO, DAPO, Magistral, and MiniMax. The Appendix A.17 descriptions show at least one meaningful deviation for DAPO (batch size 1280 with dropped zero-variance prompts rather than dynamic resizing), even if the authors argue this gives DAPO an advantage.

2. **MiniMax-M1 gap.** MiniMax-M1 in Figure 2 already uses CISPO and FP32 — two of SCALERL's core ingredients — yet achieves lower asymptotic performance (A≈0.54 vs SCALERL's A≈0.61). The paper attributes the difference to PipelineRL vs. PPO-off-policy and starting state, but does not isolate this clearly. A more careful ablation would help.

3. **Starting point heterogeneity.** Different recipes start from different SFT checkpoints. Since A is measured as an absolute pass rate, differences in the initial policy's capability (R₀) could confound comparisons of the long-run asymptote.

---

### Related Work (§6 and Appendix A.1)

Coverage is good for a fast-moving area. The paper is honest about the relationship with LitePPO (Liu et al., 2025c) and ProRL (Liu et al., 2025a). The distinction between the paper's focus (compute-scaling prediction) and those works' focus (algorithm comparison at fixed compute) is clearly drawn. The absence of discussion on RLHF scaling work beyond verifiable rewards (e.g., human feedback-based RL for chat models) is a reasonable scope limitation.

---

### Limitations & Discussion (§7)

The paper is admirably self-aware in §7:
- It correctly identifies that in-distribution validation may not fully predict downstream generalization.
- It highlights that the full 3-axis scaling law (model size × pre-training compute × RL compute) is left for future work.
- It notes that multi-task scaling is preliminary.

**Under-acknowledged limitations:**
1. The framework is validated only on verifiable-reward math/code tasks. Extension to subjective or open-ended tasks (where reward is noisier) is not discussed and would likely break the sigmoidal framework's regularity.
2. The 8B base model is specific; the SFT initialization is not publicly released ("curated data mix of reasoning traces"). Reproducibility depends on the base model, which practitioners may not have.
3. The paper does not discuss what happens when the training data is exhausted (multiple epochs over Polaris-53K). The no-positive-resampling component permanently removes problems — this changes the effective data distribution over epochs and could interact non-trivially with the sigmoidal model.

---

### Overall Assessment

This paper makes a genuine and timely contribution: it brings scaling-laws methodology to RL post-training for LLMs and backs it up with a massive empirical investment (400k+ GPU-hours) that is out of reach for most academic groups. The sigmoidal compute-performance framework is well-motivated, and the key findings — CISPO > DAPO for stability and asymptotic performance, FP32 IS-ratio fix as a hidden performance bottleneck, PipelineRL for efficiency, and the counter-intuitive inversion of efficiency-vs-asymptote tradeoffs — are practically important for the community. The 100k GPU-hour demonstration is a credible stress-test of the predictive framework.

That said, several concerns limit the paper's strength at ICLR. First, SCALERL is explicitly an integration of existing techniques rather than an algorithmic advance; the contribution is methodology and validation at scale. Second, the central framework (fit a sigmoid to held-out validation accuracy) is conceptually simple, and the "prediction" itself is somewhat conservative (fitting over 1.5k–50k to predict 100k when the curve is already visibly saturating at 50k). Third, the ablation regime for many design choices (3.5k–4k GPU-hours) conflicts with the paper's own warning that small-scale rankings can invert at scale, creating a methodological circularity that the paper does not fully resolve. Fourth, the observed divergence between in-distribution validation and downstream benchmarks (particularly for batch size) introduces doubts about whether the sigmoidal framework captures what practitioners need. These are real issues but they do not negate the paper's value — a resource of this scale, done with this rigor, merits publication if the community is willing to accept its empirical contribution as primary. For ICLR specifically, stronger justification of the framework's generality beyond math/code verifiable rewards, and a cleaner reconciliation of the small-scale vs. large-scale prediction tension, would significantly strengthen acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents the first large-scale systematic study of RL compute scaling for LLMs, utilizing over 400,000 GPU-hours to define a sigmoidal compute-performance framework and introduce the SCALERL recipe. The work demonstrates that RL training trajectories can be predicted via extrapolation from lower-compute runs, identifying key design choices that affect both asymptotic performance and compute efficiency. SCALERL achieves state-of-the-art scaling stability and performance, validating the framework across model sizes up to 17B MoE and compute budgets of 100,000 GPU-hours.

### Strengths
1.  **Unprecedented Scale:** The study's magnitude (400k+ GPU hours, including a 100k GPU-hour run) is significantly larger than typical RL scaling studies, providing high-confidence empirical evidence that is rare in the RL domain (Abstract, Figure 1).
2.  **Predictive Framework:** The proposal to model bounded RL performance using sigmoidal curves (Equation 1) rather than power laws is mathematically well-justified for metrics like pass rate and validated through successful extrapolation against extended training runs (Appendix A.4, Figure 1).
3.  **Systematic Ablation:** The Leave-One-Out (LOO) experiments (Section 4, Figure 5) rigorously isolate the impact of individual design choices (e.g., PipelineRL, FP32 precision, loss type), moving beyond the ad-hoc recipes often found in industry technical reports.

### Weaknesses
1.  **Domain Limitation:** The evaluation is heavily concentrated on math/reasoning tasks (Polaris-53k, AIME), with limited investigation into code, chat, or multi-turn RL (Section 5, Appendix A.15), raising questions about generalizability to broader LLM applications.
2.  **Model Scale Scope:** While 17B MoE is used, the "Scaling" narrative relies heavily on compute scaling within the 8B/17B range; comparisons with larger frontier models (e.g., 70B+) would strengthen the claim of universal scaling laws (Section 5).
3.  **Validation Dependency:** Predictability is primarily measured against a held-out validation set from the training distribution. While downstream correlations (AIME) are shown, the primary metric relies on in-distribution performance extrapolation, which may not always capture generalization failure modes in new contexts (Section 1, Section 7).

### Novelty & Significance
**Novelty:** The paper's primary novelty lies in establishing a *methodology* for RL scaling analysis analogous to pre-training scaling laws, rather than just proposing a new algorithm. The systematic application of sigmoidal curves to RL post-training is a novel empirical contribution compared to the existing "black box" nature of RL engineering.
**Significance:** High. Providing a principled way to evaluate RL candidate improvements without incurring full compute costs is critical for the community, especially for academic groups without massive budgets. It bridges the gap between ad-hoc RL tuning and the rigorous scaling laws of pre-training.

### Suggestions for Improvement
1.  **Expand Domain Generalization:** Include experiments on diverse domains (e.g., code generation, instruction following, tool use) to verify if the SCALERL components and scaling laws hold beyond reasoning tasks.
2.  **Broader Model Scaling:** Include model scaling dimensions (e.g., 1B, 7B, 70B) to verify if the compute curves collapse across model parameters, not just fixed model size, to establish true multi-dimensional scaling laws.
3.  **Generalization Analysis:** Provide more quantitative analysis on the correlation between validation pass rates and out-of-distribution benchmarks across different *epochs*, not just final asymptotes, to better understand scaling vs. overfitting dynamics.
4.  **Recipe Complexity Discussion:** Given the high number of components in SCALERL (Appendix A.18), discuss the trade-off between the complexity of the recipe and the marginal gains, helping practitioners decide when SCALERL is necessary versus a simpler baseline.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Fit scaling curves directly on downstream benchmarks (AIME, LiveCode) rather than just IID validation to verify that IID predictability transfers to OOD capabilities. Without this, the claim that scaling laws predict "capabilities" is unsupported.
2. Run baseline methods (GRPO, DAPO) with equivalent hyperparameter tuning budgets to ensure their reported instability is not due to suboptimal configuration compared to SCALERL. Otherwise, the performance gap may reflect tuning effort rather than algorithmic superiority.
3. Train until single-epoch completion to distinguish compute saturation from data exhaustion. The current multi-epoch setup conflates compute scaling with data reuse, undermining the validity of the compute-performance law.

### Deeper Analysis Needed (top 3-5 only)
1. Re-evaluate scaling curves using tokens processed or FLOPs instead of GPU hours to ensure the law is hardware-agnostic. GPU hours depend on cluster efficiency, making the "scaling law" non-portable to other environments.
2. Provide confidence intervals for the asymptotic parameter $A$ to confirm SCALERL's gain over baselines is statistically significant rather than within noise variance. Current error margins ($\pm 0.02$) overlap with some baseline differences, weakening the SOTA claim.
3. Analyze sensitivity of scaling parameters to off-policy staleness ($k$) to define stability boundaries. Asynchronous RL stability depends heavily on this delay, yet the robustness of the scaling law to $k$ is not quantified.

### Visualizations & Case Studies
1. Plot residuals of the sigmoid fits across compute regimes to demonstrate no systematic bias in the low or high-compute regions. This is necessary to prove the sigmoid form is not artificially forcing a fit.
2. Visualize baseline divergence points (loss spikes, truncation rates) to substantiate claims that other methods are inherently unstable. Without traces, instability claims look like hyperparameter mismatches.
3. Compare scaling parameters ($A, B$) across distinct task domains (Math vs. Code) side-by-side to test universality. Current results show parallel trends but lack direct parameter comparison to confirm the law holds across domains.

### Obvious Next Steps
1. Validate scaling laws on non-reasoning tasks (e.g., RLHF, instruction following) to test if the sigmoidal relationship generalizes beyond verifiable math. The current scope is too narrow to claim a general science of RL scaling.
2. Explicitly model the data-constrained regime where performance plateaus due to prompt exhaustion rather than compute limits. This is critical for practitioners to know when to stop training.
3. Include the curve-fitting code in the supplementary material immediately rather than promising post-acceptance release. Reproducibility of the scaling law fits is essential for verifying the core scientific contribution.

# Final Consolidated Review
## Summary

This paper presents the first large-scale systematic study of RL compute scaling for LLMs, using over 400,000 GPU-hours to establish a sigmoidal compute-performance framework. The authors show that RL training follows predictable sigmoidal curves that enable extrapolation from smaller runs, and propose SCALERL—a recipe combining PipelineRL, FP32 precision, CISPO loss, and other design choices—that achieves predictable scaling up to 100,000 GPU-hours and outperforms reimplementations of existing recipes like GRPO, DAPO, and MiniMax.

## Strengths

- **Unprecedented empirical scale.** The study's magnitude (400k+ GPU-hours, including a 100k GPU-hour validation run) substantially exceeds prior RL scaling work and provides high-confidence evidence that would be infeasible for most academic groups. The leave-one-out ablation methodology at 16k GPU-hours per run is appropriately thorough for validating component contributions.

- **Sigmoidal framework is well-motivated.** The choice of sigmoidal curves over power laws is justified both theoretically (bounded metrics saturate rather than diverge) and empirically (power-law fits overpredict asymptotes; Appendix A.4 shows power-law predicts A=1.0 for the 100k run while sigmoidal correctly predicts A≈0.645). The high-compute equivalence to power-law (C >> C_mid) provides useful conceptual grounding.

- **Key practical findings.** The FP32 precision fix at the LM head (credited to MiniMax et al.) is shown to dramatically improve asymptotic performance (A: 0.52 → 0.61), addressing a widespread implementation issue. The PipelineRL efficiency gain over PPO-off-policy is meaningful and attributed to tighter on-policy alignment rather than just throughput.

- **Honest handling of measurement uncertainty.** The ±0.02 error margin for A, established via three independent SCALERL runs, enables principled comparison. When LOO variants fall within this margin, the paper appropriately shifts emphasis to efficiency (B) rather than overstating asymptotic differences.

- **Transferability across model scales.** The 17B×16 MoE (Llama-4 Scout) experiments demonstrate that SCALERL's predictability transfers across model architectures, with the larger model achieving the 8B asymptote at 1/6 the compute—a quantitatively striking result.

## Weaknesses

- **Baseline comparisons rely on reimplementations, not original implementations.** Figure 2 compares SCALERL against authors' reimplementations of GRPO, DAPO, Magistral, and MiniMax. While Appendix A.17 describes implementation details, including a larger batch size for DAPO (1280 vs. 768) that should advantage the baseline, subtle implementation differences could affect conclusions. The paper attributes the MiniMax gap primarily to PipelineRL vs. PPO-off-policy, but this is not cleanly isolated. Direct comparisons with original published baselines or officially released training curves would strengthen confidence.

- **Divergence between validation and downstream metrics.** Section 5 notes that smaller batch sizes show continued improvement on in-distribution validation while stagnating on downstream benchmarks—a pattern observed but not deeply investigated. This raises concerns about whether the sigmoidal framework's validation-based predictions fully capture the capabilities practitioners care about. The paper acknowledges this limitation in §7, but the practical implications deserve more prominence.

- **Scope limited to verifiable-reward reasoning tasks.** All experiments use math (Polaris-53K) and code (Deepcoder) datasets with deterministic reward signals. Whether the sigmoidal regularity holds for RLHF, instruction following, or multi-turn interactions—where rewards are noisier and more subjective—is not explored. The framework's generality remains an open question.

- **Small-scale ablation methodology conflicts with "bitter lesson" warning.** The forward ablations in §3.2 run at 3.5k–4k GPU-hours—the regime the paper warns can produce misleading rankings (small-compute leaders can invert at scale). The paper notes that unstable variants cannot be extended beyond this, creating selection bias. While LOO experiments at 16k GPU-hours provide reassurance for SCALERL components, this circularity is not fully resolved.

- **Grid search bounds on asymptote parameter A (0.45–0.80) are not discussed.** Methods with asymptotes outside this range would be clipped. While no baseline appears to hit these boundaries, the constraint deserves explicit acknowledgment for practitioners applying the framework to new domains.

- **Data exhaustion dynamics are unexplored.** The no-positive-resampling component permanently removes prompts, changing the data distribution across epochs. How this interacts with the sigmoidal model—whether performance plateaus from compute saturation or data depletion—is not analyzed. Multi-epoch training conflates compute scaling with data reuse in ways the framework does not disentangle.

## Nice-to-Haves

- **Downstream benchmark extrapolation.** Fitting scaling curves directly on AIME or LiveCodeBench (rather than IID validation) would test whether predictability transfers to out-of-distribution capabilities—critical for claims about scaling "performance" broadly.

- **FLOPs or tokens as the compute unit.** GPU-hours depend on cluster configuration and efficiency; normalizing by tokens processed or FLOPs would make the scaling law more portable across hardware setups.

- **Comparison on larger model scales.** While the 17B MoE result is compelling, showing that curves collapse or translate predictably across a wider model size range (e.g., 1B–70B) would strengthen claims of universal scaling laws.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Methods achieving similar A have ambiguous individual effects."** While true that some LOO variants cluster within the ±0.02 margin, the paper correctly handles this by shifting to efficiency (B) comparisons and noting cumulative effects. This is appropriate methodology, not a weakness.

- **"DAPO comparison is unfair due to batch size."** The paper explicitly states in Appendix A.17 that DAPO was given a larger batch (1280) as an advantage—the comparison actually favors the baseline, making this criticism unfounded.

- **"FP32 fix may not matter when CISPO is present."** Appendix A.10 addresses this: FP32 still helps with other losses and shows gains on Scout MoE. The interaction is documented.

- **"Multiple epochs conflate compute scaling with data reuse."** This is partially valid (moved to weaknesses above) but the formulation here was overstated—the sigmoidal model fits observed compute vs. performance regardless of underlying mechanism.

- **"Code domain shows lower efficiency (B=1.09) without explanation."** Figure 16 shows parallel scaling trends for math and code; the B difference is visible but not anomalous. A fuller investigation would help but is not a critical flaw.

- **"Starting checkpoints differ across recipes."** While true, this is a limitation of comparing training reports, not something the paper can control. All SCALERL variants use consistent starting points; the baseline reimplementations follow published specifications.

- **"Confidence intervals for asymptote A needed for statistical significance."** The ±0.02 error margin is already established via multiple runs. Demanding additional confidence intervals on top of this is scope creep for an empirical systems paper.

## Novel Insights

Beyond the paper's contributions, three insights emerge: (1) The separation between asymptotic performance (A) and compute efficiency (B) reveals that many RL design choices are efficiency optimizations, not capability improvements—this reframes how practitioners should prioritize interventions. (2) The FP32 precision fix highlights that numerical bugs in importance sampling ratios can silently cap RL performance, suggesting that many published results may be systematically biased downward. (3) The tension between IID validation predictability and downstream stagnation (small batches) identifies a fundamental challenge for scaling-law-based development: optimizing for easily-measured proxies may not optimize for the capabilities that matter.

## Suggestions

- Add a brief analysis of how early the sigmoid fit becomes reliable (e.g., minimum GPU-hours to predict 100k with ±0.02 accuracy) to operationalize the framework for budget-constrained researchers.

- Provide residuals or goodness-of-fit metrics for sigmoid fits across compute regimes to demonstrate no systematic bias in early vs. late regions.

- Clarify in the main text (not just appendix) the interaction between CISPO and FP32—that FP32's importance is method-dependent—to prevent practitioners from applying unnecessary fixes or skipping necessary ones.

- Release the curve-fitting code immediately (even as a supplement) to enable reproducibility of the core scientific contribution; the current promise of post-acceptance release limits verifiability.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
