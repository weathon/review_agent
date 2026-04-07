=== CALIBRATION EXAMPLE 88 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is accurate but slightly misleading: "Small Transformers Don't Need LayerNorm at Inference Time" implies a theoretical insight, but the finding is empirical and practically limited to the GPT-2 family. The subtitle ("Scaling LN Removal to GPT-2 XL") frames GPT-2 XL (1.5B parameters) as a "scaling" result, which overstates the magnitude — this is a modest scale by 2025/26 standards. The abstract's claim that "the amount of fine-tuning data needed grows sublinearly" is based on only four data points (Small, Medium, Large, XL), making the extrapolative inference that "scaling to larger models is feasible" stronger than the evidence supports.

The abstract's mention of "a small increase in validation loss (+0.03 cross-entropy loss for GPT-2 XL)" refers specifically to The Pile-filtered, not The Pile, where GPT-2 XL LN-free achieves a CE of 130.22 vs. 2.44 for the original — an extreme outlier that is buried in footnotes in the main text. This selective presentation in the abstract is misleading.

---

### Introduction & Motivation

The motivation is solid. LN's nonlinearity genuinely complicates mechanistic interpretability by coupling components and invalidating linear attribution methods. The framing is well-grounded in prior work (Elhage et al., 2021; Wang et al., 2022b; nostalgebraist, 2020). The three contributions are clearly stated.

However, the introduction does not adequately discuss **what interpretability findings on LN-free models can actually teach us about models with LN**. The authors acknowledge that "similarity is not exact" but this is underqualified. If removing LN changes confidence behavior, attention sink patterns, and residual stream norms, the models may have learned qualitatively different internal representations. This limits the claim that LN-free GPT-2 models serve as transparent proxies for original GPT-2 models, and this limitation deserves sharper treatment upfront.

The precursor workshop paper (Heimersheim, 2024) is appropriately cited. The present paper extends to all GPT-2 sizes with a new auxiliary loss, sublinear scaling analysis, and mechanistic interpretability experiments — a meaningful upgrade, though readers should note that the core idea is not new to this paper.

---

### Method (Section 3: LN Removal Strategy)

**FakeLN design.** Replacing the dynamic standard deviation with a fixed scalar $\hat{\sigma}_{avg}$ is conceptually simple and practically motivated. The use of an exponential moving average for larger models to estimate this scalar is sensible. The decision to split LN into four distinct types (LN$^l_\text{qk}$, LN$^l_v$, LN$^l_\text{MLP}$, LN$^f$) and remove them sequentially is justified and demonstrated empirically to improve stability.

**Auxiliary loss.** The auxiliary loss (Equation 3) penalizes variance in token-position standard deviations. Excluding position 0 and EOS tokens from the target (not from the loss itself) is a specific design choice that merits more justification. The claim that "such positions consistently exhibit higher variance in GPT-2 models" is asserted but not shown quantitatively. Additionally, the auxiliary loss is applied only at the final LN layer — but the norm-consistency objective is motivated by all layers' instability during LN removal. Why not apply it at intermediate layers?

**Ablations are insufficient.** The paper lacks systematic ablation experiments:
- What is the performance cost if all LN blocks are removed simultaneously (briefly mentioned as "irreparably breaking the model," but no quantification is given)?
- What is the incremental effect of the auxiliary loss alone vs. sequential removal alone? The paper states "the auxiliary loss effectively absorbs some of the effects of LN removal" without a controlled comparison.
- What is the effect of MLP-first vs. QK-first removal ordering? Only qualitative reasoning is provided ("residual norm variance at beginning-of-sequence tokens affects the attention mechanism more").

**Reproducibility.** Appendix B provides hyperparameters and schedules, which is commendable. However, Appendix B.1 acknowledges substantial failure modes and the need for significant hyperparameter tuning for Large and XL models. The procedure is notably fragile — "irrecoverably high loss" failures are mentioned without statistics on success rates across random seeds. Readers hoping to apply this method will find it more difficult than the smooth loss curves in Figure 1 suggest.

---

### Results (Section 4: LN Removal Results)

**Table 1 and the outlier problem.** The GPT-2 XL LN-free model's mean CE loss of 130.22 on The Pile is alarming. The paper explains this as three samples with OOD tokens causing residual norm explosions. The explanation is plausible, but it raises an unresolved question: why does this catastrophic failure mode appear only in GPT-2 XL and not in other sizes? The paper states "we observed this phenomenon only in GPT-2 XL" without mechanistic explanation. This is not simply a distribution-shift issue, since the Pile-filtered metric carefully controls for this and GPT-2 XL LN-free performs fine there (+0.025 CE). A deeper diagnosis is warranted, especially if practitioners try to use GPT-2 XL LN-free on data with uncommon tokens.

**Persistent performance gap.** The finding that extended fine-tuning does not close the loss gap (Section 4) is important and interesting — it implies LN provides a small but irreducible functional benefit. However, the authors do not deeply investigate whether this gap is due to (a) the irreversible information loss during training-time normalization, (b) suboptimal FakeLN initialization, (c) insufficient fine-tuning, or (d) a fundamentally inexpressible function without normalization. This is mentioned as future work but feels like a key unanswered question given the paper's central claim.

**Sublinear scaling claim (Appendix B.2).** The argument rests on two data points: GPT-2 Small (300 steps × 524k tokens = 157M tokens) and GPT-2 XL (800 steps × 516k tokens = 413M tokens). A 12× parameter increase requiring 2.6× more tokens is indeed sublinear, but four data points (Small, Medium, Large, XL) is a very thin basis for claiming a scaling law, particularly when hyperparameter tuning costs are not included in the analysis. The protocol required "significant hyperparameter tuning" for Large and XL; if those tuning runs are counted, the effective data requirement is much larger.

**Benchmark results (Appendix F).** LN-free models maintain within 1-2 percentage points on BoolQ, HellaSwag, PIQA, and WinoGrande. However, these benchmarks are simple and may not be sensitive to the kinds of capability differences (e.g., calibration, distributional reasoning) that matter for mechanistic interpretability. The overconfidence observed in LN-free models (ECE nearly doubles for GPT-2 Medium) is not captured by accuracy-based benchmarks.

---

### Mechanistic Interpretability Analyses (Section 5)

**Section 5.1 (DLA becomes exact).** This is the paper's cleanest result. The NMAE dropping from ~49% to 0.00% for LN-free GPT-2 Small is mathematically expected — without the normalization nonlinearity, DLA and DE are algebraically equivalent — but empirically confirming it and quantifying the magnitude of the prior approximation error (~50%) is valuable. The 95% CI for the original model is wide [29.92%, 66.10%], indicating high per-head variability, which motivates further investigation. The choice of NMAE formulation (averaging absolute differences rather than per-sample ratios, as footnote 5 explains) is non-standard and the justification given ("we did not observe a consistent proportional relationship") should be more carefully motivated.

**Section 5.2 (Attribution patching does not improve).** This is the paper's most important interpretability finding. The result that removing LN does not improve attribution patching accuracy challenges a widely-held belief in the community (Nanda, 2023a explicitly called LN "a particularly thorny nonlinearity"). The finding is honest and well-presented. However, the evaluation is limited to a single task (IOI), a single model size (GPT-2 Small), and a single patching location (residual stream). The conclusion that "other nonlinearities (attention SoftMax or MLP activations) are the primary bottleneck" is suggestive but not empirically demonstrated. A simple comparison of attribution patching error with and without attention softmax or GELU nonlinearities — even in a toy model — would substantially strengthen this claim.

**Section 5.3 (First-position tokens are no longer special).** The finding that LN-free models have more uniform token norms is interesting and expected. The attention sink rate drops from 55.3% to 45.3% in LN-free models — a notable reduction but not elimination. The observation that "the relationship between relative token norm magnitudes and attention sink behavior is likely complex" is understated; the paper does not investigate whether this partial reduction in attention sinks affects interpretability analyses (e.g., circuit-finding methods that typically exclude attention to position 0).

**Section 5.4 (Confidence neurons are neutered).** This is a satisfying confirmation of Stolfo et al.'s (2024) proposed mechanism. The finding that confidence neurons maintain their structural signature (high weight norm, low logit variance) but lose functional impact is nuanced. The observation that vanilla fine-tuned models also show reduced confidence neuron effectiveness is worth highlighting more prominently — it suggests that the fine-tuning process itself partially disrupts confidence regulation, independent of LN removal. This confound complicates attributing all changes to LN removal specifically. Appendix H shows careful additional analysis (SVD of unembedding matrix, cumulative ablation) that is commendable.

---

### Discussion & Limitations (Section 6)

The limitations section is honest about instabilities, hyperparameter sensitivity, and overconfidence. The paper acknowledges that LN-free models "are not easily quantizable," which is a minor but practical point.

A key limitation that is not adequately discussed: **the fidelity of LN-free models as proxies for original models**. The paper repeatedly justifies LN-free models as enabling more precise interpretability research, but the fact that (a) overconfidence increases substantially, (b) attention sink behavior changes, (c) confidence neurons are disabled, and (d) a persistent performance gap exists suggests that the internal computation may have reorganized during fine-tuning in ways that are not well-understood. Any circuit-finding or component-attribution study on LN-free models would need to carefully validate that the circuits found transfer to the original model. This is mentioned only briefly: "our fine-tuned models have similar internals, but should be used with caution as similarity is not exact."

---

### Writing & Clarity

The main body is well-organized and clearly written. The division of the interpretability section into distinct phenomena (DLA, attribution patching, attention sinks, confidence neurons) is logical and easy to follow. Figure 1 (training loss curves) effectively summarizes the LN removal procedure.

Appendix E.1 (GPT-2 XL high-loss samples on The Pile) dumps raw per-token CE losses for three samples at considerable length without adding much insight. This content could be summarized more concisely.

---

### Overall Assessment

This paper provides a genuine and reproducible contribution: LN-free analogs of the full GPT-2 family, released on Hugging Face, with a carefully engineered fine-tuning procedure. The demonstration that DLA becomes exactly equal to DE (rather than an approximation) in LN-free models is clean, and the negative result on attribution patching is scientifically honest and informative. However, the paper has several notable weaknesses relative to ICLR's standards. The scope is limited to GPT-2-class models (increasingly dated for interpretability research), and the claim of "scalability" extrapolates from only four data points while acknowledging significant hyperparameter sensitivity. The GPT-2 XL outlier failure (CE = 130.22 on The Pile from 3 samples) is inadequately explained. The persistent performance gap and substantial increase in overconfidence are acknowledged but not resolved, raising questions about when LN-free models are trustworthy proxies for their LN counterparts — the paper's core justification. The most interesting mechanistic finding (attribution patching doesn't improve) lacks follow-up experiments to isolate which nonlinearity is actually responsible. As a resource paper for the mechanistic interpretability community, this work has clear value; as a research contribution meeting ICLR's innovation bar, it is borderline. Acceptance would hinge on whether reviewers weigh the open-source artifact and empirical confirmation of theoretical expectations as sufficient novelty, or whether they require deeper mechanistic insight and broader scope.

# Neutral Reviewer
## Balanced Review

### Summary
This paper demonstrates that LayerNorm (LN) layers can be systematically removed from GPT-2 family models (up to 1.5B parameters) via sequential fine-tuning with minimal performance degradation. The authors show that removing LN linearizes the residual stream, making Direct Logit Attribution (DLA) mathematically exact compared to ground-truth Direct Effects, thereby offering significant benefits for mechanistic interpretability. While the method is successfully scaled and released, the paper also quantifies side effects such as increased model overconfidence and the lack of improvement in attribution patching accuracy.

### Strengths
1.  **Demonstrated Feasibility and Scaling:** The authors provide robust empirical evidence (Table 1, Figure 1) that LN removal is scalable across the GPT-2 suite (Small to XL) and even Pythia-70M. The finding that fine-tuning data requirements grow sublinearly with model size (Section 4) is a valuable insight for future larger-scale applications.
2.  **Interpretability Breakthrough:** The result that DLA becomes an exact estimator of the Direct Effect (NMAE drops from ~49% to 0%) upon LN removal (Section 5.1) is a significant theoretical contribution. It validates assumptions made in interpretability literature and removes a critical source of approximation error.
3.  **Open Science and Repeatability:** The release of code, fine-tuned models on Hugging Face, and the specific removal schedules (Appendix B) significantly lowers the barrier for replication, a high priority for ICLR.
4.  **Comprehensive Ablation Studies:** The investigation into "confidence neurons," first-token norms, and the stability of the fine-tuning process (Section 6.1) adds depth to the empirical claims beyond simple loss metrics.

### Weaknesses
1.  **Limited Interpretability Gains (Attribution Patching):** Contrary to the authors' hypothesis that removing LN would improve interpretability broadly, Section 5.2 shows that attribution patching accuracy does not significantly improve. This suggests other nonlinearities (Softmax, MLP) dominate the difficulty, limiting the broader claim that LN-free models are "better" for all interpretability tasks.
2.  **Robustness and Calibration Issues:** The GPT-2 XL LN-free model exhibits a catastrophic average loss spike on The Pile (130.21) due to rare tokens (Appendix E), despite similar median behavior. Furthermore, the models consistently display increased overconfidence (Section 5.4, Appendix H), which raises concerns about using LN-free models for downstream safety-sensitive tasks.
3.  **Generalization Scope:** The evaluation is heavily centered on GPT-2 architecture. While Pythia-70M is tested, there is no exploration of modern architectures using RMSNorm or architectures without explicit LN (e.g., certain Mamba or recent Llama variants), limiting the claims about the universality of the method.
4.  **Fine-tuning Instability:** Section 6.1 acknowledges significant training instability (exploding gradients) during the removal process, particularly for LN_qk. This makes the protocol sensitive to hyperparameters and potentially difficult to apply to much larger models where fine-tuning budgets are tighter.

### Novelty & Significance
The novelty is moderate but practically high. A precursor to this work was presented at the NeurIPS 2024 workshop (cited in the text), so the core idea of removing LN is not entirely new. However, extending this to GPT-2 XL, providing a detailed interpretability analysis on the resulting linearized models, and releasing a full suite of models elevates this to ICLR standards. The significance lies primarily in the impact on mechanistic interpretability: proving that the LN-induced nonlinearity complicates causal attribution and that its removal yields exact analytical tools (DLA). This has high value for the interpretability community, even if the practical language modeling benefits are marginal.

### Suggestions for Improvement
1.  **Elaborate on Attribution Patching Null-Majority:** The authors should discuss in the main text whether the failure of attribution patching to improve suggests that the "thorny nonlinearity" of LN is less responsible for interpretability difficulties than previously assumed, or if the specific architecture of GPT-2 masks potential gains.
2.  **Analyze Generalization to RMSNorm:** Since RMSNorm is increasingly standard (e.g., in Llama), clarify whether the linearization technique applies to RMSNorm models, or if the mean-centering in LN is the specific feature breaking interpretability.
3.  **Address Calibration Deficit:** Given the consistent overconfidence issues, discuss whether specific post-hoc calibrations (temperature scaling) can recover the utility of these models for applications requiring reliable probability estimates.
4.  **Stability Analysis:** Provide a brief sensitivity analysis on the auxiliary loss hyperparameter (lambda) to quantify how critical it is for successful scaling versus smaller models.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Evaluate the removal protocol on modern architectures (e.g., Llama-2 7B with RMSNorm, SwiGLU, RoPE). Without this, the claim that "LLMs can function without LN" is restricted to obsolete GPT-2 architectures and lacks relevance for current research.
2. Perform a concrete circuit discovery task (e.g., Indirect Object Identification) on both original and LN-free models. Claiming "more precise interpretability" requires evidence that actual mechanisms are easier to find, not just that DLA math is exact.
3. Test robustness on adversarial or out-of-distribution datasets beyond The Pile-filtered. The massive loss spike on GPT-2 XL (130 vs 2.4) suggests catastrophic failure modes not captured by mean metrics.
4. Ablate the auxiliary loss component across all model sizes to quantify its necessity. The scaling claim depends on this stabilization mechanism, but its necessity is only qualitatively described for larger models.

### Deeper Analysis Needed (top 3-5 only)
1. Decompose the attribution patching error budget to quantify contributions from Softmax vs. LN nonlinearities. The negative result contradicts the core motivation; without quantifying other bottlenecks, the interpretability benefit is unclear.
2. Measure representational similarity (e.g., CKA) between original and LN-free residual streams. To claim LN-free models are valid "proxies" for original models, you must prove internal representations remain aligned despite architectural changes.
3. Investigate the root cause of increased overconfidence beyond disabled confidence neurons. Safety-critical interpretability research requires calibrated models; unexplained confidence shifts limit the utility of these releases.

### Visualizations & Case Studies
1. Side-by-side visualization of a discovered circuit (e.g., IOI) in original vs. LN-free models. This would reveal whether removing LN actually reduces entanglement and sparsifies the computational graph in practice.
2. Display specific prompt examples where the LN-free model fails catastrophically compared to the original. Average metrics hide the severe outlier behavior observed in GPT-2 XL on The Pile.
3. Plot layer-wise histograms of residual norms for both model types. This would visually confirm the claim that LN-free models learn to stabilize norms internally without explicit normalization.

### Obvious Next Steps
1. Scale the removal protocol to Llama-2/3 family models to ensure relevance for modern mechanistic interpretability. GPT-2 is no longer the standard benchmark for scaling laws or architecture studies.
2. Develop a method to retain calibration while removing LN nonlinearities. The observed overconfidence is a safety risk that must be mitigated before these models can be widely adopted for alignment research.
3. Train Sparse Autoencoders (SAEs) on LN-free models to test feature monosemanticity. This directly validates the hypothesis that removing LN reduces feature entanglement more effectively than standard approaches.

# Final Consolidated Review
## Summary

This paper demonstrates that LayerNorm (LN) layers can be removed from all GPT-2 model variants (up to 1.5B parameters) via a sequential fine-tuning procedure with minimal performance degradation. The authors introduce "FakeLN" blocks that replace the dynamic standard deviation with a fixed scalar, supported by an auxiliary loss that regularizes activation norms. Key findings include: (1) DLA becomes mathematically equivalent to the Direct Effect in LN-free models (NMAE drops from ~49% to 0%), (2) attribution patching accuracy does not improve despite LN removal, and (3) "confidence neurons" lose their functional role in LN-free models. The paper releases all LN-free GPT-2 variants on Hugging Face.

## Strengths

- **Exact linearization of Direct Logit Attribution**: The result that DLA and Direct Effect become mathematically equivalent in LN-free models (NMAE from 49.07% to 0.00%) is a clean theoretical contribution with empirical verification. This eliminates a major source of approximation error in mechanistic interpretability methods and quantifies the magnitude of the prior approximation error.

- **Successful scaling across model sizes with sublinear data requirements**: The authors demonstrate LN removal works across GPT-2 Small, Medium, Large, and XL. The finding that fine-tuning data requirements grow sublinearly (12× parameters require only 2.6× more tokens) has practical implications for potential extension to larger models.

- **Negative result on attribution patching is scientifically honest**: The finding that removing LN does not improve attribution patching accuracy challenges community assumptions about the primary bottleneck for this method. This suggests other nonlinearities (attention SoftMax, MLP activations) may dominate, which is valuable for directing future interpretability research.

- **Open release of artifacts**: The paper provides complete fine-tuning schedules (Appendix B), code, and all LN-free GPT-2 models on Hugging Face, significantly lowering barriers for replication and follow-up work.

## Weaknesses

- **Sublinear scaling claim rests on thin empirical evidence**: The conclusion that "fine-tuning data requirements grow sublinearly" is based on only four data points (Small through XL). More importantly, the hyperparameter tuning costs for Large and XL models are not included in this analysis. Appendix B.1 acknowledges "significant hyperparameter tuning" was required, which means the effective data requirement (including tuning runs) may be substantially larger than reported.

- **GPT-2 XL outlier failure on The Pile is inadequately explained**: The LN-free XL model achieves CE loss of 130.22 on The Pile versus 2.44 for the original—a catastrophic failure driven by just three samples containing rare tokens. The paper attributes this to "residual stream norm explosions" but does not explain why this failure mode appears only in XL and not other model sizes. Practitioners using GPT-2 XL LN-free on data containing uncommon tokens may encounter severe degradation without warning.

- **Limited systematic ablation of the proposed method**: The paper lacks controlled ablations for key design choices: (a) the auxiliary loss contribution is assessed qualitatively ("curves were more spiky without it") but not quantified in a controlled comparison, (b) the removal ordering (MLP-first vs. QK-first) is justified only with qualitative reasoning, and (c) simultaneous removal of all LN blocks is mentioned as "irreparably breaking performance" but not quantified. These omissions make it harder to understand which components are essential.

- **Overconfidence issues acknowledged but not resolved**: All LN-free models exhibit increased overconfidence (e.g., GPT-2 Medium entropy drops from 2.86 to 2.53, ECE nearly doubles). While the paper identifies disabled confidence neurons as one contributing factor, it notes "additional contributing factors" remain unexplained. This calibration deficit is a practical concern for downstream interpretability research that may rely on calibrated probability estimates.

- **Proxy fidelity concern under-addressed**: The paper positions LN-free models as tools for understanding original models, but acknowledges "similarity is not exact." The observed changes—overconfidence, altered attention sink behavior (55.3% → 45.3%), disabled confidence neurons—suggest substantial internal reorganization. The paper provides limited analysis of whether circuits discovered in LN-free models transfer to their LN counterparts, which is critical for the paper's core use case.

## Nice-to-Haves

- **Evaluation on architectures using RMSNorm**: Since RMSNorm is increasingly standard (e.g., Llama family), extending the analysis to clarify whether the linearization technique applies or requires modification would broaden impact.

- **Concrete circuit discovery comparison**: Demonstrating that actual interpretability tasks (e.g., discovering the IOI circuit) are easier or more accurate in LN-free models would strengthen the practical utility claim beyond the DLA result.

- **Temperature scaling or post-hoc calibration**: Given the consistent overconfidence, a brief analysis of whether simple calibration methods can restore utility would be valuable for practitioners.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Title criticism about "Small Transformers"**: This is a formatting/style nitpick that does not affect the paper's substance.

- **Demand for modern architecture evaluation (Llama-2/3, Mamba, etc.)**: The paper explicitly scopes its contribution to GPT-2-class models. Evaluating additional architectures would broaden impact but is outside the stated scope.

- **Request for attribution patching evaluation on multiple tasks/models**: The single-task evaluation (IOI on GPT-2 Small) provides useful data; demanding broader evaluation is scope creep. The negative result is clearly reported.

- **Claim that NMAE formulation is non-standard**: Footnote 5 explains the choice ("we did not observe a consistent proportional relationship") which is a valid methodological justification.

- **Demand for adversarial/OOD robustness testing beyond The Pile-filtered**: The paper already includes this analysis (The Pile-filtered addresses distribution shift, and the outlier behavior is discussed). Additional adversarial testing would be valuable but is not a core flaw.

- **Request for CKA representational similarity analysis**: While interesting, this is an additional analysis that would strengthen but is not essential given the paper's scope.

## Novel Insights

The most striking finding is that attribution patching accuracy does not improve despite complete LN removal, contradicting the widely-held belief in the mechanistic interpretability community that LN is "a particularly thorny nonlinearity" for this method (Nanda, 2023a). This suggests that attention SoftMax or MLP activations may be the dominant bottleneck for attribution patching accuracy, redirecting research attention toward these components. Additionally, the observation that confidence neurons maintain their structural signature (high weight norm, low logit variance) but lose functional impact in LN-free models provides elegant confirmation of Stolfo et al.'s proposed mechanism while revealing that fine-tuning alone partially degrades confidence regulation—a confound worth noting.

## Suggestions

- **Quantify the auxiliary loss contribution**: Include a controlled ablation (with vs. without auxiliary loss) for at least one model size to demonstrate its necessity, particularly for the larger models where the paper claims it was critical.

- **Provide more analysis of the XL outlier failure**: Even a brief mechanistic explanation for why residual norm explosions occur only in XL (or include this as a noted limitation to warn users) would strengthen the paper's practical guidance.

- **Clarify proxy fidelity**: Add a brief discussion of what interpretability findings from LN-free models can and cannot be expected to transfer to original models, given the observed representational changes.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
