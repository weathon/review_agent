Now I have enough calibration data. Let me synthesize the final review.

Key calibration references:
- "Machine Unlearning Fails to Remove Data Poisoning Attacks" - Accepted (Poster), scores 6/6/6/6/6. Novel problem formulation, empirical evaluation showing failures of existing methods.
- "Machine Unlearning via Simulated Oracle Matching" - Accepted (Poster), scores 6/6/8. Novel unlearning method with empirical verification.
- "Adversarial Machine Unlearning" - Accepted (Poster), scores 6/6/6/6. Novel adversarial framework for unlearning.
- "Auditing Privacy Protection of Machine Unlearning" - Withdrawn/Reject, scores 3/6/5/6/1. Issues with threat model and evaluation.
- "Machine Unlearning in Audio" - Reject, scores 5/3/3/6/6. First in new domain but incremental method and limited evaluation.
- "GIFF: Forward-Forward" - Withdrawn/Reject, scores 5/5/1/5. FF improvement paper with limited novelty.
- "Towards Understanding the Feasibility of Machine Unlearning" - Reject, scores 6/5/5/5.

Now let me verify the key claims in the paper against reviewer criticisms:

1. **G-MIA as "black-box"**: The paper claims G-MIA is a "lightweight blackbox attack" and "practical." But it requires access to goodness vectors from ALL layers (step 4 of G-MIA). This is indeed more access than typical black-box MIAs. However, the paper argues in §1/§2 that this is because data owners may not have white-box access (full parameters/gradients). The paper does distinguish between black-box (prediction-only) and their method. Let me re-read...

From §5, G-MIA requires "the goodness vectors from all layers" — this is intermediate activations, which is not final prediction output only. However, it doesn't require model parameters or gradients. This is somewhere between black-box and white-box — the paper should acknowledge this more clearly but it's not entirely wrong since they explicitly say "data owners who do not have full access to the models."

Actually, looking more carefully at the paper in §5, they say: "It is also noted that the attacker can obtain the output of the target model of attack, i.e., the goodness vectors from all layers." This is a specific FF model output — in FF models, goodness vectors are indeed a designed output for inference. So calling it "black-box" is semantically reasonable if querying an FF model naturally returns goodness vectors, but it's not "black-box" in the traditional MIA sense.

2. **G-MIA used as verification without calibration**: The paper does reference ACC and AUC, and shows comparisons. However, the harsh critic points out that differences are small (0.5245 vs 0.532) and without confidence intervals. This is a valid concern.

3. **Limited baselines**: The paper compares against RE (retraining) and GA (gradient ascent). The paper argues other methods are "not suited for FF models" but only demonstrates GA fails. This is a valid criticism.

4. **Evaluation scope**: The paper mentions 4 datasets and 3 model architectures, but main text only shows VGG13/CIFAR-10. The appendix presumably contains more.

5. **Formal unlearning guarantees**: The paper doesn't claim formal guarantees — it's an empirical unlearning paper similar to many in the field. This is a nice-to-have.

6. **The conceptual gap between goodness alignment and "removing the effect"**: This is a valid point — the paper uses goodness alignment as a proxy for unlearning but doesn't formally connect them.

Let me now construct the final review.</think>## Summary

This paper introduces FF-Erase, the first machine unlearning framework designed specifically for Forward-Forward (FF) models. FF-Erase employs a "guidance model" to generate target goodness distributions, steering the original model to unlearn forgetting data by shifting layer-wise goodness scores via KL-divergence (forgetting forward), combined with periodic "recovering forward" steps on remaining data. The paper also proposes G-MIA, a membership inference attack leveraging FF models' per-layer goodness vectors to verify unlearning effectiveness. Experiments on four image benchmarks and three FF architectures demonstrate that FF-Erase achieves unlearning 1.9–3.1× faster than retraining from scratch with 1.6–3.3% accuracy degradation.

## Strengths

- **Novel and well-motivated problem formulation.** This is the first paper to formalize and address machine unlearning for FF models. The analysis of why direct gradient ascent causes model collapse in FF models (layer-wise independent training, parameter sensitivity) is concrete and well-illustrated (Figure 1). This addresses a genuine gap in the literature.

- **Principled, architecture-aligned solution.** FF-Erase's design directly leverages FF's layer-wise goodness structure. The KL-divergence alignment to a guidance model's goodness distribution is a natural fit for FF's training paradigm, and the dual forward mechanism (forgetting + recovering) directly addresses the effectiveness-utility trade-off. The ablation (Table 1) convincingly shows that a randomly initialized guidance model causes catastrophic collapse (ACC_t = 55.53%), validating the design rationale.

- **Flexible guidance model strategies.** Offering both mini-retraining and fast-distillation strategies with tunable α₁ and α₂ provides practical flexibility across scenarios with different data availability and time budgets. The efficiency analysis (§4.3) transparently breaks down the computational cost.

- **G-MIA is a creative FF-specific contribution.** Exploiting per-layer goodness vectors as membership inference features is novel and well-motivated for FF architectures. The result that G-MIA outperforms classical black-box and even white-box MIAs on deeper models (Figure 3) demonstrates the value of this FF-specific design.

- **The paper is clearly written** with well-structured methodology, detailed algorithm descriptions, and informative visualizations (Figure 2).

## Weaknesses

### Major:

- **Limited baselines for the claim that existing unlearning methods are "not feasible" for FF models.** The paper's central motivation rests on the assertion that classical unlearning methods (§1: "Existing machine unlearning methods are not feasible for FF models") and that "direct gradient ascent induces optimization instability and frequent model collapse." However, only vanilla gradient ascent (GA) with coarse λ-sweeping is tested on a single model/dataset combination in the main text (VGG13/CIFAR-10). More sophisticated unlearning variants — such as influence-function-based methods, SCRUB-style knowledge distillation, or even GA with better regularization/early stopping/layer-wise learning rates — are claimed to be incompatible but never empirically demonstrated to fail. The blanket "not feasible" statement is overstated given the evidence. This matters because the paper's core narrative requires demonstrating that FF-specific design is *necessary*, not just beneficial. Without showing that adapted versions of modern approximate unlearning methods also fail or underperform on FF models, the motivation is weakened.

- **G-MIA is mischaracterized as "black-box" yet requires per-layer goodness vectors.** The paper repeatedly claims G-MIA is a "lightweight blackbox attack" (Abstract) and "practical for verification" (§1). However, G-MIA requires the attacker to obtain "the goodness vectors from all layers" (§5) — intermediate layer activations that go substantially beyond the final prediction output assumed by standard black-box MIAs. While it is true that FF models produce goodness vectors as a designed output, calling this "black-box" without qualification is misleading by community standards. No real deployment scenario is described where a data owner has API access to per-layer goodness but not to model weights. This matters because the practicality claim is a core contribution. The paper should either acknowledge this intermediate access level explicitly, or demonstrate G-MIA's effectiveness under stricter access constraints.

- **Evaluation relies primarily on G-MIA, the paper's own proposed metric, without calibration or error reporting.** The claims that FF-Erase achieves "effective unlearning" and is "comparable to retraining" depend heavily on G-MIA scores. However: (1) no confidence intervals or variance across multiple runs are reported; (2) the differences between methods (e.g., G-MIA ACC 0.5245 vs. 0.532 for FF-Erase(D) vs. RE) are very small — on the order of 0.01–0.05 — and may not be statistically significant; (3) no "no-unlearning" baseline (original model before unlearning) is reported to contextualize G-MIA scores, making it hard to assess what scores actually indicate effective unlearning. This creates a risk of circular evaluation: the unlearning method and verification method are co-designed for the same architecture, and small differences in a non-calibrated metric drive the paper's conclusions.

### Minor:

- **Limited experimental coverage in the main text.** While the paper claims results "on 4 standard image benchmarks" with "various FF models," detailed results are shown only for VGG13/CIFAR-10 in the main body. The 1.9–3.1× speedup and 1.6–3.3% accuracy degradation ranges are stated but not explicitly tabulated per setting in the main text, making it difficult to assess whether observations generalize consistently across all configurations.

- **Conceptual gap between goodness alignment and information removal.** FF-Erase aligns each layer's goodness distribution to that of a guidance model trained without the forgetting data. While intuitively plausible, the paper does not analyze whether this removes higher-order information about forgetting samples (e.g., correlated patterns, memorization in earlier layers) or merely matches distributional statistics. No representational similarity analysis or stronger attack evaluation is provided to address what *type* of information is removed.

- **No testing at high forgetting ratios.** All main experiments use β = 20% forgetting ratio. At higher ratios (e.g., 50%+), the remaining data shrinks, and both guidance model quality and recovering forward effectiveness may degrade substantially. The paper does not explore this regime, which is practically relevant for GDPR-type scenarios.

## Nice-to-Haves

- Formal or theoretical analysis connecting the KL-divergence alignment in goodness space to formal unlearning guarantees (even approximate ones) would strengthen confidence beyond empirical G-MIA scores.
- Comparison of at least one adapted modern approximate unlearning method (e.g., a SCRUB-style distillation or influence-function approach) on FF models, even if it fails — documenting the failure would strengthen the necessity claim for FF-Erase.
- Sensitivity analysis of G-MIA under stricter access conditions (e.g., using only the final prediction layer, or subsets of layers) would clarify the practical boundary of the verification method.
- Analysis of per-layer unlearning dynamics to validate the stated challenge that "layers diverge in update directions" during naive GA.

## Removed Points

- **Claim that FF models or their variants are unproven or have limited applicability.** The paper cites CwComp, FF-LSTM, FORWARDGNN, Deeperforward, and Symba as established FF methods. Per the hard rules, cited entities are assumed to exist.
- **Request to test on non-vision benchmarks (e.g., graph, tabular, text).** The paper's scope is FF models on standard image benchmarks consistent with prior FF work. Demanding evaluation on tasks where even the base FF algorithms are still being explored is scope creep.
- **Demand for formal unlearning guarantees or theoretical proofs.** This is a standard expectation in some unlearning theory papers, but not the norm for empirical systems papers. The paper makes no formal guarantee claims.
- **Nitpick about missing hyperparameter details or training details.** The key hyperparameters (α₁, α₂, K, λ, η) are specified. Stopping thresholds and batch-level sampling details are minor implementation concerns.
- **Formatting/style complaints.** Removed per rules.
- **Criticism that the paper should compare against SCRUB, SalUn, or other BP-based unlearning baselines adapted "naively" to FF.** This is partially valid (the motivation would be stronger with more baselines), but the paper does present evidence that the fundamental training paradigm differs (layer-wise vs. global), and directly adapting BP methods to FF's goodness-based objective is non-trivial. Keeping a weakened version as a major weakness above, but removing the demand for exhaustive BP baseline adaptation.

## Novel Insights

The paper's insight that FF models' layer-wise independent training makes gradient-based unlearning fundamentally unstable — because parameter shifts in one layer don't propagate consistent gradient signals across layers — is the core conceptual contribution. This is distinct from BP models where the chain rule provides coordinated gradient directions. The proposed solution (guidance model + KL-divergence alignment) is a natural response: instead of relying on coordinated gradient signals, provide each layer with its own stable target distribution. Whether this insight generalizes beyond the specific FF architecture to other locally-trained systems (e.g., decoupled neural interfaces) is an interesting direction suggested by this work.

## Suggestions

- **Add a "no-unlearning" G-MIA baseline**: Report G-MIA scores on the original model (before any unlearning) alongside the unlearned and retrained models, so readers can interpret what specific G-MIA values mean in absolute terms rather than only relative comparisons.
- **Report standard deviations across at least 3 random seeds** for all key metrics (G-MIA ACC/AUC, Acc_forget, Acc_test, timing) to establish statistical significance of reported differences.
- **Clearly position G-MIA's access level** as "gray-box" or "intermediate access" rather than "black-box," and discuss deployment scenarios where per-layer goodness is available.
- **Test at least one additional forgetting ratio** (e.g., β = 50%) to demonstrate robustness beyond the 20% setting.

## Score and Decision

Calibration: I compared this paper against several related works in unlearning and FF algorithms:
- "Machine Unlearning Fails to Remove Data Poisoning Attacks" (score ~6, Accept-Poster): Novel problem formulation with systematic empirical evaluation, but similar concerns about evaluation metrics and limited methodological novelty.
- "Machine Unlearning via Simulated Oracle Matching" (score ~6-8, Accept-Poster): Novel unlearning framework with strong empirical results and a clear theoretical grounding.
- "Machine Unlearning in Audio" (score ~3-6, Reject): First in a new domain, limited baselines, incremental methods.
- "Auditing Privacy Protection of Machine Unlearning" (score ~3-6, Withdrawn/Reject): Issues with threat model evaluation framework.
- "GIFF: Forward-Forward" (score ~1-5, Withdrawn/Reject): FF improvement paper with limited novelty.

This paper presents a genuinely novel problem (FF unlearning) with a well-motivated, architecture-specific solution. The core contribution is real. However, the evaluation has substantive gaps: the baseline comparison is narrow, the key metric (G-MIA) is self-designed and uncalibrated, the "black-box" claim is overreaching, and the generalizability of claims rests on appendix-only results. These are not fatal but meaningfully weaken the paper's central claims. The paper is above the "first in domain but weak method" papers (like Audio Unlearning at ~3-5), but below papers with stronger empirical grounding and more calibrated metrics (like Simulated Oracle Matching at ~6-8). Given the novelty of the problem formulation and the mechanistic quality of the solution, paired with the evaluation concerns, this falls in the weak-accept to borderline range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>