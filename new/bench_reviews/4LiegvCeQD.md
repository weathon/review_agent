## Summary

The paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation (TTA) method that adapts a set of independently pre-trained classifiers to distribution-shifted data one sample at a time. At each step, IEL selects the model whose softmax output has the highest probability for the majority-voted class, then minimizes the cross-entropy between that soft target and all ensemble members, updating all trainable parameters. Experiments on CIFAR-10C, CIFAR-100C, and ImageNet-C show improvements over static (unadapted) ensembles for most corruption types.

## Strengths

- **Addresses a practically relevant single-sample TTA setting.** The paper explicitly designs for batch-size-1 adaptation by freezing batch normalization parameters, filling a gap left by TENT and other methods that rely on batch statistics. This is a genuine and useful constraint (Sec. 1, Sec. 4).

- **Simple and clear core mechanism.** The dynamic teacher-selection rule—majority vote to determine the class, highest confidence model as soft target—is straightforward to describe and implement. The method genuinely adapts multiple models simultaneously, and individual model improvements are reported alongside ensemble improvements (Tables 1–3).

- **Broad empirical coverage.** Three benchmark datasets (CIFAR-10C, CIFAR-100C, ImageNet-C), 15 corruption types each, and multiple architectures per benchmark provide a decent empirical scope. Results are broken down by corruption type and model, which is informative about where IEL helps vs. harms.

- **Honest reporting of failure cases.** The paper explicitly reports negative improvements on noise-type corruptions (Gaussian Noise, Shot Noise, Impulse Noise) and acknowledges long-run degradation below baseline (Figure 3, Sec. 3.1), rather than hiding these.

## Weaknesses

### Major:

- **No comparison to any existing TTA method.** The paper positions itself in the TTA literature, cites TENT, EATA, CoTTA, and ROID in Sec. 2.2, but experiments compare only against a static (unadapted) ensemble. The only supported claim is "joint distillation within an ensemble beats a frozen ensemble," which is a much weaker proposition than "IEL is an effective TTA method." Without at minimum a comparison to single-model entropy minimization (TENT with frozen BN), EATA, or even simple self-training/pseudo-labeling on a single model, the core empirical thesis that IEL is a competitive TTA approach is unsubstantiated. This is not a missing-ablation issue; it is a structural gap in the evaluation that makes it impossible to assess IEL's value relative to existing work it explicitly claims to improve upon.

- **Best-epoch cherry-picking inflates reported results and obscures catastrophic forgetting.** Tables 1–3 report "Highest accuracy improvements (%) over all epochs," which corresponds to oracle epoch selection per corruption type. The paper's own Figure 3 and Sec. 3.1 acknowledge that accuracy peaks early and then degrades—sometimes below the static baseline. In a real TTA deployment, there is no oracle to select the best epoch. Reporting only peak improvements without also showing fixed-epoch results (e.g., after one pass, or at a predetermined epoch) makes it impossible for readers to assess the method's practical stability. This is especially concerning given the severe degradation (6–30 percentage points) on noise-type corruptions.

- **Misleading claim of "diversity as a new optimization signal."** Contribution bullet 1 states: "This work proposes diversity as a new optimization signal." However, the loss (Eq. 1) minimizes the cross-entropy between all models and a single teacher, which actively *reduces* diversity by forcing agreement. The paper acknowledges in Sec. 2.1 that "by minimizing [the cross-entropy] we force members to agree with the majority voted model," but then frames the contribution as "diversity-based." The actual mechanism is standard distillation from a dynamically chosen teacher—framing this as a "diversity signal" is conceptually backwards and inflates the novelty claim.

- **Evaluation protocol is easier than standard TTA settings.** Models are reset to source weights between every corruption type (Sec. 4), which avoids cumulative forgetting across distribution shifts. This sidesteps one of the core challenges of continual TTA. Combined with 90/10 tuning/evaluation splits drawn from the same corruption distribution and type, the evaluation is closer to unsupervised domain adaptation than to the more realistic TTA scenarios used in the cited literature. The paper cannot claim to address continual or online TTA under this protocol.

### Minor:

- **No ablations isolating the proposed mechanism.** The paper never tests whether the specific "majority-vote + highest-confidence" teacher selection is better than simpler alternatives (e.g., ensemble-averaged softmax as target, hard pseudo-labeling, entropy minimization on each model independently). Without these, the gains could be attributed to generic self-training effects rather than IEL's specific design.

- **The selected-teacher model is also updated.** When H(x) = h_{θ_j}(x), the j-th term in the loss is h_{θ_j}'s self-entropy, which is small but still produces non-zero gradients. The paper acknowledges this risks "overwriting the strong knowledge of the majority voted models" (Sec. 3) but defers all mitigations (e.g., KL divergence) to future work without justification.

- **No statistical significance or variance reports.** Many improvements are only a few percentage points (e.g., +0.56%/+0.20% for Brightness on CIFAR-10C), yet no standard deviations or confidence intervals are provided. The evaluation splits are also small (10% of 7k samples for ImageNet-C per corruption).

- **Computational cost not quantified.** Backpropagation through all trainable parameters of all M models per single test sample is expensive, but no wall-clock time, FLOPs, or memory comparisons are given.

### Trivial:
- The collaboration analogy ("two heads are better than one") is overextended: the algorithm is one-directional per sample (one teacher distills to all members), not truly symmetric mutual learning. This is a framing issue more than a technical one.

## Nice-to-Haves

- Comparison with at least TENT (frozen BN, batch-size 1) and EATA as TTA baselines, even on a subset of corruption types, to establish whether IEL's ensemble mechanism adds value beyond single-model adaptation.
- Fixed-epoch results (e.g., accuracy after one pass through the data) alongside best-epoch results.
- Ablations with ensemble-averaged softmax as the soft target vs. the proposed majority-vote-highest-confidence mechanism.
- Analysis of how performance scales with ensemble size (2, 3, 5, 10 models).
- A continual/non-stationary evaluation (cycling through corruption types without weight resets).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Only CNN architectures tested, no transformers"** (from human finder): This is a generic scope-expansion request. The paper uses standard TTA benchmarks with standard architectures; demanding transformers is a nice-to-have, not a core flaw for a method paper on these benchmarks.

- **"Severity levels of corruptions not specified"** (from harsh critic): The paper states it uses CIFAR-10C, CIFAR-100C, and ImageNet-C. While specifying severity would be ideal, this is a minor experimental detail rather than a fundamental flaw, since the datasets have standard severity conventions.

- **"No error bars or statistical significance"** (raised by multiple reviewers): While important, single-run evaluation is the norm in the TTA benchmark literature (e.g., TENT, CoTTA, EATA all report single runs on these corruption benchmarks). This is a community-standard practice; demanding otherwise is a nice-to-have.

- **"Incrimental novelty relative to knowledge distillation"** (from human finder): The dynamic teacher selection based on majority voting is a meaningful design choice beyond standard online distillation. While the components are individually well-known, the specific combination for single-sample TTA is what's claimed as novel, and this characterization undersells the contribution.

- **"H(x) notation is ambiguous — returns model vs. softmax vector"**: This is a minor notational clarity issue, not a substantive weakness.

## Novel Insights

The most interesting empirical finding is that IEL works well on blur-type and spatial corruptions (defocus, glass blur, motion blur, zoom blur) but consistently degrades on noise-type corruptions (Gaussian, shot, impulse noise). The paper does not analyze this failure pattern, but it suggests that the quality of the soft target (the majority-voted teacher's softmax) depends heavily on the initial ensemble's accuracy on the corruption type—when all models are already badly confused (as on noise), the majority vote reinforces errors. This is a known pitfall of pseudo-labeling methods, but the ensemble-specific dynamic (where bad agreement produces bad soft targets, causing cascading degradation) could have been examined more deeply.

## Suggestions

1. **Add TTA baselines.** At minimum, run TENT (with BN parameters frozen, batch-size 1) and EATA as single-model baselines. This is essential for the paper's claimed contribution.
2. **Report fixed-epoch accuracy.** In addition to best-epoch results, report accuracy at epoch 1 (single pass) and at a fixed final epoch so readers can assess stability.
3. **Reframe the diversity contribution.** Change "diversity as a new optimization signal" to "disagreement-driven" or "ensemble-consensus-based," since the method minimizes disagreement (lack of diversity). This is more accurate and still highlights the novelty.
4. **Analyze failure modes on noise corruptions.** Provide even a brief analysis of why IEL fails there—is it because the ensemble majority is also wrong, or because noise destabilizes the optimization?

## Score and Decision

**Calibration anchors** (similar TTA papers):
- COME (Conservatively Minimizing Entropy): Accepted poster with scores 6/6/5/8 — had full TTA baselines, thorough comparisons, and consistent improvements.
- TTE (Test-Time Ensemble): Accepted poster with scores 6/8/6/6 — integrated with existing TTA methods and had comprehensive baselines.
- FEATHER (Lifelong TTA with Lightweight Adapters): Rejected with scores 5/5/3/5 — missing key baselines and efficiency concerns.
- TTA with Auxiliary Tasks: Rejected with scores 5/6/3 — methodological gaps and ad-hoc framing.

This paper shares the most critical weakness with FEATHER and TTA-AuxiliaryTasks (missing baselines), but also has additional evaluation protocol issues (best-epoch reporting, per-corruption resets) and a misleading conceptual framing. It is weaker than COME and TTE, which had proper baselines and cleaner claims. The paper is approximately at the level of FEATHER: an interesting idea that is not properly validated against the literature it claims to advance.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>