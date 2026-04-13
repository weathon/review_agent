=== CALIBRATION EXAMPLE 24 ===

# Final Consolidated Review
## Summary
This paper studies localized machine unlearning through the lens of memorization. It first empirically examines several localization strategies motivated by memorization hypotheses, then proposes a new one-shot, batched localization method that combines channel-level selection with weighted-gradient scoring, and pairs it with reset-and-finetune to form DEL. Across the reported CIFAR-10/ResNet-18 and SVHN/ViT settings, DEL is consistently stronger than the compared localized methods and also competitive against or better than the included full-parameter baselines on the paper’s chosen proxy unlearning metrics.

## Strengths
- **The paper does more than just introduce another heuristic; it decomposes the design space of localization strategies in a useful way.** Section 4 compares Deepest, Shallowest, CritMem, and SalLoc, and Section 5/ Tables 1 and 4 isolate two concrete axes—granularity and criticality criterion—before assembling the final method. This makes the contribution scientifically more valuable than a pure “ours beats baseline” paper.
- **A specific and non-obvious empirical finding is that simple data-agnostic localization rules behave very differently than commonly assumed.** In Figure 2, Deepest preserves test accuracy but forgets poorly at small budgets, while Shallowest forgets better but hurts utility. That is a meaningful insight for localized unlearning design, not a generic strength.
- **The proposed localization strategy appears modular rather than overfit to a single downstream unlearning rule.** Figure 3 shows gains when paired with multiple unlearning algorithms, not only the authors’ DEL pairing, which supports the claim that the main contribution is the localization mechanism itself.
- **The random-mask control is genuinely informative.** Table 3 keeps the same layerwise masking structure and budget while randomizing which parameters/channels are selected, and performance degrades substantially. This directly supports the claim that the benefit comes from identifying critical parameters, not merely touching fewer weights.
- **The paper includes a nuanced analysis of forget-set specificity.** Table 4 distinguishes IID vs. non-IID forget sets and shows that tailoring localization to the forget set matters more in the non-IID case. That is a useful insight about when data-dependent localization is likely to pay off.
- **Within the paper’s evaluation protocol, DEL’s empirical performance is strong.** Table 2 and Figure 4 show clear improvements over SalLoc-RL and several full-parameter baselines on the reported unlearning metrics, while utility is often improved relative to other localized methods.

## Weaknesses

### Major:
- **The paper’s “state-of-the-art” framing is stronger than the evidence supports because the evaluation relies on limited proxy metrics.** The paper itself acknowledges in Section 2.1 that “Evaluating unlearning rigorously is an ongoing area of research” and that stronger evaluation methods exist but are expensive; nevertheless, the main claims in the abstract, introduction, and conclusion are phrased as broad SOTA claims. The actual evaluation centers on forget accuracy and a single confidence-based MIA efficacy metric, plus utility. These are standard community metrics, so using them is not itself a flaw; the issue is that the headline claims should be calibrated to “improves standard benchmark metrics” rather than implying a stronger conclusion about unlearning writ large.
- **Efficiency is a central motivation but is not quantified in a convincing end-to-end way.** The introduction and Section 2.1 frame localized unlearning around forgetting/utility/efficiency tradeoffs, and Section 5 argues the proposed method is more efficient than CritMem due to one-shot batched localization. However, the evidence presented is largely parameter budget, not runtime, FLOPs, memory, or full unlearning cost including localization and finetuning. Since localization itself can be nontrivial, parameter fraction is not enough to establish the efficiency claim.
- **The main comparative results are not organized as clean tradeoff frontiers under matched constraints.** The paper states in Section 4 that “throughout the paper, we tune hyperparameters separately for each budget and localization strategy,” and Section 6 reports each localized method using its “best identified parameter budget and its best-paired unlearning algorithm for that setting.” This is useful for showing best achievable behavior, but it weakens the stronger claim that DEL is broadly superior on the forgetting/utility/efficiency tradeoff. A more compelling presentation would compare methods along common budget/compute frontiers rather than best-picked operating points.
- **The empirical scope is still fairly limited for the breadth of the claims.** The paper evaluates on CIFAR-10/ResNet-18 and SVHN/ViT, with IID and non-IID 10% forget sets on CIFAR-10. These are reasonable benchmarks, and the paper is stronger than a single-dataset study, but the claims in the abstract and conclusion read broader than what these settings can fully support. In particular, it remains unclear how the method behaves for larger models/datasets, substantially different forget-set sizes, or repeated unlearning requests.

### Minor
- **The memorization framing is somewhat overstated relative to what the experiments actually establish.** The paper’s strongest scientific result is not that memorization hypotheses directly yield good unlearning strategies; in fact, Section 4 finds that direct translations like Deepest and CritMem are weak or impractical. The successful method is better described as a practical synthesis of design choices inspired by memorization-localization work (channel granularity, weighted gradients, one-shot batching) rather than validation of a strong memorization-to-unlearning theory.
- **A few important design choices are under-analyzed, especially the channel score hyperparameter \(h\).** Section 5.2 defines neuron criticality as the average of the top-\(h\) parameter scores in a channel, but the main paper gives little evidence about sensitivity to \(h\). Since this is part of the core localization rule, robustness analysis would strengthen confidence that the method is not delicately tuned.
- **The role of the always-updated classifier layer is not isolated.** In Section 4, the reset+finetune setup updates the identified critical parameters “and the classifier layer.” Because this appears to be part of the localized recipe, an ablation would help clarify how much of the gain comes from the localization strategy versus the classifier update.
- **Some causal language is too strong for the evidence.** For example, Section 6 says Table 3 indicates “beyond doubt” that localized unlearning succeeds due to pinpointing critical parameters. The random-mask control is strong evidence, but “beyond doubt” overstates what a single control can establish.

### Trivial
- **Some tables are harder to interpret than necessary because the \(\Delta\) conventions are not immediately intuitive.** A brief reminder of the sign/direction near the tables would improve readability, though this does not affect the substance.

## Nice-to-Haves
- Add wall-clock/FLOP/memory measurements for localization and full unlearning, not just parameter budget.
- Show common-budget or common-compute tradeoff curves for DEL, SalLoc, and strong full-parameter baselines.
- Include an ablation over forget-set size, since all main experiments appear to use 10% forget sets.
- Add sensitivity analysis for \(h\) and for the decision to always update the classifier layer.
- Include one stronger privacy audit or additional attack variant to test whether gains persist beyond the chosen MIA metric.
- Report retain accuracy more prominently in the main paper, since it is an important utility measure and is currently mostly deferred to the appendix.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Deepest at high budget amounts to retraining the whole model” is incorrect.** The harsh review rightly objected to the literal equivalence. The paper says this only loosely: “amounts to retraining (almost) the entire model.” Strictly speaking, reset+finetune of most layers with some layers frozen is not retraining from scratch, so this wording is imprecise. However, this is not a substantive methodological flaw and should not be elevated as a real weakness.
- **Complaints about omitted implementation details / reproducibility knobs.** The paper explicitly states hyperparameter details are in Section A.2 and notes per-budget/per-strategy tuning. Demands for more low-level detail would be outside scope for a main-review criticism.
- **Claims centered on whether cited methods/tools exist or are available.** Per instructions, such concerns are removed.
- **Missing-related-work style criticism in the abstract.** The user instructions prohibit external “missing related works” complaints, so these are not included.
- **“Need significance tests” as a core weakness.** The paper already reports means with uncertainty (e.g., \(\pm\) values) in tables. Asking for formal significance testing is at most a nice-to-have here, not a core flaw by community standards for this kind of empirical systems/benchmark paper.

## Novel Insights
The most interesting synthesis across the reviews and the paper is that the work’s real contribution is not “memorization localization solves unlearning,” but something subtler: direct memorization hypotheses do *not* transfer cleanly to unlearning, yet certain *inductive biases* from that literature do. In particular, the paper suggests that coarse, channel-level localization with a conservative weighted-gradient score may act as a useful regularized selection mechanism—more robust than parameter-level saliency—without requiring the literal correctness of the underlying memorization theory. This makes the paper more valuable as a study of what kinds of localization bias are useful for unlearning than as evidence for a strong mechanistic connection between memorization and forgetting.

## Suggestions
- **Recalibrate the headline claims.** Replace broad “sets a new state-of-the-art” language with a claim tied to the paper’s actual evidence: e.g., “improves standard proxy unlearning metrics on the evaluated benchmarks.”
- **Quantify efficiency directly.** Report end-to-end runtime and/or FLOPs for localization plus unlearning, and compare against SalLoc, CritMem, and representative full-parameter methods.
- **Present matched-frontier comparisons.** In addition to best-case points, show curves under fixed parameter budget and, ideally, fixed compute.
- **Clarify the scientific claim about memorization.** Emphasize that the paper finds direct memorization-localization hypotheses are not sufficient, but design principles extracted from that literature improve localized unlearning.
- **Add targeted ablations.** In particular, vary \(h\), isolate the effect of always updating the classifier layer, and vary forget-set size.
- **Strengthen evaluation if possible.** Even one additional privacy audit or attack variant would materially improve confidence in the unlearning claims.
- **Temper causal phrasing.** Replace formulations like “beyond doubt” with more measured language aligned with the strength of the evidence.



# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 6.0, 5.0]
Average score: 5.0
Binary outcome: Reject
