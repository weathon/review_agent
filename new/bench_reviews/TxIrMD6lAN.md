## Summary

This paper proposes integrating task-specific adapters with co-training of both the backbone and adapters for incremental learning, contrasting with prior adapter-based IL approaches that freeze the backbone. A backbone distillation regularizer (for prediction-based methods) or adapter parameter masking (for weight-based methods) enforces stability in the shared representational space. The method integrates cleanly across five classical regularization baselines (EWC, MAS, PathInt, LwF, LwM) and demonstrates consistent 3–5% accuracy gains across CIFAR-100 orderings and task scales, with preliminary (but underspecifed) results on ImageNet-Subset.

## Strengths

- **Method-agnostic adapter integration with consistent gains.** As shown in Figure 3, the approach lifts accuracy by 3–5% across five diverse regularization baselines (both weight-regularized and prediction-regularized) without algorithm-specific re-engineering. The modifications—backbone distillation for prediction methods (§3.2.1, Eq. 1) and adapter parameter exclusion from consolidation for weight methods (§3.2.1, Eq. after line 136)—are minimal drop-in additions.
- **Robust evaluation across task orderings and scales.** Figures 4–5 demonstrate gains persist across alphabetical, coarse-grained, and iCaRL orderings, and at 5/10/20 classes-per-task. This breadth of evaluation exceeds what is typical in many IL papers that rely on a single fixed ordering.
- **Co-training backbone empirically outperforms frozen-backbone adapter paradigm.** Table 2 shows LwF-A (co-trained) at 74.0% vs. LwF-A-FrB (frozen backbone) at 72.9%, providing direct evidence that continuous backbone updates contribute beyond the adapter architecture alone—noting this validates the architectural choice but does not fully validate the deeper mechanistic claim (see Weakness 1).
- **Clean, minimal modifications enabling broad adoption.** Unlike prior adapter-based IL methods that rely on custom complex losses (e.g., TAMiL's attention modules), this approach requires only additive loss terms or parameter exclusions.

## Weaknesses

### Fatal
None

### Major

- **The core mechanistic claim—invariant vs. task-specific feature separation—is confounded with parameter capacity and remains unverified.** The paper's central thesis (§3.2, §3.2.1) is that adapters capture task-specific knowledge while the backbone learns shared invariant features. However, every adapter-augmented baseline gains 3–5% over its non-adapter counterpart, which trivially increases the number of trainable parameters per task. The bottleneck-width ablation (Figure 6) shows that wider adapters (128, 256) sustain higher accuracy than narrower ones (16, 32), which is consistent with a capacity-scaling interpretation. No parameter-matched control baseline is provided (e.g., expanding classification heads or appending equivalent dense layers to the backbone without adapters). Without this, it is impossible to attribute the gains specifically to the proposed invariant/task-specific architectural separation rather than simply having more parameters. This matters because the paper's theoretical framing (eliminating the stability-plasticity dilemma via feature separation) is distinct from the pragmatic result (adapters help). The claim that inter-task differences are "the primary driver of catastrophic forgetting" (line 31) and that the method resolves this via architectural separation is not experimentally validated.

- **Backbone regularization (Eq. 2) is underspecified, creating a reproducibility gap.** The backbone distillation loss is defined as:
  R_φ^t = Σ_{t'=1}^{t-1} M(Linear_{d×c}(φ^{t'}(x)), Linear_{d×c}(φ^t(x)))
  with c chosen as "the number of classes of each task" (§3.2.1, line 117). The paper does not specify whether Linear_{d×c} is: (a) the task-specific classification head from task t', (b) the current task-t head applied to both features, or (c) a separate shared projection layer. In Task-IL's multi-head setting, if heads are task-specific, cross-task distillation requires head alignment or masking that is not described. If a shared projection is used, this is not stated. The phrase "implicitly a direct distillation on backbones" (line 117) does not clarify the implementation. This ambiguity is significant enough that the method cannot be faithfully reproduced from the text alone and affects the validity of the backbone stability claim.

### Minor

- **Class-IL results are buried in the appendix, insufficiently supporting broad claims.** The abstract and conclusion claim to "eliminate the stability-plasticity dilemma" and "effectively address" it broadly. Yet all main-text results (Figures 3–5, Table 1) are exclusively Task-IL, which uses a task-ID oracle and avoids cross-task output interference—the hardest and most practically relevant form of the dilemma. Class-IL results are deferred to Appendix B (line 163). While the paper does acknowledge Class-IL exists and provides appendix results, the strong claims about solving the dilemma should be grounded by at least prominently presenting the harder protocol's evidence. As-is, readers assessing the main text alone see no evidence for the harder setting. The phrasing "eliminating the stability-plasticity dilemma" (Abstract, line 15; §5, line 245) is also hyperbolic—no IL method truly eliminates this trade-off.

- **ImageNet results are compromised by experimental constraints and should be interpreted cautiously.** Section 4.2 (lines 197–198) acknowledges that CIFAR-100 hyperparameters were transferred without retuning and training was limited to 50 epochs. Table 1 shows mixed results: MAS-A and LwF-A improve over baselines, but EWC-A and LwM-A actually *underperform* their non-adapter counterparts on later tasks (e.g., EWC-A: 65.3% vs. EWC: 60.8%, but LwM-A: 56.9% vs. LwM: 58.0% by Task 10). The paper claims "non-trivial performance improvement" but this is partly driven by the baselines being under-trained. The 50-epoch constraint is acknowledged but undermines the ImageNet evidence as a scaling argument.

### Trivial

- **No layer-wise representation analysis to support the "spatial feature separation" hypothesis.** The paper asserts that adapters capture task-specific information "in the layers closer to the output" while "squeezing task-invariant knowledge into layers nearer the input" (§3.1, line 31; §3.2, line 79). This is presented as a fact but no CKA, CCA, or probing analysis is provided to verify the spatial distribution of invariant vs. task-specific information across backbone layers.

## Nice-to-Haves

- Adding a parameter-matched baseline (e.g., expanded linear heads of equivalent parameter count) would strengthen the attribution claim, though the practical utility of adapters remains regardless.
- Including Class-IL accuracy curves in the main text would better ground the "eliminating the stability-plasticity dilemma" narrative.
- Providing confidence intervals or standard deviations across the 10 seeds alongside the mean accuracy curves would give readers a sense of result variance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **~"Unstated appendix" for Class-IL results.~** The harsh critic claimed Class-IL results were relegated to an "unstated appendix." The paper in fact explicitly states on line 163: "while results for class-IL are included in Appendix B." The concern about Class-IL being outside the main text still holds, but the "unstated" characterization is a misread.

- **~Figure 1 merely confirms ordering sensitivity and validating need for task-specific modeling is a logical leap.~** This is a presentation-level interpretation critique. Figure 1 does show that coarse-grained orderings degrade performance—this is a reasonable motivational observation for the paper's direction, not an unsupported claim.

- **~CIFAR-100 validation 10/90 and ImageNet untuned tuning is a "computational inconsistency."~** This is a weak complaint. The paper honestly states resource constraints for ImageNet; the 10/90 split is standard for CIFAR-100 per cited baselines (Masana et al., 2022).

- **~"Comparison with modern methods (DualNet, iTAML, TAMiL) is cursory."~** Table 2 provides direct numerical comparisons. While the comparisons are limited in scope, they are not "cursory"—they show concrete +1% gains over DualNet/iTAML and outperform TAMiL.

- **~"Abstract claims that inter-task differences are the primary driver of catastrophic forgetting is ordering sensitivity, a known property."~** The paper's observation about ordering sensitivity is indeed known, but using it as motivation for the adapter approach is standard practice in IL papers, not an overclaim in itself.

## Novel Insights

The paper's insight—re-purposing parameter-efficient adapters from NLP/Vision for incremental learning via co-training rather than frozen-backbone fine-tuning—is a clean and practically useful contribution, though it does not rise to the level of fundamental novelty given that adapter-based IL (e.g., DualPrompt, TAMiL, Liang & Li 2024) is already an active line. The genuinely useful contribution is the *method-agnostic integration scheme*: the same lightweight adapter additions consistently improve classical regularization baselines (EWC, MAS, LwF) without needing custom algorithm-specific modifications, which is practically valuable for the IL community. However, the paper oversells this as mechanistic resolution of the stability-plasticity dilemma when the evidence primarily supports a practical capacity-and-isolation benefit.

## Suggestions

- **Run a capacity-controlled ablation.** Compare the adapter approach against a non-adapter baseline where equivalent parameters are added to the backbone (e.g., wider classification head or additional dense layers) to disentangle the "added capacity" effect from the proposed architectural separation.
- **Clarify the Linear_{d×c} projection in Eq. 2.** Specify whether it is a shared layer, the current task's head, or the previous task's head, and describe how cross-task gradient flow is handled in the multi-head Task-IL setup. This is critical for reproducibility.
- **Tone down claims.** Replace phrases like "eliminating the stability-plasticity dilemma" with "mitigating" or "improving." Claims should be proportional to evidence presented.
- **Include Class-IL results (or a subset) in the main text.** Even showing one Class-IL accuracy curve alongside the Task-IL results would strengthen the generality claim or honestly limit it if performance drops.
- **Add a layer-wise analysis** (e.g., CKA or centered kernel alignment of backbone features across tasks) to verify the claimed spatial separation of invariant and task-specific information.

## Score and Decision

Compared against calibration anchors:
- **SD-LoRA (avg score 7.5, Accept-Oral):** Similar in proposing a straightforward architectural modification for IL, but SD-LoRA evaluated on Class-IL with theoretical analysis and broader benchmarks, placing it clearly above this paper.
- **MISA (avg score 6.5, Accept-Poster):** Also a plug-in module that consistently improves multiple baselines—similar contribution style. MISA included Class-IL evaluation and had clearer methodology, scoring higher.
- **MetaAdapter (avg score 5.40, Reject):** Similar profile—strong empirical results but missing baseline comparisons and overclaiming mechanistic novelty. This paper is slightly more cohesive in its method-agnostic integration but has the equivalent capacity-confounding gap. Comparable level.
- **Amphibian (avg score 3.67, Reject):** Only evaluated on Task-IL with unclear methodology and no Class-IL evidence. The paper under review is clearly above this—it has cleaner experiments, honest reporting of limitations, and consistent gains.
- **Borderline papers around 5.0** (e.g., APER 4.75, Duct 5.0): Similar positioning—useful contributions but limited by scope or methodological gaps.

This paper is strongest at ~5.5. It offers genuine practical utility (consistent baseline improvements, clean integration paradigm) that distinguishes it from the clearly reject tier (≤4). But the uncontrolled capacity confounding, underspecified regularization, and overclaiming about the stability-plasticity dilemma place it below the accept tier (≥6.5). The empirical results are real and the contribution is useful, but the core mechanistic claim is not experimentally validated.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>