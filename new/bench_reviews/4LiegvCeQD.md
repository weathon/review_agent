## Summary
This paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method for the restrictive single-sample setting. Given several independently pretrained classifiers, IEL selects the model with the highest confidence among those predicting the majority-voted class and uses its softmax as a pseudo-target to update all models online. Experiments on CIFAR-10-C, CIFAR-100-C, and ImageNet-C show that this mutual adaptation can substantially improve over a static ensemble on many corruption types, especially blur-related shifts.

## Strengths
- The paper studies a meaningful and relatively underexplored setting: adapting **multiple pretrained models jointly** at test time, with **batch size 1** and without relying on BN-statistics updates. This is a genuine departure from standard single-model or batch-statistics-heavy TTA setups.
- The core mechanism is simple and easy to understand: use the highest-confidence model among the majority-vote models as a dynamic teacher, then adapt the ensemble online through cross-entropy. This is an intuitively plausible design and easy to implement.
- The empirical coverage is fairly broad across three standard corruption benchmarks: CIFAR-10-C, CIFAR-100-C, and ImageNet-C, with corruption-wise breakdowns rather than just aggregate numbers.
- The paper makes a good experimental choice by freezing BN parameters and using batch size 1, which reduces a common confound in TTA and supports the claim that gains are coming from the proposed update rule rather than BN-statistics adaptation.
- On many corruption types, especially blur and motion-related corruptions, the reported gains over the static ensemble are large. For example, the ImageNet-C table reports consistent positive improvements across all listed corruptions, and the CIFAR tables show strong gains on several blur-like shifts.
- The paper is reasonably candid about limitations such as catastrophic forgetting and computational cost, rather than pretending the method is uniformly stable.

## Weaknesses

###: Fatal
- **The main reported results use oracle-over-time “highest accuracy over all epochs,” which is not a deployable TTA evaluation.** Tables 1–3 are explicitly titled “Highest accuracy improvements (%) over all epochs,” while Section 3.1 also states: “**We do not include a termination step in our experiments**.” The paper further acknowledges that performance can deteriorate after the peak: “**In some experiments we found that the average model accuracy of the IEL ensemble reduced below the static model accuracy by the final epoch.**” This means the headline gains rely on retrospective label-based epoch selection unavailable at test time. For a TTA paper, this substantially weakens the central practical claim.

### Major:
- **The experimental protocol is weaker than the paper’s practical framing of online single-sample inference.** Although each update uses one sample, the actual evaluation repeatedly adapts for several epochs over a large stationary set from one corruption type, then evaluates on a held-out split from the same corruption. Section 4 says the data are split into “**90% split of tuning set samples used for IEL and 10% split of evaluation set samples**,” and the method is run for “**several epochs on the corruption types**,” resetting weights for each corruption. This is closer to stationary unsupervised adaptation within a known target domain than to the paper’s broader rhetoric of practical one-sample-at-a-time inference in dynamic real-world streams.
- **There is no comparison against actual TTA baselines.** The method is framed as a TTA contribution, and the paper discusses TENT, EATA, CoTTA, and ROID in Related Work, but experiments compare only against the static ensemble / static members. As a result, the paper shows that IEL improves over doing nothing, but does not establish how competitive it is relative to existing TTA approaches, including simple per-model test-time updating baselines.
- **The conceptual framing around “diversity as a new optimization signal” is internally inconsistent.** The method does not optimize for diversity; it explicitly reduces disagreement. The paper itself says in Section 1/2 that “**we minimize the diversity of the ensemble (we force models to agree with each other)**,” yet the contributions section claims “**This work proposes diversity as a new optimization signal**.” The actual mechanism is more accurately described as dynamic self-/mutual-distillation from a majority-consistent confident member. This mismatch matters because it overstates the conceptual novelty.
- **The method’s stability issues are real and not deeply analyzed.** The paper openly reports severe failures on some corruptions, especially noise corruptions in CIFAR-10-C and CIFAR-100-C, with large negative changes in several table entries. The paper attributes this to catastrophic forgetting risk, but does not provide a substantive analysis of when the majority-vote teacher is wrong, why noise corruptions are especially harmful, or whether simple safeguards would help.
- **Key ablations are missing for the central design choices.** Since the method hinges on a specific pseudo-target construction, it is important to compare against alternatives such as: hard majority label, averaged ensemble softmax, most-confident model without majority filtering, excluding the teacher model from the loss, or using KL divergence rather than cross-entropy. The paper itself notes that CE includes an entropy-minimization effect and even suggests KL as future work, which makes this omission more consequential.

### Minor
- **The computational cost is substantial and under-quantified.** The paper acknowledges that IEL is heavier than single-model TTA, which is true: it requires forward/backward updates through all models per sample. But there is no runtime, memory, or throughput analysis, so the practical tradeoff is hard to assess.
- **The stationary single-corruption evaluation and weight reset between corruption types limit realism.** This setup is useful for controlled study, but it does not test mixed or non-stationary shifts where error accumulation and forgetting would likely be more severe.
- **The writing sometimes overclaims.** Phrases like “significant and consistent improvement” are too strong given the oracle epoch selection and the clear failures on some corruption types in CIFAR.
- **Method details that directly affect behavior are insufficiently probed.** In particular, the selected teacher is still included in the CE loss, so it is pushed toward lower entropy on its own prediction rather than treated as fixed. The paper acknowledges this could overwrite strong knowledge, but does not study the effect.
- **Absolute accuracies and stability trajectories are not emphasized enough.** Reporting only improvement over baseline and only the best epoch obscures whether gains are volatile or modest in absolute terms.

### Trivial
- Tie handling in majority voting / teacher selection is not specified.
- The assumption that distinct weights imply distinct architectures is oddly stated and unnecessary.

## Nice-to-Haves
- Evaluate on mixed or sequentially changing corruption streams, not just one corruption at a time with resets.
- Add order-sensitivity / seed-sensitivity analysis, since online adaptation can be path-dependent.
- Measure ensemble disagreement before and after adaptation to support the claimed diversity/agreement narrative more directly.
- Provide a simple label-free stopping heuristic and report its performance.
- Include compute-efficiency metrics so readers can judge whether the gains justify multi-model online backpropagation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Pure formatting/style criticisms** from the reviews were removed per instruction.
- **Generic reproducibility nitpicks** about optimizer minutiae or every implementation detail were removed as standalone weaknesses; while some missing details exist, they are not central relative to the larger evaluation issues.
- **“Missing related work” style complaints** were not included, per instruction.
- **Claims questioning whether updating all parameters is itself a virtue** were weakened; the paper does make that design choice explicit, and the real issue is not that it updates all parameters, but that this increases instability/cost and is not adequately justified empirically.
- **Any criticism implying cited methods/models/benchmarks may not exist or may not be verifiable** was excluded by rule.

## Novel Insights
The strongest synthesis here is that the paper is not primarily undermined by the basic idea—which is simple but plausible—but by a mismatch between **what is claimed**, **what is optimized**, and **how it is evaluated**. IEL is best understood not as “diversity-based optimization,” but as a form of **online mutual distillation under a dynamic, majority-constrained teacher choice**. Under that reading, the positive results on blur-like corruptions make sense: when the ensemble remains reasonably calibrated, consensus distillation can sharpen all members. But the same mechanism becomes fragile under heavy noise, where majority-consistent confidence is a poor proxy for correctness. This suggests the core opportunity is real, but the current paper does not yet establish a reliable TTA method under realistic deployment conditions.

## Suggestions
- Replace the headline best-epoch tables with a realistic protocol: fixed epoch budget, final-epoch reporting, or preferably a label-free stopping/model-selection rule.
- Add direct comparisons to strong TTA baselines, including at least simple per-model test-time updating baselines in the same single-sample setting.
- Reframe the method more accurately as dynamic mutual distillation / agreement-based adaptation rather than “diversity optimization.”
- Add ablations on pseudo-target construction: majority hard label vs. majority-soft teacher vs. averaged ensemble softmax vs. most-confident model.
- Analyze failure modes by measuring how often the majority-vote teacher is wrong on different corruption types, especially Gaussian/shot/impulse noise.
- Test teacher stop-gradient / excluding teacher from the loss / KL-divergence variants to see whether instability comes from self-entropy minimization.
- Include mixed-corruption or continual-shift evaluation without resetting weights between corruption types.
- Report compute overhead and memory usage relative to simpler TTA baselines.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Moderate. The ensemble-mutual-adaptation angle is interesting, but the core mechanism is a fairly straightforward distillation/agreement objective rather than a fundamentally new learning principle.  
- **Importance of the research question:** Good. Single-sample TTA and multi-model adaptation are worthwhile problems.  
- **Whether the claims are well supported:** Weak-to-moderate. The strongest claims are undermined by oracle best-epoch reporting and lack of TTA baseline comparisons.  
- **Soundness of experiments:** Mixed. Broad benchmark coverage is a plus, but the evaluation protocol is not aligned with deployable TTA.  
- **Clarity of writing:** Generally readable, though conceptually imprecise in important places.  
- **Value to the community:** Potentially useful idea, but not yet convincing enough as a TTA paper in its current form.

**Calibration against human-reviewed papers:**  
- Compared with **4wk2eOKGvh (Test-Time Ensemble)**, which was accepted with scores 6/8/6/6, this submission is weaker: both papers study ensemble-style TTA, but TTE appears to have stronger comparative evaluation and a more deployable story, whereas this paper’s core results rely on oracle best-epoch selection.  
- Compared with **75PhjtbBdr (Multi-Label TTA with BEM)**, accepted with 8/6/6/5, this paper is also weaker on empirical support because that paper appears to benchmark against prior methods directly, while this one mostly beats a static baseline.  
- Compared with **PxL35zAxvT (Test Time Adaptation with Auxiliary Tasks)**, rejected with 5/6/3, the current paper is somewhat similar in having an interesting idea but evaluation gaps. I view IEL as slightly stronger in motivation and breadth of corruption-wise reporting, but still not strong enough because the practical evaluation issue is central.  
- Compared with **X4ATu1huMJ (Realistic Evaluation of TTA)**, rejected around 3s, this paper is stronger as an algorithmic contribution, but that paper is a useful anchor for why label-free model selection/stopping is not a minor detail in TTA.

Overall, this paper lands in the **4.5–5.5** range. I lean below threshold because the oracle-over-epochs issue directly compromises the main claim of practical test-time adaptation.

**Final score: 5.0 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>