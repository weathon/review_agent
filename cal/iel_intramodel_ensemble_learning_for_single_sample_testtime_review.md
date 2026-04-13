=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary
The paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method for the restrictive batch-size-1 setting. Given multiple pretrained classifiers, IEL selects, for each test sample, the model with the highest confidence among those predicting the majority-voted class, and uses that soft prediction as a target to update all ensemble members online. Experiments on CIFAR-10C, CIFAR-100C, and ImageNet-C show that this procedure can substantially improve a static ensemble and even improve individual constituent models on many corruption types.

## Strengths
- **Targets a genuinely restrictive and practically meaningful setting:** the method is explicitly designed for single-sample test-time adaptation, where batch-statistics-based methods are less applicable. The paper is clear that it uses batch size 1 and freezes BN parameters “to ensure that we are not benefiting from updating batch normalization statistics on new data.”
- **The dynamic teacher-selection rule is simple and specific:** rather than a fixed teacher or simple averaging, IEL chooses the most confident model among those supporting the majority class. This is a concrete mechanism that is easy to implement across heterogeneous models and is more interesting than a static voting ensemble.
- **The paper demonstrates that adaptation can improve not only the ensemble output but also individual members:** e.g., Tables 1–3 show large gains for specific constituent models, especially on blur/compression-type corruptions, which is more distinctive than merely improving the ensemble vote.
- **Empirical gains on ImageNet-C are substantial relative to the static ensemble baseline:** Table 3 shows consistent positive improvements across all listed corruption types, often with double-digit gains for both the ensemble and individual models.
- **The paper is unusually candid about failure modes:** it explicitly notes catastrophic forgetting, acknowledges that performance can later decline, and states below Algorithm 1 that “in some experiments we found that the average model accuracy of the IEL ensemble reduced below the static model accuracy by the final epoch.”

## Weaknesses

### Major:
- **The evaluation protocol does not fully support the paper’s practical TTA claims because results are reported as the best epoch chosen with labels after adaptation.**  
  This concern is directly supported by the paper. Tables 1–3 are explicitly titled “Highest accuracy improvements (%) over all epochs,” and Section 3.1 states “No termination is required,” while also acknowledging that accuracy can later diminish. In a deployable test-time adaptation setting, the method needs a principled stopping rule or a fixed online evaluation protocol; otherwise, best-over-epochs reporting gives an optimistic upper bound rather than realized inference-time performance. This is the most important empirical issue because it affects the central claim of practical adaptation during inference.

- **Baseline comparisons are too weak for the paper’s positioning as a TTA method.**  
  In practice, the experiments compare only against static, non-adapted models/ensembles. The paper discusses TENT, EATA, ROID, and CoTTA in the introduction/related work, but does not evaluate against them or even against simpler adaptation baselines that would isolate the contribution of the proposed teacher-selection objective. As a result, the evidence supports the narrower claim that IEL can improve over a frozen ensemble under this protocol, but not the stronger claim that it advances the state of the art in single-sample or online TTA.

- **The experimental setting is narrower than the paper’s broader framing suggests: adaptation is done separately per corruption type with model resets, under stationary shifts.**  
  Section 4 states that samples from a “single corruption type” are streamed, and after finishing one corruption, the models are reset and adaptation starts “from scratch on the new corruption type.” This is a legitimate experimental setting, and the paper does mention stationary shifts, but many of the broader claims around TTA and continual learning read more generally than what is actually validated. Because nonstationary or mixed-shift streams are where adaptation methods often fail, the current results should be interpreted as evidence for stationary, corruption-specific adaptation rather than general online robustness.

- **The method’s conceptual framing around “diversity as a new optimization signal” is imprecise and somewhat misleading relative to the actual objective.**  
  The paper itself says, “we minimize the diversity of the ensemble (we force models to agree with each other),” while the contributions claim it “proposes diversity as a new optimization signal.” The mechanism is better described as consensus-forcing or mutual distillation using pre-existing ensemble disagreement only to choose a teacher. This does not invalidate the empirical idea, but it weakens the conceptual clarity of the contribution and leaves the justification underdeveloped: why should reducing disagreement after teacher selection improve adaptation rather than amplify majority mistakes?

- **The paper does not sufficiently disentangle IEL from entropy minimization or other simpler pseudo-labeling effects.**  
  The authors explicitly note that their chosen target is likely low-entropy and that IEL minimizes entropy “as a side effect,” with Figure 1 showing correlation between IEL loss and entropy. But the paper does not compare against straightforward alternatives such as entropy minimization on each member, self-training from the majority pseudo-label, or using an averaged soft target instead of the selected majority-confident teacher. Without such ablations, it remains unclear how much of the gain comes from the specific intra-ensemble learning mechanism versus generic confidence sharpening.

- **Failure modes are real and nontrivial, especially on noise corruptions, but are not analyzed in depth.**  
  The CIFAR tables show sizable degradations for several corruption types, including Gaussian Noise, Shot Noise, and Impulse Noise, and the text acknowledges catastrophic forgetting. This honesty is appreciated, but the paper does not probe why these failures occur, whether they are tied to incorrect majority pseudo-labels, or whether simple safeguards could mitigate them. Since the method’s central risk is propagating confident ensemble mistakes, this deserves more direct analysis.

### Minor
- **Absolute accuracies are not presented in the main result tables, only improvements over the static baseline.**  
  This makes it harder to assess practical significance and compare performance across datasets/corruptions. Improvement-only reporting is useful, but it is not enough on its own for a methods paper.

- **Computational cost is acknowledged but not quantified.**  
  IEL requires forward and backward passes through multiple models at test time for every sample. The limitations section notes that the approach is more computationally expensive, but there is no runtime, FLOPs, or memory analysis to help judge whether the observed gains justify the overhead.

- **Some key design choices are left underexplained or unablated.**  
  Examples include the near-zero regularization constant (\(\alpha = 10e^{-11}\)), the choice to update the selected teacher as well as the students, and the sensitivity to ensemble size / architecture diversity. These do not make the paper unsound, but they leave the method under-characterized.

- **The algorithm is underspecified for some edge cases such as ties in majority vote or ties in confidence within the majority class.**  
  Since teacher selection is central to the method, these details should be made explicit.

### Trivial
- None.

## Nice-to-Haves
- Evaluate a more realistic nonstationary or mixed-corruption stream without resetting model weights between corruptions.
- Add ablations on ensemble size, same-architecture vs mixed-architecture ensembles, and alternate target constructions (majority hard label, average soft target, KL instead of cross-entropy).
- Report fixed-epoch results or use an unsupervised stopping criterion alongside best-epoch numbers.
- Include a small set of learning curves on both successful and failing corruptions, plus simple diagnostics for when the majority vote is wrong.
- Quantify test-time overhead with wall-clock/runtime and memory measurements.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the paper is “not true one-sample deployment” because it adapts over many samples.**  
  Removed as a criticism of the single-sample setting itself. The paper’s claim is that each *update step* requires only one sample, not that adaptation consists of exactly one total sample. The valid issue is instead the best-epoch evaluation/stopping-rule problem.

- **Criticism that “all trainable parameters” is false because BN parameters are frozen.**  
  Removed as a strawman. The paper clearly states BN parameters are frozen during testing; this is an implementation choice, not a contradiction that undermines the method.

- **Reproducibility-style complaint about missing low-level implementation details.**  
  Removed because the paper provides the core algorithm, model sets, datasets, and optimization setup, and this line of critique would be a routine nitpick rather than a substantive weakness here.

- **Strong novelty attacks based on unspecified external related work.**  
  Removed because they depend on outside literature claims that cannot be verified from the submission alone. The retained novelty concern is limited to the paper’s own framing versus its actual mechanism.

## Novel Insights
The most interesting signal in the submission is not just that an ensemble can outperform a static ensemble under shift, but that **mutual online adaptation can materially improve the constituent models themselves**, especially on structured corruptions like blur and compression, while failing much more on stochastic noise corruptions. That asymmetry suggests IEL may be leveraging cross-model agreement to recover stable structure under semantically preserving shifts, but becomes brittle when the majority pseudo-label is itself unreliable. Framed this way, the key research question is less “does reducing diversity help?” and more “when does consensus among heterogeneous pretrained models provide a trustworthy adaptation target under shift?”

## Suggestions
- Replace peak-over-epochs reporting with a deployable protocol: fixed budget, online cumulative accuracy, or an unsupervised stopping rule.
- Add direct TTA baselines, at minimum a strong single-model baseline and a simple ensemble-based pseudo-label/entropy baseline, to isolate what IEL contributes.
- Reframe the contribution more precisely as dynamic mutual distillation / consensus-based adaptation rather than “diversity optimization.”
- Analyze failure cases on noise corruptions and test simple safeguards, such as confidence-thresholded updates, not updating the selected teacher, or partial weight restoration.
- Report absolute accuracies and test-time cost alongside improvement numbers.
- Add ablations on ensemble size and target choice to verify that the gains are genuinely due to the proposed teacher-selection mechanism.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 1.0]
Average score: 2.5
Binary outcome: Reject
