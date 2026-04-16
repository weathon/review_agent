Let me verify by reading the specific section about "theoretical framework" claim from the paper content already provided:

From the paper in Section 1.1: "We first theoretically establish the scenarios under which such assumptions are violated."

And Section 2.1 is titled "THEORETICAL FRAMEWORK" but contains only notation and the stated assumptions — no theorem, lemma, or proof deriving conditions for violation.

Now let me also check the closely related paper (Pev2ufTzMv) — the predecessor paper "Why Sanity Check for Saliency Metrics Fails?" which was rejected with scores 1, 3, 8, 3. This is very closely related work, and the current paper appears to be an extended/different version of the same research direction.

Now I have enough information to write the final review. Let me consolidate my assessment:

**Key structural problems identified:**

1. **Misattributed assumptions**: The paper's P1 ("every pixel perturbation drops probability") and P2 ("rank invariance across perturbation types") are significantly stronger than what fidelity metrics actually require. AOPC/AD/IC are typically computed with a *single fixed perturbation type*, not requiring cross-perturbation consistency. The "cross-perturbation invariance" assumption (P2/Equation 5) is not an assumption made by any cited metric. This undermines the central claim that "metrics are inconsistent."

2. **No fidelity metrics computed**: The paper claims fidelity metrics are "inconsistent and unreliable" but never actually computes AOPC, AD%, IC%, W%, or faithfulness. It only computes DROP and PSim. The logical leap from DROP/PSim < 1 to "metrics fail sanity checks" is not justified without showing that actual metric outcomes differ substantially.

3. **Overclaimed theoretical contribution**: Section 1.1 says "We first theoretically establish the scenarios under which such assumptions are violated" but Section 2.1 only defines notation and states desired properties — there are no theorems, lemmas, or proofs about conditions for violation.

4. **Relation to predecessor work**: The closely related prior paper ("Why Sanity Check for Saliency Metrics Fails?") was rejected. The current paper appears to be an extension (9 perturbations vs. just Gaussian blur, segment-wise in addition to pixel-wise, adversarial models), but the fundamental issues flagged in the prior review persist.

**What the paper does well:**

- Systematic empirical sweep of 9 perturbation types across multiple models/datasets
- DROP and PSim are simple, well-defined diagnostics
- The practical recommendation (specify perturbation type, use conformity checks) is reasonable

**Calibration:**

- The closely related predecessor (Pev2ufTzMv) scored 1,3,8,3 and was rejected. The current paper is stronger in empirical breadth but shares the same structural issue of overclaiming.
- The GNN fidelity paper (up6hr4hIQHm) which actually proposed a fix to fidelity metrics got 8,8,5,3 and was accepted as poster. That paper had theoretical grounding AND proposed solutions.
- Papers with similar "identifies problem but overclaims solution without fixing it" patterns (like the benchmarking deletion metrics paper bXeSwrVgjN with 8,3,5,8 but rejected) tend to fall in the 3-5 range.
- The "understanding parameter saliency" paper (3J7foqnJkA) that assumes something then shows it doesn't hold scored 5,6,6 — similar but was rejected.

Given the structural misalignment between claims and evidence, and this being essentially an extended version of a previously rejected paper with the same core issues, I'd place this at around 3.5.

Let me finalize the review.</think>Now I have sufficient calibration information. Let me compose the final review.

## Summary

This paper examines why perturbation-based fidelity metrics (AOPC, AD%, IC%, W%, faithfulness) are inconsistent across perturbation techniques. The authors formalize two assumptions they argue underlie these metrics—[P1] that perturbing any pixel drops prediction probability, and [P2] that importance rankings are invariant across perturbation types—and propose two conformity measures (DROP and PSim) to quantify violations of these assumptions. Through experiments with 9 perturbation types, 5 models, and 3 datasets, they find that both assumptions frequently fail, and conclude that perturbation-based fidelity metrics are "inconsistent and unreliable."

## Strengths

- **Systematic empirical sweep**: The evaluation covers 9 perturbation types, 5 models (including 2 adversarially trained ones), 3 datasets, and both pixel-wise and segment-wise perturbation schemes, involving approximately 75 million prediction calls. This is genuinely comprehensive in scope.

- **Clear formalization of assumptions**: Articulating [P1] and [P2] explicitly provides a structured analytical framework for discussing why inconsistency arises, which is an improvement over merely observing it.

- **Simple, interpretable diagnostic measures**: DROP and PSim are straightforward to compute and interpret. The finding that DROP ≈ 0.5–0.6 and PSim ≈ 0.3–0.6, along with the observation that Gaussian blur yields relatively higher conformity, provides useful practical guidance.

## Weaknesses

### Major:

- **The central assumptions [P1] and [P2] do not accurately represent the requirements of existing fidelity metrics.** The paper constructs two assumptions and then shows they are violated, but these assumptions are significantly stronger than what the cited metrics require. [P1] as stated (Equation 2: p₀ > pᵢ^φ for *all* pixels i and *all* perturbations φ) demands that *every single* pixel perturbation must decrease prediction probability. Most fidelity metrics only require that *important* features, when perturbed, produce larger performance drops than *unimportant ones* on average—they do not assert a strict per-pixel monotonic decrease. [P2] as stated (Equation 5: rbo across different perturbation types ≈ 1) demands cross-perturbation rank invariance, but the cited metrics (AOPC, AD%, IC%, etc.) each adopt a *single* perturbation scheme and never require that rankings agree across different perturbation operators with different semantics. The paper's headline conclusion that "fidelity metrics are inconsistent and unreliable" therefore does not follow from its own evidence: it invalidates assumptions that the metrics do not actually make.

- **No fidelity metric is actually computed or shown to be inconsistent.** The paper claims that metrics like AOPC, AD%, IC%, W%, and faithfulness are unreliable, but the empirical evaluation only reports DROP and PSim scores. At no point does the paper compute any of these fidelity metrics for any saliency map under any perturbation, nor does it show that two saliency methods would be ranked oppositely by the same metric under different perturbation choices. Without this, the logical chain from "model responses vary across perturbations" → "metrics are inconsistent" is asserted but not demonstrated. A straightforward experiment—computing AOPC for several saliency methods under different perturbations and showing ranking reversals—would substantially strengthen the argument but is absent.

- **The claimed "theoretical framework" does not deliver theoretical results.** Section 1.1 states "We first theoretically establish the scenarios under which such assumptions are violated," but Section 2.1 ("Theoretical Framework") only introduces notation and restates the assumptions as equations. There are no theorems, lemmas, or proofs characterizing conditions under which violations occur. This is a definitional section, not a theoretical contribution.

### Minor:

- **The 50-random-pixel sampling strategy does not match how fidelity metrics are used in practice.** Fidelity metrics such as AOPC perturb pixels *cumulatively in order of saliency importance*, not 50 randomly selected pixels one by one. The paper acknowledges PIR can be computed "one by one or cumulatively" (Section 1) but restricts its analysis to single-pixel perturbations of random locations. The argument that "a subset of a ranked order list maintains ranking" (invoked from Appendix S2) does not justify sampling random locations: random subsets do not preserve any unknown ranking, and no prior ranking exists in this analysis because the PIR is being constructed via the perturbations themselves.

- **Interpretation of DROP ≈ 0.5 as invalidating [P1] conflates model insensitivity with assumption violation.** Only a small fraction of pixels in an image are truly important for classification. That approximately half of *randomly selected* pixels do not decrease the top-class probability when perturbed is expected behavior for a well-functioning classifier, not evidence that the model violates a core assumption.

- **The paper equivocates between model behavior, perturbation choice, and metric design.** The conclusion states "unreliability arises as a property of the DL models with respect to perturbations," but also that "metrics... would be rendered unreliable and fail the sanity checks." These are distinct claims: (a) models respond differently to different perturbations, (b) perturbation choice is an important hyperparameter for metrics, and (c) the metrics themselves are fundamentally flawed. The evidence supports (a) and weakly (b), but the paper's language repeatedly asserts (c).

### Trivial:

- Algorithm 1 contains a confusion where the count of non-negative δP values is appended into the same list as per-pixel differences (line 251), though this appears to be a presentation issue rather than affecting the computed metrics. The return statement also references incorrect equation numbers (6) and (7) swapped for DROP/PSim definitions.

## Nice-to-Haves

- Computing actual fidelity metrics (AOPC, AD%, etc.) for multiple saliency methods under different perturbations and showing whether metric rankings change would directly validate or refute the paper's central claim.
- Analyzing whether DROP/PSim can serve as *predictors* of fidelity metric disagreement—i.e., showing that low conformity scores predict when fidelity metrics will disagree—would validate their utility as preconditions.
- Including Vision Transformers, NLP models, or tabular data to assess whether findings generalize beyond CNN-based image classifiers.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"The paper doesn't cite model/dataset availability"** — The paper cites standard, publicly available datasets (Imagenette, Oxford-IIIT Pet, PASCAL VOC 2007) and well-known pretrained models. Removal: doubting availability is not grounds for criticism.

- **"Imagenette is a typo for ImageNet"** — Imagenette is a real, well-known subset of ImageNet distributed via tensorflow.org. This is not an error.

- **"Gaussian Blur significantly reduces model accuracy, so recommending it is contradictory"** — The paper recommends Gaussian blur as *relatively* more consistent across perturbation variants, not as semantically benign. The recommendation is comparative, not absolute. Removal: this misreads the claim.

- **"Missing related works"** — Per review rules, I should not flag missing citations.

- **Formatting and notation nitpicks** (e.g., ℝ defined as "ranks" but then used as "ranked list of pixels") — The notation could be clearer but this does not affect the substance.

- **"Writing is unclear"** — While the presentation could be tighter, this is too generic and stylistic to qualify as a substantive weakness for this venue.

## Novel Insights

The most novel empirical finding is that Gaussian blur perturbations yield relatively higher DROP and PSim conformity scores compared to inpainting- or constant-value-based perturbations, suggesting that smoother perturbations produce more consistent model responses. However, this observation is presented without mechanistic explanation (why blur produces more internally consistent rankings), and the practical implication (whether using blur leads to more reliable fidelity evaluations) is not demonstrated. The distinction the paper implicitly surfaces—between *model sensitivity to perturbations* (a model property) and *metric inconsistency* (a metric property)—is genuinely important but the paper conflates rather than clarifies it.

## Suggestions

- **Reframe the claims**: Replace "fidelity metrics are inconsistent and unreliable" with "model responses to local perturbations vary substantially across perturbation types, and this variation undermines the reliability of any fidelity metric that does not specify its perturbation scheme." This is well-supported and important.
- **Compute actual fidelity metrics**: Show at least one case where AOPC or AD% ranks two saliency methods differently under two reasonable perturbation choices. This would make the paper's core argument concrete rather than inferential.
- **Provide the theoretical characterization promised**: Either add actual theorems about when/why P1/P2 fail, or remove the claim of "theoretically establishing the scenarios" from Section 1.1.

## Score and Decision

**Calibration**: The closely related predecessor paper "Why Sanity Check for Saliency Metrics Fails?" (Pev2ufTzMv) was rejected with scores 1/3/8/3. The current paper is a substantially extended version (9 perturbations vs. just Gaussian blur, segment-wise analysis, adversarial models), which improves empirical breadth but retains the same core structural problem: overclaiming that metrics are unreliable without actually computing them or correctly characterizing their assumptions. Papers that correctly identify problems with fidelity metrics *and* propose principled fixes (e.g., "Towards Robust Fidelity for GNNs" at up6hr4hIQHm, scores 8/8/5/3, accepted poster) score much higher. Papers with novel framing but weak empirical validation of their central claim (e.g., "Understanding Parameter Saliency" at 3J7foqnJkA, scores 5/6/6, rejected) score similarly or slightly higher. This paper's empirical work is thorough, but the argument structure is undermined by the misattribution of assumptions and the absence of fidelity metric computation. I place it below papers with correct argumentative foundations and comparable or slightly above the prior rejected version.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>