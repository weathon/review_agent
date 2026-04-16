## Summary
This paper studies perturbation sensitivity in saliency-map fidelity evaluation. It proposes two diagnostic measures—DROP, measuring how often confidence decreases under perturbation, and PSim, measuring rank agreement of perturbation-induced pixel importance across perturbation operators—and evaluates them across nine perturbations, multiple CNNs, three datasets, and both pixel-wise and segment-wise schemes. The main empirical finding is that model responses and induced importance rankings vary substantially with perturbation choice, so perturbation-based fidelity scores should be interpreted cautiously and reported together with the perturbation operator.

## Strengths
- **Addresses an important and real problem.** Perturbation-based metrics are widely used for saliency evaluation, and the paper usefully highlights that perturbation choice materially changes the underlying model responses on which such metrics rely.
- **Broad empirical sweep.** The study spans 9 perturbation types, 2 perturbation schemes, 3 standard CNNs plus 2 adversarially trained ResNet50 variants, and 3 datasets. That breadth makes the descriptive perturbation-sensitivity result fairly convincing.
- **Useful descriptive diagnostics.** Even if the stronger claims are overstated, DROP and PSim are reasonable descriptive statistics for probing whether a model’s outputs and perturbation-induced rankings are stable across masking operators.
- **Some actionable empirical insight.** Across the tested perturbations, Gaussian blur variants appear relatively more stable than several replacement-based perturbations, which is practically useful as a comparative observation within the paper’s setup.
- **Generally clear practical message.** The recommendation to specify perturbation type when reporting fidelity scores is well supported by the experiments.

## Weaknesses

###: Fatal
- **The central claim is stronger than what the paper actually demonstrates.** The title, abstract, and conclusion claim to explain “why perturbation-based fidelity metrics are inconsistent,” but the experiments do **not** compute the cited fidelity metrics themselves (AOPC, AD%, IC%, W%, faithfulness) on saliency maps and show how DROP/PSim explain their disagreement. What is actually shown is that **raw model responses to perturbations are perturbation-dependent**. That is an important finding, but it is not the same as directly establishing that the cited metrics are inconsistent for the reasons claimed.

### Major:
- **The paper elevates cross-perturbation rank invariance to a necessary condition without sufficiently justifying that this is required by the criticized metrics.** In Section 2.1, Eq. (5) defines consistency via \( rbo(\mathfrak{R}(\phi), \mathfrak{R}(\psi)) \approx 1 \) across perturbation pairs. But the paper does not rigorously establish that perturbation-based fidelity metrics, in general, require near-invariant rankings across arbitrary perturbation operators rather than only behaving sensibly under a chosen perturbation scheme. Low PSim clearly demonstrates perturbation sensitivity, but the step from “sensitive across perturbations” to “therefore the metrics are inconsistent/unreliable” is insufficiently justified.
- **Assumption [P1] is too strong as stated, so violating it does not by itself invalidate perturbation-based metrics.** The paper formalizes [P1] as requiring \(p_0 > p_i^\phi\) for all perturbed pixels/segments and perturbations. That universal monotonicity condition is stronger than what many perturbation-based evaluations need in practice. Perturbing a region can increase confidence for reasons unrelated to explanation failure, including removal of distracting evidence or nonlinear interactions. Thus, Table 1 values of DROP around 0.5–0.6 do show non-monotone behavior, but they do not by themselves prove the target metrics are invalid.
- **There is a mismatch between the theoretical setup and many target metrics, which often use cumulative perturbations rather than individual perturbations.** The proposed analysis is built around single-pixel/single-segment perturbations and induced rankings, while several cited metrics are based on cumulative deletion/insertion trajectories or masked sets. The paper does not establish how instability at the individual-perturbation level translates to instability of cumulative metrics. This weakens the claimed explanatory link to the metrics named in the paper.
- **The random-pixel protocol weakens claims about pixel-importance rankings.** Section 4.1 states that the authors “select 50 random pixels” and justify this using the property that a subset of a ranked list maintains ranking. But the ranking under study is itself induced by perturbation responses; it is not a known fixed list being subsampled for evaluation. Random pixels may still reveal perturbation sensitivity, but this protocol is weaker than directly testing the salient regions or validating stability with different sample sizes. Since PSim is central evidence, this matters.

### Minor
- **The paper does not sufficiently analyze why Gaussian blur is relatively more stable.** This is one of the more practically relevant observations, yet it is mostly reported descriptively rather than explained mechanistically.
- **Some claims about model-level vs metric-level failure are under-argued.** The conclusion states that unreliability “arises as a property of the DL models with respect to perturbations.” The experiments do support perturbation sensitivity of model outputs, but the distinction between a property of the model, a property of the perturbation, and a property of the metric design is not cleanly disentangled.
- **There are some clarity/notation issues that affect technical precision.** For example, Eq. (9) is malformed as written (\(PSim = \frac{1}{|K|}\sum_{k=1}^K PSim\) lacks the image-level term), and the relationship among \(\mathfrak{R}\), \(\mathfrak{R}(\phi)\), and ordered pixels could be presented more rigorously.
- **Use of top-class probability is not fully discussed for all settings.** The paper repeatedly relies on the model’s top-class probability after perturbation, but this becomes less straightforward when perturbations alter the predicted class, and the treatment is especially ambiguous for VOC2007, which is multi-label.

### Trivial
- None.

## Nice-to-Haves
- Compute the actual target fidelity metrics (AOPC, AD%, IC%, W%, faithfulness) on the same models/datasets/perturbations and show whether low DROP/PSim predicts disagreement in those metrics.
- Evaluate DROP/PSim on top-k salient pixels from common explanation methods, not only random pixels.
- Add a controlled sanity-check setting where the assumptions are expected to hold approximately, to calibrate what “high” DROP/PSim should look like.
- Study sensitivity to the RBO persistence parameter and to the number of sampled pixels/segments.
- Clarify whether analyses are restricted to correctly classified examples and how the target probability is defined when the predicted class changes.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaints about missing inpainting perturbations or limited perturbation diversity.** Removed because the paper explicitly includes “two inpainting-based perturbations (Telea and Navier Strokes)” along with several other perturbation classes, so this criticism would be factually wrong.
- **Criticism doubting the existence/release/availability of datasets or models.** Removed per instruction; if they are cited, they are assumed real and available.
- **Pure formatting/style complaints** such as illegible figures, acronym inconsistency, or parser-induced presentation artifacts. These are not substantive enough for the main review.
- **Generic requests for many more architectures (e.g., transformers) as a core weakness.** This can be a useful extension, but given the already nontrivial model/dataset sweep, it is better treated as scope expansion rather than a central flaw.
- **Strong claims that the paper entirely lacks novelty because perturbation sensitivity was known before.** The paper does add breadth and attempts a more structured diagnostic framing; the better criticism is not “no novelty,” but that the explanatory/theoretical claim is stronger than the evidence.

## Novel Insights
The most important synthesis is that the paper is **better understood as a perturbation-sensitivity study of model behavior than as a conclusive explanation of existing fidelity-metric inconsistency**. Under that reading, the empirical contribution is meaningful and reasonably broad. Under the paper’s own stronger framing, however, the evidence falls short because the proposed conformity measures are not connected back to the actual metric outputs they are supposed to explain. This distinction is crucial: the paper has a real contribution, but it is narrower than advertised.

## Suggestions
- Reframe the paper more modestly around **perturbation sensitivity of model-response proxies used in fidelity evaluation**, unless you can directly connect DROP/PSim to actual fidelity-metric disagreement.
- Add the missing bridge experiment: compute AOPC/AD/IC/W/faithfulness under multiple perturbations and test whether low DROP/PSim predicts metric instability or saliency-method rank reversals.
- Soften [P1] and [P2] from universal assumptions to diagnostic desiderata, and clearly separate “necessary assumption,” “helpful robustness property,” and “empirical heuristic.”
- Replace or complement the 50-random-pixel analysis with experiments on top-k salient pixels and with sensitivity analyses over sample size.
- Clarify the handling of multi-label data and predicted-class changes under perturbation.
- Improve the theoretical exposition so the connection among saliency ranking, perturbation-induced ranking, and metric behavior is explicit and rigorous.

## Score and Decision
**Originality:** Moderate. The topic is important and the perturbation sweep is broader than some prior studies, but the conceptual core—perturbation choice affects saliency evaluation—is not entirely new.

**Importance of the research question:** High. Reliability of saliency fidelity metrics is a significant issue for the XAI community.

**Whether the claims are well supported:** Mixed to weak. The descriptive claim that perturbation choice changes model responses is well supported; the stronger claim that the paper explains why perturbation-based fidelity metrics are inconsistent is not fully established.

**Soundness of experiments:** Moderate. The experiments are extensive and computationally substantial, but the choice to study only proxy diagnostics, the single-perturbation setup, and the random-pixel protocol limit how directly they support the headline claim.

**Clarity of writing:** Moderate. The high-level message is understandable, but some notation, equations, and theoretical steps are imprecise.

**Value to the research community:** Moderate. The paper is a useful cautionary empirical study, but in its current form it overstates what has been proven.

**Calibration against human-reviewed anchors:**  
- Compared with **“Why Sanity Check for Saliency Metrics Fails?”** (scores 1, 3, 8, 3; reject), this submission is in a very similar space and has similar strengths/weaknesses: important problem, some useful empirical observations, but overclaiming and a gap between what is measured and what is concluded. This paper is somewhat broader empirically, so I would place it a bit above the low end of that distribution, but still below acceptance.
- Compared with **“Towards Robust Fidelity for Evaluating Explainability of GNNs”** (scores 8, 8, 5, 3; accept), that paper directly connected its critique of existing metrics to a better-founded replacement and supported it theoretically and empirically. The present paper is clearly weaker on the support for its main claim.
- Compared with **“AttributionLab”** (scores 6, 5, 6, 6; reject), this paper is less convincing because its main claim rests on a weaker bridge from diagnostics to the target object of study.
- Compared with **“Benchmarking Deletion Metrics with the Principled Explanations”** (scores 8, 3, 5, 8; reject), that paper also studied metric behavior directly and more explicitly; despite mixed reviews, it provides a stronger direct connection to the metrics themselves than the present paper.

Overall, this is **not a bad paper**, but it is **an overstated one**. The empirical observations are useful, yet the main claim is not supported strongly enough for acceptance in its current form.

**Final score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>