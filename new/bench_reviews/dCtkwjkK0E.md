Now I have a good understanding of the calibration landscape. Let me compose the final review.

## Summary

This paper proposes active learning strategies for flow matching models in shape design, framing queries in terms of diversity and accuracy as competing objectives. By analyzing flow matching models through a piecewise-linear neural network and closed-form flow matching framework, the authors derive two query strategies: QD (promoting diversity by selecting data with similar labels) and QA (promoting accuracy by selecting data with distant labels), plus a weighted hybrid. Experiments on one synthetic and three real-world shape design datasets show the strategies behave as designed.

## Strengths

- **Novel problem direction**: Applying active learning specifically to generative (flow matching) models rather than discriminative models is a meaningful and underexplored direction. The paper correctly identifies the conceptual gap between "generative models for active learning" and "active learning for generative models."

- **Practical relevance**: The application domain of shape design with continuous labels and expensive numerical simulation is well-motivated. The use of real engineering datasets (airfoil, flying wing, starship) with CFD-computed labels demonstrates genuine utility.

- **Clean conceptual framing**: Articulating the diversity-accuracy trade-off as a fundamental tension in data composition, and providing separate knobs to control each, is intuitively appealing and empirically borne out in the results. The hybrid strategy with tunable ω offers practitioners a straightforward mechanism.

- **Computational efficiency**: Decoupling the query strategy from the flow matching model training, relying only on RBF networks for label prediction, is a practical advantage that eliminates iterative model retraining during active learning.

## Weaknesses

### Major:

- **The theoretical framework relies on strong, unjustified assumptions that do not hold for practical flow matching models**: The central analysis assumes the flow matching network is piecewise-linear, which enables the key claim (Eq. 2) that interpolation in condition space yields interpolation in data space. However, real flow matching models use deep architectures with residual connections and attention that are far from piecewise-linear interpolators. The cited "condensation" phenomenon (Luo et al., 2021; Xu et al., 2025) only holds under specific conditions (small initialization, dropout) that are not standard in flow matching training. Furthermore, Eq. 2 is effectively assumed rather than derived—the paper does not show that CPWL networks satisfy linearity-in-condition across different linear regions. This is analogous to the criticism in the piecewise-linear CNN analysis paper (laKmMbx6x4), where reviewers noted that "the theory doesn't scale to real-world challenges" and there is "a large disconnect between their theoretical construction and actual behavior." The experiments use standard 8-layer MLPs with LeakyReLU—not closed-form flow models—yet there is no empirical validation that these models exhibit the interpolation behavior assumed by the theory.

- **The core Lemma 1 proof is circular**: The proof in Appendix A starts by assuming that u_t(x', a_0c_0 + ... + a_dc_d) = a_0u_t(x', c_0) + ... + a_du_t(x', c_d) (linearity of the flow field in conditions), which is the very property the lemma is supposed to establish. This is not a derivation from first principles—it is an assumption that constrains the model's behavior. The conclusion (interpolation in label space → interpolation in data space) is thus baked into the premise.

- **The proposed strategies do not meaningfully exploit flow-matching specifics**: Both QD and QA operate entirely at the dataset level using label-space distances (via RBF predictions) and data-space distances. They never use any property of the trained flow matching model—its uncertainties, gradients, score estimates, or training dynamics. The conclusion section acknowledges this: "the framework shifts the focus from model-internal diagnostics to data-centric selection." This means the methods are generic continuous-label active learning heuristics (essentially coreset + label-balancing for diversity, and label-space coverage for accuracy) that could be applied to any conditional model without modification, weakening the claimed contribution of "active learning for flow matching."

- **Experimental evaluation does not validate the theoretical claims**: The experiments show that QD increases the pairwise-distance diversity metric and QA increases label MSE accuracy, but they do not test whether: (a) the trained models actually exhibit the interpolation behavior assumed by the theory, (b) the error bound structure (quadratic in max label distance) is reflected empirically, or (c) the combinatorial diversity analysis (mn upper bound) predicts actual generative diversity. Without such validation, the theory remains motivational rather than explanatory.

### Minor:

- **Lemma 2 error bound is opaque**: The bound |f(x*) − c*| ≤ K max||c_i − c_j||² relies on a constant K only described as "related to f and d." No regularity assumptions on f are stated, making the bound difficult to use for principled strategy design. The quadratic dependence on label distance is not standard and lacks rigorous justification.

- **Diversity metric conflates spread with diversity**: The diversity score (average pairwise Euclidean distance) measures how spread out generated samples are in data space, not how well they cover the conditional distribution. This makes QD's success unsurprising since QD explicitly maximizes data-space distances. The paper acknowledges (end of Section 2.2) that Eq. 3 is an upper bound with potentially zero-probability samples, but the diversity analysis in Section 2.3 reasons purely about combinatorial counts, not actual model behavior.

- **No error bars or statistical significance tests**: Results appear to be from single runs. Active learning is sensitive to initialization; without variance estimates, claims of superiority over baselines are uncertain.

- **RBF prediction errors are unanalyzed**: Both QD and QA depend on RBF-predicted labels for unlabeled data, especially in early rounds with few labeled samples. Poor predictions could bias selection without any analysis of robustness.

### Trivial:

- The paper occasionally has minor notational issues (e.g., inconsistent indices in Eq. 2/3, "subject of interest which transforming" in the introduction).

## Nice-to-Haves

- Empirical validation that the piecewise-linear assumption approximates real model behavior (e.g., comparing interpolation behavior of actual vs. idealized models).
- Comparison with simple label-space or data-space coverage baselines tailored to continuous labels (e.g., farthest-point sampling in label space for accuracy, stratified sampling for diversity).
- Statistical significance tests across multiple random seeds.
- Extension to categorical/discrete label spaces, or explicit discussion of limitations.
- Sensitivity analysis of α, β, γ, and ω across different datasets and budget sizes.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Limited applicability to discrete/categorical label spaces"** (from Neutral Reviewer): The paper explicitly scopes itself to continuous labels (shape design with continuous performance requirements). Criticizing it for not handling categorical labels is scope creep—the paper is about continuous-condition generative models.

- **"No comparison against generative model AL methods"** (from Spark): The paper compares against the most relevant available baselines. GAL and related methods are designed for discriminative models, which is exactly the gap this paper identifies. Comparing with "naive or heuristic data selection strategies for generative models" is a fair suggestion but not a fatal flaw.

- **"Scalability to higher-dimensional label spaces"** (from Spark): The paper tests label dimensions 1 through 4. The concern about curse of dimensionality is valid but speculative—raising it as a nice-to-have would be more appropriate than as a weakness.

- **"QD outperforming full dataset in diversity suggests metric degeneracy"** (from Spark): This is already partially addressed in the paper, which notes this comes at the cost of accuracy. It's not necessarily a degeneracy—selectively choosing diverse samples can indeed increase pairwise spread over random inclusion. This is worth mentioning but not as a fatal flaw.

- **"FID should be used instead of custom diversity metric"** (from Spark): The paper explicitly argues for separate diversity and accuracy metrics rather than combined ones like FID. Their domain (shape design) makes FID, which is designed for images, less appropriate. This is a methodological choice, not a weakness.

- **"Incomplete hyperparameter specification"** (from Neutral Reviewer): The specific values of α, β, γ are tunable parameters of a heuristic. This falls under reproducibility of implementation details, which is a minor concern at best for a methodology paper.

- **"Missing related works"** (from various): Per the hard rules, I do not flag missing related works since I cannot verify their existence.

## Novel Insights

The paper's key conceptual insight—that same-label data enhances diversity while different-label data enhances accuracy in conditional generative models, and that these objectives are fundamentally antagonistic—is elegant and well-articulated. However, this insight is derived from an idealized model with strong assumptions that may not transfer to practical settings, and the resulting strategies are generic label/data-space sampling heuristics rather than flow-matching-specific innovations.

## Suggestions

1. **Empirically validate the interpolation assumption**: Show that conditioning on interpolations between labels in the trained flow model actually produces geometric interpolations in data space, versus comparing against the piecewise-linear prediction. This would bridge the theory-practice gap.

2. **Test against direct label-space baseline**: Compare QA against simple farthest-point sampling in label space (since QA essentially does this through the RBF) and QD against data-space coreset with label-balanced sampling. This would clarify whether the theoretical framework adds value beyond intuitive design.

3. **Report variance across seeds**: Run each experiment 3-5 times with different initial selections to establish statistical significance.

4. **Acknowledge the disconnect between theory and method more explicitly**: The paper should clearly state that QD and QA are motivated by the theoretical framework but operate as model-agnostic heuristics, and discuss when the theoretical predictions may fail.

## Score and Decision

**Calibration anchors**:
- Papers with strong theoretical claims undermined by unjustified assumptions but interesting empirical direction: piecewise-linear CNN paper (laKmMbx6x4) scored 3,3,1,5,3 (avg ~3), rejected/withdrawn.
- Papers with reasonable but limited experimental validation and moderate novelty in active learning: Diffusion Active Learning (73Q9U0vcja) scored 5,8,5,6 (avg ~6), rejected.
- Papers with similar domain (shape design, generative models) but clearer contribution: GINNs (zpX0teJu9Z) scored 3,5,5,6 (avg ~4.75), rejected. Compositional Generative Inverse Design (wmX0CqFSd7) scored 8,8,6,6 (avg ~7), accepted spotlight—but that paper had a clearer and more validated core method.
- FALCUN (yZBpnKpBCw), another competing-objectives AL paper with limited baselines and questionable generalization: scored 3,5,5,5 (avg ~4.5), rejected.

This paper sits in similar territory to the piecewise-linear CNN analysis paper and the AL competition papers. The theoretical contribution is the main novelty claim, but the theoretical framework has a fundamental circularity (Lemma 1 assumes linearity in conditions) and relies on assumptions that are not validated empirically. The practical strategies, while reasonable, are generic heuristics that do not exploit flow-matching specifics. The experimental validation is limited (4 datasets, no statistical testing, no validation of theoretical predictions). This places it below the "accepted" threshold: it is an interesting pilot study direction, but the gap between theoretical claims and actual contributions is too large in its current form.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>