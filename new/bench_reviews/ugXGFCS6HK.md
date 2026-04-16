Now I have enough information to write the consolidated review. Let me synthesize the reviewer claims against the paper content.---

## Summary

This paper introduces *principal distortions*, a framework for comparing the local geometries of N ≥ 2 image representations using the Fisher Information Matrix (FIM). The authors define a pseudometric on local geometry based on log sensitivity ratios and optimize a pair of distortions that maximize the variance of models under this metric—a natural generalization of pairwise "generalized eigen-distortions" (Zhou et al. 2023). The method is demonstrated on a nested family of early visual system models and on deep neural networks varying in architecture (AlexNet vs. ResNet50) and training procedure (standard, SIN, adversarial training), producing interpretable distortions that reveal architecture- and training-type-specific differences in local geometry.

---

## Strengths

- **Principled and elegant extension of prior work.** The reformulation of generalized eigen-distortions as a pseudometric (Eq. 3) is clean, and the natural extension to N models via variance maximization (Eq. 4) is well-motivated. The metric's scale-invariance and permutation-invariance properties are sensible and well-motivated; the connection to Fisher-Rao distance for mean-zero Gaussians (Appendix A) provides additional theoretical grounding.

- **Compelling efficiency argument.** The reduction from O(2N) or O(N²) distortions to just 2 (or O(log N) with iterative elimination) is practically significant for psychophysics experiments. This is a genuine practical advantage over prior approaches.

- **Novel and interpretable DNN findings.** The finding that principal distortions separate AlexNet/ResNet50 by architecture (textured/high-variability vs. smooth/low-frequency regions), while adversarial training separates by training type (unstructured noise vs. smooth color patches), is as far as can be determined genuinely novel. The result that SIN training does *not* override the architecture-driven separation is an interesting negative finding. Consistency is reported across 100 base images (with standard deviation), across multiple random seeds, and is validated by differential effects on classification decisions (Supp. Fig. SI.4), which provides some degree of independent corroboration.

- **Honest and substantive limitations section.** The authors clearly acknowledge the local approximation issue, the Gaussian noise assumption, and the qualitative nature of the human-perception comparisons. This transparency strengthens the paper.

- **Clear writing and logical progression.** The conceptual Figure 1 effectively frames the relationships among eigen-distortions, generalized eigen-distortions, and principal distortions. The progression from simple hand-crafted models to DNNs is logical.

---

## Weaknesses

### Fatal
*None identified.*

### Major

1. **Partial circularity in the evaluation.** Principal distortions are defined to maximize variance of log sensitivity ratios across models (Eq. 4), and the main evidence of success is showing that these distortions produce separated log sensitivity ratios (Figs. 2A, 3D–E, 4A, 5A). That is not a fully independent validation—it is largely a demonstration that the optimization achieves its own objective. The paper mitigates this partially (100 images, random seeds, classification effects), but there is no fully external criterion such as: prediction of held-out models, recovery of known architectural features from log-ratios alone, or correlation with human sensitivity thresholds. The stronger interpretive claims—that the distortions *reveal* architecture- or training-driven geometry—rest on this partly circular evidence combined with qualitative inspection.

2. **Key empirical claims are qualitative without quantitative support.** For the early vision models, the human-relevance comparison (Fig. 2C) is explicitly described as "visual inspection." For the DNNs, the claims that principal distortions "remarkably consistently" separate models by architecture or training type are supported by averaged log-ratio plots with standard deviation, but without formal clustering accuracy, separability statistics, or effect-size measures. A simple quantitative test—e.g., can one predict architecture or training type from the log-ratio pair with >X% accuracy?—would substantially strengthen the claims made in Sections 4.1–4.2.

3. **The Gaussian noise assumption is unexamined and material.** For deterministic models, the FIM is computed as I(s) = J_f(s)ᵀ J_f(s), i.e., assuming isotropic additive Gaussian noise. The paper acknowledges this limitation in Section 5 ("not generally representative of neural responses in the brain"), but provides no sensitivity analysis. Since FIM shape determines which distortions are selected as "principal," different noise assumptions (Poisson, heteroscedastic, fitted covariance) could produce qualitatively different distortions. It is unclear whether the architecture and training-type findings are robust to this assumption or are artifacts of it.

### Minor

4. **No empirical baseline comparison.** A natural empirical baseline—selecting the pairwise generalized eigen-distortion pair with highest variance, or randomly sampled distortions—is not compared to principal distortions in terms of how well each actually separates the models. The theoretical efficiency argument (2 vs. O(N²) stimuli) is convincing, but it would be reassuring to show empirically that the proposed distortions carry at least as much discriminative information as alternatives.

5. **The two-distortion limit is underdiscussed.** The paper does not analyze how much of the total inter-model variance (in the defined metric space) is captured by the first pair of principal distortions, analogous to the explained variance ratio in PCA. For complex or large model sets, two dimensions may be insufficient, and this limitation is noted only briefly.

6. **Optimization landscape characterization is missing.** The objective (Eq. 4) is non-convex. The paper refers to a gradient-based algorithm in Appendix B (omitted from the reviewed text), but does not report whether multiple random initializations are used, how often solutions differ, or what the optimized objective values are. This leaves open the question of whether reported solutions are local optima.

### Trivial

- The pseudometric footnote (footnote 1, p. 5) is technically correct but could be integrated into the main discussion more explicitly, since indistinguishability of non-identical FIMs under the chosen distortion pair is a practical limitation worth foregrounding.

---

## Nice-to-Haves

- **Pilot psychophysics experiment.** The paper describes exactly the setup needed (Fig. 2C, Supp. Fig. SI.1) and the authors explicitly flag this as future work. Even a small-scale threshold measurement from human observers for the early-vision principal distortions would transform a qualitative claim into a testable quantitative one.

- **Second and third principal distortions.** Showing the second pair (analogous to PC2 in PCA) and reporting explained variance would help assess whether two dimensions are sufficient for the model sets tested.

- **Sensitivity to image choice.** A systematic analysis of when and why distortions change qualitative character across base images would clarify the conditions under which the discovered architecture effects are stable.

- **Broader model diversity.** Testing on Vision Transformers and self-supervised models (briefly mentioned in supplement) is noted, and broadening this coverage in the main paper would increase confidence in the generality of the findings.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic — "Introduction overstates causal attribution to architecture/training":** The paper uses careful hedging throughout ("suggest," "as far as we know," "qualitative"). The abstract phrase "reveal differences in local geometry that arise due to architecture and training types" is accurate given the controlled model sets used. The paper does not claim a full causal experiment; the framing is appropriate. *Removed as nitpick.*

**Harsh Critic — "Efficiency argument (log₂N) is speculative":** The O(2 log₂ N) argument rests on the assumption that each iteration of the iterative elimination procedure discards half the models, which is framed explicitly as a best-case assumption. This is a thought experiment illustrating potential gains, not an empirical claim requiring evaluation. *Removed as scope creep.*

**Spark — "Sign ambiguity of ε₁ vs ε₂ not formally discussed":** The paper notes that convention is used (e.g., positive log ratio for the last tested AlexNet layer). This is a minor reproducibility clarification, not a substantive flaw. *Removed as formatting/trivial.*

**Neutral Reviewer — "Connection to prior work on DNN as brain models is insufficient":** The paper does not claim to fully solve model-brain alignment; it proposes a tool. Critiquing absent scope is scope creep. *Removed as outside stated scope.*

**Human Finder — "Limited model diversity (no ViTs, self-supervised)":** The paper explicitly justifies the AlexNet/ResNet50 choice as a controlled experiment and shows ViT/EfficientNet in the supplement. *Weakened to Nice-to-Have.*

---

## Novel Insights

The most genuinely novel observation in this work is the dissociation between architecture-driven and training-driven local geometry: SIN training (which radically changes global texture statistics) does *not* override the architecture-induced local geometry differences between AlexNet and ResNet50, while adversarial training does override them. This is a specific, non-obvious, and testable empirical claim that goes beyond the method demonstration. The further characterization that AlexNet's sensitivity advantage is specifically in high spatial frequency / high-variability regions regardless of foreground/background (Supp. Fig. SI.8) adds anatomically and architecturally grounded specificity. These findings merit independent follow-up.

---

## Suggestions

1. **Add a quantitative separation score.** For each experiment (architecture, SIN, AT), compute a simple classification accuracy predicting model type from the pair of log-sensitivity-ratios per image, and report it with a confidence interval. This directly addresses the circularity concern without requiring new experiments.

2. **Report optimization diagnostics inline.** Even a single-sentence report ("across 100 images and 5 random initializations, variance of the objective at convergence was X%") would allow readers to assess convergence quality.

3. **Run a noise-model ablation for the early vision models.** For the early vision models, Poisson noise is biologically motivated and analytically tractable; compute principal distortions under both Gaussian and Poisson assumptions and compare qualitatively. This is low-cost and would directly address a stated limitation.

4. **Report explained variance for the first distortion pair.** Compute the metric between all model pairs under the *optimized* distortions vs. under the *worst-case* 2D projection, and report the fraction of total inter-model variance captured.

---

## Score and Decision

**Calibration anchor papers:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| z7K2faBrDG | Fisher info metrics, perception | 8,5,5,3 | Accept (poster) |
| ih3BJmIZbC | Interpretable representational similarity (new method) | 6,8,6,8,6 | Accept (poster) |
| WyZT4ZmMzf | Evaluating rep. sim. metrics (no new method) | 3,3,3,5 | Reject |
| 4GfEOQlBoc | Image statistics and perception, limited validation | 5,5,6,5 | Reject |

**Positioning:** The paper is clearly above WyZT4ZmMzf (which had zero methodological novelty) and above 4GfEOQlBoc (which had debatable definitions and weak validation). It is comparable to ih3BJmIZbC, which introduced a new interpretable method for model comparison and was accepted with scores of 6–8, but the paper under review has weaker quantitative validation (qualitative inspection vs. explicit concept discovery and evaluation in ih3BJmIZbC). The z7K2faBrDG paper was accepted despite mixed reviews (3–8) because of its theoretical contribution and genuine empirical work—an approximate parallel to this paper.

**Assessment axes:**
- *Originality:* Good. Genuine methodological extension to N models, principled metric, novel DNN findings.
- *Importance of research question:* High. Local geometry comparison is underexplored and relevant to NeuroAI.
- *Claims well-supported:* Partially. DNN findings are supported qualitatively with some robustness checks; human-perception claims are explicitly deferred.
- *Soundness of experiments:* Moderate. The circularity concern is real; quantitative tests are missing.
- *Clarity:* Good. Well-structured and clearly written.
- *Value to research community:* Moderate-to-high as a methodology tool; the DNN findings are interesting independently.

The paper makes a genuine contribution and is honest about its limitations. The main weaknesses (qualitative validation, circularity, unexamined noise assumption) are real and substantive but are not fatal—they suggest the paper is strong as a proof-of-concept methodology contribution. Based on calibration against accepted methodology papers with similar empirical profiles, I place the score at **5.5**—borderline, leaning toward reject absent rebuttal, but not a confident reject given the genuine contributions.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>