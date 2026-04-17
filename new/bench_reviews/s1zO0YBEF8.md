Now I have a good understanding of the paper and the calibration landscape. Let me synthesize my final review.

## Summary

This paper proposes the Structured Identity Mapping (SIM) task—learning the identity function on Gaussian mixture data with structurally organized centroids—as a theoretical abstraction for compositional generalization in diffusion models. The authors analytically solve the learning dynamics of one-layer and symmetric two-layer linear networks on SIM, explaining known phenomenology (sequential concept learning, signal/diversity-dependent generalization order, terminal phase slowing) and predicting a novel "Swing-by Dynamics" mechanism causing non-monotonic OOD test loss. The predicted Swing-by phenomenon is then verified in text-conditioned diffusion model experiments.

## Strengths

- **Well-defined and tractable abstraction.** The SIM task cleanly distills concept-space structure (signal strength, data diversity, compositional hierarchy) into a Gaussian mixture identity-mapping problem, enabling exact or semi-analytic dynamics. The mapping from concept space to SIM is clearly motivated: in the real setting, a good generator behaves as an identity mapping in concept space (Sec. 1, line 40-42).

- **Technically competent theoretical analysis.** Theorem 4.1 provides a closed-form decomposition of one-layer dynamics into growth and noise terms, with explicit rates controlled by $a_k = \sigma_k^2 + \mu_k^2/s$. The two-layer analysis (Eq. 4.3) decomposes Jacobian updates into growth, suppression, and noise terms, revealing a multi-stage mechanism. The mathematical framework for connecting Jacobian evolution stages to output trajectory stages (Fig. 5) is conceptually insightful.

- **Coherent mechanistic story.** The paper provides a unified explanation for three distinct phenomena—generalization order, terminal slowing, and Swing-by—through a single framework of differential growth rates and depth-dependent cross-direction interactions. The connection between major entry growth / minor entry suppression cycles and the non-monotonic loss curve is clearly articulated.

- **Novel prediction empirically verified.** The prediction of non-monotonic OOD test loss and its observation in diffusion models (Fig. 6b) goes beyond explaining known phenomenology and constitutes a genuinely new finding, even if the connection is primarily qualitative.

## Weaknesses

### Major

- **Insufficient evidence that SIM is a "faithful" abstraction of diffusion model concept learning.** The paper claims SIM is a "meaningful" and "faithful" abstraction (Abstract, Sec. 1, Sec. 6) and that its theory "explains" diffusion model phenomenology. However, the bridge between SIM and diffusion models is almost entirely qualitative—visual similarity of curves rather than quantitative prediction and verification. The key structural differences (identity regression vs. conditional generation, axis-aligned Gaussians vs. learned representations, MSE training vs. denoising objectives) are not argued to preserve the causal mechanisms responsible for compositional generalization. The paper does not test any quantitative prediction from the SIM theory (e.g., explicit rate dependence on $\mu_k$ and $\sigma_k$) in the diffusion model setting. This is a meaningful gap because the paper's core interpretability claim—that SIM provides mechanistic understanding *of* diffusion models—rests on this bridge being more than correlational.

- **Swing-by as a "novel mechanism" is overstated relative to the evidence.** The paper distinguishes Swing-by from epoch-wise double descent in footnote 2 (line 120), claiming it is a "distributional phenomenon" rather than one caused by noise or over-parameterization. However, (i) the two-layer linear model dynamics are a concrete instance of coupled mode learning in deep linear networks (related to Saxe et al. 2013/2019, Lampinen & Ganguli 2018, etc.), and the non-monotonicity arises simply from evaluating at an OOD point where competing modes make opposing contributions—a change in evaluation metric, not a fundamentally new mechanism; (ii) the diffusion verification (Fig. 6b) shows a double-descent-like curve with no error bars, no statistical characterization, and no attempt to rule out standard explanations (overfitting, optimization transients, classifier noise). The paper itself acknowledges Swing-by can be viewed as a special case of epoch-wise double descent but distinguishes it "largely by fiat" rather than by falsifiable criteria or controlled experiments.

- **The theory–experiment quantitative gap is substantial.** While Theorem 4.1 predicts explicit exponential rates $a_k$, the empirical sections do not quantitatively verify these predictions—e.g., no extraction of observed convergence rates or comparison with theoretical $a_k$ values. Similarly, the two-layer analysis provides rich qualitative descriptions (multi-stage growth/suppression) but no systematic exploration of how the number, magnitude, or timing of non-monotonic bumps depend on $s, d, \mu, \sigma$, or initialization scale beyond a single illustrative case (Fig. 5). For the diffusion experiments, only "signal" (color contrast) is manipulated; "data diversity" ($\sigma$) has no explicit counterpart, so the joint theoretical prediction of signal and diversity controlling order remains half-validated.

### Minor

- **Narrow OOD evaluation.** Both SIM and diffusion experiments evaluate on a single OOD test point (the "corner" combining all training-cluster means for SIM; the missing (blue, small) combination for diffusion). The paper mentions multiple OOD points in App. B, but the main mechanistic explanation (Sec. 4.2.2) is developed around the specific corner point structure. The authors acknowledge Swing-by is "rather modest" in high dimensions (line 114), which is the practically relevant regime for diffusion models, but this limitation is stated without deeper analysis of its implications.

- **Assumptions restrict generality.** The theoretical results depend on small initialization ($\omega \ll 1/(d \max a_i)$), sufficiently distinct $a_k$ values (Assumption D.5), and large $n$ (population covariance). The paper does not explore robustness to violations of these assumptions or establish how broadly they apply to realistic diffusion model training (e.g., Adam optimizer, batch normalization, large initialization scales).

### Trivial

- The optional cluster at the origin (Sec. 2.1) is included in the formal definition but never analyzed.

## Nice-to-Haves

- Testing the multiple-descent prediction for $s \geq 3$ concepts in diffusion models, as the theory explicitly predicts this and it is the most distinctive novel claim.
- Providing concept-space trajectory visualizations for diffusion models analogous to Fig. 2, which would more directly test whether SIM captures the diffusion model's dynamics at a finer grain.
- Exploring non-axis-aligned or correlated concept representations, which would test the robustness of the theory's structural assumptions.
- A controlled experiment disentangling Swing-by from epoch-wise double descent (e.g., varying OOD-ness independently from noise level).

## Novel Insights

The decomposition of deep linear network Jacobian dynamics into growth, suppression, and noise phases—connecting staged Jacobian evolution to staged OOD output behavior—is a genuinely useful analytical framework. The observation that minor Jacobian entries can create an "illusion of generalizing" before being suppressed is an insightful mechanistic description of how depth-dependent cross-direction interactions cause transient non-monotonic OOD behavior, even if the novelty of the resulting phenomenon relative to epoch-wise double descent is debatable.

## Suggestions

- Soften language around "faithful abstraction" and "explaining" diffusion model phenomena to more accurately reflect the qualitative nature of the bridge, or substantiate these claims with quantitative predictions tested in the diffusion setting.
- Add error bars / multiple seeds to the diffusion model experiments (Fig. 6b) to establish that the observed double-descent curve is not a high-variance artifact.
- Test at least one quantitative prediction from the theory (e.g., relative convergence rates as a function of $a_k$) in the MLP setting to validate that the theory accurately captures even the simplified experimental setting.

## Score and Decision

**Calibration comparison:**

- **Provable Compositional Generalization for Object-Centric Learning** (Accept oral, scores 8/8/6): Strong theoretical contribution with clean formalization, provable guarantees, and empirical validation. More rigorous theory–experiment connection and clearer novelty of the mechanism.
- **Grokking in Linear Estimators** (Accept poster, scores 3/8/5/6, avg ~5.5): Similar pattern—simplified linear model for a phenomenon originally observed in complex models (grokking), with analytical dynamics and some empirical validation. Controversial reviews, with one reviewer arguing the model is too simple to constitute "real" grokking. Accepted mainly because the deflationary insight (grokking as metric artifact) was genuinely new and well-argued.
- **On Provable Length and Compositional Generalization** (Reject, scores 6/5/5/5, avg ~5.25): Theoretical results on compositional generalization in simplified models, but with idealized assumptions and limited quantitative connection to practice.
- **Grokking in Linear Estimators** is the closest comparator: both papers study a well-known phenomenon via simplified linear models, derive analytical dynamics, and provide qualitative + some empirical verification in more complex settings. That paper was accepted at ~5.5 average with controversy.
- **Interaction Asymmetry** (Accept poster, scores 8/6/8/6): Stronger theoretical contribution with clearer formalization of a new principle, better-validated experiments, and more rigorous proof structure. Above the current paper.

The current paper has a clear theoretical contribution in analyzing SIM dynamics and identifying the growth/suppression mechanism, but its main claims about "faithful abstraction" and "novel mechanism" overreach the evidence. The qualitative-only bridge to diffusion models and the debatable novelty of Swing-by relative to known non-monotonic training phenomena are significant weaknesses. The paper is stronger than the rejected compositional generalization papers (which had even weaker theory–practice connections) but weaker than the accepted oral/spotlight papers in this space. It sits in a similar position to "Grokking in Linear Estimators" but with somewhat less sharp insight and more overclaiming.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>