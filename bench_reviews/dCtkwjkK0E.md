## Summary

This paper presents a theoretical framework for active learning in flow matching models with continuous conditions, motivated by expensive label acquisition in engineering design. By analyzing flow matching through a piecewise-linear neural network lens, the authors derive how dataset composition influences generation diversity (same-label data) and accuracy (different-label data), propose two competing query strategies ($Q_D$ and $Q_A$), and introduce a weighted hybrid to navigate the trade-off. Experiments on synthetic and three shape-design benchmarks show improvements over discriminative AL baselines.

## Strengths

- **Clear articulation of an under-explored problem**: The paper draws a sharp distinction between "generative models for active learning" (data augmentation for classifiers) and "active learning for generative models" (efficiently training the generator itself). This framing is precise and identifies a genuine gap—existing AL literature overwhelmingly targets discriminative settings, and the paper explains why those strategies fail for conditional generation with continuous labels.

- **Mechanistic insight linking data composition to model behavior**: The theoretical framework yields a specific, non-obvious claim: data sharing the same label drives generation diversity (via combinatorial interpolation), while data with different labels drives accuracy (via reducing interpolation error bounds). Even under the strong assumptions discussed below, this provides actionable guidance that goes beyond generic uncertainty-sampling heleneistics and is validated by the ablation in Fig. 9 and the consistent $Q_D$/$Q_A$ divergence across datasets.

- **Practical decoupling of query from generative model training**: Because $Q_D$ and $Q_A$ operate on dataset-level computations (distances and entropy in label/data space) rather than requiring intermediate FM retraining cycles, the annotation budget is allocated before any expensive generative model training. This is genuinely useful in the target domain where simulation labels are costly.

## Weaknesses

### Major:

- **The piecewise-linear assumption is central but unverified**: The entire theoretical derivation—Eq. 2 (linear interpolation of vector fields), Eq. 3 (linear interpolation of generated outputs), and the resulting query strategies—rests on the hypothesis (stated in Sec. 2.2) that flow matching networks exhibit "piecewise-linear interpolation behavior" due to condensation. While LeakyReLU networks are technically CPWL, condensation into the simple interpolation regime described by Eq. 3 typically requires specific conditions (e.g., infinite width, special initialization). The paper provides no empirical verification that the trained 8-layer, 512-unit networks actually obey Eq. 3 in practice (e.g., by testing whether generated samples at interpolated conditions match the linear combinations predicted). If the networks learn significantly non-linear mappings—as deep networks generally do—the "generation law" (Eq. 3) is invalid, and the theoretical justification for $Q_D$ and $Q_A$ collapses to heuristic motivation. This gap between strong theoretical assertions and unverified assumptions is the paper's most significant vulnerability.

- **Mishalignment between theoretical diversity definition and evaluation metric**: The theory in Sec. 2.3 defines diversity as the number of distinct combinatorial interpolation types at a *fixed* condition $c^*$—essentially measuring multimodality conditional on $c$. However, the experimental metric (Eq. 8) computes average pairwise Euclidean distance integrated over the entire condition space $dc_Y$. A model that simply outputs distinct shapes for distinct conditions (basic conditional generation functionality) would score high on this metric without being "diverse" in the theoretical sense. The paper needs to either (a) use a conditional diversity metric that aligns with the theory, or (b) explicitly reconcile why the unconditional metric validates the conditional theory. Without this, the empirical validation of $Q_D$'s theoretical motivation is weakened.

- **RBF surrogate quality is a hidden dependency without analysis**: Both $Q_D$ and $Q_A$ depend on RBF neural network predictions of labels for unlabeled data (Sec. 2.3, 2.4). The quality of query selection is entirely bottlenecked by RBF accuracy, yet the paper reports no RBF prediction error, no sensitivity analysis of how selection degrades with surrogate noise, and no comparison against using ground-truth labels for selection (an oracle upper bound). For complex physics tasks (e.g., 4D starship labels), RBF performance on sparse training data may be poor, potentially causing the query strategy to select suboptimal points. This is especially concerning given the "decoupled" design philosophy: the strategy is blind to the FM model's actual failure modes and relies entirely on the surrogate.

### Minor:

- **$Q_A$ is coreset in label space**: The paper acknowledges that $Q_A$ (Eq. 6, maximizing label-space distance) "performs the coreset algorithm in the label space." The novelty here is the *derivation* from the error bound (Eq. 5) rather than the algorithm itself. The contribution is the insight that the theoretical framework yields coreset as the accuracy-optimal strategy for this setting, not a new algorithm. This should be framed more carefully to avoid overstating algorithmic novelty.

- **$Q_D$ ablation reveals limited contribution of theory-driven terms**: Fig. 9 shows that the `distance(x, X)` term (standard data-space coreset diversity) is the most important component of $Q_D$, while the theory-motivated terms ($-distance(y, Y)$ and $\Delta entropy$) have comparatively minor effects. This suggests that the primary driver of $Q_D$'s performance is a well-known data-space diversity term, diminishing the practical impact of the novel theoretical framework for the diversity strategy specifically.

- **Claim of outperforming the full dataset needs explanation**: Sec. 3.2 states "$Q_D$ achieves the highest diversity, even outperforming the model trained on the full dataset." While this is plausible for the intentionally uneven synthetic dataset, the paper does not explain whether the real-world datasets share similar imbalance properties, or whether this result reflects a metric artifact rather than genuine improvement. A brief explanation would strengthen credibility.

- **Lemma 1 proof has a logical gap**: The proof in Appendix A (Eq. 17–19) compares two vector fields that are acknowledged to be "not exactly the same" but claims "their final generated results are consistent." The justification—that the difference lies only in noise schedules—is stated without formal proof that different noise schedules yield identical marginals. This should either be proven rigorously or the claim of equivalence should be softened.

### Trivial:

- **Notation inconsistency between condition $c$ and label $y$**: The paper uses $c$ for conditions and $y$ for labels, but in this setting they refer to the same physical quantities (e.g., lift coefficient). The interchangeable usage can confuse readers about whether the model is $p(x|c)$ or $p(x|y)$. A clarifying sentence would help.

## Nice-to-Haves

- Empirical verification of the piecewise-linear interpolation behavior (e.g., measuring linearity along condition-space paths in trained models) would substantially strengthen the theoretical foundation.
- An oracle experiment showing $Q_D$/$Q_A$ performance with ground-truth labels vs. RBF-predicted labels would quantify the surrogate's impact.
- Comparison with a generative-specific AL baseline (e.g., adapting GALIS to continuous conditions) would better position the contribution beyond "we beat discriminative methods."
- Evaluation on a non-shape-design task with continuous conditions (e.g., conditional image generation with continuous attributes) would demonstrate broader applicability, though the paper's scope to shape design is clearly stated.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Unfair comparison with baselines**: The claim that comparing against discriminative AL baselines is unfair is removed because the paper's core argument is precisely that generative-specific strategies should outperform discriminative ones in this setting. Showing that discriminative methods underperform is the point, not a flaw. (Removed: weakness about unfair comparison)
- **Missing related works / insufficient citations**: Per review rules, I cannot verify the existence of uncited works and should not flag missing references. (Removed: weakness about missing citations to recent AL methods)
- **Reproducibility concerns about undisclosed hyperparameters**: The specific values of α, β, γ are implementation details. While sensitivity analysis would be valuable, the absence of disclosed values is a minor reproducibility concern not rising to the level of a core flaw. (Removed: weakness about undisclosed hyperparameters)
- **Formatting and typographical issues**: Removed per rules against formatting/style nitpicks. (Removed: weakness about typos, equation numbering, figure formatting)
- **Demand for evaluation on image benchmarks (CIFAR-10)**: The paper explicitly scopes itself to shape design with continuous conditions. Requesting evaluation on fundamentally different tasks is scope creep. (Removed: weakness about limited to shape design domain—demoted to nice-to-have)
- **Statistical significance / confidence intervals**: Single-run evaluation with curve plots is standard in this community for active learning papers; demanding confidence intervals is not the field's norm. (Removed: weakness about lack of error bars)
- **Ethics statement**: Not a methodological concern. (Removed)
- **Skepticism about DALL-E-3/Veo3 using flow matching**: These are cited as motivating examples of advanced systems; the paper's contribution does not depend on this claim being precisely accurate. (Removed: minor factual concern about citations)

## Novel Insights

The paper's most interesting insight is the *formal demonstration that diversity and accuracy are inherently antagonistic from a dataset perspective in conditional generative models*: same-label data increases multimodality at each condition (by expanding combinatorial interpolation types), while different-label data reduces interpolation error (by shrinking convex hulls in label space). This is distinct from the usual GAN-style diversity-accuracy trade-offs (mode collapse vs. fidelity) because it operates purely at the dataset level before any model training, suggesting that the trade-off is a property of the data itself rather than the optimization process. The practical implication—that one cannot simultaneously maximize both without a hybrid strategy—is non-obvious and useful for practitioners allocating simulation budgets.

## Suggestions

- **Verify Eq. 3 empirically**: Train a flow matching model, pick two conditions $c_0, c_1$ in the dataset, and test whether the generated sample at $c^* = 0.5c_0 + 0.5c_1$ is approximately $0.5x_0 + 0.5x_1$. Report the deviation. Even partial validation would significantly strengthen the theoretical foundation.
- **Add an oracle-label experiment**: Run $Q_D$ and $Q_A$ using ground-truth labels instead of RBF predictions for the unlabeled pool. The gap between oracle and RBF-based performance directly quantifies the surrogate's impact and addresses the hidden-dependency concern.
- **Use a conditional diversity metric**: Report diversity at fixed conditions (e.g., variance of generated samples for a given $c$) in addition to the current unconditional metric, to align evaluation with the theoretical claims in Sec. 2.3.

---

**Axis assessments:**

- **Novelty**: Moderate. The application of active learning to flow matching with continuous conditions is novel, and the theoretical framework provides genuine insight. However, $Q_A$ is coreset-in-label-space and the most effective term of $Q_D$ is data-space coreset, limiting algorithmic novelty.
- **Technical soundness**: Mixed. The theoretical framework is creative but rests on an unverified assumption (piecewise-linear condensation), and the Lemma 1 proof has a gap. The RBF surrogate dependency is unanalyzed. The diversity metric misaligns with the theory.
- **Empirical support**: Adequate for the claimed application domain. Four datasets with consistent trends, but missing critical ablations (surrogate sensitivity, oracle comparison, conditional diversity).
- **Significance**: Moderate-to-good for the engineering design community. The decoupled query strategy and tunable diversity-accuracy trade-off are practically useful. Broader impact on generative AL is currently limited by the unverified assumptions.
- **Clarity**: Adequate. The paper is generally readable with helpful figures (Fig. 1, 2), but notation inconsistency ($c$ vs. $y$) and the abrupt transition to the piecewise-linear framework in Sec. 2.2 hinder full accessibility.