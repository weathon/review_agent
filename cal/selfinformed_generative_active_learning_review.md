=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary
SIGnAL proposes a reinforcement learning–based generative active learning framework that optimizes a large language model (Qwen2.5-7B-Instruct) via PPO to synthesize informative training instances. A novel acquisition function combining KL-divergence–based informativeness with an embedding-distance–based relevance penalty is introduced to prevent out-of-distribution generation. Experiments on three text classification datasets (SST-2, AGNEWS, QNLI) show SIGnAL outperforming pool-based baselines, particularly in severely limited data regimes (0.1% unlabeled pool).

---

## Strengths

- **Principled formulation of relevance-aware acquisition for generative AL.** The acquisition function (Section 4.2, Eq. 3) explicitly combines informativeness (KL divergence between neighbor predictions) with a distributional relevance term (inverse embedding distance). This directly solves an acknowledged failure mode of prior generative AL methods (GAAL, ASAL): producing out-of-distribution instances that score high on uncertainty but hurt downstream performance. Dividing by embedding distance is a clean, lightweight mechanism, not found in CAL or its predecessors.
- **Meaningful experimental setting.** Evaluating at 0.1% and 1% unlabeled pool sizes captures a genuinely underserved regime — one where pool-based methods are structurally disadvantaged. The two-axis comparison (dataset scale × pool restriction) provides clearer insight into when generative AL is valuable.
- **Adaptive generation policy.** The observed self-correction on QNLI (Section 5.4) — the generator initially biased toward entailment labels gradually learns to produce balanced labels as entailment data become uninformative — demonstrates that the RL loop drives qualitatively meaningful behavioral change in the generator, not just random variation.

---

## Weaknesses

### Fatal
*(None that individually invalidate the paper, but the combination of the two Major issues below collectively prevents acceptance in their current state.)*

### Major

- **Missing ablation on the RL component — the paper's core claim is unvalidated.** SIGnAL's key technical contribution is the RL-based optimization of the generator. Yet there is no experiment comparing against a non-RL variant: e.g., using the same LLM (Qwen2.5-7B-Instruct) to generate data with static prompts, with the same acquisition-based selection applied afterward. Without this ablation, it is impossible to determine whether the performance gains stem from (a) the RL optimization loop, (b) the acquisition-based selection over synthetic data, or (c) simply having access to a 7B LLM's pre-existing world knowledge. This ablation is not a "nice-to-have" — it is required to substantiate the core contribution.

- **Severe model-capacity asymmetry between SIGnAL and all baselines.** SIGnAL uses Qwen2.5-7B-Instruct (7 billion parameters) to generate data; all baselines are BERT-BASE-only. The datasets (SST-2 movie reviews, AGNEWS news articles) are precisely the kind of data that 7B instruction-tuned LLMs have absorbed in pretraining, giving SIGnAL a massive information advantage that pool-based BERT methods cannot access. The observed improvements may be entirely explained by LLM world knowledge rather than by anything specific to the SIGnAL framework. The paper neither controls for this nor discusses it. A meaningful evaluation would require at minimum a comparison against a baseline where the LLM generates data without RL optimization and the same selection strategy is applied.

### Minor

- **No computational cost analysis.** Running PPO on a 7B model inside each AL iteration is orders of magnitude more expensive than pool-based sampling on BERT. The paper does not provide wall-clock times, GPU hours, or any cost analysis. Without this, the practical claim — that generative AL is preferable when unlabeled data collection is expensive — is hollow: the compute cost of the generator may far exceed the annotation savings.

- **Acquisition function: distance metric d(·,·) unspecified.** Algorithm 1 (line 6) uses d(Φ(xᵢ), Φ(xⱼ)) but never defines the metric (L2, cosine, etc.). This is a reproducibility gap. Similarly, the PPO β coefficient and number of PPO update epochs per iteration are not reported.

- **Acquisition function: no treatment of numerical instability when d → 0.** When two instances are close in embedding space, the score diverges. No clamping, regularization, or minimum-distance cutoff is mentioned. In practice this likely occurs, especially in low-diversity early iterations.

- **Number of trials not reported.** Section 5.3 states results are averaged over "multiple trials with varying initial labeled datasets" but does not specify how many trials are run, making it impossible to assess the stability of the reported means and standard deviations.

- **Simulated oracle introduces unquantified label noise.** Synthetic labels come from a fine-tuned BERT achieving 90–94% accuracy (Section 5.3). The sensitivity of the RL reward signal and downstream classification accuracy to this noise is not analyzed. The paper acknowledges the issue but does not measure its effect.

- **QNLI label distribution claim is unquantified.** Section 5.4 asserts that the generator "gradually learns to produce more not-entailment data," but no label distribution curves or token-level statistics across iterations are provided. This is the most mechanistic claim made about the RL loop's behavior and deserves empirical support.

- **Early underperformance is acknowledged but not mitigated.** Section 5.4 confirms that SIGnAL underperforms pool-based methods in early iterations due to repetitive generation. Given that many practical annotation budgets are small, this is a real limitation. The paper suggests an adaptive strategy but does not evaluate it.

### Tiny

- Equation for data generation (Eq. 2) uses argmax notation, which formally implies greedy decoding but is a mismatch with standard LLM sampling practice. A clarifying note on the actual decoding strategy (temperature, top-p, etc.) would improve reproducibility.
- The conclusion's claim that SIGnAL "can be applied to other forms of data, such as images" is unsupported — no image experiments or architectural adaptation for images is described. This should be weakened to a research direction, not a contribution claim.

---

## Nice-to-Haves

- **Label distribution visualization over iterations.** A simple curve showing label ratios of generated QNLI instances across AL rounds would directly validate the key adaptive-generation claim in Section 5.4.
- **Embedding-space visualization.** A t-SNE or UMAP of synthetic versus real data, colored by acquisition score, would empirically validate that the relevance term in the acquisition function is keeping generated instances in-distribution.
- **Hybrid cold-start strategy evaluation.** The paper itself suggests an adaptive budget allocation (pool-based early, generative later). Even a simple experiment testing this would address the acknowledged cold-start weakness.
- **Reward convergence curves.** Showing that PPO reward improves across iterations (and does not collapse) would build confidence in the RL optimization's stability.
- **Using symmetric KL or Jensen-Shannon divergence.** Asymmetric KL in the acquisition function is inherited from CAL; a brief justification or comparison with symmetric alternatives would strengthen the methodological grounding.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Self-informed" title is misleading** (Harsh Critic): Pure style/labeling preference. The term is defensible — the generator is informed by its own outputs' downstream reward.
- **Abstract's "limited data scale" claim is non-discriminating** (Harsh Critic): Minor framing preference. Not a substantive weakness.
- **Argmax implies greedy decoding is a reproducibility flaw** (Harsh Critic): Argmax notation is standard in formal problem descriptions even when sampling is used in practice. This is a notational convention, not an implementation claim. A tiny clarification note at most.
- **Related work should include EDA/back-translation/text augmentation** (Harsh Critic): The paper's related work covers generative AL predecessors (GAAL, ASAL, BGADL). Data augmentation for NLP (EDA, back-translation) is a distinct field; the absence of augmentation baselines in the *experiment* is already captured under the RL ablation weakness. Demanding a richer related work section without external sources to verify is out of scope here.
- **Problem formulation "two exceptions" are trivial, not novel** (Harsh Critic): Formalizing a new setting's conditions is standard practice, not a false claim of novelty. The paper does not claim this as a contribution.
- **KNN search scope (labeled set only) is unjustified asymmetry** (Harsh Critic): Measuring relevance against the labeled set is the *design intent* — the function is asking whether a synthetic point is close to already-labeled real points, which is a principled choice for preventing distribution shift. This is not an error.
- **Existing query-synthesizing methods (GAAL, ASAL) should be included as baselines** (Spark Finder): The paper explicitly notes (Section 5.2) that all existing query-synthesizing AL methods target image data, making direct comparison impossible without substantial re-implementation. This is a fair scope limitation, not a gap — the criticism about an LLM-without-RL baseline is the correct version of this concern and is already captured.
- **Extended budget comparison is unfair (SIGnAL evaluated to 200%, baselines to 100%)** (Harsh Critic): The paper explicitly limits SIGnAL's evaluation to 200% "sufficient for understanding its behavior" (Section 5.3). The comparison at equal annotation budgets (≤100%) is visible in Figure 3 and is the primary comparison of interest; the extension beyond 100% is exploratory. This is not a fundamental fairness violation.
- **KL reference policy is unclear** (Harsh Critic): The paper explicitly defines π^Pretrained as the pretrained policy in the PPO objective (Section 4.3). This criticism is factually incorrect.

---

## Novel Insights

The most interesting structural insight buried in the paper is the generator's emergent self-correction on QNLI: when early generations are dominated by entailment labels, those labels progressively become uninformative to the classifier, which causes the acquisition function to assign them lower rewards, which in turn pushes the generator toward not-entailment instances. This is a demonstration of curriculum dynamics that emerge naturally from the RL loop without explicit curriculum design — the informativeness signal creates implicit label-balancing pressure. This phenomenon is described only qualitatively and deserves rigorous quantification. If verified, it would be a genuinely interesting property of the framework worth emphasizing as a standalone contribution.

---

## Suggestions

1. **Add the essential ablation.** Run SIGnAL's LLM with static prompts (no PPO), apply the same acquisition-based selection, and report accuracy. This single experiment is necessary to attribute the performance gains to the RL mechanism rather than to LLM pretraining knowledge.
2. **Add a fair baseline that controls for model capacity.** Even reporting SIGnAL performance when the LLM generator uses random selection (no acquisition scoring, no RL) would let readers isolate the contribution of the acquisition function and RL loop independently.
3. **Report wall-clock time and GPU hours** per active learning iteration for SIGnAL versus pool-based baselines. This is essential for any practical claims.
4. **Specify the distance metric, β coefficient, number of PPO epochs, and trial count** in the implementation details for reproducibility.
5. **Quantify the QNLI label distribution shift** with iteration-level curves; this is the paper's most mechanistically interesting claim and should have supporting data.
6. **Add a minimum-distance clamp** to the acquisition function denominator (e.g., max(d(·,·), ε)) and report the value used, addressing the numerical instability issue.

---

**Axis evaluations:**

- **Novelty:** Moderate — applying PPO to optimize a generator within an AL loop is the right direction, and the relevance-weighted acquisition function is a clean contribution; but the integration of these ideas is incremental given prior RLHF and AL literature.
- **Technical soundness:** Below bar — the acquisition function has unresolved numerical and specification issues; the RL objective's key hyperparameters are unreported; the argmax/sampling mismatch is unexplained.
- **Empirical support:** Weak — three classification datasets with no RL ablation, no model-capacity-controlled baseline, and unspecified trial counts are insufficient to substantiate the core claims. The most important experiment (LLM-without-RL) is entirely missing.
- **Significance:** Potentially meaningful direction, but in current form not demonstrated — without isolating the RL component's contribution, the practical value of the full SIGnAL pipeline (including its heavy computational cost) over simple LLM-based augmentation is unestablished.
- **Clarity:** Adequate — the framework description and algorithm are clear; the analysis section (5.4) is qualitative where it should be quantitative.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
