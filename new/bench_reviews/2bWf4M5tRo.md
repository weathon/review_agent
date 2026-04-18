Now let me read the comparison papers to calibrate my score.Now I have enough to write the final review. Let me synthesize.Now I have enough context. Let me write the final review.

---

## Summary

This paper proposes injecting uniform noise into the MLP outputs of intermediate transformer layers as an additional source of randomness to complement standard prediction-layer sampling for LLM hallucination detection. The key insight is that prediction-layer sampling preserves token likelihood orderings regardless of temperature, while intermediate-layer perturbation can disrupt these orderings, yielding complementary uncertainty signals. The paper introduces Answer Entropy as a metric suited to reasoning tasks and validates empirically—via ablations on K, temperature, noise magnitude, layer position, and model architecture—that combining noise injection with sampling improves AUROC over sampling alone on several datasets and metrics.

---

## Strengths

- **Novel and intuitive mechanism.** The observation that prediction-layer sampling is order-preserving while intermediate-layer noise is not provides a principled motivation for why the two sources of randomness are complementary. This is a genuine conceptual contribution, not just an engineering trick.
- **Consistent, non-trivial gains on the primary setting.** The flagship GSM8K + Answer Entropy result shows a +5.26 AUROC gain (Table 2), and Mistral shows a +5.92 gain (Table 6). These are not marginal.
- **Thorough ablation coverage.** The paper systematically varies number of generations (Fig. 4), temperature, noise magnitude (Table 4), injection layer band (Table 5), and model architecture (Table 6), giving a reasonably complete picture of the method's behavior space.
- **Answer Entropy metric.** The motivation for focusing on the final answer rather than all tokens is well-stated for chain-of-thought reasoning tasks, and the metric is simple and reproducible.
- **Algorithm is clear.** Algorithm 1 specifies that noise ε is sampled once per generation (not per decoding step), and the marginal distribution is correctly written out.
- **Figure 4 reports variance.** Mean and standard deviation across 20 groups of runs are shown, which is good practice.

---

## Weaknesses

### Fatal
None.

### Major

- **No comparison against existing SOTA hallucination detectors.** The paper benchmarks only against its own noise-free variant. Methods like INSIDE (EigenScore, which also exploits internal activation geometry; Chen et al. 2024, cited but never compared) use internal states without needing K noisy forward passes and have demonstrated strong AUROC numbers on overlapping datasets. Without even one head-to-head comparison on a shared dataset, it is impossible to assess whether the noise-injection approach is competitive, complementary, or inferior to existing internal-state-based methods. The contribution is framed as advancing hallucination detection, but all we learn is that noise helps relative to one particular baseline configuration of the same method.

- **Marginal gains on most non-GSM8K settings without significance testing.** While GSM8K and Mistral show substantial improvements, the majority of entries in Table 3 show gains of +0.2–1.7 AUROC (and one is −0.31), and Table 7 shows +1.85/+1.89. No confidence intervals, standard deviations, or significance tests are reported for these table entries. Given that K=5 is used and AUROC can fluctuate substantially at small K, many of these differences may not be reproducible. The paper's claim that noise injection "generally enhances performance" and "significantly improves detection" is not well-supported for the weaker results. Importantly, Figure 4 does show variance bands but only for varying K; Table 3's cross-dataset numbers lack any variance estimate.

### Minor

- **Table 5 text/table inconsistency.** The paper states "upper-layer injection is the most effective," but Table 5 shows middle-layer noise achieving the highest AUROC (79.36 vs. upper 78.55). Upper layer achieves the highest accuracy (36.65), which the paper conflates with detection. The text should be corrected.

- **Positive-only noise U(0, α) is unjustified.** This is asymmetric and systematically shifts activations upward, which is a distributional shift in addition to variance injection. No rationale is given for choosing U(0, α) over zero-mean Gaussian or symmetric uniform. Given that the direction of the noise relative to the model's activation distribution could matter substantially, this choice warrants at least an empirical justification.

- **Hallucination operationalization is standard but partially circular for Answer Entropy.** Defining hallucination as "majority of K answers are incorrect" is the same protocol used by Kuhn et al. and INSIDE and is appropriate here. However, since both the label (majority-vote correctness) and the detection metric (Answer Entropy) are derived from the same K generations, high-entropy examples are more likely by construction to fail majority vote. This inflation of apparent AUROC is worth acknowledging in the paper, even if it is not unique to this work.

- **Black-box applicability limitation not discussed.** The method requires access to intermediate MLP activations, making it inapplicable to black-box APIs. This is a meaningful practical constraint that should appear in the limitations section.

- **Hyperparameter sensitivity acknowledged but understated.** The noise magnitude differs by model (α=0.05 for Llama2-13B and Llama2-7B, α=0.02 for Mistral), and the optimal magnitude also varies per layer band (Table 5). The paper notes α=0.05 is not optimal for all datasets, but provides no guidance on how to set it without a labeled validation set. This limits plug-and-play applicability.

### Trivial

- The claim in §3.3 that r=0.67 demonstrates a "complementary relationship" is slightly overinterpreted in the text. A moderate correlation indicates partial independence, which is the right point, but "complementary" suggests near-orthogonality. A softer phrasing would be more accurate.

---

## Nice-to-Haves

- Experiment with zero-mean noise (Gaussian or symmetric uniform) to separate the effect of variance injection from systematic activation shifting.
- ROC curves rather than scalar AUROC for key comparisons, to reveal whether gains are broad or concentrated in specific operating regions.
- Qualitative examples showing instances where noise injection changes the detection outcome, to give mechanistic intuition.
- An experiment adding noise on top of semantic entropy clustering (not just using semantic entropy as a metric), to test whether the two are genuinely synergistic beyond TriviaQA.
- Analysis of when noise injection hurts (GSM8K Predictive Entropy drops −0.31): what properties of a metric or dataset make noise unhelpful?

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Hallucination = incorrectness is a fundamental structural flaw.** The paper uses the same operationalization as Kuhn et al., INSIDE, and other cited baselines. This is standard in the field, not a unique error of this paper. Removed as a misunderstanding of community norms.

- **Harsh Critic: Algorithm underspecified — is ε resampled per decoding step?** Algorithm 1 clearly places the noise sampling (Line 2) inside the generation loop but outside the decoding-step loop, making it unambiguous that ε is sampled once per generation. Removed as a misread.

- **Harsh Critic: No noise at T=1.0 → AUROC drops, so noise is required.** This is not a criticism of the method; the paper acknowledges that different T × α combinations have different optima. Removed as a scope-creep critique.

- **Harsh Critic / Spark: Lack of confidence intervals as a fatal flaw.** While significance testing is important, many single-run AUROC comparisons on large datasets (TriviaQA has 9,960 questions) are standard in this community. Reporting variance is a nice-to-have, not a fatal flaw. Moved to minor/nice-to-have tier.

- **Spark: Confound — noise injection changes the model's accuracy (34.95→36.32), so we're detecting a different model's hallucinations.** This is a somewhat confused critique. The hallucination labels are determined by ground-truth correctness, not by the model's own behavior; noise perturbs the generation process but the gold answers are fixed. The accuracy shift shows noise also improves performance, which is a secondary positive finding, not a confound in the detection task. Removed as a misunderstanding.

- **Human Finder: Compare to GPT-3.5/4 performance.** This is scope creep; the paper is about uncertainty-based detection using open-source models. Removed.

---

## Novel Insights

The central mechanistic insight — that prediction-layer temperature scaling is rank-preserving (it changes probabilities but not the ordering of the token distribution) while intermediate perturbation is order-disrupting — provides a principled geometric reason for why the two randomness sources are complementary rather than redundant. This is more principled than simply observing empirical diversity, and connects to ideas from the representation geometry and steering literature. If validated more rigorously (e.g., via mutual information analysis or per-instance case studies), this framing could justify a broader program of internal-state perturbation as a probe of model coherence, extending beyond hallucination detection to calibration and robustness more generally.

---

## Suggestions

1. Run at least one head-to-head comparison against INSIDE (EigenScore) on a shared dataset (e.g., TriviaQA) to situate the method in the SOTA landscape.
2. Report bootstrap confidence intervals or per-seed variance for Table 3 and Table 7 entries, given the small gains on non-GSM8K settings.
3. Fix the Table 5 text to accurately describe which layer band is best on which metric (AUROC vs. accuracy).
4. Add a limitations section acknowledging (a) white-box requirement, (b) need for per-model hyperparameter tuning, and (c) the positive-only noise design choice.
5. Experiment with zero-mean noise to test whether the gain comes from added variance or from the systematic positive shift.

---

## Score and Decision

**Calibration anchors:**

- **INSIDE** (Chen et al., 2024 — internal states for hallucination detection, same venue): Scores 8,6,6,6 → Accept. INSIDE provides clear methodological novelty, significant AUROC improvements, and meaningful baseline comparisons. The key advantage over this paper is that INSIDE directly benchmarks against multiple established methods.
- **SEP** (Semantic Entropy Probes): Scores 5,6,5,6 → Reject. Simple idea, limited improvements, insufficient baselines — closer to this paper in terms of evidence quality.
- **Randomized Feature Defense** (vZ6r9GMT1n): Scores 3,6,6,8 → Accept (poster). Also uses intermediate-layer noise injection, but has formal theoretical grounding (even if imperfect) and stronger empirical baselines.

**Assessment:** This paper is stronger than SEP (it has a cleaner conceptual story and more ablations) but weaker than INSIDE (no SOTA comparison, weaker gains on most metrics, unjustified noise design). The GSM8K and Mistral gains are real and substantial; the gains on the majority of other metric/dataset combinations are small and lack statistical backing. The missing SOTA comparison is a genuine hole that prevents a confident accept. On balance, this is a borderline paper with an interesting idea and adequate—but incomplete—validation. It sits below the threshold for acceptance primarily due to the absence of any comparison with INSIDE or other internal-state baselines (a directly comparable method that was accepted at the same venue), and because the empirical story outside the flagship GSM8K setting is weak.

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>