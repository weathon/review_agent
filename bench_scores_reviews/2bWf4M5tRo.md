## Summary
This paper proposes augmenting standard temperature-based sampling for LLM hallucination detection by additively injecting noise into the MLP outputs of intermediate transformer layers. The central argument is that hidden-representation perturbation and prediction-layer sampling are *complementary* sources of randomness—operating at different stages of computation—and that combining them improves the separation between hallucinating and non-hallucinating model responses, as measured by entropy-based uncertainty metrics. The method is validated across GSM8K, TriviaQA, CSQA, and ProntoQA with Llama2-7B/13B-chat and Mistral-7B.

---

## Strengths

- **Figure 4 provides the strongest piece of evidence.** The AUROC and accuracy curves for K=1 to K=20 include mean and standard deviation across 20 random seed groups. At every value of K, noise injection (α=0.05) consistently dominates standard sampling (α=0), with the gap visible and persistent. This is more convincing than single-point AUROC comparisons and directly addresses the "more K vs. noise" trade-off: at K=5 with noise, AUROC ≈ 79, whereas no-noise at K=10 reaches only ≈ 74. This is a specific empirical strength most papers in the area do not provide.

- **Table 4 provides concrete evidence for complementarity beyond the correlation argument.** The table shows that at T=1.0, standard sampling alone *drops* AUROC to 66.65, but adding noise brings it back up to 76.68–79.90. This asymmetric recovery—noise compensating for a degraded sampling temperature—is a more compelling argument for complementarity than the Pearson correlation figure alone.

- **Answer Entropy for reasoning tasks is a sensible, under-explored contribution.** Treating token-level entropy as the primary signal on chain-of-thought datasets is known to be noisy; focusing entropy estimation on the final answer string (Eq. 4) is a practical improvement over Equations 1–2 for tasks like GSM8K. Most uncertainty-estimation papers in this space do not make this distinction.

- **The method is white-box-only but trivially composable**: it adds no new architectural components, requires no fine-tuning, and integrates with any existing entropy-based detector as a drop-in augmentation. The composability with Semantic Entropy (Table 7) demonstrates this cleanly.

---

## Weaknesses

### Fatal
None. The paper's core empirical claim—that noise injection improves hallucination detection AUROC—holds in the majority of settings.

### Major

- **Factual inconsistency in Table 5 and its caption.** The table caption reads "with the upper layer demonstrating the greatest effectiveness," and Section 4.4 repeats "upper-layer injection is the most effective." However, Table 5 shows: No Noise = 73.15, Lower = 78.70, **Middle = 79.36**, Upper = 78.55 AUROC. The *middle layer* achieves the highest AUROC. The claim that upper layers are best is correct only on the ACC metric. Since the paper's primary metric throughout is AUROC, this is a factual inconsistency between the reported data and the stated conclusion. This should be corrected and the layer-selection recommendation revisited.

- **No discussion of MC Dropout, the closest conceptual prior work.** Applying dropout at inference time (Gal & Ghahramani, 2016) produces multiple stochastic forward passes from hidden-layer perturbations—conceptually the same operation as this paper. The Related Work section does not mention it at all. The paper need not numerically compare against MC Dropout (the mechanisms differ: zeroing vs. additive shifting), but must explain how the proposed method differs, why additive uniform noise is preferred over dropout-style perturbation, and what is genuinely novel given this prior art. Its omission leaves the novelty claim poorly situated.

- **Statistical testing is absent for main results in Tables 3, 6, and 7.** Many cells in Table 3 show improvements under 1 AUROC point (+0.20, +0.28, +0.33, +0.39), and one entry (Predictive Entropy on GSM8K) is negative (−0.31). Without confidence intervals or multi-seed statistics (which are provided for Figure 4 but not these tables), it is impossible to determine whether these small gains are reliable. The negative result is particularly concerning—it is not discussed anywhere in the paper. This is not a fatal flaw given Figure 4's robust evidence, but it significantly weakens the generalizability argument of Section 4.1.

- **The choice of uniform noise U(0, α) with a non-zero lower bound is unjustified and potentially problematic.** Because all sampled noise values are non-negative, the perturbation introduces a systematic *positive bias* to MLP activations across all injected layers—this is not an isotropic perturbation but a directional shift. The paper neither justifies this choice nor explores zero-mean alternatives (e.g., Gaussian N(0, σ²)). Whether the improvement stems from the direction of the shift rather than the stochasticity is left unexamined and undermines the theoretical interpretation.

- **No comparison against leading hallucination detection baselines.** The paper only compares "noise vs. no noise" within its own entropy framework. INSIDE (Chen et al., 2024, cited in the paper) uses intermediate activations directly for detection; DoLA (Chuang et al., 2023, also cited) uses layer contrast for factuality. Neither is used as a performance baseline. The paper cannot claim meaningful "improvement" in hallucination detection without placing the absolute AUROC numbers (e.g., 79.12 on GSM8K) in context against competing methods. This is perhaps the most significant gap for a systems-level contribution.

### Minor

- **No discussion of the white-box deployment constraint.** The method requires access to intermediate layer activations, making it inapplicable to closed-source API models (GPT-4, Gemini, Claude). This is a meaningful practical limitation entirely absent from the limitations section.

- **Unexplained numerical inconsistency between Table 2 and Table 3.** Table 2 reports the no-noise baseline at 73.86 AUROC; Table 3 reports the same configuration (Answer Entropy, T=0.8, GSM8K, Llama2-13B-chat, K=5) at 73.15. The 0.71-point discrepancy is not explained and suggests random-seed variance that, given the lack of reported uncertainty in the tables, makes the reported values hard to trust.

- **Hyperparameter sensitivity across architectures without a selection heuristic.** Section 4.5 notes that Llama2-7B uses α=0.05 while Mistral-7B requires α=0.02. No guidance is provided for choosing α without a held-out validation set, which limits practical applicability to settings where such data is available.

- **No computational overhead discussion.** The method requires storing and adding noise vectors to intermediate activations across multiple layers for every generation. While the K-sample computation budget is unchanged, the per-generation overhead of hooking into transformer layers is non-trivial in practice and should be quantified.

- **Weak theoretical justification for directional effect.** The paper's explanation for why noise disproportionately increases entropy on hallucinated responses is purely intuitive ("incorrect answers are less robust"). No mechanistic analysis (e.g., activation geometry, probing of hallucination-prone layers) supports this claim, which makes it unclear whether the effect generalizes beyond the tested models.

### Tiny

- Notation conflict: τ is used as both the binary classification threshold (Eq. on p.2) and as the sequence length index (Equations 1 and 2).

- Algorithm 1 samples noise ε once per generation (Step 2) and applies the same vector at every decoding step (Step 6). This design choice is not discussed or justified. Resampling per step would be an obvious alternative.

---

## Nice-to-Haves

- **Ablation on noise distribution type.** Comparing U(0, α) against N(0, σ²), dropout-style (Bernoulli mask), or symmetric U(−α, α) would clarify whether the directionality of the shift matters or if any stochastic perturbation works equally well.

- **Compute-normalized ablation.** Figure 4 nearly makes this comparison, but an explicit table showing noise@K=5 vs. no-noise@K=10 vs. no-noise@K=20 on AUROC would directly address whether noise injection is worth the engineering complexity relative to simply running more standard samples.

- **Visualization of which decoding steps diverge under noise.** A per-step analysis showing where hallucinated vs. correct chains diverge under noise injection would substantiate the "robustness" hypothesis with direct mechanistic evidence.

- **Failure case analysis.** When does noise injection hurt? The GSM8K Predictive Entropy case (−0.31) and Table 4's T=1.0 instability (76.68 at α=0.05 vs. 79.90 at α=0.01) suggest boundary conditions that deserve analysis rather than omission.

- **Calibration analysis.** Checking whether the uncertainty score is monotonically calibrated against actual hallucination rate (reliability diagram) would validate that the method is useful beyond classification thresholding.

- **Expanding to one or two larger open-weight models** (e.g., 70B-class) would strengthen the generalizability claim, though the current model zoo is already reasonable for an empirical paper.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"Title is misleading"**: Pure stylistic nitpick with no bearing on scientific content.

- **"The Pearson 0.67 is not evidence of complementarity"**: The paper does not rely solely on this number; Table 4's T=1.0 recovery and Figure 4's consistent advantage across K both provide independent empirical evidence. Partially addressed.

- **"Semantic Entropy evaluated only on TriviaQA is insufficient"**: The paper explicitly justifies this (ProntoQA/CSQA formats incompatible with Rouge-L clustering; GSM8K numerical answers incompatible with semantic clustering). The justification is sound and reasonable. Removed.

- **"The distinction between case study and main experiments is blurred"**: Pure organizational style criticism with no impact on correctness or reproducibility.

- **"Improvements in Table 3 could be overfitting because α=0.05 was optimized on GSM8K"**: The paper explicitly states this ("α=0.05 is not the optimal noise magnitude for each dataset and performance can be further boosted through hyper-parameter search"). The fact that cross-dataset improvement persists even with a non-optimized hyperparameter is arguably a *strength*, not a weakness. Removed.

- **"Unfair comparison if noise degrades output quality"**: The paper does check model accuracy (Table 2, Figure 4) and shows noise does not degrade ACC. The criticism about incoherent outputs is speculative and not demonstrated.

---

## Novel Insights

The most genuinely novel observation—underexplored even in the paper itself—is the asymmetric recovery in Table 4: at T=1.0, standard sampling alone degrades below T=0.8 performance (66.65 vs. 73.70), but noise injection at T=0.8 surpasses both (80.72). This suggests the two randomness sources are not merely additive but interact non-linearly: noise injection may compensate for regimes where token-probability entropy is already high and thus uninformative. If this interaction were mapped systematically (e.g., across temperature × noise level grids for multiple models), it could yield a deeper understanding of when internal-state vs. output-stage stochasticity is the binding constraint on uncertainty quality—an insight that would have broader applicability to uncertainty estimation in autoregressive models beyond hallucination detection.

---

## Suggestions

1. **Correct the Table 5 caption and Section 4.4 text.** Middle-layer injection achieves the highest AUROC (79.36). If upper layers are preferred for combined AUROC+ACC considerations, state this explicitly with the trade-off justified.

2. **Add a Related Work paragraph on MC Dropout and test-time perturbation methods.** Explain the conceptual difference (additive vs. zeroing; token-probability-order-preserving vs. not) and position the proposed method accordingly.

3. **Report confidence intervals for Tables 3, 6, and 7.** Since Figure 4 already shows the random-seed infrastructure exists, backfill these for at least the most important entries. Discuss the −0.31 Predictive Entropy result on GSM8K explicitly.

4. **Replace or supplement U(0, α) with a zero-mean variant** (e.g., U(−α, α) or N(0, σ²)) and report the comparison as an ablation. This directly addresses whether directionality is load-bearing.

5. **Add at least one SOTA baseline AUROC for reference.** Even a single row reporting INSIDE or DoLA's AUROC on the shared datasets (e.g., TriviaQA) would contextualize whether 75–82 AUROC is competitive, mediocre, or strong relative to the field.

6. **Add a limitations section** covering: (a) white-box access requirement, (b) the need for per-model α tuning without a validation-free heuristic, and (c) conditions under which noise injection reliably fails (high-T regime in Table 4).

---

**Evaluation axes:**
- **Novelty:** Moderate. The core idea—treating hidden-state noise injection as a complementary uncertainty source to output sampling—is clean and not obviously present in prior work, but the relationship to MC Dropout and test-time perturbation methods is not acknowledged. The contribution is primarily empirical rather than conceptual.
- **Technical soundness:** Below expectations for ICLR. The factual inconsistency in Table 5, the unjustified uniform-noise choice, and the absence of statistical testing in main tables are concrete technical gaps. The algorithm itself is sound.
- **Empirical support:** Mixed. Figure 4 is strong; Table 3 is weak for several metrics/datasets with several near-zero gains and one negative result and no uncertainty quantification. The absence of comparison to competing methods leaves the absolute performance level unanchored.
- **Significance:** Moderate if empirical claims hold robustly. The method is simple and composable, which is practically valuable. However, the white-box constraint limits scope to open-weight models, and the missing baselines leave significance unestablished.
- **Clarity:** Generally good. The algorithm is clearly presented, Figure 2 is effective, and the case-study-then-generalization structure is pedagogically sensible. The Table 5 inconsistency and the Table 2/3 discrepancy are clarity failures.