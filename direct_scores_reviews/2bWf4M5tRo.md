## Summary
This paper proposes injecting additive uniform noise into the intermediate MLP layer outputs of LLMs during inference to generate more diverse samples for uncertainty-based hallucination detection. The core hypothesis is that hallucinated answers are less robust to internal representation perturbations than truthful answers, and that this intermediate-layer noise provides a *complementary* source of randomness to standard prediction-layer sampling. Experiments across three models, four datasets, and multiple uncertainty metrics show AUROC improvements, with the largest gains on Answer Entropy for reasoning tasks.

---

## Strengths

- **Answer Entropy as a task-appropriate metric (Section 3.1, Table 1):** The introduction of Answer Entropy — an uncertainty metric that counts the distribution over extracted final answers rather than all tokens — is a concrete and sensible contribution for reasoning tasks with lengthy chain-of-thought reasoning. The motivation (avoiding length bias when measuring uncertainty over mathematical proofs) is clearly demonstrated with a worked example, and most prior work has only used token-level entropy metrics.

- **Complementarity empirically validated across K (Figure 4):** Figure 4 shows, with mean and standard deviation across 20 random seeds, that noise injection consistently outperforms the no-noise baseline at every K from 1 to 20. This is substantially stronger evidence than a single K=5 point comparison, as it rules out the possibility that the gain is an artifact of a particular sample size.

- **Dual benefit: detection and accuracy (Table 2, Section 3.4):** The observation that noise injection also improves majority-vote accuracy (34.95% → 36.32%) has a coherent explanation — incorrect hallucinated answers are less stable under noise and thus less likely to dominate the majority vote — and is a useful side effect beyond pure detection improvement.

- **Practical, training-free method:** The algorithm requires no model retraining and operates as a wrapper over any existing LLM inference pipeline. This is a genuine usability advantage over methods that require fine-tuning or training auxiliary models.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Gains are concentrated almost entirely in Answer Entropy; standard metrics show marginal or negative improvement.** Table 3 reveals a striking pattern: for Predictive Entropy and Normalized Entropy — the two most widely used metrics in the literature — noise injection yields gains of ≤ +1.08 AUROC and even causes a regression on GSM8K (−0.31). Only Answer Entropy shows substantive gains (+5.40 on GSM8K, +1.26–+1.76 on other datasets). Since Answer Entropy is a metric the authors themselves define and optimize α on, the broader claim that "noise injection generally enhances model performance across various uncertainty metrics" is overstated. The method may primarily be improving answer-space consistency rather than improving model uncertainty estimation in general, which substantially narrows the claimed contribution. The paper does not analyze or acknowledge this discrepancy.

- **No comparison to competitive detection baselines.** The only comparison throughout is between "noise injection" and "no noise" variants of the same metrics. INSIDE (Chen et al., 2024), Semantic Entropy (Kuhn et al., 2023b; Farquhar et al., 2024), and Lexical Similarity (Lin et al., 2022) are cited as related work but are never used as baselines. Table 7 compares noise vs. no-noise for Lexical Similarity and Semantic Entropy, but does not situate absolute AUROC numbers against the state of the art. A method proposing to advance hallucination detection should be placed against the best-performing detectors, not only against itself.

- **No discussion of MC-Dropout / Bayesian uncertainty estimation.** Injecting stochastic perturbations into intermediate layer representations at inference time to obtain a distribution over outputs is the core mechanism of MC-Dropout (Gal & Ghahramani, 2016). That entire body of work is absent from the related work and the paper makes no attempt to distinguish the proposed method's noise injection from MC-Dropout in terms of mechanism or performance. This is a substantive gap in novelty attribution that the authors must address.

- **Table 5 claims contradict the AUROC data.** Section 4.4 states "upper-layer injection is the most effective," but Table 5 shows Middle Layer Noise achieves AUROC = 79.36 while Upper Layer Noise achieves AUROC = 78.55 — a clear reversal on the primary detection metric. (Upper layers lead on ACC: 36.65 vs. 36.00, but ACC is secondary.) This internal inconsistency undermines confidence in the analysis and the claim in the text must be corrected.

### Minor

- **Figure 3 label inconsistency.** The x-axis of Figure 3 reads "Answer entropy, T = 0.5, No Noise," but Section 3.3 specifies that prediction-layer sampling uses T = 0.8. This is a labeling error that must be corrected, as it creates ambiguity about which temperature was actually used for the complementarity analysis.

- **Same noise vector reused across all decoding steps.** Algorithm 1, line 2, samples ε once per generation and applies the same vector at every decoding step t within that generation. This is not a technical error, but it means the noise acts as a fixed per-generation "representation offset" rather than step-level exploration noise. This design choice is never discussed or motivated. The implication — that all tokens in a response see the same additive bias — deserves at least a brief justification or comparison to step-resampled noise.

- **No statistical significance for marginal improvements.** Several AUROC differences in Table 3 are <1 point (e.g., +0.20, +0.28, +0.39). Given that K=5 is small and there is inherent stochasticity in generation, the significance of these differences is unclear. Figure 4 includes standard deviation bands for the K-ablation, which is commendable, but significance is not assessed in Table 3.

- **Noise distribution choice is unjustified.** The paper uses additive, non-zero-mean (asymmetric) uniform noise $U(0, α)$. This produces a systematic positive bias shift in intermediate representations at every perturbed layer. There is no ablation over noise distribution (e.g., zero-mean Gaussian, symmetric uniform), and no justification for why a biased shift rather than a zero-mean perturbation is preferred.

### Tiny

- **White-box requirement is not acknowledged as a limitation.** The method requires access to intermediate MLP layer activations, ruling out black-box API models (GPT-4, Claude, Gemini). This is a meaningful deployment constraint that the paper should acknowledge explicitly.

- **No expected calibration error (ECE) analysis.** AUROC measures ranking ability but not calibration of the uncertainty score. For deployment (thresholding at τ), a well-calibrated score is critical. Reporting ECE or reliability diagrams alongside AUROC would make the paper's practical claims more complete.

---

## Nice-to-Haves

- **Compute-matched baseline:** Compare noise injection at K generations against standard sampling at 2K generations. If simply doubling K achieves the same gain, the noise injection mechanism adds no value beyond increased sampling diversity. This baseline would directly test whether the complementarity claim holds under a fixed compute budget.

- **Layer-wise AUROC heatmap:** Replace the broad Lower/Middle/Upper ablation in Table 5 with a per-layer or per-group-of-5-layers AUROC curve. This would clarify whether "upper layers are best" is a smooth gradient or a sharp peak at specific layers, and would guide principled hyperparameter selection.

- **Principled noise magnitude selection:** The paper acknowledges that α = 0.05 is not optimal per dataset and that Mistral requires a different level than Llama. A heuristic for selecting α without labeled validation data (e.g., based on layer output norms) would meaningfully improve practical applicability.

- **Difficulty vs. hallucination confounding analysis:** High entropy under noise could reflect question difficulty rather than falsehood. An analysis separating "hard but correct" questions from hallucinated ones would validate the claim that the method targets hallucination specifically.

- **Larger and more diverse models:** Evaluation on a model ≥30B parameters and/or a non-instruction-tuned model would strengthen generalizability claims. The current 7B–13B range, all using the same Llama or Mistral family, limits the scope of the conclusion.

- **Accuracy claim scope clarification (Section 3.4):** Table 2 shows accuracy improvement on GSM8K, but Table 3 does not report accuracy for other datasets. It should be explicitly stated whether the accuracy boost is specific to reasoning tasks with majority-vote answering or whether it generalizes.

---

## Removed Points
*These points are flagged for removal — treat them with caution; they were raised by reviewers but are factually incorrect, out of scope, or insufficiently supported.*

- **Concern about "high temperature reverses token ordering"** (Harsh Critic, Concern 2): The paper's theoretical claim is that *sampling preserves token likelihood ordering for any temperature* — i.e., the most probable token is still most probable before the sample is drawn. This is correct: temperature sampling draws from the softmax distribution but does not reverse the ranking of token probabilities *per se*. The harsh critic's objection ("at high T, many orderings are reversed") confuses the probability ordering with the sample outcome. The argument that noise can actually invert which logit is largest by modifying hidden states is plausible and distinct from temperature. This concern should be removed.

- **Circularity in hallucination labeling** (Harsh Critic, Concern 6): The labeling scheme (majority of K=5 answers incorrect = hallucinating) is standard in the literature. The critic's circularity concern (noise changes majority outcome → changes the label) is theoretically possible but practically minor at K=5 with moderate noise, and the paper uses this labeling consistently across all conditions, not to selectively favor noise injection. This concern is removed as overstated.

- **Missing related works criticism** (all reviewers): Per review policy, no missing related work complaints are retained, as external existence cannot be confirmed.

- **Concerns about unfair comparison direction** (Harsh Critic, Concern 7, partially): The comparison "noise vs. no noise" is intentionally designed to isolate the contribution of noise injection, and the asymmetry is not favorable to the authors' method in an improper sense. However, the *absence* of any competitive baselines remains a legitimate concern and is preserved in the Major weaknesses above.

- **"The improvements are < 1 point are not real without statistical significance"** as applied to the Answer Entropy gains (+5.40 GSM8K): The +5.40 gain is large enough that significance testing is not required to accept it. The statistical concern is valid only for the sub-1-point gains and is scoped accordingly.

---

## Novel Insights

The most genuinely novel intellectual contribution is the observation that intermediate-layer noise and prediction-layer sampling are complementary sources of diversity for hallucination detection, with a Pearson correlation of 0.67 between their induced uncertainty distributions — high enough to show relatedness, low enough to justify combination. This complementarity, if robust across models and tasks, suggests a general principle: hallucination detection benefits from diversity at multiple levels of abstraction, not only at the output distribution. The further observation that incorrect answers under majority vote are *more* destabilized by noise than correct answers (yielding both better detection and higher majority-vote accuracy) is an interesting and somewhat counterintuitive finding about the geometry of hallucinated versus truthful representations in intermediate layers. However, the mechanism underlying this asymmetry (why hallucinated representations are more sensitive to additive shifts) is not yet explained, representing an important open question.

---

## Suggestions

1. **Directly address the Answer Entropy concentration issue:** Run an analysis testing whether noise injection is primarily randomizing the final answer extraction step (e.g., making the model sometimes produce a different number at the very end) versus perturbing the reasoning chain itself. One test: apply noise only after the reasoning chain is generated (i.e., during answer-extraction steps only) and compare AUROC. This would distinguish a representation-level effect from an answer-extraction artifact.

2. **Add a MC-Dropout comparison and discussion:** Implement a simple MC-Dropout baseline with dropout applied to MLP layers at inference time (using the model's existing dropout masks if present, or inserting them). Compare AUROC against the proposed uniform noise approach. Even a brief comparison establishes novelty relative to the closest prior art.

3. **Fix Figure 3 label:** Correct the x-axis label from "T = 0.5" to "T = 0.8" (or whatever temperature was actually used) and verify that the text and figure are consistent.

4. **Fix Table 5 prose:** Change the claim "upper-layer injection is the most effective" to accurately reflect the data — middle layers achieve higher AUROC (79.36) while upper layers achieve higher accuracy (36.65). If the authors believe upper layers are preferable for another reason (e.g., noise tolerance), that reasoning should be stated explicitly.

5. **Report standard deviation or significance for Table 3 results:** For the sub-1-point gains in Predictive and Normalized Entropy, include error estimates (variance across random seeds, as already done in Figure 4) so readers can assess which improvements are reliable.

---

**Evaluation on key axes:**

- **Novelty:** Moderate. Intermediate-layer noise injection for hallucination detection has not been explored in this form, and the complementarity framing is fresh. However, the connection to MC-Dropout is not addressed and reduces the novelty claim.
- **Technical soundness:** Moderate. The algorithm is clear and the ablations are reasonable. However, the same noise reused across steps, unjustified noise distribution, and the Table 5/Figure 3 inconsistencies indicate gaps in execution.
- **Empirical support:** Uneven. Gains on Answer Entropy are convincing; gains on standard Predictive/Normalized Entropy metrics are marginal and largely not statistically validated. The absence of competitive baselines is a significant gap.
- **Significance:** Low-to-moderate. If the method truly and reliably improves any hallucination detector with a simple plugin, that would be significant. But the current results suggest the gains are primarily limited to the authors' own Answer Entropy metric.
- **Clarity:** Mostly clear, with two notable internal inconsistencies (Figure 3 label, Table 5 claim).

MY FINAL SCORE: <pineapple>4.8</pineapple>