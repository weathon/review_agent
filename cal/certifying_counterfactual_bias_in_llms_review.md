=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary
LLMCert-B proposes a framework for statistically certifying counterfactual bias in LLMs by providing high-confidence bounds (via Clopper-Pearson intervals) on the probability of unbiased responses over distributions of counterfactual prompt sets. Three prefix distributions are instantiated — random token sequences, mixtures of manually-designed jailbreaks, and embedding-space perturbations of jailbreaks — to define specifications around fixed pivot sets. Applied to nine contemporary LLMs (including GPT-4, Mistral, Gemini, Llama, Claude) across two bias datasets (BOLD, Decoding Trust), the framework reveals latent biases in models that appear well-aligned under standard benchmarking, most notably in Mistral-7B and GPT-3.5.

---

## Strengths

- **Genuine methodological advance over point-wise benchmarking.** The shift from evaluating a fixed dataset to certifying over a *distribution* of prompt sets directly addresses test-set leakage and lack of generalization guarantees. The probabilistic certification view (rather than adversarial worst-case or empirical average) is a meaningful and well-motivated middle ground for auditing LLMs at scale.

- **Black-box applicability to closed-source models.** Two of the three specification types (random prefixes and jailbreak mixtures) require only input–output access, making them directly applicable to commercial APIs (GPT-4, Claude, Gemini). This is a practically significant feature not offered by neural network certifiers such as those based on abstract interpretation.

- **Concrete, non-obvious empirical findings.** The paper reveals that Mistral-7B, which achieves 100% unbiased responses under standard benchmarking (no prefix), collapses to [0.22, 0.42] under jailbreak mixture specifications on BOLD — a finding that standard evaluation would entirely miss. Similarly, GPT-4's latent bias under BOLD mixture specifications ([0.80, 0.96]) contrasts sharply with its near-perfect baseline, demonstrating the diagnostic value of the framework over conventional evaluation.

- **Novel relational property framing.** Specifying counterfactual bias as a *relational* property (responses across demographic groups must not be disparate) is the first such formulation for trustworthy LLM certification. This is a conceptually clean and technically well-grounded contribution relative to prior LLM safety certification work, which focuses on non-relational, single-output properties.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Bias detector uncertainty is not propagated into the certificates.** The regard-based bias detector achieves only 76% agreement with human judgment (Appendix G.1). The Clopper-Pearson bounds are, therefore, bounds on an imperfect proxy of bias — not on true bias itself. A detector with 24% error can produce systematic false negatives (calling biased outputs "unbiased"), resulting in certificates that are unconditionally optimistic. The gap between the *formal* guarantee (over D-detected bias) and the *claimed* guarantee (over actual bias) is never quantified or bounded. For a paper whose core contribution is providing formal guarantees, this is a significant unaddressed gap. A sensitivity analysis showing how detector error rates deform the certificate bounds is essential.

- **Refusal conflated with unbiased behavior.** Section 5.1.1 explicitly notes: "we observe unbiased behaviors because the model simply refuses to respond." However, the results in Table 1 do not distinguish between genuinely unbiased responses and refusals. This substantially inflates the certificate quality for safety-aligned models (Llama, Claude) and potentially for GPT-4 under DT specifications. More critically, models with disparate refusal rates across demographic groups (e.g., refusing more for prompts about one race than another) would score as "unbiased" under this framework, masking allocation harms entirely. The paper identifies the problem but takes no corrective action, yet the certificates for these models are presented at face value.

### Minor

- **Averaging confidence interval bounds is misleading.** Table 1 reports "the average of the certification bounds for all pivot prompt sets in Q_BOLD (250 items) and Q_DT (48 items)." Averaging the endpoints of 250 independent Clopper-Pearson intervals does not produce a valid aggregate confidence interval — the coverage guarantee of the individual certificates does not transfer to the averaged summary. If a practitioner reads the averaged bounds as providing a 95%-confidence guarantee over all 250 pivot sets jointly, that interpretation is incorrect. The paper should clarify that the reported averages are purely descriptive summaries of individual certificates (each valid in its own right) and not a new aggregate statistical guarantee.

- **Formal Definition 1(3) does not match operational practice.** Definition 1 requires that for an unbiased generator f, ∀i, f(P_i) = f(X_i) — i.e., the output must be *identical* to what would be produced without the sensitive attribute. This is an extremely strong, arguably impossible condition for stochastic generative models. In contrast, the operational bias detector (BOLD setup) measures *sentiment disparity* across responses, not output identity. The formal definition and the instantiated detector are not aligned. The paper should either revise the formal definition to reflect disparity-based notions (e.g., D(L(P_1), ..., L(P_s)) = 0 capturing group-level equitable outcomes) or explicitly flag the gap between formal specification and operational instantiation.

- **Gemini safety filter disabling is under-disclosed.** The choice to disable Gemini's safety filters (Section 5.1.2) is mentioned only in the model-specific discussion and is not called out in the experimental setup or Table 1. Certifying a model with production safety layers disabled produces certificates for a system that real users never interact with, limiting the practical interpretation of those results. This should be prominently disclosed alongside the Gemini results.

### Tiny

- **Scaling claim is unsupported.** The conclusion that "there are no consistent trends in fairness with scaling of model sizes" (Section 1) rests on comparisons of 7B vs. 13B variants of exactly two model families (Vicuna, Llama). Two data points per family over a 2× size range is insufficient evidence for a general empirical claim about scaling behavior. This should be hedged as a preliminary observation.

- **n=50 produces wide intervals.** With n=50 and 95% confidence, Clopper-Pearson intervals near p=0.5 span approximately ±14 percentage points. Several cells in Table 1 show interval widths of ~0.20. The ablation in Appendix E addresses this, but the main paper would benefit from acknowledging the practical precision limitations of the default configuration.

---

## Nice-to-Haves

- **Detector-noise-aware confidence intervals.** Modifying the certification algorithm to incorporate the detector's confusion matrix (false positive/negative rates from the human study) would yield bounds that formally account for measurement error, substantially strengthening the connection between the statistical guarantee and actual bias.

- **Disparate refusal reporting.** Explicitly reporting refusal rates per demographic group, separately from the unbiased response rate, would prevent the conflation issue above and reveal allocation harms that the current framework cannot detect.

- **Coverage validation experiment.** A meta-experiment verifying that the computed 95% intervals actually contain the true bias probability (estimated via large-scale sampling) in approximately 95% of trials would empirically validate the calibration of the Clopper-Pearson bounds in the LLM setting, addressing concerns about LLM-response non-stationarity.

- **Query-based approximation for soft prefixes on closed-source models.** While the paper correctly excludes closed-source models from the embedding-space specification, exploring query-based approximations (e.g., via finite-difference gradient estimates of embedding perturbations) would extend the strongest stress-test to the most widely-deployed models.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **"Certify" terminology is inappropriate (Harsh Critic).** In the probabilistic certification literature (e.g., randomized smoothing for adversarial robustness), "certify" is standard for statistical confidence bounds. The paper is explicit that it targets statistical, not formal/deterministic, guarantees. The use of "certifying" in the title is consistent with established usage in the field. REMOVED.

- **i.i.d. assumption invalidity due to LLM caching/tokenization (Harsh Critic).** The i.i.d. assumption applies to the *prefix sampling* process, which is independent by construction (each prefix is drawn independently from Δ_pre via Algorithm 2, 3, or 4). The critic's concern about correlated LLM responses due to in-context caching is speculative and not evidence of a genuine violation of the mathematical i.i.d. requirement on Bernoulli observations. REMOVED.

- **Random prefixes being uninformative garbage (Harsh Critic).** The paper explicitly frames random prefixes as a baseline specification revealing "denoising capabilities" of LLMs — not as the primary adversarial contribution. The empirical finding that some models (Vicuna-13B) are sensitive even to random prefixes is a valid result, not an absence of insight. REMOVED.

- **Algorithm 3 mixture-of-jailbreaks hypothesis unvalidated (Harsh Critic).** The empirical results directly validate the effectiveness of mixture prefixes: Mistral collapses to [0.22, 0.42] on BOLD and GPT-3.5 to [0.44, 0.67], while baselines show near-100% unbiased rates for these models without prefixes. The claim that such prefixes are "potential jailbreaks" is thus empirically supported. REMOVED.

- **Uniform noise for soft prefixes is unjustified (Harsh Critic).** The paper explicitly explains the choice: "we are not aware of any adversarial distributions of noise that could be added to manual jailbreaks to make them stronger, we select a uniform distribution." This is a principled default under uncertainty. REMOVED.

- **Soft prefix inapplicability to closed-source models is a major limitation (Harsh Critic, Positive Reviewer).** The paper is fully upfront about this in Section 1 and Table 1. The paper also notes that only two of three specifications require white-box access; the third (random and mixture) apply to closed-source models. This is a known constraint of the specification type, not a methodological failure. WEAKENED to a Tiny-level note, absorbed into the minor weakness on asymmetric comparison.

---

## Novel Insights

The most genuinely novel observation — surfaced by all three reviewers and confirmed by the paper — is that the *conflation of model refusal with unbiased behavior* creates a structural blind spot in the certification framework: models that refuse all adversarial prompts achieve near-perfect certificates not because of genuine fairness but because of blanket refusal, which itself may be disparately applied across demographic groups (an allocation harm). This is not merely a limitation of this paper; it is a latent confound in *any* safety evaluation framework that uses refusal as a safe response category without separately accounting for disparate refusal rates. A bias certification framework that cannot distinguish "equitably helpful" from "equitably unhelpful" provides an incomplete fairness picture, and this interaction between safety alignment (refusal) and fairness evaluation deserves dedicated attention from the research community.

---

## Suggestions

1. **Report refusal rates per demographic group separately** from the bias rate in all certification results. A model refusing at 90% for "female" prompts and 50% for "male" prompts would score well on counterfactual bias but exhibit severe allocation bias — the current framework cannot catch this.

2. **Clarify the statistical status of averaged bounds in Table 1.** Add a footnote or prose clarification that the reported values are descriptive averages of individual (each valid) certificates, not a single aggregate certificate with 95% coverage over all pivot sets simultaneously.

3. **Provide a formal or empirical treatment of detector uncertainty.** At minimum, a sensitivity analysis showing how the 76% detector accuracy (with plausible false-negative rates) shifts the reported lower bounds would allow practitioners to calibrate their trust in the certificates.

4. **Revise Definition 1(3)** to align with the operational bias notion (disparity in group-level sentiment) rather than the unattainable condition of output identity.

5. **Prominently disclose the Gemini safety-filter-disabled configuration** in the experimental setup table and Table 1, so readers can correctly interpret those certificates.

---

**Overall assessment:** The paper addresses a timely and genuine gap in LLM evaluation by introducing a principled probabilistic certification framework with real empirical payoff — revealing non-trivial vulnerabilities in Mistral and GPT-3.5 that standard benchmarking misses entirely. The novelty in applying relational property certification to LLM fairness is meaningful, and the black-box compatibility is practically important. The primary issues — detector accuracy not propagated into certificates, and refusal conflated with unbiased behavior — are significant but correctable, and they do not undermine the core conceptual contribution. The paper is technically sound in its use of Clopper-Pearson intervals; the methodological concerns are about the *interpretation* and *presentation* of the results rather than the statistical machinery itself. **Novelty** is high; **technical soundness** is moderate (sound core, with the two major gaps noted above); **empirical support** is solid and the case studies are compelling; **significance** is meaningful for the emerging LLM safety auditing community; **clarity** is generally good with the noted exception of Definition 1 and the averaging presentation.

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 8.0, 8.0]
Average score: 6.2
Binary outcome: Accept
