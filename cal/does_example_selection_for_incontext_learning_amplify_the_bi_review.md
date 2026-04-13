=== CALIBRATION EXAMPLE 30 ===

# Final Consolidated Review
## Summary
This paper studies whether demonstration/example selection for in-context learning changes the social bias profile of LLMs, focusing on gender and race bias in sentiment classification. It introduces a paraphrased bias-evaluation dataset derived from EEC, reports that example selection can increase worst-case bias across random seeds even when accuracy improves, and proposes ReBE, a prompt-tuning method with a demographic-aware contrastive objective intended to reduce such bias while preserving ICL performance.

## Strengths
- **The paper isolates an underexplored failure mode of ICL selection: fairness tail risk rather than just mean performance.** Section 3.3 explicitly shows a nuanced pattern: example selection can reduce **mean** bias while increasing **maximum** bias across seeds, and Figure 2/Table 2 make this distinction concrete across multiple model families and four common selection methods. This is more specific and more useful than a generic “bias exists” claim.
- **The empirical diagnosis spans a reasonably broad set of LLMs and selection strategies for the analysis phase.** The main analysis covers 8 LLMs and 4 example selection baselines, including random, similarity, perplexity, and DPP selection. For the core observational claim that seed-dependent selection can worsen worst-case fairness outcomes, this breadth is a meaningful asset.
- **ReBE is a practical, parameter-efficient intervention that is designed to compose with existing ICL pipelines.** The method freezes the base LLM and learns virtual prompt tokens via prompt tuning, making it operationally lightweight relative to full fine-tuning. The compatibility angle is borne out at least partially by the DPP+ReBE and Random+ReBE results in Tables 3 and 5.
- **The bias-contrastive construction is sensible and task-aware.** In Eq. (4), positives are samples with the same label but different demographic attribute, while negatives are samples with different label and the same demographic attribute. This directly targets the intended invariance structure rather than applying a generic contrastive loss.
- **The paper does provide some mechanism-oriented analysis rather than only aggregate metrics.** Figure 3’s group-conditioned confusion matrices are useful for showing how the disparity manifests (e.g., sadness→fear asymmetry), and the ablation in Table 4 does support the claim that the fairness-accuracy tradeoff is driven by the proposed two-term objective rather than accuracy loss alone.

## Weaknesses

### Major:
- **The central claim “example selection for ICL amplifies the biases of LLMs” is overstated relative to the evidence.** The paper’s strongest empirical finding is that example selection often increases the **maximum** bias across random seeds, while Section 3.3 itself states that mean bias often decreases. That supports a narrower and important claim about **increased tail risk / seed sensitivity of bias**, not a blanket statement that example selection generally amplifies bias. The current framing conflates “worse worst-case outcomes under prompt/example variability” with “higher bias overall,” which weakens the scientific precision of the main contribution.
- **The paper does not cleanly isolate the effect of *selection* from the effect of using ICL context at all.** Much of the headline argument compares selected-example ICL against zero-shot. That comparison establishes that adding demonstrations can change fairness behavior, but it does not by itself show that the *selection mechanism* is the causal driver, as opposed to ICL context, prompt length, or demonstration composition more broadly. The paper does include multiple selection baselines, which helps, but the claim “example selection amplifies bias” would be better supported by a more controlled attribution specifically to selection policy.
- **The mechanistic claim that example selection “contributes to spurious correlations” is suggestive but not established at the level stated.** Section 3.4 uses one case study (OPT-6.7B) plus null prompts to argue that the observed asymmetry is not fully explained by native parameter bias. That is interesting evidence, but it falls short of demonstrating that example selection is the source of the spurious correlation in a causal or general sense. At most, the paper shows that some prompt-conditioned ICL behavior introduces or accentuates such correlations beyond what is visible from the null-prompt probe.
- **The evaluation of ReBE is too selective to support the broad effectiveness claims made in the abstract and conclusion.** While the diagnosis stage uses 8 models, Section 5.1 evaluates debiasing only on “the two LLMs with the largest AvgGF in each baseline,” and excludes OPT-30B and Llama-2-70B. That is an understandable practical limitation, but it means the paper does not substantiate claims like “ReBE effectively mitigates biases of LLMs” or “is highly compatible with existing example selection methods” in a broad sense.
- **ReBE’s empirical gains are mixed, and the paper does not confront those mixed results candidly enough.** Table 3 includes several cases where post-debiasing metrics worsen, especially under the Perplexity baseline, and some maximum metrics also increase. The text emphasizes the improved entries and the overall trend, but for a method paper the failure cases deserve direct analysis. As written, the claims are stronger than the evidence supports.

### Minor
- **The dataset contribution is useful but modestly validated in the main paper.** EEC-paraphrase is still built from EEC-style demographic substitutions and emotion statements, now paraphrased by GPT-3.5. That likely improves surface naturalness, but the main text does not fully establish the stronger claim that it “better” captures real-world bias phenomena beyond Appendix-based quality validation.
- **Race-bias evidence is underexposed in the main paper despite being part of the stated contribution.** The paper repeatedly claims gender and race coverage, but the debiasing section in the main text focuses on gender, with race results moved to the appendix. This makes the main-paper support for generality across bias types weaker than the framing suggests.
- **The reporting and metric presentation could be clearer.** Table 2 and Table 3 are hard to parse, and the signed fairness metrics are not always easy to interpret when the concern is disparity magnitude or volatility across seeds. This is not fatal, but it makes it harder to assess the claims precisely.
- **The practical accuracy-preservation claim should be stated more carefully.** In several settings, accuracy drops after ReBE, and although some drops are small, the phrase “without significantly compromising accuracy” is stronger than what the mixed table entries justify unless backed by formal significance analysis.

### Trivial
- **The paper would benefit from clearer wording around what exactly is being claimed as novel:** bias amplification in the mean, amplification in the maximum over seeds, or increased fairness instability under example selection. These are not the same.
- **A brief discussion of sensitivity to the tradeoff parameter \(\alpha\)** would improve usability, since \(\alpha\) is the core fairness-accuracy knob in Eq. (5)/(total loss), but this is more completeness than a core flaw.

## Nice-to-Haves
- Add a controlled experiment that separates **ICL with arbitrary/fixed demonstrations** from **ICL with strategically selected demonstrations**, to attribute the observed fairness change specifically to selection policy.
- Expand ReBE evaluation to more of the 8 analyzed models, even if only smaller ones, to better support broad compatibility claims.
- Report the race-bias debiasing results in the main text, not only the appendix.
- Provide a sensitivity curve over \(\alpha\) to show the fairness-accuracy Pareto tradeoff.
- Include a small qualitative analysis of selected prompts/demographic composition to help explain why some selectors have worse tail fairness behavior.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaint that the paper lacks enough debiasing baselines from the literature.** I do not keep this as a core weakness because, based on the paper text, the authors explicitly state that there are no other debiasing methods specifically for ICL and compare against context-augmentation alternatives. Without external verification, it would be inappropriate to fault them for omitting unspecified prior methods.
- **Pure formatting/style complaints about table notation.** The tables are indeed somewhat hard to read, but this is minor presentation friction, not a substantive scientific flaw.
- **Reproducibility complaints centered on omitted low-level implementation details.** The paper provides code and enough methodological detail for a conference submission; demanding exhaustive hyperparameter minutiae would be disproportionate here.
- **Strong endorsement of the null-prompt analysis as rigorous causal isolation.** Review 2 overstates this. The null-prompt probe is informative, but it does not truly isolate causality, so that strength was weakened rather than kept in full.

## Novel Insights
The most interesting synthesis across the reviews and the paper itself is that the work is strongest when interpreted not as proving a universal “bias amplification” law, but as uncovering a **fairness reliability problem in ICL**: example selection can make fairness substantially more unstable across prompt seeds, creating harmful worst-case outcomes even when average accuracy and even average bias look acceptable. This reframing is more precise, better aligned with Figure 2/Table 2, and potentially more significant than the paper’s current wording because it identifies a deployment-relevant risk mode that standard average-case evaluations would miss.

## Suggestions
- Reframe the main empirical claim from broad “bias amplification” to **increased worst-case bias / fairness tail risk under example selection**, unless stronger causal evidence is added.
- Tone down the mechanistic claim in the abstract and introduction from “example selection contributes to spurious correlations” to a more defensible statement such as “ICL with selected examples can induce or exacerbate group-specific error patterns consistent with spurious correlations.”
- Expand ReBE evaluation beyond the selected worst-AvgGF models, or explicitly narrow the claimed scope of effectiveness.
- Analyze the negative cases in Table 3 directly: identify when ReBE hurts and whether that correlates with selector type, model family, or baseline bias profile.
- Add a controlled attribution experiment comparing zero-shot, fixed random-context ICL, and strategic selection ICL to distinguish the effect of **having context** from the effect of **how examples are selected**.
- Surface at least one race-bias debiasing result in the main paper if race is part of the headline contribution.
- Clarify in the writing that the paper’s strongest contribution is an **empirical diagnosis of fairness instability in ICL selection**, while ReBE is a promising but not yet comprehensively validated mitigation.

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 6.0]
Average score: 4.7
Binary outcome: Reject
