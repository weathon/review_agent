Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

This paper proposes two contributions for reducing hallucinations in Multimodal Large Language Models (MLLMs): (1) HalfScore, a graph-based hallucination metric that decomposes evaluation into precision (hallucination degree) and recall (coverage) at the concept level using triplet-extracted language graphs, and (2) PerturboLLaVA, a training strategy that injects adversarially perturbed text during instruction tuning to reduce the model's reliance on language priors and force greater attention to visual content.

## Strengths

- **HalfScore's precision/recall decomposition is a genuine conceptual advance** over existing metrics. CHAIR measures only object-level hallucination, while MMHalBench produces a single holistic score. HalfScore separates hallucination (precision) from coverage (recall) using structured triplet representations (Eqs. 1–3, Figure 3), addressing the real problem that models can game CHAIR by producing very short, safe captions. Table 2 demonstrates disaggregation into object, attribute, and relation hallucination types.

- **PerturboLLaVA is practical and deployment-friendly**. Table 1 shows it operates at 1× inference cost versus 2–5× for OPERA and 2× for VCD, requires no extra training stages, and avoids the costly preference data collection of RLAIF-V. This is a meaningful practical advantage.

- **The method improves rather than degrades general multimodal capabilities**. Unlike VCD (MMB 66.2) and RLAIF-V (MMB 63.7), which hurt general benchmarks relative to the LLaVA1.5 baseline (MMB 67.3), PerturboLLaVA improves them (MMB 68.9, SEED +0.3, CCBench +1.2), supporting the claim that reducing language prior reliance benefits visual understanding broadly (Table 3).

- **Complementarity with decoding-based methods**. Table 3 shows PerturboLLaVA + OPERA yields further gains (HalFscore 52.8, CHAIRs 33.1), demonstrating the training-based approach addresses an orthogonal optimization direction.

- **Honest and informative ablation**. Table 5 systematically varies perturbation strength (Versions 1–3) and relevance (Random), revealing trade-offs between hallucination reduction and recall/general performance. The finding that random perturbation also helps is honestly reported despite partially undermining the paper's narrative.

## Weaknesses

### Fatal
None.

### Major

- **The "outperforming existing approaches" claim is selectively true and overstated.** RLAIF-V dominates on CHAIR (18.1 vs. 36.1 CHAIRs), the most widely-adopted hallucination metric, by a very large margin. PerturboLLaVA's advantage exists only on HalFscore (the paper's own metric, 52.2 vs. 51.9) and MMBench. The abstract's claim of "outperforming existing approaches in handling multimodal hallucinations" is misleading without qualification. The paper's response—that they "plan to design targeted perturbation texts for objects"—acknowledges the gap but does not address it.

- **The mathematical framework (Section 4.2) has unjustified assumptions that undermine its explanatory power.** The derivation from Eq. 5→10 relies on a conditional independence assumption (Eq. 8) that $x_{<k}^p$ and $x_{<k}^{-p}$ are mutually independent given $x_k$ and $I$. These represent language-prior-influenced and image-influenced components of the *same* autoregressive prefix—deeply entangled in practice, not independent. Furthermore, the key claim that "the world knowledge embedded in the language model remains unchanged" during instruction fine-tuning, making $p(x_k|x_{<k}^p)$ a fixed perturbation term, is incorrect: the language model weights *are* updated during instruction tuning, so the language prior is being modified. While the section is framed as "explanation" rather than proof, the claimed mechanism—that perturbation training steers optimization toward an image-only model—lacks rigorous support.

- **The random perturbation ablation partially undermines the claimed mechanism.** Table 5 shows that random (non-adversarial, non-contextual) text perturbations also reduce hallucinations (HalFscore 50.7 vs. baseline 49.2; CHAIRs 52.4 vs. 54.2). While targeted perturbations improve over random (HalFscore 52.2 vs. 50.7), the gap is modest. This suggests the primary effect is generic disruption of text processing that forces the model toward the visual modality—not specifically the "adversarial perturbation that aligns with general knowledge but conflicts with visual content" as claimed. The paper's analysis (Section 5.3) attributes the difference to "more relevant perturbations exert a stronger disruptive effect," but provides no direct evidence that targeted perturbations activate language priors in the hypothesized manner (e.g., no attention weight analysis or probing experiments).

- **Figure 2, the paper's showcase comparison, demonstrates a severe scene-level hallucination from PerturboLLaVA.** The method misidentifies the sport entirely (describing tennis players as "playing badminton games" with "badminton rackets" during "the Rio 2016 Olympics"). While the caption acknowledges these as hallucinations (red text), the figure caption claims PerturboLLaVA "describes rich image details more accurately"—directly contradicted by a wholesale scene misidentification that is more severe than the competing methods' errors (extra chairs, ball position). This single example does not invalidate the quantitative results, but it raises questions about what *types* of hallucinations remain, and whether the method trades minor attribute errors for rarer but more severe scene-level errors.

### Minor

- **Evaluation limited to a single model backbone (LLaVA1.5-7B)**. Without testing on other model families (e.g., Qwen2-VL, InternVL2), it is unclear whether perturbative training generalizes beyond LLaVA1.5's specific architecture and training pipeline. The paper evaluates other models for HalFscore validation (Table 2), but not for the perturbation method itself.

- **HalFscore validation is limited in scale.** The human evaluation (Table 6) uses only 4 methods × 12 pairwise comparisons, and does not report inter-annotator agreement or variance across GPT-4o runs. While the Pearson correlations (80.7% precision, 78.1% recall) are encouraging, the sample size is small for establishing metric reliability.

- **The "no additional computational overhead" claim in the abstract is imprecise.** It refers only to inference cost; generating 160k perturbation texts via GPT-4o is a substantial upfront data generation cost. Table 1 is transparent about this ("No extra data generation: ✗"), but the abstract's wording is misleading.

- **Potential confound in perturbation text generation.** The GPT-4o prompt for generating perturbation texts provides access to the image, question, *and answer* (Section 4.1). Although the instruction says "without disclosing the answer," there is no verification that perturbation texts never leak answer information. If perturbation texts sometimes contain hints about the correct answer, the training signal becomes "trust this text" rather than "ignore language priors and look at the image."

### Trivial
None.

## Nice-to-Haves

- Probing experiments measuring attention weights on perturbation tokens vs. question tokens before/after training would directly support or refute the claimed mechanism of language prior suppression.
- Fine-grained error analysis by hallucination type on a larger sample would reveal whether PerturboLLaVA systematically shifts hallucination types (as Figure 2 suggests) rather than uniformly reducing them.
- Evaluation on at least one additional model family to demonstrate generalizability of the perturbation training approach.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic: "scaling has not proven effective" claim is asserted without citation.** This is a motivation statement about the current state of MLLM hallucination research, not a formal claim requiring citation. The paper's overall argument does not hinge on this assertion.

- **Harsh critic: perturbation texts leak answer information is a "significant confound."** While worth flagging (moved to Minor), calling it a "significant confound" without evidence that leakage actually occurs is speculative. The concern is reasonable to raise but should be at Minor level.

- **Strength finder: "Mathematical grounding links perturbative training to Bayesian decomposition."** This strength is removed because the mathematical framework has been identified as having unjustified assumptions; a strength that conflicts with a verified weakness should be dropped.

- **Harsh critic: Table 5 shows cherry-picking of the reported configuration.** The paper tests multiple versions and reports Version 1 as the main result. While Version 1 gets the best HalFscore but worst CHAIR among the targeted versions, this is a genuine trade-off the paper discusses, not cherry-picking. The ablation is transparent about these trade-offs.

- **Harsh critic: HalFscore inherits the reliability concerns of all LLM-as-judge metrics.** While true at a general level, this is a generic concern applicable to any GPT-4o-based metric. The paper provides human correlation data (Table 6) showing HalFscore outperforms MMHalBench in alignment with human judgment, which partially addresses this concern. The specific concern about validating the triplet extraction and graph matching subcomponents is kept as Minor.

## Novel Insights

The most interesting observation across the reviews is the tension between PerturboLLaVA's two contributions: HalFscore reveals that hallucination and coverage are *separable* dimensions (a model can reduce hallucinations at the cost of completeness), yet the perturbation training method itself exhibits this exact trade-off—stronger perturbation versions reduce hallucinations but produce shorter, less complete captions (Table 5, recall drops from 46.5 to 46.0–46.1). This suggests the method is partially achieving hallucination reduction by making the model more cautious rather than more visually grounded, which is consistent with the random perturbation baseline also working. The real test for the "language prior suppression" claim would be whether perturbation training specifically reduces *language-biased* errors (e.g., describing a green banana as yellow) while preserving *visually grounded* details, which the paper does not isolate.

## Suggestions

- Replace the Figure 2 example with one where PerturboLLaVA genuinely produces a more accurate description, or add a second example showing success. The current showcase actively undermines the paper's claims.
- Add a targeted experiment isolating language-prior-specific errors: construct a test set where the correct answer contradicts common knowledge (e.g., green bananas, upside-down text) and measure whether PerturboLLaVA specifically reduces these errors more than random perturbation does.
- Temper the "outperforming existing approaches" claim to specify which metrics and dimensions, acknowledging RLAIF-V's advantage on CHAIR.
- Report variance across random seeds and GPT-4o generation runs for the main results.

## Evaluation

**Originality:** Moderate. The perturbation training idea is creative and the HalFscore precision/recall decomposition is a genuine advance, but both have limitations. The mathematical framework is adapted from Clark et al. (2019) with assumptions that don't hold well in the MLLM setting.

**Importance of research question:** High. Hallucination in MLLMs is a critical problem with significant practical implications.

**Claim support:** Mixed. Quantitative improvements exist but are modest on the paper's own metric (+3.0 HalFscore over baseline), non-existent on CHAIR vs. RLAIF-V, and the claimed mechanism is not well-isolated from generic noise effects.

**Soundness of experiments:** Limited by single-model evaluation, small human validation study, and lack of mechanistic analysis.

**Clarity:** Good. The paper is well-structured and the intuition (Figure 1, 5) is clearly communicated.

**Value to community:** Moderate. HalFscore could be a useful addition to the evaluation toolkit if better validated; the perturbation method is a practical but not clearly superior approach.

## Calibration

**Anchors compared:**
- **Differential Transformer** (avg 8.0, Oral): Novel architecture with rigorous evaluation across many tasks. Far stronger contribution than PerturboLLaVA.
- **TAME** (avg 7.0, Poster): Eigenspectrum-based decoding for MLLM hallucination. Similar domain, simpler method with cleaner theoretical grounding. PerturboLLaVA is weaker due to its questionable math and selective claims.
- **CHiP** (avg 6.33, Poster): DPO for MLLM hallucination with strong empirical results but unclear theory (soundness 2). PerturboLLaVA is comparable—both have real results but shaky theoretical foundations. CHiP has stronger empirical gains but no metric contribution.
- **RITUAL** (avg 5.0, Withdrawn): Random image transformations for hallucination. Simple method, limited novelty, weak theory. PerturboLLaVA is stronger due to the HalFscore contribution and more thorough ablation.
- **GACD** (avg 4.75, Withdrawn): Gradient-based hallucination mitigation with weak experiments and overclaimed results. PerturboLLaVA is somewhat stronger.
- **EDU-RAG** (avg 2.33, Withdrawn): Trivial contribution. PerturboLLaVA is clearly stronger.

PerturboLLaVA sits between RITUAL (5.0) and CHiP (6.33), closer to the lower end due to the selective "outperforming" claim, questionable math framework, poor showcase figure, and RLAIF-V's dominance on the most established metric. The HalFscore contribution adds value but with limited validation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>