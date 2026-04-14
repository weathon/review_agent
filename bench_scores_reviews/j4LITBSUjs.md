## Summary

This paper introduces two contributions targeting hallucination in Multimodal Large Language Models (MLLMs): (1) *HalfScore*, a graph-based precision/recall metric that decomposes caption fidelity at the level of objects, attributes, and relations using GPT-4o-extracted triplet structures; and (2) *PerturboLLaVA*, a training strategy that prepends adversarially generated misleading text during instruction tuning to force the model to rely on visual rather than linguistic evidence. Experiments on LLaVA1.5-7B show consistent gains in HalfScore, CHAIR, HallusionBench, MMBench, and CCBench, and the training method complements decoding-based strategies such as OPERA.

---

## Strengths

- **Training-free inference with measurable improvement across hallucination and general benchmarks.** Unlike OPERA and VCD, which incur 2–5× inference cost (Table 1), PerturboLLaVA adds zero inference overhead while achieving +3.0 HalfScore over the LLaVA1.5 baseline and improvements across all general benchmarks in Table 3 (MMB +1.6, CCBench +1.2). Both VCD and RLAIF-V degrade general performance, making this a distinctive result.

- **Complementarity with decoding strategies is empirically demonstrated.** OPERA+Ours achieves HalfScore 52.8, higher than either technique alone (51.9 OPERA, 52.2 Ours), and further reduces CHAIRs from 36.1 to 33.1. This is concrete evidence that training-time and decoding-time mitigations capture different aspects of the problem.

- **Perturbation relevance ablation is informative.** Table 5's Random baseline shows that random text perturbation yields only partial gains (HalfScore 50.7 vs. Version1's 52.2), demonstrating that targeted, semantically adversarial perturbations are materially important — not just any textual disruption. This is a non-obvious result that advances understanding of why the method works.

- **HalfScore captures precision–recall tradeoff neglected by CHAIR and MMHalBench.** Existing metrics either favor conservative models (CHAIR rewards brevity) or give holistic LLM-judge ratings without decomposing accuracy vs. completeness. The graph-based F-score formulation fills a genuine gap for evaluating dense captioning. Human correlation coefficients of 78.1/80.7 (Recall/Precision) and the comparison showing HalfScore correlates better with human precision judgments than MMHalBench (80.7 vs. 71.7) are meaningful supporting data.

---

## Weaknesses

### Fatal

*(None, but the first two Major issues are serious enough to require revision before acceptance.)*

### Major

- **Figure 2 contains an outright hallucination in PerturboLLaVA's own output.** The method's primary qualitative showcase describes a tennis image as "two women playing badminton games on a court during the Rio 2016 Olympics" and refers to "badminton rackets." The figure's caption simultaneously asserts that this output "contains more accurate descriptions." This directly contradicts the paper's central contribution and appears to have gone undetected. Either Figure 2 is the wrong example (must be replaced with a correct one) or the highlighted text is miscategorized. This cannot stand as-is.

- **The precision improvement is substantially confounded by output shortening.** Table 5 reveals a clear pattern: as perturbation strength increases from Version1 to Version3, output length drops from baseline 100 to 89, 82, and 78. Precision improves (+6.2 vs. baseline for Version1) while recall barely changes (+0.7), and stronger versions reduce recall further. This pattern is consistent with a model that simply generates shorter, more hedged captions rather than one that has genuinely improved visual grounding. Without a length-controlled evaluation—running all methods at matched output lengths and re-computing precision—it is impossible to distinguish "fewer hallucinations" from "fewer tokens overall." This confound directly challenges the core mechanistic claim.

- **Generality of PerturboLLaVA is unestablished.** All experiments use a single base model (LLaVA1.5-7B). The paper claims a general training strategy for any MLLM, but without at least one replication on another architecture (e.g., LLaVA-NeXT or a Qwen2-VL derivative), the conclusion that the method is a general solution is unsupported. This is especially important for an ICLR submission proposing a "standard strategy."

### Minor

- **The mathematical derivation in Section 4.2 is not convincing and may do more harm than good.** The critical step at Equation (8) invokes conditional independence between $x_{<k}^p$ and $x_{<k}^{-p}$, but these quantities are defined as complementary decompositions of the same autoregressive context — making their independence deeply implausible. The handling of $p(x_k)$ as negligible under a "sufficiently uniform dataset" is asserted, not shown. The conclusion does not follow rigorously. This section reads as post-hoc rationalization rather than analysis. It would be stronger to recast it as an explicit analogy to Clark et al. (2019)'s debiasing framework, without claiming a derivation that does not hold.

- **GPT-4o dependency for both metric computation and training data generation raises reproducibility concerns, and the costs should be quantified.** Table 1 correctly marks "No extra data generation: ✗" for Ours, but the abstract asserts the method incurs "no additional computational overhead" — a statement that covers inference but not training pipeline cost. Generating perturbations with GPT-4o for 160k samples is a real one-time cost that should be quantified in the main paper (not just the appendix). Similarly, per-evaluation API costs for HalfScore should be disclosed to allow others to assess adoption feasibility.

- **HalfScore's reliability as a metric is under-validated.** The metric depends entirely on GPT-4o for triplet extraction and graph matching. Key validation gaps: (a) no prompt sensitivity or run-to-run variance study; (b) no failure-case analysis for extraction errors (e.g., granularity mismatches, synonym handling, relational triplet ambiguity); (c) the human study uses only 4 methods × 12 pairwise comparisons, which is small for a metric intended as a new standard; (d) no inter-annotator agreement is reported. The correlations reported are encouraging, but insufficient to fully validate a new benchmark tool.

- **The "over-reliance on language prior is the root cause of hallucination" is stated too strongly.** The paper provides illustrative examples (Figure 1) and a plausible mechanism, but other factors (weak visual grounding in the encoder, instruction-tuning distribution shift, exposure bias) are not seriously considered. Framing this as a contributor rather than the singular root cause would be more accurate and defensible.

### Tiny

- RLAIF-V dramatically outperforms all methods on CHAIR (18.1 vs. 36.1) and HalBench (51.3 vs. 47.5), while Ours exceeds RLAIF-V on HalfScore and general benchmarks. The paper acknowledges that RLAIF-V uses the much stronger LLaVA-Next 34B as reward model, making comparison difficult to interpret. The discussion of which method is actually better should be more nuanced — both have different trade-off profiles.

- Table 5 shows Version3 achieves 49.6 on HalBench (best among versions) but the final system uses Version1. A brief explanation of this operating-point choice would help readers understand the design decision.

---

## Nice-to-Haves

- **Run on a second base model** (e.g., LLaVA-NeXT-7B or InternVL2-8B) to establish transfer, even in the appendix. This would substantially strengthen the generality claim.

- **Ablation on perturbation data scale** (e.g., 25/50/100% of the 160k samples) to show the method is practical even with fewer GPT-4o API calls, and to understand the cost-performance curve.

- **Compare GPT-4o-generated perturbations against open-source LLM alternatives** (e.g., LLaMA-3-8B-generated perturbations). Given that the Random baseline still shows partial effects, a cheaper perturbation source might achieve most of the gains, which would be a meaningful finding for accessibility.

- **Attention visualization** comparing PerturboLLaVA vs. LLaVA1.5 on the same input would provide direct mechanistic evidence for the claimed shift from linguistic to visual attention, rather than relying on behavioral output comparisons.

- **Evaluate on VQAv2 or GQA** to confirm that perturbation training does not hurt performance on standard VQA benchmarks.

- **Discuss the risk of over-skepticism toward legitimate textual context** (e.g., in retrieval-augmented or document-reading settings where prepended text genuinely should be trusted). This is a real deployment concern worth noting in Limitations.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **[REMOVED — misread]** Harsh critic claimed the abstract says the method incurs "no additional computational overhead" and read this as covering training cost. On re-reading, the abstract sentence refers to inference overhead, and Table 1 correctly marks "No extra data generation: ✗". This is a factual misread of the abstract, though the intro does separately make a slightly broader claim ("without additional training or inference costs") that is worth softening.

- **[REMOVED — scope creep]** Demanding formal theoretical proofs for the training-based mechanism. This is an empirical/systems paper; the mathematical section is adequately flagged in the review as informal but the absence of a theorem is not a weakness for this type of contribution.

- **[REMOVED — standard practice]** Requests for statistical significance tests / confidence intervals on MMBench, SEED, CCBench results. Single-run evaluation is the accepted norm for these large-scale benchmarks in the MLLM community; demanding multiple seeds is not standard.

- **[REMOVED — non-standard demand]** Requesting that HalfScore be tested with an alternative judge model as a validation. While interesting as a nice-to-have, this level of metric validation is not required for ICLR.

- **[REMOVED — generic strength]** "The paper is well-written and the topic is important." True but applies to any paper.

---

## Novel Insights

The most genuinely novel observation to emerge from the reviewing process — not explicitly foregrounded in the paper — is the tension between the paper's own ablation data and its mechanistic claim. Table 5 shows a monotonic trade-off: stronger perturbation → shorter output → higher precision → lower recall. The "Random" baseline already achieves partial precision gains with random text injection. Together, these results suggest that the method may operate partly through a *response conservatism* mechanism — training the model to output shorter, more hedged captions by increasing uncertainty in the generation prior — rather than (or in addition to) a genuine *visual grounding* improvement. This is testable: a length-controlled re-evaluation would either confirm or refute the grounding interpretation. If confirmed, it would actually be a useful contribution in its own right — showing that conservatism injection is itself a viable hallucination-reduction strategy — but the paper should acknowledge this interpretation rather than claiming purely improved visual attention.

---

## Suggestions

1. **Replace Figure 2** — PerturboLLaVA's output in this figure says "badminton" and "badminton rackets" for a tennis image. Either select a different qualitative example or, if this is a rendering error in the submission PDF, correct and verify it.

2. **Conduct a length-matched evaluation**: generate captions with all baselines truncated or constrained to the same length distribution as PerturboLLaVA, then re-run HalfScore and CHAIR. This is the critical experiment to isolate whether precision gains are attributable to conservatism or genuine grounding.

3. **Reframe Section 4.2** as an intuition/analogy to Clark et al. (2019) rather than a formal derivation. Remove Equations (5)–(10) or relabel them as "informal sketch." The current form makes verifiably unjustified independence assumptions and weakens the paper's credibility.

4. **Add a cost table** (main paper, not appendix): report GPT-4o API calls, estimated cost in USD, wall-clock data preparation time, and GPU-hour difference between baseline LLaVA1.5 training and PerturboLLaVA training. This resolves the "minimal cost" ambiguity and helps reproducibility.

5. **Report inter-annotator agreement** for the human study and clarify the evaluation protocol in the main paper (how many annotators, what instructions, how ties were resolved).

6. **Add at least one additional backbone** (even in the appendix) to support the generality claim before submitting a revision.

---

**Overall evaluation:** The core training idea is simple, practical, and shows real consistent gains on a representative base model — these are genuine virtues. However, the paper's primary qualitative example contains a hallucination from the proposed method itself; the central mechanistic claim (better visual grounding) is confounded by output length effects; and the method is only demonstrated on one model. The metric contribution (HalfScore) is useful but under-validated. These issues make the paper's current form fall below the ICLR acceptance bar. With a corrected Figure 2, a length-controlled ablation, and a second backbone experiment, this would be a solid contribution.

- **Novelty:** Moderate — adversarial text injection during training is not conceptually novel, but its application to MLLM hallucination is sensible and the design choices are carefully motivated.
- **Technical soundness:** Weak-to-moderate — the mathematical justification is formally incorrect, and the key confound (length vs. grounding) is unaddressed.
- **Empirical support:** Moderate — results are consistent across multiple benchmarks on one model, but critically lack length-controlled analysis and multi-model validation.
- **Significance:** Moderate-to-high practically — inference-free hallucination reduction is a deployable contribution if the method generalizes.
- **Clarity:** Adequate overall, but the blurring between "HalfScore as a graph metric" and "HalfScore as a GPT-4o pipeline" and the misleading cost claims need correction.