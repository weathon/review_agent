Now I have a thorough understanding of the paper and the calibration landscape. Let me write the final review.

## Summary

The paper proposes Copy-Paste, a generation paradigm for RAG that encourages high lexical copying from context as a proxy for contextual faithfulness, instantiated as CopyPasteLLM through a two-stage pipeline: (1) Copy-Paste-Prompting methods (CP-Order, CP-Link, CP-Refine) that generate high-copying response candidates, and (2) DPO training on preference data curated from those candidates using multi-criteria filtering and Elo-style LLM-as-Judge ranking. CopyPasteLLM achieves 12.2%–24.5% accuracy improvements on FaithEval counterfactual settings over the best baseline using only 365 training samples.

## Strengths

- **Strong empirical results with impressive data efficiency.** CopyPasteLLM achieves 92.8% accuracy on FaithEval (Llama-3-8B) using 365 seed pairs compared to Context-DPO's 80.2% with 18,000 samples. The improvements are consistent across three model families and four datasets, and the method also improves performance on original (non-counterfactual) settings (Table 3), suggesting it doesn't catastrophically degrade parametric knowledge on the tested benchmarks.

- **Clear and intuitive motivation.** The inverse correlation between copying degree and hallucination density on RAGTruth (Section 2.2, Figure 1) provides a simple, empirically grounded starting point. The three prompting strategies span a sensible faithfulness–fluency design space, with CP-Refine achieving the best balance.

- **Informative mechanistic analysis.** The Context-Parameter Copying Capturing method extends KTC to full CoT trajectories, and the finding that CopyPasteLLM recalibrates parametric knowledge confidence (rather than enhancing contextual processing) is an interesting and somewhat counterintuitive result that adds mechanistic depth beyond showing *that* it works.

- **Ethics acknowledgment.** The paper explicitly acknowledges the risk of over-reliance on potentially biased or incorrect context (Section 7), which is important for a method that trains models to trust context.

## Weaknesses

### Major:

- **Copying as a proxy for faithfulness is a conceptual gap in non-counterfactual settings.** The paper's core move—treating high copying degree as an operational proxy for contextual faithfulness—works reasonably in the evaluated counterfactual settings (FaithEval, ConFiQA) where the context is explicitly designed to be the ground truth that the model should follow. However, the paper makes broader claims: the abstract states that "copied content itself serves as direct evidence of faithfulness," and the conclusion asserts it is an "elegant solution to RAG attribution challenges." These claims overreach what the evidence supports. In real-world RAG systems, retrieved context may be irrelevant, partially incorrect, or contradictory. A model trained to copy verbatim from context may propagate errors from these contexts. While Section 7 briefly acknowledges this risk, the experimental evaluation does not test behavior with noisy, irrelevant, or adversarial contexts—only contexts that should be trusted. This is a significant limitation given the paper's framing around medical applications where context quality cannot be guaranteed.

- **The pipeline entangles multiple interventions; it is unclear what drives the gains.** The CopyPasteLLM training pipeline combines (a) specialized copy-paste prompting strategies, (b) multi-criteria filtering (AlignScore, MiniCheck, κ, δ, perplexity, embedding similarity), (c) Elo-style LLM-as-Judge ranking with hallucination-type detection, and (d) gold-answer augmentation of preference pairs. With all of these operating simultaneously, it is impossible to tell from the main paper which components are essential. While Appendix G reportedly contains ablations, the main text provides no evidence that the gains come specifically from the copy-paste paradigm rather than, e.g., the high-quality curation pipeline or the gold-answer engineering. The headline claim of "copy-paste as paradigm" is therefore under-supported causally.

- **No evaluation of parametric knowledge degradation.** Related work ("Is Factuality Enhancement a Free Lunch?," BALCONI) demonstrates that improving context faithfulness can degrade a model's ability to use its own parametric knowledge when context is absent or irrelevant. While Table 3 shows improvements on PubMedQA and ConFiQA original contexts, these are all settings where context is provided. The paper does not evaluate on standard benchmarks (e.g., MMLU, GSM8K) without context to verify that CopyPasteLLM has not learned a blanket "always trust context" heuristic that harms general reasoning. This is a critical gap for a method whose mechanistic analysis shows it "reduces reliance on parametric knowledge."

### Minor:

- **The "50× data efficiency" claim is somewhat misleading.** While 365 seed pairs are used, the pipeline generates ~6 candidates per pair and constructs ~5 preference pairs per sample (Section 3.2), yielding roughly 1,825 preference comparisons. The total inference compute for candidate generation, filtering, Elo tournaments, and CP-Refine writer-reviewer loops using 72B/671B models is substantial. Comparing by seed count without acknowledging this expansion inflates the efficiency narrative.

- **The Context-Parameter Copying Capturing method uses a coarse proxy for knowledge source classification.** Classifying tokens as "contextual" based on string overlap with the provided context and "parametric" based on preference in context-free runs conflates multiple effects. Common words, discourse markers, and function words appear in virtually any context and would be classified as "contextual" regardless of their actual knowledge source. The UMAP visualizations (Figure 4) provide qualitative support but no quantitative cluster separability metrics. These limitations should be stated more explicitly.

- **Fluency-cost tradeoff deserves deeper analysis.** Table 2 confirms that CP-Order (the highest-copying variant) achieves top faithfulness but notably worse fluency. The paper acknowledges this but does not provide human evaluation of response naturalness or analyze the Pareto frontier systematically.

## Nice-to-Haves

- **Evaluate on tasks requiring multi-hop reasoning or synthesis** (e.g., HotpotQA, 2WikiMultiHopQA) where verbatim copying is less effective, to delineate the boundaries of the Copy-Paste paradigm.

- **Test behavior with noisy/irrelevant retrieval** where some retrieved documents are off-topic or contain errors, which is the realistic deployment scenario for RAG systems.

- **Evaluate on general capability benchmarks without context** (e.g., MMLU) as a sanity check for parametric knowledge preservation.

- **Provide failure case analysis** showing when and why CopyPasteLLM produces suboptimal outputs, rather than only reporting aggregate metrics.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "GPT-4o comparison is unfair"** — The paper notes this as "GPT-4o's reported 47.5%" (line 177) and references Appendix Table 6. This is a standard reference to prior reported numbers, not a head-to-head comparison. It provides context for the difficulty of the benchmark, not a claim of methodological superiority over GPT-4o.

- **Harsh Critic: "No numerical correlation coefficients or regression analyses" for RAGTruth** — The visual analysis in Figure 1 showing kernel density estimation with hallucination density overlays is a legitimate form of evidence. Correlation coefficients would strengthen the claim, but their absence does not invalidate the observed pattern across 6 models.

- **Harsh Critic: "No ablation"** — The paper explicitly references ablation studies in Appendix G (line 178: "for ablation studies and training dynamics, see Appendix G"). The claim that there are no ablations is factually incorrect; the valid critique is that key ablation results are not presented in the main text.

- **Harsh Critic: "Hit Rate metrics... not fully defined"** — FaithEval's evaluation protocol is defined in its original publication (Ming et al., 2025). Hit rate with exact match on lengthy answers is a standard evaluation metric for this benchmark.

- **Harsh Critic: "CP-Refine uses same model as base"** — The paper states the writer-reviewer loop uses specific models; this is an implementation detail, not a methodological flaw. The use of larger models (Qwen-72B, DeepSeek-V3) for candidate generation is specified in Table 2.

- **Spark: "CP-Refine uses Qwen-72B and DeepSeek-V3 as writers/reviewers" as a weakness about hidden cost** — While the computational cost is relevant, reviewers should not question the existence or availability of these models. The cost concern is valid but addressed under the data efficiency point above.

- **Harsh Critic: "Context-Parameter Copying Capturing is not validated with ground truth"** — While a ground-truth sanity check would strengthen the method, this tool is presented in service of interpretation (RQ3), not as a primary contribution. The claims about "recalibration" are supported also by the logit analysis (Figure 3), not solely by the UMAP visualization.

## Novel Insights

The finding that CopyPasteLLM achieves its improvements primarily by suppressing parametric knowledge confidence rather than enhancing contextual knowledge representations (Figure 4) is genuinely interesting. This suggests that the key mechanism may not be "learning to attend to context more" but rather "learning to distrust one's own internal knowledge when context is present." This aligns with and extends the "Context-Parametric Inversion" phenomenon identified by Goyal et al. (2025), but from the opposite direction: rather than instruction finetuning reducing context reliance, here preference optimization explicitly increases it via parametric suppression. This connection is not drawn in the paper but could enrich the broader understanding of context-parametric dynamics.

## Suggestions

- **Move key ablation results to the main paper.** At minimum, show (1) DPO with the same pipeline but without copy-paste enforcement, and (2) Copy-Paste preference data without LLM-as-Judge filtering. This would establish whether the copy-paste paradigm itself or the curation pipeline drives performance.

- **Add a "no-context" evaluation** on a standard benchmark like MMLU to confirm CopyPasteLLM does not degrade parametric capabilities when context is absent—this is a basic sanity check for any method that trains models to favor context.

- **Test on an irrelevant/noisy context setting.** Even a simple experiment where random or off-topic documents are prepended as context would reveal whether CopyPasteLLM blindly copies from context or can discriminate relevant from irrelevant content.

## Evaluation

**Originality**: The Copy-Paste paradigm as a generation strategy is intuitive rather than deeply novel, but the two-stage pipeline connecting prompting to preference learning and the mechanistic analysis add useful novelty. The three CP-Prompting strategies are incremental (extract-then-reorder, extract-with-connectors, iterative refinement).

**Importance of research question**: Contextual faithfulness in RAG is an important and timely problem, especially for medical applications.

**Claims well supported**: Partially. The empirical gains are strong, but the causal attribution to "copy-paste" specifically (rather than the multi-component pipeline) is under-supported. The broader claims about copied content being "inherent faithfulness evidence" are not demonstrated beyond counterfactual QA settings.

**Soundness of experiments**: The experiments cover multiple models and datasets with consistent improvements, but there are significant gaps—no irrelevant/noisy context evaluation, no general capability benchmarks, and key ablations in appendix only.

**Clarity**: The paper is generally well-written and structured, though the pipeline complexity makes it somewhat hard to follow. The distinction between Stage 1 (prompting) and Stage 2 (DPO) is clear.

**Value to research community**: The method demonstrates strong results on an important problem, and the mechanistic insight about parametric suppression is valuable. However, the overclaiming about data efficiency and the scope of applicability limit the immediate impact.

**Calibration**: Compared to SCOPE (poster, 6,6,6,6,6 — self-supervised faithfulness method with similar profile and limitations), BALCONI (reject, 6,6,3,6 — method for balancing context vs. parametric knowledge with weaknesses in evaluation scope and novelty), and "Context-Parametric Inversion" (oral, 8,8,8,8 — strong mechanistic analysis of context-parametric interaction), this paper falls between these anchors. It has stronger empirical results than BALCONI and more practical impact than the inversion analysis, but weaker mechanistic grounding than the oral paper and similar evaluation concerns to SCOPE. The overclaiming about data efficiency and lack of evaluation beyond counterfactual QA settings are notable weaknesses.

## Score and Decision

This paper presents a well-motivated and empirically strong method for contextual faithfulness, with meaningful improvements across multiple benchmarks and an interesting mechanistic finding. However, the core causal narrative is entangled by a multi-component pipeline without main-text ablations, the broader claims about copying-as-faithfulness-proxy extend beyond the evaluated settings, and critical evaluations (no-context capability preservation, noisy/irrelevant context behavior) are missing. These are not fatal flaws but they significantly weaken the paper's claims and generalizability argument.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>