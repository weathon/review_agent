## Summary

The paper proposes "Copy-Paste," a generation paradigm for RAG that directly embeds contextual fragments into responses to reduce hallucinations, motivated by an observed inverse correlation between copying degree and hallucination density on RAGTruth. The paradigm is instantiated through CopyPasteLLM, which uses two-stage high-copying preference training (DPO) with only 365 input query–context pairs. The paper also introduces Context-Parameter Copying Capturing for mechanistic analysis, claiming that CopyPasteLLM recalibrates parametric knowledge confidence rather than enhancing contextual representations. CopyPasteLLM achieves 12.2%–24.5% accuracy improvements on FaithEval over the best baseline.

## Strengths

- **Strong empirical gains with remarkable data efficiency.** CopyPasteLLM achieves substantial improvements on FaithEval counterfactual (up to 24.5% over baselines) using only 365 input pairs to construct preference data—orders of magnitude less labeled data than Context-DPO (18,000), Canoe (10,000), or ParamMute (32,580). Even accounting for synthetic expansion (~5 pairs per sample), this is a meaningful efficiency result. Table 3 also shows consistent improvements on non-counterfactual settings (PubMedQA, ConFiQA).

- **Well-designed progressive prompting pipeline.** The three Copy-Paste-Prompting methods (CP-Order → CP-Link → CP-Refine) span hard constraints to soft iterative refinement, and Table 2 demonstrates that CP-Refine achieves the best faithfulness–fluency trade-off. The automated pipeline from candidate generation through multi-criteria filtering to Elo-style LLM-as-Judge ranking is fully specified and requires no human annotation.

- **Insightful mechanistic analysis.** The Context-Parameter Copying Capturing algorithm extends prior short-answer probing (KTC) to full Chain-of-Thought trajectories, and Figures 3–4 reveal that CopyPasteLLM suppresses parametric knowledge rather than enhancing contextual representations—a non-obvious finding with implications for understanding faithfulness interventions.

- **Comprehensive experimental scope.** Evaluation spans 4 base model sizes (7B–72B prompting, 7B–8B for trained models), 4 benchmarks, and both counterfactual and original contexts, providing broad coverage of the method's behavior.

## Weaknesses

### Major:

- **Conflating counterfactual context obedience with contextual faithfulness.** The headline gains (12.2%–24.5%) come primarily from FaithEval's counterfactual setting, where context deliberately contradicts world knowledge. Performing well there means the model overrides correct parametric knowledge to trust wrong context—an effect better described as "context obedience" than "faithfulness." The paper acknowledges risks in its Ethics Statement ("over-reliance on copied content may lead to verbatim reproduction of potentially biased or incorrect source material") but does not empirically evaluate this failure mode. No experiments test CopyPasteLLM under noisy, partially incorrect, or adversarial contexts—precisely the conditions most relevant to the paper's own medical motivation. This gap means the core empirical claim ("mitigate hallucinations") is only demonstrated in a synthetic setting that does not reflect realistic RAG deployment.

- **Copying degree and faithfulness metrics are partially circular.** The paper proposes copying degree (κ, δ) as both the optimization target and (indirectly) the evaluation criterion. High-copying responses will by construction score well on context-overlap metrics like AlignScore and MiniCheck, which themselves measure n-gram or semantic overlap with context. Table 1 reports only Accuracy and Hit Rate, which partially avoids this for the trained model, but Table 2 (prompting stage) relies heavily on AlignScore and MiniCheck where the circularity is most acute. The foundational motivation (Figure 1: inverse correlation between copying and hallucination) is also purely correlational—no experiment establishes that *forcing* a model to copy more *causally* reduces hallucinations rather than merely that models which already attend better to context also happen to copy more.

- **Preference data construction entangles copying with correctness.** The pipeline appends gold answers to preferred (high-copying) candidates and incorrect answers to rejected candidates (Section 3.2). This means chosen responses are simultaneously more copying AND more correct, making it impossible to determine whether CopyPasteLLM learned "prefer context" or "produce correct answers." No ablation disentangles these: e.g., high-copying but wrong preferred responses, or low-copying but accurate preferred responses. Without such controls, the claim that CopyPasteLLM "internalizes contextual trust" is not convincingly supported—an equally plausible read is that DPO trained on responses that are both more correct and more copy-like.

- **The sample-efficiency comparison is not apples-to-apples.** The paper compares "365 query–context pairs" against baselines counted in total training samples (18,000 for Context-DPO). Since each input pair generates ~5 preference pairs, the actual training data is ~1,800 pairs—not ~1/50th of 18,000 but closer to ~1/10th. The comparison also mixes different DPO objectives, training data sources, and possibly different evaluation splits. The data-efficiency narrative is directionally correct (substantially less data) but the "1/50th" framing is misleading.

### Minor:

- **No evaluation of answer quality beyond accuracy in CopyPasteLLM stage.** Section 2.1 explicitly balances faithfulness, query relevance, and fluency as objectives. Table 2 shows these trade-offs for prompting, but Tables 1 and 3 report only Accuracy and Hit Rate, leaving whether CopyPasteLLM degrades fluency or relevance unverified.

- **Mechanistic conclusions overclaim relative to probe granularity.** Context-Parameter Copying Capturing classifies tokens as "contextual" if they appear in the provided context and "parametric" if preferred in a context-free run. Common words, subword tokens, and semantically driven generations that don't overlap lexically with context are misclassified. The UMAP visualizations (Figure 4) are qualitative—the paper reports no quantitative cluster separation metrics, and no causal interventions (e.g., ablating identified components) confirm the "recalibration" claim.

- **No simple copy-inducing prompt baseline.** A natural comparison is simply instructing the model to "quote relevant sentences from the context verbatim in your answer." Without this, it is unclear how much gain comes from the pipeline's complexity vs. the basic instruction to copy.

### Trivial:

- Some metric values in Table 2 appear anomalous (e.g., perplexity of 330.8 for Llama-3.1-8B on ConFiQA-QA Attributed), which the paper flags as possible parser issues. This does not affect the core conclusions.

## Nice-to-Haves

- **Evaluation under noisy or adversarial context** to quantify the over-reliance risk empirically, even on a small scale.
- **SFT ablation on same data** (train via SFT on the 365 high-copying responses) to determine whether DPO is necessary.
- **Ablations decorrelating copying from correctness** in preference data to isolate the true mechanism.
- **Per-category breakdowns on FaithEval/ConFiQA** to reveal whether gains generalize across question types or concentrate in easy categories.
- **Qualitative output examples** showing what CopyPasteLLM responses actually look like to assess fluency, coherence, and informativeness.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Copying does not guarantee faithfulness because context can be wrong."** This is addressed by the paper's Ethics Statement and by the design intent—Copy-Paste is explicitly about trusting the provided context. The question is whether the method *over-trusts*, which is a valid concern (kept above), but the bare statement that copying wrong context is unfaithful misrepresents the paper's definition of faithfulness (fidelity to provided context, not to ground truth).

- **"Insufficient novelty—the copy idea is intuitive/historical."** The paper's contribution is not merely "copy more" but the specific two-stage pipeline with automated preference data construction and preference optimization. The mechanistic analysis is also novel. Novelty concerns are noted but are not a standalone weakness given the empirical and engineering contributions.

- **"No confidence intervals / statistical significance testing."** Single-run evaluation without variance reporting is the norm in the LLM fine-tuning literature for benchmarks at this scale. Requesting confidence intervals for 365-sample training runs is a methodological preference, not a core flaw.

- **"Cost of preference data construction not analyzed."** The pipeline uses an LLM-as-Judge and multi-criteria filtering, but the paper commits to full automation with no human annotation. The total computation cost is a practical concern but not a methodological flaw, and the paper focuses on sample efficiency of *labeled* data.

- **"Limited model scale (only up to 8B for CopyPasteLLM)."** Copy-Paste-Prompting is evaluated on models up to 671B (DeepSeek-V3). CopyPasteLLM is trained on 7B–8B models due to compute constraints, which is standard for fine-tuning papers. Whether gains transfer to 70B+ is a reasonable extension but not a missing baseline.

- **"No human evaluation of response quality."** The paper uses well-established automated metrics (AlignScore, MiniCheck, accuracy on standard benchmarks). Human evaluation would strengthen but is not standard for this type of work and is a nice-to-have.

## Novel Insights

The paper's most intellectually novel finding is the mechanistic result that CopyPasteLLM achieves greater contextual trust not by enhancing contextual processing but by suppressing parametric knowledge confidence. This "subtraction rather than addition" mechanism, visible in the UMAP analysis (Figure 4, where contextual representations remain nearly co-distributed with the base model while parametric representations diverge), suggests that faithfulness interventions may operate primarily through reducing internal competition rather than strengthening external signal processing—a hypothesis with implications beyond this specific method. However, as noted in Weaknesses, this finding rests on indirect probing and would benefit from causal validation.

## Suggestions

1. **Add evaluation under noisy/adversarial context conditions** (even a small experiment with 10–20% corrupted contexts) to empirically characterize the over-trust boundary and directly address the counterfactual-vs-realistic-faithfulness tension.

2. **Run an ablation with same-size DPO on standard (non-copy-paste) preference data** to isolate the contribution of the copying paradigm from the Elo-ranking and filtering pipeline.

3. **Decouple copying from correctness in preference data** by constructing at least one ablation where preferred responses are low-copying-but-accurate, to test whether the model truly learns contextual trust vs. a stylistic copying heuristic.

4. **Report fluency and relevance metrics for CopyPasteLLM** (not just the prompting stage) to verify that the faithfulness–fluency–relevance trade-off remains balanced after DPO training.

5. **Tone down claims about "contextual faithfulness" and "mitigating hallucinations"** to accurately reflect that the method primarily demonstrates context obedience on synthetic counterfactual benchmarks, and reframe the narrative to acknowledge the trade-off with parametric knowledge preservation.

## Score and Decision

Calibration papers:
- **Trust-Align** (scores: 8,8,8,8) — strong paper on RAG grounding with clean methodology; this paper has weaker theoretical grounding and overclaims.
- **BALCONI** (scores: 6,6,3,6) — context-vs-parametric knowledge trade-off paper; similar concern about over-trusting context, but BALCONI explicitly studied the trade-off this paper ignores.
- **"Is Factuality Enhancement a Free Lunch?"** (scores: 6,6,8) — studies the factuality–faithfulness tension; this paper's core issue mirrors the "free lunch" concern.
- **SCOPE** (scores: 6,6,6,6,6) — self-supervised faithfulness; moderate results, moderate novelty.

This paper has genuinely strong empirical results (large improvements on established benchmarks) and a creative pipeline, but suffers from: (1) equating context obedience on synthetic counterfactuals with real faithfulness, (2) circular metrics in the prompting stage, (3) entangled preference data that confounds copying with correctness, and (4) overclaimed mechanistic conclusions. These are substantive but not fatal—the empirical gains are real and the data efficiency is meaningful. The paper lands below BALCONI (which explicitly studied the trade-off this paper elides) and above papers with weaker results. A score of **5.5** reflects solid engineering and empirical contributions undermined by conceptual overclaiming and evaluation gaps that directly affect the core narrative.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>