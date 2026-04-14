## Summary
This paper introduces **situated faithfulness** — the ability of LLMs to dynamically calibrate trust between internal parametric knowledge and external contexts when the two conflict, as arises in RAG pipelines. To benchmark this, the authors evaluate several QA datasets paired with correct and incorrect contexts, contributing **RedditQA**, a new dataset featuring human-written real-world misinformation from Reddit. They propose two method classes — Self-Guided Confidence Reasoning (SCR) and Rule-Based Confidence Reasoning (RCR) — and a fine-tuning approach, **CR-DPO**, which trains Llama-3-8B via preference optimization over self-sampled confidence-reasoning traces, yielding an average +8.9% gain on situated faithfulness.

---

## Strengths

- **RedditQA fills a genuine benchmark gap.** Unlike all prior work (ClashEval, FaithEval, DynamicQA), which relies on synthetically perturbed incorrect contexts, RedditQA sources incorrect contexts from naturally occurring Reddit posts, providing a qualitatively different test of model robustness to misinformation-as-it-appears-in-the-wild. This is a substantive contribution that no concurrent benchmark offers.

- **The insight that calibration quality ≠ better decision-making is non-obvious and empirically substantiated.** Section 5.3 / Table 3 demonstrates that improving confidence calibration (via isotonic regression, threshold tuning, percentile correction, or self-consistency) does not reliably improve situated faithfulness. The explanation — that a well-calibrated confidence score can still be misaligned with the accuracy-maximizing decision rule — is conceptually precise and not a generic observation. This analysis is one of the most valuable contributions in the paper.

- **The SCR vs. RCR split across model capability levels provides actionable practical guidance.** The finding that strong models (GPT-4o, GPT-4o-mini) benefit from end-to-end SCR while weaker models (Llama-3-8B) are better served by RCR — especially InternalConf via sequence probability — offers concrete guidance for practitioners choosing methods. Tables 1 and 2 clearly support this pattern.

- **CR-DPO ablations are unusually informative.** Table 4 isolates the contributions of CoT, DPO vs. SFT, training task diversity, and trace source (self-sampled vs. GPT-4o). The finding that self-sampled reasoning paths outperform GPT-4o-sourced traces — because confidence reasoning is grounded in the model's own knowledge, which a stronger model cannot share — is a surprising and theoretically motivated result.

- **The RCR signal-rule misalignment taxonomy is a useful contribution.** The paper identifies three distinct failure modes for RCR: (1) flawed/biased rules, (2) noisy/biased confidence signals, and (3) structural misalignment between signal and rule objective. This structured diagnosis goes beyond simply reporting that RCR underperforms.

---

## Weaknesses

### Fatal
None identified.

### Major

- **The central model-capability finding rests on only three models, two of which are closely related GPT-4o variants.** The claim that "stronger reasoning models benefit more from SCR while weaker models benefit from RCR" is a key insight of the paper, but it is supported by exactly three data points: GPT-4o, GPT-4o-mini, and Llama-3-8B. GPT-4o and GPT-4o-mini are not independently diverse architectures. This is far too narrow a base to draw a general conclusion about model capability and method fit. Without additional models spanning a capability spectrum (e.g., Llama-3-70B, Mistral, Gemma), this finding remains suggestive, not established. The paper's framing should be substantially qualified.

- **CR-DPO, the paper's central training contribution, is evaluated on a single model (Llama-3-8B).** All claims about CR-DPO's generality — to unseen tasks, to varying context types — are derived from one architecture and scale. Without evidence across even one additional open-source model, the method cannot be characterized as a general approach to improving SCR in smaller LLMs.

- **No measurement of general capability degradation after CR-DPO.** The paper does not evaluate whether CR-DPO hurts performance on standard benchmarks (e.g., MMLU, reasoning tasks). A model that improves situated faithfulness by learning to systematically distrust external contexts could be practically harmful if it simultaneously becomes overconfident in its parametric knowledge in general QA settings. This omission is a significant gap for any claim of practical utility.

- **Heavy GPT-4o involvement in dataset construction creates potential evaluation circularity.** GPT-4o participates in claim filtering, question generation, context modification, and context verification for several datasets. GPT-4o and GPT-4o-mini are then among the primary evaluated models. This introduces a plausible stylistic or inferential alignment between training data generation and evaluation that is not analyzed. The paper does not quantify or mitigate this risk.

- **The benchmark uses binary correct/incorrect contexts, limiting practical relevance.** All experiments pair each question with a completely correct or completely incorrect context. Real RAG pipelines encounter contexts that are partially correct, partially outdated, or partially relevant. The core claim of "situated faithfulness" in realistic deployment conditions is untested under this most natural setting. This is acknowledged nowhere as a limitation.

### Minor

- **No analysis of source-selection behavior.** The paper measures only final answer accuracy, yet its central framing is about *trust* and *confidence reasoning*. Without a breakdown of how often each method selects the internal vs. external answer across the four quadrants (internal correct/wrong × context correct/wrong), it is impossible to determine whether SCR succeeds by reasoning carefully or simply by defaulting to internal knowledge. A confusion-matrix-style analysis would make the "confidence reasoning" claim more than interpretively asserted.

- **The Figure 2 CR-DPO example contains factual errors in reasoning.** The CR-DPO output states "Richard M. Daley served as the mayor of Chicago from 1955 to 1976" — this conflates Richard J. Daley (father, 1955–1976) with Richard M. Daley (son, 1989–2011). The final answer (Chicago) is correct, but the reasoning contains clear factual inaccuracies. As the paper's only qualitative success case for CR-DPO, this is concerning: it suggests the model learns to *argue against* misleading contexts more forcefully, not necessarily to reason from better-organized knowledge.

- **RedditQA is multiple-choice while other datasets use open-ended QA.** Answer format can substantially affect model behavior under conflicting context (e.g., guessing probability, distractor salience). The "Total" metrics aggregate these into a single number without discussing whether this conflation is appropriate.

- **Dataset statistics for RedditQA are not reported in the main text.** Final dataset size, topic distribution, inter-annotator agreement, and filtering rates are deferred to an appendix. For a benchmark contribution at a venue like ICLR, these statistics should be prominent in the paper body.

- **TACS(LR) is a substantially weakened approximation of the original TACS method.** The paper substitutes the original hidden-state classifier with an LLM prompting approach because hidden states are inaccessible for proprietary models. Conclusions drawn from TACS(LR) performing poorly (e.g., that preprocessing approaches fail) are conflated with conclusions about the original method. This should be framed much more carefully: TACS(LR) failing does not mean TACS fails.

### Tiny

- The SF metric equally weights Acc_t and Acc_f, but in many real deployments, correct contexts are far more frequent than incorrect ones. This makes SF overweight robustness relative to utility. This is acceptable as a benchmark stress-test metric but is sometimes written as if it were a deployment objective.

- The conclusion that "SCR operates more effectively in text space" is a plausible interpretation but is asserted mechanistically rather than demonstrated. It could equally be explained by end-to-end prompting avoiding brittle intermediate decomposition errors, which is a simpler explanation.

---

## Nice-to-Haves

- A scaling plot of model capability (e.g., MMLU score or benchmark-derived proxy) vs. SCR–RCR performance gap across more models would be highly impactful and provide genuine support for the capability-conditioned recommendation.

- An evaluation with partially correct or mixed-quality contexts would stress-test situated faithfulness under more realistic RAG conditions and substantially strengthen the practical relevance claim.

- A general capability evaluation (MMLU or similar) before and after CR-DPO would significantly increase confidence in the method's safety and deployability.

- Evaluating CR-DPO on one additional open-source model (e.g., Llama-3-70B or Mistral-7B) would establish whether the training approach generalizes across architectures and scales.

- A confusion matrix decomposing source-selection behavior per method (internal vs. external choice, conditioned on which source is correct) would directly validate or challenge the "confidence reasoning" interpretation of SCR's success.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"DIA is too weak a baseline to justify the +24.2% claim" (Harsh Critic):** DIA is an intentionally naive upper bound on standard RAG systems. The paper also compares against TACS and a full suite of RCR methods. Comparing against DIA to demonstrate the magnitude of the vulnerability is appropriate; the RCR methods serve as the more principled comparison. The asymmetry here is not unfair — it favors the baseline to make a stronger point about the problem severity.

- **"No confidence intervals or statistical significance" (Harsh Critic / Spark Finder):** Single-run evaluation is the standard norm for large-scale QA benchmarks with API models, and many of the reported differences are large (>5 percentage points). This does not meet the bar for a substantive weakness in this research community's standards.

- **"The upper bound of Acc_f is not valid without stronger assumptions" (Harsh Critic):** The paper's claim that "wrong contexts can't help a model answer questions it cannot answer" is a reasonable and approximately correct assumption given the experimental design, where false contexts are specifically constructed to point to wrong answers. While a false context could theoretically contain tangential helpful information, this is an edge case that does not undermine the benchmark design.

- **"The related work does not deeply engage with calibration/uncertainty literature" (Harsh Critic):** The related work is selective but adequate for the paper's scope. Requiring an extensive survey of calibration literature imposes scope creep on a paper whose contribution is empirical and applied, not a theoretical contribution to calibration.

- **"Concurrent work is not deeply compared" (Harsh Critic, multiple reviewers):** The paper cannot be penalized for not comparing against methods whose specifics are not established by an external reviewer; the paper positions against the concurrent work's benchmarks (which it uses in evaluation) and is explicit about what is and is not compared.

- **Requests for missing related works:** Per review instructions, not included, as these cannot be verified without external sources.

---

## Novel Insights

The most genuinely novel insight in this paper — largely surfaced by the spark finder but empirically supported in Section 5.3 / Table 3 — is the **structural misalignment between confidence calibration and accuracy maximization in rule-based systems**. Improving a model's calibration (via isotonic regression, threshold tuning, etc.) does not reliably improve situated faithfulness because a well-calibrated confidence score need not track the binary source-selection decision optimally. This is a theoretically clean insight: calibration optimizes expected calibration error, while the situated faithfulness objective optimizes accuracy — and these are different optimization targets. This finding has implications beyond this paper for any system that uses predicted confidence as a proxy for decision-making in retrieval-augmented settings.

---

## Suggestions

1. **Expand model coverage with at least one intermediate-capability open-source model** (e.g., Llama-3-70B or Mistral-7B) to substantiate the capability-conditioned SCR vs. RCR recommendation, and apply CR-DPO to at least one additional model beyond Llama-3-8B.

2. **Add a general capability evaluation** (e.g., MMLU, ARC, or similar) before and after CR-DPO training to demonstrate the method does not degrade parametric knowledge or broad reasoning — this is essential for any practical deployment claim.

3. **Add a source-selection confusion matrix** breaking down how often each method selects internal vs. external answers conditioned on (internal correct × context correct/wrong), to provide evidence for the "confidence reasoning" interpretation rather than a "default to internal" alternative explanation.

4. **Explicitly discuss the GPT-4o data generation circularity** in the main paper limitations section, and ideally provide a disaggregated analysis of GPT-4o performance on RedditQA vs. non-GPT-4o-generated datasets.

5. **Report RedditQA's key statistics** (final N, domain distribution, inter-annotator agreement, discard rate) in the main body rather than the appendix, given this is a benchmark contribution.

6. **Qualify the Figure 2 CR-DPO example**: either correct the factual dates in the reasoning or acknowledge that CR-DPO teaches adversarial argumentation rather than improved knowledge organization, as this distinction matters for interpreting what the model has learned.

7. **Include at least one experiment with mixed-quality contexts** (partially correct, partially misleading) to probe the robustness of situated faithfulness methods in more realistic RAG conditions.

---

**Evaluation axes:**
- **Novelty:** Moderate-to-good. RedditQA and the SCR/RCR taxonomy are meaningful contributions. CR-DPO is technically a task-specific DPO application, not an algorithmic advance, but is a practical contribution. The calibration-vs-accuracy insight is genuinely novel.
- **Technical soundness:** Moderate. The formulation is clean, but the experimental scope is narrow (3 models, 1 training model, binary context setup), and several core claims are under-supported by evidence.
- **Empirical support:** Adequate for the narrower claims; insufficient for the broader generalization claims (capability-SCR/RCR relationship, CR-DPO generality).
- **Significance:** Good. Knowledge conflicts in RAG are a real, growing problem. The datasets, methods, and analysis fill a genuine gap in the literature.
- **Clarity:** Good. The paper is well-organized, tables are informative, and the SCR/RCR taxonomy is easy to follow. Key implementation details are too heavily deferred to appendices.