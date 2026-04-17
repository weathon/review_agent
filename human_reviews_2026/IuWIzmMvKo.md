# ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
LLMs require efficient knowledge editing (KE) to update factual information, yet existing methods exhibit significant performance decay in multi-hop factual recall. This failure is particularly acute when edits involve intermediate implicit subjects within reasoning chains. Through causal analysis, we reveal that this limitation stems from an oversight of how chained knowledge is dynamically represented and utilized at the neuron level. We discover that during multi-hop reasoning, implicit subjects function as query neurons, which sequentially activate corresponding value neurons across transformer layers to accumulate information toward the final answer—a dynamic prior KE work has overlooked. Guided by this insight, we propose ACE (Attribution-Controlled Knowledge Editing), a framework that leverages neuron-level attribution to identify and edit these critical query-value (Q-V) pathways. Ace provides a mechanistically grounded solution for multi-hop KE, empirically outperforming state-of-the-art methods by 9.44% on GPT-J and 37.46% on Qwen3-8B. Our analysis further reveals more fine-grained activation patterns in Qwen3 and demonstrates that the semantic interpretability of value neurons is orchestrated by query-driven accumulation. These findings establish a new pathway for advancing KE capabilities based on the principled understanding of internal reasoning mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores multi-hop knowledge editing. The authors hypothesizes that the failure is because the missing tracking of intermediate implicit subjects within reasoning chains. Using a suite of attribution metrics, they aim to uncover the mechanism of multi-hop reasoning. Based on the analysis of identifying important attention layers and FFN layers, Attribution-Controlled Knowledge Editing is proposed. Evaluation on a single dataset Mquake-3k is performed to demonstrate the effectiveness of the proposed method.

### Strengths
- This paper explores an interesting and important direction of multi-hop knowledge editing.

- This paper not only presents a solution, but also gives the analysis on which the approach is conditioned.

### Weaknesses
- The two takeaways are well-explored and not novel. For example, *Takeaway 2: Information of the final answer is accumulated through implicit query neurons that sequentially activate corresponding value neurons across the reasoning chain.* is verified in [1]. 

- The analysis in Section 4 lacks sufficient rigor. In particular, the selection of the subset from MQuAKE-3K is not clearly justified. Although the analysis is entirely based on this subset, no details about it are provided. For instance, the number of samples or their distribution across different knowledge types. Given that MQuAKE-3K is not a large dataset and the analysis only involves forward passes, it would be more convincing to conduct the analysis on the full MQuAKE-3K dataset rather than on an unspecified subset.

- The deduction from the analysis is not convincing： a27, a26, a7 ranks top in all knowledge-> MHSA stores general knowledge and capabilities in LLMs.

- Authors claim that FFN layers tends to primarily extracts its own knowledge. But f_26 ranks top for all knowledge types, which contradicts the conclusion.

- In the identifying process, forward passes on all multi-hop questions in the dataset and computed the sum of the importance scores at the last token position are performed. Is this setting reasonable? Is this instance-level or dataset-level editing?

- The whole evaluation is based on one benchmark MQuAKE-3K, which cannot demonstrate the generalization of the conclusion. It would be better to conduct the evaluation on more multi-hop knowledge editing datasets.

[1] CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Attribution-Controlled Knowledge Editing (ACE), a framework aiming to enhance multi-hop factual recall after knowledge editing. The authors claim prior locate-then-edit methods (ROME, MEMIT, PMET) degrade on multi-hop questions, esp. when edits involve implicit subjects. The framework builds on the empirical observation that implicit subjects act as query neurons that activate value neurons in subsequent layers. ACE identifies critical query layers and value layers and performs sequential edits (value-first then query) using a PMET-style closed-form update.
On a curated subset of MQuAKE-3K, ACE improves multi-hop accuracy vs. baselines by approx. 9% (GPT-J-6B) and 37% (Qwen3-8B) over PMET. Ablations that skip editing top-ranked query or value layers reduce the performance.

### Strengths
1. Multi-hop knowledge editing is definitely under-researched and practically important
2. The Q-V pathway view provides a clear hypothesis (drawn from empirical observations) to test and design around
3. Simple and clear pipeline: (1) Identify (via attribution),  (2) edit value layers (deep),  (3) edit query layers (mid); built atop PMET
4. Significant improvement over baselines
5. The authors analyze and highlight the distinct roles of query and value edits. Skipping value layers hurts most, which aligns with the intuition that query layers rather transmit, whereas value layers rather store information. These are interesting findings.

### Weaknesses
1. I am not convinced that the additivity of log-prob importance (Eq. 9) is well-suited. It is simply asserted by the authors, not derived or tested. The same holds for Eq. 10. Both equations ignore underlying nonlinearities.
2. The authors use a subset of MQuAKE-3K in the experiments. It is not clear what they did to prevent selection bias. LLM-based labeling lacks human validation. Also, it is not clear how prompts were handled across baselines. What did you do to guarantee a basic level of uniformity?
3. From a statistical analysis viewpoint, the evaluation is quite weak. There are no reports of significance tests, CIs, or multi-seed variability. There are no "negative" control analyses (e.g., wrong layers, randomized layer ranks, randomly swap query/value roles, etc).
4. The most critical point is the evaluation on a single dataset. To show generalizability, the authors should extend the experiments and evaluate on additional datasets.
5. It would be very helpful to include a human audit sample (e.g., 300 instances) with inter-annotator agreement of GPT-4o labels.

### Questions
See comments above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce ACE, a neuron-level editing framework. They first attribute multi-hop recall to coordinated query (Q) and value (V) neurons across transformer FFNs; then, they apply that attribution to perform staged edits. Specifically, ACE achieves this by first overwriting value layers (closed-form, like PMET) to encode the new fact; then, it adjusts query layers which control access to said edited value such that the updated knowledge propagates through multi-hop chains. The authors test ACE on a filtered multi-hop subset of MQuAKE-3K and report sizable accuracy gains over FT/ROME/MEMIT/PMET on GPT-J and Qwen3-8B. Their ablation experiments also indicate that both Q- and V-stage edits are necessary, as evidenced by attribution metrics (log-probability change and inner-product-based query scores) and intervention results. The experiments and method design support the validity of the authors' interpretation that the query-to-value activation mechanism mediates multi-hop recall.

### Strengths
1. The mechanistic motivation behind the two-stage editor is excellent. The method aligns with observed Q-to-V activation timing, rather than relying on heuristics for layer selection.
2. The formalization of neuron-level importance is clear and convincing, with consistent localization patterns across semantic categories. The ablations demonstrate strong causal sensitivity.
3. The multi-hop gains across the two tested models are substantial.

### Weaknesses
1. While the authors frame their work as multi-hop reasoning editing, the evidence predominantly demonstrates improved propagation of edited facts rather than modification of the intermediate reasoning process itself.
    * The method edits FFN value weights and query gating to make the correct value more retrievable along existing chains. However, the evaluations measure only final-answer accuracy, without verifying whether intermediate computations actually changed.
    * Trajectory-level tests may be needed to fully accept the implied mechanistic claim. Specifically, one or both of the following would provide definitive evidence: (i) activation patching for each hop pre/post edit; and (ii) tracing attention and activations to confirm that the edited value propagates through the intended intermediate subject rather than being directly re-inserted at the end. The current results are valuable in that they clearly demonstrate better retrieval of a rewritten fact, but they don't establish that the reasoning process itself has been modified.
2. Attribution-based layer rankings may be tightly coupled to the specific prompt and decoding context. To my knowledge, the paper does not explicitly demonstrate stability across different queries, so it is possible that the importance score (log-probability gain on the final token when a value neuron is restored) and query score (inner product with learned subkeys) can both shift with template, subject position, and relation phrasing.
    * Because ACE edits only the top-ranked layers, it implicitly assumes the rankings will remain the main routing and storage sites across all different wordings of the same relation. The paper does not seem to verify this via rank-stability tests.
    * The authors should check whether critical layers stay on top under paraphrases, alternate templates, or different relation types. That way, it will be more clear whether ACE can generalize beyond the identification prompt and avoid overfitting to it.
3. Relatedly, the mechanistic claims lack per-instance causal verifications.
    * The authors show that ablating a small neuron set greatly reduces accuracy, but they do not isolate whether those neurons are causal for specific chains or globally influential.
    * Per-instance causal scrubbing along the chain may be needed to fully accept the causal claims.
4. The current evaluation setting may bias the reported gains toward cases in which the model already does well.
    * The authors evaluate on a curated subset where the base model already answers correctly (selected via zero-shot answerability).
    * In this setting, ACE performs counterfactual edits and then measures whether that new fact propagates along the model's existing multi-hop reasoning path (including the paraphrase and specificity checks).
    * This does not necessarily probe whether ACE works for failure modes wherein the path itself is unreliable. In other words, it remains unclear whether ACE can repair a broken intermediate link (e.g., when the base model originally follows the wrong entity mid-chain).

### Questions
1. Could you please provide hop-wise activation patching and intervention results that show that the edited value is carried through the intended implicit subjects rather than injected late?
2. How stable are Q/V layer/neuron rankings and attribution results across prompt templates, paraphrases, and relation types?
3. Could you please evaluate cases where an intermediate relation is wrong and must itself be edited, and include locality/specificity metrics on unrelated facts?

I am willing to raise my score if these points are addressed via new results, or if the authors can justify why they are not needed to support the main claims.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the failure of existing Knowledge Editing (KE) methods in multi-hop factual recall scenarios, particularly when edits involve intermediate, implicit subjects. The authors provide a new mechanistic insight, suggesting that multi-hop reasoning relies on a "query-value" (Q-V) activation pathway, where implicit subjects act as "query neurons" that sequentially activate "value neurons" across layers. Based on this, they propose ACE, an editing framework that uses attribution to identify and modify both the critical query (reasoning path) and value (factual storage) neurons. The method shows significant performance improvements on multi-hop recall tasks over existing baselines on GPT-J and Qwen3-8B.

### Strengths
1. **Valuable Mechanistic Insight**: The paper's core strength is its deep dive into why multi-hop editing fails. The proposed Q-V activation mechanism  offers a plausible and fine-grained explanation for how information is accumulated during complex reasoning, moving beyond layer-level heuristics.
2. **Novel Editing Framework**: The ACE framework is a direct and logical application of this insight. By targeting not only the "value" FFNs (for the fact itself) but also the "query" FFNs (for the reasoning path), the method presents a more complete solution that is mechanistically grounded.
3. **Strong Empirical Gains**: The reported results are impressive, with ACE substantially outperforming SOTA methods like PMET by 9.44% on GPT-J and 37.46% on Qwen3-8B in multi-hop accuracy. The ablation studies also clearly validate the necessity of editing both Q and V components .

### Weaknesses
1. **Potential Data Leakage in Evaluation** : I am concerned about the experimental setup regarding the MQuAKE dataset. The paper seems to use MQuAKE's multi-hop question templates (as seen in Appendix H.1 ) to construct the few-shot and CoT prompts for the evaluation.
If the model is evaluated on MQuAKE questions after being primed with prompts structured exactly like MQuAKE's own examples, does this not constitute a form of "teaching to the test"?
2. **Practicality and Efficiency of Attribution**: The ACE framework introduces a "Stage 1: Identifying" step  that relies on attribution to locate critical Q-V neurons before editing. This "attribution-locate-then-edit" paradigm appears to add significant computational overhead. Is this attribution step a one-time analysis required per model, or is it a dynamic analysis required per edit? The paper suggests Qwen3-8B has domain-specific, dynamic alignment, implying the latter. If this attribution must be run for every new edit, the method's practical utility is questionable compared to "on-the-fly" editors like ROME or PMET. The authors must provide a clear analysis of the computational cost (e.g., latency, FLOPS) per edit versus the baselines.
3. **Missing SOTA Comparisons and Efficiency Analysis**:
The set of baselines , while foundational, seems incomplete. The field has advanced quickly. Why are more recent and highly relevant SOTA methods, such as AlphaEdit or other concurrent works focusing on multi-hop editing, omitted from the main comparison in Table 2? Furthermore, the paper lacks a rigorous analysis of batch editing efficiency.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
