# Old Memories Die Hard: Understanding Challenges of Privacy Unlearning in Large Language Models

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Large language models (LLMs) often memorize private information during training, raising serious privacy concerns. While machine unlearning has emerged as a promising solution, its true effectiveness against privacy attacks remains unclear.
To address this, we propose Prileak, a new evaluation framework that systematically assesses unlearning robustness through three-tier attack scenarios: direct retrieval, in-context learning recovery, and fine-tuning restoration; combined with quantitative analysis using forgetting scores, association metrics, and forgetting depth assessment.
Our study exposes significant weaknesses in current unlearning methods, revealing two key findings: 1) unlearning exhibits ripple effects across gradient-based associated data, and 2) most methods suffer from shallow forgetting, failing to remove private information distributed across multiple model layers.
Building on these findings, we propose two key strategies: association-aware core-set selection that leverages gradient similarity, and multi-layer deep intervention by progressive learning rates and representational constraints. These strategies represent a paradigm shift from shallow forgetting to deep forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the robustness of privacy unlearning methods for large language models (LLMs). It introduces PriLeak, a new evaluation framework that tests unlearning effectiveness against active attacks: direct retrieval, in-context recovery, and fine-tuning restoration. Using the Enron dataset and LLaMA-3.2-3B, the authors benchmark 19 existing unlearning methods and find that many approaches only achieve shallow forgetting, with sensitive information quickly recoverable. Their analysis points to two core issues: forgetting ripples across associated data and fails to penetrate deeper network layers. Based on these insights, they propose strategies including association-aware core-set selection and multi-layer intervention to strengthen privacy forgetting.

### Strengths
The motivating problem is important and timely, and the paper demonstrates clear originality by shifting the evaluation of unlearning into more realistic active attacker scenarios. The PriLeak benchmark is a meaningful contribution to the community, offering nuanced and multi-tiered measurements of privacy persistence that go beyond passive-output testing. The identification of ripple effects and shallow forgetting shows careful empirical analysis, revealing mechanisms that prior work did not clearly articulate. The proposed strategies are incremental but directionally interesting, suggesting a practical path toward deeper and more resilient forgetting. The scope of the evaluation (19 methods, both known/unknown private data, multiple datasets) supports the value of the empirical findings.

### Weaknesses
The technical novelty of the proposed strategies feels modest relative to the strong emphasis on negative benchmarking results. The empirical section is heavily overloaded with tables and metrics, making key insights harder to follow; clearer narrative summarization would help. The benchmark relies on a single primary dataset for privacy testing, limiting the generalizability of the findings in real-world PII contexts. Some terminology such as “deep forgetting” remains conceptual without rigorous formalization or theoretical insight. The study of known vs. unknown private data is compelling but deserves deeper exploration: why does forgetting propagate similarly, and what constraints does this impose on future algorithm design? Finally, while the experiments are extensive, clear ablation results are needed to isolate where improvements truly come from in the proposed approach.

### Questions
One concern is whether the strong performance gaps between different methods are sensitive to hyperparameter choices. Is PriLeak intended to be a fixed evaluation standard or a benchmark whose scores vary significantly depending on tuning choices? The ripple-effect analysis suggests that privacy removal conflicts with utility preservation, yet the proposed method still appears vulnerable under P3. How do you envision closing the remaining >30% recovery gap? It would be helpful to clarify whether the benchmark could incorporate adaptive adversaries rather than predefined fixed attacks. The current selection of PII types from Enron seems narrow; could the proposed metrics generalize to more complex private attributes such as implicit identifiers? Finally, the proposed representation-anchoring loss uses noise-perturbed base states. Could you justify this choice more rigorously or compare with alternative anchoring methods (e.g., teacher-student consistency with privacy-filtered representations)?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes PriLeak, a new evaluation framework that assesses unlearning robustness through three-tier attack scenarios: direct retrieval, in-context learning recovery, and fine-tuning restorationn; combined with quantitative analysis using forgetting scores, association metrics, and forgetting depth assessment. Empirical studies expose weaknesses in current unlearning methods --- ripple effects across gradient-based associated data and shallow forgetting. The paper then proposes  association-aware core-set selection based on gradient similarity and multi-layer deep intervention as two strategies to mitigate the issues.

### Strengths
**S1.** The new benchmark is well motivated with principled designs.

**S2.** Empirical studies cover extensive unlearning methods and present interesting insights.

**S3.** The proposed two strategies effectively improve the unlearning.

**S4.** The paper is well presented.

### Weaknesses
**W1.** The empirical analyses are constrained to relatively small LLMs (LLaMA-3.2-3B and GPT-2).

**W2.** The paper's presentation may be further improved by highlighting the findings previous benchmarks did not yield in empirical study analyses.

**W3.** While I like the idea of fine-tuning-based recovery, I think it will be more interesting to check if fine-tuning on related but non-private data restores unlearned private data, e.g., email addresses of public figures or organizations.

**W4.** Minor Presentation Issues.
- The full name of CKA should be provided at its first appearance.
- Notations $\mathbb{D}_{uk}$ and $\mathbb{D}_k$ should be explicitly defined at their first appearance (L207) despite analogous definitions at L107.
- There are two consecutive "with" at the end of L257.
- The presentation of Table 1 can be potentially improved by using different colors to group numbers in different ranges.

### Questions
N.A.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the problem of machine unlearning in large language models (LLMs), aiming to evaluate how effectively models can remove specific knowledge while maintaining general utility. The authors propose a unified benchmark that measures residual memorization through multiple levels of privacy leakage and a utility metric capturing general performance retention. They conduct large-scale experiments on the Enron and MUSE datasets, benchmarking various representative unlearning methods. The results reveal that existing methods still struggle to achieve thorough and reliable forgetting, highlighting the challenge of ensuring complete unlearning in LLMs.

### Strengths
1. The paper conducts extensive experiments, covering a wide range of unlearning methods.


2. The writing is clear and well organized, making the experimental setup and findings easy to follow.


3. The experimental design is generally complete, with systematic evaluation across multiple models, datasets, and metrics.

### Weaknesses
1. My main concern is the novelty of the paper. The core idea that unlearning certain knowledge propagates to semantically related facts via shared representations closely parallels prior studies on ripple effects in knowledge editing [1-5]. Similar analyses of entanglement and edit locality have been thoroughly explored, making the framing here appear incremental rather than conceptually new.


2. The proposed measurements resemble established notions such as locality, retention, and causal entailment used in both knowledge editing and unlearning literature. Prior works including RippleEdits [4], MUSE [9], WMDP [10], and Deep Unlearning [11] have already operationalized comparable metrics for quantifying cross-fact interference. The paper does not convincingly justify why its definitions capture fundamentally different or deeper dynamics [6–11].


3. The paper’s setup is closely related to knowledge editing frameworks. It is better to also include knowledge editing methods as well.

References:

 [1] Locating and Editing Factual Associations in GPT Models.

 [2] Mass-Editing Memory in a Transformer.

 [3] Editing Factual Knowledge in Language Models.

 [4] Evaluating the Ripple Effects of Knowledge Editing in Language Models.

 [5] Evaluating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering.

 [6] TOFU: Benchmarking Factual Unlearning in LLMs.

 [7] Selective Forgetting: Advancing Machine Unlearning Techniques and Evaluation in Language Models.

 [8] Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness.

 [9] MUSE: A Benchmark for Evaluating Unlearning in LLMs.

 [10] WMDP: Unlearning Harmful Knowledge in LLMs.

 [11] Evaluating Deep Unlearning in Large Language Models.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the limits of machine unlearning for large language models and introduces PriLeak, a benchmark for evaluating how well different unlearning methods resist privacy-related attacks. The framework tests models under three settings: direct extraction, recovery through in-context prompts, and recovery after fine-tuning. This goes beyond the usual unlearning benchmarks by allowing the attacker to access and modify model weights. The study compares 19 existing approaches and shows that many achieve only superficial removal of memorized information. 

The authors identify two main issues: (1) cross-sample “ripple effects,” where unlearning one example inadvertently alters or forgets related data through shared gradient directions, and (2) “shallow forgetting,” where parameter changes remain concentrated in higher network layers while earlier representations continue to encode sensitive content. To address these, the paper proposes two strategies: an association-aware core-set selection method that identifies the most influential samples to forget based on gradient similarity, and a multi-layer intervention approach that adjusts learning rates and representational constraints across network depth to promote more complete forgetting without excessive performance loss.

### Strengths
1. Focuses on adaptive / active privacy attack via fine-tuning, whereas prior papers are focused on the case where the user only has API access to the model (can attack via ICL or via QA).
2. Identifies additional insights into how/where the model has stored the data that is meant to be unlearned, which allows them to improve the unlearning procedure to ensure information is removed throughout all the layers. The insights are consistent across two model architectures which makes them more convincing.
3. Experiments are thorough and provide interesting information on the existing unlearning algorithms.

### Weaknesses
1. The fine-tuning–based attack scenario assumes that an adversary has access to modify model weights. This setting is not clearly justified and may not reflect realistic deployment conditions, where most users interact only through APIs. Clarifying when and why this is a threat model we would care about is important to make the arguments of th epaper.
2. The paper does not sufficiently connect its metrics and analysis to existing work on memorization and knowledge localization in language models. There is a rich literature on tracing and diagnosing memorized content (e.g., influence functions, causal mediation, or representation probing), and the relationship between those approaches and the proposed metrics is not discussed. This makes it difficult to assess the conceptual novelty of the diagnostic tools. Moreover, the proposed evaluation metrics and the “deep forgetting” interpretation rely on several design choices (layer selection, gradient normalization, metric thresholds) that are not analyzed for robustness. Without sensitivity studies, it is unclear how stable these results are across architectures or training configurations.
4. The link between representational change and actual privacy protection remains partly qualitative. The evidence that deeper layer modification corresponds to stronger unlearning is plausible but not rigorously demonstrated. Can the authors provide more direct evidence that these changes correspond to actual reductions in recoverable private information, rather than general representational drift?
5. Enron and MUSE News contain highly structured PII as I understand it. So would the results hold in cases where PII is more diffuse? And what about forgetting more general knowledge?
6. Recent literature has drawn into question the utility of the traditional unlearning definition (matching re-training from scratch). The authors do not comment on this at all, and this relates to my first question about why looking at FT attacks in this setting is even interesting in the first place.


Separately from this specific paper's methods, I think there are a lot of unlearning papers flooding the space without making meaningful improvements or insights. The lack of true, sociotechnical motivation for the new setting in this paper is just further evidence that the authors may not have thought through what is really important for unlearning from a social / legal standpoint. There is little to no discussion beyond the usual GDPR citation. I am also concerned, on the technical side, that the authors made little effort to connect to any literature that does not directly discuss unlearning (eg memorization literature).

### Questions
The weaknesses above contain the questions that I have about the paper.

### Soundness
3

### Presentation
2

### Contribution
2
