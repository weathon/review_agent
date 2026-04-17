# Do LLMs Perform Multilingual Multi-step Reasoning?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Ideally, large language models (LLMs) should be able to exploit information sources from all available languages to achieve strong performance for diverse tasks, including reasoning. However, most evaluations of multilingual reasoning focus on symbolic domains, e.g., mathematics and coding, and it remains unclear how effective LLMs handle multilingual reasoning in linguistic tasks. In this paper, we introduce a controlled multilingual two-hop question answering setting, where answering a question requires two reasoning steps across two documents in different languages: the first-hop document provides bridging information, and the second-hop document links it to the final answer. Despite the equal importance of both hops, we find that the performance of a strong multilingual LLM (i.e., Gemma-3) is substantially affected by language variation in the second-hop documents more than in the first-hop. To analyze each hop's reasoning process, we evaluate the decomposed sub-questions of a two-hop question. Surprisingly, the model often fails on the first sub-question for inferring bridging entities, yet still answers the overall two-hop question correctly. Our implicit context attribution analysis shows that the model still attends to bridging documents for correct answer generation, despite struggling to interpret them. This shows that the LLM's multilingual multi-hop reasoning does not follow a faithful step-by-step decomposition for sub-question answering. We also find that the absence of reasoning decomposition leads to about 18% composition failures, where both sub-questions are answered correctly, while failing to answer the two-hop question. To mitigate this, we propose a three-stage SubQ prompting method to guide the multi-step reasoning with sub-questions, which boosts accuracy from 10.1% to 66.5%. Overall, our findings shed light on the multilingual multi-step reasoning mechanism and the potential of explicit reasoning decomposition for future tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether LLMs are capable of performing multi-hop reasoning over documents written in mixed languages. The authors designed a carefully controlled two-hop QA experiment using a subset of questions and translated documents from HotpotQA, covering four languages (excluding English). In this setup, each question requires information from two documents in different languages. A question can always be decomposed into two subquestions: the first document provides the answer needed to identify a bridging entity, which is then used to retrieve the final answer from the second document. Several analyses were performed using Gemma-3-Instruct.

The authors make several key contributions. First, they propose a controlled experimental setting for multi-hop QA across mixed-language documents. Second, they show that Gemma-3-Instruct does not perform faithful multilingual multi-hop reasoning. Gemma-3-Instruct can sometimes arrive at the correct final answer using incorrect intermediate reasoning, or conversely, answer all subquestions correctly but fail to produce the correct final answer. Finally, they demonstrate that a three-stage prompting strategy that explicitly guides the model through subquestion-based multi-step reasoning improves overall performance.

### Strengths
S1: The study introduces a controlled and interpretable setting for evaluating multilingual multi-hop reasoning over documents of mixed languages, which the direction is valuable. 

S2: The paper discovers that models can produce correct answers with incorrect reasoning highlights an important limitation in current evaluations.

S3: The proposed three-stage prompting method provides a simple yet effective method.

### Weaknesses
W1: From existing work on LLMs reasoning over general benchmarks (e.g., GSM8K etc.), it is known that LLMs can arrive at the correct final solution despite having incorrect intermediate steps or flawed reasoning. Hence, the finding here is less surprising.

W2: While the paper provides comprehensive results for different combinations of multilingual experiments and a quick analysis on the correlation between performance and linguistic similarity , a deeper analysis of the multilinguality of LLMs, for example, into embedding homogeneity/heterogeneity across languages. This would help strengthen the findings and uncover why current LLMs are limited in multi-hop reasoning over documents in mixed languages (i.e., whether the limitation arises from language representations or reasoning capability).

W3: The experiments focus only on Gemma-3-Instruct. By including more LLMs would strengthen the generality of the findings. Similarly, the filtered Hotpot QA dataset used for this study contains 182 instances in total, this may be too limited to draw generalized conclusions.

### Questions
Q1: Figure 7 in the appendix, why in the instruction of the zero-shot prompt it says only 1 article is provided? Further, why is a question asked twice in this prompt?

### Soundness
2

### Presentation
3

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
Introduce a controlled multilingual two-hop question answering setting. Results reveal that performance is more sensitive to language variation in second-hop documents, and that models often answer two-hop questions correctly without faithfully solving sub-questions. Proposing SUBQ prompting, a three-stage sub-question–guided prompting method.

### Strengths
1.	The proposed multilingual two-hop QA setting offers a controlled and interpretable testbed.

### Weaknesses
1.	The filtered dataset (182 examples) may limit statistical robustness.
2.	The large performance jump with SUBQ prompting may partly reflect prompt engineering rather than improved reasoning ability, which should be discussed more explicitly. Moreover, the proposed three-stage prompting substantially increases computational cost and inference time.
3.	Experiments rely solely on Gemma-3-27B-Instruct, without testing other multilingual models (e.g., Qwen, Mistral, LLaMA). Broader model comparison would strengthen generality.
4.	The proposed “controlled multilingual two-hop question answering setting” is highly structured and artificial, which may not fully reflect real-world multilingual reasoning scenarios where information is distributed in more complex and less clearly decomposed forms. There is a gap between the controlled setup and realistic model usage.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduced a controlled multilingual two-hop QA setup. Hop-1 provides a bridging entity and Hop-2 contains the answer, possibly in different languages. It used HotpotQA to extend original entries to multi-language EN/FR/RU/AR/ZH, and the authors mainly used Gemma-3-27B-Instruct for evaluation. Key findings: (1) changing the Hop-2 (answer-span) language hurts performance far more than changing Hop-1; (2) the model often fails SubQ1 (bridge identification) yet still gets the overall two-hop answer, indicating unfaithful step-by-step reasoning. Attribution and distractor tests show strong reliance on Hop-1 even in unfaithful cases, and topic-relevant distractors degrade performance notably. A three-stage SUBQ prompting (answer SubQ1 -> use it for SubQ2 -> compose final answer) mitigates compositional failures, boosting accuracy from 10.1% to 66.5%.

### Strengths
(1) Controlled and decomposable setup with explicit sub-questions enables fine-grained diagnosis of faithfulness and composition. 

(2) The analysis is relatively in-depth, distinguishing two failure types: unfaithful failure and compositional failure, and further probing unfaithful failure with ablation-style investigations: attribution, distractor, and order effects.

### Weaknesses
(1) The experiments rely solely on Gemma-3-27B-it (with greedy decoding), which makes it hard to argue that the findings generalize across other LLM families; the evidence feels somewhat thin, may need to verify the cross-model and cross-decoding conclusions. 

(2) The evaluation uses fewer than 200 test examples, which may be too small to support strong conclusions.

(3) may include trying multi-hop (more than 2) questions.

### Questions
(1) A fundamental question: for multilingual multi-hop reasoning, why not translate all source documents into a single language? A model’s capabilities are typically best “unleashed” in one language. What is the distinctive benefit of keeping the task multilingual instead of normalizing to a single language?

(2) Are there 2-hop reasoning datasets that are more challenging than HotPotQA? If so, it might be valuable to evaluate on those as well.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a controlled multilingual two-hop QA dataset derived from HotpotQA and utilizes it to analyze the multilingual, multi-step reasoning capabilities of large language models. The resource is carefully curated and methodologically sound, and the accompanying experiments offer insightful empirical analyses on reasoning decomposition across languages.

### Strengths
1. The paper is clearly written and well organized, making the experimental setup and findings easy to follow. The presentation quality is strong, with clear figures and well-structured explanations of each prompting strategy.
2. The proposed resource—a controlled multilingual two-hop QA dataset derived from HotpotQA—is useful and thoughtfully designed. It also enables interesting findings and analysis and would be a great resource for other researchers.

### Weaknesses
1. Dataset scope and justification. 

The dataset comprises only 182 examples per language, and the authors describe it as “a controlled multilingual two-hop QA setting.” While such a compact testbed is appropriate for fine-grained analysis, the paper would benefit from a clearer justification of its small size along two dimensions:

- Representativeness. How do these 182 items capture the diversity of linguistic phenomena, topic coverage, and reasoning types observed in the full HotpotQA distribution? Some indication (e.g., category statistics, lexical diversity, or reasoning-type balance) would help substantiate the claim that results from this subset are indicative of broader multilingual behavior.
- Technical feasibility. If scaling to a larger set is infeasible, the paper should specify what practical or computational constraints prevent expansion (e.g., human validation cost, attribution analysis overhead). Otherwise, readers may interpret the small size as arbitrary.
Finally, the authors should elaborate on the analytical limitations imposed by the small sample size. A brief discussion of sampling variance, statistical confidence, or generalizability would strengthen methodological transparency. Small test sets can indeed be efficient if their indicative power is demonstrated, but this requires explicit reasoning beyond the current single sentence in Section 3.2.

2. Uneven supervision and limited real-world applicability. 

The SubQ Prompt setup benefits from gold sub-question decompositions (and sometimes gold intermediate answers), while Two-Hop and Zero-shot CoT must infer both steps autonomously. This uneven access blurs causal interpretation: the performance gain (10.1 → 66.5 F1) partly reflects privileged supervision rather than the decomposition method itself. Moreover, such oracle access is unrealistic in deployment. The current configuration should therefore be interpreted as an upper bound rather than a deployable setting. Clarifying this limitation would improve transparency. Figure 5 compares three prompting strategies: Default Prompt, Zero-shot CoT, and SubQ Prompt. To improve interpretability, I suggest renaming the current SubQ Prompt configuration as SubQ Prompt (Gold), since it relies on gold sub-question decompositions. This will make the supervision asymmetry explicit.

Building on this, two additional variants could further enrich the analysis:

- SubQ Prompt (Self-Decompose): This variant removes the gold decomposition information privilege. The model generates its own decomposition before solving, removing the gold decomposition and testing autonomous reasoning.
- SubQ Multistep (Gold): This presents an alternative to SubQ prompting. Since the gold decomposition steps are already available, they can be used to invoke two single-hop reasoning calls directly.

Together with the original Default and Zero-shot CoT settings, these additions would yield a clearer picture of how performance scales with decomposition quality and supervision level, and help separate the benefits of explicit structure from the effects of oracle information.

### Questions
See weaknesses.

I also have a suggestion on writing improvements. The presentation would greatly benefit from an upfront declaration of the central thesis. Currently, the reader only realizes this in the conclusion that the study aims to investigate whether large language models can perform faithful multilingual multi-step reasoning. Making this intent explicit early on would help readers understand that the new dataset functions as an instrument for analysis rather than an end in itself, and that the empirical results are meant to illuminate the fidelity of reasoning rather than to achieve performance gains.
In its current form, the “In this paper” paragraph (Line 074) lists what the paper provides, but not to what end these offerings serve. Clearly articulating the inquiry’s purpose at that point would greatly improve narrative coherence.

### Soundness
3

### Presentation
3

### Contribution
3
