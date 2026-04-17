# CLARA: Clarification-Driven Measurement of Input Ambiguity in LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Large Language Models (LLMs) perform well on question-answering tasks with well-specified inputs, but real-world queries are often vague or underspecified, leading to ambiguity and unreliable responses. Existing methods for ambiguity detection typically use a two-stage framework: (a) generating multiple clarifying reformulations of the input, and (b) answering each version to assess ambiguity based on the variation in responses. We introduce CLARA, a novel and complementary approach that quantifies ambiguity using only the clarification generation phase. We hypothesize that ambiguous inputs elicit a greater number and diversity of clarifications. CLARA estimates ambiguity by measuring the semantic dispersion of these LLM-generated clarifications, without requiring subsequent answering. This method requires no additional task-specific training, relying instead on an off-the-shelf similarity model, and thus offers two key benefits: (1) it is lightweight—reducing API calls and computational cost, and (2) it is more robust across LLMs—avoiding dependence on model-specific factual knowledge and reducing susceptibility to hallucinations. Empirical results across multiple LLMs and benchmark datasets demonstrate that CLARA provides an intuitive, scalable, and effective alternative to answer-based techniques, achieving comparable or superior performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a simple method for measuring the ambiguity of a question given to an LLM, based on the consistency of the clarification utterances themselves, rather than that of following answers, as normally done in the literature.
The approach is relatively simple: given an ambiguous query, a user requests the model to generate different clarifications, and the diversity of the generated clarifications is directly utilized as a proxy for measuring the extent to which the question is ambiguous.
For gauging the diversity of generated clarifications, a BERT-based similarity model, fine-tuned on the Quora Question Pairs (QQP) dataset, is employed, and the similarities of each pair of clarifications are aggregated to score a query-level amgbiuity.

### Strengths
- It is an insightful idea to suggest that answer generation may not be necessary for consistency-based ambiguity measurement, and that relying solely on clarification can be sufficient.
- The proposed method is straightforward and easy to follow.
- The overall writing quality is clear and well-organized.

### Weaknesses
- The experiments are conducted on only two datasets, consisting of 200 and 364 instances, respectively. Incorporating additional and more diverse datasets would significantly strengthen the empirical claims of the paper.
- Furthermore, the performance on these limited datasets is not satisfactory, particularly on AmbigInst. Although the authors attribute this to potential overfitting of BERT-based score functions to question-type cases, this explanation itself exposes a key weakness of the proposed method. Specifically, the score functions appear to lack robustness and require fine-tuning for each domain, which undermines their generalizability and practical applicability.
- While the simplicity of the proposed method can be considered a merit, its subpar performance suggests that further variations should be explored. For example, combining the proposed approach with existing answer consistency–based methods could serve as a straightforward yet meaningful direction for improvement. In addition, extending the range of score function choices would broaden the scope and contribution of the paper. Beyond encoder-based scoring, employing LLM-based scoring or other alternatives may yield better performance. Overall, the current version of the paper lacks sufficient exploration and technical depth to be considered ready for acceptance.
- It is also unclear whether the proposed method is cost-efficient. The authors are encouraged to include additional analyses addressing this aspect. To the best of my understanding, the proposed approach appears to require a larger number of clarification steps (grouped clarifications) rather than directly generating answers. If such an approach introduces additional computational or annotation costs, it raises the question of whether generating answers directly would be a more practical and efficient alternative. A discussion of this potential trade-off would help clarify the real advantages of the proposed method.

### Questions
Please refer to the Weaknesses section.

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
This paper proposes a simple, effective, and cost-efficient method for LLMs to decide when to ask clarifying questions given an input user query. The query ambiguity is measured by the diversity and semantic dispersion of LLM-generated clarification questions for that query. The experiments also show the effectiveness and cost-efficiency of the proposed method.

### Strengths
1. I'm interested in the propose method, especially given its focus on simplicity and cost-efficiency.
2. The method is model-agnostic and training-free, depending only on an off-the-shelf similarity model.

### Weaknesses
The paper's reliance on a score threshold is a major practical limitation.The paper reports results based on an optimal threshold, likely found by sweeping across all possible values on the test set to maximize the F1 score. It is unclear how to set this value in a real-world application. A static threshold is unlikely to generalize across different scenarios, user queries, and backbone LLMs. This lack of a robust threshold-setting mechanism significantly hinders the method's practical utility and generalizability.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CLARA, a novel method for detecting ambiguity in user queries. Unlike traditional two-stage approaches that require both clarification generation and answer evaluation, CLARA estimates ambiguity solely from the diversity and semantic dispersion of generated clarifications. By eliminating the need for answer generation, the authors claim, CLARA reduces computational cost and avoids reliance on model-specific knowledge. It uses an off-the-shelf similarity model and requires no task-specific training. Empirical evaluations across multiple LLMs and benchmark datasets show that CLARA performs comparably or better than existing answer-based methods.

There are several concerning issues that need to be addressed before this manuscript is ready for publication. Please refer to the list of weaknesses for more details.

### Strengths
- The idea that the level of ambiguity in input may be proportional to the number and diversity of LLM-generated clarifications is intuitive.

### Weaknesses
1. Focusing solely on ambiguity detection limits the contribution given there have been several papers in the past few years tackling ambiguity detection as well as answer generation, such as soliciting clarifications or generating a long-form answer addressing diverse interpretations of a given ambiguous input. In addition, the latest related works cited by the authors as examples of only detecting ambiguity actually go beyond the task. Shi et al. (2025) does uncertainty calibration of the LLM-generated response, and Hou et al. (2024) performs uncertainty decomposition, which involves quantifying uncertainty caused by the model.
2. The survey of related works falls short of expectation. There are several missing papers even if you limit the scope to ambiguity detection in question answering, [1,2] to name a few. Those that are cited in this paper are not described adequately, making it difficult to situate this work in the existing literature. In fact, Tian et al. (2023) is misrepresented as detecting ambiguity in questions, whereas it is a paper on calibrating uncertainty in LLM-generated responses. 
3. Several important claims are not adequately substantiated:
- Clarification generation still relies on the parametric knowledge of the LLM used, thus it is inaccurate to say that this approach avoids such reliance.
- The claim that this approach is more cost effective due to not generating potential answers in unconvincing, because existing methods can use those potential answers for answer generation or uncertainty calibration, which this work does not address. In this way, the comparison lacks fairness.
- This approach is claimed to be more scalable, but only compared to one of the baselines, AU, with respect to computational complexity.


[1] Lee et al. 2023. Asking Clarification Questions to Handle Ambiguity in Open-Domain QA
[2] Zhang et al. 2024. CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous
Information Needs in Large Language Models

### Questions
- Why  isn’t this work compared to the best results reported in Hou et al. (2024)?
- Why weren’t the benchmarks like ABG-COQA used in Shi et al. (2025) used in this work? Testing on 564 data points can limit the generalizability of the results.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CLARA, Clarification-Driven Measurement of Input Ambiguity in LLMs, which is a relatively lightweight framework that detects if an input query is vague or underspecified by quantifying the semantic dispersion of LLM-generated clarifications, thereby eliminating the need for resource-intensive answer generation. The proposed approach is efficient, requiring substantially fewer API calls than traditional answer-based methods, yet it achieves comparable or superior performance in reliably distinguishing ambiguous from unambiguous queries.

### Strengths
- The paper is generally well-written regarding the flow of the proposed method.
- The paper proposes a method that does not require answer generation for ambiguity detection.
- Empirical analysis of the paper demonstrates CLARA’s computational efficiency over answer-based methods.

### Weaknesses
- I have some concerns regarding the baseline Ask4conf. Rather than directly asking the confidence to LLMs, as the authors did, can we also check the similarity (or clarification) using the generated questions by self?
- The similarity model is fine-tuned on the QQP dataset and as the authors note, it seems like a domain mismatch. Do we really need QQP fine-tuning for this task?
- The paper mainly uses a BERT-based similarity model, which seems somewhat outdated. Are there better ways to measure the similarity between the questions for the proposed method? In other words, while effective, relying on a fixed BERT architecture may be viewed as less optimal compared to leveraging more recent embedding models.
- While the formalization is present, the paper focuses on the utility and results rather than the explicit definition of the metrics. I cannot understand the exact setup of the experiment without checking other papers.

### Questions
I need more explanations about the exact evaluation setup for the paper, as I noted in the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
