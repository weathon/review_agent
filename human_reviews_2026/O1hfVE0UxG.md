# PrefDisco: Benchmarking Proactive Personalized Reasoning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Current large language model (LLM) development treats task-solving and preference-alignment as separate challenges, optimizing first for objective correctness, then for alignment to aggregated human preferences. This paradigm fails in human-facing applications where solving a problem correctly is insufficient if the response mismatches the user’s needs. This challenge intensifies in just-in-time scenarios where no prior user interaction history exists due to cold-start conditions or privacy constraints. LLMs need to proactively identify what they don’t know about the user, strategically elicit preference values through questioning, then adapt their reasoning processes and responses accordingly—a complicated chain of cognitive processes which we term personalized reasoning. We introduce PREFDISCO, an evaluation methodology that transforms static benchmarks into interactive personalization tasks using psychologically-grounded personas with sparse, context-dependent preferences, and define PREFALIGN as a fine-grained rubric-based metric for measuring preference alignment. PREFDISCO builds scenarios where identical questions require different reasoning chains depending on user context, as optimal explanation approaches vary by individual expertise and preferences while maintaining factual accuracy. Evaluation of 21 frontier models across 10 tasks reveals 29.0% of naive personalization attempts produce worse preference alignment than generic responses, yet generic responses also fail to serve individual user needs. These findings suggest personalized reasoning requires dedicated development rather than emerging naturally. PREFDISCO provides a foundation for developing systems that can adapt to individual users in education, healthcare, and technical domains where personalization is critical.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces personalized reasoning, the ability of large language models (LLMs) to adapt their reasoning and explanations to individual user preferences rather than offering one-size-fits-all answers. It presents PREFDISCO, a benchmark that converts static reasoning tasks into interactive personalization challenges where models must elicit user preferences through questioning and tailor responses accordingly. Evaluating 21 frontier models across 10 domains, the authors find that 29% of personalization attempts reduce alignment compared to generic responses and often lower task accuracy, especially in mathematical reasoning. The study concludes that effective personalization requires dedicated mechanisms for preference discovery and adaptive reasoning, as current LLMs fail to perform just-in-time personalization despite strong general capabilities.

### Strengths
They have assessed the 21 frontier models across multiple benchmarks.

### Weaknesses
I believe addressing the following concern might help to improve the quality:

1. Although the task is interesting, the exploration of the user preference has already been conducted in UserBench [1]. The paper lacks a comprehensive comparison with existing works. No table or section is provided for such comparisons. I encourage the authors to conduct a thorough literature review and include a detailed comparative analysis with current studies.


2. Another concern relates to the five interaction turns used in the experiments. The rationale behind selecting this specific number is not explained. An ablation study exploring the impact of different conversation counts would be helpful to justify this choice.


3. I believe that preference reasoning presents a valuable direction to explore for planning-related tasks, such as travel planning. I suggest that the authors include this domain in the benchmark to enhance the coverage and applicability of the study.

4. The process used to generate user instructions is not clearly described. Only one incomplete example is presented in Figure 1. It is recommended that the authors release a portion of the benchmark and provide a detailed visualization of the data generation process.


5. The authors mention that the user simulator is passive. However, in practice, an assistant faced with a passive user would likely disengage and terminate the conversation. The reasoning for considering a passive user in the experiments remains unclear and requires further justification.


6. The authors state that MATH and LogiQA exhibit the most performance degradation, whereas SocialIQA benefits the most from interactive personalization. However, no analysis or explanation is provided to clarify these observations. A deeper analysis would help readers better understand the underlying performance dynamics.


7. The results in Table 1 show that, for most benchmarks, the performance of 3-Opus surpasses that of Sonnet-4. However, Sonnet-4 is a stronger model. The authors should provide an explanation for this phenomenon.


8. The judgment process used in the evaluation is insufficiently explained. Including the evaluation prompts and methodology details would enhance transparency and reproducibility.


---
**Reference**

[1] UserBench: https://arxiv.org/pdf/2507.22034

### Questions
Please read the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper defines personalized reasoning in just-in-time scenarios, where the model actively seeks answers to discover user preferences and adjusts its reasoning accordingly. It then proposes PREFDISCO, transforming a static benchmark into an interactive task using psychology-driven sparse preference profiles. Experimental results demonstrate the limitations of current LLMs in personalized reasoning.

### Strengths
1. Extending from personalized content to personalized reasoning is interesting and reasonable.

2. A psychology-driven evaluation method is proposed.

3. Experimental results highlight the limitations of current LLMs, emphasizing the need to consider this characteristic.

### Weaknesses
1. Many details in the proposed evaluation framework remain unclear. For example, how are sparse preference profiles generated? What are the 20 scenarios? What are the relevant and irrelevant attributes for each scenario? How are importance weights calculated? How is LLM-based assessment performed? The numerous vague descriptions make the paper incomplete.

2. While passive user simulation can simulate more challenging scenarios, it may overlook more informative interactions in general. Furthermore, real-world scenarios are not always extreme and may not accurately reflect actual situations. Therefore, this rationale requires further discussion.

3. The paper's analysis is not sufficiently thorough. For example, the paper attributes the significant drop in accuracy of the mathematical task to "over-optimization," but can this be further demonstrated experimentally? Why does the accuracy of the mathematical task continue to decline with increasing question count?

4. Case studies would be beneficial. For example, showcasing various failure cases to support the paper's arguments.

5. Minor: The range of the value of $g_j(r,v_j)$ is inconsistent between Sections 2.3 and 3.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors created a benchmark for simulating online exploration of user preferences with a simulator. The authors creates a set of preference attributes and allow the llm to query the value of those attributes' numerical values. They defined the eval metric as the ratio of the improvement performance upon zero shot prompting comparing to oracle where all the preference values are given in context. They argue that personalization is a reasoning capability that current model lacks.

### Strengths
1. A novel interactive benchmark for personalization of llm that did not use llm simulate human, which decrease confounding factors.

2. The evaluation spans all frontier models and show that all models failed under the current setting.

### Weaknesses
1. While I understand (and even appreciate) the authors’ motivation to control for confounding by simplifying the user simulation with scalar value feedback, the framework is far from realistic human llm interactions. Several components are significantly out-of-distribution: (a) models are explicitly forced to ask preference-related questions via system prompting, and (b) feedback is reduced to scalar or categorical values, whereas real users express preferences through text. These design choices make the evaluation more of a controlled experiment rather than a realistic personalization eval benchmark, though I acknowledge the inherent difficulty of modeling authentic human feedback at scale.
2. Personalization through online interaction and exploration is not a novel topic. Several papers have explored this under llm setting, not even to mention previous bandit literatures. The method authors explored in the previous setting seems to be a soley based on the system prompt, rather than sth like uncertainty quantification. I would wonder if more advanced method can boost the performance.
3. The exact system prompt for exploration is not shown, which I assume would be a very important part of the whole algorithm.

### Questions
1. What are the system prompts exactly?

2. Can some more advanced method like UQ based solve this problem better? 

3. see weakness 1.

### Soundness
2

### Presentation
2

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
This paper proposes PrefDisco, an evaluation methodology that transforms existing static benchmarks into interactive personalization tasks layered on top of the original task. The authors introduce a multi-dimensional numerical profile to instantiate user preferences. Their experiments show that attempts at personalization can hurt performance accuracy over the original task, suggesting naively attempting proactive personalization often degrades alignment for generic responses. Furthermore, they show that models generally do not ask questions, highlighting that interactive personalization remains under-explored.

### Strengths
1. Explores a very interesting and relevant problem of multi-turn interactive preference elicitation. Current LLMs tend not to actively discover user preferences, even though this is critical for many real-world problems.
2. The framework is creative and very flexible, allowing researchers to turn any existing benchmark into an interactive personalization task, making it broadly applicable across domains.
3. The evaluation is extensive with various metrics and abundant models and static benchmarks. The counterintuitive finding that personalization attempts can often hurt performance is insightful and valuable to know.

### Weaknesses
1. Evaluation with LLM judge: the scoring system relies heavily on an LLM-based judge to score personalization quality. However, the paper does not provide sufficient evidence that the judge’s assessments are reliable or aligned with actual human preferences.  Since the grader model itself embodies particular stylistic biases, it is unclear whether higher scores genuinely reflect improved personalization rather than conformity to the judge model's implicit preferences. It would be helpful to have some sort of evaluation over the reliability of these model-based judges. For example, user studies, human-in-the-loop experiments, calibration checks etc.

2. Persona representation: even though the personas are grounded in psychological research, it remains unclear how well these explicit preference dimensions capture the richness and diversity of real user preferences. User preferences are generally assumed to be high-dimensional, unstructured, and sometimes even internally inconsistent, these modeling dimensions are still fundamentally low order approximations to these preferences.  As a result, models evaluated on this benchmark may be learning to match stylized persona traits rather than genuine personalization applicable to real-world users.

### Questions
See weakness above

### Soundness
3

### Presentation
3

### Contribution
3
