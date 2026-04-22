# Why is the LLM unsure? Profiling the Causes of LLM Uncertainty for Adaptive Model and Uncertainty Metric Selection

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Large Language Models (LLMs) frequently produce fluent yet factually inaccurate outputs, termed hallucinations, which compromise their reliability in real-world applications. Although uncertainty estimation offers a promising approach to detect these errors, existing metrics lack interpretability and offer limited insight into the underlying causes of uncertainty. In this work, we introduce a novel prompting-based framework for systematically analyzing the causes of uncertainty in LLM responses. We design dedicated indicators to quantify each distinct cause and profile how existing uncertainty metrics align with them. Our findings reveal systematic variations in uncertainty characteristics across metrics, tasks, and models. Leveraging these insights, we propose a task-specific metrics/models selection method guided by the alignment of uncertainty characteristics with task requirements. Experiments across multiple datasets and models demonstrate that our selection strategy consistently outperforms non-adaptive baselines, achieving 3-4\% performance improvements and enabling more reliable and efficient uncertainty estimation for LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical issue of hallucinations in Large Language Models (LLMs) by proposing a novel framework to understand the underlying causes of model uncertainty. The authors identify four interpretable causes—Syntax Sensitivity, Semantic Ambiguity, Indecisiveness among Outputs, and Unconfidence when Challenged—and design a prompting-based pipeline to quantify each cause with dedicated indicators. By profiling how existing uncertainty metrics align with these causes across various tasks and models, the paper reveals systematic differences in their behavior. Leveraging these "uncertainty profiles," the authors propose an adaptive method for selecting the most suitable uncertainty metric or model for a given task, which demonstrates consistent performance improvements over non-adaptive baselines.

### Strengths
1.	The study research topic is meaningful and sheds light on future research
2.	The methodology is novel.

### Weaknesses
1.	The paper should be further polished for better presentation.
⦁	In Line225, what does $M$ means?
⦁	In Figure4, what does cutoff ranks mean?
⦁	In Table2, what do the two settings (worst, random) mean in detail?
⦁	In Line 429, how is the NDCG metric computed? Is it proposed by yours? If this is a common metric, please cite with proper reference.
2.	The setting is questionable:
⦁	The authors define 4 possible causes of LLM uncertainty. Is such a taxonomy sound and complete? There is no evidence that these causes could span all causes of LLM uncertainty. The conclusion drawn based on this premise may be erroneous.
⦁	In Table 1, the datasets include CommonsenseQA, MATH, TruthfulQA. However, in later experiments, the authors additionally introduced TriviaQA. The two settings are misaligned and the authors do not explain the hidden reason.
⦁	In Figure3, the two indicators (SA, UC) are generally the lowest, and are presented with a very small value (e.g., 0.02). This also validates my concern about the validity of these four causes of LLM uncertainty.
3.	The experiment result is hard to interpret
⦁	In Table2, the random setting could give a >90% accuracy. The authors do no give a reason for this unusual phenomenon.
⦁	Why the authors use a smaller K (K=5) for scenario 2, but use a larger K (K=8) for scenario 1?
⦁	In the scenario2 setting, triviaQA's accuracy is extremely high compared to other datasets. The authors do not give an explanation for this.

### Questions
1.	In Line378, the authors present a conclusion that 'there is no clear distinction between smaller and larger models in terms of their average uncertainty cause values'. However, this conclusion is given based on just one series of models (Gemma2). Llama3.2-3B and Llama3-8B are not strictly small/large models as their training data are different. Meanwhile, only two sizes for comparison is not sufficient to draw this conclusion. Could the author compare more sizes of LLMs using the same series of models? (Qwen2.5 series: 0.5B, 1.8B, 3B, 7B, 14B (which could be deployed in two 3090))
2.	The four indicators are computed by prompting the LLMs to be evaluated. This may introduce bias, as the result is not only reflects the uncertainty score, but also reflects the instruction following ability of the LLM to be tested. Could the author try to separate any distracting factors to conduct the experiments?

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
3

### Summary
The paper proposes a prompting-based framework that decomposes LLM uncertainty into four interpretable causes and defines indicators to estimate each cause. Using these indicators, the authors build uncertainty profiles for tasks, models, and existing uncertainty metrics, then use profile alignment to adaptively select models/metrics.

### Strengths
1. Understanding the sources of LLM uncertainty is important for evaluation, deployment, and safety. The paper targets a question many researchers face.
2. Framing uncertainty via SS/SA/IO/UC is intuitive and maps neatly to known interventions.
3. Interesting solutions that considering adaptive uncertainty metric selection.

### Weaknesses
1. Cause set may be incomplete. The four causes are plausible but not exhaustive. Other factors (e.g., retrieval grounding quality, tool-use failures, long-context drift, alignment-induced hedging) can also drive uncertainty. The paper should explicitly acknowledge this and discuss extension paths.
2. The pipeline is sequential: paraphrasing → clarification → answering → self-check. Uncertainty introduced (or reduced) at earlier stages may leak into later ones, potentially inflating/deflating UC or IO.
3. The paper would benefit from direct evidence that the SS/SA indicators capture what they claim.

### Questions
1. Do you have human annotations of ambiguity or sensitivity to confirm that SA and SS correlate with actual ambiguity/surface-form brittleness rather than proxy artifacts?
2. What happens if you shuffle the stages (e.g., answer before clarification) or run stages independently? How stable are UC/IO estimates under such changes?
3. Citation format at Line 141 appears incorrect.
4. From Section 5.2 onward, consider adding a one-sentence takeaway at the end of each paragraph to reinforce the main point and improve readability.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a prompting-based framework to systematically analyze and profile the causes of uncertainty in LLM responses. The authors design interpretable indicators for four distinct causes of uncertainty (syntax sensitivity, semantic ambiguity, indecisiveness among outputs, and unconfidence when challenged), and empirically study how existing uncertainty metrics align with these causes. The work further introduces an adaptive model/metric selection strategy based on uncertainty profiles, achieving consistent improvements over non-adaptive baselines across multiple datasets and models.

### Strengths
⦁	Clear motivation and practical relevance for LLM deployment.
⦁	Strong empirical results and broad coverage of datasets/models.
⦁	The framework is interpretable and actionable for downstream applications.

### Weaknesses
⦁	Some experimental and implementation details are lacking (e.g., ablation on indicator importance, prompt sensitivity).
⦁	The method's effectiveness may depend on the quality of prompt engineering and may not generalize to all LLM architectures.
⦁	Limited discussion of failure cases or scenarios where the indicators may be misleading.

### Questions
⦁	How sensitive are the results to the choice/design of prompts for each indicator?
⦁	Can the authors provide more details or ablation on the relative importance of each indicator?
⦁	How does the framework perform on closed-source or instruction-tuned LLMs?
⦁	Are there cases where the indicators disagree or provide misleading signals?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a framework to analyze the cause of LLM uncertainty, categorizing the causes into four types and proposing corresponding metrics to quantify each of them. The cause profiling results are used to guide the selection of existing uncertainty metrics based on the models and tasks.

### Strengths
The taxonomy of four primary causes of LLM uncertainty is well motivated. It forms a good foundation for future work in analyzing the uncertainty causes for LLMs.

### Weaknesses
1.	The design of the estimation pipeline for uncertainty causes has several fundamental issues, as outlined below:
-	For estimating Syntax Sensitivity (SS), the quantitative metric used is doubtful. Model being able to paraphrase the original questions in syntactically very different ways does not necessarily mean that the model's outputs are sensitive to paraphrasing of the question, right? This metric simply measures how well the model is at paraphrasing a question.
-	In estimation of the Indecisiveness among Outputs (IO), the answers are for different prompts, so simply using their likelihood in an entropy-calculation way does not really returns an entropy. Entropy is a property of one single distribution, but what we actually have here are multiple responses from multiple output distributions, each corresponding to one clarified question. This makes the entropy interpretation invalid.
-	For estimating the Unconfidence while being Challenged (UC), it has the same problem as in IO estimation: the answers are not from the same distribution, each answer is from a different distribution (each corresponds to one clarified question), this makes the mathematical interpretation invalid.
2.	From the results in Table 1, the SS score only provides AUROC values slightly higher than 0.5 and the SA score even gives AUROC lower than 0.5. Considering the fact that a random guess can provide an AUROC score of 0.5, it seems that the proposed SS and SA scores do not really indicate any useful uncertainty information of the LLMs.
3.	The literature review in the related work section is limited.

### Questions
1.	For estimating the Semantic Ambiguity (SA), does the distance in the internal embedding space exactly reveal the difference in semantic meanings? Can you provide any supporting literatures?
2.	In the “Metric-Level Attribution” paragraph, how do you calculate P(u,m) here for indicator value and metric score? How do you get the probabilities for them? This seems mathematically invalid. Please provide the concrete math equations for verifications.
3.	How do we get the Metric-Vec? The cause indicator values are corresponding to specific model and task, but irrelevant to the uncertainty metrics. This is confusing.
4.	For the tested SOTA uncertainty metrics, most of them are prompt-wise. Can you also include more response-wise uncertainty scores, e.g., semantic density [1]?

[1] Xin Qiu, Risto Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space, Advances in Neural Information Processing Systems (NeurIPS), 2024

### Soundness
1

### Presentation
2

### Contribution
1
