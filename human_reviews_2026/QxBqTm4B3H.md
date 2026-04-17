# Scientific logicality enriched methodology for LLM reasoning: A practice in physics

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
With the continuous advancement of reasoning abilities in Large Language Models (LLMs), their application to scientific reasoning tasks has gained significant research attention. Current research primarily emphasizes boosting LLMs' performances on scientific QA benchmarks by training on larger, more comprehensive datasets with extended reasoning chains. However, these approaches neglect the essence of scientific reasoning process -- logicality, which is the rational foundation to ensure the validity of reasoning steps leading to reliable conclusions. In this work, we make the first systematic investigation into the internal logicality underlying LLM scientific reasoning, and develop a scientific logicality enriched methodology, including a set of assessment criteria and data sampling methods for logicality-guided training, to improve the logical faithfulness as well as task performance. Further, we take physics, characterized by its diverse logical structures and formalisms, as an exemplar discipline to practise the above methodology. For data construction, we extract scientific problems from academic literature and sample a high-quality dataset exhibiting strong logicality. Experiments based on three different backbone LLMs reveal that: 1) the training data we constructed can effectively improve the scientific logicality in LLM reasoning; and 2) the enriched scientific logicality plays a critical role in solving scientific problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work focuses on the logicality of the scientific reasoning process.
The authors propose three new metrics to assess the logicality of an LLM’s reasoning process: Logical Fidelity, Causal Connection, and Inferential Progress.
They construct an 80K logicality-related SFT dataset and a 864-example in-domain test set.
Experiments have shown that their constructed training dataset can effectively improve LLM logicality in physics reasoning and the final task performances.

### Strengths
1. The idea to check logicality from different perspectives is interesting and proven useful for SFT.
2. The proposed PHYSLOGIC benchmark might be useful for related domains.
3. The experiments are comprehensive, and the whole paper is full of details.
4. The collected SFT dataset in itself can be useful for enhancing the physics reasoning ability of LLMs.

### Weaknesses
1. This paper focuses exclusively on physics reasoning. It might be better to discuss other possible related domains, for example, math and code. I recommend that the authors at least evaluate their trained models as well as baselines on other popular benchmarks (e.g., AIME25, LiveCodeBench) to see the generalizability of their method.
2. As the PHYSLOGIC is collected from arXiv papers. It is possible that PHYSLOGIC has a data contamination issue.
3. Lack of an ablation study on the different percentiles for the Logical-Distillation (L-D) method. It seems that the authors use 50% directly.
4. The proposed metrics heavily rely on "reasoning step segmentation." How to segment such long responses from large reasoning models into "reasoning steps"? I think there is no consensus on this problem, and I believe a different segmentation strategy will influence these metrics.
5. I recommend that the authors spend more text explaining the definition of the new metrics, especially Causal Connection. It is not very intuitive to see the meaning from the formula.
6. It is better to report the new proposed three metrics on the training data of different methods.
7. I am uncertain about the reliability of the new metrics. You know, many works try to define some process-level metrics, but many of these are not that reliable.

### Questions
1. How do you judge the answer correctness of those open-ended questions on PHYSLOGIC? Are you relying on the rule-based method?
2. I am curious about the range of the proposed three metrics.
3. How to set the coefficients in the formula in L225?
4. How to get the Nexus and Weight for each question? Is the Nexus unique for a certain question? (Maybe we can combine the two steps in Nexus into a "bigger" one)
5. In your metrics, you mainly use the cosine similarity of text embeddings. Have you ever tried NLI models, like [1-2].
6. I am also curious whether Logical Nexus is specific to physics reasoning? It seems that we could also define "Nexus" or extract it using your prompt for other domains (math, other STEM domains).


[1] Golovneva, O., Chen, M., Poff, S., Corredor, M., Zettlemoyer, L., Fazel-Zarandi, M., & Celikyilmaz, A. (2022). Roscoe: A suite of metrics for scoring step-by-step reasoning. arXiv preprint arXiv:2212.07919.

[2] Xu, X., Diao, S., Yang, C., & Wang, Y. (2024). Can We Verify Step by Step for Incorrect Answer Detection?. arXiv preprint arXiv:2402.10528.

### Soundness
3

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
4

### Summary
This paper provides a comprehensive study on logical reasoning in physics domain. The authors model reasoning process as a set of logical nexuses with corresponding weights, and propose to measure logicality via 3 novel metrics: Logical Fidelity, Causal Connection, and Inferential Progress. These metrics are combined to give an overall logical score. Based on this score, they design 2 logic-enhanced data sampling methods for downstream SFT, and create the PhysLogic benchmark. Comprehensive experiments are conducted to show model performances on PhysLogic, SFT improvements of several baseline scientific datasets, and the benefits of logicality-guided data sampling.

### Strengths
1. Propose to measure logicality via 3 novel metrics that each covers one aspect of logical reasoning, and verified the design via ablation study on data sampling performance.
2. A novel data sampling strategy guided by logical score, which help curated a high quality SFT dataset as well as the PhysLogic benchmark.
3. The paper is well written with much of the necessary details provided.

### Weaknesses
1. Lack of justification for using a sentence encoder to embed logical steps. While sentence encoder performs well on encoding contexts and general semantics, it may fail to differ subtle logical differences and especially hallucinations. Since the sentence embedding similarity is the basis for all proposed logical metrics, the authors should first verify the feasibility of using a sentence encoder for encoding logical reasoning steps.
2. Lack of analysis on the properties of the three logical metrics ($\mathcal F,\mathcal O,\mathcal P$). These metrics are intuitively defined but not verified whether they truly reflect respective logical properties. I understand that verifying this is hard for the general case, but a few typical examples would also help.
3. While PhysLogic covered proof questions, the authors did not establish a robust way of evaluating LLM generated proofs, making these type of data less meaningful.

### Questions
1. When finetuned for logical reasoning on physics questions, how does that affect out-of-domain performances, such as on math benchmarks?
2. In Fig.3 Direct Sampling Baseline, did you make sure that $A'$ is aligned with $A$ (or $\mathcal N$)? Since $A$ is curated answer, it is more likely to be of higher quality and correct. Could this be the reason that logic-guided sampling achieved better performance?
3. Following Question 2, if all three data sampling methods have the same set of questions $Q$ and final answer $A$, would the improvements of logic-guided sampling still be as much?

### Soundness
3

### Presentation
4

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
The paper explores how to improve the scientific logicality of large language models in physics reasoning. The authors argue that current reasoning models often produce fragmented and loosely connected reasoning chains that lack human-like logical coherence. To address this issue, they propose a Scientific Logicality Enriched Methodology with three quantitative measures, which are logical fidelity, causal connection, and inferential progress, to evaluate and enhance reasoning quality. They also build a physics-specific dataset where each problem is annotated with logical nexuses that capture the essential steps of scientific reasoning.

Based on this framework, the authors propose two logic-guided data sampling methods for supervised fine-tuning: Reasoning Style Transfer (RST), which turns structured logical nexuses into natural-language reasoning, and Logical-Distillation (L-D), which selects high-logicality samples for training. Experiments on their PHYSLOGIC benchmark and several public physics QA datasets show that these methods improve both reasoning coherence and answer accuracy, suggesting that incorporating logical structure into training data can strengthen scientific reasoning ability.

### Strengths
The paper tackles an underexplored but meaningful problem of improving the logical coherence of scientific reasoning in LLMs. Its proposed framework divides reasoning quality into measurable components such as logical fidelity, causal connection, and inferential progress, providing a structured basis for studying logic-aware model training in the future.

### Weaknesses
1. The definition of the ground truth logical nexus is potentially unstable. Scientific reasoning problems often admit multiple valid solution paths, and some reasoning steps can be rearranged or occur in parallel without affecting correctness. The authors do not discuss how such variability is handled. If many correct nexuses are omitted or only one canonical sequence is annotated, the proposed metrics may underestimate model performance and weaken the validity of the conclusions.

2. In Table 4, the out-of-domain results show very limited improvement on the DeepSeek-R1-Distill-Qwen-7B model, suggesting that the claimed generalization benefit may not hold consistently across architectures.

3. The data scales used for RST (80k) and Logic-Distill (40k) differ substantially, which makes it difficult to attribute performance differences purely to the proposed methodology. Although Section 4.4 includes a scaling-law analysis, the discussion remains qualitative and does not include experiments where both methods are trained on exactly the same number of samples, making it unclear how much of the reported gain comes from logicality rather than data quantity.

4. The evaluation heavily relies on the authors’ own logicality metrics (logical fidelity, causal connection, and inferential progress), which are derived from the same annotated nexus structure used for training. This creates a strong overlap between the training signal and the evaluation criteria. Without human judgments or external benchmarks, it is difficult to confirm whether higher metric scores truly correspond to more logically sound reasoning, or simply reflect closer alignment to the annotated schema. Moreover, the relationship between these logicality metrics and task accuracy has not been empirically demonstrated, leaving it unclear whether improvements in logicality genuinely translate into better reasoning performance.

### Questions
1. In the computation of logical fidelity, the greedy matching strategy seems overly simplistic. Would more structured alignment methods (e.g., dynamic programming or global matching) yield more reliable results?

2. I understand that SFT typically operates on natural language sequences, but I am curious about the RST design choice. Since the logical nexuses already represent explicit reasoning structures, is converting them back into a natural-language reasoning chain essential? Would direct supervision over structured logic be more effective?

3. Could the authors clarify why the RST and Logic-Distill datasets were constructed with different sample sizes? Was the smaller size of Logic-Distill due to high-score filtering, and have the authors examined whether the improvements stem from logicality rather than data quantity?

### Soundness
2

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
In this work, the authors discussed a better approach in measuring the reasoning progress of scientific reasoning problems (using physics reasoning as an example), which follows a more discipline-specific general strategy than regular reasoning problems like math reasoning. Therefore, the authors propose a nexus-based scoring method to measure the logical fidelity, casual connection and inferential progress of reasoning steps by comparing the vector representations of the reasoning steps with the logical nexuses created by human experts. They curated a dataset from preprint scientific papers with reformulated corresponding SFT data employing their proposed pipeline. The result shows that the reasoning trajectories incorporated their logical filtering achieves better reasoning gain than directly employing the raw question-answer data on the Qwen-instruct and Qwen-distill models.

### Strengths
The faithful measurement of reasoning progress is an important problem in sharpening the understanding of the reasoning ability of model beyond benchmark performance. The authors unveil this problem from the physics reasoning perspective with measureable dimensions. The result also brings reasoning improvements on several benchmarks, which excludes the impact of several other factors like data volume.

### Weaknesses
- Although there are various ways to create such high-quality dataset for training, the entire pipeline relies on the dot-product of step representations, which is conducted through a sentence encoder and would bear significant information compression loss. Therefore the reliability of metrics derives from the embedding dot products becomes questionable specially for problems of higher complexities.
- The effectiveness seems to be undermined with reasoning-based model (the Qwen-distill ones) where the original model achieves the best performance on two of the benchmarks.

### Questions
- i am still not clear about how the reasoning style transfer is conducted in terms of the incorporation of the weighted nexuses. Are they directly fed into the model generator?
- What is the step separation strategy to segment the reasoning traces if not specified, I was not able to find it?

### Soundness
3

### Presentation
2

### Contribution
2
