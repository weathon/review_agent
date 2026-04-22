# Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Large language models demonstrate impressive results across diverse tasks but are still known to hallucinate, generating linguistically plausible but incorrect answers to questions. Uncertainty quantification has been proposed as a strategy for hallucination detection, requiring estimates for both global uncertainty (attributed to a batch of responses) and local uncertainty (attributed to individual responses). While recent black-box approaches have shown some success, they often rely on disjoint heuristics or graph-theoretic approximations that lack a unified geometric interpretation. We introduce a geometric framework to address this, based on archetypal analysis of batches of responses sampled with only black-box model access. At the global level, we propose Geometric Volume, which measures the convex hull volume of archetypes derived from response embeddings. At the local level, we propose Geometric Suspicion, which leverages the spatial relationship between responses and these archetypes to rank reliability, enabling hallucination reduction through preferential response selection. Unlike prior methods that rely on discrete pairwise comparisons, our approach provides continuous semantic boundary points which have utility for attributing reliability to individual responses. Experiments show that our framework performs comparably to or better than prior methods on short form question-answering datasets, and achieves superior results on medical datasets where hallucinations carry particularly critical risks. We also provide theoretical justification by proving a link between convex hull volume and entropy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a geometric framework for understanding uncertainty in large language models, distinguishing between global and local uncertainty. The authors propose a black-boxed method to both detect and mitigate hallucinations by modeling uncertainty from a holistic perspective. The paper evaluates the approach on several benchmarks, showing moderate performance improvements in some cases.

### Strengths
This paper proposes a novel and conceptually interesting perspective on hallucination detection and mitigation through geometric modeling of uncertainty. It introduces a distinction between global and local uncertainty, which offers a potentially useful framework for understanding LLM confidence. The paper also propose an appealing high-level idea to address hallucination detection and mitigation in a unified framework.

### Weaknesses
(1) My main concern is the lack of essential ablations. As a sampling based method, key hyperparameters such as sampling temperature and number of samples are not studied, but rather fixed to the same value through out the paper. This makes it difficult to assess the generalizability of the proposed approach.

(2) While the authors highlight the importance of medical data, the proposed method on MedicalQA underperforms baselines in AUROC on most models in Table 1.  Also, there's typo in the color code in table 1-- on CLAMBER dataset llama3.1 8B, performance of proposed method should not be highlighted for AUROC.   

(3) Some simple baselines are missing. On the detection task on open-source model, how does the performance compared to simple perplexity based detection? On the mitigation task, how does the performance compared to majority vote among the generated answers? -- this is particularly interesting since the design of detection score is to find consensus.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a hallucination detection method that lies within a geometric framework that uses black-box model access. The framework includes both global uncertainty estimates through measuring geometric volume, while local uncertainty estimates use geometric suspicion. In doing this, the framework is able to capture both global and local uncertainty in the LLM responses, quantifying the reliability of the response through semantic boundary points.

### Strengths
- The theoretical framework (Appendix A)  to justify allows for a higher-level understanding of the ideas proposed in the paper. I would suggest you find a way to include this in the main paper. 
- The benchmarks used to evaluate the framework are adequate and diverse. Although most are focused on medical data, there are more general purpose datasets used as well.

### Weaknesses
- The performance, as reported in table 1, is not consistently higher than other baselines. More experiments need to be conducted to understand why this is the case. It seems that P(True), the simplest baseline out of all of them, outperforms in certain scenarios, so considering the complexity associated with the proposed approach in comparison with P(True), there needs to be better justification by the others. 
- A complexity and/or time analysis of the framework is missing from the paper.

### Questions
see above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This manuscript presents a new framework to quantify the uncertainty of LLM responses from a geometric perspective. The framework provides both a group-wise uncertainty score and a response-wise uncertainty score based on archetypal analysis. Experiments on several QA benchmark are performed to validate the effectiveness of the approach.

### Strengths
1.	Using archetypal analysis to quantify the uncertainty of LLM responses provides a new perspective.
2.	Providing response-wise uncertainty is of more practical values than existing prompt-wise uncertainty metrics.

### Weaknesses
1.	The main claim “this is the first sampling-based black-box method that differentiates high and low uncertainty responses within a batch.” is not true. See [1], where the proposed semantic density metric also measures sampling-based response-wise confidence/uncertainty without accessing the internal LLM state.
2.	The design of the local uncertainty, i.e., the sum of ranks in three simple heuristic metrics, is not supported by any solid mathematical justifications. 
3.	The interpretation of the experimental results need to be further clarified. See “Questions” below.
4. Several related baseline methods are missing in the current experiments. See “Questions” below for more details.

### Questions
1.	Can you add more discussions and compare to Semantic density [1], which is also a sampling-based response-wise confidence/uncertainty metric without access to the internal LLM state? Moreover, please also consider comparing to Degree [2] and the length-normalized likelihood [3].
2.	When converting the responses into embedding space, the prompt is not considered as a context. This may cause problems for measuring the semantic relationships among the responses. One example: Q: “What is the capital of France?” A1: “Paris”, A2: “the capital of France is Paris”. A1 and A2 actually mean the same thing under this context, but they will have very different embeddings without considering the original questions as context. Have you considered this limitation? 
3.	In the "Convex Hull Approaches" part of “Related Works”, existing works are criticized to oversimplify the problem by using PCA to reduce the embedding dimensionality to two. However, the proposed approach also uses PCA to reduce the embedding dimension (to 15 dimensions). How do you make sure 15 dimensions are sufficient to preserve important information? Do you have an ablation study with different PCA dimensions?
4.	The likelihood of each sampled response is not utilized in the local uncertainty calculations. Using this information can potentially reduce the cost of sampling, i.e., we don’t need to sample the same response multiple times to estimate the output distribution. What is your consideration here?
5.	In the experimental setup, it is stated “to classify response sets as reliable or hallucinated…”. What do you exactly mean here? Are you using the default answer to represent the response set?
6.	Did you handle the long-form question/answer in medicalQA in a different way than other shorter QA benchmarks? If not, do you think one uncertainty score for the entire long answer, which may include multiple claims, is sufficient?
7.	Do you have an explanation or analysis about why the answer selection does not work well in Qwen3-8B? What makes the performance difference so different across different model?

[1] Xin Qiu, Risto Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space, Advances in Neural Information Processing Systems (NeurIPS), 2024

[2] Zhen Lin, Shubhendu Trivedi, Jimeng Sun, Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models, Transactions on Machine Learning Research, 2024

[3] Kenton Murray, David Chiang. Correcting Length Bias in Neural Machine Translation. In Proceedings of the Third Conference on Machine Translation, 2018.

### Soundness
2

### Presentation
2

### Contribution
2
