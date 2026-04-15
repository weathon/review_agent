# Divide, Reweight, and Conquer: A Logit Arithmetic Approach for In-Context Learning

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
In-Context Learning (ICL) emerges as a key feature for Large Language Models (LLMs), allowing them to adapt to new tasks by leverageing task-specific examples without updating model parameters. However, ICL faces challenges with increasing numbers of examples due to performance degradation and quadratic computational costs. In this paper, we propose Logit Arithmetic Reweighting
Approach (LARA), a novel framework that enhances ICL by using logit-based ensembling of multiple demonstrations. Our approach divides long input demonstrations into parallelizable shorter inputs to significantly reduce memory requirements, and then effectively aggregate the information by reweighting logits of each group via a non-gradient optimization approach. We further introduce Bi-
nary LARA (B-LARA), a variant that constrains weights to binary values to simplify the search space and reduces memory usage by filtering out less informative demonstration groups. Experiments on BBH and MMLU demonstrate that LARA and B-LARA outperform all baseline methods in both accuracy and memory efficiency. We also conduct extensive analysis to show that LARA generalizes well to scenarios of varying numbers of examples from limited to many-shot demonstrations. Our codes can be found in https://anonymous.4open.science/r/LARA-F55B.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents the Logit Arithmetic Reweighting Approach (LARA) for enhancing in-context learning (ICL) in large language models (LLMs). LARA improves ICL performance by dividing long prompts into shorter, parallelizable inputs and using logit-based ensembling to optimize memory usage and accuracy. This method also includes a Binary LARA (B-LARA) variant, which constrains the weight of each subgroup to binary values, reducing memory needs and focusing on the most informative inputs. The experiments on the BBH and MMLU benchmarks demonstrate that LARA and B-LARA outperform baseline methods in both efficiency and accuracy across different models.

### Strengths
1. Clear Presentation: The methodology is explained thoroughly, making LARA and B-LARA easy to understand.
2. Practical Relevance: LARA’s memory and computational efficiency make it well-suited for real-world applications, including black-box models.
3. Comprehensive Experiments: Extensive experiments across multiple benchmarks and ablation studies validate the effectiveness of the proposed approach.

### Weaknesses
1. While the empirical results are strong, the paper lacks a deeper theoretical analysis explaining why the logit reweighting improves performance, which could strengthen the foundational understanding of the approach.
2. The proposed method achieves good results on the MMLU and BBH datasets; however, it appears applicable only to multiple-choice questions or those with single-token answers, as it relies on the exact next-token logits for prediction, suggesting limited generalization capability.

### Questions
My main concerns have been listed above. I look forward to the authors' response and am willing to reopen and adjust the score upward.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposed a new approach LARA to enhance in-context learning in LLM. The devised framework divides the demonstration examples into several subgroups, and ensemble the logit results from different groups based on Covariance Matrix Adaptive Evolution Strategy. They also proposes a binary version of weight selection, which is equivalent to subgroup pruning, and only selects more informative groups as the contributing demonstration. The evaluation experiments show the effectiveness of the proposed methods compared to original ICL and other variations.

### Strengths
1. LARA enhances ICL with a non-gradient optimization prompting strategy, and it shows performance gain in different settings.
2. The division strategy can handle the situation of longer context naturally.
3. The paper is clearly write and easy to follow.

### Weaknesses
1. In weight optimization algorithm for B-LARA, it requires about 20 iterations to obtain the final weights, which may results in inefficiency. Although finally only a subset of groups is involved, the intermediate steps still consume a lot. And there is no ablation study on the impact of the iteration number.
2. The method is limited to the case that the vocabulary space of the target answer is already known.

### Questions
In Figure 4, when number of examples used is 16, LARA without weight re-weighting seems performs better than the one with reweighting, could you provide more insights. And how could you verify the effectiveness of the re-weighting strategy if so.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a method to address the issues of performance degradation and quadratic computational costs in the context of many-shot in-context learning (ICL). By grouping demonstrations for parallel input and employing Logit Arithmetic for reweighting and concatenation, the authors cleverly leverage the effective information from demonstrations. This approach alleviates the resource consumption associated with large-shot inputs and enhances the performance of ICL to some extent.

### Strengths
1. The issues of performance degradation and quadratic computational costs in many-shot in-context learning (ICL) are currently hot research topics in the AI community. The approach presented in this paper, which involves Logit Arithmetic concatenation after grouping inputs, is both novel and intriguing. The experiments effectively validate the method's improvements on these problems.

2. The method is supported by a solid theoretical foundation. The design of validation and cross-entropy loss can be viewed as a form of non-gradient fine-tuning for special tasks in the ICL context. B-LARA represents a further optimization of the LARA method's weights. The paper provides detailed formulas and algorithmic derivations to substantiate these claims.

### Weaknesses
1. The use of Logit Arithmetic reweighting and concatenation to leverage effective information from demonstrations is indeed a good approach. However, I am curious about how the authors address the issue of output prefix inconsistency during the decoding process after grouping different demonstrations as inputs. For example, if the first token of the output differs under different shot inputs, how can the subsequent tokens be rigorously and effectively subjected to Logit Arithmetic reweighting? This concern arises because most of the ground truth in the datasets used in this paper consists of a single token.

2. In the "Related Work" section, should the authors discuss more relevant methods for addressing many-shot ICL, such as FocusICL、STRUCTICL? Additionally, would it be beneficial to include these methods as baselines for comparison?

3. ICL is a broadly applicable and effective paradigm; however, the benchmarks used in the experiments lack diversity. For instance, I am curious about the performance of the proposed method on mathematical reasoning problems, such as those found in MATH or GSM8K.

### Questions
As mentioned in the above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes LARA, a new framework that enhances in-context learning (ICL) by ensembling logits from multiple demonstrations, achieving accuracy gains without requiring any parameter updates. The authors also introduce Binary LARA (B-LARA), which further refines efficiency by selectively excluding less informative demonstrations. Experiments on BBH and MMLU benchmarks validate that both LARA and B-LARA outperform standard ICL.

### Strengths
1. The proposed logits re-ensemble provides partial evidence supporting the effectiveness of grouped ICL, serving as an additional rationale for future research on extending this approach toward long ICL.
2. The proposed Binary LARA can further enhance performance while reducing costs.

### Weaknesses
1. The proposed method does not effectively address or mitigate the disadvantages of the related work introduced in the introduction. For instance, while OpenAI's API provides some logits information, LARA fundamentally remains a white-box approach. Additionally, Binary LARA also carries the risk of losing critical information. As a result, the motivation for the method is unclear, and there is a lack of in-depth analysis regarding why LARA is effective.
2. The selected baseline does not effectively demonstrate the validity of the proposed method. The comparison baseline does not specifically address the issue of large-scale demonstrations. Additionally, other relevant grouping-based methods, such as [1] and [2], were not included in the comparison. Additionally, the main text uses RAE, while the table uses IRE.
3. LARA requires iterative computation of optimal weights for the model and dataset before inference, which introduces latency. When there are few samples to infer, or even just a single sample—common in real-world interactions with LLMs—the proportion of additional costs relative to the inference becomes particularly high.

  [1] Structured Prompting: Scaling In-Context Learning to 1,000 Examples.

  [2] Focused Large Language Models are Stable Many-Shot Learners.

### Questions
1. For the experiments in Section 5.5, please provide additional details. For instance, does decoding each token require re-feeding the entire prefix to the API? If so, this would result in significant costs.
2. Where are the results for different candidate numbers of groups in Table 1 and 3?

### Soundness
2

### Presentation
2

### Contribution
3
