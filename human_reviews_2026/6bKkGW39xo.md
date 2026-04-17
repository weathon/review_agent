# Mitigating Position Bias in Transformers via Layer-Specific Positional Embedding Scaling

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) still struggle with the ''lost-in-the-middle'' problem, where critical information located in the middle of long-context inputs is often underrepresented or lost. While existing methods attempt to address this by combining multi-scale rotary position embeddings (RoPE), they typically suffer from high latency or rely on suboptimal hand-crafted scaling. To overcome these limitations, we introduce a layer-specific positional embedding scaling (LPES) method that assigns distinct scaling factors to each layer. LPES achieves a more balanced attention distribution without fine-tuning model parameters or increasing inference delay. A specially designed genetic algorithm is employed to efficiently select the optimal scaling factors for each layer by incorporating Bézier curves to reduce the search space. Extensive experiments demonstrate that LPES effectively mitigates positional attention bias and delivers consistent improvements across multiple long-context benchmarks, yielding up to an $11.2$\% accuracy gain on the key-value retrieval dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel method to mitigate the position bias problem of LLMs. The method uses genetic algorithm to search layer-wise ROPE scaling factors, and use Bézier curve to reduce the number of points to be searched. Compared to previous methods such as MsPoE, this method achieves better mitigation of position bias, while does not need additional computation in inference.

### Strengths
1. Clear structure, and easy to understand
2. There are adequate baselines and datasets in the experiments
3. This method doesn't need to fine-tune the model, and no additional computation is required during inference, which is superior to previous methods.

### Weaknesses
1. My main concern is that, this method's main process is to use genetic algorithm to search for scale factors, which is very similar to LongRoPE's [1] search algorithm. Just like the same method is transferred from pre-training dataset to QA dataset, from dimension-wise to layerwise.

[1] Ding, Yiran, et al. "LongRoPE: extending LLM context window beyond 2 million tokens." Proceedings of the 41st International Conference on Machine Learning. 2024.
    
2. Although using Bezier curves to search for scale factors is an innovative thought, its necessity is not so strong. From section 4.3.1, it seems that the Bezier curve does not have a significant advantage over linear interpolation.
    
3. The evaluation results of Qwen2.5 only appeared in Experiment 1, not in Experiments 2 and 3. And the other evaluated models are relatively old, and only have a 4k context window (which is too short for long-context problems). So there are doubts about the generalization and practical application value of the method.

4. This method may not applied to models already using LongRoPE or Yarn.

### Questions
Why the results of Qwen2.5 are not shown in experiment 2 and 3?

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
3

### Summary
This paper presents a layer-specific positional embedding scaling (LPES) method designed to improve the LLM's attention on the information located in the middle of long contexts.
The core contribution of LPES is an efficient search method that uses a genetic algorithm to optimize the control points of a Bézier curve , which in turn defines the scaling factor for each layer of the LLM, significantly reducing the search space.
Experiments on several long-context benchmarks demonstrate that LPES improves accuracy and is faster at inference than baselines like Ms-PoE and MoICE.

### Strengths
1. This paper introduces a novel method to solve the non-trivial problem of selecting optimal scaling factors for each layer, which
utilizes Bézier curves to constrain the search space and employs a specially designed genetic algorithm.
2. This  paper focuses on achieving a balanced attention distribution without fine-tuning and without increasing inference cost, which is highly significant and practical for deploying LLMs in real-world, long-context applications.
3. This paper provides a clear and valuable analysis justifying its choice of a genetic algorithm over gradient descent.

### Weaknesses
1. The necessity of Bézier curves is not strongly justified. As shown in Table 6, the Bézier curve's performance is only marginally better than the simpler linear interpolation method. This small gain calls into question whether the added complexity of implementing and optimizing Bézier curves is necessary. The authors should provide a more critical analysis of this trade-off. 
2. The method's effectiveness hinges on the scaling factors found by the genetic algorithm, which in turn depend on a small search dataset  and a weighted fitness function. However, it is still unclear how sensitive the "optimal" Bézier curve is to the choice of this small search dataset. 
3. Formatting errors: A minor but notable issue is the inconsistent bold formatting in the results tables. By convention, the best-performing metric (typically the maximum value) is bolded to help the reader quickly identify the top method. However, this rule is not applied consistently. For example, in Table 2, for the StableBeluga-7B model, the NarrQA column incorrectly bolds the score 9.910(LPES), rather than the maximum value (10.73 for Baseline). This type of error, where non-maximum values are incorrectly bolded, appears in Tables 1, 2, and 3, and may mislead a reader who is scanning the tables.

### Questions
1. Is there a stronger theoretical reason, or an experiment on a larger LLM with more layers, that demonstrates a more critical advantage for the non-linear, smooth curve-fitting that Bézier provides over simple linear interpolation? Besides, a fine-grained evaluation—for example, testing at 10% intervals in Table 6 may help to reveal the differences.
2. What happens if you optimize the factors on the key-value retrieval dataset and then apply them to the MDQA benchmark?  Could you clarify how much the "optimal" curve depends on the choice of this search set?
3. The authors should perform a careful review of Tables 1, 2, and 3 to ensure that the bold formatting is corrected and applied consistently, highlighting only the best score in each column for each model-specific comparison.

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
The paper proposes LPES, which assigns unique scaling factors to each transformer layer, it uses Bezier curves to simplify the search for optimal factors and a genetic algorithm to find them efficiently, enabling balanced attention without extra inference latency.

### Strengths
1. A major advantage is that LPES achieves better performance on MDQA tasks with no extra overhead during inference compared to baselines.
2. The paper is exceptionally clear and well-structured.

### Weaknesses
1. The layer-wise assignment of scaling factors in LPES is not a novel concept. While the introduction of the Bézier curve to further constrain the search space is appreciated, it is neither a necessary nor a unique approach, which ultimately weakens the overall novelty of this paper.

2. The authors lack experiments using larger-scale models and larger context windows to fully validate the effectiveness of LPES.

### Questions
1. Given that different attention heads are known to possess distinct functional roles, a natural extension would be to assign an individual scaling factor to each head. LPES, however, employs a single, shared scaling factor across all attention heads within a layer. While this choice significantly reduces the search space, it potentially leads to sub-optimal performance. Could the authors provide a more detailed justification for this design choice?

2. While the paper describes the genetic optimization algorithm, providing a performance curve during the optimization process—specifically, showing the performance of each generation on the training data—would significantly strengthen the argument and be more convincing.

### Soundness
2

### Presentation
3

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
This paper aims to address the “lost-in-the-middle” problem by assigning a distinct rotary position embedding (RoPE) scaling factor to each layer. While previous works combine the output from multiple inferences with different scaling factors, the proposed method is claimed to require inference only once. In determining the scaling factor for each layer, this paper proposes a genetic algorithm to adjust Bezier curves corresponding to different scaling factor settings.

### Strengths
* The proposed method seems straightforward. The empirical evidence presented demonstrates that when the scaling factor is appropriately set, the proposed method can achieve good results with a single forward inference.

* The paper is clear and easy to follow; the proposed method is clearly demonstrated.

* The design of the genetic algorithm for the scaling factor setting is interesting and creative. Generally, the perspective of searching and adjusting the scaling factor provides a brand new perspective.

### Weaknesses
* The time complexity of the algorithm is of primary concern. Despite this paper emphasizing that previous methods require multiple forward passes, the proposed method requires first determining the scaling factor. The proposed genetic algorithm involves inference on an additional dataset and gradually adjusting the scaling factor for each layer, which seems to require extensive computation.

* The extensive efforts devoted to adjusting the scaling factor of each layer further raise my concern about the robustness of the proposed method, where the good performance may highly rely on an appropriate scaling factor setting and may be limited when there is no high-quality dataset to apply the genetic algorithm on. In Tables 2 and 3, the improvement of the proposed method over the previous work, MoICE, appears marginal.

* While the paper provides results showing that Bezier curves achieve better performance than the step function and linear interpolation, the choice of the curve type could use more insights and reasons to explain why Bezier curves are a good choice.

### Questions
My questions are generally related to the concerns I mentioned above.

* Could the authors provide a time complexity analysis of the proposed genetic algorithm? Running the genetic algorithm for a dozen epochs seems computationally expensive.

 * More analyses on how the scaling factor would affect the model's performance would be appreciated. Since it is the core contribution of the proposed method, the current analyses on the scaling factor do not appear to be in-depth. Why are Bezier curves a good choice for scaling factor setting? Are there any insights that could inform the setting of the scaling factor?

* While the scaling factor of RoPE is the focus of the proposed method and many previous methods, could the proposed genetic algorithm be applied to other methods that involve changing the scaling factor?

### Soundness
2

### Presentation
3

### Contribution
2
