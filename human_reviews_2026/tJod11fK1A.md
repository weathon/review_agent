# Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Long-context models are essential for many applications but face inefficiencies in loading large KV caches during decoding. Prior methods enforce fixed token budgets for sparse attention, assuming a set number of tokens can approximate full attention. However, these methods overlook variations in the importance of attention across heads, layers, and contexts.

To address these limitations, we propose Tactic, a sparsity-adaptive and calibration-free sparse attention mechanism that dynamically selects tokens based on their cumulative attention scores rather than a fixed token budget. By setting a target fraction of total attention scores, Tactic ensures that token selection naturally adapts to variations in attention sparsity. To efficiently approximate this selection, Tactic leverages clustering-based sorting and distribution fitting, allowing it to accurately estimate token importance with minimal computational overhead.

We show that Tactic outperforms existing sparse attention algorithms, achieving superior accuracy and up to 5.14x decode attention speedup. This improvement translates to an overall 1.51x end-to-end inference speedup, making Tactic a practical and effective solution for long-context LLM inference in accuracy-sensitive applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Tactic, an adaptive and calibration-free sparse attention mechanism for long-context LLMs. Unlike prior methods that use a fixed token budget, Tactic dynamically selects tokens by targeting a fraction of the total attention scores, allowing it to naturally adapt to sparsity variations across heads and layers. To achieve this efficiently, it leverages clustering-based sorting and distribution fitting. The method reports significant improvements, including up to a 7.29× speedup in decode attention and a 1.58× end-to-end inference speedup while maintaining accuracy.

### Strengths
1.  The core idea of targeting a fraction of attention scores instead of a fixed budget is elegantly designed to address the limitations of fixed-budget methods in handling variable attention sparsity.
2.  The method is calibration-free, which significantly lowers the barrier for practical adoption and simplifies its integration into existing inference pipelines.
3.  The combination of clustering and distribution fitting appears to be a computationally efficient approach to enable the adaptive selection mechanism.
4.  The reported performance gains are substantial, suggesting that Tactic is a promising solution for accelerating long-context LLM inference.

### Weaknesses
1.  The method has two main components, clustering and cumulative attention score based selection. It's not clear how the two components work together to achieve the accuracy gains. An ablation study is needed to isolate the contributions of the two components.

### Questions
1.  Could you provide an ablation study that isolates the contributions of the clustering-based sorting, the distribution fitting components and the cumulative attention score based selection to the overall performance?

### Soundness
3

### Presentation
3

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
Paper points out that using a fixed number of tokens for attention is inefficient because different inputs require appropriate or different amounts of focus. To solve this, TACTIC adjusts the number of tokens dynamically—it keeps only as many tokens as needed to capture most of the important information in the attention layer. It does this by grouping similar tokens with K-means clustering during setup and by modeling how attention changes during generation. This method makes the model run much faster (7× faster in attention calculations) while keeping accuracy almost the same.

### Strengths
It is a smart and adaptive approach to improving efficiency in large language models. Instead of using a fixed number of tokens for attention, which can waste computation on unimportant parts it dynamically selects only the most relevant tokens using clustering and distribution fitting. This makes the smaller model(8~9B)  much faster while keeping accuracy almost unchanged, and it’s done without extra training or calibration, making it easy to apply to existing models.

### Weaknesses
This method mainly focuses on the decode phase, so, it doesn’t speed up the initial setup (prefill) as much. Additionally, although the results are strong, the paper doesn’t thoroughly examine how the method performs on large models(e.g., 70B and above) or various types of tasks, and it lacks detailed statistical analysis across multiple runs to confirm consistency.

### Questions
1. Since the paper focuses on a few model families, is the adaptive token selection strategy architecture-agnostic, as it would support broader applicability?

2. Can the authors provide results across multiple random seeds or include confidence intervals to confirm reproducibility and eliminate the possibility that observed improvements are dataset or run-specific?

3. Has there been consideration to compare against the very latest “heavy hitter / speculative decode / KV eviction + reuse” on real hardware latency, not just attention microbenchmarks?

### Soundness
3

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
3

### Summary
The paper presents a novel adaptive sparse attention mechanism designed for efficient inference in long-context language models (LLMs). 

Unlike existing fixed-budget methods, Tactic dynamically selects tokens based on cumulative attention scores, allowing it to adapt to variations in attention sparsity. This flexibility enhances the model's efficiency in processing long sequences.
Tactic employs clustering-based sorting and distribution fitting techniques to accurately estimate the importance of tokens while minimizing computational overhead. This approach improves the model's ability to handle large amounts of data effectively.
The experimental results demonstrate that Tactic achieves up to 7.29 times speedup in decoding attention and 1.58 times overall inference speedup compared to traditional methods, all while maintaining high accuracy.
The framework is designed to be practical for long-context LLMs, making it a valuable contribution to the field of natural language processing, particularly for applications requiring efficient handling of extensive textual data.
In summary, Tactic represents a significant advancement in adaptive attention mechanisms, providing both efficiency and accuracy for long-context language models.

### Strengths
1. Tactic employs a dynamic approach to token selection based on cumulative attention scores rather than a fixed budget. This adaptability allows the model to efficiently handle variations in attention sparsity across different contexts, leading to improved performance in long-context language models  
2. The experimental results demonstrate that Tactic achieves up to 7.29 times speedup in decoding attention and 1.58 times overall inference speedup compared to traditional methods. This efficiency is crucial for practical applications of long-context language models, making Tactic a valuable contribution to the field  
3. Despite the focus on efficiency, Tactic maintains high accuracy levels. The method's design ensures that the attention distance error is minimized, providing a theoretical guarantee on the accuracy of the attention approximation  
4. Tactic utilizes clustering-based sorting and distribution fitting techniques to estimate token importance effectively. This innovative approach reduces computational overhead while ensuring that critical tokens are selected based on their relevance to the current query, enhancing the overall performance of the model

### Weaknesses
1. The adaptive nature of Tactic, which involves dynamic token selection based on cumulative attention scores, may introduce additional complexity in implementation compared to simpler fixed-budget methods. This complexity could pose challenges for practical deployment in certain environments.
2. Tactic relies heavily on accurately estimating attention scores for effective token selection. Any inaccuracies in this estimation could lead to suboptimal performance, potentially affecting the overall effectiveness of the model
3. While Tactic uses clustering to improve efficiency, the initial clustering step may introduce computational overhead, especially for large datasets or models. This could negate some of the efficiency gains achieved during the decoding phase 
4. The performance improvements demonstrated in the paper are based on specific models (e.g., Llama-3.1-8B-Instruct). There may be concerns regarding how well Tactic generalizes to other architectures or tasks, which could limit its applicability in diverse scenarios

### Questions
1. Could you provide more detailed insights into the clustering methodology used in Tactic? Specifically, how do you determine the optimal number of clusters, and what criteria do you use to evaluate the effectiveness of the clustering?

2. How well does Tactic generalize to other language models beyond Llama-3.1-8B-Instruct and MegaBeam-Mistral-7B-512k? Are there any limitations observed when applying Tactic to different architectures or tasks?

3. What is the impact of different hyperparameter settings on the performance of Tactic? For instance, how does varying the average cluster size or the number of iterations in K-means affect the results?

4. Could you elaborate on the real-time performance metrics used to evaluate Tactic? How do you measure the efficiency gains during actual inference as opposed to theoretical evaluations?

### Soundness
3

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
4

### Summary
This paper aims to address critical inefficiencies in long-context LLM inference by proposing a calibration-free sparse attention mechanism that dynamically selects tokens based on cumulative attention scores to meet a target fraction, rather than relying on a fixed token count.

### Strengths
1. The paper is well structured and easy to follow.
2. This paper addresses a real, practical problem, and this area is promising.
3. The evaluation is comprehensive.

### Weaknesses
1. Vague practical cost tradeoffs: Clustering overhead is <6% of prefill time, but no breakdown of how this scales with ultra-long sequences (>128K) or batch inference—critical for real-world high-throughput serving.
2. The idea is similar to several other papers, such as Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning.
3. Typo: line 43, reference

### Questions
1. Any ablation study on distribution fitting? Compare \(y=\frac{a}{x}+b\) with power-law and analyze the fitting error impact on results.
2. The selector will cause any latency overhead; some papers have mentioned this [1]. Can you add some analysis?

[1] HShare: Fast LLM Decoding by Hierarchical Key-Value Sharing.

### Soundness
2

### Presentation
2

### Contribution
2
