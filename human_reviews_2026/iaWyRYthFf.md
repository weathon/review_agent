# Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Large Language Models (LLMs) face significant inference latency challenges stemming from their autoregressive design and large size. To address this, speculative decoding emerges as a solution, enabling the simultaneous generation and validation of multiple tokens. While recent approaches like EAGLE-2 and EAGLE-3 improve speculative decoding using dynamic tree structures, they often neglect the impact of crucial system variables such as GPU devices and batch sizes.

Therefore, we introduce a new dynamic tree decoding approach called CAST that takes into account inference costs, including factors such as GPU configurations and batch sizes, to dynamically refine the tree structure. Through comprehensive experimentation across six diverse tasks and utilizing six distinct LLMs, our methodology demonstrates remarkable results, achieving speeds up to 5.2 times faster than conventional decoding methods. Moreover, it generally outperforms existing state-of-the-art techniques from $5\%$ to $20\%$. The code is available at \url{https://github.com/EAGLE-Research/sglang-eagle4}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the latency challenge of autoregressive large language model inference by improving speculative decoding efficiency. Prior dynamic draft-tree methods such as EAGLE-2 and EAGLE-3 adapt token proposal structures, but they overlook system-level factors, including GPU characteristics and batch size. The proposed method models inference cost and dynamically adjusts tree depth, branching width, and token verification count to balance acceptance rates against computation overhead. Experiments demonstrate consistent gains and improvement over previous state-of-the-art methods.

### Strengths
- Adeaute related background along the timeline and solid motivation analysis.

- The paper explains its concepts clearly, and the writing is smooth and well-organised.

- The paper presents clear experimental settings and includes a comprehensive set of comparison models.

### Weaknesses
- For the conclution "merely increasing the tree depth and node numbers may not always result in better performance", is there some quantitative analysis / results to support?
- The results are mainly presented in table form. Is there a visualization of the performance trends? The paper also mentions a trade-off — are there additional experiments that explore multi-factor trade-offs in more depth? Just like Figure 5 in Appedndix E.1. Maybe more info in the Appendix could be included in the main body and cut off 1 table.
- 3.2.2 typo "valuesinterpreted" 
- The core contribution of this paper lies in incorporating cost into the model’s decision process. However, the work does not provide concrete examples or detailed formulation for this component. Although the paper mentions that GPU hardware characteristics and batch size should be considered, it is necessary to further clarify the modeling procedure—for instance, which GPU parameters are included and how they quantitatively influence the cost function. This aspect is central to the method and requires empirical evidence to validate its effectiveness. Since the high-level idea is relatively straightforward, the design and justification of the cost function should serve as the main technical contribution.

### Questions
- General questions are given in the weakness part.

- Is there a theoretical upper bound or lower bound under specific settings? Furthermore, do the experimental results align with the theoretical analysis?

- Can you provide an example of how to represent hardware, such as parameters like memory capacity and FLOPS? Is there a unified quantitative representation for hardware characteristics?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the inference latency of llms by improving speculative decoding. The authors argue that existing dynamic tree-structured methods, such as EAGLE-2 and EAGLE-3, are suboptimal because they ignore system-level inference costs, particularly the impact of GPU hardware and batch size. This paper introduces Cost-Aware Speculative Tree, a new dynamic tree approach that explicitly models this trade-off. CAST uses pre-computed cost look-up tables to guide the dynamic construction and reranking of the draft token tree, pruning branches where the computational cost outweighs the expected utility. Experiments across six models and six tasks show that CAST achieves state-of-the-art speedups, outperforming prior methods by 5-20% and autoregressive decoding by up to 5.2x .

### Strengths
- It correctly identifies that SOTA dynamic tree methods ignore critical system costs like batch size and GPU type, which can negate speedups . The proposed cost-utility model, which uses precomputed lookup tables to guide tree construction, is an good solution. 
- A key contribution is the generalization of prior SOTA (EAGLE-2/3), demonstrating they are special cases of this new framework (Theorem 4.1) . The empirical results are strong, showing consistent 5-20% gains over EAGLE-3 and demonstrating superior scalability as batch size increases—a vital metric for production systems.

### Weaknesses
- One of the weakness is the reliance on a new set of hyperparameters, specifically the cost thresholds $C_1$, $C_2$, and $C_3$ and the buffer size $R$, whose selection and sensitivity are not discussed or ablated. 
- The method's practicality hinges on pre-computing cost-lookup tables $S_T(B)$ and $S_D(B)$. While practical, the paper does not sufficiently analyze the cost and complexity of this profiling step, which must be run for different hardware and batching configurations. It is unclear how the $select(c)$ approximation for context length impacts the cost model's accuracy.

### Questions
- How are the crucial thresholds $C_1, C_2, C_3$ and the FIFO buffer size $R$ determined? Please provide a sensitivity analysis for these new hyperparameters.
- What is the practical overhead (e.g., in hours) of generating the $S_T(B)$ and $S_D(B)$ lookup tables for a new model and hardware configuration? How sensitive is performance to the accuracy of this precomputed cost model?
- Algorithm 1 is used for both breadth pruning (Sec 4.1) and reranking (Sec 4.2), but with different cost functions ($c_k^{(i)}$ vs. $c_k$). Can you elaborate on the rationale for using the normalized draft cost for pruning but the normalized target cost for reranking?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces CAST, a dynamic tree-based speculative decoding method that addresses a key limitation in SOTA approaches like EAGLE-3: their failure to account for system-level inference costs. By pre-computing the actual hardware latency for different batch sizes and token counts, CAST reframes tree construction as a cost-utility optimization, dynamically pruning the draft tree's breadth and depth to maximize accepted tokens per unit of time. This cost-aware approach delivers state-of-the-art speedups (up to 5.2x), demonstrating a particularly strong advantage over prior methods in practical, high-throughput batched-inference scenarios where cost-agnostic heuristics fail.

### Strengths
1. The method replaces prior heuristics with a novel and principled cost-utility framework. This formal optimization of acceptance "utility" versus hardware "cost" is a more robust and generalizable approach .

2. Comprehensive and Rigorous Experimentation: Claims are exceptionally well-supported by extensive experiments across 6 models, 6 tasks, and 3 different GPU architectures. This thoroughness confirms the method's effectiveness and generality .

3. The authors correctly use "Speedup Ratio" as the primary metric and insightfully argue against the misleading "Average Acceptance Length," demonstrating a mature understanding of the evaluation problem. The ablation in Table 3 clearly isolates the individual contributions of the new components (DR, DP, BP), validating the paper's design choices.

### Weaknesses
1. Unquantified Profiling Overhead: The method relies on pre-computing cost lookup tables, but the paper never quantifies the one-time profiling cost (e.g., in GPU-hours), which could be a significant practical barrier to adoption.

2. Lack of Hyperparameter Sensitivity Analysis: The new thresholds ($C_1, C_2, C_3$) are critical to the method, but their robustness and the strategy for tuning them are not discussed, leaving a key practical question unanswered.

3. Unclear Intuition for Generalization Claim: The paper claims (Theorem 4.1) that prior work is a "special case" of CAST, but the intuition for why a "cost-agnostic" method maps to a specific linear cost model is not well-explained.

### Questions
1. Cost of Pre-computation: Can the authors quantify the one-time profiling cost (e.g., in GPU-hours) required to generate the cost lookup tables for a single model and hardware setup?

2. Hyperparameter Tuning Strategy: What is the recommended procedure for tuning the thresholds $C_1, C_2,$ and $C_3$, and how sensitive is the method's performance to these values?

3. Cost Model Granularity: What is the performance impact of approximating the context length $c$ using the $select(c)$ function, and how sensitive is the method to this granularity?

4. Intuition for Theorem 4.1: Can the authors provide more intuition for why the cost-agnostic EAGLE-2/3 algorithms are mathematically equivalent to Algorithm 1 with a specific linear cost model?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper argues existing dynamic tree speculative decoding methods (like EAGLE) are "cost-agnostic". They ignore system variables like GPU and batch size, which can paradoxically increase latency. The paper proposes CAST (Cost-Aware Speculative Tree). This method explicitly models these inference costs, often using a lookup table. CAST then dynamically prunes its tree by balancing token utility (acceptance probability) against the actual hardware cost. This cost-aware approach delivers up to 5.2x speedup over standard decoding and outperforms SOTA methods like EAGLE-3 by 5-20% , showing particular strength in batching scenarios.

### Strengths
- The idea of using utility function to consider resource problem in the speculative decoding setting is novel and motivated. And the paper is well organized.

- The way the paper formulates the utility function based on acceptance rate and how to choose the depth is new.

- The experimental results are comprehensive and convincing. The authors validate their CAST method across a wide array of 6 distinct LLMs and 6 diverse tasks, ranging from multi-turn conversation to code generation.

### Weaknesses
- The method introduces precomputation overhead. If the hardware, batching strategy, or even the model (which changes the cost profile) is modified, this entire precomputation step must be redone.

- There are multiple thresholds (at three new cost-utility thresholds) for tuning. What are the overheads?

- The paper considers batch size as a factor and motivation but lacks more comprehensive experiments for that.

- Theorem 4.1: Try to make it self-contained. What does j mean in the formula c_j. And also it is not clear what $\lambda, \delta$ means.

### Questions
- What is the difference between c and n in Section 4?

See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
