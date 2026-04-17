# EAMET: ROBUST MASSIVE MODEL EDITING VIA EMBEDDING ALIGNMENT OPTIMIZATION

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Model editing techniques are essential for efficiently updating knowledge in
large language models (LLMs). However, the effectiveness of existing approaches
degrades in massive editing scenarios, particularly when evaluated with
practical metrics. Their robustness is also limited in context-rich settings or
when editing multiple facts of the same subject simultaneously. We attribute
these failures to the embedding misalignment among knowledge items, which
undermines editing reliability at scale. To address this, we propose EAMET
(Embedding Alignment Model Editing in Transformers), which addresses this issue
by aligning the space of key and residual embeddings. Extensive experiments
across six LLMs and three datasets demonstrate that EAMET consistently
outperforms existing methods, achieving about 90\% editing efficacy when editing
10k facts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper aims to address the sharp performance degradation of large language models (LLMs) when performing massive simultaneous factual edits. The authors attribute this failure to embedding misalignment, a geometric inconsistency between the key embeddings representing knowledge and the residual embeddings responsible for executing updates, which leads to information loss during aggregated updates.

To tackle this issue, they propose EAMET, a novel model editing method whose core idea is to progressively and proactively align the structures of the key and residual embedding spaces during the optimization of each knowledge update, guided by KL divergence and MSE losses. Extensive experiments demonstrate that EAMET significantly outperforms existing methods, achieving higher accuracy and robustness when editing thousands of facts, particularly in challenging and realistic scenarios such as long-prefix interference and multi-point edits on the same subject.

### Strengths
* This paper attributes the performance degradation of existing model editing methods in large-scale realistic editing scenarios to **embedding misalignment**, which is a novel and interesting perspective.
* The paper conducts an in-depth theoretical and empirical analysis of **embedding misalignment**.
* The paper proposes **EAMET** to address the problem of **embedding misalignment**.
* Extensive experiments demonstrate the effectiveness of **EAMET**.

### Weaknesses
The assumption regarding **embedding misalignment** is overly strong, which concerns the theoretical foundation of the paper. Although **EAMET** shows strong experimental results, the theoretical aspect is an essential part of the paper’s contribution. Please refer to the **Questions** section for details.

### Questions
* In Equation (20), the paper assumes that in a large-scale editing batch, any knowledge update vector $(r_i)$ can be approximately represented as a weighted average of all other update vectors $(r_j)$ within the same batch. While this assumption might hold with a small $\epsilon_i$ in semantically related cases, it may not hold in semantically unrelated batches, where $\epsilon_i$ could be large. In such cases, the reconstruction residual would be excessively high and lose its interpretative significance. How do the authors explain this issue?
* Is the cosine similarity in Equation $9$ order-invariant? In massive editing, the order of knowledge updates should be arbitrary, so I believe an order-invariant definition should be provided here. If it is not order-invariant, what would the empirical results look like when the order is randomized?
* Why is preserving original knowledge defined as $\Delta C_p = 0$? Theoretically, $C_p$ should be a positive value to ensure the protection of existing knowledge. Practically speaking, removing $C_p$ should lead to a significant drop in editing performance (intuitively), since it serves as a regularization term, especially for preserving existing knowledge (specificity metric). How do the authors justify this design choice?
* Robustness when editing the same subject is indeed important. However, what happens if the facts about the same subject are potentially contradictory? Would the post-edit LLM produce inconsistent answers in such cases?
* Figure 1 is not mentioned in the main text, and the last case (i.e., {Sentences | $𝑠_i$ = Jeep Commander}) requires further clarification of its meaning to make it more reader-friendly.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies why massive editing (e.g., 10k facts at once) breaks many model-editing methods and proposes EAMET, which aligns key and residual embedding spaces during the edit step. Under the stricter metric and in harder settings, EAMET claims high editing efficacy.

### Strengths
1. The problem is interesting where massive editing fails. 
2. It introduces a stricter, more practical success metric tied to generation.
3. The results show the effectiveness of their method.

### Weaknesses
1. The primary experiments appropriately focus on large-batch editing, but the performance under single-edit or small-batch scenarios (e.g., editing only one or a few facts) remains unexplored. It would be valuable to examine whether the proposed alignment mechanism still provides benefits in these simpler settings.
2. The use of KL divergence over similarity-based softmax distributions to measure embedding misalignment is unconventional. The paper should clarify whether similar formulations have been used in prior literature.
3. The strategy for choosing the number of neighbors M in the pairwise MSE alignment term is not discussed.
4.The method introduces multiple additional hyperparameters, making the approach potentially fragile.
5. Although EAMET achieves notable gains in certain large-scale settings, its improvements are not consistently observed across all datasets and model families

### Questions
Please refer to the Weakness.

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
4

### Summary
The paper introduces an innovative approach, EAMET, for enhancing the robustness of large-scale model editing, focusing on aligning key and residual embeddings to improve the editing efficacy in large language models. The research addresses a pertinent issue in model editing, and the proposed method demonstrates promising results across various benchmarks. However, there are several areas that need further clarification and improvement for better understanding and impact.

### Strengths
1. The proposed EAMET framework brings an innovative solution to improve the effectiveness of model editing, particularly in massive editing scenarios. By aligning key and residual embeddings, it overcomes limitations seen in traditional methods, making it a valuable contribution to the field.

2. The paper includes comprehensive experiments on multiple datasets and models, demonstrating the effectiveness of the proposed method in real-world scenarios. The experimental design is solid, and the results are promising, showing EAMET’s superiority over existing methods.

### Weaknesses
1. The current empirical analysis (starting at line 200)  would benefit from a deeper investigation into how the success rate of editing varies across different categories of knowledge, particularly considering their varying degrees of representation inconsistency. This would provide stronger evidence for the challenges addressed by the proposed method.

2. The paper does not provide a detailed complexity analysis of the Key Embedding Preparation step, which involves calculating a large number of cosine similarities.

3. While not a major issue, some points in the writing need improvement. For example, the abbreviation for "CF ZS" in the experiments section is not defined; Formula (15) lacks punctuation; The definition of "N" in Formula (14) is not provided.

4. Formula (14) is central to the paper’s approach, but it is difficult to fully understand without a more detailed explanation. While I can infer its meaning, the explanation provided is insufficient, and I spent considerable time trying to understand it.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes EAMET, a massive model editing method that aims to address a key failure of existing locate-then-edit approaches such as MEMIT/PMET: the embedding misalignment between key embeddings and residual/memory embeddings when many facts are edited simultaneously. The authors provide a theoretical analysis linking reconstruction error in closed-form edits to the mismatch between similarity structures in key space and residual space, then introduce an alignment-based optimization of residual embeddings integrated into a MEMIT-style update. Empirically, EAMET seems to show consistently improved efficacy, robustness, and portability across multiple LLM architectures and factual datasets, with minimal degradation to general capabilities.

### Strengths
+ This paper clearly identifies and theoretically characterizes embedding misalignment as a core scalability bottleneck for existing massive edits approaches.

+ The proposed EAMET is architecturally compatible with MEMIT-style pipelines. It introduces alignment-based optimization of the derived residual embeddings, which makes sense to address the embedding misalignment problem.

+ The experiments with 10k+ edits or long prefixes seem to demonstrate the effectiveness of the proposed method.

### Weaknesses
- It seems that the per-fact residual optimization and alignment steps may be expensive at very large scales. I'm curious about the detailed runtime, memory, and scalability trade-offs with the MEMIT-style baselines.

- The optimization of the alignment seems to be sequential. I also wonder if the optimization order can have a difference to the editing results.

### Questions
Please refer to my summary of weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
