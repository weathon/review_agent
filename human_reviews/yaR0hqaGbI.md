# Hierarchical Demonstration Order Optimization for Many-shot In-Context Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 1, 3

## Abstract
In-Context Learning (ICL) is a technique where large language models (LLMs) leverage multiple demonstrations (i.e., examples) to perform tasks. With the recent expansion of LLM context windows, many-shot ICL (generally with more than 50 demonstrations) can lead to significant performance improvements  on a variety of language tasks such as text classification and question answering. Nevertheless, ICL faces demonstration order instability (ICL-DOI), which means that performance varies significantly depending on the order of demonstrations. Moreover, the ICL-DOI phenomenon persists and can sometimes be more pronounced in many-shot ICL, validated by our thorough experimental investigation. Current strategies handling ICL-DOI, however, are not applicable to many-shot ICL, since they cannot overcome two critical challenges: (1) Most metrics measuring the quality of demonstration order rely on subjective judgment, lacking a theoretical foundation to achieve precise quality characterization. These metrics are thus non-applicable to many-shot situations, where the order quality of different orders is less distinguishable due to the limited ability of LLMs to exploit information in long input contexts. (2) The requirement to examine all orders is computationally infeasible due to the combinatorial complexity of the order space in many-shot ICL. To tackle the first challenge, we design a demonstration order evaluation metric based on information theory for measuring order quality, which effectively quantifies the usable information gain of a given demonstration order. To address the second challenge, we propose a hierarchical demonstration order optimization method named HIDO that enables a more refined exploration of the order space, achieving high ICL performance without the need to evaluate all possible orders. Extensive experiments on multiple LLMs and real-world datasets demonstrate that our HIDO method consistently and efficiently outperforms other baselines. Our code can be found at https://anonymous.4open.science/r/HIDO-B2DE/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces HIDO, a hierarchical optimization method, and ICD-OVI, an information-theoretic metric, to address demonstration order instability (ICL-DOI) in large language models. HIDO reduces computational complexity by separately optimizing order within clusters and between clusters. Meanwhile, ICD-OVI measures the information gain from each order to help select the optimal sequence. Experimental results show that HIDO, with dynamic updates to ICD-OVI, outperforms baselines in both predictive accuracy and stability.

### Strengths
1. Innovative Method: The paper introduces HIDO, a novel hierarchical optimization approach that effectively addresses demonstration order instability (ICL-DOI) in many-shot in-context learning, meanwhile reducing the computational complexity of order optimization, a persistent challenge in large language models.
2. Theoretical Rigor: Introducing ICD-OVI, an information-theoretic metric, provides a robust, quantifiable measure of information gain for different demonstration orders, adding a solid theoretical foundation to the method.
3. Strong Experimental Validation: Extensive experiments on multiple datasets and models demonstrate that HIDO outperforms baseline methods in both predictive accuracy and stability, highlighting its practical effectiveness.

### Weaknesses
Although the method shows strong results on several datasets, additional validation on a broader range of tasks, especially outside of text classification, would strengthen the claim of its general applicability in many-shot in-context learning.

### Questions
Could you please specify the hardware used (e.g., type of processors/GPUs, memory) and the approximate computation time required to run the proposed HIDO method on the datasets in the experiments? Thank You

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses the problem of demonstration order instability (DOI) in the in-context learning (ICL) of large language models (LLMs). To study this issue, the authors propose an information theory-based metric called ICD-OVI and introduce a hierarchical demonstration order optimization method named HIDO. They validate their method on five LLMs across nine tasks, comparing it with three baselines.

### Strengths
1. The paper provides a strong and comprehensive theoretical guarantee for their proposed metric for demonstration example ordering.
2. The experiments show consistent improvement of their method against all the baselines presented.
3. The paper is clearly written and easy to follow.

### Weaknesses
1. **Missing Baselines:** Only three baselines are included in the experiments, and there are other key baselines that need to be compared. For example:
    - [1] provides an information-gain-based metric to evaluate the effect of demonstrations, which also has a strong theoretical foundation.
    - [2] selects demonstration examples based on the embedding space distance between examples and input queries.
Further experiments are crucial for enhancing the rigor of this paper.

2. **Outdated Tasks and Datasets:** All the tasks used are text classification tasks, and the datasets are somewhat outdated. Though not a major issue, experiments on newer tasks and datasets (e.g., MMLU, HellaSwag) are necessary for validating the proposed metrics and methods.

[1] Hongfu Liu and Ye Wang. 2023. Towards Informative Few-Shot Prompt with Maximum Information Gain for In-Context Learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15825–15838, Singapore. Association for Computational Linguistics.

[2] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What Makes Good In-Context Examples for GPT-3?. In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics.

### Questions
There is a typo on line 176: it should be “**(3)** Empirically Effective”.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
3

### Summary
In-context learning performance is heavily dependent on examplar order as demonstrated by previous work. This work proposes a score to rank a particular ordering and a hierarchical optimization framework that does not require evaluating every permutation.

### Strengths
The problem of determining an optimal ordering of exemplars is a crucial problem.

### Weaknesses
## Weaknesses
- **[Major]** Why are PDO and GlobalE presented as separate baselines? One minimizes the KL-divergence between the distribution of predicted labels on the probe set and a uniform prior and the other maximizes the entropy of the distribution. These are mathematically identical.
- **[Major]** What is the point of introducing the framework of V-usable information? It is easy to compute the KL divergence between the distribution of labels in your training set and the distribution of predicted labels directly. The problem stated in Eq. 1 can be solved by simply choosing the ordering that attains the lowest loss. 
- **[Major]** What is the interpretation of $\log_2 P_{\text{LLM}}(a \mid \Pi(\mathcal{A}) \oplus \emptyset)$? From my understanding, you are simply concatenating a bunch of labels and  prompting the LLM to get a probability score for the label of a probing example. However, it seems as though prompting the LLM in such a manner can result in gibberish since this type of prompt clearly is very contrived. Why not directly compute an empirical distribution on the true training examples? 
- **[Major]** The framework proposed in this work fixes the order of the exemplars. However, the order should be **query dependent.** Based on the observation of [1], many ICL/RAG works order the exemplars based on their similarity to the query [2,3,4], but this is not discussed or used as a baseline in this work. Compared to LLM inference with long contexts, a simple similarity based reordering should be very cheap.
- **[Major]** It seems like there is no statistically significant difference in performance in the vast majority of the experiments...
- **[Minor]** What is the point of using LLM generate probes? I understand that this saves annotation cost, but in the many shot setting where we are already required to annotate ~150 examples, why not manually annotate a few extra samples to use as a validation set? I am skeptical of whether or not the validation set is reliable when it is purely synthetic.
- **[Minor]** The writing for Section 3 needs significant improvement. In its current state, one must repeatedly look at prior work in order to get any understanding of what the actual problem set up is. See comments below:
1. Many terms are used without being defined. For example, "probing samples" in Line 125 or "optimal ignorance requirement" in Line 184 are not terms that are used in the broader ML community, and should be properly defined.
2. What is $\pi(i)$ in line 194? This is never defined.
3. What is the difference between $\mathcal{A}$ and $A$ in Equation 4? Is this a typo? 
4. It is not mentioned that the "probing samples" are LLM generated until line 199.


## References
[1] Lost in the Middle: How Language Models Use Long Contexts (https://arxiv.org/abs/2307.03172)
[2] In-Context Learning for Text Classification with Many Labels (https://aclanthology.org/2023.genbench-1.14.pdf)
[3] What Makes Good In-Context Examples for GPT-3? (https://arxiv.org/pdf/2101.06804)
[4] Retrieval-Augmented Generation for Large Language Models: A Survey (https://arxiv.org/pdf/2312.10997)

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
As large language models (LLMs) grow in scale and capability, in-context learning (ICL) and, in particular, many-shot learning have become predominant approaches for machine learning practitioners. This work addresses a significant challenge currently facing ICL and many-shot learning: the dependency of LLM performance on the order of examples within prompts. Existing literature has shown substantial performance variations when examples (or demonstrations) are reordered within a prompt. The authors first propose the ICD-OVI score as a metric for measuring the impact of example order in ICL, building upon V-usable information. They then introduce a hierarchical framework based on clustering and inter- and intra-cluster ordering to enable scalable refinement in the example order space. The HIDO clustering approach effectively searches the permutation space, making ICL-DOI a feasible optimization problem.

### Strengths
The lack of a precise, quality-measuring metric for demonstration order has been a major challenge in ICL and many-shot learning research, given the demonstration order instability in LLMs. Many existing works rely on human annotation or heuristic metrics, which are either not scalable or lack accuracy. The authors build on V-usable information to develop a theoretical foundation as an order evaluation metric, effectively quantifying the usable information gain from a specific demonstration order. This metric is not only interpretable (based on information content) but also computationally viable and effective. The authors also propose a model agnostic hierarchical optimization technique to find the optimal demonstrations order based on ICL-DOI metric

### Weaknesses
There are some weaknesses in the authors' proposed methodology for efficiently finding the “optimal” order of demonstrations in ICL.

1. **Assumption of Probing Set Effectiveness**: The authors assume that “the demonstration order optimized for answer prediction also works well for sample generation.” However, figures in the appendix show that demonstration embeddings and probing set embeddings generated by various LLMs lack a clear decreasing trend, which does not support this assumption.

2. The authors limit their many-shot learning setting to 50 examples, which is considerably smaller than certain state-of-the-art many-shot learning setups that require up to 2,000 shots (see source).

3. **Computation-Performance Trade-offs**: It would be beneficial to see an analysis and experiments exploring the relationship between computational cost (by varying the number of clusters or the number of samples per cluster) and model accuracy.

4. The authors claim that “intra-cluster demonstrations share proximate embeddings, which significantly decreases ICL performance variance when demonstration orders vary.” This claim lacks supporting analysis or experimentation. It is not immediately clear that closer embedding proximity (implying more similar samples) necessarily results in reduced variance in demonstration order.

### Questions
* L175: (2) Empirically Effective → (3) Empirically Effective

* L184: Could you clarify “optimal ignorance requirement for a predictive family”?

* L226: $\Pi_1(D)$ and $\Pi_1(D)$ → $\Pi_1(D)$ and $\Pi_2(D)$

* Could you clarify why Theorem 2 can’t be applied to the entire demonstration set and why it is only limited to demonstrations within a cluster?

* what is the sensitivity of HIDO on choice of embedding?

### Soundness
2

### Presentation
2

### Contribution
2
