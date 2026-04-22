# Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Mixture-of-Experts (MoE) enables efficient scaling of large language models (LLMs) with sparsely activated experts during inference.
To effectively deploy large MoE models on memory-constrained devices, many systems introduce expert offloading which caches a subset of experts in fast memory, leaving others on slow memory to run on CPU or load on demand.
While some research has exploited the locality of expert activations, where consecutive tokens activate similar experts, the degree of this **local routing consistency** varies across models and remains understudied.
In this paper, we propose two metrics to measure local routing consistency of MoE models:
(1) **Segment Routing best Performance (SRP)**, which evaluates how well a fixed group of experts can cover the needs of a segment of tokens, and
(2) **Segment Cache best Hit rate (SCH)**, which measures the hit rate of an expert cache utilizing a length of future information under a cache limit.
We analyze 20 MoE LLMs with diverse sizes and architectures and use toy models to verify key factors related to local routing consistency.
We find a strong trade-off between local routing consistency and *local* load balance, while showing that *global* load balance can coexist with local routing consistency.
Meanwhile, settings like shared experts that decrease expert combination space can lead to low local routing consistency.
We further reveal that domain-specialized experts contribute more to routing consistency than vocabulary-specialized ones, and that most models balance between cache effectiveness and efficiency with cache sizes approximately twice the active experts.
These findings pave the way for memory-efficient MoE design and deployment without compromising inference speed.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
MoE architectures have recently enabled scaling of LLMs to even larger sizes without introducing the same amount of computation. To reduce the memory footprint, expert offloading has become a major strategy for inference under memory-constrained environments. However, limited studies have examined the potential cache hit rate of different MoE LLMs in recent progress. The authors performed experiments across a wide range of MoE LLMs and datasets and identified that routing consistency can vary across models. Based on this observation, the authors proposed several metrics to estimate the capability of a particular MoE model to perform expert offloading.

### Strengths
- The idea is intuitive and based on an interesting observation that the activation consistency could change due to MoE architectures. Early studies in this field were unable to cover the later architectural changes in MoEs, so the proposed study addresses an area that has not been well studied.

- The selected models and datasets are comprehensive, covering both classic and recent variants of MoE LLMs (within hardware constraints). Based on these empirical results, the authors have provided some great observations.

### Weaknesses
- The abstract is poorly written. It does not clearly mention the different MoE architectures (shared experts, small/large experts) as the potential reasons for the local consistency discrepancy, which I believe is the most important contribution of this paper. 

- While the authors have shown that the different MoE models could have vastly different expert activation consistency, the authors did not perform an end-to-end empirical evaluation on the expert offloading performance with the proposed strategy. As the MoE model architectures themselves could be vastly different, whether a certain model suit expert offloading would not only depends on the cache hit rate.

- I personally believe that it would be better to describe the concepts of SRP and SCH using visualizations. While the listed formulas are helpful, they do not effectively convey the core idea. For SCH in particular, since it describes the cache hit rate, the authors should consider including a figure to illustrate the concept, as is common in other papers that discuss caching.

- Figure 2-6 are way too small. It's really difficult for me to identify the relevant colors even with a large screen. These figures give me the sense that they were copy pasted from some research reporting slides, instead of being curated for this manuscript.
The author should consider ways to improve the rendering of these figures, especially for figure 2 and 3, given that these figure consist of the most important observation of this paper.

- Conclusion is badly written, as it does not make any suggestion on the MoE architecture as promised by the abstract.

- Minor
  - L150 -- single expert is a weird paragraph title here. "Single expert case" might be better as it's mostly related to SRP, instead of any particular expert. Similar comment could be made to line 180 and 256 ("Dataset").

### Questions
I would like to see some of the discussion regarding the actual performance of offloading that takes the architectural differences into the consideration.

It would also be interesting to see why particular training strategy or model architecture design leads to the split between the domain specialization / vocabulary specialization in the MoE models.

### Soundness
3

### Presentation
1

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
This paper investigates a key property of Mixture-of-Experts (MoE) models, termed "local routing consistency", which determines a model's suitability for expert offloading systems. To quantify this property, the paper proposes two novel metrics: Segment Routing Best Performance (SRP) and Segment Cache Best Hit Rate (SCH). Through an extensive empirical study on 20 MoE large language models with diverse architectures, the authors obtain several findings, for example, models applying MoE on every layer and without shared experts exhibit the highest routing consistency.

### Strengths
This paper offers a novel analytical perspective on the important and practical problem of efficient MoE model deployment. The concept of "local routing consistency" is insightful. It provides a clear and quantifiable framework for evaluating and comparing the deployment potential of different MoE models in resource-constrained environments. 

The experimental evaluation is thorough and comprehensive, standing out as a primary strength of this work. The authors analyze up to 20 representative MoE models, covering a wide range of parameter scales and architectural variants, which lends strong credibility and generalizability to their conclusions.

### Weaknesses
Although the analysis is insightful, its conclusions are primarily based on correlation rather than causation. For example, the study observes that architectures with "MoE on every layer" and "no shared experts" correlate with high routing consistency and conjectures that dense modules might "interfere with or weaken routing signals". Ablation studies, such as modifying these architectural features on the same backbone (even a small one is ok) and observing the change in consistency, would greatly strengthen the reliability of these claims and uncover the underlying mechanisms.

The work did not explore the potential trade-off between this routing consistency and the model's core performance (e.g., accuracy). It is plausible that forcing or guiding a model to produce highly consistent routing could limit its expressive power, making it harder to capture fine-grained, token-level semantic shifts and thus hurting its prediction accuracy. The paper lacks a discussion on this "efficiency-accuracy" trade-off, which makes it difficult for readers to assess whether the design choices made to pursue high SRP/SCH might come at an unacceptable performance cost.

The paper's definition of "local routing consistency" primarily focuses on the processing of contiguous text segments, which corresponds mainly to the autoregressive decoding phase. However, in practical applications, the prefilling phase for long prompts is also a significant performance bottleneck. The routing patterns during prefilling may differ from those during decoding, but the current analytical framework does not explicitly distinguish between or investigate consistency in these two modes. 

The paper's core claim is to guide memory-efficient model design and deployment, yet this claim is not well demonstrated. The study focuses primarily on analyzing proxy metrics like cache hit rates but does not provide implementation of a real model or a real expert offloading system.

### Questions
1. Could you design an experiment to more directly verify that dense components indeed "interfere" with sparse routing, thereby decreasing consistency?

2. Is there an inherent trade-off between maximizing local routing consistency (for inference efficiency) and maintaining the model's predictive performance (e.g., accuracy)? Could designs that pursue high SRP potentially limit the model's expressive capacity?

3. Does your analysis and conclusions apply equally to the prefill stage for long context inputs? 

4. How well do the findings in this paper guide model or system design? Specifically, is there any action item (with validated performance) for the MoE model/system developers to optimize their design?

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
4

### Summary
Mixture-of-expert (MoE) models typically have high parameter count and not all parameters are always present on the GPUs. Efficiency and throughput of the model will be higher if active experts are present in the GPUs and there is minimal computation on CPU / communication between CPU and GPU. This work proposes two metrics to analyze how correlated expert activations are within a segment of tokens (termed local routing consistency). The authors empirically analyze various MoE models and suggest architecture design choices that might impact the local consistency. They also propose a way to empirically select the ideal cache size for a given model.

### Strengths
The authors tackle a specific and important problem in LLM deployment. The work proposes two metrics (SRP and SCH) to analyze the local routing consistency. The empirical analysis is quite thorough with 20 different MoE models and several datasets from both general training corpora and downstream tasks. The paper considers several reasons that could be responsible for the consistency - model architecture, expert specialization to specific domains or vocabulary subsets and load balancing. The findings are interesting - shared experts have lower local consistency (SRP), load-balanced models can have high SRP and cache size of $2\times$ number of experts results in a high SCH for most models.

### Weaknesses
1. There is no explicit connection of proposed metrics to throughput. For instance, how would SRP/SCH affect throughput given cache size, communication time and LLM forward propagation time? There are no measurements of throughput in any of the experiments either. 
2. What advantage does SCH have compared to common cache algorithm hit rate? Can we not determine required cache size by analyzing LRU hit rate vs $\rho$? 
3. In analysis in Section 3.3, the authors suggest that applying MoE on every layer and not sharing experts results in higher routing consistency. While there might be a correlation between the two, it does not imply causation. Particularly in the case of ‘MoE on every layer’, there are just 4 out of 20 models that apply at every ‘k’ layer instead and of those four, two of them have extremely high ratio of active to total experts (1:64 and 1:128). It is difficult to come to such conclusions without decoupling the factors and with such limited data. The conjecture on why these might cause inconsistency (L:334-336) is also unclear. 
4. Writing needs significant improvement: \
i. The proposed metrics in Section 2 require a lot of time to understand. Providing intuitive / textual explanations for the equations, removing repeated mentioning of variables (for e.g., saying expert $e \in E$ at all locations instead of just saying expert) and moving unnecessary equations to appendix and replacing them with text in the main section would vastly improve the reader’s experience. \
ii. The term $\rho$ is overloaded - it is used in both SRP for segment routing size ratio and in SCH for segment cache size ratio - causing a lot of confusion. \
iii. Some of the figures (e.g. Figures 2 and 4) are hard to parse. 
5. Since the proposed metric requires access to expert activation, it is not possible to measure them for black box models.

### Questions
1. Questions are primarily based on the weaknesses above. Provide answers to weaknesses (1) and (2). If possible provide a function connecting the proposed metrics to throughput / inference time and empirical results for the same.
2. Clearly explain why the dense modules might decrease local consistency (SRP).
3. Is it useful to have the marker size representing model size in Figure 2? Similarly, can expert specialization vs SRP be used in the plot in Figure 4 instead of correlation? Why do LL2 and Y2 have a very low correlation while they have high SRP and relatively high expert domain specialization? Why do STe have high correlation but low domain specialization?

### Soundness
2

### Presentation
2

### Contribution
3
