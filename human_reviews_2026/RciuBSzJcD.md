# Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Visual retrieval-augmented generation (VRAG) augments vision–language models (VLMs) with external visual knowledge to ground reasoning and reduce hallucinations. Yet current VRAG systems often fail to reliably perceive and integrate evidence across multiple images, leading to weak grounding and erroneous conclusions. In this paper, we propose EVisRAG, an end-to-end framework that learns to reason with evidence-guided multi-image to address this issue. The model first observes retrieved images and records per-image evidence, then derives the final answer from the aggregated evidence. To train EVisRAG effectively, we introduce Reward-Scoped Group Relative Policy Optimization (RS-GRPO), which binds fine-grained rewards to scope-specific tokens to jointly optimize visual perception and reasoning abilities of VLMs. Experimental results on multiple visual question answering benchmarks demonstrate that EVisRAG delivers substantial end-to-end gains over backbone VLM with 27\% improvements on average. Further analysis shows that, powered by RS-GRPO, EVisRAG improves answer accuracy by precisely perceiving and localizing question-relevant evidence across multiple images and deriving the final answer from that evidence, much like a real detective.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies the limitation of poor cross-image evidence integration and weak grounding problems in current Visual Retrieval-Augmented Generation (VRAG) systems, and addresses this limitation by proposing EVisRAG, an end-to-end framework for multi-image evidence-guided reasoning. EVisRAG introduces explicit steps: (1) observe retrieved images, (2) record per-image evidence, (3) reason over aggregated evidence to generate answers. To optimize this framework, the authors present Reward-Scoped Group Relative Policy Optimization (RS-GRPO), which applies fine-grained rewards (perception, derivation, format) to task-specific token scopes to sharpen credit assignment. Empirical results across show EVisRAG outperforms baselines with impressive improvement on accuracy and F1 score.

### Strengths
1. EVisRAG’s step-by-step evidence recording solves the problem of implicit, ungrounded reasoning in VRAG systems.
2. Reward scoping addresses the blurring of credit assignment in mixed-reward training, leading to stable and superior performance.
3. Works across single/multi-hop tasks and diverse document types (charts, slides, docs), with prompt-based gains even for untrained models.

### Weaknesses
1. Efficiency: No detailed analysis of output token counts in the experiments. Explicit evidence steps may increase output length, but this trade-off is not quantified. Although the inference latency is provided in Sec. A.9, output token counts s a more appropriate metric for measuring inference costs than latency.
2. Failure Cases Exploration: Limited discussion of failure cases where EVisRAG misinterprets evidence or fails to handle the retrievals. How does EVisRAG perform with these situations?

### Questions
See Weaknesses.

### Soundness
3

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
This paper presents a new visual retrieval-augmented generation framework, which introduces observation and evidence before think. And it introduces RS-GRPO to train the model. The approach demonstrates significant performance improvements across multiple VQA benchmarks and is well-structured in presentation. However, several concerns remain regarding its novelty and experimental design.

### Strengths
It achieves significant performance improvement.

The paper is clearly written and well-organized.

### Weaknesses
1. Although effectivenss, the motivation of introducing observation and evidence is unclear. The motivation of RS-GRPO is clear.
2. The introduction of observation and evidence appears to be an incremental improvement and not essential enough.
3. The selection of VRAG baselines is limited, and methods such as VisRAG are omitted. Moreover, the results in Table 2 show that a simple think-then-approach (without perception) already outperforms all baselines in Table 1.
4. The process for constructing SFT data with the newly introduced observation and evidence fields is not discussed.
6. Eq. 3-6 describe standard autoregressive generation processes and do not contribute meaningfully to the methodology. The rollout process is straightforward and does not require detailed exposition.

### Questions
See weaknesses.

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
4

### Summary
The authors present a single model, EVisRAG, that does a visual-RAG-like operations in the chain-of-thought to yield better results compared to the visual reasoning models. Specifically, when given a question and a set of images, the model will first go through all i mages, determine if they're relevant or not, do reasoning, and answer the question eventually. The authors proposed RS-GRPO, a variant of GRPO, to provide rewards at different part (such as one reward for reasoning, another for relevance prediction) during the RL training. The proposed pipeline improves the performance compared to the baseline on multiple benchmarks.

### Strengths
1. The paper is clear and easy to follow.
2. The proposed method is effective on multiple benchmarks and the analyses in "insufficient to answer" is interesting.

### Weaknesses
- While the paper is technically sound, the gain of the perception reward and the RS-GRPO seems to be small in both in-distrubiton and OOD settings (Table 2). The difference of attention maps in Figure 3 seems to be small as well. As the RL part is the key technical contribution of the paper, it would be great if the author can provide the error bar (bootstrapped average with confidence interval) to show that the proposed method is actually useful or not. Also, it would be great if we can try natural images (beyond documents only) with large-scale settings [1, 2].

- The reviewer believes that the system is slightly different from the actual two-stage RAG setting. What's the advantage of training one large model as a whole compared to having one retriever and one generator. The most efficient setting would be a light-weight retriever to get top-K images, let's say CLIP or other post-trained variants [1]. In a real-world setting with large-scale documents, going through each image one by one in a reasoning model can be time-consuming and easily runs out of the context window. Another baseline that worth comparing would be calling a VLM twice with the first time doing retrieval and the second one doing generation based on retrieved images.


[1] Wu, Tsung-Han, et al. "Visual haystacks: A vision-centric needle-in-a-haystack benchmark." ICLR 2025.

[2] Chen, Jun, et al. "Document haystacks: Vision-language reasoning over piles of 1000+ documents." CVPR 2025.

### Questions
please read the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
