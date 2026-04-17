# Distilling the Thought, Watermarking the Answer: A Principle Semantic Guided Watermark for Reasoning Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 2

## Abstract
Reasoning Large Language Models (RLLMs) excelling in complex tasks present unique challenges for digital watermarking, as existing methods often disrupt logical coherence or incur high computational costs. Token-based watermarking techniques  can corrupt the reasoning flow by applying pseudo-random biases, while semantic-aware approaches improve quality but introduce significant latency or require auxiliary models. This paper introduces ReasonMark, a novel watermarking framework specifically designed for reasoning-intensive LLMs. Our approach decouples generation into an undisturbed Thinking Phase and a watermarked Answering Phase. We propose a Criticality Score to identify semantically pivotal tokens from the reasoning trace, which are distilled into a Principal Semantic Vector (PSV). The PSV then guides a semantically-adaptive mechanism that modulates watermark strength based on token-PSV alignment, ensuring robustness without compromising logical integrity. Extensive experiments show ReasonMark surpasses state-of-the-art methods by reducing text Perplexity by 0.35, increasing translation BLEU score by 0.164, and raising mathematical accuracy by 0.67 points. These advancements are achieved alongside a 0.34% higher watermark detection AUC and stronger robustness to attacks, all with a negligible increase in latency. This work enables the traceable and trustworthy deployment of reasoning LLMs in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a watermarking technique designed for reasoning-intensive large language models (LLMs). The proposed method identifies critical tokens from the reasoning phase and assigns additional weighting to tokens in the green list based on their semantic relevance to these critical tokens. The detection process follows the standard green–red list framework used in prior work. Experimental results demonstrate that the proposed method achieves improved text quality and higher detection accuracy compared to existing approaches.

### Strengths
+ The topic is interesting and important.
+ The evaluation results demonstrate its effectiveness.

### Weaknesses
- The threat model could be clarified and illustrated in more detail. It is unclear whether users or adversaries are assumed to have access to the reasoning procedure and whether the reasoning-phase outputs themselves should also be watermarked.
- The system design appears largely empirical and lacks theoretical grounding. The global causal contribution (GCC) and competitive persistence scoring (CPS) are introduced as empirical measures, but the rationale behind their formulation and their effectiveness in preserving semantic integrity are not fully analyzed. Similarly, the motivation for introducing the principal semantic vector (PSV) and the reasoning for aligning selected tokens with their PSV should be more thoroughly explained.
- The evaluation does not adequately assess reasoning integrity. While the results include standard text-quality metrics such as PPL, BLEU, and mACC, they do not measure whether the generated text faithfully reflects the underlying reasoning process. Including an evaluation of semantic consistency between reasoning steps and final outputs would strengthen the work.

### Questions
+ How many tokens are used for detecting watermarks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes ReasonMark, a semantically guided watermarking framework for Reasoning Large Language Models. The authors argue that existing text watermarking methods are primarily designed for general text generation tasks, and directly applying them to reasoning LLMs may disrupt their internal chain of thought. Therefore, this paper divides the model generation process into two stages: "Thinking" and "Answering," embedding the watermark only in the Answering stage. Specifically, the authors first extract critical tokens (CTs) from the Thinking stage, obtain the principal semantic vector (PSV) through PCA, and then adaptively adjust the watermark bias strength based on the semantic similarity between the token and the PSV in the Answering stage.

### Strengths
This paper works on the watermarking for large language models, which is significant for determining provenance and responsibility for generated content.

### Weaknesses
1. The paper does not adequately discuss why existing "distortion-free watermarks" such as [1] and [2] cannot be applied to the reasoning LLM scenario, resulting in insufficient motivation to propose a new framework. Besides, although the unbiased sampling technique [3] that the was cited in the paper was claimed inefficient by the authors, there several other unbiased watermarking papers [4] [5] which do not require access to the original LLMs and is efficient.

2. ReasonMark only uses the first principal component (PCA1) of the key token embedding as the main semantic direction, but does not explain why other principal components (such as PCA2) are ignored. If the CT embedding has high noise or contains multiple semantic clusters, PCA1 may only reflect the direction of maximum variance rather than the true semantic main line of reasoning, which may weaken the semantic guidance role of PSV.

3. According to equations (5) and (6), both GCC and CPS involve pairwise comparisons of probability distributions at each step. For large vocabularies and long inference chains, this process is extremely time-consuming in practical deployments. While the paper reports an overall latency increase of approximately 8%, it does not explicitly state whether this includes CT construction and PCA computation time, thus the performance evaluation may be incomplete.

4. As seen in Figure 2, the improvements in ReasonMark's AUC and TPR@1%FPR are small (only about 1–2%). Without statistical significance analysis, such improvements may not be sufficient to demonstrate the method's effectiveness.

5. the parameter $\delta_{i, w}$ in equation (11) is crucial for performance, but the paper only provides the ranges of $\delta_0 \in [1, 2]$ and $\delta_\lambda \in [1, 5]$, without specifying their final set values ​​or their comparability with baselines such as KGW, making the interpretation of the results less transparent.


[1] Undetectable Watermarks for Language Models, Miranda Christ et al., COLT 2024

[2] Scalable watermarking for identifying large language model output, Sumanth Dathathri et al., Nature 2024

[3] Unbiased watermark for large language models, Zhengmian Hu et al., ICLR 2024

[4] A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models, Yihan Wu et al., ICML 2024

[5] StealthInk: A Multi-bit and Stealthy Watermark for Large Language Models, Ya Jiang et al., ICML 2025

### Questions
Can current distortion-free or unbiased watermarking be applied for reasoning LLM scenario? If not, could you explain why?

### Soundness
1

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
3

### Summary
This paper proposes ReasonMark, a watermarking framework tailored for reasoning-intensive LLMs. It splits generation into an untouched thinking phase and a semantically-guided answering phase, and uses a Criticality Score to select pivotal tokens from the reasoning trace and distills them into a Principal Semantic Vector (PSV) that adaptively modulates watermark strength. This preserves logical coherence while embedding a strong, detectable signal.

### Strengths
1. The paper is well written.
2. The proposed method is well designed with reasonable motivations.
3. The paper presents several nice output examples to help readers get a better sense of the output picture.

### Weaknesses
1. No analysis is performed to verify whether the proposed watermarking method affects the generation quality.
2. I suggest that the authors add a limitation part to the paper to clarify some possible improvements.
3. The method is designed specifically for reasoning models. Does this mean that it will not work for a non-thinking model? What if the user turns off the thinking mode? In such a case, how do you select the critical tokens?

### Questions
No, see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces ReasonMark, a semantic-guided watermarking framework tailored for reasoning-intensive language models. By separating the generation into a reasoning (thinking) phase and a watermarked answering phase, the method preserves logical coherence while embedding a robust watermark. A Principal Semantic Vector (PSV), distilled from key reasoning tokens, adaptively guides watermark strength during output generation. Experiments show that ReasonMark outperforms prior methods in text quality, task accuracy, and watermark robustness, with minimal latency overhead.

### Strengths
1. The paper addresses a timely problem: watermarking for reasoning-intensive language models, where tranditional methods often fail to preserve logical coherence.

2. The proposed method is conceptually clear, introducing a separation between thinking and answering phases, guided by a semantically meaningful representation.

3. Experimental results demonstrates improvements in watermark detection and task performance.

### Weaknesses
1. The main concern is that the performance gains appear marginal: improvements of 0.67 in accuracy on reasoning tasks and 0.34% in AUC are relatively small and may not justify the complexity of the method.

2. The distinction between this approach and prior semantic-aware watermarking methods is not clearly articulated. Although the paper critiques existing methods for latency and reliance on auxiliary models, ReasonMark appears to share similar characteristics, and Table 4 shows comparable latency across methods.

3. The presented theorem (2.2) lacks a formal proof and does not seem to provide theoretical guidance for the algorithm’s design.

### Questions
No

### Soundness
3

### Presentation
2

### Contribution
2
