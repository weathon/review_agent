# Combination Generalization of Capability-Specific Neurons in LLMs

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Although Large Language Models (LLMs) exhibit various exciting capabilities, understanding the mechanisms behind these abilities remains a challenging problem. In this paper, we aim to understand these mechanisms from the perspective of neurons. Specifically, we first propose a Detecting Capability-Specific Neurons (DCSN) method. Extensive enhancement and erasure experiments demonstrate that the detected neurons are highly correlated with specific capabilities, exhibiting strong cohesion and separability, which we define as capability-specific neurons. Moreover, leveraging these neurons, we conducted compositional experiments and, for the first time, discovered that capability neurons exhibit compositional generalization. Inspired by these findings, we propose a Capability Neuron-Level Fine-tuning method (CNLF) that fine-tunes specific capability neurons to achieve performance improvements across datasets and tasks. Extensive experiments validate the effectiveness of this method and provide a low-cost, highly generalizable fine-tuning paradigm. Our research offers interpretable insights into the capability mechanisms of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a method for detecting capability-specific neurons in LLMs and claims to discover that these neurons exhibit compositional generalization. The authors construct a dataset based on arithmetic operators (+, -, *, /), propose a Detecting Capability-Specific Neurons (DCSN) method, and introduce a Capability Neuron-Level Fine-tuning (CNLF) approach.

### Strengths
The paper is well-organized with a logical flow from neuron detection to compositional generalization to fine-tuning applications.

The introduction of cohesion and separation metrics provides a principled way to evaluate neuron localization quality, which is valuable for comparing different approaches.

The observation that capability-specific neurons exhibit compositional generalization (e.g., addition and multiplication neurons both activate for "1+3*5") is noteworthy.

### Weaknesses
There is absolutely no technical novelty in the way neurons are selected. NeFT (COLING 2025) and Robustness-Preserving Fine-tuning (ECCV 2024) show that the finetuning method is also not a novel parameter efficient finetuning method

The domain in which the results are evaluated is very narrow (four arithmetic operators), far below ICLR standard

The mechanistic interpretability literature is crowded with similar work:
Knowledge neurons (Dai et al., 2021)
Task neurons (Leng & Xiong, 2025)
Causal tracing (Meng et al., 2022)

### Questions
No additional questions

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel framework for interpreting Large Language Models (LLMs) by localizing capability-specific neurons in Feed-Forward Networks, which exhibit strong cohesion and separability. The authors propose the Detecting Capability-Specific Neurons (DCSN) method. validated via enhancement and erasure experiments on models like LLaMA-2-7B. Key contributions include: (1) first demonstration of compositional generalization in these neurons during multi-operator tasks,  (2) Capability Neuron-Level Fine-tuning (CNLF) for efficient, controllable multi-capability boosts, and (3) interpretable insights into LLM invocation mechanisms, advancing beyond overlapping knowledge/task neurons

### Strengths
1. Proposes a simple yet effective detection method (DCSN) with interpretable metrics (cohesion & separability).
2. Demonstrates strong empirical evidence for compositional generalization at the neuron level.
3. Introduces efficient neuron-level fine-tuning (CNLF) that improves capability performance with minimal parameters.
4. Clearly defines and isolates capability-specific neurons, moving beyond prior “knowledge/task neuron” work.

### Weaknesses
1. All experiments are conducted on relatively small and mid-sized models (up to 13B parameters), so scalability to larger LLMs remains untested.
2. The detection threshold for neuron identification (σ = 6) is arbitrarily chosen, without robustness or sensitivity analysis.
3. Downstream evaluation tasks are few and domain-specific, providing weak external validation beyond arithmetic reasoning.
4. The assumption that capability-specific neurons are fully separable is oversimplified, as inter-capability overlap is observed but not deeply analyzed.
5. The work lacks theoretical grounding to explain why such neurons emerge or how they mechanistically interact to yield compositionally.

### Questions
1. How well do the DCSN neuron identification results generalize to more complex or fuzzy capabilities beyond arithmetic, such as reasoning, summarization, or memorization?
2. are there any mechanistic insight into why capability-specific neurons emerge and how they interact across layers to support compositional generalization?
3. Could other attribution methods, such as Integrated Gradients or other feature attribution techniques, be used for detecting capability-specific neurons, and how might their results compare to DCSN?.
4. I suggest adding the following recent works to the Related Works section for better context: [1, 2, 3]. These works focus on neuron-level interventions in LLMs. detecting and pruning neurons that harm generalization or cause copying bias, and are directly relevant to this paper’s theme.



[1] Detecting and Pruning Prominent but Detrimental Neurons in Large Language Models.

[2] Mitigating Copy Bias in In-Context Learning through Neuron Pruning.

[3] Finding Safety Neurons in Large Language Models.

### Soundness
4

### Presentation
3

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
This paper aims to understand the effects of neuron composition within large language models (LLM) by identifying neurons linked to specific skills, exploring their compositional generalization, and demonstrating the utility of sparse fine-tuning. The authors first proposed a method called DCSN to detect capability-specific neurons. They curated datasets targeting distinct task capabilities and employed statistical analyses to isolate neurons whose activations are significantly higher for those capabilities. For example, separate neuron groups were found to correspond to addition and multiplication in mathematical reasoning. They further validated the compositional effects of these capacity-specific neurons, confirming that the identified units indeed encode skill-specific functions (Cohesiveness, Separability metrics). Finally, to showcase the practical implications, the authors applied their method to locate neurons associated with mathematics, programming, and other language tasks. Leveraging these neurons for sparse fine-tuning (i.e., CNLF method in this paper) led to superior downstream performance compared to both full-parameter and LoRA fine-tuning, despite involving far fewer trainable parameters.

### Strengths
1. The concept of compositional generalization of capability neurons is inspiring. Although previous studies have explored knowledge neurons [1] and capability neurons [2], the combinatorial effects between them, and their implications for model fine-tuning have rarely been investigated. This work provides valuable insights into understanding the inner workings of LLMs at the neuron level (e.g., the mutual interactions between neurons responsible for addition and subtraction).
2. The proposed metrics of cohesiveness and separability offer effective tools for assessing the accuracy and reliability of neuron localization.
3. The demonstrated effectiveness of the CNLF fine-tuning method is impressive, as it enables model training under resource-constrained conditions. The experiments show that fine-tuning only a small subset of neurons can achieve comparable or even superior performance to full-model fine-tuning on tasks such as mathematics.
4. The paper is well written and structured, forming a coherent narrative that progresses from neuron localization to understanding compositional effects, and finally to neuron-level fine-tuning.

### Weaknesses
1. Most of the experiments related to mathematical capabilities focus on well-defined basic arithmetic operations. However, the concept of capability neurons may extend far beyond **elementary arithmetic**. Even within mathematics, could “mathematical capability” be decomposed into more **complex sub-capabilities**, such as logical reasoning or symbolic manipulation? It may be worthwhile to further analyze neurons corresponding to these higher-level skills.
2. Related to the first point, the evaluation datasets could be more challenging. GSM8K is relatively basic. Would the localization and fine-tuning of neurons responsible for arithmetic operations (`+, -, *, /`) yield consistent results and efficiency when applied to more difficult benchmarks such as AIME24 or AIME25?
3. The definition of Eq. 3 requires refinement. It currently describes the intersection of sets rather than a quantifiable metric (e.g., a ratio or percentage). However, the subsequent discussion of cohesiveness and separability suggests that these should indeed be ratio-based measures.
4. The main text introduces a hyperparameter $\alpha$, while Appendix D.2 presents an ablation study involving a hyperparameter $\beta$. However, $\beta$ is never defined or mentioned in the main text. It seems likely that $\beta$ corresponds to $\alpha$ in Eq. 2, but this should be clarified explicitly.

### Questions
1. How these findings might generalize to other architectures, particularly Mixture-of-Experts [3] models. For example, would specific capabilities tend to cluster within certain experts?
2. This study primarily focuses on neurons within the FFNs. Have you considered the role of attention heads [4] in realizing these capabilities, especially in combining different ones? Is it possible that certain capabilities emerge from the synergistic interaction between FFNs and the attention mechanism?

**Reference**

[1] Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). Knowledge Neurons in Pretrained Transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 8493-8502).

[2] Song, R., He, S., Jiang, S., Xian, Y., Gao, S., Liu, K., & Yu, Z. (2024). Does Large Language Model Contain Task-Specific Neurons?. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 7101-7113).

[3] Cai, W., Jiang, J., Wang, F., Tang, J., Kim, S., & Huang, J. (2025). A survey on mixture of experts in large language models. IEEE Transactions on Knowledge and Data Engineering.

[4] Yin, K., & Steinhardt, J. Which Attention Heads Matter for In-Context Learning?. In Forty-second International Conference on Machine Learning.

### Soundness
2

### Presentation
2

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
The paper presents an attempt to identify “capability” specific neurons through correlation of individual neuronal activations and the model’s output tokens. The authors identify about 5% of the top contributing neurons in the FFN layer of the transformer layers for each arithmetic operation. The paper conducts several evaluations to establish the benefits of identifying capability-specific neurons including compositional behavior, erasure, and fine-tuning methods. 

Overall, while the paper provides some compelling results, the draft leaves out various questions related to choices made in the work as well as the broader implications of the work.

### Strengths
Approach to ranking neurons is simple, although the authors do not provide any sensitivity to the number of samples needed to identify the neurons. 

Performance on enhancement, separability and cohesion evaluations show validity of the some of claims made in the work. 

Extension to non-mathematical tasks are interesting, although this evaluation is quite limited.

### Weaknesses
I. The authors primarily considered FFN layers of the transformer layers. While this may have allowed operation simplicity, it is unclear why the self-attention neurons are ignored in the current analysis. The authors do not provide a strong justification for this choice. 

II.  The large part of the work dwells significantly on the arithmetic operations of “+”, “-”, “*” and “/”. It is only in the very last part of the results that the authors provide experimental evidence for other non-arithmetic tasks. Several questions pertain to these experiments. Are the neurons identified similar to the arithmetic operations. 

III. Tasks like emotion recognition and coding involve multiple atomic tasks like, semantic parsing, logical understanding, contextual modeling and function reasoning. It is not clear how these are modeled in Section 6.4 (Table 6). The discussion on this part is rather slight, leaving much of the details to the imagination of the reader.

IV. Various modes of introducing arithmetic operations are discussed in Table 1. It is not clear how these modes of introduction change the neuronal activity pattern, for example, if DG is used for identifying the neuronal patterns, does it also generalize to other types of invoking arithmetic operations.

V. Several choices are made throughout the work, without a strong justification. For example, choices like capability specific neurons of 0.05% are used, and in the experiments with erasure 10% of capability neurons are disabled. Secondly, in some of the Tables which report experiments, different models are used for enhancement and erasure experiments (like Table 2). Why do we not have the same model used for both enhancements and erasures. These choices, without a clear articulation, raises significant questions about the generalization of the behaviors reported in the work. 

VI. The paper suffers from a lack of novelty. Identifying neural activity using correlation patterns is the only choice considered, while prior works (like Leng et al [2025]) have advocated the use of gradient based relevance scores, while knowledge editing efforts use different similarity metrics and measures. 

VII. Further, the authors have focused on individual neurons versus layers or chains of neurons. The authors do not provide any experimental comparison with other prior works which have proposed layers of connected neurons or those with other measures. This makes the experimental comparisons very shallow and do not provide sound justification for the significance of this work.

### Questions
As noted in the weakness, several aspects can be elaborated and explored to strengthen the work.
Providing more comparative experimental evidences with other works that have proposed identifying task specific neurons and layers. 

Justifying many of the choices made the work, including the motivation for the correlation based metrics used.

Discussing various experiments in detail to highlight the aspects to related applications in non-arithmetic tasks.

Measuring sensitivity and robustness of the neuron identification task when the number of samples are limited. 

Improving the limitations sections - it is currently in Appendix with the only mention of computational choices made in terms of the LLMs considered in the work.

### Soundness
3

### Presentation
2

### Contribution
3
