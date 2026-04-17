# SPIDER: Multi-Layer Semantic Token Pruning and Adaptive Sub-Layer Skipping in Multimodal Large Language Models

- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Multimodal Large Language Models (MLLMs) face significant efficiency challenges that stem from two distinct yet coupled sources: data redundancy and computational redundancy. While most methods focus on data redundancy by pruning visual tokens from the output of the visual encoder or computing redundancy in LLM decoders using blockwise importance, the finer-grained inter-layer representation shifts and the distribution differences within the layers themselves have not been fully explored. In this work, we comprehensively investigate this dual-level inefficiency. We posit that intermediate layer tokens from vision encoders should be considered for effective visual token pruning, as semantic focus shifts across layers, with middle-layer tokens capturing more detailed object-centric information that deeper layers may abstract away. Furthermore, we reveal the differential contributions of Attention and FFNs across distinct LLM decoder layers. Building upon these discoveries, we propose \textbf{SPIDER}, a training-free framework that integrates multi-layer \underline{\textbf{S}}emantic visual token \underline{\textbf{P}}run\underline{\textbf{I}}ng with an a\underline{\textbf{D}}aptive sub-lay\underline{\textbf{ER}} skipping mechanism. Experimental evaluations demonstrate that SPIDER consistently maintains strong performance across various MLLM architectures and reduction ratios. Notably, SPIDER achieves a reduction to $20\%$ in FLOPs for LLaVA-Next-7B, while preserving $98.7\%$ performance to the baseline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SPIDER, a training-free framework for VLM token pruning. It introduces two complementary components: (1) Multi-layer Semantic Visual Token Pruning (MSV-Prune) : selects tokens using semantic features from both middle and deep layers of the vision encoder to preserve fine-grained object information. (2) Adaptive Sub-layer Skipping (ASL-Skip): dynamically skips attention or FFN sub-layers based on per-token entropy and image–text correlation, fused with precomputed sub-layer contribution scores. From result, the proposed methods performance well on most of benchmark with 80% reduction.

### Strengths
This paper provide a insight view of token pruning, as attention indeed shift between different layer, so maybe according to situation of each layer to choose token should be a promising idea. 

This work provide comprehensive experiment on various benchmarks as well as hyperparameter anlyses.

### Weaknesses
1.The paper's writing needs improvement; the content too dense. The authors could perhaps provide a clearer description of the different components, such as following the data pipeline.

2.The method presented here is largely experimental, with a weak theoretical foundation.

3.If possible, author could experiment with more backbones to see how the proposed method generalizes.

### Questions
1. How stable are SLC scores across different datasets or tasks? Could a dynamic adjustment mechanism further improve performance? like classification task maybe much easier than qa.

2. Prior work has shown that token pruning methods may raise knowledge boundary drift. For example, instances that were previously answered correctly may become incorrect after pruning, and vice versa. Such inconsistencies could be problematic in applications where reliable responses to critical queries are essential.
Would it be possible for the authors to conduct an instance-level stability analysis to examine whether the proposed method causes any notable shifts in per-sample predictions or reasoning consistency?

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
3

### Summary
This paper addresses computational inefficiencies in Multimodal Large Language Models (MLLMs), focusing on data redundancy in vision encoders and computational redundancy in LLM decoders.

### Strengths
SPIDER does not require additional fine-tuning, making it practically attractive for deployment on pre-trained models.

### Weaknesses
1. While the proposed method demonstrates strong results on LLaVA-based models, the evaluation lacks coverage of a broader range of MLLMs, such as Qwen-VL. This raises questions about the generalizability of SPIDER across different model architectures.

2. The comparative experiments focus solely on training-free baselines. While this is aligned with SPIDER's training-free design, it remains unclear how the method would perform relative to training-based approaches.

3. Although the training-free nature of SPIDER is advantageous for deployment efficiency, it is worth investigating whether the method could be extended into a training-aware framework, potentially allowing learnable pruning or adaptive skipping to further enhance performance.

### Questions
1. Although the training-free nature of SPIDER is advantageous for deployment efficiency, it is worth investigating whether the method could be extended into a training-aware framework, potentially allowing learnable pruning or adaptive skipping to further enhance performance.

2. Have the authors considered evaluating SPIDER on text-dense VQA tasks,  such as DocVQA, or ChartQA? Given that such tasks often rely on fine-grained textual and spatial cues, it would be interesting to see whether the proposed pruning and sub-layer skipping mechanisms retain effectiveness in these more information-dense settings.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SPIDER, a multi-layer semantic pruning framework designed to compress large transformer-based models while maintaining semantic fidelity. Instead of relying on traditional local metrics, SPIDER evaluates semantic contribution across layers using latent-space similarity and hierarchical optimization.

### Strengths
[1] The semantic-level pruning criterion is conceptually novel. Instead of focusing on local magnitude or gradient, it prunes neurons based on their semantic alignment contribution.      
[2] Solid empirical results across multiple datasets and model scales. Ablation studies demonstrate that semantic objectives significantly outperform magnitude-based and movement pruning.   
[3] The paper is well written and easy to follow.

### Weaknesses
[1] The theoretical analysis assumes layer-wise independence, treating semantic discrepancy as decomposable across layers. However, some works also show that transformers exhibit strong inter-layer coupling [a].       
[2] Computing multi-layer semantic similarity via latent embeddings requires multiple forward passes per layer, increasing pre-pruning cost.       
[3] The paper does not examine how pruning affects layers with high semantic dependency.   



[a] Merullo, Jack, Carsten Eickhoff, and Ellie Pavlick. "Talking heads: Understanding inter-layer communication in transformer language models." Advances in Neural Information Processing Systems 37 (2024): 61372-61418.

### Questions
[1] Why was cosine similarity chosen for semantic alignment?   How about MSE?  Have you evaluated alternatives such as KL divergence or mutual information?     
[2]  Can semantic similarity estimation be approximated using smaller calibration subsets?     
[3] It is better to add more layer-specific analysis, such as (a) identifying which layers are most sensitive to pruning and whether SPIDER adapts sparsity accordingly. (b) Visualize layer-specific semantic drift compared to magnitude-based pruning.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a training-free framework named SPIDER, which aims to enhance the inference efficiency of MLLMs through the co-optimization of visual token pruning and LLM decoder computation. The core idea demonstrates significant novelty and practical value, particularly in its utilization of intermediate-level semantic information from the visual encoder and the introduction of a fine-grained sub-layer skipping mechanism. The experimental design is comprehensive, validating the method's effectiveness across two model architectures and multiple benchmarks. However, some key details in the methodology require clearer explanation, the depth of comparison with related work needs strengthening, and certain aspects of the experiments could be further improved.

### Strengths
1. The co-design of token pruning and layer skipping is a natural yet powerful idea. The SPIDER framework demonstrates that even among the tokens retained after pruning, a significant amount of computation remains skippable. This "double-filtering" mechanism enables greater efficiency gains.
2. The method can be applied to existing models without requiring any additional training.

### Weaknesses
1. Could the authors further clarify the primary innovations of this work compared to existing studies? Given the substantial number of existing works on visual token pruning, what specifically distinguishes the proposed approach?
2. The experimental conclusions are primarily drawn from evaluations on the LLaVA family of models. It would be valuable to conduct further experiments to verify whether these conclusions generalize to other MLLM architectures, such as the Qwen-VL or InternVL series.
3. The paper mentions that intermediate layers capture fine-grained object-centric details, but the specific layer indices defining an "intermediate layer" (Layer M) versus a "deeper layer" (Layer L) are not clearly defined.
4. The formulas present a conceptual framework for the multi-level similarity score S_ij, but they do not specify how the values of the hyperparameters α and β are determined. It is recommended to supplement this with an analysis of hyperparameter stability, for instance, by showing the variance in performance across a range of different hyperparameter values.
5. It is suggested to include more diverse examples, particularly an analysis of failure cases. Illustating scenarios where SPIDER might incorrectly prune crucial tokens or skip computations that should not be skipped would help in understanding the method's limitations.

### Questions
Please refer to the above Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
