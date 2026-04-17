# Task-Related Token Compression in Multimodal Large Language Models from an Explainability Perspective

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Existing Multimodal Large Language Models (MLLMs) process a large number of visual tokens, leading to significant computational costs and inefficiency. Instruction-related visual token compression demonstrates strong task relevance, which aligns well with MLLMs’ ultimate goal of instruction following. Previous works generally assume that visual tokens achieve better vision–language alignment in the shallow layers of LLMs, which have led to task-related token compression being primarily applied in intermediate LLM layers. In contrast, our study reveals that with proper selection, task-related token compression is feasible at the input stage of LLM with negligible performance loss. This new paradigm significantly reduces task-irrelevant visual tokens and its model-agnostic design enables application without modifying the LLM architecture. Specifically, we suggest that explainability methods for transformer-based architechtures can evaluate the global importance of each visual token with respect to the given instruction, which can effectively guide the task-related token compression for MLLMs. Furthermore, we propose to learn a mapping from the attention map of the first LLM layer to the explanation results, thereby avoiding the need for a full inference pass. Interestingly, this mapping can be learned using a simple and lightweight convolutional network, whose training is efficient and independent of MLLMs. Extensive experiments on 13 image and video benchmarks across three leading MLLMs (Qwen2-VL, LLaVA-OneVision, and VILA1.5) demonstrate the remarkable effectiveness and strong generalization of our approach. Additionally, our new compression paradigm achieves faster inference with reductions in both prefilling time and KV-cache memory.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a visual token compression method for Multimodal Large Language Models (MLLMs) based on explainability approaches, enabling task-related token compression at the LLM input stage. The core innovation lies in using explainability methods to evaluate the importance of visual tokens and training a lightweight network to predict these importance scores. Evaluations on 3 MLLM models and 11 image/video benchmarks show that the method maintains high performance while reducing computational complexity, prefilling time, and KV-cache usage.

### Strengths
1. It achieves an innovative paradigm shift, as the study challenges the previous view that shallow-layer visual tokens are indispensable.
2. The authors conducted comprehensive experiments to demonstrate the superiority of their method compared to other existing approaches.

### Weaknesses
Requires more discussion and ablation analysis; see Questions for details.

### Questions
1. Why is the network depth set to 5 layers? No ablation analysis is provided to justify this choice.
2. Can relevance prediction be improved using more advanced architectures without sacrificing efficiency?
3. The motivation explanation in Section 3.1 is insufficiently clear. Specifically, the causal link indicated by "therefore" in Lines 186–187 is not clear enough.
4. Why is one layer of attention sufficient? The paper mentions "first-layer attention suffices" but provides no theoretical or empirical analysis to support this claim.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to compress visual tokens for accelerating inference of multi-modal large language models (MLLMs).

Specifically, it transfers the explainability methods for transformer-based architectures to visual token pruning.
The explainability methods often keep a relevance map that could be used to measure the importance of visual tokens.

To validate the effectiveness of the method, experiments on various benchmarks (MME, MMStar, MMVet, Video-MME, MVbench, and MMBench-V) are conducted, demonstrating improvements over previous methods.

### Strengths
(1) The paper writes clearly and is easy to follow.

(2) It is interesting that the relevance map from explainability methods can accurately identify important visual tokens.

### Weaknesses
(1) To derive the relevance map, it requires the ground truth labels to calculate gradients. Thus, the paper proposes a lightweight model to distill knowledge from the derived relevance maps. However, it is challenging to make it generalizable to various data, as the lightweight model is small and the size of training data is also much smaller than that used for MLLMs.

It would be better to evaluate the model performance on various benchmarks, like GQA, SQA, VQAv2, VizWiz, and MMB.


(2) For the lightweight model, it takes the first-layer attention scores as inputs.
     Why not use the original visual tokens as inputs?     
     Is it helpful to use multiple-layer attention scores?

### Questions
See weaknesses.

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
The paper proposes a task-related token compression paradigm for Multimodal Large Language Models (MLLMs) to address the high computational cost and inefficiency associated with a large number of visual tokens in the LLM input. The authors first use an explainability method (e.g., gradient-weighted multi-head attention) to assess the global importance of each visual token relative to a given instruction, generating a "ground truth" importance score. Building on this insight, they train a lightweight convolutional network to predict this importance score based solely on the LLM's first layer attention map. Experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is well-written and easy to understand. Figure 1 and 2 are intuitive to understand the motivation and the big picture of the proposed method.

2. The authors successfully demonstrate the feasibility and efficiency of performing task-related token compression at the LLM input stage with negligible performance loss, providing a new, more efficient compression strategy.

3. Experiments validate the effectiveness across three distinct MLLM architectures (Qwen2-VL, LLaVA-One Vision, VILA1.5) and 11 diverse image and video benchmarks, showcasing its robustness and wide applicability.

### Weaknesses
1. The lightweight convolutional network is trained using only the first-layer attention map ($A^0$). A more in-depth analysis is needed to justify how such a small network, using only shallow information, can accurately and robustly predict the global, task-specific importance score, which typically requires information propagated through multiple LLM layers.

2. This train-based methods may overfit to the training data. I wonder the performance on some OCR related benchmarks (TextVQA, Chartqa, DocVQA, OCRBench), since they are more challenging to validate the effectiveness of the proposed method.

### Questions
The paper mentions training using only 10K image samples. Please elaborate on the data sampling strategy (e.g., are they sampled from the test benchmarks? Are they general-domain? How is diversity ensured?) Is the small scale of the training data a potential limitation when generalizing to open-world or other benchmarks scenarios beyond the 11 evaluation benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel approach for task-aware visual token compression in multimodal large language models (MLLMs), aiming to eliminate instruction-irrelevant tokens at the LLM input stage to enhance inference efficiency—without modifying the underlying architecture. The key idea is to utilize explainability techniques to assign relevance scores to visual tokens with respect to a given instruction, guiding the compression process accordingly. Furthermore, the authors demonstrate that a lightweight convolutional network can be trained to map first-layer attention maps to explainability-derived importance scores, enabling token importance prediction without a full forward pass.

### Strengths
1. The work tackles a critical bottleneck in MLLMs that high computational and memory overhead from large numbers of visual tokens by introducing an instruction-aware token compression paradigm that operates without architectural changes.
2. The proposed compressors are lightweight, transferable, and require minimal retraining across diverse MLLM architectures, underscoring strong generalizability.
3. Evaluation spans 11 benchmarks for both images and videos, with comprehensive ablations, efficiency analyses, and baseline comparisons.

### Weaknesses
1. The paper relies heavily on empirical findings and visual evidence, offering limited theoretical grounding. The mathematical formulation (e.g., the relevance propagation equation for 𝑅t in Section 3.2) lacks a deeper analysis of properties such as linearity, gradient behavior, and attribution faithfulness, especially under ambiguous or multi-factor instructions.

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
3
