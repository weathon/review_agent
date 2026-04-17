# HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
The quadratic computational cost of processing vision tokens in Multimodal Large Language Models (MLLMs) hinders their widespread adoption. While progressive vision token pruning offers a promising solution, current methods misinterpret shallow layer functions and use rigid schedules, which fail to unlock the full efficiency potential. To address these issues, we propose HiDrop, a framework that aligns token pruning with the true hierarchical function of MLLM layers. HiDrop features two key innovations: (1) Late Injection, which bypasses passive shallow layers to introduce visual tokens exactly where active fusion begins; and (2) Concave Pyramid Pruning with an Early Exit mechanism to dynamically adjust pruning rates across middle and deep layers. This process is optimized via an inter-layer similarity measure and a differentiable top-$k$ operator.  To ensure practical efficiency, HiDrop further incorporates persistent positional encoding, FlashAttention-compatible token selection, and parallel decoupling of vision computation to eliminate hidden overhead associated with dynamic token reduction. Extensive experiments show that HiDrop compresses $\sim$90\% visual tokens while matching the original performance and accelerating training by 1.72$\times$. Our work not only sets a new state-of-the-art for efficient MLLM training and inference but also provides valuable insights into the hierarchical nature of multimodal fusion. The code is released at https://github.com/EIT-NLP/HiDrop.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes HiDivDrop, a token pruning methods for vlm. The author claim there two main challenging in exist works. They are shallow layers are misinterpreted and pruning schedules are overly rigid. The author combined late Injection, concave pyramid pruning in the middle layers, and early exit to sort these challenges. The experiment shows the author proposed methods performance well.

### Strengths
The experiments are comprehensive and validated the effectiveness and acceleration performance of the proposed method.

The proposed method is well-designed, and the experiments further validate the claims.

### Weaknesses
1. The writing of this paper needs improvement. There are too many paragraphs. The purpose of Chapter 2 is unclear. Could it serve as the first section of the methodology, validated experimentally? As a separate chapter, it feels redundant with Chapters 1 and 3.

2. Tables 1 and 2 are too far removed from the analysis of the results to be easily readable.

### Questions
1.The authors only tested two backbones. If possible, could they test different sizes or other backbones to observe the generalization of the proposed method?

2.Does the proposed method exhibit knowledge drift? That is, in individual cases, answers that were previously correct may now be incorrect, and vice versa.

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
This paper introduces a novel visual token pruning methodology for VLMs, aimed at enhancing their inference efficiency. The proposed approach is methodologically straightforward: it involves skipping visual tokens in the shallow and deep layers while employing a 'learnable top-k selection mechanism' for tokens in the middle layers. Through experimental evaluation, the paper demonstrates that the proposed method outperforms baseline approaches under identical settings. Furthermore, the finding that visual tokens in the shallow layers play a limited role is an insightful observation. This insight holds significant heuristic value for the VLM research community.

### Strengths
1.  The finding that visual tokens in most shallow layers are also dispensable is a novel insight that holds significant heuristic value for the VLM research community.
2.  The methodology itself is sound; skipping non-essential layers combined with a learnable token selection in the intermediate layers allows for maximal visual token compression, a claim that is substantiated by the experimental results.

### Weaknesses
1.  The paper lacks a detailed justification for *why* visual tokens in shallow layers are non-essential. This finding serves as a critical premise for the proposed method, yet the supporting argumentation provided is neither detailed nor sufficient.
2.  The experimental validation relies on relatively older, and arguably undertrained, VLM models (e.g., LLaVA-1.5). It remains unclear whether the conclusions generalize to more recent, powerful models (e.g., Qwen2.5-VL, Gemma3-VL, Qwen3-VL) and across a broader spectrum of tasks.
3.  The description of the training strategy is unclear and requires polishing. It is not specified at which stage the learnable token selection mechanism is trained (e.g., during pre-training, SFT, or task-specific fine-tuning). This ambiguity hinders a clear understanding of the methodology.

### Questions
1.  Is the pruning strategy (e.g., skipping policy, selection parameters) uniform across different tasks? 
2.  How general is the proposed method? Could it, for example, be applied during the pre-training phase rather than just as a post-hoc optimization or fine-tuning strategy?
3.  The experiments are conducted on LLaVA-1.5, which may be undertrained. Do these conclusions consistently hold for more recent, powerful models (e.g., Qwen2.5-VL, Gemma3-VL, Qwen3-VL) and across a broader spectrum of tasks?

### Soundness
3

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
The manuscript unveils several key mechanisms as to how the visual information are processed in MLLMs, and use these findings to guide the design of HiDivDrop, a new visual token reduction method for MLLMs. The first finding is that shallow layers are merely propagators of visual contents into middle layers where true vision language fusion happens. Hence, HiDevDrop only inserts visual tokens at the beginning of middle layers. The second finding is that the fusion process happening in the middle layers is highly sparse, leading to the drastic reduction in the number of visual tokens during the middle layers in HiDevDrop. The final finding is that deep layers does not possess the capability of vision-language interaction. Thus, HiDevDrop drops all tokens in deep layers. Experiments show that HiDevDrop can retain strong performance under low visual token number regime.

### Strengths
1. The complete method is built on close observation of the vision-language fusion process in MLLMs, which not only produces reasonable model structures, but also provides important insights for future vision-language model research. 

2. The proposed method achieves a strong performance under low token number regime, retaining 96.5% of the performance using 48 tokens compared to original 576 tokens.

### Weaknesses
1. While the identified internal mechanisms are potentially insightful, the presentation and explanation of these mechanisms is of limited quality. Key concepts lack sufficient clarification, making readers incapable of following the reasoning process that helps produce the conclusion. This is especially the case for Figure 2 and 3, (questions detailed in the Questions section below). 

2. The analysis seems to be limited to a single type of language model, which lowers the credibility of the generality of the observed underlying vision-language fusion mechanism.

3. Under a similar training budget, it seems the proposed method achieves lower performance than PDrop. Though at a lower inference computation budget, the important metric is the inference latency, which the manuscript fail to provide. It is also confusing why the method, reducing so many visual tokens (64 compared to 270 in PDrop), requires similar training budget. 

4. The experimental analysis on the proposed method is not sufficient. For example, it would be crucial to know how the the training/inference cost and performances change when the injection becomes later (starting from first-layer injection). 

5. It seems costly to determine the injection layer and the exit layer for each given MLLM, how can the hyperparameter be efficiently determined?

### Questions
1. What is the difference between the two figures in Figure 2? 

2. Further related to Figure 2, how is the statement in L151~L152 reflected? ('Fig. 2 shows that text embeddings are nearly invariant to the visual input in shallow layers')

3. What does p value mean in Figure 3?

### Soundness
3

### Presentation
2

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
This paper studies the visual token reduction problem for the multimodal large language models (MLLMs). The author analyzes the diverse impact of the visual tokens in shallow, middle, and deep layers for LLaVA-1.5-3B and LLaVA-1.5-7B, and proposes a hierarchical division-based vision token dropping (HiDivDrop) method where the shallow layers are handled with Late Injection and the deep layers are handled with Early Exit, and apply the Concave Pyramid Dropping in the middle layers to progressively reduce vision tokens. Experiments are conducted on the LlaVA-1.5 architecture with different LLM backbones and 11 mainstream benchmarks, verifying the effectiveness of the proposed method.

### Strengths
1. The proposed method is effective in reducing a large amount of visual tokens compared to the state-of-the-art methods on extensive benchmarks.

### Weaknesses
1. Generalizability of the analysis in Section 2. The analysis in Section 2 serves as the primary motivation for the proposed method. However, the details about the training and evaluation datasets, training configuration, and the exact model type (e.g., the LLM backbones) are missing. It is challenging to convince the reviewer that the empirical observation in Section 2 is generally applicable to various pretrained datasets, evaluation tasks, training configurations, and model types for MLLM. The author should provide more context for the empirical observation to let the reader understand the generalizability of the observation.

2. The applicability of the Joint Visual Layer Reduction. In Section 3.1, lines 227-228 and lines 232-234, the author selects the specific layers for Late Vision Injection and Early Vision Exit based on the empirical observations in Section 2. However, as mentioned in Weakness 1, the pretrained datasets, evaluation tasks, training configurations, and model types for MLLM are not clarified in the main paper, making it hard for the reviewer to understand whether this layer selection is also applicable when we change to other pretraining datasets, evaluation tasks, and model types. 

Moreover, the author does not provide a concrete description of how to identify those in practice; for example, should we evaluate the MLLM performance, as in Section 2, every time we are given new datasets, evaluation tasks, and model types? Should we analyze the training dataset or the validation dataset to identify the optimal layers? How can we guarantee that the selection is optimal when we change the model training recipe? 

3. The choice of MLLM architecture. In Section 4, the author only evaluates the LLaVA framework as the MLLM. However, it is not clear whether the proposed method can be applied to other MLLM frameworks. 

4. Technical Contribution is limited. Most of the methods used in Section 3 have existed previously, and the overall novelty of the proposed method is limited.

### Questions
Please refer to the Weaknesses section for details.

### Soundness
2

### Presentation
3

### Contribution
2
