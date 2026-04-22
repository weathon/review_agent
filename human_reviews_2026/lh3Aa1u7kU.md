# Stacked from One: Multi-Scale Self-Injection for Context Window Extension

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 2

## Abstract
The limited context window of contemporary large language models (LLMs) hinders broader application. In this work, we present SharedLLM, a novel approach grounded in the design philosophy of multi-grained context compression and query-aware information retrieval. SharedLLM is composed of two short-context LLMs: a lower moel (compressor) and an upper model (decoder). The lower model compresses context information, while the upper model processes compressed, context information from the lower model and performs context-aware modeling. Information transfer between the compressor and decoder occurs only at the lowest layers to reduce redundant computation. Based on this architecture, we introduce a specialized tree-style data structure to efficiently encode, store and retrieve multi-grained contextual information from text chunks. This entire process, wherein the sender and receiver are derived from the same LLM layer, is referred to as self-injection. In our evaluation on long-context modeling and understanding tasks, SharedLLM achieves superior or comparable results to several strong baselines, striking an effective balance between efficiency and performance. Meanwhile, with the aforementioned design choices, SharedLLM can greatly reduce memory consumption, and demonstrates substantial speed-ups over other advanced baselines. The core code of our implementation along with training and evaluation is available in appendix and supplementary.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the limitation of finite context windows in Large Language Models (LLMs), this paper proposes the SHAREDLLM framework. Grounded in the concepts of multi-grained context compression and query-aware information acquisition, SHAREDLLM consists of two stacked models derived from the same base short-context LLM: a lower "compressor" and an upper "decoder". It enables "self-injection" through shared key-value (KV) states at the lower layers, thus avoiding redundant computations. A core innovation is the context tree—a binary tree structure that dynamically encodes long context into coarse-to-fine representations, expanding only task-relevant nodes. Trained on 8K-length sequences, the model can generalize to 128K+ tokens. It outperforms baselines such as CEPE and Activation Beacon on language modeling (perplexity) and long-context understanding tasks (LongBench/InfiniBench), with an inference speed 2× faster than streaming architectures and 3× faster than encoder-decoder architectures.

### Strengths
The dynamic expansion of the context tree adapts to task requirements, balancing information retention and efficiency, representing a relatively novel idea. It provides researchers with a lightweight alternative, lowering the threshold for long-context research; efficiency improvements expand the application of LLMs in long-text scenarios.

### Weaknesses
The model is only tested on up to 128K tokens, yet it claims to "generalize to arbitrary lengths"—supplementary experiments on 256K tokens or theoretical analysis are needed to support this claim.Experiments only provide quantitative results without qualitative examples (e.g., cases where the model correctly identifies key information in passkey retrieval tasks), making it difficult to intuitively demonstrate advantages.

### Questions
Can experiments on 256K tokens be supplemented? If not, theoretical justification for the "arbitrary length generalization" claim is required.

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
This paper proposes a two-LLM architecture for long-context inference. One LLM compresses the context, and the other LLM operates based on compressed context, achieving 2x speedup without hurting accuracy.

### Strengths
* A more end-to-end approach that also includes LLM training stage.

### Weaknesses
* This is a crowded space, and needs to be more embedded into related work to justify the novelty of this work.

### Questions
* The proposed method --- using LLM to compress context --- reminds me of MemGPT and generative agent paper, where they also uses the LLM to compress the context. How do you compare with these two approaches? I understand that you train the model end-to-end, but algorithm-wise is there any reason to believe your approach is better?
* Given the accuracy number of your system is close to token dropping based approach, and SnapKV, though it is a strong baseline, is not state-of-the-art based on NVIDIA's KVPress measurement (https://huggingface.co/spaces/nvidia/kvpress-leaderboard), I would suspect other baselines may have higher accuracy than SnapKV and be better than your approach. Is there any reason to believe that is not the case?
* Regarding training --- why SFT instead of RL?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes SharedLLM, a novel two-stage LLM framework designed to extend the context window of short-context language models. Specifically, it consists of 1) a lower model (i.e., compressor) that segments long input sequences and compresses each into hierarchical “context trees” and 2) an upper model (i.e., decoder) that retrieves relevant information from them. The mechanism is called self-injection. Cross-attention from query to the shared KV between the same base model’s layers efficiently injects the compressed/selected input context information. SharedLLM demonstrates long-context generalization (up to 128K tokens trained only on 8K) capability and outperforms baselines such as CEPE and Activation Beacon. In addition, the paper reports accelerated inference and memory usage savings.

### Strengths
* The idea of using a single model for two complementary purposes in long-context processing is novel. Leveraging the same base model to ensure compatibility is both intuitive and efficient.
* The context tree construction, which identifies relevant segments within each chunk without requiring an additional similarity computation module, is original and practical.
* The method avoids complex attention mechanisms, enabling the reuse of existing optimization techniques such as FlashAttention.

### Weaknesses
* As shown in Figure 4, performance improvements with respect to tree height and compression ratio appear somewhat inconsistent, suggesting potential sensitivity to hyperparameters. Furthermore, although the method is generally robust within moderate ranges, its reliance on rule-based policy selection and heuristic coarse-to-fine downsampling may limit task generalization.
* The lack of inter-chunk dependency modeling is understandable for parallelization and optimization efficiency, but it may introduce limitations in long context integration.
* The Passkey task might be too well aligned with SharedLLM’s query-aware design, making it an easier benchmark for this method. Additional justification or analysis would strengthen the empirical evaluation.

### Questions
* The proposed mechanism seems to be designed for the prefill phase; it may have a limited impact during decoding.
* When positional indices are added in the cross-attention module, are they implemented as sinusoidal position embeddings directly added to the key–value states, rather than as RoPE-style rotations?
* During fine-tuning, are only the cross-attention components trained while other parameters are frozen, or is the model fully fine-tuned?
* (suggestion) The near-half-randomness in context tree construction is interesting, but its benefits are not entirely clear. Given that later tokens within a chunk often implicitly encode earlier information, maybe a deterministic segmentation strategy, such as splitting at points of large neighboring vector similarity differences, would be better.

### Soundness
3

### Presentation
4

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
**Summary**

This work presents SHAREDLLM, a novel framework addressing the challenge of efficient long-context inference. Its key innovation is a two-stage process involving a lower model, which compresses context chunks via a Context Tree structure, and an upper model, which decodes  successive tokens from the compressed KV-cache. This architecture enables an extended context window while enhancing inference efficiency.

### Strengths
**Strengths**

（1）This study presents a novel method for addressing the challenges of long-context inference, with its architectural details thoroughly elaborated. 

（2）The efficacy of the proposed framework is demonstrated through experiments, which confirm its capability to extend the context window and improve inference efficiency. Furthermore, ablation studies provide evidence for the effectiveness of the introduced context information injection mechanism.

### Weaknesses
**Weakness**

(1) The research motivation is not sufficiently clear. The point raised in the introduction—that "specialized attention patterns may cause incompatibility with high-performance attention implementations"—does not adequately motivate the proposed method. For instance, prompt compression methods (e.g., ICAE and UniICL) can also improve long-context inference efficiency without requiring specialized attention mechanisms. However, the authors neither compare their method with these alternatives nor clarify the uniqueness of the problem their approach aims to solve.

(2) The abstract requires smoother expression. For example, the definition of "self-injection" is repeated, resulting in redundant content. Additionally, the mention of "sender" and "receiver" concepts in the abstract may lead readers to believe they are important modules, yet these terms are only mentioned there, which causes confusion.

(3) If no information has been overlooked, Tables 2 and 3 do not report the memory usage or inference latency of the baseline methods. Without aligning these factors, it is difficult to ensure a fair comparison between different methods.

### Questions
**Suggestions**

(1) Improve the expression in the Abstract and Introduction to enhance readability.

(2) Include the memory usage or inference latency of all compared methods in Tables 2 and 3 to ensure a fair comparison.

(3) If feasible, conduct experiments on larger models to validate the scalability of the approach.

### Soundness
2

### Presentation
1

### Contribution
2
