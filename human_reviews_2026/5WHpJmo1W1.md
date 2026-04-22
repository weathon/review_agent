# ChunkLLM: A Lightweight Pluggable Framework for Accelerating LLMs Inference

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Transformer-based large models excel in natural language processing and computer vision, but face severe computational inefficiencies due to the self-attention’s quadratic complexity with input tokens. Recently, researchers have proposed a series of methods based on block selection and compression to alleviate this problem, but they either have issues with semantic incompleteness or poor training-inference efficiency. To comprehensively address these challenges, we propose ChunkLLM, a lightweight and pluggable training framework. Specifically, we introduce two components: QK Adapter (Q-Adapter and K-Adapter) and Chunk Adapter. The former is attached to each Transformer layer, serving dual purposes of feature compression and chunk attention acquisition. The latter operates at the bottommost layer of the model, functioning to detect chunk boundaries by leveraging contextual semantic information. During the training phase, the parameters of the backbone remain frozen, with only the QK Adapter and Chunk Adapter undergoing training. Notably, we design an attention distillation method for training the QK Adapter, which enhances the recall rate of key chunks. During the inference phase, chunk selection is triggered exclusively when the current token is detected as a chunk boundary, thereby accelerating model inference. Experimental evaluations are conducted on a diverse set of long-text and short-text benchmark datasets spanning multiple tasks. ChunkLLM not only attains comparable performance on short-text benchmarks but also maintains 98.64% of the performance on long-context benchmarks while preserving a 48.58% key-value cache retention rate.  Particularly, ChunkLLM attains a maximum speedup of 4.48× in comparison to the vanilla Transformer in the processing of 120K long texts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ChunkLLM, a training and inference pipeline that employs a sentence-segmentation approach. It isolates the attention calculation for different sentences and uses a retrieval mechanism designed to compute attention only for the most relevant sentences.

### Strengths
Overall, the paper is comprehensive and clearly written. The experimental validation is thorough and supported by key visualizations.

### Weaknesses
In my opinion, while this paper is sufficiently novel, many of its designs are overly engineered and lack rigorous theoretical or empirical justification. Although this provides acceleration, it simultaneously introduces significant drawbacks: it fails to generalize to all task types, and its application scenarios are potentially ad-hoc.

### Questions
1.  How can the attention statistics of an entire chunk be estimated using only the separate token of that chunk? Theoretically, the relationship between the two is very distant, and a simple FFN seems insufficient for this process. Even with training, it is intuitively difficult to approximate, likely leading to inaccurate chunk retrieval. Could the authors explain this or provide convincing empirical evidence?

2.  Clearly, different layers, and even different heads within each layer, serve distinct functions. Intuitively, forcing all layers to vote should be significantly worse than allowing each layer to retrieve its own chunks. Why then, in the ablation study, did performance decrease after removing the vote mechanism? Furthermore, what is the motivation for this vote mechanism, and what advantages does it have over layer-specific retrieval?

3.  Regarding the ICAC property, is it related to the vote mechanism? It seems the vote mechanism inherently smoothes the chunk retrieval, causing all tokens within the same sentence to retrieve identical chunks. Do the authors agree with this assessment? Moreover, is this ICAC property present in the vast majority of examples, or does it only manifest in specific tasks like summarization or passkey retrieval?

4.  The sentence separation mechanism introduces calibration data, which prevents generalization across all datasets. For data types such as code and math, can the training method proposed by the authors still be effectively applied for adaptation?

5.  Why was LongBench not fully evaluated? I recommend the authors complete the evaluation on more subsets, especially the synthetic tasks.

### Soundness
1

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
This paper proposes ChunkLLM, a lightweight pluggable framework, featuring QK Adapter and Chunk Adapter to accelerate LLM inference.

### Strengths
1. The idea is straightforward and makes intuitive sense.
2. The results of LLaMA-8B and Qwen-7B are good.

### Weaknesses
1. The main concern I have with the paper is the practicality of the proposed adapters. The added complexity of training adapters for each model makes it much less scalable than existing training-free methods, such as StreamingLLM [1] and H2O [2], despite some accuracy performance improvements. To the best of my knowledge, this work falls more into `incremental' work with over-engineering techniques, where I fail to see any valuable insights for the ICLR communities.

2. How would the proposed method perform against prior training-free methods in terms of latency? 

3. How would the proposed method compare with newer baselines, such as ChunkKV [3] and Quest [4]?

4. How would the proposed method perform on larger models, such as 13B/30B?

5. The presentation needs some improvements. For instance, Figures 1 and 2 could be merged into one figure; Figure 3 is very hard to see. 

6. Some system-related works on KV cache compression are missing [5-7].

[1] Efficient Streaming Language Models with Attention Sinks, ICLR 2024.

[2] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models, NeurIPS 2023.

[3] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference, NeurIPS 2025.

[4] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference, ICML 2024.

[5] InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management, OSDI 2024.

[6] Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference, MLSys 2024.

[7] ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching, ISCA 2024.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

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
This paper introduces ChunkLLM, an efficient framework designed to enhance the computational efficiency of LLMs when processing long sequences. The core challenge addressed is the complexity of the self-attention mechanism in Transformers. ChunkLLM proposes a solution by integrating two lightweight, trainable modules into pre-existing LLMs: a Chunk Adapter and a QK Adapter. The Chunk Adapter, attached to the first layer, performs chunk boundary prediction using a simple feed-forward network, thereby segmenting the input into semantically coherent units. The QK Adapter, operating in parallel at each Transformer layer, compresses the full attention map into a chunk-level attention score via a knowledge distillation process that minimizes the KL divergence between the full and chunk-based attention distributions. The authors demonstrate that ChunkLLM generalizes effectively to long contexts, achieving inference speedup over the vanilla Transformer while retaining its performance on long-context benchmarks.

### Strengths
(1) The problem discussed in the paper, improving the efficiency of transformer-based LLMs, is important and challenging. 

(2) The design of the proposed method is nice. The proposed method is pluggable, which allows for direct integration into existing LLMs without requiring costly retraining of the entire model backbone.

(3) The paper provides a thorough evaluation across multiple dimensions. Experiments on long-context benchmarks, short-context tasks, and inference efficiency demonstrate the method's superiority over baselines. The ablation studies effectively validate the contribution of individual components.

### Weaknesses
(1) The paper selects StreamingLLM and SepLLM as baselines. However, it omits comparison against other long-context methods for the inference time evaluation, for example, H2O or SnapKV (dynamic KV cache compression methods cited in the related work).

(2) The hyperparameters top-k and the number of "local chunks" are set empirically. There is no sensitivity analysis showing how performance and efficiency scale with these parameters. The optimal k is likely dependent on the total context length and task, and the paper provides no guidance on how to select it for a new application, representing a significant practical limitation for deployment.

(3) The paper focuses on computational speedup and KV cache size. However, it does not discuss potential memory overheads. The chunk voting mechanism and the storage of chunk-level attention scores introduce additional memory footprints. It would be great if the authors can introduce more about the full memory overheads.

### Questions
(1) How does the performance of ChunkLLM degrade as the semantic granularity required for chunking becomes more complex? For instance, would it struggle with documents with rich structures, such as tables?

(2) The chunk voting mechanism is shown to be beneficial, but what is the computational cost of this cross-layer consensus? Is there a risk of this process becoming a bottleneck itself when the number of layers and chunks is very high?

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
3

### Summary
The paper presents ChunkLLM, a lightweight, pluggable framework designed to accelerate LLM inference by reducing the quadratic cost of self-attention. The approach integrate two components to the existing Transformer achitectures: Chunk Adapter that predicts chunk boundaries based on contextual semantics and QK Adapters that used to distill attention. During training, the backbone model remains frozen, and only adapters are optimized via KL-divergence–based distillation. They perform experiments on Qwen2.5-7B and LLaMA3.1-8B show that ChunkLLM maintains comparable accuracy on short-text tasks and 98.6% on long-context benchmarks while achieving up to 4.48× inference speedup and about half KV-cache usage reduction.

### Strengths
1. The two modular adapters can be easily attached to existing LLMs without retraining the full model. And the functions of the two adapters are clear.

### Weaknesses
1. The papers does not have formal analysis of why attention distillation preserves semantic completeness or guarantees ICAC stability. Same for analysis of the chunk boundaries.
2. The paper is not compared with some current state-out-of-the art efficiency model such as FlashAttention 2, RingAttention, Mamba variants. 
3. The selection for parameters like top-k chunk seem empirically tuned. 
4. The figures look hand drawn which is a little bit weird.

### Questions
see weakness section

### Soundness
2

### Presentation
2

### Contribution
2
