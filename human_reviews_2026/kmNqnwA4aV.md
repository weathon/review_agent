# Bridging Transformers and RWKV: Towards Efficient Multimodal Video Understanding

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Transformer-based Multimodal Large Language Models (MLLMs) struggle to process hour-long video inputs due to the quadratic computational complexity of causal self-attention, leading to prohibitively high computational costs during both training and inference. 
Existing token compression approaches reduce the number of video tokens, but often suffer from significant information loss and remain inefficient for extremely long sequences. In this work, we propose a hybrid RWKV-Transformer model that distills transformer layers into linear RNNs by reusing their attention projection weights, guided by a progressive distillation strategy. Without any token reduction, when replacing about 25\% of standard Transformer layers with RWKV modules improves throughput by 20\% compared to the original Transformer model, while matching its performance on multiple video understanding benchmarks such as Video-MME and MLVU, and even outperforming it on VNBench and LVBench, with average scores of 74.0\% and 46.8\%, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a hybrid RWKV-Transformer architecture to mitigate the quadratic computational complexity of processing long videos in MLLMs. The core idea is to replace a portion of the Transformer's self-attention layers with linear-complexity RWKV modules. To make this feasible, the authors introduce three main contributions: (1) a weight initialization strategy that remaps pre-trained attention weights ($W_Q, W_K, W_V$) to the new RWKV modules ($W_r, W_k, W_v$); (2) a parallel cross-attention mechanism that uses "scene tokens" to provide global context and mitigate RWKV's history decay; and (3) a three-stage progressive distillation process to train the hybrid model. The authors claim their model, when replacing 25% of layers, can match or exceed the baseline's performance while improving throughput by 20%.

### Strengths
1. The paper tackles the critical and well-known challenge of $O(N^2)$ complexity in long-context video modeling, which is a significant bottleneck for MLLMs.

2. The idea of re-using the pre-trained $W_Q, W_K, W_V$ weights to initialize the RWKV modules is a smart approach to leverage the knowledge from the original model and reduce the training cost and instability of introducing new, randomly-initialized architectures.

3. The paper provides informative ablation studies. For example, the study on replacement layer position (Table 3) clearly shows that replacing late layers is effective while replacing early layers is detrimental. Furthermore, the ablations in Table 4 effectively demonstrate the necessity of both the weight transfer and the cross-attention module.

### Weaknesses
1. The central claim that the 25% model "match[es] its performance on multiple video understanding benchmarks"  is an overstatement. While it does show marginal gains on VNBench (+0.6%) and LVBench (+1.5%), it shows clear performance degradation on other key benchmarks compared to the Qwen2.5-VL-7B baseline. On Video-MME, the hybrid model scores 61.3% (Avg), which is 2.5 points lower than the baseline's 63.8%. On MLVU, the hybrid model scores 68.0% (m-avg), which is 2.4 points lower than the baseline's 70.4%. This is not "matching" performance; it is a clear trade-off (sacrificing performance on some benchmarks for gains on others) that is not adequately acknowledged in the abstract or introduction.

2. The efficiency gains are presented in a potentially misleading manner. The abstract and introduction prominently claim a throughput increase of "up to nearly 2x". However, this 2x gain is only achieved when all Transformer layers are replaced (as shown in Figure 2b). The performance of this 100%-replacement model is never reported, and the catastrophic failure of the 50%-replacement model  strongly implies the 100% model's performance would be unusable. The actual model being advocated for (the 25% variant) achieves only a 20% throughput gain. This is a modest improvement, not a 2x one.

3. The proposed method appears to be extremely brittle. The ablations reveal that the approach only works in a very specific "sweet spot": 
- It fails when replacing early layers.
- It fails when replacing 50% of layers, causing a "dramatic decrease" in performance. This suggests the method is not a robust, generalizable hybridization strategy. Instead, it's a highly-tuned workaround that breaks if a user attempts to push for more efficiency by replacing more layers. This severely limits the method's primary utility.

4. The 20% inference speedup (which comes with a performance trade-off, not a clear win) is achieved at the cost of a significant increase in training and architectural complexity.

### Questions
Please see Weaknesses. While the paper presents an interesting direction for hybridizing models, the execution is not convincing. The performance claims are overstated (the model underperforms the baseline on several key benchmarks), the efficiency gains of the usable model are modest (20%, not 2x), and the method itself appears brittle, complex, and difficult to scale. Therefore, the paper in its current form falls below the acceptance threshold.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a hybrid RWKV-Transformer architecture to address the quadratic complexity bottleneck of Transformers in long-video understanding. Instead of reducing visual tokens (which causes information loss), the authors replace selected Transformer self-attention layers with RWKV modules, a linear RNN-style architecture, achieving pp to 2× inference speed-up without token reduction
20% throughput gain by replacing just 25% of layers. The model (based on Qwen2.5-VL-7B) outperforms efficient baselines like LongLLaVA and VAMBA while using less training data and no token compression.

### Strengths
1. The paper introduces the first hybrid RWKV-Transformer architecture for video MLLMs, combining the efficiency of linear RNNs with the expressiveness of Transformers.
2. The paper proposes a parameter remapping strategy that reuses pre-trained Transformer weights to initialize RWKV modules—no full retraining required. Enables fast adaptation and knowledge preservation from powerful pre-trained models like Qwen2.5-VL.

### Weaknesses
1. RWKV Trade-offs: RWKV’s constant-memory recurrence is efficient, but its data-dependent decay still causes slight degradation on the longest videos.
2. Training Cost. Although pre-trained weights are reused, the three-stage distillation still requires ~2.5 M video samples and 8×A800 GPUs; total GPU-hours and carbon footprint are not disclosed, limiting cost-benefit analysis.
3.  Over-hyped speed-up: the “≈2×” claim only holds if every attention layer is swapped (a configuration whose accuracy is never even reported). The promoted 25 % hybrid yields a mere 20 % throughput boost—far less than the headline suggests.

### Questions
1. Long-video degradation with RWKV. Figure 2 shows that 50 % layer replacement already hurts accuracy, yet the 100 % RWKV model is never evaluated. Please provide full benchmark scores for the all-RWKV configuration or present evidence that its performance is unusable; otherwise the “≈ 2 ×” speed-up lacks practical relevance.
2. Training cost and reproducibility. The three-stage distillation uses ~2.5 M videos on 8 × A800 GPUs. Please report the total GPU-hours.

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
This paper addresses the significant computational bottleneck in Transformer-based Multimodal Large Language Models for long video understanding, which stems from the quadratic complexity of self-attention. The authors propose a hybrid RWKV-Transformer architecture that replaces a subset of the standard self-attention layers with Hybrid Decoder Layer. This new layer has two key components:An RWKV module that processes all visual and text tokens with linear complexity, providing the primary efficiency gain. A cross-attention module that runs in parallel. This module allows the text tokens to attend to a set of "scene tokens", which are adaptively pooled global representations of the video. This component is designed to mitigate the history decay common in RNNs by providing global context. To avoid costly retraining, the model is initialized by remapping weights from a pre-trained MLLM (Qwen2.5-VL), and a 3-stage progressive distillation strategy is used to align the new hybrid layers. The authors demonstrate that replacing 25% of the Transformer layers achieves a 20% throughput increase while matching or even slightly exceeding the baseline's performance on benchmarks like VNBench and LVBench.

### Strengths
1. The paper is generally well-written and well-structured. The problem is clearly stated, and the proposed architecture is explained in detail, clarifying the distinct paths for the RWKV and cross-attention modules. The 3-stage distillation process is also well-defined.

2. The paper tackles a practical problem. The "hybrid" approach and cross attention are practical and make sense.

3. The empirical validation is strong. The ablation studies in Section 5.3 are thorough and provide convincing evidence for the authors' design choices. Table 4 clearly shows the necessity of the cross-attention module to mitigate performance drops, and Table 3 rightly identifies that replacing late-stage layers is more effective than early-stage ones. The adaptive scene pooling based on similarity is also a better approach than naive uniform pooling.

### Weaknesses
1. The technical novelty is limited. The proposed architecture and distillation training are very similar to the discussed related work, Vamba, except that it uses RWKV instead of Mamba

2.  While the "up to 2x" throughput is highlighted, this is for a 100% replacement model that, based on the steep performance drop of the 50% model, is likely unusable. The practical, high-performance model (25% replacement) only achieves a 20% speedup.

3. The paper's design routes both text and visual tokens through the same RWKV module. An alternative, and perhaps cleaner, architecture was not explored: Apply RWKV layers only to the visual tokens (the bottleneck). Keep text tokens in standard Transformer layers. Have the text tokens cross-attend to the (now $O(N)$) RWKV-processed visual representations.

4. Lack of Memory Analysis: A critical component of "efficiency" is entirely missing from the analysis: inference memory. The primary appeal of models like RWKV is their $O(1)$ memory complexity during autoregressive generation, which contrasts with the $O(N)$ memory of a Transformer's KV-cache. Since this model is a 75% Transformer hybrid, it presumably still maintains an $O(N)$ KV-cache for the majority of its layers. This means that for "hour-long" videos, the memory bottleneck from the visual token KV-cache would still be enormous, even if the throughput is 20% faster. The paper's focus on throughput (tokens/s) alone feels incomplete.

### Questions
See W4. Could the authors please provide a detailed analysis of training/inference memory usage.
How does the memory consumption scale with the number of input video tokens?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a hybrid RWKV-Transformer model for efficient long-video understanding. It focuses on distilling a retrained Transformer model into a linear RNN-based model, achieving 2× higher throughput. In addition, the authors employ cross-attention to help the RWKV capture global context, mitigating its inherent history-decay issue

### Strengths
* Motivation is good: focus on archeiture design of long vide-LLMs
* method is reasonable, efficiently replace the complex attention block with linear attention block 
* the cross-attention is effecively to mitigate  RWKV's inherent history-decay issue of RWKV

### Weaknesses
* Some clarifications are not clear.  
  For example, in the abstract, the authors mention that the throughput increases by nearly 2×.  
  Under what setting is this achieved? Does it maintain the same performance?

* Insufficient baselines:  
  * Throughput comparison with other baselines such as VAMBA and SlowFast-MLLM.  
    It would be better to include a throughput-versus-performance curve to compare your model with these baselines.  
  * Ablation using the same fine-tuning data on the original architecture.
  * What is the performance when replacing the causal attention with RWKV without any additional training?

### Questions
Is the difference between Stage 2 and Stage 1 that Stage 2 includes MLP training and uses different data?

Also, why did the authors specifically choose RWKV? There are many types of linear attention mechanisms to choose from — I would like to know the authors’ reasoning behind this choice.

### Soundness
2

### Presentation
2

### Contribution
3
