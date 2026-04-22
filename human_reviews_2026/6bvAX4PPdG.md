# ARTS: Alleviating Hallucinations in Large Vision–Language Models via Redundancy-Aware Token Selection

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Large Vision--Language Models (LVLMs) demonstrate significant potential in multimodal tasks, yet they are prone to hallucinations, where generated outputs deviate from the visual evidence. A mainstream approach to mitigate hallucinations in LVLMs is to develop training-free decoding strategies. Most of these methods posit that hallucinations stem from insufficient attention to relevant information and therefore focus on strengthening the model’s utilization of informative content. Beyond this perspective, we reveal a new source of hallucination: Visual tokens in intermediate decoder layers often become redundant or noisy, thereby misleading multimodal reasoning. Next, we evaluate commonly used token-importance metrics and observe that they cannot effectively identify redundant visual tokens in this context. To address this problem, we introduce ARTS, a decoding strategy that 
first reintegrates the original visual embeddings to enrich essential visual information, and then employs a novel sink-token-based method to select important visual tokens in intermediate decoder layers. Extensive experiments on multiple benchmarks and LVLM architectures demonstrate that our approach consistently reduces hallucinations and improves factual alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the hallucination problem in Large Vision-Language Models (LVLMs) from the perspective of visual token redundancy. It proposes ARTS, a decoding-time method that reintegrates original visual embeddings to enrich essential visual information and then leverages sink-token-based attention to identify and retain informative visual tokens in intermediate decoder layers. Extensive experiments demonstrate the effectiveness of ARTS in reducing hallucinations and improving factual alignment across multiple LVLMs and benchmarks.

### Strengths
- The paper is clearly written, logically organized, and supported by well-designed figures that effectively illustrate both the motivation and the method.
- ARTS leverages sink tokens as reliable anchors to estimate token importance, offering a principled and adaptive way to identify redundant visual tokens without additional training.
- Extensive experiments on multiple LVLMs and benchmarks show consistent gains over various decoding-based baselines.

### Weaknesses
- **Incomplete baseline comparison**: Several recent hallucination mitigation approaches, such as VTI [1], VASparse [2], and CMI-VLD [3], are not included in the comparison. These methods respectively address latent-space steering, vision-aware decoding, and adaptive cross-modal consistency. Including them would strengthen the empirical validation and contextual positioning of this work. In addition, OPERA and SID are missing in Table 7 without explanation.
- **Experimental Models**: The evaluation focuses on relatively early LVLMs (e.g., LLaVA-1.5, InstructBLIP), whereas more recent and capable models such as Qwen2.5-VL or LLaVA-NEXT are not considered. This limits the assessment of scalability and relevance to current-generation systems.
- **Restricted benchmark scope**: The evaluation could be more comprehensive by including additional hallucination-oriented benchmarks such as CHAIR [4], MMBench [5], or the GPT-4-assisted hallucination benchmark [6], which would provide stronger evidence of robustness and generalization.
- **Hyperparameter sensitivity**: ARTS depends on several hyperparameters (e.g., starting layer K, retention ratio P), which appear to require model- and dataset-specific tuning. This sensitivity may limit generalizability and increase the deployment cost across different LVLMs.
- **Time and computational consumption**: As a decoding-time method, a key concern lies in the trade-off between performance gain and computational overhead. The steps of sink-token localization and redundant-token pruning may introduce additional latency and memory cost. Reporting metrics such as inference time, FLOPs, or speedup ratios would help quantify the quality–efficiency trade-off of ARTS.
- **Clarity in Comparison**: Some baseline implementations (e.g., OPERA with reduced beam size) may not fully reflect their best-reported performance, potentially affecting fairness.
- **Missing ablation**: The effect of the number of clusters in the sink-token localization step is not discussed. Such an ablation would clarify whether two clusters are optimal or if finer-grained partitioning could yield additional benefits.

[1]Liu Sheng, et al. Reducing hallucinations in vision-language models via latent space steering. In ICLR 2025.

[2]Zhuang Xianwei, et al. Vasparse: Towards efficient visual hallucination mitigation for large vision-language model via visual-aware sparsification. In CVPR 2025.

[3]Fang Hao, et al. Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs. In NIPS 2025.

[4]Rohrbach, Anna, et al. Object hallucination in image captioning. In EMNLP 2018.

[5]Liu Yuan, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV 2024.

[6]Zhao, Zhiyuan, et al. Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839 (2023).

### Questions
Refer to Weaknesses.

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
The paper introduces a training-free decoding strategy aimed at mitigating hallucinations in LVLMs. The authors identify a novel source of hallucination—redundant or weakly aligned visual tokens in intermediate decoder layers—that disrupt multimodal reasoning. ARTS reinforces informative visual signals by reinjecting original visual embeddings and employs sink–visual attention to evaluate token importance. By utilizing sink tokens as global information anchors, ARTS effectively prunes redundant visual tokens at intermediate layers without additional training. Extensive experiments across multiple LVLMs and benchmarks demonstrate consistent improvements in factual accuracy and lower hallucination rates, highlighting ARTS’s robustness and generalizability.

### Strengths
1. The paper reframes the hallucination problem by highlighting redundant token accumulation as a new perspective, extending beyond conventional explanations centered on insufficient visual attention or weak alignment. 

2. The use of sink–visual attention for redundancy estimation is creative and theoretically justified. Exploiting sink tokens as global semantic anchors is both efficient and intuitively interpretable.

### Weaknesses
1. Sink-token detection and redundancy scoring (via clustering and cross-token attention analysis) introduce additional computation during decoding, partially offsetting the benefits of token pruning, especially for large-scale inference.

2. ARTS assumes that attention correctly reflects token importance. In noisy or adversarial visual conditions, this assumption may not hold, potentially leading to incorrect pruning decisions.

3. There is a lack of comparison with current state-of-the-art (SOTA) models, and the experimental results are not satisfactory. Compared with other SOTA methods such as MCA[1], PATCH[2], and Less is More[3], the performance is not clearly superior.

[1] MCA-LLaVA: Manhattan Causal Attention for Reducing Hallucination in Large Vision-Language Models

[2] From Pixels to Tokens: Revisiting Object Hallucinations in Large Vision-Language Models

[3] Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective

### Questions
Please refer to Weaknesses. If the author can respond to my question directly, I will increase my score.

### Soundness
3

### Presentation
3

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
ARTS is a training‑free, mid‑layer decoding strategy for LVLMs. It detects sink tokens (via 2‑means on column‑wise attention), reinjects original visual embeddings to refresh visual signal, and then keeps only top‑p visual tokens by sink‑visual attention, pruning the rest. This aims to remove redundant/noisy mid‑layer visual tokens that contribute to hallucinations. Across MME/POPE/VizWiz/MM‑Vet and multiple backbones (7B and 13B), ARTS consistently improves factual alignment over strong decoding baselines.

### Strengths
1. Fresh diagnosis & evidence for mid‑layer redundancy. Section 3.1 explains infomation flow rationale. 

2. Simple, training‑free, model‑agnostic insertion point. ARTS operates at inference in intermediate layers (no finetuning.

3. The use of sink tokens as anchors is novel. Prior works reallocate away from sinks; ARTS leverages sinks to rank visual tokens.

4. Consistent gains across multi‑backbone, multi‑benchmark

### Weaknesses
1. There’s no runtime or memory comparison. This is important for deployment to compare how much time and compute needed for extra steps.

2. No direct “random vs ARTS” at the same K/P on the same tasks. Fig. 1b shows random pruning helps in mid layers; Tables 2–4/5/7 show ARTS improvements, but a side‑by‑side matched comparison is missing.

3. Sink detection stability unreported. In section 5.1, it proposes 2‑means on column‑wise sums but provides no statistics (e.g., typical number of sink tokens, layerwise stability, per‑head variability). Only Table 1 indirectly shows the importance of sinks.

### Questions
1. What are the latency/throughput/memory effects of ARTS vs. vanilla and vs. pruning baselines (visual‑text/CLS)? Please report wall‑clock, FLOPs (pre‑/post‑pruning), and GPU memory

2. Can you add a matched‑budget comparison (same K and P) against random pruning on MME perception and POPE? Fig. 1b suggests random helps mid‑layer; quantifying ARTS’s lift over random would isolate its value.

3. Please report typical number of sinks per layer/input, variance across heads, and stability across images/prompts; Table 1 (p. 5) shows sinks matter, but not how many you find.

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
4

### Summary
The paper introduces ARTS, a training-free decoding strategy to reduce hallucinations in LVLMs. Unlike prior work that attributes hallucinations mainly to insufficient attention to relevant information, this paper identifies that redundant or noisy visual tokens in intermediate decoder layers that mislead reasoning. ARTS first re-injects original visual embeddings to preserve essential visual information and then employs a sink-token-based cross-attention mechanism to assess token importance, pruning redundant visual tokens dynamically. Experiments across multiple LVLMs (LLaVA-1.5, mPLUG-Owl2, InstructBLIP, MiniGPT-4) and datasets (MME, POPE, VizWiz, MM-Vet) show that ARTS outperforms existing training-free methods (e.g., DoLa, VCD, DAMO, SID, OPERA) in reducing hallucinations while maintaining reasoning capability.

### Strengths
- **Clear Writing.** The paper's writing is clear and easy to follow. 
- **Novel Perspective.** The paper identifies visual token redundancy in intermediate layers as a new and underexplored source of hallucinations, extending beyond the common “insufficient attention” explanation.
- **Simple method.** ARTS is a training-free and lightweight decoding approach, making it practical to integrate into existing LVLMs without retraining or architecture changes.

### Weaknesses
+ **Incremental improvement of ARTS.** The biggest concern I have is the (very) limited performance gains. Compared to the baselines, ARTS seems to be not strong. For example, in Table 3, most of the improvement is within 1 point, without reporting statistical significance. This may raise questions about practical significance versus added inference complexity. 
+ **Lack of Intuitive/Theoretical Grounding.** The design of the current method lacks intuitive explanation and theoretical grounding. For example, there is almost no explanation of the 2-means clustering and the assumption of the Euclidean distance measure; there is no discussing of the form of re-injecting the visual tokens' information back to the hidden states. 
+ **Not Comprehensive Enough Evaluation Benchmarks.** It looks like the authors break down the POPE benchmark into multiple subsets to report (Table 3) instead of reporting the average scores. The paper also lacks the results on CHAIRs benchmark. 
+ **Missing Crucial Baselines.** Two recently published strong baselines are not discussed in the paper: 
  + Liu, Shi, Kecheng Zheng, and Wei Chen. "Paying more attention to image: A training-free method for alleviating hallucination in lvlms." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.
  + Li, Zhuowei, et al. "The hidden life of tokens: Reducing hallucination of large vision-language models via visual information steering." ICML, 2025.
+ **Multiple Formatting Issues.**
  + Tables' caption needs to be placed on top of the table body; 
  + The citation format is inconsistent and incorrect.

### Questions
See above (Weaknesses).

### Soundness
1

### Presentation
1

### Contribution
1
