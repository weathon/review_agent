# RepSpec: Structural Re-parameterized Draft Model Training for Speculative Decoding

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
As the parameter size of large language models (LLMs) continues to grow, the latency of autoregressive inference increases due to memory-bound computational inefficiency. To address this, speculative decoding has been proposed, where a large target model verifies multiple tokens generated in parallel by a smaller draft model. However, the performance of speculative decoding is fundamentally limited by the draft model’s capacity, which stems from the parameter gap between the two models. To overcome this limitation, we propose RepSpec, which combines structural re-parameterization with draft model training. During training, redundant linear structures are introduced and later merged into the backbone network during inference, thus enhancing the draft model’s training effectiveness without increasing inference cost. By applying our method to improve the current state-of-the-art approach, EAGLE, we achieve a significant improvement in accepted sequence length. Furthermore, considering the specific characteristics of the speculative decoding scenario, we explore a hybrid training strategy that combines linear and nonlinear structures, which yields a further improvement in acceptance length.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper adapts structural re-parameterization to draft model training for speculative decoding. While the idea is technically sound, the work suffers from marginal gains, limited novelty, and insufficient practical impact. Below are the key criticisms supporting rejection.

### Strengths
1 The paper presents a well-motivated adaptation of structural re-parameterization techniques to the emerging domain of speculative decoding. While this technique has been widely used in convolutional networks, its application to draft model training in autoregressive decoding is timely and relevant. 

2 The method effectively decouples training-time complexity from inference-time efficiency, maintaining the lightweight nature of draft models while enhancing their capacity during training.t

### Weaknesses
1 The work directly adapts structural re-parameterization—a well-established technique in convolutional networks—to draft model training without conceptual innovation. The hybrid variant introduces non-mergeable nonlinear components but fails to justify the increased inference costs, contradicting the low-latency objective of speculative decoding.

2 Performance improvements are modest: pure linear re-parameterization improves acceptance length by only 7–10% on LLaMA-8B, while the hybrid method incurs additional latency.

3 Experiments are confined to small models (≤13B) and academic benchmarks. The paper lacks comparisons to larger models.

### Questions
The paper focuses on comparing with EAGLE, Medusa, and Hydra, but how does RepSpec perform against other draft model optimization strategies such as knowledge distillation or dynamic architecture methods? Are there scenarios where training-free approaches (e.g., self-speculative decoding) might be more practical despite potentially shorter acceptance lengths?

### Soundness
2

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
This paper proposes a structured re-parameterization method, REPSpec, which enhances the training of draft models by introducing additional layers during training and merging them during inference. Extensive experimental results demonstrate that the proposed method significantly improves the performance of existing approaches.

### Strengths
1. The application of structured re-parameterization to the field of speculative decoding aligns well with the specific requirements of draft models.

2. Extensive experiments have been conducted to explore the effectiveness of different architectural designs.

### Weaknesses
1. The proposed method introduces a significant increase in training overhead.

2. Although it improves the acceptance rate, the hybrid approach also incurs additional inference costs, resulting in limited overall end-to-end acceleration.

### Questions
1. Although Appendix A provides some implementation details, the specific placement and strategy for incorporating nonlinear factors remain somewhat unclear.

2. If my understanding is correct, is there a fundamental difference between introducing unmergeable nonlinear factors and directly increasing the size of the draft model?

### Soundness
3

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
4

### Summary
This article proposes RepSpec, a method of training draft models in speculative decoding using structural re-parameterization. This method enhances the draft model's capacity during training without increasing its inference cost. The core idea is to augment the draft model's linear layers with redundant, mergeable branches (Pre, Post, Bypass) during training, which are then fused into a single layer for inference. Furthermore, the authors introduce a hybrid method that incorporates a minimal, non-mergeable nonlinear component, justified by the fact that the draft model's inference time is a small fraction of the total SD latency. Experiments on various SD methods (EAGLE, Medusa, Hydra) and LLMs (LLaMA-3.1-8B, LLaMA-2-13B, Vicuna-7B) demonstrate that RepSpec consistently improves the accepted sequence length and end-to-end decoding speed.

### Strengths
1. The article applies the structural re-parameterization techniques previously used in convolutional neural networks to the training of the draft model for speculative decoding, perfectly adapting to the characteristics of the draft model that are insensitive to training costs and sensitive to inference costs.
2. The experimental results fully demonstrate the effectiveness of its method, including end-to-end acceleration and draft acceptance rate.
3. This method has certain value in practical applications.

### Weaknesses
1. Limited Theoretical Insight: The paper provides a solid empirical foundation but offers limited theoretical analysis of why the re-parameterization helps in this specific context (beyond general optimization benefits). The discussion in Appendix E is a good start but could be more integrated.
2. Training cost (minor): Although the draft model is not very sensitive to training costs (as mentioned above), it still presents certain challenges in resource limited scenarios, especially when the base model is large.

### Questions
N/A

### Soundness
3

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
2

### Summary
This paper introduces RepSpec, a training framework to fix the key bottleneck in speculative decoding: the draft model's weak capacity. The core idea is structural re-parameterization. During training, authors expand the draft model by adding mergeable linear structures (like Pre and Bypass layers) to boost its capacity. At inference time, these structures are merged back into the original layers, resulting in zero additional inference cost. This ``train-large, infer-small" method improves the draft model's effectiveness, leading to better acceptance lengths and overall speedup. A Hybrid version also adds non-linear blocks for a minimal cost, which pays off on larger target models.

### Strengths
- The idea of merging the linear part of the model architecture during inference is novel and motivated. 

- The results shows speedups compared to the SOTA EAGLE. For example, there is the 7.3% improvement over EAGLE1 on LLaMA-3.1 8B (Table 1, T=0). The ablation studies are comprehensive.

### Weaknesses
- Pure linear method gives limited gains. Though the bypass path may make the training more effective, there is a performance ceiling for that. The ceiling can be related to the model size. The paper's own results show that while this method works well on the 8B model, the "Hybrid" method outperforms it on the larger 13B model. This implies that the zero-cost benefit comes with a performance ceiling that the authors themselves had to address with the costlier hybrid variant.

- The benefits are not as simple as adding more layers. Will adding too much linear re-parameterization can actually degrade training performance? This brings up the question that whether the re-parameterization structure is a sensitive hyperparameter that must be carefully tuned, rather than a simple, scalable fix.

- The training overhead is also a concern. The required training GPU memory increases and the training speed is also reduced.

- How about larger models?

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
