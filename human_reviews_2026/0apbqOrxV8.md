# Energy Efficient Language Models through Dynamic Sparsity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Transformer models, despite their impressive performance, often face practical limitations due to their high computational requirements driven largely by the memory-bound KV-cache. State-space Models (SSMs) attempt to address this issue with linear attention, easing memory pressure and improving compute and memory efficiency. However, their efficiency is instead limited by dense linear layers with inherently low arithmetic intensity, again leading to a memory-bound landscape, posing challenges for deployment on hardware-constrained edge devices where these models might otherwise excel. In this work, we present a technique to induce high activation sparsity in quantized SSMs with minimal performance degradation, both for smaller-scale models suitable for edge-deployment and larger billion scale models. We nullify activations within a trainable threshold ($\pm \Delta$), which preserves outliers that are crucial for high performance. With only 1/4 of the effective MAC (Multiply-Accumulate) operations of a dense model, our sparse MatMul-free models maintain competitive performance compared to the dense base model. As GPUs offer limited support for unstructured sparsity during inference, we target a neuromorphic hardware platform that efficiently supports this dynamic and unstructured activation sparsity on a silicon level. Based on previous deployment results of a dense model, our sparsified models can increase throughput by 37$\times$ while decreasing power consumption by 16$\times$ compared to an edge GPU-based deployment of a comparable transformer-based LLM. Compared to a baseline dense model on the same hardware, we show improvements of 5.4$\times$ in both metrics, paving the way for future explorations of highly efficient language models leveraging dynamic activation sparsity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a novel method to induce high activation sparsity in quantized SSMs (MMFreeLM).

### Strengths
1. The paper's experiments show that this method can achieve 76% reduction in effective MAC  operations with negligible impact on task performance.
2. The method is easy but useful. Using a learnable, per-projection threshold $\Delta$ allows the model to assign different sparsity levels based on layer sensitivity.

### Weaknesses
1. The paper's most impactful claims (e.g., 75mJ/token and 224.1 throughput in Table 3) do not appear to be actual measured results from running the sparse model on Loihi 2. As described in Section 4.5, these figures are calculateed from a performance model based on measurements of the dense model. The paper's claims would be strengthened if the authors could provide real acceleration data from Loihi 2, even on a single, small-scale sparse layer.

2. The proposed method requires keep training on top of the pre-trained model. This process introduces additional learnable parameters and a new loss term. The paper does not discuss the additional computational overhead or convergence time this adds to the training phase. Could author discuss and add those cost?

### Questions
See weakness above.

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
3

### Summary
This paper proposes a method to induce high activation sparsity in quantized State Space Models. The approach uses learnable threshold-based pre-activation gates that zero out activations within a delta while preserving outliers. The authors demonstrate that their sparse models maintain competitive performance with up to 72% activation sparsity and project significant efficiency gains when deployed on neuromorphic hardware.

### Strengths
1. The sensitivity analysis provides valuable insights into which projections and layers can tolerate sparsity, and provides proper motivation for the paper. The authors offer an orthogonal complement to pruning and quantisation, while building upon existing techniques.
2. The paper evaluates both 370M and 2.7B parameter models across multiple benchmarks, showing consistent results.
3. The learnable two-sided ReLU with smooth surrogate gradient is computationally lightweight and unlike prior ReLU-based sparsification it generalises across all projections and learns per-projection thresholds.
3. Targeting neuromorphic hardware that can actually exploit unstructured sparsity is a good choice that was thoroughly explained.

### Weaknesses
1.  Results are extrapolated from dense deployments rather than measured on actual sparse models. Real deployment results on Loihi 2 would strengthen credibility.
2. Competing SSM-based efficiency techniques (Mamba pruning, LoRA-style compression, structured token pruning) are not directly compared experimentally.
3. The paper claims “minimal additional training cost” but does not report training time or compute the overhead introduced by sparsity regularisation.
3.  The paper can include more important prior work on activation sparsity for neuromorphic computing. demonstrates similar sparsity-inducing techniques for neuromorphic hardware on simpler models. [ Activity Sparsity Complements Weight Sparsity for Efficient RNN Inference (https://arxiv.org/abs/2311.07625), Sparsity-Aware Hardware-Software Co-Design of Spiking Neural Networks: An Overview (https://arxiv.org/abs/2408.14437)
]

### Questions
1. Were the sparsity thresholds initialised globally or per-layer? How sensitive are results to this initialisation?
2. Have you explored structured sparsity variants such as block or channel for compatibility with GPU inference? Were the more stringent sparsity setups too detrimental to results achieved?
3. How stable is training with high lambda, beyond what values are there cases where too much sparsity leads to collapse?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a sparsification algorithm for SSMs targeting neuromorphic hardware, namely Loihi 2.

### Strengths
The sparsification algorithm of the paper seems to perform well, and introduces significant activation sparsity.

### Weaknesses
Sparsification is an old topic and I find the paper's contribution to be very limited to a single proposal that was laid out in one figure. They then introduced a modification to the loss function to "encourage" the pushing of the values to +/- \Delta. There was no explanation why the function is a good one, and also no details about how \Delta is learned - how is \Delta updated in the training process?

The other major weakness of the paper is that it just evaluated its proposal with other activation functions. Since sparsification is not new, there ought to be an evaluation against other sparsification methods. Without this, it is hard to place the contribution of the work.

### Questions
1. How would your method compare to other state-of-the-art sparsification algorithms, even for traditional ANNs?

2. Any reason for k = 10 (page 5) being a "good" value? Also, what's the intuition behind Eq. 3?

3. Would the approach also work for transformers?

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
3

### Summary
To address the high computational requirements introduced by the memory-bound KV-cache, State-space Models (SSMs) are proposed to ease memory pressure. But the new bottleneck of SSMs (linear layers) is still memory-bound, making it hard to be deployed on resource-constrained edge devices.
This paper proposed a trainable dynamic-sparsity mechanism for quantized State-Space Models (SSMs). The sparse model can achieve 37x better throughput on Intel Loihi 2 when compared with the dense model on edge GPU & 5.4x better throughput when compared with the dense model on the same hardware.

### Strengths
Over 60% MAC sparsity can be achieved with ~1% accuracy loss.

### Weaknesses
Loihi 2 sparse results are modeled, not measured.
Novelty compared with other work? It seems both TurboSparse & Q-Sparse have already introduced activation sparsity.

### Questions
How is the paper compared with other sparse LLM work?
What is the extra parameter size for the sparsification (delta, etc)?

### Soundness
2

### Presentation
2

### Contribution
2
