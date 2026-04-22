# Positional Encoding for Spiking Transformers

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Spiking Neural Networks (SNNs) offer superior energy efficiency compared to Artificial Neural Networks (ANNs). Recent Transformer-based SNNs have achieved promising performance by integrating spike-driven computation with Transformer architectures.
Positional information is essential in sequential tasks. However, existing positional encoding methods designed for ANNs cannot be directly applied to SNNs, as they interfere with the spike-driven computation paradigm, highlighting the need for SNN-specific solutions.
We propose Spiking Positional Encoding (SPE), a novel positional encoding specifically designed for Spiking Transformers that captures both absolute and relative positional information. Its key component is the Positional Encoding Leaky Integrate-and-Fire (PE-LIF) neuron layer, which encodes positional information directly into neuron thresholds. Through continuous spike firing and membrane potential reset processes, this positional information is effectively reflected in the emitted spikes while preserving the spike-driven computation paradigm.
Comprehensive experiments across seven datasets, including three time-series forecasting tasks and four natural language processing benchmarks, demonstrate that SPE consistently outperforms existing positional encoding methods and achieves state-of-the-art performance.
SPE provides a tailored positional encoding solution for Spiking Transformers, bridging the performance gap between ANNs and SNNs, thus advancing neuromorphic computing applications in sequential modeling tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors address the lack of absolute and relative positional encoding in Spike Transformers and, drawing inspiration from positional encoding designs in the ANN domain, propose a set of design principles for positional encoding in SNNs. Based on these principles, they design a novel neuron model, PE-LIF, which integrates absolute positional information into its threshold dynamics. Theoretical analysis further demonstrates that this neuron model, while encoding absolute positional information, is also capable of representing relative positional information. Moreover, the authors introduce an MPR loss to maintain the theoretical assumptions. Finally, extensive experimental results are presented to validate the effectiveness of the proposed method.

### Strengths
1. The motivation of this paper is clear and well-defined, directly targeting a major pain point in the current Spike Transformer research — the lack of positional encoding, which in turn limits the effectiveness of SNNs in sequential modeling.

2. The paper provides thorough analysis and well-founded theoretical proofs, accompanied by comprehensive experiments that validate the proposed method’s effectiveness across multiple datasets.

### Weaknesses
1. The letter notations in the paper are somewhat inconsistent. In Section 3.1, t denotes the time step, N denotes the number of tokens, and D represents the dimension; in Section 4.3, L refers to the number of PE-LIF neuron layers; yet in Table 2, L is again used to represent the input length.

2. The novelty of this paper remains open to question. Although the problem it addresses is indeed an important one in the SNN field, the proposed method is largely based on RoPE — both in terms of the underlying formulation and the way absolute positional encoding is extended to relative positional encoding through the attention mechanism. Moreover, there is no ablation study specifically analyzing the effect of the newly proposed MPR loss; only a comparison of one related metric is provided, and the improvement is not particularly significant. Overall, the work appears to focus primarily on the PE-LIF neuron design derived from RoPE, rather than introducing a fundamentally new mechanism.

3. The remaining details can be found in the Questions section below.

### Questions
1. In Proposition 2, the authors state that as the relative distance between tokens increases, their dependency weakens and eventually approaches zero, thereby suggesting that the proposed encoding method enables dependency decay with distance. However, wouldn’t proving monotonicity or approximate monotonicity better demonstrate this characteristic?

2. In Proposition 1, the authors introduce a hypothesis, and in Section 4.2 they incorporate the MPR loss to ensure that this hypothesis is approximately satisfied during training. Could the authors visualize the variation of MPR loss throughout the training process?

3. According to Figure 4, the introduction of MPR loss seems to have little impact on the R² and RSE losses. Have the authors conducted an ablation study where MPR loss is removed and only PE-LIF is considered?

4. In this paper, the variable T used during sequence modeling with SNNs appears to be obtained through repetitive encoding. Although such repeated encoding is widely used in image processing, its role in sequential modeling is not entirely clear. Could the authors further explain the rationale behind adopting this approach for sequences?

5. The authors encode positional information into the thresholds of spiking neurons, resulting in each neuron having a distinct threshold. Since the number of neurons is directly related to the input token length, could the authors discuss the implementation complexity of such independently parameterized thresholds when deploying the model on actual neuromorphic hardware?

### Soundness
2

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
5

### Summary
This paper proposes a novel positional encoding approach for Spiking Transformers; however, the experimental verification is insufficient.

### Strengths
The paper identifies a fundamental limitation in directly applying ANN-based positional encodings to Spiking Neural Networks (SNNs), effectively motivating the need for SNN-specific solutions. and proposes an Integer-firing neuron for positional encoding in spiking transformers.

### Weaknesses
1. The explanation of the complexity of SSA in the paper is unsatisfactory(shown in "A key advantage of SSA lies in its linear attention property, whereby the time complexity of SSA, line 179-184). This paper focuses on "Positional Encoding for Spiking Transformers"; however, the spiking self-attention in spiking transformers (or self-attention in general) is primarily designed for large models, where the embedding dimension $ D$ is substantial, and $D^2$ is certainly not negligible.
2. The model used in the experiments is too small（1-2M） to thoroughly validate the effectiveness of the positional encoding in the Spiking Transformer. Moreover, the experiments are relatively simple — for instance, this paper lacks evaluations on more widely recognized time series benchmarks such as the ETT dataset.
3. Lack of comparison with the SOTA ANN methods.
4. Lack of discussion on deployment on neural chips.
5. Lack of comparison with "Toward Relative Positional Encoding in Spiking Transformers"

### Questions
see above

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
5

### Summary
This paper addresses the critical issue of incorporating positional information into Spiking Transformers, as conventional methods from Artificial Neural Networks (ANNs) disrupt the essential spike-driven computational paradigm of Spiking Neural Networks (SNNs). The authors introduce a novel method called Spiking Positional Encoding (SPE), which is specifically designed to overcome this limitation. The core of SPE is the Positional Encoding Leaky Integrate-and-Fire (PE-LIF) neuron layer, which ingeniously encodes both absolute and relative positional information directly into the firing thresholds of neurons, thereby preserving the event-driven nature of SNNs. The authors provide a theoretical foundation for their method, proving that SPE can represent relative positions and exhibits a desirable long-term decay property. Through comprehensive experiments on seven datasets spanning time-series forecasting and natural language processing, SPE is shown to consistently outperform existing approaches and achieve state-of-the-art results, effectively bridging a significant performance gap between ANNs and SNNs in sequential modeling tasks.

### Strengths
This paper presents a compelling and well-executed study, exhibiting significant strengths across originality, quality, clarity, and significance. The originality of this work lies in its novel formulation of positional encoding specifically for Spiking Transformers. Instead of adapting ANN-based methods that disrupt the spike-driven paradigm, the authors introduce Spiking Positional Encoding (SPE), a creative solution that embeds positional information directly into the firing thresholds of a newly proposed PE-LIF neuron layer. This approach elegantly captures both absolute and relative positional information within a unified, spike-native framework. The quality of the research is exceptionally high, substantiated by both rigorous theoretical analysis and comprehensive empirical validation. The authors provide formal proofs (Propositions 1 and 2) to demonstrate SPE's capability to represent relative positions and its possession of a desirable long-term decay property. These theoretical claims are backed by extensive experiments across seven diverse datasets, where SPE consistently achieves state-of-the-art performance. Thorough ablation studies further strengthen the findings by methodically demonstrating the contribution of each component. The paper is presented with outstanding clarity; it logically progresses from a clear problem analysis to a set of well-defined design principles, which then guide the development of the proposed solution. The manuscript is well-written, and the high-quality figures effectively illustrate the core concepts. Finally, the work is highly significant as it addresses a critical bottleneck that has limited the application of SNNs to complex sequential tasks. By providing a principled and effective solution, this research substantially bridges the performance gap between ANNs and SNNs, advancing the field of neuromorphic computing and paving the way for the development of more powerful and energy-efficient, brain-inspired architectures.

### Weaknesses
1. The paper's persuasiveness is somewhat weakened by the lack of validation in the computer vision domain.
2. Hand-crafted, non-learnable positional thresholds. The PE-LIF thresholds adopt fixed sinusoidal formulas (Eq. 8), with even-dimension requirement and a global $\lambda$. This design may be under-adaptive across modalities, sequence scales, or layers, and the paper does not explore learnable or data-driven variants (e.g., per-layer amplitudes/phases or low-rank adapters).

### Questions
1. I would like to see your performance on static vision datasets as well as neuromorphic vision datasets.
2. Learnability of PE. Why not make the threshold modulation learnable (per-layer/per-head amplitudes, learnable frequencies/phases) and regularize toward sinusoidal priors? Would a learned variant outperform fixed sin/cos?
3. Where to place PE-LIF? Beyond the current placements (input/MLP tail and Q/K activations), did you evaluate (i) only Q, (ii) only K, (iii) V, or (iv) alternating layers? A placement study would clarify where positional signals are most beneficial.

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
5

### Summary
This paper proposes Spiking Positional Encoding (SPE) for spiking Transformers. The core mechanism replaces LIF layers with PE‑LIF layers whose firing thresholds are position-dependent, thereby embedding positional information into the spikes without breaking the spike‑driven computation paradigm. Absolute position is injected by using PE‑LIF in the first spike‑encoding layer and at the end of each MLP, while relative position is encoded by using PE‑LIF for the Q and K activations in Spiking Self‑Attention (SSA). They further introduce a membrane‑potential regularizer (MPR‑Loss) to approximately enforce the required expectation condition. Experiments on time‑series forecasting and text classification show consistent improvements.

### Strengths
1. SNN-compatible method. Encoding position via threshold modulation inside PE‑LIF is a clean, spike‑domain design that preserves binary spikes. This approach is simple, yet novel.
2. The proposed approach is supported by theoretical analysis.
3. MPR‑Loss operationalizes the expectation‑matching precondition and, in ablations on Solar with L=24, improves R² by about +0.022 and reduces RSE by 0.029. The accompanying histograms also show FR–mean alignment improvements in Q/K layers.
4. Well-written and organized manuscript.

### Weaknesses
1. The “longer‑horizon” advantage is not convincingly substantiated. The paper claims SPE is particularly effective for longer prediction horizons, but the deltas vs. the spike‑driven baseline are mixed. For example on Electricity, the R² gain of SPE over the spiking baseline is +0.027 at L=6 (0.983 vs. 0.956) and +0.021 at L=96 (0.964 vs. 0.943), which does not evidence a stronger effect at longer horizons; trends on other datasets should be similarly quantified, not only averaged.
2. Limited breadth of baselines on spiking Transformers with relative PE. The paper argues that relative PE usually breaks linear SSA; however, several linear‑attention–compatible relative/bias schemes (e.g., simple phase rotations on Q/K, ALiBi‑style biases) can be adapted without explicitly forming attention maps. A direct comparison (or at least a careful adaptation study) is missing, so it remains hard to isolate the value of encoding position in thresholds versus in Q/K phases or biases.
3. Where and how absolute PE is injected deviates from conventional approach and needs stronger justification. The method places APE at the very first spike‑encoding layer and at the end of each MLP. The rationale, and the effect of each insertion point are not fully analyzed.
4. Limited applications. The authors present experimental results applying their method to various tasks, but their application is limited compared to other related papers. How does the method perform on image classification benchmarks commonly used in ViT, such as ImageNet?
5. Energy analysis is absent. 
6. Limited hyperparameter exploration and robustness anlaysis.
7. To verify that the proposed method works well as a PE, visualization of the proposed PE is required.

### Questions
Please refer to Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
