# Communication-Efficient Multi-Device Inference Acceleration for Transformer Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Transformer models power many AI applications but suffer from high inference latency, limiting their use in real-time settings. Multi-device inference can reduce latency by parallelizing computation. Yet, existing methods require high inter-device bandwidth, making them impractical for bandwidth-constrained environments. We propose ASTRA, a communication-efficient framework that accelerates Transformer inference through a novel integration of sequence parallelism and a Mixed-Precision Attention mechanism designed to minimize inter-device communication. ASTRA compresses non-local token embeddings via vector quantization and preserves task accuracy through two optimizations, Noise-Augmented Quantization and Distributed Class Tokens. Experiments on ViT and GPT2 across vision and NLP tasks show that Astra achieves up to 2.64$\times$ speedups over single-device inference and up to 15.25$\times$ speedups over state-of-the-art multi-device inferences, while operating under bandwidths as low as 10 Mbps.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the significant challenge of high inference latency in Transformer models, particularly in multi-device scenarios where inter-device communication becomes a dominant bottleneck in low-bandwidth environments. The authors propose ASTRA, a communication-efficient inference framework that builds on sequence parallelism but introduces a novel Mixed-Precision Attention mechanism to drastically reduce communication overhead. This mechanism computes attention using full-precision embeddings for local tokens while using low-bit vector-quantized (VQ) representations for non-local tokens transmitted between devices. To preserve model accuracy under such aggressive compression, ASTRA introduces two key optimizations: Noise-Augmented Quantization (NAVQ), a training-time regularization strategy that injects noise into quantized embeddings to improve generalization , and Distributed Class Tokens, which replicates the class token to each device and aggregates the outputs to mitigate information bias. Experiments on ViT and GPT-2 models show that ASTRA achieves substantial end-to-end speedups -- up to 2.64x over single-device inference and 15.25x over other multi-device methods -- in bandwidth-constrained settings (as low as 10 Mbps), while incurring only minor accuracy degradation.

### Strengths
The paper is pretty interesting wtih 2 proposed optimizations to speedup inference across multiple device. I am not a systems expert, but the math behind the optimizations is correct and makes sense.

### Weaknesses
The biggest weakness and drawback from the paper is the lack of large-scale experiments. Espeically the model choices in this paper is so small that none of these optimizations matter. Given the prevelance of open models of much larger sizes even ViTs and Qwen series models, it is empirically needed to make sure the proposed optimizations are quality neutral while providing the benefits. Without these results it hard to make a case for accepting the paper. I hope the authors can scale up the benchmarking further.

### Questions
see above.

### Soundness
2

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
4

### Summary
This paper introduces ASTRA, a communication-efficient framework for multi-device Transformer inference under bandwidth-constrained settings. Existing multi-device methods suffer from high inter-device communication overhead, which dominates latency when bandwidth is limited. ASTRA addresses this by combining sequence parallelism with a Mixed-Precision Attention mechanism: local attention is computed at full precision, while remote tokens are compressed via low-bit vector quantization.

### Strengths
This paper observes a real bottleneck in multi-device Transformer inference for low-bandwidth or edge environments, which is increasingly relevant for real-time AI applications.

### Weaknesses
1. ASTRA integrates known techniques (sequence parallelism + token quantization + noise augmentation), so the main contribution is in practical integration and bandwidth optimization, not a fundamentally new inference algorithm.
2. Lacks formal characterization of attention approximation error due to vector quantization and noise injection. Consider adding error bounds or theoretical analysis of how quantization and noise affect attention computation and model accuracy.
3. Experiments assume stable bandwidth; network variability or heterogeneous device scenarios are not tested.
4. Latency and energy claims are based on simulation; real-world multi-device deployment is not evaluated. Incorporate hardware-level profiling (multi-GPU, FPGA, or edge devices) to substantiate speedup and efficiency claims.

### Questions
see weakness

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
3

### Summary
This paper introduces ASTRA, a framework for accelerating transformer inference across multiple devices in bandwidth-constrained environments. The key part is a Mixed-Precision Attention mechanism that computes local attention with full precision while using vector-quantized embeddings for non-local tokens, reducing communication overhead. To preserve accuracy under aggressive compression, the authors propose Noise-Augmented Vector Quantization (NAVQ) and Distributed Class Tokens. Experiments on ViT and GPT-2 models across vision and NLP tasks demonstrate speedups of up to 2.64x over single-device inference and 15.25x over existing multi-device methods at bandwidths as low as 10 Mbps, while maintaining accuracy within 3.58% of the original models.

### Strengths
1. Addresses a real bottleneck: The paper identifies and tackles a genuine problem, that communication dominates latency (58.6-93.5%) in bandwidth-constrained multi-device inference.
2. Novel compression approach: The Mixed-Precision Attention mechanism is creative, using full-precision for local tokens and VQ for remote tokens.
3. Good evaluation: Extensive experiments across multiple architectures (ViT, GPT-2), tasks (classification, language modeling), and conditions (bandwidth, device count, heterogeneity) demonstrate broad applicability.
4. Practical compatibility: The framework integrates with existing quantization methods (8-bit, 4-bit), showing additional speedups of 1.35 to 2.73x when combined.

### Weaknesses
1. Limited architectural types: The evaluation focuses only on ViT and GPT-2, which are relatively small and dated models. Modern applications use much larger models (e.g., LLaMA variants). The scalability claims are weakened without evidence on contemporary, production-scale models.

2. Severe zero-shot degradation: Table 3 shows large performance drops in zero-shot settings (e.g., GPT-2M perplexity increases from 43.22 to 62.29, a 44% degradation). This is a critical limitation for practical deployment where generalization is essential. 

3. Baselines potentially unfair: The comparison with BP, TP, and SP assumes these methods use full float32 precision. However, these methods could also be combined with standard compression techniques (gradient compression, activation compression). A more fair comparison would evaluate "BP+8bit quantization" vs "ASTRA+8bit quantization" to isolate ASTRA's contribution. Additionally, recent methods like FlexGen or DistServe are not compared, making it unclear how ASTRA compares to the current sota.

4. Communication model oversimplified: The paper assumes fixed bandwidth and doesn't account for real-world network variability, packet loss, or latency jitter. The latency model appears to assume perfect overlap of computation and communication, which is rarely achievable. Dynamic bandwidth fluctuations common in WiFi environments (cited as the target deployment) could significantly impact the practical speedups. The authors should evaluate under realistic network conditions with variable bandwidth and packet loss, or at minimum discuss how ASTRA degrades under non-ideal conditions.

### Questions
Please address the implicitly listed questions in the weakness.

Can you provide a decision tree or heuristic for practitioners to select:
1. Number of groups G based on task type (vision vs. NLP) and target accuracy?
2. Commitment loss weight ε based on model architecture and dataset?
3. When to use distributed vs. single class tokens?

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
2

### Summary
The paper introduces a way to improve communication efficiency for multi device Transformer inference using sequence parallelism and mixed-precision. They show significant speedups on ViT and GPT2 scale models.

### Strengths
- The paper is well-presented and easy to follow
- Communication overhead is significant in large Transformer model distributed settings
- The use of codebooks is interesting

### Weaknesses
- The models used for inference are small and it is not clear to me that these hold at scale.

### Questions
- It would be helpful to further motivate the wireless and edge deployment motivation for this framework. Why are we doing inference requests on wifi?

### Soundness
3

### Presentation
4

### Contribution
3
