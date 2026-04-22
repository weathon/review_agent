# TINY BUT MIGHTY: A SOFTWARE-HARDWARE CO- DESIGN APPROACH FOR EFFICIENT MULTIMODAL IN- FERENCE ON BATTERY-POWERED SMALL DEVICES

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Large Multimodal Models (LMMs) are inherently modular, consisting of vision
and audio encoders, projectors, and large language models. Yet, they are almost
always executed monolithically, which underutilizes the heterogeneous accelera-
tors (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency.
In this paper, we present NANOMIND, a hardware–software co-design inference
framework for Large Multimodal Models (LMMs) that breaks large models into
modular “bricks” (vision, language, audio, etc.) and maps each to its ideal accelera-
tor. The key insight is that large models can be broken into modular components and
scheduled to run on the most appropriate compute units. It performs module-level
dynamic offloading across accelerators on unified-memory SoCs. By combining
customized hardware design, system-level scheduling, and optimized low-bit com-
putation kernels, we demonstrate our framework with a compact, battery-powered
device capable of running LMMs entirely on-device. This prototype functions as
a self-contained intelligent assistant that requires no network connectivity, while
achieving higher throughput and superior power efficiency under strict resource
constraints. The design further bypasses CPU bottlenecks and reduces redundant
memory usage through token-aware buffer management and module-level coordi-
nation. Our system outperforms existing implementations in resource efficiency,
cutting energy consumption by 42.3% and GPU memory usage by 11.2%. This
enables a battery-powered device to run LlaVA-OneVision-qwen2-05B with a
camera for nearly 20.8 hours.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents NANOMIND, a hardware-software co-design framework for efficient on-device LMM inference. The key idea is decomposing LMMs into modular components and dynamically scheduling each to the most suitable accelerator (NPU/GPU/CPU) on unified-memory SoCs. The authors build a custom battery-powered device with RK3566 SoC, achieving 42.3% energy reduction and enabling 20.8 hours of operation. The system features Token-Aware Buffer Manager for zero-copy transfer, battery-aware execution modes, and custom low-bit quantization kernels.

### Strengths
1. **Holistic system design**: The paper presents a rare end-to-end co-design spanning algorithm (model decomposition, quantization), system (scheduling, memory management), and hardware (custom PCB with PMU, parallel memory). This comprehensive approach addresses multiple bottlenecks simultaneously rather than optimizing in isolation.

2. **Practical hardware validation**: Unlike many systems papers that only simulate or use off-the-shelf platforms, the authors designed and fabricated custom hardware, providing concrete evidence of feasibility and real power measurements through integrated PMU.

3. **Novel heterogeneous scheduling**: The insight of mapping modular LMM components to different accelerators based on their computational characteristics (NPU for low-bit vision encoding, GPU for FP16 LLM decoding) is well-motivated and demonstrates clear benefits in the unified memory architecture context.

4. **Strong empirical results**: The battery life improvements (20.8 hours for voice interaction) and energy efficiency gains (42.3% reduction) are substantial and practically meaningful for edge deployment scenarios.

### Weaknesses
1. **Venue mismatch concern**: This work is fundamentally a hardware systems paper with custom PCB design, power management circuitry, and hardware-specific optimizations. While it has ML applications, the core contributions (hardware architecture, cross-accelerator scheduling, unified memory optimization) align more naturally with hardware/systems venues like HPCA, ISCA, or MICRO rather than ICLR, which focuses on machine learning methods and representations. The ML community may lack the expertise to properly evaluate the hardware contributions, and the authors would likely receive more targeted feedback from hardware systems reviewers.

2. **Limited generalizability**: The framework is tightly coupled to the RK3566 SoC and Rockchip's RKNN ecosystem. Key components (NPU offloading, RKNN model conversion, specific driver optimizations) may not transfer to other mobile SoCs (Qualcomm Snapdragon, MediaTek Dimensity, Apple Silicon). The paper lacks discussion of how the design principles would adapt to different hardware platforms or what abstractions could enable portability.

3. **Insufficient ablation studies**: While the paper shows end-to-end improvements, it doesn't systematically isolate individual contributions. What is the specific gain from: (a) parallel LPDDR4x vs. standard configuration? (b) zero-copy TABM vs. traditional buffer management? (c) NPU vs. GPU for vision encoding? (d) custom GEMM kernels vs. existing implementations? This makes it difficult to understand which design decisions are most impactful.

4. **Weak baseline comparisons**: The comparisons are primarily against llama.cpp on various platforms, but the paper doesn't compare against other recent edge inference frameworks (PowerInfer-2, llm.npu) on the same hardware. The NanoVLM comparison is limited to Jetson platforms. Additionally, the claim that llama.cpp is inefficient on unified memory architectures needs more rigorous support—is the inefficiency inherent to the framework or the specific platform/configuration?

5. **Missing accuracy-efficiency tradeoffs**: Figure 7 shows accuracy across quantization strategies but doesn't correlate these with latency, throughput, or power consumption. What accuracy degradation is acceptable for different battery levels? How do users navigate the performance-accuracy-power tradeoff space?

### Questions
1. **Portability strategy**: How would NANOMIND adapt to SoCs without dedicated NPUs (e.g., Mali-only systems) or with different NPU architectures (e.g., Qualcomm HTP)? What abstraction layer could make the framework hardware-agnostic?

2. **Static shape limitation**: You mention NPUs require static input shapes, which you address by fixing image resolution. How does this impact accuracy on datasets with varying native resolutions? Did you experiment with multiple fixed resolutions or dynamic resolution selection?

3. **Real-world deployment**: Your experiments use controlled benchmarks (MMBench, InfoVQA, etc.). How does the system perform in real-world usage with variable user interaction patterns, thermal throttling over extended use, and battery degradation over time?

### Soundness
2

### Presentation
2

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
The paper introduces NANOMIND, a hardware–software co-design framework targeting efficient on-device inference of Large Multimodal Models (LMMs) on battery-powered systems. The authors propose decomposing multimodal models into modular components (e.g., vision encoders, LLM decoders) and dynamically mapping each module to the optimal heterogeneous accelerator (GPU, NPU, CPU) under a unified memory architecture.

### Strengths
- On-device LMM/VLM execution is increasingly important for privacy, latency, and offline use. The focus on battery-powered compact devices differentiates this work from existing edge-accelerator papers.

- The modular execution model, accelerator-aware scheduling, and token-aware buffer manager offer practical and technical novelty, especially under UMA constraints.

- The combination of FP16 encoders with W4A16 LLMs, fused dequant-GEMM OpenCL kernels, and linear attention demonstrates solid engineering toward performance and power efficiency.

### Weaknesses
- Baseline gaps: lacks rigorous, same-hardware comparisons against state-of-the-art mobile stacks such as MLC-LLM, llm.npu, and PowerInfer-2, weakening external validity.

- Incomplete utility evidence: limited end-task accuracy and qualitative results for multimodal workloads leave the user-perceived quality underexplored.

- Scalability uncertainties: generalization to larger models and to NPUs with static-shape constraints is not demonstrated, and ablations isolating hardware choices are minimal.

### Questions
- How does the scheduling handle rapid mode switching (camera and audio streams) under burst workloads without degrading responsiveness?

- Can the system support multi-image or temporal encoder models given the static-shape NPU limitations?

- What is the quantitative energy-latency trade-off curve of three power modes over long-running workflows?

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
3

### Summary
This paper proposes NANOMIND, a novel software–hardware co-design framework for enabling efficient on-device inference of large multimodal models (LMMs) on resource-constrained, battery-powered devices. It leverages the modularity of LMMs by decomposing them into “bricks” and dynamically offloading modules to optimal compute units (NPU/GPU/CPU). The paper introduces a custom inference pipeline, quantization strategies, token-aware buffer management, and custom hardware design, achieving notable energy and memory efficiency.

### Strengths
- The software–hardware co-design for efficient on-device inference of LMMs is very interesting and impactful.

- Clear research motivation: tackling latency and energy inefficiency of LMMs on edge devices.

- Practical systems-level contribution, combining model decomposition, dynamic workload scheduling, and embedded hardware design.

### Weaknesses
- While the proposed deployment strategy is evaluated on the authors’ custom SoC, the paper does not provide validation on other commercial or widely available SoCs.

- In Figure 6, Throughput (tokens/s) and Latency appear to vary according to the device’s power state, which is controlled by the proposed Power-Efficiency Strategy. If I understand correctly, the battery level determines whether the system operates in a high-performance parallel mode or in the low-power “On-Demand Cascade Inference” mode. However, the paper does not clearly explain how throughput and latency are measured or computed under different power levels.

- While Figure 7 compares different quantization strategies, the paper does not include an accuracy comparison between the proposed system and existing implementations. Such a comparison is necessary to clearly demonstrate the performance advantage of the proposed design.

### Questions
- A key architectural component is the use of a token-aware ring buffer to facilitate zero-copy data flow between heterogeneous compute units (e.g., NPU and GPU). While this design significantly optimizes memory bandwidth and latency, how does it manage the Key–Value (KV) caches?

**Minor Comments**:

- The references in the current version of the paper are incomplete in formatting and lack hyperlinking.

- The information presented in the paragraph starting at line 354 overlaps considerably with the earlier section “Token-Aware Buffer Management” (beginning at line 327). The two sections convey similar ideas and could be merged into a single, more concise paragraph to improve the paper’s flow and avoid redundancy.

I am not deeply familiar with prior work on on-device inference frameworks, so I am unsure whether other closely related studies exist. I would be happy to discuss this further during the rebuttal and consider increasing the score accordingly.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes NanoMind, a hardware-software co-design framework for efficient on-device inference of LMMs. The key idea is to decompose LMMs into modular components and dynamically offloads them to the most suitable heterogeneous accelerators including NPU, GPU, and CPUs. This paper proposes a token-aware buffer manager for zero-copy data transfer and a dynamic power management strategy. The framework is implemented on a custom-designed device based on the RK3566 SoC. Evaluations demonstrate reduced memory usage, lower latency, and improved power efficiency.

### Strengths
- The problem is well-motivated, addressing the critical challenge of efficient LMM deployment on resource-constrained edge devices.

- The paper conducts practical system implementation and evaluation on real hardware.

### Weaknesses
- The paper claims to be a system-algorithm co-design method; however, the algorithm-level innovations appear to be minor, primarily leveraging existing quantization and model decomposition ideas rather than introducing novel algorithmic contributions.

- The framework's adaptability to different devices with varying computational resource budgets is not explored. The experiments are conducted only on RK3566 SoC, limiting the generalizability of the proposed framework.

- Lack of strong quantitative metrics demonstrating the method effectiveness. As seen in Figures 5, 6, and 7, the improvements in memory usage, latency, and accuracy over baselines appear marginal, lacking of quantitative evidence for a substantial performance improvement.

### Questions
- How can the proposed framework be adapted to other edge SoCs with different accelerator configurations (different NPU/GPU capabilities)? What modifications or adjustments would be required?

- Can you provide more compelling quantitative evidence or statistical analysis to demonstrate that NanoMind offers a significant improvement in key metrics like latency reduction, memory efficiency, or accuracy preservation?

### Soundness
3

### Presentation
2

### Contribution
3
