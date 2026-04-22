# SF-PE: A Synergistic Fusion of Absolute and Relative Positional Encoding for Spiking Transformers

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Positional signals in spiking neural networks (SNNs) suffer distortion due to spike binarization and the nonlinear dynamics of Leaky Integrate-and-Fire (LIF) neurons, which compromises self-attention mechanisms. We introduce Spiking-RoPE, a spiking-friendly relative rotary positional encoding that applies two-dimensional spatiotemporal position-dependent rotations to queries/keys prior to
binarization, ensuring that relative phase kernels are preserved in statistical expectation under LIF dynamics while maintaining content integrity. Building on this core, we propose Spiking Fused-PE (SF-PE), a scheme that fuses absolute CPG-based spikes with Spiking-RoPE. The resulting attention score decomposes into complementary row/column (absolute) and diagonal (relative) structures, thereby expanding the representable function space. We validate our method across two diverse domains (time-series forecasting and text classification) on Spikformer, Spike-driven Transformer, and QKFormer backbones. SF-PE consistently improves accuracy and enhances length extrapolation capabilities. Ablations on rotation bases and 1D vs. 2D variants support the design. These results establish rotary encoding as an effective, spiking-friendly relative PE for SNNs and demonstrate that fusing absolute and relative signals yields synergistic benefits under spiking constraints. Code: https://anonymous.4open.science/r/SNN-RoPE-F6DE.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies how spike binarization and LIF dynamics distort positional signals in spiking Transformers, weakening self-attention. 
It introduces **Spiking-RoPE**, which applies rotary positional encoding to queries/keys *before* binarization so relative phase information is preserved under LIF. 
A **2-D Spiking-RoPE** decouples sequence and time rotations to explicitly model spatiotemporal relations. 
To combine absolute and relative cues, the authors fuse CPG-based absolute spikes with Spiking-RoPE into **Spiking Fused-PE (SF-PE)**, yielding complementary row/column vs. diagonal attention patterns. 
A theoretical analysis supports pre-spike rotations as intrinsically compatible with LIF, and experiments across time-series forecasting and text classification on multiple spiking backbones show consistent improvements and better length extrapolation. 
Comprehensive ablations on rotation bases and 1-D vs. 2-D variants corroborate the design’s robustness.

### Strengths
The paper redesigns RoPE specifically for SNNs by applying it pre-spike, extends it to a 2-D spatiotemporal variant, and fuses absolute CPG-PE with relative Spiking-RoPE into SF-PE to capture complementary row/column and diagonal structures. It provides a statistical-expectation analysis showing phase preservation through LIF, includes thorough ablations (1-D vs 2-D; RoPE base), and reports solid results across time-series and text on multiple spiking backbones. The method is clearly motivated, the pre-spike rotation pipeline is well articulated with equations and design rationale, and assumptions are explicitly stated.

### Weaknesses
1) The paper motivates Spiking-RoPE as a **pre-spike** operation (before LIF binarization) to justify “phase preservation under LIF.” In the released code (`spikformer_cpg_rope.py`, self-attention forward), Q/K pass through `self.q_lif` / `self.k_lif` **first**, and only then `apply_spiking_rotary_pos_emb(...)` is applied—i.e., RoPE is **post-spike**. Under this code path, the pre-spike theoretical claim does not hold as implemented.

2) The manuscript emphasizes 2D spatiotemporal rotation and fusing absolute CPG with Q/K inputs. The public model instantiates a 1D rotary embedding (`RotaryEmbedding1DSpatial`) and injects CPG via `CPGLinear` on the encoder path rather than at Q/K. If this is intended to be equivalent, that needs justification; otherwise it should be documented as an implementation deviation or the stated 2D variant should be released.

### Questions
Q1. The paper argues for **pre-spike** RoPE; the released code implements **post-spike**. Which path produced the reported results? If post-spike was used, please revise the theory accordingly; if pre-spike was used, provide the corresponding implementation and report a direct pre- vs post-spike comparison.

Q2. Current code uses `RotaryEmbedding1DSpatial` and fuses CPG via `CPGLinear` on the encoder path rather than at Q/K. (i) Are these implementations theoretically equivalent to the paper’s formulas? If not, please mark as a deviation. (ii) Release and evaluate the full **2D** variant and the **Q/K-side fusion** described in the manuscript, with side-by-side results.

### Soundness
3

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
5

### Summary
This paper introduces Spiking Fused-Positional Encoding (SF-PE) for spiking transformers, which integrates absolute positional encoding (CPG-PE) and relative positional encoding (Spiking-RoPE) to resolve the distortion of positional signals in spiking neural networks (SNNs)—a problem caused by spike binarization and the nonlinear dynamics of Leaky Integrate-and-Fire (LIF) neurons. The authors provide theoretical proof that Spiking-RoPE preserves relative phase kernels in statistical expectation under LIF dynamics, and validate SF-PE across two tasks (time-series forecasting and text classification) and three spiking backbones (Spikformer, SDT-V1, QKFormer). Results show consistent accuracy improvements and enhanced length extrapolation, with ablations supporting the design of 2D Spiking-RoPE and rotation bases.

### Strengths
1. It rigorously proves that Spiking-RoPE preserves relative phase kernels in statistical expectation under LIF dynamics, addressing the lack of theoretical analysis for positional information preservation in existing SNN transformers (Gap 1).
2. The 2D Spiking-RoPE explicitly models spatiotemporal relationships by decoupling sequence and time axes, solving the limitation of most PEs treating position as one-dimensional (Gap 3), and ablation experiments confirm its superiority over 1D variants.

### Weaknesses
1. I checked the code and found that the authors merely fused the RoPE code crudely into Spiking Self-Attention without analyzing the properties of spikes; theoretically, RoPE cannot work on binary matrices, so I am skeptical about the performance improvements claimed by the authors.
2. The theoretical analysis of Spiking-RoPE’s phase preservation under LIF dynamics relies on the strong assumption that the firing probability function of LIF neurons operates almost linearly, yet the paper provides no verification of the validity range or error bounds of this assumption.
3. In the text classification experiments, the paper shows that neither CPG-PE nor SF-PE improves performance on the RTE task, but it fails to analyze the reasons for this task-specific ineffectiveness (e.g., whether it is related to spatiotemporal modeling or relative position encoding).
4. The paper claims that SF-PE enhances length extrapolation capabilities, but it only mentions the analysis in Appendix D without presenting key results (e.g., performance degradation trends on sequences longer than training lengths) in the main text.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents a novel method that combines Spiking-RoPE with CPG positional encoding in spiking neural networks, resulting in a fused position encoding mechanism called Spiking Fused-PE. The approach first injects absolute positional information from CPG-PE into token embeddings via linear projection, then applies a two-dimensional Spiking-RoPE rotation to encode relative positions along both spatial (sequence) and temporal dimensions. Following this 2D rotation, the attention computation is decomposed into amplitude terms carrying absolute information and trigonometric terms encoding relative positional differences. The goal of this design is to jointly capture absolute and relative positions across space and time, yielding richer and more structured spatiotemporal representations. Experiments demonstrate the effectiveness of Spiking Fused-PE through quantitative performance gains and ablation studies.

### Strengths
This paper extends the RoPE mechanism to the temporal dimension of spiking neural networks and integrates it with CPG absolute positional encoding to form a new positional encoding scheme. This combination is somewhat novel. Additionally, the paper provides detailed mathematical derivations, and includes visual illustrations of the Spiking-RoPE computation process in the appendix, which together enhance clarity and facilitate understanding.

### Weaknesses
1) The paper’s overall innovation appears somewhat limited. It mainly extends RoPE to the time-step dimension and then merges it with CPG positional encoding, which results in a method that integrates existing ideas rather than introducing a fundamentally new mechanism. Although the proposed Spiking Fused-PE achieves good performance, it lacks deeper methodological novelty or theoretical breakthroughs. 

2) The paper does not explore other potential fusion strategies between CPG and Spiking-RoPE. For instance, it remains unclear whether the CPG component must be injected only at the input embedding stage or if it could be integrated after the 2D RoPE rotation or within later layers, which might lead to different positional interaction effects.

3) The experimental section lacks sufficient discussion of computational cost. Specifically, Table 2 reports identical inference times across all compared methods, including SF-PE, which theoretically requires additional computation for the 2D RoPE operation. This inconsistency raises concerns about the rigor of the experimental setup and whether the runtime measurements were properly controlled or averaged.

### Questions
1) Could the authors explore whether CPG can be applied at other stages of the model. For example, after the 2D RoPE rotation or within later processing blocks rather than being limited to the input x? Would such variations affect the encoding of absolute and relative positional information?

2) In Table 2, why are the inference times completely identical for all three methods? Since SF-PE introduces additional 2D RoPE computations on top of CPG-PE, one would expect a measurable increase in inference time. How were these runtimes obtained and were they averaged over multiple runs, and was the hardware or implementation optimized in a way that masked the computational overhead?

### Soundness
2

### Presentation
3

### Contribution
2
