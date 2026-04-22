# Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Prevailing LLM (Large Language Model) serving engines employ expert parallelism (EP) to implement multi-device inference of massive Mixture-of-Experts (MoE) models. However, the efficiency of expert parallel inference is largely bounded by inter-device communication, as EP embraces expensive all-to-all collectives to route tokens to the remote experts if not collocating on the same GPU/NPU device. Nevertheless, state-of-the-art schemes treat expert device-placement and request (or token) device-scheduling as separate concerns, triggering excessive communication between them and compromising inference efficiency

This paper proposes Semantic Parallelism, a novel parallelism paradigm that minimizes the steep communication costs in EP-centric MoE serving via model-data collaborative scheduling. We implement Semantic Parallelism in a framework called Sem-MoE. Sem-MoE maximally collocates experts and their activating tokens onto the same device using proactively modeled activation likelihood between them and introduces three key techniques: (1) Offline model scheduling, which preliminarily clusters and collocates experts onto devices based on their co-activation tendencies for certain classes of input. (2) Online inter-request data scheduling for Attention-DP setups, which proactively rebatches incoming requests onto the device that hosts experts most likely and frequently activated by the corresponding requests. (3) Online intra-request data scheduling for Attention-TP setups, which seamlessly fuses a token reshuffling procedure into the original inference pipeline and proactively reschedules tokens to devices to reduce dispersed remote routing. We build Sem-MoE into a prevailing LLM serving engine SGLANG. Experiments show our collaborative scheduling approach can effectively reduce the all-to-all communication volume in EP and achieve superior inference throughput compared to existing solutions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Sem-MoE, a mixture-of-experts (MoE) scheduling framework that leverages offline profiling of token–expert activations to optimize communication and load balance. The method first computes a token–expert activation matrix $$C_{p,jk}$$ from model logs, then formulates an integer linear program (ILP) minimizing cross-expert communication and balancing expert loads. The ILP solution is later applied at runtime (§3.4) to pre-group experts and improve all-to-all efficiency. Experiments on an 8×A100 NVLink setup report a 2.78× throughput gain and 25% latency reduction compared with FasterMoE.

 The overall mathematical formulation and implementation are coherent and elegant, yet the theoretical and empirical justifications for using offline token–expert frequencies as stable, context-independent features are insufficient. Experimental validation focuses mainly on efficiency and omits semantic and multi-node evaluation.

### Strengths
(1) The ILP modeling in §3.3 is mathematically elegant and well aligned with the MoE scheduling problem. It directly encodes both communication minimization and load balancing as discrete constraints. The objective function depends explicitly on $$C_{p,jk}$$, linking statistical observations to optimization variables. The derivation is concise and solvable, and the implementation (§3.4) confirms that this optimization is executed in practice.

(2) The pipeline from offline profiling to runtime integration is clear and systematic. The ILP-generated token–expert and expert–expert tables are concretely used for expert grouping, confirming that the method bridges theory and system.

(3) The experiments (§4) show consistent and interpretable performance gains, demonstrating real engineering improvement, not mere theoretical potential.

### Weaknesses
(1) The ILP depends on the token–expert frequency matrix $$C_{p,jk}$$, which the authors assume to be stable and context-independent (§3.2). However, this assumption lacks theoretical grounding or rigorous validation. Appendix A reports stability (96.3%) and F1 (78.8%) computed on a single corpus (ShareGPT) and layer, but provides no variance analysis across datasets, seeds, or layers. Since the gating probability $$P(E_k|x_j,h_{L,j})$$ inherently depends on contextual hidden states $$h_{L,j}$$, averaging over contexts removes critical variance information. Moreover, although the authors describe this as an “offline distribution,” the collection of $$C_{p,jk}$$ itself is derived from attention representations obtained during **normal Transformer model inference on full sentences**. Thus, each token’s recorded expert activation already embeds contextual dependencies from its neighboring tokens. This makes the so-called offline mapping not truly context-independent but an average over context-conditioned activations. Consequently, the statistical input to the ILP may not reflect real routing behavior when context distributions shift.

(2) Figures 6(b)(c) mainly show high-frequency function tokens such as “the”, “and”, “you”. Their activation stability is trivial and does **not** generalize to some specific semantic or rare tokens, which dominate real workloads. Furthermore, Figure 6 shows that activation entropy increases in deeper layers, yet the paper interprets this as “uniform expert reuse.” Uniformity here indicates uncertainty, not stability, thus contradicting the claimed “context independence.” The analysis therefore captures lexical regularity rather than semantic invariance.

(3) The communication cost term in the ILP objective aggregates token-level co-activation frequencies into expert–expert communication weights (§3.3), but no strong empirical correlation between these weights and actual communication traffic is shown. The paper lacks ablation studies testing ILP sensitivity to noise or sampling bias in $$C_{p,jk}$$.

(4) The reported results measure only throughput, latency, and load variance, **lacking evaluation** of model quality (perplexity, accuracy, downstream metrics). So it is unclear whether the offline routing preserves semantic fidelity or harms inference correctness.

(5) All experiments are performed on a single 8×A100 NVLink node (§4). The paper suggests that the method could achieve larger gains under slower interconnects (PCIe or InfiniBand), but this claim remains untested. Including cross-node or heterogeneous-network experiments would strengthen the scalability argument.

### Questions
(1) The paper defines “context-independent routing” (§3.2) based on the observed stability of token–expert activation frequencies computed from full-sentence attention representations. However, the gating function $$G_L(h_{L,j})$$ inherently depends on contextual hidden states. How do the authors justify that the offline probability $$P(E_k|x_j)$$ estimated under contextualized inputs approximates the true context-independent term $$P(E_k|x_j,h_{L,j})$$? In particular, what does “semantic independence” mean in this formulation—is it statistical stability of frequent tokens, or an actual invariance of routing under different semantic contexts?

(2) The authors report stability (96.3%) and F1 (78.8%) on ShareGPT, but the analysis seems limited to a single layer and dataset. Could they clarify whether similar stability holds across different layers, seeds, or corpora?

(3) Figures 6(b)(c) analyze only common function tokens. The text briefly mentions that rare tokens are “more uniformly distributed,” but no quantitative results are given. Could the authors elaborate on the behavior of low-frequency or semantic tokens?

(4) The author mentioned “the benefit will be even larger under slower interconnects (PCIe or InfiniBand)” in §4 and suggests that greater improvements would be observed under slower interconnects (PCIe or InfiniBand), yet this claim is untested. Have the authors evaluated or simulated multi-node scenarios to confirm that the ILP-based grouping remains effective when communication latency and bandwidth vary?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Sem-MoE, a semantic-aware model–data collaborative scheduling framework that improves inference efficiency for large-scale MoE models. It jointly optimizes expert placement (offline model scheduling) and token/request routing (online data scheduling) to reduce all-to-all communication in expert parallelism.

### Strengths
1. This paper addresses a key bottleneck in MoE inference—the high communication overhead associated with expert parallelism.
2. It introduces a novel idea that aligns expert placement with token–expert affinities through semantic-aware co-scheduling.
3. The proposed framework demonstrates a comprehensive design, effectively supporting both DP and TP configurations.

### Weaknesses
1. The experimental setup is relatively limited, with evaluations conducted only on DeepSeek and Qwen; I recommend validating the conclusions across a broader range of models and datasets.

2. The reported performance gains in E2E latency appear limited, which limits the practical significance of the proposed scheme.

### Questions
1. Beyond SGLang and MoETuner, were any additional baselines included for comparison? The current comparison seems limited.
2. Do tasks or queries from different knowledge domains require extra training to maintain accurate token–expert mappings? 
3. How often must Algorithm 1 be executed in practice, and does this introduce notable or impractical computational overhead?
4. The paper mentions online inter-request data scheduling to proactively rebatch incoming requests. In the experiments, what stochastic process assumptions or models are used to generate the query arrivals?

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
5

### Summary
The Sem-MoE paper introduces an efficient method to enhance the performance of Mixture-of-Experts (MoE) models by reducing heavy communication between GPUs during inference. The authors do not treat model placement and data routing separately. Instead, they design a semantic-aware scheduler. This scheduler predicts which experts each token will likely activate. It aims to keep those experts and tokens on the same GPU. The method combines offline expert clustering with online request routing. It also uses customized communication kernels to minimize overhead. Experiments on models like DeepSeek-V2-Lite and Qwen3-30B-A3B show significant gains in throughput and latency, demonstrating the effectiveness of the approach. However, the method relies on static token–expert correlations, requires costly optimization, and may struggle to adapt to real-world, changing workloads. Overall, it is a smart and well-engineered step toward more efficient MoE inference.

### Strengths
1. The semantic-guided approach is intuitive and elegant. It leverages predictable token–expert correlations that were mostly ignored by other MoE schedulers.
2. Extensive experiments on DeepSeek-V2-Lite and Qwen3-30B-A3B show consistent and significant improvements in both throughput and latency.

### Weaknesses
1. The semantic prediction is based on historical activation statistics. In real-world, open-domain inference, input distributions can shift significantly. Such shifts may invalidate prior correlations and reduce performance.
2. The optimization process is largely offline and static. When the model or workload changes, Sem-MoE requires re-optimization, which can be computationally expensive.
3. Although the method is non-intrusive, Sem-MoE modifies the runtime scheduler and introduces custom collective operations. This makes integration into production-level inference frameworks more complex than suggested.
4. There is no analysis of energy consumption or communication bandwidth efficiency in the paper.
6.  The author should consider more types of MoE LLMs, such as Qwen3, Moonlight-A3B, gpt-oss-120b, and gpt-oss-20b, should be considered.

### Questions
Summarized in Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
