# UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
While Mixture of Experts (MoE) models achieve remarkable efficiency by activating only subsets of parameters, they suffer from high memory access costs during inference. Memory-layer architectures offer an appealing alternative with very few memory access, but previous attempts like UltraMem have only matched the performance of 2-expert MoE models, falling significantly short of state-of-the-art 8-expert configurations. We present UltraMemV2, a redesigned memory-layer architecture that closes this performance gap. Our approach introduces five key improvements: integrating memory layers into every transformer block, simplifying value expansion with single linear projections, adopting FFN-based value processing from PEER, implementing principled parameter initialization, and rebalancing memory-to-FFN computation ratios. Through extensive evaluation, we demonstrate that UltraMemV2 achieves performance parity with 8-expert MoE models under same computation and parameters but significantly low memory access. Notably, UltraMemV2 shows superior performance on memory-intensive tasks, with improvements of +1.6 points on long-context memorization, +6.2 points on multi-round memorization, and +7.9 points on in-context learning. We validate our approach at scale with models up to 2.5B activated parameters from 120B total parameters, and establish that activation density has greater impact on performance than total sparse parameter count. Our work brings memory-layer architectures to performance parity with state-of-the-art MoE models, presenting a compelling alternative for efficient sparse computation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
UltraMemV2 refines the Memory Layer architecture as a new sparse alternative to MoE models.
While the original UltraMem only reached the performance level of a 2 active expert MoE, UltraMemV2 achieves performance comparable to an 8 active expert MoE, marking a significant advancement.
The authors introduce several key improvements, including a memory layer in each Transformer block, a simplified implicit value expansion (IVE), a PEER-based feedforward mechanism, improved initialization for stable training, and optimized compute ratios between memory to  FFN.
With these enhancements, UltraMemV2 shows notable gains (+6 to +8 points) on long-context, multi-step reasoning, and memory-intensive tasks, while efficient memory access comparable to existing MoE models.

### Strengths
1. Considering that MoE architectures have become a de facto standard component in LLM training, the direction of this work is highly meaningful. Improving the cost efficiency and performance of such sparse architectures is an important and timely.

2. The reported performance is very promising, matching the top-k=8 MoE configuration that is commonly used in recent LLM models.

3. The experiments are conducted at a large scale up to 120B parameters with 2.5B active and trained on 4.4T tokens which makes the results convincing and demonstrates the method’s scalability.

### Weaknesses
1. It would be valuable to include more discussion or quantitative results about GPU hours, training/inference latency, and throughput, especially compared to existing MoE models.
These metrics would strengthen the claim that UltraMemV2 offers practical efficiency gains.

2. While hyperparameter search is generally required for LLM training, the proposed model appears to be more sensitive to hyperparameter choices, such as initialization and learning-rate scheduling.

### Questions
1. It is mentioned that UltraMemV2 cannot easily achieve higher sparsity. Did the authors analyze whether its sparsity scaling behavior differs from that of MoE models?

### Soundness
3

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
3

### Summary
The paper proposed a sparse memory architecture by tweaking memory layer architecture to match the performance with MOEs with 8 experts. The paper does a good job of providing a Comprehensive analysis and detailed ablation studies. The paper also discusses scalability, though it’s unclear if the same scaling law of LLMs holds with trainable memory parameters. 
It improves inference efficiency over prior work, by  simplifying value expansion with single linear projections, demonstrating that parameter efficiency of non-shared linear layers is actually not high. This paper also overcomes the limitation of number of memory layers, where previous work has shown degradation of performance if the number of memory layers are too high.

### Strengths
1. Good ablation for number of layers and  overcomes the limitation of number of memory layers
2. Matched performence with MOEs with 8 experts
3. Uses strong benchmarks and evaluation
4. Simplifies the  value expansion, making inference more efficient

### Weaknesses
Overall the contribution is light, the paper aims to bridge the gap(in performance) between MOEs and Memory layer architectures. In terms of scientific novelty, some of the approaches seem incremental and this approach seems to combine multiple incremental tweaks to achieve performance improvements over baseline. For example, “Memory Layer at Scale”(Berges, 2024) paper demonstrated that  was multiple memory layers increase performance significantly over having a single layer(In their case, performance degraded going beyond 3 layers).  Another example is adoption of simplified value expansion with a small tweak of single linear projection.

### Questions
1. A more rigorous scaling law analysis and discussion around scaling trends are required if this is proposed as an alternative architecture to MoEs
2. Can this approach be introduce during mid-training or post training and achieve good performance. Any discussion around the performance gap with CT would be great. 
3. There are other parametric memory work, such as memory Layers at scale. It would be good to compare the results with such alternative approaches.  
4. It would be good to share results on how high quality RAG combines with this approach and how does this approach compares with RAG for long context tasks

Minor comments:
1. Discussion about how can this be combined with MOEs

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents UltraMemV2, a memory-layer architecture intended to close the performance gap between memory-layer sparse approaches and Mixture-of-Experts models. The authors introduce five main changes: placing memory layers in every transformer block, simplifying implicit value expansion to a single linear projection, using FFN-based value processing inspired by PEER, a new initialization for the memory layer, and rebalancing memory and FFN computation proportions. They evaluate UltraMemV2 across proprietary and open benchmarks and report that UltraMemV2 reaches parity with 8-expert MoE under matched compute/parameters while requiring much lower memory access. They highlight particularly strong gains on memory-heavy tasks and validate scaling up 120B parameter models.

### Strengths
Strong architectural contributions: the five design changes proposed by the authors are all justified through ablations and contribute to improved model performance

Strong empirical evaluation: multiple model scales up to 120B and a diverse selection of benchmarks make the authors claims very convincing.

Initialization analysis: the paper contributes a new initialization scheme to stabilize training of the memory layer, which addresses a common failure mode for large sparse modules.

Practical use: UltraMemV2 is a compelling architecture for deploying models under memory bandwidth constraints because of the relatively lower memory accesses.

### Weaknesses
The paper motivates UltraMemV2 with lower memory access and inference cost, but it would be more convincing to see latency and bandwidth comparisons vs. traditional MoE models

Proprietary data: this might be unavoidable but the proprietary nature of the benchmarks and data limits the reproducibility of the methods in this paper

The UltraMemV2 model has significantly worse benchmark performance on multi-hop reasoning. The paper would be improved if the authors investigated this further and demonstrated through other benchmarks whether UltraMemV2 has worse overall reasoning abilities or if it is specific to this benchmark.

### Questions
The method lags behind early in training as mentioned by the authors. Do you have more details and possible intuitions why UltraMemV2 is slower to train initially? And what does "early" mean in general - how much data does it need to catch up?

Is there a reason for the drop in multi hop reasoning? Is this reflective of the model's overall reasoning capabilities on downstream tasks?

Could you quantify the inference cost improvements that you alluded to eg the reduction in memory access?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a memory‑layer architecture intended as an alternative to MoEs, which is an extension of UltraMem. The novelty with respect to the origianl UltraMem are, adding a memory layer to every Transformer block, simplifying the implicit value expansion (IVE) to a single linear projector, replacing value embeddings with an FFN with 1-dimensional inner layer, introducing a variance‑matching initialization, and rebalancing compute between memory and FFN. They claim parity with 8‑expert MoEs at similar active parameters/compute and advantages on long‑context tasks.

### Strengths
The paper clearly describes its position relative to MoE, PKM/UltraMem, and PEER.

Experiments show that increasing the number of UltraMemV2 layers improves downstream accuracy even when validation loss plateaus.

The proprietary long‑context suite shows non‑trivial gains of 6.2 on multi‑round memorizing and 7.9 on in‑context learning.

The paper is explicit that UltraMemV2 underperforms early in training and benefits from continued training, and also notes dependence on per‑block placement.

### Weaknesses
The paper asserts matching compute and parameters, but does not report KV‑cache costs, routing FLOPs, or memory traffic for both MoE and UltraMemV2.

The claim that this work is the first memory layer to match 8‑expert MoE is not accurate in light of the Memory Layers at Scale paper [https://arxiv.org/abs/2412.09764].

### Questions
How are KV‑cache footprint and router/TDQKR indexing costs accounted for in the iso‑compute comparisons in Tables 1–3?

Could you report token/s and latency vs. batch size, plus HBM read/write estimates, for representative model sizes?

### Soundness
3

### Presentation
3

### Contribution
2
