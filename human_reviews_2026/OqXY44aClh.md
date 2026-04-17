# Efficient Unified Multimodal Understanding and Generation with Gated Hybrid Attention

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Unified multimodal learning requires attention mechanisms that are both efficient and expressive. 
Softmax attention provides strong modeling capacity but suffers from quadratic complexity, while 
linear attention achieves near-linear efficiency at the cost of weaker expressivity. We identify two 
major expressivity challenges in efficient unified multimodal models: (\emph{i}) modality imbalance, 
where dominant signals suppress weaker modalities during fusion, and (\emph{ii}) loss of global context, 
as efficient variants tend to over-smooth long sequences. We propose \textbf{Gated Hybrid Attention (GHA)}, 
a multimodal-specialized operator that augments linear attention with (\emph{i}) a selective gating mechanism 
to balance modality contributions and stabilize training, and (\emph{ii}) agent-token softmax aggregation 
to restore adaptive global context while preserving near-linear complexity. To demonstrate generality, we validate GHA in two representative paradigms: autoregressive-only(AR-only)
and autoregressive+diffusion(AR+Diffusion). In both settings, GHA consistently improves multimodal alignment, 
long-context retention, and efficiency over comparable Transformer and efficient attention baselines. 
These cross-paradigm results highlight that GHA functions as a plug-and-play building block, offering 
a lightweight and extensible approach that is orthogonal to scaling trends and modality complexity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Summary:
This paper focuses on efficient unified multimodal understanding and generation with gated hybrid attention. Specifically, Standard Softmax attention offers strong expressivity but suffers from $O(N^2)$ quadratic complexity, creating a bottleneck for long sequences. In comparison, efficient alternatives like Linear Attention achieve near-linear complexity but at the cost of less modeling capacity. To address this, the authors propose Gated Hybrid Attention (GHA), which is a novel attention operator that enhances linear attention with two key components: 1) A selective gating mechanism to balance the contributions of different modalities and stabilize training, addressing the "modality imbalance" problem; 2) a softmax-based agent-token bottleneck. To ensure the theoretical efficiency of GHA translates into practice, the authors further provide a hardware-optimized implementation based on FlashAttention style chunking and a custom Triton kernel. The authors implement GHA as a "plug-and-play" module in a unified model. They validate this model in two representative paradigms including Autoregressive-only (AR-only) and Autoregressive+Diffusion (AR+Diffusion). Experimental results show that, compared to strong baselines of similar scale (like the Transformer-based and Mamba-based approaches), OmniGHA achieves competitive performance on both multimodal understanding (e.g., POPE, VQAv2) and visual generation (e.g., FID) benchmarks.

### Strengths
Strengths:
- The paper is well organized and easy to follow;
- The paper conducts comprehensive experimental evaluation, i.e., evaluating GHA in both AR-only and AR+Diffusion frameworks, which shows the potential of GHA to be served as a general-purpose and plug-and-play module for various unified architectures;

### Weaknesses
Weakness & Questions:
- In Section 4.1(3), the gate $G$ is described as "a learnable, data-dependent gating matrix". However, Equation 5, $O=Q^{\prime}(G\odot(K^{\prime}V))$, it may suggest that it is a learnable parameter matrix (i.e., nn.Parameter). To avoid confusion, maybe the gate $G$ can be written as $G = f(x)$. This would be helpful for understanding the mechanism;
- Training of GHA. The three-phase training curriculum (Representation Warm-up, Long-context Adaptation, Joint Alignment) is non-trivial. The ablation in Table 8 confirms that skipping the first two phases leads to a significant performance drop. This suggests GHA may be highly sensitive to its training curriculum, which somewhat reduces its simplicity as a "plug-and-play" module, since we do need to carefully design the training steps;
- In the main results (Table 2), the OmniGHA (AR) model achieves a very strong FID of 5.12. However, in the ablation studies (Table 7 and Table 8), the reported FID for the full GHA configuration is 8.8 and 9.2, respectively. While it is likely due to a simplified setup for the ablations, this discrepancy is not explicitly explained and may make readers confused;

I would like to see the author rebuttal in terms of the weakness & questions part.

### Questions
Please see Weakness & Questions.

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
This paper proposes Gated Hybrid Attention (GHA), which aims to tackle the trade-off between efficiency and expressiveness in unified multimodal learning. The method highlights the joint use of a key-value gating module and a softmax-based agent-token bottleneck to yield a near-linear complexity and long-context attention mechanism. The paper validates the effectiveness of GHA under two paradigms, autoregressive-only (AR-only) and autoregressive+diffusion (AR+Diffusion). Experiments show that it outperforms Transformer and other efficient attention baselines in terms of multimodal alignment, long-context retention, and generation quality.

### Strengths
+ The paper is well-written
+ The efficiency is impressive
+ The design demonstrates a strong focus on practical implementation by delivering a hardware-optimized solution using Triton

### Weaknesses
- The analysis of the key hyperparameter 'n' is missing: The attention complexity is O(Nnd), where 'n' is the number of introduced agent tokens. The paper does not discuss how this hyperparameter was chosen, its sensitivity on model performance and inference speed, or its impact on expressiveness.

- Insufficient justification for 'mitigating modality imbalance': A key motivation for GHA is that its gating mechanism can 'mitigate modality imbalance'. However, the paper only supports this indirectly via the ablation study in Table 5 (where 'Linear + Gate' performs better than 'Linear' on metrics like POPE/MME-P). This only shows that gating improves overall performance, but provides no direct evidence that it is actually balancing the contributions of different modalities.

- Missing recent baselines (Table 1): In Section 5.2, the authors claim OmniGHA outperforms "Understanding-only" systems. However, some of recent works such as Qwen2.5-VL-3B, InternVL3 are not included in the Table 1.

- Missing critical hyperparameter: GHA's efficiency claim of near-linear complexity O(Nnd) hinges on the number of agent tokens (n), yet the paper fails to specify n in either the main text or appendix. If n is large (e.g., n=1024), the O(Nnd) cost may not be as low as implied. Without this crucial detail, readers cannot assess the true computational cost or reproduce the efficiency results in Table 3, making the claims unverifiable and incomplete.


- Insufficient analysis of the AR+Diffusion paradigm: The paper shows an OmniGHA (AR+Diffusion) variant (see Table 1 and Table 2) that performs even slightly better than the AR-only variant. However, the paper merely mentions this, attributing it to "the additional conditioning projector acting as a form of representation regularization". This analysis is too shallow. The AR+Diffusion model is much slower in generation speed (1.95 images/s vs 5.26 images/s) as shown in Table 4, and the authors do not adequately discuss this performance/efficiency trade-off or deeply analyze how GHA functions specifically within this more complex paradigm.

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
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Gated Hybrid Attention (GHA), a multimodal attention mechanism combining linear attention with selective gating on key-value pathways and agent-token softmax aggregation. The paper applies GHA to address modality imbalance and loss of global context while maintaining efficiency. GHA is validated in both autoregressive-only and autoregressive + diffusion paradigms through OmniGHA, a small-scale model trained from scratch. Results show improvements over existing baselines on understanding and generation benchmarks, with significant inference speedups over Transformer baselines.

### Strengths
- The exposition is clear and addresses a well-motivated problem of balancing efficiency and expressivity in unified multimodal models.
- The approach demonstrates strong empirical results across understanding and generation tasks, supplemented with hardware-aware implementation that achieves measurable wall-clock speedups.
- Ablations properly isolate contributions of gating and agent-token components, validating that both contribute to performance gains.

### Weaknesses
- A major shortcoming is the lack of pure Transformer baseline trained from scratch with the same pipeline. Most compared baselines use pretrained LLMs while OmniGHA trains from scratch, confounding the comparison. Observed gains could stem from the proposed attention mechanism, or any of training from scratch versus using pretrained models, the training curriculum, encoder choices, or hyperparameter tuning. Without a controlled Transformer baseline, the key claim about the architectural contribution of GHA cannot be verified.

- The proposed framework is limited in novelty as components already exist in prior work. The contribution is primarily combining existing techniques rather than fundamental innovation. In this regard, the lack of justification for why this combination is optimal or any analysis of how it specifically helps for multimodal learning, besides just performance gains, is a shortcoming.

- Small-scale experiments raise generalization concerns, as modern competitive models operate at much larger scales. While limited scale alone could be overlooked for an architectural contribution, combined with the missing controlled baseline and insufficient analysis, it significantly weakens the empirical validation of the claimed contributions.

Overall: While this paper achieves good empirical results, the lack of a controlled Transformer baseline makes it impossible to isolate GHA's contribution from confounding factors. The limited novelty, small scale, and insufficient analysis further weaken the contribution. If GHA's potential can be demonstrated more conclusively, the reviewer is willing to raise their evaluation of the paper.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
