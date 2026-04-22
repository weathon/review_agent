# SAISA: Towards Multimodal Large Language Models with Both Training and Inference Efficiency

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Multimodal Large Language Models (MLLMs) mainly fall into two architectures, each involving a trade-off between training and inference efficiency: embedding space alignment (e.g. LLaVA) is inefficient during inference, while cross-attention space alignment (e.g. Flamingo) is inefficient in training. In this paper, we compare these two architectures and identify key factors for building efficient MLLMs. A primary difference between them lies in how attention is applied to visual tokens, particularly in their interactions with each other. To investigate whether attention among visual tokens is necessary, we propose a new self-attention mechanism, NAAViT (No Attention Among Visual Tokens), which eliminates this type of attention. Our pilot experiment on LLaVA-1.5 shows that attention among visual tokens is highly redundant. Based on these insights, we introduce SAISA (Self-Attention Input Space Alignment), a novel architecture that enhances both training and inference efficiency. SAISA directly aligns visual features with the input spaces of NAAViT self-attention blocks, reducing computational overhead in both self-attention blocks and feed-forward networks (FFNs). Compared with the LLaVA-1.5 architecture, SAISA reduces the inference FLOPs by 66% and the training budget by 26%, while achieving superior performance in terms of accuracy. Comprehensive ablation studies further validate the effectiveness of SAISA across various LLMs and visual encoders. The code and models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an approach to improve the training and inference efficiency of MLLMs by exploiting redundancy among visual tokens. The key idea is to eliminate the attention computations and FFNs operations related to visual token. The proposed architecture demonstrates significant computational savings with competitive performance to baselines. Experiments across multiple benchmarks, vision encoders, and LLM backbones validate the approach.

### Strengths
* The paper tackles an important problem: improving the efficiency of MLLMs, which are known to be computationally demanding.
* The writing is clear, well structured and the work is well motivated.
* The approach is validated across different architectures and setups

### Weaknesses
* The redundancy among visual tokens has been explored in prior work. The paper should better position itself in relation to existing studies that analyze or reduce visual redundancy e.g.  [1].

* The experiments primarily use the LLaVA setup, which lags behind more recent MLLM work. It is unclear how the proposed method compares with state-of-the-art (SOTA) models or whether it can be integrated effectively into stronger baselines.

* The paper does not discuss scaling behavior—how the proposed method performs when model or dataset size increases (also related to the previous point).

* Many modern VLM benchmarks (e.g., OCR, document understanding, PDF reasoning) require high-resolution images and fine-grained visual features. It remains unclear whether removing attention among visual tokens will negatively affect performance in these scenarios.

[1] "Implicit multimodal alignment: On the generalization of frozen llms to multimodal inputs." NeurIPS2024.

### Questions
Please refer to the weaknesess section.

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
4

### Summary
This paper addresses the well-known trade-off between training and inference efficiency in MLLMs. The authors categorize existing architectures into two types: embedding space alignment (e.g., LLaVA), which is training-efficient but inference-inefficient due to long visual token sequences, and cross-attention space alignment (e.g., Flamingo), which is inference-efficient but training-inefficient due to a large number of new parameters.

To tackle this, the paper first proposes NAAViT, a self-attention mechanism that eliminates the computationally expensive attention among visual tokens. A pilot study suggests this form of attention is redundant. Building on this, the authors introduce SAISA (Self-Attention Input Space Alignment), a novel architecture that employs NAAViT blocks and aligns visual features directly with the input spaces of each self-attention layer in the LLM. This approach avoids both the quadratic complexity of visual self-attention and the need to pass visual tokens through the FFNs. The experimental results demonstrate that SAISA significantly reduces inference FLOPs (by 66%) and training costs (by 26%) compared to its primary baseline, LLaVA-1.5, while achieving competitive or superior performance on several benchmarks.

### Strengths
1. Well-Motivated Problem: The paper targets a critical and practical challenge in the MLLM domain. Improving both training and inference efficiency is a highly valuable research direction.
2. Intuitive and Simple Core Idea: The central concept of NAAViT—that attention between visual tokens within the LLM might be redundant—is simple to understand and implement. The pilot experiment provides good initial evidence for this hypothesis.
3. Strong Empirical Efficiency Gains: The reported improvements in computational efficiency are substantial. A 66% reduction in inference FLOPs and a 26% reduction in training GPU hours compared to LLaVA-1.5 are impressive results that clearly demonstrate the practical benefit of the proposed architecture.
4. Clear Presentation: The paper is well-written, and the diagrams (especially Figure 2 and 3) effectively illustrate the differences between existing architectures and the proposed SAISA model.

### Weaknesses
1. While the combination of components is new, the core ideas feel more incremental than foundational.
The primary mechanism, NAAViT, essentially modifies the self-attention in embedding-space models to mimic the behavior of cross-attention models like Flamingo, where text queries attend to visual keys/values without any V-V interaction. The novelty lies in achieving this within a standard self-attention block via masking, rather than by introducing new cross-attention modules, thereby saving parameters. However, the conceptual leap is not substantial.
The idea of projecting features into multiple layers of a transformer is also not entirely new and has precedents in other architectures that seek to inject conditioning information throughout a network. The contribution seems to be a clever engineering combination of existing principles rather than a fundamentally new architectural paradigm.
2. The paper's ablations are too limited to fully substantiate its claims and disentangle the sources of improvement.
The most critical missing ablation is a direct comparison between a model using only NAAViT (as in the pilot experiment) and the full SAISA model (NAAViT + layer-wise projection). The current presentation conflates the benefits of two distinct architectural changes. It is unclear how much of the performance gain and efficiency improvement comes from simply removing visual self-attention versus the more complex layer-wise projection scheme. A thorough analysis is needed to justify the added complexity of SAISA's projector.
The paper argues for projecting visual features to every layer. Is this necessary? An ablation studying the impact of projecting to only a subset of layers (e.g., early, middle, or late layers) would provide deeper insights into how and where visual information is most effectively integrated.
3. The experimental comparison, while thorough against LLaVA-1.5, feels dated and lacks engagement with the latest state-of-the-art in efficient MLLMs.
The MLLM field is advancing at an extremely rapid pace. While LLaVA-1.5 is a crucial baseline, many newer and more efficient models have been proposed since its release. The comparison set in Table 2 feels somewhat stale, missing more recent architectures that have been published in late 2024 or 2025 (as this paper targets ICLR 2026).
More importantly, the paper does not adequately compare its architectural approach to orthogonal efficiency methods, particularly token pruning/merging. Models like FastV (which is cited but could be compared more directly) or other token reduction techniques achieve inference efficiency by dynamically removing visual tokens. SAISA's approach is to keep all tokens but reduce computation per token. A direct and fair comparison between these two competing strategies for efficiency is essential to contextualize SAISA's contribution. Is it better to prune tokens or to keep them all with cheaper attention? This question is left unanswered.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the trade-off between training and inference efficiency in MLLMs. Traditional embedding space alignment models (e.g., LLaVA) are efficient to train but slow to infer, while cross-attention space alignment*models (e.g., Flamingo) offer fast inference but incur high training costs. After analyzing the computational bottlenecks of both paradigms, the authors argue that self-attention among visual tokens is largely redundant. They propose NAAViT to remove visual–visual attention and build upon it to design SAISA, which maps visual features directly to the input spaces of self-attention layers in the LLM while skipping FFN operations on visual tokens. Experiments show that SAISA achieves a 26% reduction in training cost and a 66% reduction in inference FLOPs, with slightly improved accuracy—thus realizing efficiency in both training and inference.

### Strengths
1. **Clear problem statement and method design.** The paper tackles the key challenge of improving MLLM computational efficiency—a crucial factor for deployment and energy optimization. The combination of NAAViT’s `removal of visual–text attention` and SAISA’s `layer-wise input-space alignment` provides a novel trade-off between existing embedding- and cross-attention-based architectures.

2. **Fair experimental comparison.** All main experiments are conducted under identical data and training settings as LLaVA-1.5, ensuring fair comparison. The ablations demonstrate generality across different LLMs (Vicuna, Llama3, Mistral) and visual encoders (CLIP, SigLIP, ConvNeXt), while also studying projector design and pretraining strategies.

3. **Strong reproducibility.**  The paper includes detailed hyperparameters, training settings, and FLOPs derivations, providing sufficient information for replication.

### Weaknesses
1. **Insufficient evidence for the `redundancy` claim.** The paper concludes that attention among visual tokens is redundant, but this is supported only by a marginal performance gain of NAAViT over standard self-attention. Such improvement could stem from optimization dynamics or implicit regularization rather than true redundancy. This would be meaningful: Providing attention heatmaps or token correlation analyses, and testing on multi-object or spatial reasoning tasks would help verify whether visual–visual interactions are indeed dispensable.

2. **Unclear source of efficiency gains.** The study does not isolate the contributions of NAAViT versus the removal of FFNs on visual tokens. Without reporting FLOPs and performance when using NAAViT alone, it is unclear which component drives the observed acceleration.

3. **Potential incompatibility of NAAViT masks with inference accelerators.** In multi-image or interleaved dialogue scenarios, NAAViT introduces fragmented attention masks that may reduce the efficiency of Flash-Attention 2 or vLLM kernels.

4. **Outdated baseline selection for token compression.** Easy token unshuffle methods can reduce token counts to 1/4 or 1/9 with negligible information loss. Yet the paper lacks comparison with recent **parameter-free token optimization methods**, and only contrasts with older pruning-based approaches from 2024 (e.g., FastV, VTW), which weakens its claims.

5. **Limited experimental scope.** All primary results are based on Vicuna-7B and LLaVA datasets, without evaluation on larger models (e.g., 13B) or other modalities such as videos and multi-image inputs.

### Questions
1. Could the authors provide results using NAAViT alone (while keeping FFNs for visual tokens) to disentangle the sources of efficiency?
2. Does the sparse attention mask in NAAViT affect compatibility or speedup when using Flash-Attention 2 or vLLM inference engines, especially for interleaved or long-context inputs?
3. See `Weaknesses` for additional clarifications.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SAISA (Self-Attention Input Space Alignment), a novel and highly efficient architecture for Multimodal Large Language Models (MLLMs). The work is motivated by a key finding: that self-attention among visual tokens within the LLM is "highly redundant." The authors first validate this by proposing NAAVIT (No Attention Among Visual Tokens), a modified self-attention mechanism that eliminates this $O(v^2)$ computation, and show it improves performance over standard attention.Building on this, the SAISA architecture aligns visual features directly with the input spaces of each NAAVIT self-attention block in the LLM. This clever design allows visual tokens to participate in attention (text can attend to visual tokens, visual tokens can attend to text) but bypasses two major computational bottlenecks: (1) self-attention among visual tokens and (2) all FFN computations for visual tokens.The paper demonstrates that, compared to the LLaVA-1.5 architecture, SAISA reduces inference FLOPs by 66% and the training budget by 26%, all while achieving superior performance on a wide range of MLLM benchmarks.

### Strengths
- A 66% reduction in inference FLOPs and a 26% reduction in training budget (vs. LLaVA-1.5) are extremely significant practical contributions. This makes SOTA-level MLLMs more accessible to train and cheaper to deploy.

- The core finding that attention among visual tokens is redundant—and that FFNs for visual tokens can also be skipped—is a fundamental and novel insight into MLLM architecture.

- This is not just an efficiency-focused paper (e.g., pruning, quantization). SAISA outperforms its LLaVA-1.5 baseline across a wide array of benchmarks (Tables 2, 3, 5), indicating the architectural changes also serve as a good regularizer or are simply a better design.

- The ablation studies in Table 5 compellingly show that SAISA's benefits are not specific to one model pair but apply broadly across different LLM families (including GQA models like Mistral) and visual encoder types (ViT, ConvNeXt).

### Weaknesses
- The paper does an excellent job proving that attention among visual tokens is redundant but doesn't fully explore why. Is it because the visual encoder has already "bound" all necessary spatial/object relationships? Does removing it act as a regularizer, preventing the model from "overthinking" the visual features? A deeper analytical probe (e.g., attention visualization, feature analysis) into why NAAVIT works better would make this paper even stronger.

- The SAISA projector consists of n separate 2-layer MLPs (one for each of the n LLM layers), whereas the LLaVA projector is a single 2-layer MLP. This significantly increases the number of new parameters. The paper shows that training is still 26% faster (which is the more important metric), likely due to reduced computation in the frozen LLM blocks. However, an explicit comparison of the number of new, trainable parameters (SAISA projector vs. LLaVA projector) would be helpful for transparency. The paper's "shared MLP" pre-training strategy seems to be the key to managing this, but this could be clearer.

### Questions
- The finding that NAAVIT (no visual self-attention) outperforms vanilla self-attention (Table 1) is fascinating. Do the authors have a hypothesis for why this is the case? Does it act as a regularizer? Or is it possible that the visual encoder's features are already so robust that further self-attention only adds noise?

- Could you please provide the concrete number of new, trainable parameters introduced by the SAISA projector (e.g., for Vicuna-7B with 32 layers) and compare this to the number of parameters in the LLaVA-1.5 projector? This would clarify the trade-off being made (more parameters in the projector for far less computation in the LLM).

- How does SAISA compare to orthogonal efficiency methods like visual token pruning (e.g., FastV, cited in the paper)? SAISA's approach is to keep all 576 tokens but process them more cheaply. Would it be better to (e.g.) prune to 64 tokens and use a standard LLaVA architecture? Or can these methods be combined (e.g., use SAISA with a resampler, as in Table 7)?

### Soundness
3

### Presentation
3

### Contribution
3
