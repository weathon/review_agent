# Norm$\times$Direction: Restoring the Missing Query Norm in Vision Linear Attention

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Linear attention mitigates the quadratic complexity of softmax attention but suffers from a critical loss of expressiveness. We identify two primary causes: (1) The normalization operation cancels the query norm, which breaks the correlation between a query's norm and the spikiness (entropy) of the attention distribution as in softmax attention. (2) Standard techniques for enforcing non-negativity cause destructive information loss by nullifying valid inner-product interactions. To address these challenges, we introduce **NaLaFormer**, a novel linear attention mechanism built upon a norm\$\times\$direction (ND) decomposition of the query and key vectors. We leverage each component to solve a distinct problem: The *query norm* is injected into our kernel to create a query-norm-aware map that restores the attention distribution's spikiness. The *direction vectors* are processed by a geometric, cosine-based similarity metric that guarantees non-negativity while preserving the rich, fine-grained information of the inner product. We validate NaLaFormer through a comprehensive multi-modal evaluation, where it sets new state-of-the-art benchmarks for linear attention. Our model achieves up to a 7.5\% accuracy gain on ImageNet-1K and a 4.7\% mIoU improvement on ADE20K over comparable baselines. It demonstrates profound efficiency, reducing peak memory by a transformative 92.3\% in token-intensive super-resolution tasks (70K+ tokens). NaLaFormer's versatility is further confirmed as it surpasses strong baselines like Mamba on common-sense reasoning and sets a new state-of-the-art on the Long Range Arena (LRA) benchmark. Source code can be found in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides an in-depth analysis of the performance of linear transformer, which is inferior to that of vanilla transformer, and proposes that it is due to the inability to preserve the Q-norm in the process of linear attention computation, resulting in the loss of the negative correlation between the entropy of the attn matrix and the Q-norm, which leads to the excessive smoothing of the distribution of the attn matrix distribution to be excessively smooth, thus reducing the performance of the transformer.

### Strengths
- This paper has a unique idea, proposes a novel angle, the line is fluent and clear, and the proof process is complete.
- The paper presents its own conjecture for the problem that linear attention does not perform as well as vanilla attention, focusing on the negative correlation between the distributional spikes (entropy) of the ATTENTION MATRIX and the Query-norm, which is an interesting entry point, and there is a novelty in the motivation of the paper.
- This paper gives an ingenious, efficient injection of query-norm while preserving the computational complexity of linear attention, which takes into account the nonnegativity required by the linear attention kernel function and at the same time reconstructs the q-norm-entropy relation that is lost in linear attention, and gives a concrete reasoning process with sufficient experiments to prove its effectiveness.
- Overall, I found this article enlightening in that it gives an entry point for analyzing transformer performance and broadens the ideas for future related analyses.

### Weaknesses
Overall, I found this article enlightening, but I still have a few minor questions I'd like to raise:
I don't see too many problems, but I personally have a small comment, I would suggest that the q-norm/entropy correlation plots of linear vs. vanilla attention should be compared in a more forward position, so that the reader can more quickly visualize the core ideas of the paper.

### Questions
Also I have a couple of minor questions:
- Notice that the transformer also exhibits different performance variations when the transformer is encoded with different positions, are they also due to a change in the query-norm-entropy relationship or is it something else?
- Is the INJECTION scheme unique and are there other possible INJECTION schemes for controlling the query-norm-entropy relationship? Also, do different positional coding schemes have an impact on the INJECTION calculation process?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This manuscript introduces NaLaFormer, a novel linear attention mechanism addressing the expressiveness gap between linear and softmax attention by leveraging norm×direction (ND) decomposition of query and key vectors. It restores the query norm’s role in regulating attention distribution spikiness through a norm-aware feature map and preserves non-negativity without information loss via a cosine-based direction similarity metric. Validated across multi-modal tasks, NaLaFormer achieves state-of-the-art performance—including up to 7.5% accuracy gain on ImageNet-1K, 4.7% mIoU improvement on ADE20K, and 92.3% peak memory reduction in super-resolution—while outperforming baselines like Mamba on language tasks and the LRA benchmark.

### Strengths
1. This paper resolves two core limitations of linear attention (query norm cancellation and destructive non-negativity enforcement) through theoretically grounded ND decomposition, which is novel to me.
2. The experiment are conducted on competitive benchmarks, e.g., ImageNet, COCO, ADE20K and DIV2K.
3. The formula and figure are clear and well-illustrated.

### Weaknesses
1. How the proposed linear attention work in diffusion is not clear. Since existing works show that linear attention can also perform well in diffusion transformers[1], it is encouraged to add some DiT experiments.
2. The RoPE is used in the proposed method, but not discussed.
3. The ablation study in λ and τ is missing.

[1] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators, in ECCV 2024.

### Questions
1. Does NaLaFormer need infra optimization when serving?

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
This paper introduces NaLaFormer, a Norm-Aware Linear Attention mechanism for Transformer models. It targets two major weaknesses in conventional linear attention: the disregard for query norm information and the loss of expressive negative inner-product interactions due to enforced non-negativity. The authors propose a decomposition of query and key vectors into norm and direction components, allowing dynamic entropy reduction modulated by query norms and a norm-preserving cosine mapping that enforces non-negativity while retaining directional richness. Substantial theoretical analysis is provided regarding the role of query norms in attention entropy, and empirical evaluations show performance gains on vision and language modeling benchmarks.

### Strengths
1. To avoid negative values, the re-mapped cosine direction is a somewhat clever technical contribution.

2. The experiment results show strong improvements over many baselines. On ImageNet-1K, NaLaFormer variants outperform recent linear attention and some softmax-based ViT models across all model sizes, often by substantial margins.

3. Both vision (understanding and super-resolution) and language tasks are included, which extends the breadth of the applications of this paper.

### Weaknesses
1. The paper does not cite or compare to MetaLA [1] and InLine [2], which both focus on matching softmax’s spikiness or optimizing the linear approximation. Empirical comparisons and deeper methodological discussion are crucial, especially in language tasks, to establish both novelty and superiority.

2. The exact model definitions and training details are not fully disclosed in the appendix, such as Swish activation before the classifier, Layerscales, 1024-dim classifier, convolution patch embedding, RoPE, and CPE, which is a convolution bypass in the attention module. Some or many of the baselines lack these enhanced designs; therefore, a clean version of the proposed method should be carefully ablated.

3. The cosine direction works similarly to rotary position embeddings (RoPE) [3] in large language and vision models; this paper does not discuss the technical insights between the proposed method and RoPE.

[1] Chou, Yuhong, Man Yao, Kexin Wang, Yuqi Pan, Rui-Jie Zhu, Jibin Wu, Yiran Zhong, Yu Qiao, Bo Xu, and Guoqi Li. "MetaLA: Unified optimal linear approximation to softmax attention map." Advances in Neural Information Processing Systems 37 (2024): 71034-71067.

[2] Han, Dongchen, Yifan Pu, Zhuofan Xia, Yizeng Han, Xuran Pan, Xiu Li, Jiwen Lu, Shiji Song, and Gao Huang. "Bridging the divide: Reconsidering softmax and linear attention." Advances in Neural Information Processing Systems 37 (2024): 79221-79245.

[3] Su, Jianlin, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing 568 (2024): 127063.

### Questions
I have no other questions about this submission.

### Soundness
2

### Presentation
3

### Contribution
2
