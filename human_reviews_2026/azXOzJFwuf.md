# FuseNorm: Achieving the Best of Both Worlds from PreNorm and PostNorm

- Decision: Reject
- Scores: 2, 2, 6, 4, 4

## Abstract
The success of Large Language Models (LLMs) hinges on the stable training of deep Transformer architectures. A critical design choice is the placement of normalization layers, leading to a fundamental trade-off: the PreNorm architecture ensures training stability at the cost of potential performance degradation in deep models, while the PostNorm architecture offers strong performance but suffers from severe training instability. In this work, we propose FuseNorm, a novel technique designed to resolve this dilemma by integrating the strengths of both paradigms. FuseNorm adopts the clean residual path of PreNorm to stabilize signal propagation while employing a PostNorm-style computation that normalizes the output of the residual connection, thereby enhancing model performance. We provide a theoretical analysis demonstrating that FuseNorm, combined with a principled scaling strategy, maintains bounded signal variance throughout the network, preventing the gradient issues that plague PostNorm models, and alleviating the representation collapse of PreNorm. Empirically, FuseNorm consistently outperforms standard normalization schemes in both dense and Mixture-of-Experts (MoE) scenarios, paving the way for more powerful and stable Transformer architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the trade-off between PreNorm and PostNorm Transformer architectures. PreNorm stabilizes training but often underperforms at scale. In contrast, PostNorm achieves better final performance but is prone to instability in deep networks. To reconcile these opposing properties, the authors propose FuseNorm, a simple yet effective modification that preserves the clean gradient path of PreNorm while applying Layer Normalization at the output like PostNorm.

### Strengths
The core ideas are well-motivated and supported by theoretical insights.

The method shows improvements in performance across various tasks and model scales.

### Weaknesses
The performance improvement is marginal, and in some cases (e.g., LMB perplexity, LMB accuracy, and ARC-c accuracy at 740M scale) the method underperforms compared to PreNorm. Moreover, on the 5B model, HybridNorm achieves better performance on ARC-c accuracy.

In addition, the paper lacks completeness in several aspects:

**Lack of comparison with relevant baselines**: The paper does not include a comparison with Peri-LN (Peri-LN: Revisiting Normalization Layer in the Transformer Architecture, ICML 2025), which is a closely related normalization method.

**Confounding factors in experimental design**: The method couples FuseNorm with Scale Init. Since initialization itself can significantly impact both training stability and final performance, it is unclear how much of the gain comes from FuseNorm vs. Scale Init. For example, a comparison with PreNorm + Scale Init would provide a clearer attribution of the source of improvement.

**Missing ablation experiments**: The paper does not provide standalone ablation experiments to disentangle the contribution of different architectural components (residual shortcut structure vs. normalization sequence) to training stability and performance.

**Reproducibility**: No implementation or code is provided, which makes it difficult to fully verify the empirical claims.

### Questions
Minor Errors

Line 505: In in -> In

Line 658: to twice as 740M -> to twice that of the 740M model

Line 659: maximize -> largest

Line 666: slimpajama -> SlimPajama

Line 674: Slimpajama -> SlimPajama

Line 676: Winograde -> WinoGrande

Line 676: Hellaswag -> HellaSwag

Line 675: LM harness evaluation -> the LM Harness evaluation

Line 174: LN(MHA(X′_l)) + X′_l) -> LN(MHA(X′_l) + X′_l)

Lines 175 and 190: The prime notation appears to be inconsistently rendered in mathematical expressions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes to achieve a best of both worlds between PreNorm and PostNorm in the Transformer architecture, by modifying the skip-connection/LayerNorm placements. Specifically, in a PostNorm architecture, the authors remove the skip connection from the FFN block, instead replacing it with a longer skip connection across both Attention and FFN blocks. The authors show improvements compared to PreNorm on multiple model sizes, and compared to some prior baselines.

### Strengths
1. The paper shows improvements over PreNorm architecture in pre-training across multiple model sizes, and over some prior works.
1. The proposed method achieves improved representation diversity across layers

### Weaknesses
1. A lot of the derivations are extremely handwavy, with numerous approximations and assumptions throughout in the derivations. ( E.g. $\approx$ occurs 15 times in the manuscript.).
1. Section 4 of the paper, covering depth and width scaling, are not proposing anything new, and are extremely redundant. For example, the  same/similar scaling has often been proposed before (e.g., see section K.3 of https://arxiv.org/pdf/2403.09635), and the "width scaling" is the authors simply verifying that prior works for LR transfer work with their architecture as well.
1. In section 5, the authors explicitly admit to an incomplete baseline study. A claim to "consistently outperform...advanced variants" is premature unless all baselines are working correctly.
1. In section 6, the authors point to their architecture being trainable with higher LR than PostNorm as proof of "mitigated gradient decay", but this conclusion does not necessarily hold. For example, a model with reduced sensitivity (eg. a linear layer with very large param values, followed by a layernorm) will be trainable with a much larger LR, but this does not say anything about inherent stability or instability of that model.
1. Even assuming all the derivations of the authors are correct, their proposed method only allows training 2x deeper post-norm models (equation 11). Prior works such as DeepNorm instead extend it to 100s of layers.

### Questions
1. Was the learning rate and initialization hyper-parameters set from known-good values from prior works, or perhaps hyper-parameter searched for all the baselines in Table 21? Inefficient setting of these can significantly affect performance.
1. (minor, no author rebuttal needed) The pdf seems to be somewhat bugged - searching and highlighting text is broken in several pages across multiple pdf readers. I have not observed the same in other papers.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FuseNorm, a normalization design that keeps a direct input–output skip connection through each Transformer block while applying normalization after both the attention and MLP sublayers. This layout aims to combine the gradient stability of PreNorm and the representational strength of PostNorm. The authors analyze its gradient and variance behavior, propose a depth-dependent initialization scheme, and validate the approach on dense and MoE models up to 16B parameters. FuseNorm yields small but consistent gains over PreNorm and maintains stable training where PostNorm becomes unstable.

### Strengths
* The method is simple, clearly defined, and easy to integrate into existing architectures. While training stability for large-scale Transformers has been extensively studied in recent months through various architectural innovations, the proposed method still contributes a refreshing perspective to this ongoing line of work.
 * Theoretical and empirical analyses are coherent: the scaling rule derived from variance control matches practical stability trends.
 * Results are consistent across model sizes and architectures, with diagnostic evidence (gradient norms, inter-layer similarity) that supports the claims.

### Weaknesses
* The paper omits comparisons to recent normalization schemes such as Peri-LN (arXiv:2502.02732), which has been adopted by several major LLMs and addresses the same issues. The paper’s motivation and positioning substantially overlap with prior works such as MixLN and HybridLN, which limits its perceived novelty.
 * The reported improvements are modest relative to the added complexity.

### Questions
- How does FuseNorm compare with recently proposed peri- or sandwich-style normalization schemes such as Peri-LN under identical hyperparameters and compute budgets?

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
4

### Summary
This paper introduces FuseNorm, a normalization strategy that unifies the training stability of Pre-LayerNorm and the performance advantages of Post-LayerNorm in Transformer architectures. The key idea is to retain a residual path while applying a Post-LN-style normalization at the block output, thereby resetting variance and maintaining balanced gradients across depth. Theoretical analysis shows that FuseNorm preserves stable Jacobian spectra, and empirical studies confirm consistent stability and performance improvements across both dense and Mixture-of-Experts models.

### Strengths
- The architectural modification is simple and intuitive, replacing the FFN residual with the original block input and normalizing the final output.  
- The theoretical condition for depth scaling provides a practical guideline for stable training.  
- Empirical results show consistent improvement trends across dense and MoE settings, suggesting potential practical value.

### Weaknesses
1. **Figures and visual evidence are unconvincing and inconsistently presented.** Many of the figures (Figs. 2–7) lack consistent axis ranges, normalization, and clarity in what they aim to demonstrate, making cross-figure interpretation difficult. For example, Figure 3 is intended to illustrate training collapse without scaling, yet it only shows three layer traces (layers 1, 12, 24) from a single run, without comparison to Pre-, Post-, or scaled FuseNorm variants. This makes it impossible to assess generality, effect size, or reproducibility. A unified plot contrasting “no-scale vs. scale” across normalization types with mean ± std over multiple seeds is needed. Moreover, Figure 6—claimed to demonstrate severe representation collapse in Pre-LN and its prevention by FuseNorm—does not visually support such a conclusion. When inspecting the heatmaps, the difference between Pre-LN and FuseNorm appears marginal, with large red regions (high similarity) persisting in both cases. The improvement is not visually evident enough to justify the strong qualitative statement in the text.
Each figure and caption should clearly specify the experimental setup, number of runs, normalization scheme, and intended takeaway, ensuring visual evidence aligns with the claimed phenomena.

2. **Missing strong baselines, fairness issues, and lack of comparison to Gemma-style architectures.** The paper lacks head-to-head results with strong baselines such as DeepNorm and NormFormer, both of which explicitly target the same stability–performance trade-off. In addition, the OLMO2 and LayerNorm-Scaling results are reported as “being re-run,” raising questions about hyperparameter fairness. A rigorous comparison should fix dataset, token count, learning-rate schedule, initialization, and random seeds (≥ 3), with clearly defined criteria for “training failure.” Furthermore, recent open-source models such as Gemma 2/3 employ a Peri-LN-style normalization [1], which shares similar motivations of balancing gradient flow and variance growth. A theoretical and empirical comparison between FuseNorm and such Peri-LN architectures would substantially strengthen the paper’s positioning and clarify whether the proposed method provides distinct or complementary benefits. Adding Peri-LN to Table 2, along with gradient-norm profiles or stability-transfer plots (Fig. 4), would make the evaluation more comprehensive and fair.

3. **Ablation insufficiency – unclear contribution of structure vs. initialization.**  The current results do not isolate whether the observed improvements stem from the new FuseNorm block design itself or from the “Scale-Init” strategy introduced for depth scaling.

4. **Missing quantitative validation of theoretical claims.** The theoretical analysis predicts that the variance of each residual branch should scale and that FuseNorm reduces gradient-decay rates relative to Post-LN. However, the paper provides no empirical evidence verifying these relationships. 


[1] Kim et al. "Peri-ln: Revisiting normalization layer in the transformer architecture." ICML2025.

### Questions
All major questions and requests for clarification are already integrated into the Weaknesses section for clarity and conciseness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **FuseNorm**, a normalization strategy that integrates the advantages of both PreNorm and PostNorm while addressing their respective drawbacks. In particular,  FuseNorm leverages the training stability benefit of PreLN and combines it with the performance benefits of PostLN.

The authors provide theoretical justifications for how FuseNorm preventing representational collapse in deeper layers, while also improving gradient flow to counteract gradient decay during training.  Empirical results show consistent performance improvements over recent hybrid normalization schemes that attempt to mix or modify PreLN and PostLN strategies.

### Strengths
1. LayerNorm placement is still a very active research topic, with several recent works exploring position-specific variants like QK-, QKV-, and FFN-LayerNorm [1,2, 3], or hybrid approaches such as MixLN that aim to balance stability and performance in deeper models. This paper fits naturally into that discussion and and provide a unified and principled way to combine the strengths of existing normalization strategies


2. Authors have provided a clear mathematical analysis on how FuseNorm helps prevent representational collapse and mitigate gradient decay which offer an intuitive  understanding of normalization behavior in large and deep LLMs.

3. The experimental evaluation is thorough, comparing FuseNorm against recent hybrid normalization strategies across both dense and MoE-based FFN architectures, with model sizes ranging from 0.74B to 5B parameters. The performance gains in deeper LLMs is convincing for showing the utility of FuseNorm.


[1] Dehghani et al., Scaling vision transformers to 22 billion parameters, ICML 2023

[2] Zhuo et al., HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization, 2025

[3]  Rybakov et al., Methods of improving llm training stability, 2024

### Weaknesses
1. The novelty of the work is quite limited. The idea of combining the benefits of PreLN and PostLN has already been well explored in prior works such as MixLN [1] and other hybrid normalization schemes. The proposed FuseNorm is an **incremental modification** rather than a fundamentally new concept, and the paper does not convincingly articulate what distinguishes it in terms of design principle or innovation.


2. The Authors  do not demonstrate how FuseNorm improves the **quality of internal representations**. Although it claims to prevent representational collapse, there’s no clear analysis---such as probing FFNs, or representation visualizations. Without showing how the internal representations, the paper’s claim about representational improvement is not convincing.  For example  eigen-value distribution or Rank-analysis shown in  [2,3].  


[1] Li et al., Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN, ICLR 2025

[2] Loshchilov et al., nGPT: Normalized Transformer with Representation Learning on the Hypersphere, ICLR 2025

[3] Jha et al.,  Spectral Scaling Laws in Language Models: How Effectively Do Feed-Forward Networks Use Their Latent Space? EMNLP 2025

### Questions
Could the authors provide the eigenvalue distribution of FFN post-activations or weight matrices, or a layer-wise rank comparison between FuseNorm, PreLN, PostLN, and MixLN models? This would offer concrete evidence of how FuseNorm impacts internal representation quality and whether it genuinely prevents representational collapse in deeper layers.

### Soundness
2

### Presentation
3

### Contribution
2
