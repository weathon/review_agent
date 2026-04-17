# Understanding Cross-layer Contributions to Mixture-of-Experts Routing in LLMs

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Mixture-of-Experts (MoE) has been a prevalent method for scaling up large language models at a reduced computational cost. Despite its effectiveness, the routing mechanism of MoE still lacks a clear understanding from the perspective of cross-layer mechanistic interpretability. We propose a light-weight methodology at which we can break down the routing decision for MoE to the contribution of model components, in a recursive fashion. We use our methodology to dissect the routing mechanism by decomposing the input of routers into model components. We study how different model components contribute to the routing in different widely used open models. Our findings on four different LLMs reveal patterns such as: a) MoE layer outputs usually contribute more than attention layer outputs to the routing decisions of subsequent layers, b) *MoE entanglement* at which MoE firing up in layers consistently correlate with firing up of MoE in subsequent layers, and c) some components can persistently influence the routing in many following layers. Our study also includes findings on how different models have different patterns when it comes to long-range and short-range inhibiting/promoting effects that components can have over MoE in subsequent layers. Our results indicate the importance of quantifying the impact of components across different layers on MoE to understand the routing mechanism.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the routing mechanisms in Mixture-of-Experts (MoE) LLMs from a cross-layer perspective. The core idea is to recursively decompose the expert assignment score, i.e., the result of a dot product between an expert's routing weight vector and the MoE layer's input, into a linear sum of contributions from all preceding model components. These components include the initial token embedding, the outputs of all previous attention layers, and the outputs of all previous MoE layers.

The authors define a metric for "influence" based on the variance of the scores a single component contributes to all experts in a receiving MoE layer; high variance implies a strong influence on the routing decision. Using this framework, the paper analyzes OLMOE, DeepSeek-V2-Lite, Qwen3-30B-A3B, and Mixtral-8x7B.  Here are some of their key findings:

The variance of the initial token embedding scores decays rapidly with layer depth.

The authors identify a "cross-layer entanglement" phenomenon, where specific MoE layers exhibit a strong, persistent, long-range influence on routing decisions in many subsequent layers.

The methodology could be extended to finer granularities, showing that specific attention heads or a small subset of experts are disproportionately influential on routing.

### Strengths
- Intuitive and straightforward method: The paper presents a simple and computationally light method for MoE interpretability. Shifting the focus from expert specialization (what experts do) to the composition of the router's input (why a router chooses) is a good contribution to the field. 

- Interesting Findings: The findings of this paper clearly demonstrate that MoE routing is not just shaped by the precedent attention layer but also previous transformer blocks.

### Weaknesses
1. **My greatest concern** for this paper is whether variance is a good metric for an expert’s contribution. The paper's metric (variance) measures "flatness" of gating scores, but this is an imperfect proxy for influence on the Top-K routing decision (expert selection and weight). For example, a component's contribution that changes scores (assuming selecting 2 experts out of 4 experts) from [10, 9, 1, 0] to [12, 7, 2, -1] with contribution [2, -2, 1, -1] (Top-K={E1, E2}) would have a high variance. A different component that changes scores from [5, 4.9, 4.8, 4.7] to [6, 3.9, 4.8, 4.7] (Top-K={E1, E3, a changed routing decision) with contribution[1, -1, 0, 0]. The latter alter the routing results but have lower variance.  The metric is not sensitive to the "gaps" between the K-th and (K+1)-th expert, which is what actually determines the decision. Mean while, the variance also does not reflect the final contribution/weight of each expert. For example, a component contribute [100, 100, 100, 100] to the gating score and another component contribute [1, 1, 0, 0]. The first component’s contribution is zero accroding to variance. but it actually make the final selective expert’s contribution more flat, which is quite important and cannot be ignored when understanding routing mechanism in MoE.  

2. **Does this paper re-find the massive activation phenomenon?** The observations discovered by authors closely relates to massive activation and attention sink. I hope authors could discuss the relation with these two phenomenons.  Furthermore, the author should to provide the most direct piece of evidence for illustrating the contribution of each components: a heatmap of component L2-norms, directly comparable to Figure 3. The findings in Figure 3 (e.g., high influence from M1, M4) could very well be a simple re-discovery of massive activation of that layer. Without this comparison, the paper's core claim of novelty is undefended. 

3. **Fragmented Narrative**: I feel that the paper's presentation is highly fragmented. This is a major weakness because it buries the paper's core discovery. Making readers easy to conclude that the findings are just a restatement of known concepts (i.e., massive activation). By the way, the title in the PDF is different from the submission.

4. **Speculative Practical Applications**: The abstract and conclusion claim the findings "highlight the opportunities... for effective model design and model serving." However, these claims are speculative and undeveloped.

### Questions
1. Can the authors provide a layer-level heatmap (identical in structure to Figure 3) but plotting the average L2-norm of the component outputs instead of the score variance? If this norm-heatmap is nearly identical to the variance-heatmap, the paper's claim to novelty is significantly weakened.

2. Could the authors comment on the limitations of variance as a metric, specifically its insensitivity to the score "gaps" between the Top-K and non-Top-K experts and selected expert’s contribution? Do we need additional metrics for this?

3. Assuming the variance metric is not just a proxy for norm (as per Question 1), what is the authors' primary hypothesis for the observed divergence between high-norm and high-influence components? 

4. Have the authors investigated whether these cross-layer influence patterns remain consistent during autoregressive generation on specific tasks (e.g., question-answering or MATH reasoning)? Is it possible that the observed routing patterns are an artifact of the C4 dataset and the forward-pass-only analysis?

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
4

### Summary
This paper investigates the routing mechanism in MoE-LLMs from a cross-layer mechanistic interpretability perspective. The authors propose a lightweight, recursive decomposition methodology to attribute the routing score assigned to each expert back to the contributions of preceding model components (token embeddings, attention layers/heads, and MoE layers/experts). Applying this methodology to four different large-scale MoE models (OLMOE, DeepSeek-V2-Lite, Qwen3, Mixtral), the study identifies several common patterns: (1) MoE layer outputs generally contribute more significantly and persistently to downstream routing decisions compared to attention layer outputs; (2) Evidence of "MoE entanglement," where the activation of experts in earlier layers correlates with the activation of experts in later layers; (3) Certain components exhibit long-range influence, affecting routing decisions many layers downstream; (4) Models exhibit distinct patterns regarding short-range versus long-range promoting or inhibiting effects from components on subsequent routing.

### Strengths
1. The paper addresses a timely and important gap in understanding MoE models by focusing on the cross-layer dynamics influencing routing decisions.
2. The empirical results reveal non-trivial cross-layer interactions, such as the stronger, persistent influence of MoE outputs compared to attention outputs, the "entanglement" effect, and long-range influences. These findings challenge simplistic, localized views of routing and provide valuable insights into the complex dynamics within these models.

### Weaknesses
1. Equation 8 uses an approximation ($\overline{LN}_{i}^{l}(z)$) where the RMS term is calculated over the *total* layer input, but the contribution of a *single* component $z$ is scaled. Given the non-linear nature of RMSNorm (depending on the norm of the entire input vector), the accuracy and potential biases introduced by this approximation are unclear and not quantified. This could affect the relative attribution scores between components.

2. While the paper successfully identifies several interesting cross-layer patterns (e.g., MoE persistence, entanglement stripes), it provides limited mechanistic explanations for *why* these patterns emerge. For instance, what structural or functional properties of MoE layers enable their outputs to have a more persistent influence compared to attention layers? What specific information pathways lead to the observed "entanglement" between expert activations across layers?

3. The paper could be clearer about how metrics like variance, APS, and ANS are aggregated (e.g., averaged across which tokens, which samples?) when presenting layer-level heatmaps (e.g., Figure 3).

### Questions
1. Could you elaborate on the validity of the $\overline{LN}_{i}^{l}(z)$ approximation used in Equation 8, particularly for RMSNorm? 

2. Regarding Proposition 1, how strongly does the variance of a component's score contributions correlate with its actual impact on changing the set of top-k selected experts?

3. You suggest using cross-layer contribution insights for expert prefetching. Could you sketch a potential algorithm or strategy for how the variance or entanglement patterns identified could be used to predict which experts are likely to be needed in future layers, thereby informing a prefetching policy?

### Soundness
2

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
This paper proposes a lightweight recursive decomposition method for quantifying the cross-layer contributions of different model components to MoE routing decisions. Through experiments on four production-level MoE models, the authors find:

  -Dominant Role of MoE Layer Outputs: The outputs of MoE layers exert the strongest and most persistent influence on the routing decisions of subsequent layers.

  -Cross-Layer Entanglement: Routing decisions are influenced by long-range cross-layer interactions, challenging the assumption that routing is a local process.

  -Component Effects: Attention layer outputs exhibit local facilitative effects in the bottom and top layers, while MoE layer outputs demonstrate global suppression in the middle layers.

  -Impact of Key Experts and Attention Heads: A small number of super experts have a sustained impact on routing.
These findings highlight the importance of cross-layer interpretability and provide new opportunities for the efficient design of MoEs.

### Strengths
-Lightweight Design: The recursive decomposition method requires no architectural modifications or retraining, achieving decomposition solely through input manipulation with low computational overhead. This offers a new paradigm for MoE interpretability research.

  -Model Diversity: Covers four mainstream MoE architectures, validating the universality of the conclusions. Supplementary evidence in the appendix, including t-SNE clustering and heatmaps, provides intuitive visualization of the spatial and hierarchical distribution of component contributions.

  -Theoretical Rigor: Propositions 1 and 2 prove the rationality of using variance as an influence metric, establishing a mathematical foundation for the analysis.

  -Discovery of Super Experts and Long-Range Suppression: Identifies super experts and long-range suppression effects, offering new ideas for optimizing MoE systems.

### Weaknesses
-Limited Model Scope: Experiments are confined to open-source models, excluding systems like GPT-4 MoE or other closed-source implementations. The generalizability of conclusions to larger-scale or heterogeneous architectures remains questionable.

  -Task Simplicity: Only the C4 and IOI tasks are used, lacking validation on complex downstream tasks such as reasoning or multimodal tasks. This makes it difficult to prove the task invariance of routing patterns.

  -Missing Attribution Method Comparison: No comparison is made with gradient-based attribution methods or causal intervention methods. The advantages of the variance metric are not fully substantiated.

  -Sensitivity to Hyperparameters: The analysis does not examine the impact of routing hyperparameters like the Top-k value or number of layers on the conclusions. For instance, Mixtral's Top-k value of 2 might amplify locality effects.

### Questions
Same as weakness.

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
4

### Summary
The paper analyzes the routing mechanism in Mixture-of-Experts (MoE) language models from a cross-layer mechanistic interpretability perspective.

The authors propose to understand routing by recursively decomposing the expert assignment score. The score for a given expert is a dot product of its weight vector and the MoE layer's input vector (Eq. 6). The paper's key insight is to decompose this input vector into a sum of its constituent parts: the original token embedding plus the outputs from all preceding attention and MoE layers (Eq. 8).

some of the key findings include:

- MoE layer outputs generally have a stronger and more persistent influence on downstream routing decisions than attention layer outputs.
- The paper finds that routing is not a purely local decision. Certain MoE layers (and specific experts within them, e.g., M1E9 and M4E14 in OLMOE) exhibit high influence on routing decisions many layers deeper, creating "stripes" of influence (visible in Fig 3d).

### Strengths
- The paper's primary strength is its shift in perspective. Moving from "what do experts specialize in?" to "what components influence the choice of expert?" is a significant contribution. The method of decomposing the router's input vector is elegant and interpretable.

- Clear and Justified Metrics: The use of variance as a proxy for "influence" (Proposition 1) and APS/ANS for "tendency" (Proposition 2) is well-defined and effective. This provides a clear, quantitative framework to move beyond simple co-activation analysis.

- The discovery of "MoE entanglement" (the long-range "stripes" in Fig 3d and 5a) is a major finding. It compellingly argues that MoE routing is a complex, non-local phenomenon, which has significant implications for how these models are understood.

### Weaknesses
Correlation vs. Causation: The analysis is entirely correlational. While it shows that the output of expert M1E9 correlates with routing decisions in layer 10, it does not prove a causal link. The paper lacks interventional experiments (e.g., ablating a high-variance expert and measuring the actual change in downstream routing) to confirm that these high-variance components are truly driving the decisions.

Underdeveloped Link to "Super Experts": The paper attempts to connect its high-influence experts to the "Super Experts" identified by Su et al. (2025) but finds the link is weak (Section 7, Appendix H). This is an interesting anti-finding, but it feels incomplete. It raises the question: which metric matters more? This paper's routing "influence" (variance) or the "Super Expert" output magnitude? The paper opens this door but doesn't walk through it.

Some of the notation are made deliberately complicate, which is hard to read at first, especially in section 3 and 4.1

### Questions
Could you elaborate on the `LN_bar`? from Eq 8, it seems that The methodology's decomposition (Eq. 8) relies on attributing scores to normalized components (`LN_bar`). However, Layer Normalization (LN) is a non-linear function applied to the sum of all components. $LN(x+y) ≠ LN(x) + LN(y)$. The paper defines $LN_bar(z) = \frac{(z * γ)}{RMS(total input)}$, which means the "contribution" of one component is scaled by a denominator that depends on all other components. So is such decomposition really make sense?

On Model Design Implications: Your findings suggest some layers/experts are far more important to routing. Based on this, would you recommend alternative MoE architectures? For instance, any new architecture designs or distribution of MoE layers in the network?

On Training Dynamics: This analysis is a snapshot of pre-trained models. Have you investigated how these influence maps evolve during training? Does the long-range "entanglement" (the stripes) emerge early, or is it a late-stage phenomenon as the model refines its pathways?

### Soundness
3

### Presentation
2

### Contribution
3
