# Conjuring Semantic Similarity

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
The semantic similarity between sample expressions measures the distance between their latent `meaning'.These meanings are themselves typically represented by textual expressions.  We propose a novel approach whereby the semantic similarity among textual expressions is based not on other expressions they can be rephrased as, but rather based on the imagery they evoke. While this is not possible with humans, generative models allow us to easily visualize and compare generated images, or their distribution, evoked by a textual prompt. Therefore, we characterize the semantic similarity between two textual expressions simply as the distance between image distributions they induce, or 'conjure.' We show that by choosing the Jeffreys divergence between the reverse-time diffusion stochastic differential equations (SDEs) induced by each textual expression, this can be directly computed via Monte-Carlo sampling. Our method contributes a novel perspective on semantic similarity that not only aligns with human-annotated scores, but also opens up new avenues for the evaluation of text-conditioned generative models while offering better interpretability of their learnt representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel, visually grounded approach to semantic similarity between textual expressions. Rather than defining similarity based on rephrasing or textual embeddings, the authors propose measuring the distance between the image distributions “conjured” by a text-to-image diffusion model when conditioned on different prompts. Formally, the method computes the Jensen–Shannon divergence between the reverse-time SDEs of two conditional diffusion processes, estimated via Monte Carlo sampling. The resulting metric — termed Conjuring Semantic Similarity (CSS) — enables interpretable visualization of semantic relations and quantifies the alignment between a diffusion model’s learned semantic space and that of human annotators.

### Strengths
- It suggests a fresh and intellectually stimulating inversion of the usual approach: comparing texts via images rather than images via text.
- The use of JS divergence between SDEs is mathematically principled and connects diffusion dynamics with semantic distance.
- The qualitative visualizations (e.g., “Snow Leopard” ↔ “Bengal Tiger”) concretely illustrate semantic drift in the image domain.

### Weaknesses
- Comparisons are limited to Stable Diffusion v1.4; modern models (e.g., SDXL, Flux, SD3 Large) or text encoders like T5/CLIP-L) are not systematically tested. I believe Stable Diffusion v1.4 cannot reflect the latent space sufficiently.
- The correlations to human similarity scores, while encouraging, are moderate and inconsistent across datasets. Also, the Spearman correlation 0.65 is not enough for me.
- The paper primarily showcases successful examples (e.g., “Snow Leopard” vs. “Bengal Tiger,” “Bag of Chips” vs. “Bag of Fries”) where semantic differences are clearly visually representable. However, it provides no concrete failure analyses demonstrating when the proposed Conjuring Semantic Similarity method breaks down (e.g., Abstract or non-visual concepts, Metaphorical or figurative expressions, etc.).

### Questions
- Your experiments are limited to Stable Diffusion v1.4, which relies on an older CLIP text encoder. Given that SDXL, SD3, or Flux employ stronger text encoders (e.g., OpenCLIP-L or T5-XXL), how would you expect your Conjuring Semantic Similarity metric to behave when the underlying latent space better captures linguistic semantics? Would the observed moderate correlation (~0.65) improve, or do you think the limitation is intrinsic to visually grounded similarity?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes to measure the semantic similarity between text phrases by comparing the image trajectories produced by pre-trained text-to-image generative diffusion models. They propose doing so with a symmetrized KL divergence and shows how it compares to human-annotated scores and previous approaches in the literature.

### Strengths
This paper gives a refreshing and novel approach to measure semantic similarity. The paper is well-written and was fun to read. Experiments are fairly well-done and convincing.

### Weaknesses
I have some minor concerns, which should be easy to address.

* The paper notes that the proposed measure has a large standard deviation in line 318. I wonder if this can be reduced by averaging over multiple reverse SDE trajectories. Regardless, it would be good to see some experiments on this, as $p(\text{image} | \text{text})$ has some inherent variability. 
* I find it hard to interpret figure 2. It would easier to interpret based on the output of some clustering algorithm (e.g. hierarchical clustering) on this matrix to visualize the clusters.
* It would be interesting to see how the proposed approach is better than computing the KL divergence between $p(\cdot | y_1)$ and $p(\cdot | y_2)$ directly in embedding space, similar to FID. Concretely, a suggested procedure is:
     - sample a few images from $p(\cdot | y_1)$ and $p(\cdot | y_2)$
     - get their embeddings from CLIP or a some other model and reduce their dimensionality with PCA (since we probably do not have too many samples)
     - Calculate their means and covariances $(\mu_1, \Sigma_1), (\mu_2, \Sigma_2)$, and use the formula for KL between Gaussians $N(\mu_1, \Sigma_1)$ and $N(\mu_2, \Sigma_2)$. 

For example, see the parametric approximation method of https://arxiv.org/pdf/2212.14578 (this citation is also probably worth adding in the related work). 
* The divergence in line 242, namely $1/2 \big( KL(P_1 \Vert P_2) + KL(P_2 \Vert P_1)\big)$, is known as the Jeffreys divergence. This is not the Jensen-Shannon divergence, which is defined as $1/2 \big( KL(P_1 \Vert M) + KL(P_2 \Vert M)\big)$ where $M = (P_1 + P_2)/2$.

### Questions
* How does one obtain the expression in line 234 for the KL divergence? It would be good to see a formal derivation, at least in the appendix. Unrelated side question: what other divergences can be written in this form?
* It would be good to see some details about the data, e.g. domain, how many data points, etc. Also, some examples would be nice.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a method for defining semantic similarity between text expressions, not through textual embeddings such as BERT or CLIP, but by comparing the image distributions generated by the texts when used as prompts in a text-conditioned diffusion model. Instead of operating in embedding space, the method treats the diffusion model’s denoising process as a stochastic differential equation and, for two prompts y1 and y2, computes the Jensen–Shannon divergence between their corresponding reverse-time SDEs, representing the model’s denoising trajectories. This divergence, estimated through Monte Carlo sampling, serves as a visually grounded measure of semantic distance, meaning that two texts are considered semantically similar if they produce comparable denoising trajectories or similar image distributions in the diffusion model.

### Strengths
- The paper introduces a novel perspective on semantic similarity by grounding it in the visual space of diffusion models rather than the traditional textual or embedding space.
- By formulating distances between diffusion trajectories using stochastic differential equations and the Jensen–Shannon divergence, the method establishes a sound mathematical foundation instead of depending on heuristic measures of similarity.
- The proposed idea yields interpretable visual explanations showing how two prompts differ semantically through their generated imagery.

### Weaknesses
- One of the major shortcomings of the paper is that it does not clearly justify the need for its proposed methodology. Although the authors argue that contrastive-trained embedding models such as CLIP fail to capture textual semantic similarity fully, they neither provide concrete examples nor analyses demonstrating when or why these models fail nor explain how their approach remedies these limitations. Furthermore, it is not evident from the results or discussion that their method offers any improvement over existing embedding-based models, making the motivation for introducing a new framework less convincing.                  
    - The proposed method also underperforms these contrastive models by roughly 10%, which further weakens the argument for its necessity. Moreover, there is no quantitative or qualitative evidence presented in the paper demonstrating situations where traditional methods break down, but the proposed technique succeeds. 
    - The authors’ statement in line 317, "that large standard deviations imply that correlation with human annotators matters more than achieving state-of-the-art scores," raises additional concerns. If benchmark reliability is questioned, then the validity of their evaluation results becomes uncertain, making the overall empirical evidence for the method less convincing.
- Missing technical details: A central component of the proposed approach is the Monte Carlo (MC) sampling procedure, yet the paper provides very limited discussion of it. The authors report using up to only five MC samples, which appears insufficient for accurately estimating an expectation. Moreover, in Figure 3, the reported Spearman correlation is higher for three samples than for five, which raises several important questions:
     - How can the empirical mean be reliably estimated with as few as three samples? Wouldn’t such small sample sizes produce highly noisy estimates?
     - Why does the correlation decrease when the number of samples increases to five? Five samples are still far too few to ensure convergence.
    - Should a baseline or variance-reduction technique be introduced, as is common in Monte Carlo estimation?

Because these issues are never addressed in the paper, it becomes difficult to assess the soundness and stability of the proposed method.

- (Minor) Ambiguity: Some parts of the paper are unclear or lack sufficient explanation. For example, the metric reported in Table 1 is not described in the caption, leaving readers unsure whether higher or lower values indicate better performance. Similarly, Figure 2 presents distance matrices without clarifying that smaller values correspond to higher semantic similarity, which makes the figure confusing at first glance and difficult to interpret without additional context.

### Questions
- L107 – Missing citation for the original GAN paper [1].
- Clarification needed on STS datasets: The paper mentions several variants of the Semantic Textual Similarity (STS) benchmark (e.g., STS-B, STS-12, STS-13, etc.). Still, it remains unclear which specific datasets were used and how they differ in terms of setup or evaluation.
- Possible extension: Could a variant of the proposed method be designed for image–image matching, similar to Maximum Mean Discrepancy (MMD)-based distance measures such as FID or KID?
- Relation to existing perceptual metrics: Since the approach compares image distributions during the denoising process to estimate semantic similarity, is there a conceptual or mathematical connection to the Learned Perceptual Image Patch Similarity (LPIPS) metric [2]?

[1] Goodfellow, Ian J., et al. *“Generative Adversarial Nets.”* *Advances in Neural Information Processing Systems* 27 (2014)

[2] Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose to evaluate semantic similarity via the image that the text evokes, which is achieved by performing diffusions and then evaluating the JS divergence.

### Strengths
Originality: I find the core idea of using images to evaluate semantic similarity to be highly original and interesting. This is a major strength of the paper. 

Clarity: the paper is written and motivated in a clear way that is broadly accessible to a general interest ML/AI audience. 

Quality: I find the mathematical/technical portion of the paper to be sufficiently in depth for ICLR. The experiments performed are rather comprehensive and present evidence that supports the claims of the paper and the proposed method. 

Significance: The idea/topic of evaluating semantic similarity of generative models is timely and interesting. The main contribution of the authors approach is that it provides a new form of interpretability, which is sufficiently significant for ICLR.

### Weaknesses
My overall impression of the paper is largely positive, and I don't find any major weaknesses along quality/clarity/originality/significance in this paper. I raise some follow up questions in the next section.

### Questions
1. In principle, what limits this approach to diffusions? Could flows, or any other conditional text to image generative model also work in principle?

### Soundness
3

### Presentation
3

### Contribution
3
