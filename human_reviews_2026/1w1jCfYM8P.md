# Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Mixture-of-Experts (MoE) has emerged as a powerful paradigm for scaling model capacity while preserving computational efficiency. Despite its notable success in large language models (LLMs), existing attempts to apply MoE to Diffusion Transformers (DiTs) have yielded limited gains. We attribute this gap to fundamental differences between language and visual tokens. Language tokens are semantically dense with pronounced inter-token variation, while visual tokens exhibit spatial redundancy and functional heterogeneity, hindering expert specialization in vision MoE. To this end, we present $\textbf{ProMoE}$, an MoE framework featuring a two-step router with explicit routing guidance that promotes expert specialization. Specifically, this guidance encourages the router to $\textit{first}$ partition image tokens into conditional and unconditional sets via conditional routing according to their functional roles, and $\textit{second}$ refine the assignments of conditional image tokens through prototypical routing with learnable prototypes based on semantic content. Moreover, the similarity-based expert allocation in latent space enabled by prototypical routing offers a natural mechanism for incorporating explicit semantic guidance, and we validate that such guidance is crucial for vision MoE. Building on this, we propose a routing contrastive loss that explicitly enhances the prototypical routing process, promoting intra-expert coherence and inter-expert diversity. Extensive experiments on ImageNet benchmark demonstrate that ProMoE surpasses state-of-the-art methods under both Rectified Flow and DDPM training objectives. Code and models will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors present ProMoE, a two-step routing framework for Diffusion-MoE models. In the first step, tokens are routed according to functional heterogeneity, sending unconditional tokens to dedicated unconditional experts. The remaining conditional tokens are then routed using a cosine similarity router to experts, with an additional routing contrastive loss designed to enhance intra-expert semantic similarity and inter-expert semantic differences.

### Strengths
1. Experimental results on ImageNet are strong, with uniform improvements in generation quality and diversity

2. The authors address scalability comprehensively, with experiments covering four model sizes. 

3. The paper is clear and easy to follow

### Weaknesses
**Limited technical novelty**. The authors claim the prototypical routing mechanism is novel, but I struggle to see the novelty here. The learnable prototypes appear to just be standard learnable expert embeddings and semantic similarity is computed using a cosine similarity between inputs and expert embeddings, which is fairly conventional in MoE [1,2, 3]. The authors do use an identity activation function which is non-standard, but this alone does not really qualify the entire routing mechanism as novel in my view. 

**Missing experimental results**. The reported results in Fig 3 show that ProMoE indeed offers substantive improvements over dense and MoE baselines up to 500k training steps, but the performance differences do appear to be converging on one another very quickly, with the differences starting to look more marginal towards 500K and with increasing cfg to 1.5. Given that the samples generated in figure 4 required 2 million training steps and a cfg of 4.0, it raises the question of whether the performance gains shown in Fig 3 and Table 4 are meaningful, as it seems the model is unlikely to be near convergence at 500K samples. Indeed, the loss curves seem to suggest that even at 1.2M steps the model is far from convergence. Given the trend in performance visible in Fig 3, it looks possible that at 2M and cfg=4.0 the improvement of ProMoE may no longer substantive, but the authors haven't included these important results. 

**Single dataset for experimental validation**. The authors present their experimental results on just ImageNet-1K. Though the authors do a good job of validating at multiple model sizes, the empirical contribution would be much more persuasive if the findings could be validated across multiple datasets. 


[1] On the representation collapse of spare moe [Chi et al, NeurIPS 2022]
[2] Statistical advantages of perturbing cosine router in moe [Nguyen et al, ICLR 2025]
[3] Sparse moe are domain generalizable learners [Li et al, ICLR 2023]

### Questions
I'd strongly recommend the authors to include analysis of the kind seen in Table 4 but at training step=2M and cfg=4.0 across the MoE baselines and the dense baseline. This would provide a comprehensive analysis of the empirical benefits of ProMoE at convergence. Just choosing one size, ideally the largest, would be sufficient. If the authors can demonstrate the strong empirical results hold up at higher training step and cfg settings, I would consider raising my score, but for now it seems possible that the reported gains are too far from convergence to be meaningful. 

If the authors could include an additional dataset that would also help boost the empirical contribution.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ProMoE, a framework that successfully applies Mixture-of-Experts (MoE) to Diffusion Transformers (DiTs), addressing why previous attempts failed. The authors argue that visual tokens, unlike language tokens, have high redundancy and functional differences, which hinders expert specialization. ProMoE solves this with a novel two-step router that first separates tokens by function and then assigns them to experts based on semantic content using learnable prototypes. This guided approach, enhanced by a new contrastive loss, enables strong expert specialization and achieves state-of-the-art results on ImageNet.

### Strengths
- This paper clearly diagnoses the problem of vision MoE and proposes an innovative ProMoE to solve it.

- The ProMoE achieves validated, state-of-the-art results on the ImageNet benchmark.

- The presentation is clear and easy to understand.

### Weaknesses
- What is the fundamental difference between prototypical routing and conventional MoE routing mechanisms, such as one using a standard linear layer? The paper introduces "learnable prototypes", but this seems functionally very similar to using the learnable weights of a linear layer to calculate token-expert affinities. Could you clarify what makes this prototypical approach a genuine innovation, rather than just a conceptual re-framing of a standard linear gating mechanism?

- The routing mechanism in ProMoE appears to rely on pre-defined structures tailored for specific categories, unlike the autonomous expert specialization seen in LLMs. This raises questions about its generalizability—how would ProMoE handle open-ended conditional inputs, such as a natural-language prompt, rather than predefined categories? This design appears less flexible and general.

- How is the number of experts determined, and what is the rationale for that specific choice? Are there more detailed ablation studies on the impact of varying the number of experts?

### Questions
See Weaknesses.

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
4

### Summary
The paper proposes ProMoE, a novel MoE framework for Diffusion Transformers that addresses the failure of prior MoE designs in vision via a two-step router with explicit routing guidance. It introduces conditional routing to separate functional roles and prototypical routing with learnable prototypes, enhanced by a routing contrastive loss.

### Strengths
1. The paper effectively addresses the core challenges of visual token redundancy and functional heterogeneity in Diffusion Transformers, introducing mechanisms that enable true expert specialization within the Mixture-of-Experts framework.
2. The proposed method demonstrates strong and consistent scaling behavior across multiple model sizes, validating its robustness and efficiency under both Rectified Flow and DDPM training paradigms.

### Weaknesses
1. The experiments are conducted solely on ImageNet-1K for class-conditional generation, without evaluations on other datasets or modalities, which limits the evidence of generalization.
2. The paper does not report quantitative expert utilization, such as the proportion of tokens or capacity per expert, making it hard to assess balance and specialization.

### Questions
1.Could the authors provide quantitative statistics of expert utilization (e.g., token-per-expert ratios or activation entropy) to substantiate the claimed specialization and balance?
2.Could the authors compare ProMoE with other unsupervised clustering methods that support top-K routing, such as GMM or deep clustering?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduce a new expert routing method for Diffusion Transformers, by treating conditional and unconditional tokens independently. Routing guidance and contrastive learning are further introduced to enhance the performance.

### Strengths
1. The rationale for separating conditional and unconditional tokens is clear and well-founded.  
2. The investigation into routing guidance and load balancing is insightful and valuable.

### Weaknesses
1. It would be beneficial to include ablation studies on dense models with conditional routing to determine whether the performance gain stems solely from conditional routing itself or requires combination with routing enhancements.

2. Since one key advantage of MoE models is improved computational efficiency, the authors are encouraged to report training and inference times, as well as FLOPs, in comparison to both dense models and other MoE variants.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3
