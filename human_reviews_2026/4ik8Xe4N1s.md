# Uncertainty Preservation in Generative Visual Autoregression

- Decision: Reject
- Scores: 2, 2, 2, 0

## Abstract
Autoregressive (AR) models are among the most popular frameworks for visual generation. Recently, they have demonstrated competitive performance in visual generation through next-scale prediction. However, the training pipeline relies on cross-entropy as an objective, which enforces a precise target distribution for each token. This can lead to overconfident, sharp predictions and ultimately, a lack of diversity or mode collapse in the generated samples. In supervised learning, techniques such as label relaxation have been proposed to address the shortcomings of cross-entropy by replacing precise targets with sets of plausible distributions. However, in generative models, uncertainty should be considered as dispersion over a range of plausible outcomes rather than ambiguity about a single correct label. Building on this view, we introduce *uncertainty preservation* for visual autoregressive models. Specifically, we obtain predictive distributions as *second-order likelihoods* and penalize any deviations in their dispersion from a calibrated reference. Moreover, we implement a semantic entropy loss to provide a complementary measure of uncertainty that aligns with consistency at the meaning level. In theory, we demonstrate that our approach is a special case of second-order regularization, whereby penalizing variance deviation is equivalent to controlling the scale component of a Wasserstein distance between credal distributions. Through extensive experiments on multiple settings and datasets, we demonstrate that our model, despite its simplicity and low computational overhead, improves generation quality and diversity. Moreover, we highlight the practical utility of uncertainty quantification in image generation by showcasing additional applications where our approach could be beneficial.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Uncertainty Preservation in Generative Visual Autoregression to address the limitations of standard Visual Autoregressive (VAR) models trained using cross-entropy (CE) loss. CE training often enforces overly sharp, overconfident predictions, leading to reduced diversity, mode collapse, and codebook under-utilization. The proposed framework augments the standard likelihood training with two regularizers: Second-Order Regularization which penalizes deviations in the predictive distribution's dispersion (variance relative to a calibrated reference, and Semantic Entropy Loss which quantifies uncertainty at the meaning level by aggregating probabilities of semantically equivalent code sequences before computing entropy. Empirical results on AFHQ and ImageNet-100 demonstrate consistent improvements in generation quality and diversity.

### Strengths
1. Originality: Although the introduction of these two regularizations are from other fields like language modeling, it is highly original in this VAR domain. Also, the theoretical connection drawn between variance penalization and controlling the scale component of a Wasserstein distance provides a principled geometric foundation for the method.

2. Quality: The quality of the execution is high, evidenced by the principled loss design and the comprehensive empirical validation. The method achieves consistent quantitative improvements on AFHQ and ImageNet-100, outperforming the standard VAR baseline in FID and IS metrics (Table 1). The method's effectiveness in conditional generation tasks, preventing repetitive patterns in out-painting and yielding sharper, semantically plausible fills in in-painting, further demonstrates its robustness.

3. Clarity: The paper clearly articulates the problem of overconfident predictions and the limitations of first-order likelihoods (cross-entropy). The conceptual distinction between the proposed second-order method and alternatives like Label Smoothing and Label Relaxation is well-illustrated (Figure 1). The ablation study explicitly separates the contribution of L_2nd and L_SE, providing transparency into the necessity of both components for optimal performance (Table 2).

4. Significance: The framework offers a simple and efficient way to enhance generation fidelity and diversity without incurring additional inference costs. By effectively tackling mode collapse and codebook under-utilization, the work provides a highly practical and scalable solution for improving high-fidelity image generation, confirmed by its performance on large datasets like ImageNet-100.

### Weaknesses
1. Over-reliance on Gaussian Approximation for Theoretical Justification: The crucial connection between the variance penalization term L_2nd  (Equation 4) and the Wasserstein geometry relies heavily on the assumption that the first-order categorical distributions θ (in logit space) and the second-order distributions Q and Q* can be accurately approximated by Gaussian families (Proposition 3.0.1 and 3.0.2). Since VAR outputs a categorical distribution p_θ^(i)∈Δ^{V−1}, the fidelity of this high-dimensional Gaussian approximation, especially when the predictions become highly concentrated (sharp) or extremely diffuse, is questionable and not empirically substantiated in the sources, potentially weakening the principled nature of the loss term.

2. Lack of Detail in Visual Semantic Equivalence Class Construction: The effectiveness of the Semantic Entropy Loss (L_SE, Equation 5) hinges entirely on how the equivalence classes C of code sequences s (that reconstruct the same semantic content) are defined and constructed. The paper mentions that this ensures consistency across semantically equivalent token reconstructions, but fails to provide the specific methodology used to determine these classes for visual tokens within the VQ-VAE codebook of size V=2048. Without this crucial detail, the claim that L_SE aligns uncertainty with meaning rather than arbitrary code redundancy is difficult to verify or reproduce.

3. Lack of More Baselines and Results and Inconsistency between Best Results and Ablation Study: I'm not in the field of VAR but I think the author had better compare more other methods which try to improve the basic VAR to show their effects on the problems this paper focused on and conduct more detailed experiments. I think the current results are too limited and not substantial enough to be convincing for a paper in the field of visual generation. There may be a quantitative discrepancy between the model's overall best result reported in the main comparison table and the ablation study. The best achieved FID on AFHQ is 46.97 (Table 1), yet the best result achieved in the component ablation (using both λ_1 and λ_2) is 47.4 (Table 2).

4. Choice of "Minimal Variance" Reference Distribution: The authors state they opt for an "approximately normal distributions with fixed location parameter and minimal variance as q*" for the reference distribution in L_2nd. While this choice is cited to allow for efficient approximation, choosing a distribution with minimal variance might inherently conflict with the stated goal of preserving the appropriate level of dispersion consistent with the data manifold. If the natural generative uncertainty (the true optimal dispersion) is significantly higher than this "minimal variance," the regularization might improperly suppress diversity rather than preserve it.

### Questions
1. Computational Tractability of Semantic Entropy (L_{SE}): The Semantic Entropy H_sem (Equation 5) aggregates probabilities over sequences s∈c: ∑ s∈c p_θ(s). Given that VAR generates sequences across multiple scales (Equation 1), defining s as a full code sequence would involve summing over an exponential number of possible sequences, which is computationally infeasible. 
- Question: Please clarify what precisely the sequence s represents in the context of VAR’s multi-scale prediction. How is the summation over the potentially enormous set of semantically equivalent sequences c (or an efficient approximation thereof) performed in practice, consistent with the claim of low computational overhead?

2. Specific Calibration of Reference Variance \sigma^2_\star(k): The second- lordeross L_2nd requires a reference variance σ_*^2(k) that specifies the preferred level of dispersion at scale k. The chosen reference distribution q^* is an approximate normal distribution with "minimal variance". 

- Question: How was the numerical value or functional form of this "minimal variance" σ_*^2(k) determined for each scale k? Since the optimal dispersion may vary greatly between coarse (global structure) and fine (local detail) scales, please elaborate on how this scale-dependent reference σ_*^2(k) was chosen to ensure it preserves the appropriate uncertainty rather than merely suppressing variance.

3. Empirical Verification of Gaussian/Wasserstein Link: The theoretical foundation rests on the Gaussian approximation showing L_2nd is a tractable surrogate for the variance component of the Wasserstein distance W(Q,Q*) (Proposition 3.0.2). 

- Question: Did the authors empirically monitor the correlation between the calculated value of the L_2nd term and an estimate of the true variance component of W(Q,Q*) during training? Providing empirical metrics (e.g., correlation coefficients or comparative plots across training steps) would substantially strengthen the claim regarding the fidelity of the Gaussian approximation for the actual categorical output distributions in VAR.

4. Ablation Study on ImageNet-100: The ablation analysis (Table 2) investigating the influence of λ is provided primarily, if not exclusively, for the AFHQ dataset, which is less diverse than ImageNet-100. 

- Suggestion/Question: To robustly demonstrate the scalability and necessity of both regularization terms across diverse data domains, please provide the complete ablation study (similar to Table 2) for the ImageNet-100 dataset. This would confirm whether the optimal balance between second-order regularization and semantic entropy holds consistent or requires domain-specific tuning.

5. Instantiation of the Second-Order Distribution Q (Section 3.1): The paper defines the predicted second-order distribution Q^{(i,k)}\in P(\Delta^{V-1}) as a distribution over first-order distributions. It is later stated that Q is instantiated with a Dirichlet family Q=Dir(\alpha) for tractability, yielding closed-form first-order moments θ. 

- Question: Does the neural network θ directly parameterize the Dirichlet concentration vector α for Q^(i,k) at each prediction step (i,k)? If so, what is the output dimensionality of the final layer of the VAR model, and what non-linearity (e.g., softplus) is applied to ensure α components are positive? If not, how is the second-order object Q derived from the standard logit output that normally produces the first-order distribution p_θ^(i)?

6. Specific Discrepancy Measure d(\cdot, \cdot) used for L_{2nd} (Equation 4): The Second-Order Regularizer L_2nd (Equation 4) is defined using a discrepancy measure d(σ^2, σ_*^2). The text notes that d can be the squared difference, KL divergence, or a Wasserstein distance. 

- Question: Which specific measure, d(⋅,⋅), was used to obtain the quantitative results reported in Table 1 (FID 46.97)? Since the theoretical justification focuses on the Wasserstein link (Proposition 3.0.2), was the squared difference of variances used as the practical, tractable implementation of L_2nd? Please specify the exact functional form of d(⋅,⋅).

7. Role of the Codebook Ground Metric d_0 in L_{SE} (Section 3.2): The theoretical foundation introduces a codebook ground metric d_0(u,v):=∥e_u−e _v∥_2 based on VQ-VAE embeddings. This metric defines semantic proximity and induces Wasserstein metrics (Section 3.2). The Semantic Entropy loss, however, relies on predefined, hard equivalence classes C. 

- Question: How does the geometry induced by the ground metric d_0 directly influence the calculation or construction of the hard equivalence classes C used in L_SE? Are the classes C defined purely based on reconstruction quality (semantics), or are they derived by clustering nearby tokens using d_0 distance in the embedding space?

8. Scale-Specific Optimization vs. Shared Codebook Constraint (Section 2.1 & 2.3): The VAR model uses a shared codebook Z across all scales k∈{1,…,K} (Section 2.1). Yet, the reference variance σ_*^2(k) in L_2nd is designed to be scale-dependent (Equation 4). 

- Question: Given the shared codebook, how does the model effectively manage different optimal dispersion levels σ_*^2(k) (e.g., high uncertainty for coarse structure prediction vs. low uncertainty for fine details) without conflicting gradients arising from the L_2nd term, which operates on the same token vocabulary V regardless of scale? Does the model enforce a simple monotonic relationship (e.g., decreasing variance) as k increases?

9. Impact of Sampling Strategy on Diversity (Section 4.1): The reported results are achieved using standard constrained sampling techniques, specifically Top-k (k=900) and Top-p (p=0.95) (Section 4.1, line 330). The paper's main contribution is enhancing diversity by preserving natural dispersion (uncertainty preservation). 

- Question: Since the goal is to prevent the inherent distribution collapse caused by CE, which should ideally allow for high-quality ancestral sampling without heavy truncation, how do the FID/IS scores of "Our Model" compare to the baseline VAR when no aggressive sampling constraints (i.e., Top-k/Top-p) are applied? This comparison would better isolate the intrinsic diversity improvement achieved by the training objective L.

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
3

### Summary
This paper proposes an uncertainty-preserving training framework for Visual Autoregressive (VAR) models. The key idea is to move beyond standard cross-entropy objectives, which often produce overconfident and sharp token predictions that reduce sample diversity. The authors introduce two complementary regularization terms: a second-order uncertainty loss that penalizes deviations in the variance of predictive distributions from a calibrated reference, and a semantic entropy loss that aligns uncertainty with semantically equivalent codebook tokens. Theoretically, the paper connects variance regularization to the Wasserstein geometry of second-order distributions. Experiments on AFHQ and ImageNet-100 demonstrate moderate improvements in FID and Inception Score over baseline VAR models.

### Strengths
1.	The paper introduces a novel perspective on uncertainty modeling for autoregressive visual generation by explicitly controlling second-order dispersion rather than only first-order likelihoods.
2.	The proposed semantic entropy loss creatively adapts ideas from language modeling to the vision domain, promoting semantically consistent diversity.
3.	The method is simple and computationally efficient, requiring minimal architectural changes while improving generation fidelity and diversity.

### Weaknesses
1.	**Limited technical novelty.**
The main contribution lies in adding two auxiliary loss terms, which extend existing uncertainty regularization ideas rather than introducing a fundamentally new framework.

2.	**Small improvement on ImageNet-100.**
While results on AFHQ are encouraging, the performance gain on ImageNet-100 is marginal, suggesting limited robustness and general impact.

3.	**Scalability and generality concerns.**
While VAR is currently a strong autoregressive (AR) baseline, the proposed uncertainty-preserving losses should in principle apply to other AR architectures (e.g., text-to-image models). However, the paper only evaluates on the smallest VAR configuration (VAR-d16). Given the scaling law behavior of VAR models, it is unclear whether the observed gains will persist or even diminish as model size increases.

In summary, while the paper presents an interesting perspective on incorporating uncertainty preservation into visual autoregressive models, the technical contribution is relatively incremental, and the performance improvements are not significant. Moreover, the experimental evaluation is not comprehensive enough to convincingly demonstrate the generality or robustness of the proposed approach.

### Questions
I checked the original VAR paper, where the reported FID for VAR-d16 on ImageNet is 3.30, while in this paper, it is around 20. The Inception Score also differs significantly. What could be the reason for such a large performance gap? Is the large discrepancy simply because this paper uses the ImageNet-100 subset?

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
3

### Summary
This paper introduces uncertainty preservation objectives for generative visual autoregression. Prior methods use a cross-entropy loss which can cause overconfident predictions and loss of diversity in generated samples. To counter this, they propose a second-order uncertainty preservation objective that penalizes deviation from a calibrated reference distribution. They also provide theoretical support for the proposed method. Finally, with experiments on ImageNet-100 and AFHQ, they find improved FID scores and Inception scores compared to the baseline.

### Strengths
* The limitations of VAR presented makes intuitive sense and the proposed approach is also supported by theoretical analysis.

* The paper is fairly easy to read and follow. As a minor thing, I would have appreciated an overview figure showing the overall pipeline of VAR (and the added losses).

### Weaknesses
* Compared to the original paper on VAR by Tian et al. (NeurIPS 2024), the baseline results seem to be much worse. For example, Tian et al. reports ImageNet-1k (256x256) FIDs to be 3.3 and Inception score to be 274.4 (for the same VAR-d16 model that this paper uses). But this paper's baseline result on ImageNet-100 (which should be somewhat easier than ImageNet-1k) is 20.45 for FID and 15.2 for IS (significantly worse than 3.3 and 274.4). Hence, it is unclear to me if the baselines were properly implemented or trained for enough time or with the proper hyperparameter tuning. And then the improvements by adding the proposed losses are also questionable since the baseline itself is not good enough.

* It is unclear to me why the proposed approach applies only to the VAR method. Could it also apply to other autoregressive models like [W1-W2] or to diffusion models [W3-W4] (these are compared in the original VAR paper, so I suggest also having these and other generative models like diffusion models in the table)? Showing applicability to more generative models would strengthen the results significantly.
    * The results in this paper are only limited to 256x256 resolution and a single model VAR-d16, while original VAR showed results on 512x512 as well as on different model sizes. It would be good to have at least one result each at higher resolution and with a different model size to show that the approach works even when the resolution or model size is scaled.

* Minor issues (typos):
    * L212: "a semantic-aware reference distributions leverage the geometry..." $\to$ "a semantic-aware reference distribution that leverages the geometry..."
    * L215: "distributions" $\to$ "distribution"

### References

* [W1] Lee et al., "Autoregressive image generation using residual quantization", CVPR 2022

* [W2] Yu et al., "Vector-quantized Image Modeling with Improved VQGAN", ICLR 2022

* [W3] Peebles and Xie, "Scalable diffusion models with transformers", ICCV 2023

* [W4] Rombach et al., "High-resolution image synthesis with latent diffusion models", CVPR 2022

### Questions
Please clarify my questions from the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces the notion of uncertainty preservation for visual autoregressive models, where the core idea is to incorporate second order uncertainty/regularization, which goes beyond just labels and looks at the dispersion of the whole distribution. The authors also claim to introduce the notion of semantic entropy to visual generative modeling.

### Strengths
Originality: The second order uncertainty/regularization idea for visual generation is original to the best of my knowledge. 

Clarity: The paper is motivated well. However there are unclear parts.  Details on this are given in the next section  

Quality: I find the quality of the idea to be fair, even though in terms of presentation there is room for improvement (see section below). I address the quality of experiments/empirical work in the weaknesses section. 

Significance: The topic of study, which is developing better visual generative models that have better uncertainty preservation, is a timely and high significant topic. The method proposed is original, but I discuss the significance of that in the questions section.

### Weaknesses
1. Much of the foundation of the paper depends upon visual autoregressive modeling (Tian et al 2024), which in turn depends upon a literature on VQ-VAEs. However, the background that the authors provide is minimal (with no supplement/appendix that detail this). This makes the paper less self contained for a broad ML audience, which negatively impacts the clarity of the paper. 

2. The authors claim that they "introduce semantic entropy from language modeling to
visual generation", but they seem to have missed some potentially relevant citations (Note: I am not saying that the authors are not the first. I am just saying that there are relevant papers with adjacent ideas, and that if the authors could further clarify how they are different this would make the paper more clear and make their case more convincing)  For example, 

Vision-Amplified Semantic Entropy for Hallucination Detection in Medical Visual Question Answering by Liao et al 

Semantic Entropy Can Simultaneously Benefit Transmission Efficiency and Channel Security of Wireless Semantic Communications

SEEN-DA_SEmantic ENtropy guided Domain-aware Attention for Domain Adaptive Object Detection by Li et al

3. If the goal is uncertainty quantification, then the experimental section seem to be insufficient to justify the claims of improvement in this paper. the authors compared with one baseline competitor over two datasets in their main experiment. They used FID and IS as the metrics. However, ample work in the literature show that FID and IS are not good metrics to capture uncertainty/dispersion! I highly suggest looking into the literature to find more statistically principled/uncertainty aware metrics of comparison

4. there are many typos and omitted proofs, statements and details that undermine the quality of this paper. In many crucial places, the paper is vague and do not offer proofs/rigorous/sufficiently detailed statements.  More on this in next section.

### Questions
several parts of the paper are not clearly conveyed, and I am some minor questions as well, which are all collected below. 

1. In line 195, when p_\theta^{(i)} is defined, is there a reason why no k index is included in the notation? 

2. Certain claims, e.g. those regarding Wasserstein connections in line 220-231, are not given references nor rigorously proven. Not all readers are familiar with these results, so either the claims have to be substantiated in the appendix/supplement, or references should be given. 

3. In 197, the authors mention functionals of p_\theta^{(i)}. However, they only consider the variance. Why not other functionals, such as other moments in addition to the variance? 

4. Line 229 and 230 contains inconsistent notation (which is it, q or Q) and spelling mistakes. 

5. Given that the Gaussian is continuous, but the token distribution is discrete, I do not see why we want reference q* to be approximately Gaussian in line 215? You use q and Q inconsistently. Also, why is the Guassian a good/reasonable approximation to the Dirichlet? 

 6. Propositions 3.0.1 and 3.0.2 are mentioned in line 231, but the statements are stated in general/vague terms, with no proofs offered.

### Soundness
1

### Presentation
2

### Contribution
2
