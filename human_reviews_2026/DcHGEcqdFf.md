# Disentanglement of Variations with Multimodal Generative Modeling

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6

## Abstract
Multimodal data are prevalent across various domains, and learning robust representations of such data is paramount to enhancing generation quality and downstream task performance. To handle heterogeneity and interconnections among different modalities, recent multimodal generative models extract shared and private (modality-specific) information with two separate variables. Despite attempts to enforce disentanglement between these two variables, these methods struggle with challenging datasets where the likelihood model is insufficient. In this paper, we propose Information-Disentangled Multimodal VAE (IDMVAE) to explicitly address this issue, with rigorous mutual information-based regularizations, including cross-view mutual information maximization for extracting shared variables, and a cycle-consistency style loss for redundancy removal using generative augmentations. We further introduce diffusion models to improve the capacity of latent priors. These newly proposed components are complementary to each other. Compared to existing approaches, IDMVAE shows a clean separation between shared and private information, demonstrating superior generation quality and semantic coherence on challenging datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Information-Disentangled Multimodal VAE (IDMVAE), a new multimodal generative framework designed to improve disentanglement between shared and modality-specific representations.

### Strengths
1. The cross-view regularization for disentangling the shared latent variable z and modality-specific w is well-motivated.

2. The experiments consider cross-domain data, and the ablation studies are complementary.

3. Using diffusion to model latent priors results in better alignment between prior and posterior distributions, leading to improved unconditional generation and coherence.

### Weaknesses
1. Compared to multimodal VAEs, the novelty is based on the cross-view regularization. For the diffusion part, CMVAE also integrates the diffusion model for improved performance. The conceptual advance is relatively modest.

2. The paper does not provide training efficiency comparisons or runtime analyses (especially for the diffusion part). It unclear whether the performance comes from the model complexity.

3. While figures illustrate disentanglement, generated samples on CUB are of relatively low fidelity, suggesting the model’s generative capacity is still limited.

### Questions
1. Can you provide a way to visualize the learned shared latent variable and modality-specific variables? Comparing the visualizations for the MMVAE+ and the proposed method could be straightforward.

2. Can this method scale up to a large-scale Multimodal dataset (e.g., MSCOCO)? 

3. How does the proposed method compare to other diffusion models in terms of, for example, CLIP score?

4. How do you handle the text input? Any pre-trained embedding models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the Information-disentangled Multimodal VAE (IDMVAE), a new model for learning from data with multiple modalities. IDMVAE achieves this improved disentanglement by using Mutual Information (MI) regularizations and latent diffusion priors. They try to learn a common latent z that captures all the shared information encouraged by the added new regularization losses introduced in this work. In addition, they utilize diffusion models to create more complex and capable latent priors, thereby moving beyond simple Gaussian priors and enhancing the model's representational capacity. The authors demonstrate that IDMVAE achieves better shared and private information, resulting in superior generation quality and enhanced semantic coherence on some simple datasets compared to existing methods.

### Strengths
The paper is well-written and well-presented. The paper also seems to have strong theoretical grounds for the idea they presented in this work. Their MI-based regularization methods could be useful for multimodal VAE works in general. The proposed losses are also useful for learning a shared representation, which is important for downstream tasks. Their proposed idea of using practical contrastive loss for cross-view MI maximization and their disentanglement learning with generative augmentation losses is a good contribution. They propose using diffusion models for learning complex latents, but this has already been proposed and implemented in other works (look at the weaknesses part).

### Weaknesses
Even though the paper is well presented, there are considerable weaknesses in this paper that render the contribution marginal.


1. $ \textbf{Used datasets are toy datasets} $  

The datasets used in this work are very small and not significant enough to show the practicality of the presented method. I encourage the authors to look at [1], which is also a work done on multimodal VAEs using a score-based model that used the CelebAMask-HQ multimodal dataset, which is larger than the datasets used in this paper, as a starting point. The paper is also not cited and not compared as a baseline. The authors can also add other larger datasets than this to prove their method is still useful in these settings.

2. $\textbf{Latent diffusion proposal}$

 Using a latent diffusion/score-based model to learn complex latents for multimodal VAEs is also presented in [2]. But it hasn't been cited in this work and presented as a new thing.

3. $\textbf{Limited baselines}$

 The baselines compared in this work are limited. Additional recent baselines should also be added to the work. E.g., [1,2,3,4]. 

4. $\textbf{Evaluation of Quality}$

 The paper doesn't use FID for evaluating the quality of generated unconditional and conditional images as in [1]. I believe a quantifiable measure of quality is important in multimodal VAE works.

5. $\textbf{Evaluation of method of PoE Multimodal Models}$

 The paper proposes that the method can be used in PoE/MoPoE multimodal VAE models but those haven't been explored as ablations. Can the authors try their method in at least one setup of these multimodal VAEs as a proof of concept or ablation study.


References
1. Wesego, Daniel, and Amirmohammad Rooshenas. "Score-based multimodal autoencoders." arXiv preprint arXiv:2305.15708 (2023).
2. Wesego, Daniel, and Pedram Rooshenas. "Multimodal ELBO with Diffusion Decoders." arXiv preprint arXiv:2408.16883 (2024).
3. Bounoua, Mustapha, Giulio Franzese, and Pietro Michiardi. "Multi-modal latent diffusion." Entropy 26, no. 4 (2024): 320.
4. Palumbo, Emanuele, Laura Manduchi, Sonia Laguna, Daphné Chopard, and Julia E. Vogt. "Deep generative clustering with multimodal diffusion variational autoencoders." In The Twelfth International Conference on Learning Representations. 2024.

### Questions
Please review the Weaknesses section and attempt to address the points raised there.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper “Disentanglement of Variations with Multi-Modal Generative Modeling (IDMVAE)” proposes a new multimodal variational autoencoder designed to cleanly separate shared and private sources of variation across data modalities. Standard multimodal VAEs, such as MMVAE and MoPoE, often suffer from latent leakage, where modality-specific information contaminates the shared latent space, causing incoherent cross-modal generation and poor disentanglement. IDMVAE introduces three complementary mechanisms to address this.

1.	A cross-view mutual information maximization term aligns shared representations by encouraging different modality-specific encoders to capture the same underlying latent factors.
2.	A generative augmentation cycle removes redundancy between shared and private latents by decoding “mixed” latent pairs into synthetic samples and enforcing that shared latents remain invariant while private latents retain modality-specific structure.
3.	A latent diffusion prior replaces the standard Gaussian prior with a learned diffusion model that can represent complex, multimodal latent distributions, reducing prior–posterior mismatch and improving unconditional generation quality.

The resulting objective extends the MMVAE+ ELBO with information-theoretic and generative-consistency regularizers, all trainable end-to-end. Experiments on PolyMNIST-Quadrant, CUB (image–text), and TCGA (multi-omics) show that IDMVAE achieves substantially cleaner separation of shared vs. private factors, higher conditional and unconditional coherence, and improved latent alignment compared to strong baselines. Ablation studies demonstrate that cross-view MI alignment improves cross-modal reconstruction, the generative augmentation reduces leakage, and the diffusion prior enhances sample quality.

### Strengths
Originality:\
Introduces a principled formulation of multimodal disentanglement by combining cross-view mutual-information maximization with generative augmentation, directly targeting the separation of shared and private factors.\
Conceptually unifies ideas from contrastive learning, information bottleneck, and generative modeling in a single probabilistic framework.

Quality:\
Methodologically robust: each loss term is clearly motivated by known weaknesses of multimodal VAEs (e.g., posterior collapse, shared–private leakage).\
The generative augmentation mechanism leverages the model’s own decoder for cycle consistency, avoiding handcrafted data augmentations\
Comprehensive experiments across synthetic (PolyMNIST-Quadrant) and real (CUB, TCGA) datasets demonstrate consistent improvements in disentanglement, coherence, and generation quality.\
Ablation studies isolate the contribution of each component (CrossMI, GenAug, diffusion prior) and show interpretable, consistent improvements.\

Clarity:\
The paper is well-structured and clearly written, with each component (CrossMI, GenAug, diffusion prior) well motivated and described.\
Mathematical notation and objective formulation are concise and consistent.

Significance:\
Addresses a challenge in multimodal learning of separating  shared vs. private information with a practical, modular solution applicable to existing architectures.\
Demonstrates that learned latent priors (diffusion models) can substantially improve multimodal VAEs, opening a new direction for hybrid diffusion–VAE research. However, it lacks a clear comparison with other multi-modal (latent) diffusion models, such as [1,2].

Refs: 
[1] Bounoua, Mustapha, Giulio Franzese, and Pietro Michiardi. "Multi-modal latent diffusion." Entropy 26.4 (2024): 320. \
[2] Chen, Changyou, et al. "Diffusion models for multi-modal generative modeling." arXiv preprint arXiv:2407.17571 (2024).

### Weaknesses
The connection between maximizing $I(z_m; z_n)$ and achieving conditional sufficiency $I(x_m;x_n|z)=0$ remains heuristic. Under what assumptions (e.g., conditional independence or additive latent structure) do the suggested losses approximate or minimal-sufficiency conditions.\
The InfoNCE estimator can be sensitive to negative-sample size, temperature, and batch composition. The paper fixes these hyperparameters without ablation or analysis.\
Adding contrastive, augmentation, and diffusion losses likely increases computational cost. The paper lacks a discussion of these computation costs.\
You compare to MMVAE variants but not to contrastive approaches such as [3-4], or as mentioned above, latent diffusion models [1-2]. \
The generative augmentation term is described as reducing redundancy between z and w_m, but the paper does not analyze how effectively this enforces $I(z; w_m) \to 0$ in the experiments.



Refs: \
[3] Liang, Paul Pu, et al. "Factorized contrastive learning: Going beyond multi-view redundancy." Advances in Neural Information Processing Systems 36 (2023): 32971-32998.\
[4] Dufumier, Benoit, et al. "What to align in multimodal contrastive learning?." arXiv preprint arXiv:2409.07402 (2024).

### Questions
1.	Is the diffusion prior applied only to the shared latent z, or also to private w_m? If only to z, why is a simple Gaussian sufficient for the private components?
2.	How do the MI and cycle constraints correspond to orthogonality or covariance restrictions in linear Bayesian joint additive factor models such as [5]?
3.	The paper does not mention a β-VAE coefficient. Is β fixed at 1, and do $\lambda_{\text{MI}}$, $\lambda_{\text{GenAug}}$ effectively play this role? How do these hyperparameters influence the average posterior entropies $H[q(z|x_m)] $ and $H[q(w_m|x_m)] $ over training.
4.	Some VAE models e.g. [6] address the inference-consistency problem in multimodal VAEs, i.e. the inability to obtain coherent subset posteriors such as $p(w_2 \mid x_1)$ via a tighter ELBO and flexible encoder aggregation. Does IDMVAE provide an alternative route to this goal by enforcing independence between $w_m$ and the shared latent z, thereby avoiding aggregation bias or looser bounds for modalities with high conditional mutual information?
5.	The CrossMI loss is motivated by encouraging $I(z_m;x_n)\approx I(x_m;x_n)$, implying that the shared latent fully explains cross-modal dependence. Would this not render private latents redundant? How is this balanced to prevent collapse of private variation, and does this depend on dataset redundancy?
6.	Integrating a diffusion prior end-to-end into a multimodal VAE can introduce instability, as seen in recent works like REPA-E [7] or uses some annealing tricks [8]. The paper does not discuss training schedules, warm-up strategies, or whether gradients from the diffusion loss are stopped for encoder updates.

Refs:
[5] Anceschi, Niccolo, et al. "Bayesian joint additive factor models for multiview learning." arXiv preprint arXiv:2406.00778(2024).\
[6] Hirt, Marcel, et al. "Learning multi-modal generative models with permutation-invariant encoders and tighter variational objectives." Transactions on Machine Learning Research.\
[7] Leng, Xingjian, et al. "Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers." arXiv preprint arXiv:2504.10483 (2025).\
[8] Vahdat, Arash, Karsten Kreis, and Jan Kautz. "Score-based generative modeling in latent space." Advances in neural information processing systems 34 (2021): 11287-11302.

### Soundness
3

### Presentation
3

### Contribution
3
