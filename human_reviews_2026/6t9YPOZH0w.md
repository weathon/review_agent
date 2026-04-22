# Secure Inference for Diffusion Models via Unconditional Scores

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
As diffusion model-based services expand across various domains, safeguarding client data privacy has become increasingly critical. While fully homomorphic encryption and secure multi-party computation enable privacy-preserving inference, their high computational overhead poses challenges for large-scale diffusion applications. Recent work alleviates computational costs by substituting non-linear operations with low-degree polynomial approximations. While such relaxations reduce latency, they incur significant degradation in generative fidelity, and inference remains considerably slower than plaintext execution. To further accelerate secure inference while preserving performance, we explore more relaxed approximations and propose a score-correction framework that rectifies the conditional score shift induced by the relaxed approximation, rather than decreasing the approximation error itself. The key insight is that unconditional generation can be executed without approximation and thus provides a high-fidelity score signal. Leveraging this unconditional score as corrective guidance enables more relaxed approximations while preserving semantic and perceptual quality. In experiments, we demonstrate that our method significantly alleviates the performance degradation caused by relaxed approximations across various benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies efficient implementation of inference for diffusion models (IDM) under secure multi-party computation (MPC). A computational challenge encountered in previous work (CipherDM, Zhao et al., 2024) is that inference requires the computation of non-linear functions on the ciphertext, which is known to be computationally intensive. CipherDM addressed this issue by switching to polynomial approximations of the non-linear function, and so improving the scalability.

This paper revisits CipherDM with a coarser polynomial approximation, showing improved efficiency at the cost of a further deterioration in output quality. To mitigate the drop in output quality, the authors propose to augment the inference process by using unconditional and exact generation in plaintext to “steer” the diffusion process. The key idea is that the unconditional generation has no sensitive content, and so this signal can be computed fast and exactly in plaintext, and can consequently be used to guide the diffusion process to correct for errors induced by the polynomial approximation at a low computational overhead. Experiments confirm that the augmented signal can compensate for much of the approximation error, even to the point of beating the exact baseline in some cases.

### Strengths
1. The paper is well-written with a clear problem statement supported by nice figures.
2. The key idea of using unconditional sampling in plaintext space to (with small overhead) inform the generation process is natural and clever.
3. The problem is well-motivated and fits the scope of ICLR.
4. The experimental results are strong, and suggest that activation functions can be polynomially approximated at little to no cost in image quality (Table 1 and 2).
5. The empirical setup is given in enough detail (including an attached code artifact) to be reproducible.

### Weaknesses
1. The alternative of fine-tuning the DM with poly-approximated activations is not considered in depth. Since the main contribution of the paper is to make secure inference for DM more efficient, it would be fitting to have a more extensive discussion of this alternative approach and its accuracy. Line 212-214 correctly makes the point that the corresponding fine-tuning of large DMs is expensive, but one could nevertheless imagine large industry actors training DMs for the express purpose of enabling high-performance secure inference. If there is a gain in output accuracy from fine-tuning, they may find the associated cost worth taking on. See later questions.
2. (Conditioned on the validity of the previous point) If fine-tuning the DM with poly-approximated activations can achieve better accuracy than the proposed method in this paper, then the impact of the contribution is reduced. The authors improve the accuracy/efficiency trade-off for MPC-IDM only in the particular case where a diffusion model is not trained with MPC in mind, and is then “retrofitted” for the purpose. Nevertheless, I concede that (1) this might be an important case, and (2) in the case of precise approximation the empirical work suggests that fine-tuning the model may be excessive.
3. A minor point. While it is mentioned that CipherDM (Zhao et al., 2024) also employed polynomial approximation of activation functions (line 49) to improve performance for secure inference, the details of their approximation are (as far as I can tell) not discussed. Section 3.1 discusses polynomial approximations (“precise” and “loose” variants), but does not relate this to CipherDM. For completeness, it could be nice to include their approximation scheme in the appendix. It would also be useful from the point of view of having more interpretability of Table 4, here the signal steering, while helpful, appears to still lag far behind the baseline---is it due to the coarseness of the approximation in CipherDM? I don't think this can be inferred from the paper currently.

### Questions
I find the paper’s topic of efficient secure inference for diffusion models interesting, its proposed technique clever and its writing of high quality. The empirical work is clean and appears reproducible. My only concern is related to fine-tuning as an alternative to the “signal steering” approach proposed, which I think is dismissed with insufficient motivation. As things stand, I lean towards acceptance, and feel more inclined to raising my score than lowering it.

I have some questions. As they are a few, feel free to give brief answers where you see fit.
1. My understanding of your approach is that you use the signal from the “uncondtional+exact” latent diffusion model to steer the approximated latent diffusion model to correct for approximation errors. Nevertheless, from Figure 3 it seems like your text encoder and decoder are also approximated. Did you try investigating how much of the drop in image quality is due to these particular approximations for your approach? If your technique corrects for most of the approximation error in the latent model, then I would expect the performance to increase if the encoder/decoder are exact, at least in the case of the coarse approximation.
2. Related question: with no approximations, how much (roughly) of the latency is driven by the encoder/decoder step from Figure 3? My understanding is that Table 1 and 2 do not measure their influence, so I am curious how important they are to approximate.
3. What is the cost for instead fine-tuning the DM with polynomially-approximated activations driven by? Is it only the cost of fine-tuning a “normal” DM, or is there some additional cost that is imposed from the setting of secure inference?
4. You observe in Table 1 and 2 that your signal steering can not only compensate for the approximation error, but improve on the baseline if the approximation is precise enough. You hypothesize in Section 4.1 that this improvement could be driven by that your steering has a stabilizing influence on the process. Did you consider using your steering technique on the *exact* baseline itself? It would be interesting to see if the 'vanilla' model + steering also led to some improvement.
5. In the related work, you mention sampling guidance (Hong et al., 2023; Ahn et al., 2024). Could your idea of "signal steering" with the exact model in plaintext be combined with their techniques, or are they incompatible? Can "sampling guidance" be implemented without additional computation on ciphertext?
6. While it might be computationally expensive in general, did you consider testing fine-tuning of the DM for e.g., the MNIST dataset? E.g., fine-tune CipherDM and see how it compares with the results in Table 4?
7. Is the last sentence of the caption to Table 1 related to FID and CLIP-Score important for interpreting the accuracy results (computations being done in plaintext)? I would expect the reported image quality to be independent of this detail.

### Soundness
3

### Presentation
4

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
This paper proposes a novel add-on to an existing privacy-preserving inference method (CipherDM) by addressing the issue of high latency and low fidelity of generated images. The key contribution is that the proposed method leverages the score guidance from the unconditional generation, which provides a higher fidelity signal for image generation. The results show that the image quality has increased after applying the new method on top of CipherDM.

### Strengths
1, The paper is well written and easy to follow.

2. The idea of using the unconditional generation as a guidance signal is interesting and novel. 

3. Through experiments, the authors have shown that the new add-on to the CipherDM can effectively increase the fidelity of images.

### Weaknesses
1, All experiments are conducted on a pretrained Stable Diffusion v1.5 Model, while the CipherDM paper was mainly evaluated on DDPM and DDIM. It might be better to address the transferability issue of the proposed method across different diffusion architectures. 

2. In parallel with private inference, another line of research called differentially private diffusion models are serving for the same purpose of privacy preservation. Recent works such as DPDM (https://arxiv.org/abs/2210.09929) and RAPID (https://iclr.cc/virtual/2025/poster/28006) are also trying to provide privacy guarantees to diffusion models. From both papers, I can see that their FID scores on MNIST dataset are much lower than this work or CipherDM. It might be more helpful to address this issue by either comparing their performances through experiments under the same setting or addressing their differences in a separate subsection.

### Questions
1. Would it be possible to apply the proposed method in this paper on traditional structures such as DDPM or DDIM? What will be the effect? 

2. What is the difference or pros and cons between using private inference or Differentially Private Diffusion Models?

### Soundness
3

### Presentation
3

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
The paper targets private inference for conditional diffusion under MPC/HE by replacing non-polynomial ops (SiLU, GeLU, exp/softmax) with piecewise polynomial approximations. Relaxing these approximations lowers latency but corrupts the conditional score; the core idea is to use a plaintext, high-fidelity unconditional score (computed without approximations) to correct the encrypted conditional score during sampling. The method derives a guidance update, yielding a simple correction term. Experiments on Stable Diffusion (MS-COCO, Flickr8k) and CipherDM (binary MNIST) show reduced degradation.

### Strengths
1. **Clear presentation and reasonable motivation**
    - The paper motivates the problem well: because the text condition is inaccessible during secure inference, the authors refine the conditional path by injecting a clean, uncorrupted unconditional score as a corrective signal.

2. **Simple, architecture-agnostic mechanism**
    - The method is a small, drop-in change to the sampler and is model-agnostic. And it only introduces essentially two scalar knob: the guidance strength and the learning rate.

3. **Good experimental design with clear effectiveness** 
     - The experiments probe both sides of the trade-off (latency/communication vs. fidelity) across multiple approximation regimes and datasets, and consistently show that the guidance improves quality to relaxed polynomial approximations.

### Weaknesses
1. **Minor quality gains at similar cost** 
    - Relative to no score guidance, the improvements are modest, and under the loose approximation, even with the proposed guidance, the generation quality still lags behind the precise approximation without guidance at roughly comparable latency (cf. Tables 1–2). 
2. **Incremental contribution; missing comparisons to alternative test-time fixes**
    - The method functions as a test-time refinement/scaling of the sampler, and the “implicit discriminator” view echoes prior discriminator-style interpretations of likelihood models (albeit those are usually applied during training rather than at test time). To justify the novelty and practical value, it would be helpful to compare this approach with other lightweight test-time or post-sampling corrections.

### Questions
1. **Iterative rectification within a step**
    - Since the guidance explicitly rectifies the score, can it be applied iteratively within the same time step? If so, does increasing the number of inner refinements (k=1,2,3,…) improve fidelity in a cost-aware manner.

2. **Few-/one-step regimes**
    - As diffusion/flow models move to few-step or even one-step samplers (e.g., consistency models, meanflow), how does the method specialize? In a single-step setting, is there a principled way to form the unconditional reference (score/velocity) and apply your correction so that it still yields a measurable gain?

3. **Budgeted trade-off vs. more steps**
    - The method is a test-time scaling that adds overhead. Under a fixed latency/communication budget, how does guidance compare to simply adding a few extra sampling steps (or slightly tightening the polynomial approximation) without guidance?

### Soundness
3

### Presentation
3

### Contribution
2
