# Aligning Latent Spaces with Flow Priors

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
This paper presents a novel framework for aligning learnable latent spaces to arbitrary prior distributions by leveraging flow-matching generative models as priors. Our method first pretrains a flow-matching model on the prior features to capture the underlying distribution. This fixed flow model subsequently regularizes the latent space via an alignment loss, which reformulates the flow matching objective to treat the latents as optimization targets. We formally prove that minimizing this alignment loss establishes a computationally tractable surrogate objective for maximizing a variational lower bound on the log-likelihood of latents under the prior distribution. Notably, the proposed method eliminates expensive likelihood evaluations and avoids ODE solving during optimization. As a proof of concept, we demonstrate in a controlled setting that the alignment loss landscape closely approximates the negative log-likelihood of the prior. We further validate the effectiveness of our approach by regularizing the latent spaces of autoencoders in large-scale ImageNet image generation, with diverse prior distributions, accompanied by detailed discussions and ablation studies. With both theoretical and empirical validation, our framework paves a new way for latent space alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The works aims at aligning the latent space of a generative model to arbitrary prior distributions. The approach consist to (1) train a flow-matching model on features to have an estimate of the arbitrary prior (2) allign the latent space to this learnt prior through mean square error minimization of the latents.

The approach is evaluated through a toy example (mixture of 2D Gaussian) and image generation with a ViT based auto-encoder and ImageNet $256\times 256$ images, with four different types of priors modelled by a flow prior consisting of a 6 layer multi layer perceptron, the image generation being processed with a masked autoregressive model. The alignement is estimated by comparing the change during training of the proposed alignement loss with an estimate of the negative log likelihood (this last relying on k-NN density estimation). Then the results of the generation with these prior and a masked autoregressive model are evaluated according to several metrics reflecting the reconstruction and the quality of images, as well as their similarity to the ImageNet classes.

An ablation studies the robustness of the approach with regards to important hyper-parameters, using the prior with textual embeddings.

### Strengths
* the toy example with a mixture of five isotropic 2D Gaussian (section 5.1) nicely illustrated the proposed approach and helps the reader to understand its principle. The paper also demonstrates a genuine desire of clarity by providing an "intuitive explanation" of the method in section 4.2.

* for image generation, the experiments are conducted with four very different prior, namely low-level and semantic embedding (visual features), quantified visual features and even textual features.
  - the choice of a masked auto-regressive model is relevant, although one can regret that any experiment was conducted with diffusion-based models (but this choice is justified in the manuscript).
  - the experiment also independently estimates the incluence of classifier-free guidance in this context.

* the proposed approach allows theorically to get a tractable approach to maximize a variational lower bound on the log-likelihood of latents under the prior distribution. The formal proof is provided in a simple case in the mainpaper and further precise development are given in the appendix.

### Weaknesses
* results of alignment in section 5.2 (line 397-412) are made through the *observation* of two curves and commenting their (asuumed) correlation, without reporting this last. There is no comparison to any baseline.

* the result of image generation (section 5.2 and Table 1) are not really convincing. If one considers the results with classifier free guidance (that is, the best) the performance are close to the basic prior (AE, KL, SoftVQ) and sometimes worse, in particular in terms of Precicion and Recall. As well:
  - The results are reported for one set of generation only and it is thus to estiamte the significance of the results. It may have been relevant to estimate a standard deviation, at least for some of the models (e.g the AE and the Dino that gives the best results)
  - the fact that the textual feature lead to results comparable to other prior (best for IS withour CFG and Recall with CFG) is interpreted by authors (appendic D, line 71156-1162) as "[suggesting]  that high-quality textual representations capture abstract semantic structures that are transferable across modalities. The 896-D Qwen embeddings provide a rich semantic space that can effectively constrain the visual latent space without being overly restrictive.". However, this assertion is not convincing since, on the contrary, one could also consider that *any* prior could lead to similar results and that it mainly shows that the proposed approach does not model anything usefull. Hence it make arguable the initial assumption of the paper, namely that modeling an arbitrary prior is interesting
  - by the way, the results in terms od IS and FID for $256\times 256$ imageNet images is much better in the original paper of MAR: cf. Table 4 of (Li et al., 2024a) for AR/MAR-B: FID=3.48 and IS=192.4 without CFG and FID=2.31 IS=281.7 with CFG. The precision is also much better in both case. The proposed approach is only competitive in terms of recall whre it is better with CFG and on-par (depending on the prior...) without CFG.

* it is surprising to conduct the ablation studies (section 5.3) with the textual embedding (Qwen) as prior since it is not the model that performs best nor the most "natural" to choose to generate images (settings of Tab 1: cf. line 461)

**minor**:
  - on line 397-412, the value of $k$ (for $k$-nearest neighbors desity estimation) is not reported
  - several references are arxiv preprints while the article has been published e.g (Nichol and Dhariwal, 2021) at ICML 2021 or (Lipma et al., 2022) at ICLR 2022. All preprint that have been further published should refer to the reviewed paper.
  - the "simple" baselines AE, KL and SoftVQ (Table 1) are not presented in the paper
  - some assumptions in theoretical derivation are arguable
    - in the proof of proposition 1, it is assumed that $\lambda=1$ (line 282) that seems quite high in practice. For example, in the abblation study (Table 2) $\lambda$ is less than 0.05. However the appendix provides a proof for any arbitrary positive $\lambda$ for all steps.
    - in practice, assumption 1 (line 311) is also arguable since it assumes that the flow matchinging minimizes perfectly equation (3). This assumption seems important to have latents that goes to the prior. Thus, in practice, it would be interesting to estimate to xhich extent this assumption is valid.

### Questions
* what is the "standard toolkit" used to compute the FID and IS (line 1133)? More precisely, on which model does it rely? For example, (Esser et al. 2024) rely on CLIP L/14 but older works rely on Inception.
* what are the score in Table 2 if $\lambda=1$ as assumed in the proof of proposition 1 ?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to use a pretrained flow model trained on the features of images to be the prior for a variational autoencoder.
The paper proposes to use a flow-matching-like objective using the pretrained flow model velocity field to approximate the prior log likelihood without running ODE solver to compute the likelihood.
The paper then runs experiments on synthetic dataset and ImageNet-1K using various fixed prior distributions.

### Strengths
- Simple (heuristic) objective that doesn't require ODE evaluation of flow models.
- Did an experiment on ImageNet-1K.

### Weaknesses
- "vθ encapsulates the dynamics that transport probability mass from the base distribution pinit to the prior distribution pprior along linear path" - I don't think the $v_\theta$ captures movement along linear paths. It is trained with linear paths but the velocity field itself does not produce a linear path. Only optimal transport maps would produce linear paths but that is not solved via flow matching.
  - Figure 2 while intuitive in this case will break down when the init and prior are overlapping but very different distributions. In this case, the flow matching vector field may be highly non-linear. I'm not convinced
  - Consider mixture of Guassians to mixture of Gaussians. A flow matching objective will map them to each other but the paths are not linear.
  - This seems to be an incorrect intuitive explanation for the objective.
  - "vθ precisely captures the dynamics required to transform initial noise samples x0 into prior features x along straight interpolation paths. Specifically, it has
learned to predict the exact velocity x − x0 at any point (1 − t)x0 + tx along such a path" - Again, I believe this is a misunderstanding of flow matching. Flow matching takes the average velocity over all possible pairings of points in a sense. It does not correctly predict the linear velocity since multiple pairs of points could produce the same latent. 

- Proposition 1 seems vacuous since you don't compute $C(z)$ and (as far as I can tell) $C(z)$ could be positive or negative. Thus, $L_{align}$ does not form a lower bound even up to constants since $C(z)$ is not constant w.r.t. $z$. If it was constant w.r.t. $z$, then it would be okay for training the encoder. But as it stands, this bound isn't meaningful. 
  - "and C(z) is dependent on z and vθ ." - This should be shown notationally as $C(z,\theta)$. This hides the fact that $C$ depends on $z$ and $\theta$.
  - "We analyze the behavior of C(z) in Appendix A to show that if z aligns with a more concentrated prior distribution (making Lalign(z; θ) small), C(z) tends to be positive and larger, contributing favorably to the ELBO." - Again, this highlights that $C(z)$ is not understood. Since $z$ depends on the encoder parameters, you cannot just ignore this $C(z)$ term.

- "Assumption 1 (Optimality of vθ ). The velocity field vθ : Rd1 × [0, 1] → Rd1 is (pre-trained) and optimal, satisfying vθ ((1 − t)x0 + tx1, t) = x1 − x0 ∀x0 ∼ pinit, x1 ∼ pprior, t ∈ [0, 1]." - Again as above, this is almost never  true for Flow matching velocity fields. Even if you use flow matching between two Gaussian distributions this will not be true. To make them "straight", you would have to do rectification steps like in rectified flow matching multiple times to converge on the optimal transport map. But, in general, flow matching objectives do not do this. This is a critically incorrect assumption.

- Fig 3--- This shows that the latents distribution does NOT converge to the prior distribution even for this toy example. But rather it converges towards the modes of the prior distribution. This aligns with the problems in the theory above.

- A broader motiviation question is why should the prior distributions be fixed? I'm still not convinced this is actually useful when learning an encoder. Can you provide a convincing example where you do an experiment and it directly improves some task? Like can you show that if you use Gaussian vs yours as a prior, it actually produces better robustness or classification accuracy for downstream tasks using linear probing? I'm not convinced arbitrary fixed priors are better than Gaussian fixed priors. It is more interesting like IAF to use learnable flexible priors but to have a fixed prior, it's not clear that this is actually helpful and is likely to hurt overall performance. This is important motivation issue with the proposed work.

### Questions
- How does this compare to the following relevant paper [Gong et la.,2025] on score-based priors for latent variable models (diffusion-based velocity fields but may be generalizable to flow-based velocity fields)? This paper also avoids running the ODE flow when training the encoder. If you just keep the velocity field fixed, then the encoder can be directly trained.

Gong, Z., Lim, J., & Inouye, D. I. Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization. In Forty-second International Conference on Machine Learning.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a novel method for aligning a learnable latent space with any arbitrary prior distribution. It first trains a flow matching model to model the prior distribution. It then optimizes the target latents to minimize an alignment loss computed by the flow model. The paper provides theoretical and empirical analyses to support the proposed technique. In particular, in the experiments, it successfully aligns latents with a ViT-L-based encoder-decoder to 4 different prior distributions, including low-level visual features from a VAE, continuous semantic visual features from DinoV2, discrete visual codebook embeddings from LlamaGen VQ, and textual embeddings from Qwen. These latents can be integrated into MAR-B models, yielding competitive conditional generation results on ImageNet 256x256.

### Strengths
- The paper handles an interesting problem using a novel mechanism
- The paper provides theoretical and empirical analyses to support the proposed technique
- The proposed method has high practical value and can be applied in a wide range of problems. In the experiments, the paper demonstrates it by aligning simple latents formed by ViT-L-based encoder-decoder systems to 4 different prior distributions, including low-level visual features from a VAE, continuous semantic visual features from DinoV2, discrete visual codebook embeddings from LlamaGen VQ, and textual embeddings from Qwen.
- The aligned latents can be integrated into MAR-B models, yielding competitive conditional generation results on ImageNet 256x256.
- To solve the dimension mismatching issue, it shows that among linear projection options, Random Projection is surprisingly a strong and more stable option

### Weaknesses
- The method requires two training steps to align the space; thus, it aggregates errors from both steps. First, it depends heavily on the quality of the flow-matching model, which cannot capture the prior distribution perfectly. Second, the latent optimization process is also not guaranteed to converge. The authors should analyze error accumulation and the system's failure modes.
- The method requires two training steps, which are expensive. Computation cost should be reported.
- While the proposed method can align the latent spaces to the target space in terms of distribution, it is interesting to incorporate it with semantic alignment. For example, when aligning the ViT-L-encoder-based latent space to the textual embeddings from Qwen, it is better to ensure that the encoded textual embedding aligns with the input image content.

### Questions
- The method requires two training steps to align the space; thus, it aggregates errors from both steps. First, it depends heavily on the quality of the flow-matching model, which cannot capture the prior distribution perfectly. Second, the latent optimization process is also not guaranteed to converge. The authors should analyze error accumulation and the system's failure modes.
- The method requires two training steps, which are expensive. Computation cost should be reported.
- While the proposed method can align the latent spaces to the target space in terms of distribution, it is interesting to incorporate it with semantic alignment. For example, when aligning the ViT-L-encoder-based latent space to the textual embeddings from Qwen, it is better to ensure that the encoded textual embedding aligns with the input image content.

### Soundness
3

### Presentation
3

### Contribution
4
