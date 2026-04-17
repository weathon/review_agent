# Optimal Stopping in Latent Diffusion Models

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
We identify and analyze a surprising phenomenon of $\textit{Latent}$ Diffusion Models (LDMs) where the final steps of the diffusion can $\textit{degrade}$ sample quality. In contrast to conventional arguments that justify early stopping for numerical stability, this phenomenon is intrinsic to the dimensionality reduction in LDMs. We provide a principled explanation by analyzing the interaction between latent dimension and stopping time. Under a Gaussian framework with linear autoencoders, we characterize the conditions under which early stopping is needed to minimize the distance between generated and target distributions. More precisely, we show that lower-dimensional representations benefit from earlier termination, whereas higher-dimensional latent spaces require later stopping time. We further establish that the latent dimension interplays with other hyperparameters of the problem such as constraints in the parameters of score matching. Experiments on synthetic and real datasets illustrate these properties, underlining that early stopping can improve generative quality. Together, our results offer a theoretical foundation for understanding how the latent dimension influences the sample quality, and highlight stopping time as a key hyperparameter in LDMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a theoretical analysis of an interesting observation: Early stopping in sampling may improve sample quality in latent diffusion models in a way that does not hold for ambient-space diffusion models. The authors briefly motivate their theory by demonstrating this phenomenon on a pair of latent and pixel-space models, and then introduce a toy model in which one seeks to sample from a Gaussian target distribution with diagonal covariance in a linear "latent space", and the only learnable parameters for the score function are the diagonals of the target covariance matrix. In the context of this model, the authors show that the Wasserstein distance between the model distribution and tha target distribution is not necessarily monotonic in the sampling stopping time, derive the optimal latent dimension for a given stopping time, and characterize optimal stopping times and latent dimensions for data supported on a subspace. They finally show how to generalize their results to general Gaussian targets with non-diagonal covariance matrices.

### Strengths
- This is generally a well-written paper, and I was able to follow the main ideas without too much trouble. The authors provide sufficient intuition for their theoretical results. There are a few instances of awkward phrasing, but these can easily be corrected in revisions. (For instance, the statement of the hypothesis "In latent diffusion models, the last diffusion steps do not improve, or even degrade, sample quality" is a bit confusing -- I initially parsed it as "the last diffusion steps neither improve nor degrade sample quality".) 
- The results in this paper are correct to my knowledge.
- The main finding in this paper -- that early stopping in sampling may result in a better estimate of the target distribution when the latter is supported on a subspace -- is interesting and merits further investigation. Relaxing the Gaussian-data assumption and considering multimodal or manifold-supported distributions would be a particularly interesting future theoretical direction. However, as I explain below, I believe that the need for further empirical validation is a notable weakness of this paper.

### Weaknesses
My main critique of this paper is that it does not adequately demonstrate that its main hypothesis (line 50) is true in practice. The authors experiment with a single pair of latent and pixel-space models and find that early stopping in sampling improves sample quality for the latent model but not for the pixel-space model. However, it is unclear whether this is simply an artifact of the target distribution, the model architecture, or even the training procedure. (Indeed, the models are trained on different datasets, which could conceivably explain their different behavior wrt early stopping.) If this phenomenon has previously been observed in the empirical literature, the authors might include a citation. Otherwise, I'd like to see a bit more evidence that this phenomenon actually exists. Without this evidence, it is possible that the authors have developed a collection of interesting results that predict the behavior of their toy Gaussian model but that do not meaningfully bear on the behavior of real-world diffusion models.

### Questions
- Has the main hypothesis (line 50) been empirically validated in previous work?
- In lines 350-352, the authors write that Proposition 3 "suggests a principled guideline for practitioners: identifying and restricting the diffusion process to the data’s intrinsic dimensionality leads to more accurate and robust generative models." Is this guideline actionable in practice? I am aware of methods for estimating a data's intrinsic dimensionality using a pretrained generative model (e.g. the FLIPD estimator from Kamkari et al. 2023), but training a pixel-space model to first identify the data's intrinsic dimensionality seems prohibitively costly.

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
3

### Summary
The paper investigate a phenomenon in Latent Diffusion Models (LDM): the final steps in LDM can degrade sample quality more intensively compared to general diffusion models. The authors hypothesize that this phenomenon is intrinsic to the **dimensionality reduction** in LDMs. To validate this hypothesis, the authors conduct analysis in a simplified setting: Gaussian framework with linear autoencoders, quantifying the Wasserstein distance between the generated and the data distributions. The theoretical results reflect that

(1) in lower dimensional latent space, early stopping can increase the generation quality;

(2) there is an optimal latent dimension and an optimal stopping time to achieve optimal generation quality when the data has low-dim structure.

### Strengths
1. **Rigorous math**: the analysis is grounded with clear and clean mathematical statements reflecting the relation between dimensionality, early stopping time and sample quality in LDM.

2. **Great motivation and impact**: the theory brings a new perspective for improving the sample quality in LDM, which could also be useful in other diffusion-based tasks, such as conditional generation.

### Weaknesses
1. **Simplified setting**: the current setting considers only Gaussian data and linear autoencoders, which are far from practical settings. It is not clear how useful these results in this simplified setting are in practice. While some qualitative statements may extend, the quantitative statements may not. Although the paper provides numerical results on CelebA, it does not validate any of the quantitative statements in the paper directly. 

2. **Split theory and experiments**: except for the degradation at the late stage, the theory and the practical numerical experiments are not very related. For example, is there a similar results to what's in Figure 4 for the LDM on CelebA?

### Questions
1. line 214-215: it should be replacing $\hat p_T$ by standard Gaussian;

2. line 257-260, some signs should be reflected. The current lower bound is over-conservative.

3. In Proposition 2, what does *well-ordered* mean?

4. In the training phase of the diffusion model, the score is learned by Harmonic features [1]. Is it possible to replace the dimension in the autoencoder by the features and show similar degradation phenomenon? 

[1] Kadkhodaie, Zahra, et al. "Generalization in diffusion models arises from geometry-adaptive harmonic representations." arXiv preprint arXiv:2310.02557 (2023).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper assesses the effect of early denoising stopping on the quality of generated images in Latent Diffusion Models (LDMs). The authors theoretically analyze this phenomenon by studying diffusion processes on Gaussian mixtures and independent components, and evaluate the generated distribution using the Fréchet (Wasserstein-2) distance as a function of the stopping time. They demonstrate that, depending on the underlying data distribution and latent dimensionality, the Fréchet distance may not monotonically decrease as denoising progresses. Consequently, excessive denoising can degrade sample quality. The authors then propose two generalizations of their framework to arbitrary Gaussian mixtures and provide conditions under which early stopping is theoretically optimal.

### Strengths
- The paper addresses an interesting and practically relevant phenomenon—early stopping in LDMs—that has been empirically observed but rarely analyzed theoretically.  
- The Gaussian analytical framework is elegant and allows for closed-form insights into the interplay between latent dimensionality, stopping time, and sample quality.  
- The proofs are rigorous, and the link between early stopping and dimensionality reduction is well-motivated.  
- The work provides a first theoretical grounding for empirical practice in generative modeling.

### Weaknesses
- The paper lacks a clear empirical conclusion linking the theoretical findings to the observed advantages of early stopping in real LDMs versus pixel-space diffusion models.  
- There is a noticeable discrepancy between the theoretical setup (independent Gaussian components) and real diffusion models trained on complex datasets. The analysis is an important first step but remains far from validating the phenomenon empirically.  
- The choice of Fréchet distance (FD) as the evaluation criterion is not fully justified, and it is unclear how sensitive the results are to this choice. Other metrics like KL divergence or MMD could yield different insights. 
- The statement “One well-documented challenge in this method is the onset of numerical instability as the timestep t approaches 0” cites only one work and would benefit from a broader empirical basis or clearer evidence.  
- Experimental validation is minimal. For a venue like ICLR, testing only one AE and one DM on CelebA-HQ is insufficient to substantiate claims about optimal stopping or to demonstrate generality across datasets or architectures.

### Questions
- The authors attribute the degradation at late timesteps to a low signal-to-noise ratio in the latent space.  Would this phenomenon persist in more modern architectures such as EDMs, where the model enforces unit-variance inputs and outputs at each noise level?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper identifies that in Latent Diffusion Models (LDMs), continuing the diffusion process to the very end can degrade sample quality, a phenomenon not seen in standard pixel-space diffusion models. The authors hypothesize this is an intrinsic effect of the LDM's dimensionality reduction. To analyze this, they propose a simplified theoretical framework using Gaussian data and linear autoencoders. Within this setup, they analyze the Wasserstein-2 (Fréchet) distance to the target distribution, showing that lower-dimensional latent spaces benefit from earlier stopping times. The analysis is extended to consider the impact of score matching constraints and generalized from diagonal to arbitrary Gaussian covariance matrices.

### Strengths
* The paper highlights a specific, observable, and relevant phenomenon in LDMs: the degradation of sample quality (rising FID) in the final sampling steps. This is contrasted with pixel-space models where quality consistently improves.
* The core hypothesis—linking this degradation to the interaction between latent dimensionality and stopping time—is plausible and provides a novel perspective beyond typical arguments about numerical instability.
* The work attempts to provide a theoretical foundation for the interplay between latent dimension, stopping time, and model constraints.
* The analysis within the simplified Gaussian setting is analytically tractable and produces concrete, testable results, such as the characterization of optimal projection dimensions for given time intervals.

### Weaknesses
* ** Theoretical model:** The entire analysis is built on a "Gaussian framework with linear autoencoders". This is a big oversimplification. Real LDMs, like the one used in the paper's own introductory experiment (Figure 1), use highly **non-linear** autoencoders (specifically VQ-VAEs). In summary, perfect autoencoders correspond to nonlinear functions (enc and dec) such that dec(enc(x))=x - p_{0} a.s
The linear projection model $P$ does not capture the essential non-linear manifold learning performed by the autoencoder, making the relevance of the paper's entire theoretical analysis to practical LDMs questionable.

* **Technical incorrectness of the core SDEs:** The formalization of the latent diffusion process appears to be incorrect.
    1.  The paper defines the latent forward SDE as 

$dP \vec{X}_{t}=-w_{t}^{2}P\vec{X_{t}}dt+\sqrt{2w_{t}^{2}}dP\vec{W_{t}}$

 for a general matrix $P \in \mathbb{R}^{d \times D}$. If $ P $ is not an orthogonal projection (i.e., if $ PP^\top \neq I_d$), then $P\vec{W}_t$ is **not** a standard d-dimensional Brownian motion (I_d covariance). This defines a different diffusion, and its time-reversal is more complex than stated.

 The backward SDE in Equation (2) is (**when P is not an orthogonal projection**) wrong. As shown in Anderson (1982, eq. 3.12), the correct reverse drift involves metric tensor terms ($g^{ij}$) induced by the projection, which are missing from Equation (2).

* **Misleading justification:** The paper claims its simplified setting "already exhibits phenomena similar to the larger-scale evidence", but this is just an assertion. The link between the non-monotonic FID in Figure 1 (a complex, non-linear LDM on CelebA) and the non-monotonic Fréchet distance derived from a misformulated linear-Gaussian model (Proposition 1) is never established. The "phenomenon" may be similar, but the causes are possibly distinct.
* **Esperimental validation**. As the considered distributions are particularly simple (dimensionality and structure) the authors could have included some validation of the results with a *real* score network and compare the results

### Questions
*  How can the theoretical results, derived from a simplified linear-Gaussian model, be claimed to explain a phenomenon observed in a highly non-linear VQ-VAE-based LDM? What evidence supports the claim that the *causes* are the same, rather than just the high-level behavior (non-monotonicity)?
*  The backward SDE in Equation (2) appears to be valid *only if* $PP^\top = I_d$. However, the setup in Equation (1) is for a general $P \in \mathbb{R}^{d \times D}$, which does not guarantee this. Can the authors clarify on this in the text? 

*  In Proposition 1, the condition for non-monotonicity depends entirely on the *estimation error* ($\sigma_{d'}$ vs $\hat{\sigma}_{d'}$). This suggests the phenomenon is an artifact of estimation. However, the paper's introduction claims the phenomenon is "intrinsic to the dimensionality reduction", *not* estimation error. How do you reconcile this? The proof for the *true* score (first part of Prop. 1) is shown to be monotonic.

* The notation d_F(...) =min_{} d_F is highly confusing. Can the authors clarify in the text?

* Minor: why just Frechet Distance? Being the distributions Gaussian, other divergences (e.g. KL) are available. Can the authors present some results on that as well?

### Soundness
3

### Presentation
3

### Contribution
3
