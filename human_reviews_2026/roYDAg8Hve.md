# How private is diffusion-based sampling?

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it remains impractical for large-scale diffusion models. In this work, we take a different route by analyzing privacy leakage during the sampling process. We introduce an empirical denoiser that enables tractable computation of per-step sensitivities, allowing each denoising step to be interpreted as a Gaussian mechanism. Building on this perspective, we apply Gaussian Differential Privacy (GDP) to derive tight privacy bounds. Furthermore, we identify critical windows in the denoising trajectory—time steps where salient semantic features emerge—and quantify how privacy loss depends on stopping relative to these windows. Our study provides the first systematic characterization of privacy guarantees in diffusion sampling, offering a principled foundation for designing privacy-preserving generative pipelines beyond DP training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies privacy leakage during the sampling process of diffusion models. It replaces the intractable neural denoiser with an empirical denoiser by dataset average so that each reverse step can be written as a Gaussian mechanism with sensitivity controlled by clipping. This allows per-step \mu-GDP accounting and composition across timesteps. The paper also argues that privacy loss concentrates in “critical windows” when semantics emerge; it further suggests a hybrid pipeline that switches to a public (non-private) denoiser outside those windows.

### Strengths
- The “critical window” framing offers a coherent qualitative perspective on where privacy loss concentrates along the sampling trajectory.
- A GDP-based accounting is presented, leveraging the exact $\mu$-composition property for Gaussian mechanisms.

### Weaknesses
1. **Lack of experimental baselines and validation.**
    - No comparison to other DP methods.
    - Image quality is not measured (the paper explicitly avoids FID/IS), so practical impact on generation quality is unclear.

2. **“Critical window” claims lack an operational detector.**
    - The idea is qualitative only; no quantitative rule (e.g., change-point in $\mu_t$, SNR threshold, or semantic-classifier stability) is provided or evaluated.

3. **Clarity and notation issues.**
    - Line ~163: What is $\Delta$?
    - Lines ~185–186: What is $C$? Is this the clip norm used to bound sensitivity?
    - Line ~201: What is $\mu_{t_i}$ and why is it defined that way?
    - Line 460: “Figure 3” is referenced but not linked/connected.

### Questions
- Can you provide baselines against DP-trained diffusion (e.g., DP-SGD) at matched privacy budgets, and report FID/IS (or CLIP-based metrics) to quantify utility. 
- Can you provide a quantitative detector for the “critical window” (e.g., a change-point on per-step or cumulative \mu, an SNR threshold, or a classifier-stability metric), with ablations?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates privacy leakage during the sampling process of diffusion models, proposing an alternative approach to differentially private (DP) training. The authors introduce an **empirical denoiser** that replaces the intractable neural denoiser, enabling computation of per-step sensitivities in the denoising process. By framing each denoising step as a Gaussian mechanism, they apply **Gaussian Differential Privacy (GDP)** theory to derive tight privacy bounds through composition. The analysis reveals that privacy loss is non-uniform across the sampling trajectory, with critical windows emerging where semantic features materialize. The paper explores both full-batch and mini-batch (subsampled) settings, demonstrating that subsampling provides substantial privacy amplification. Experiments on CIFAR-10 validate the framework and propose a hybrid strategy using public denoisers for non-critical timesteps to preserve privacy while maintaining generation quality.

### Strengths
- Analyzing privacy at the sampling stage rather than training is an interesting and underexplored angle, particularly relevant for proprietary models where only outputs are accessible.
- Properly applying GDP composition to multi-step stochastic processes is technically non-trivial, and the subsampling analysis (Section 4) adds value.
- The identification of non-uniform privacy loss across timesteps and the proposed hybrid strategy (Section 3.4) are potentially useful.
- Effective use of figures (especially Figures 4-5 showing ε-δ curves alongside generated samples)
- Well-structured progression from single-step to multi-step analysis

### Weaknesses
- The fundamental assumption—that the empirical denoiser $\hat{\mathbb{E}}[x|x_t; D]$ adequately approximates the neural denoiser $D(x_t, t; \theta(D))$—is insufficiently validated. While Figure 1 shows cosine similarity convergence at later timesteps, this does not guarantee that privacy bounds derived from the empirical denoiser translate to the neural case. The bias-variance argument (Section 3.3) is heuristic and relies on questionable assumptions (e.g., equal MSE between estimators). **This gap undermines the paper's central claim** that the analysis characterizes real-world privacy leakage.
- The authors acknowledge (end of Section 4) that GDP CLT requires no single mechanism to dominate, yet their late-stage denoising steps contribute disproportionately due to reduced noise scales. While they suggest early stopping as mitigation, this doesn't resolve the theoretical inconsistency—the CLT-based bounds may not be valid for the full trajectory.
- There is no empirical validation (e.g., through membership inference attacks on actual neural denoisers) to confirm that the derived privacy bounds reflect real privacy risks. The paper provides mathematical analysis but no evidence that ε ≈ 90 (full-batch, σ_min = 0.2) corresponds to actual vulnerability.

### Questions
- Can you provide formal bounds on $|\mathbb{E}[x|x_t] - \hat{\mathbb{E}}[x|x_t; D]|$ that would translate to error bounds on the privacy parameters?
- Have you considered Lipschitz-based sensitivity analysis for neural denoisers, even if looser, to validate the empirical denoiser bounds?
- Why not test membership inference attacks on neural-denoiser-generated samples and compare measured privacy leakage to your predicted ε values?
- Can you show examples where the empirical denoiser produces samples violating the predicted ε bound?

### Soundness
3

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
4

### Summary
The paper tackles the important problem of obtaining differential privacy in diffusion models by incorporating Gaussian noise in the sampling step. It identifies windows in the reverse process where semantic features emerge and appropriately utlize this for limiting privacy loss and also introduce an empirical denoiser to enable the computation of per-step sensitivities. Experiments are shown to match neural and empirical denoisers and their performance on sampling, privacy loss with respect to critical windows, and batch size for denoising.

### Strengths
* Analyzes the privacy loss with respect to the sampling steps and appropriately identifies critical regions where semantic information is generated and deals with it appropriately. The final steps are dealt with by using public datasets. 
* Utilized Gaussian diffusion process to identify the per-step sensitivities of the sampling and utilized central limit theorem to do noise accounting over multiple steps. 
* Experiments are shown to show that the neural and empirical denoisers match each other in the critical windows while diverging in the later steps (which are replaced by public data). The size of the batch size going from full dataset to subsampled is shown and it trades-off privacy with the quality of the generation as the empirical denoiser performance goes down.

### Weaknesses
(a) It does early stopping which limits the quality of the generated data. It is unclear how much data is needed for the public denoisers.        
(b) It only tackles the continuous version of the diffusion process and would be interesting to see how it compares to discrete diffusion where similar regimes are detected for privacy leakage (*).
(c) The connection from empirical denoiser to neural denoiser is not rigorous and shown with empirical experiment.

* On the inherent privacy properties of discrete denoising diffusion models. https://arxiv.org/abs/2310.15524

### Questions
(1) It seems that to obtain high quality denoised samples, we need to have public denoisers: (a) does it need to be from the same domain as the original private dataset? How much data i
s typically required to train the final steps of the denoising process? 
(2) Does it help to replace the mini-batch sampling with an importance-weighted sampler to enable a good estimate for the denoiser?

### Soundness
3

### Presentation
3

### Contribution
3
