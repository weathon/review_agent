# The Diffusion Duality, Chapter II: $\Psi$-Samplers and Efficient Curriculum

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
Uniform-state discrete diffusion models excel at few-step generation and guidance due to their ability to self-correct, making them preferred over autoregressive or Masked diffusion models in these settings. However, their sampling quality plateaus with ancestral samplers as the number of steps increases. We introduce a family of Predictor-Corrector (PC) samplers for discrete diffusion that generalize prior methods and apply to arbitrary noise processes. When paired with uniform-state diffusion, our samplers outperform ancestral sampling on both language and image modeling, achieving lower generative perplexity at matched unigram entropy on OpenWebText and better FID/IS scores on CIFAR10. Crucially, unlike conventional samplers, our PC methods continue to improve with more sampling steps. **Taken together, these findings call into question the assumption that Masked diffusion is the inevitable future of diffusion-based language modeling.** Beyond sampling, we develop a memory-efficient curriculum for the Gaussian relaxation training phase, reducing training time by 25% and memory by 33% compared to Duo while maintaining comparable perplexity on OpenWebText and LM1B and strong downstream performance. We release code, checkpoints, and a video-tutorial on [https://s-sahoo.github.io/duo-ch2/](https://s-sahoo.github.io/duo-ch2/)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a family of Predictor-Corrector (PC) samplers for discrete diffusion models, referred to as $\Psi$-samplers, which extend and generalize existing PC methods to work efficiently with arbitrary noise processes. The authors also propose a scalable and memory-efficient curriculum learning strategy for training these models.

### Strengths
1. The paper presents $\Psi$-samplers by generalizing the PC approach to discrete diffusion models, providing a unified framework that extends prior work.

2. The provided experiment shows that this memory-efficient curriculum method is likely effective, while empirical improvements are achieved in both language modeling and image generation tasks.

### Weaknesses
1. The $\Psi$-sampler introduces non-Markovian posteriors, but the theoretical discussion heavily relies on Gaussian diffusion analogies, and the impact on sample quality, variance, and convergence remains underexplored. 
  

2. The method is highly sensitive to hyperparameters like $\kappa_t$ and noise schedules, and the evaluation primarily focuses on OpenWebText and CIFAR-10. 

3. The paper claims that generation quality improves with more sampling steps, but this holds true for language modeling tasks, not image generation.

### Questions
1. Why is the Predictor-Corrector approach effective in improving sampling efficiency and generative quality in discrete diffusion models? Does the paper provide a detailed theoretical foundation explaining why this approach works, particularly regarding the non-Markovian dynamics introduced by the $\Psi$-samplers?

2. How might the proposed Predictor-Corrector $\Psi$-sampler relate to techniques like [1], [2], and [3], which have recently been proposed to improve generative quality by directly optimizing conditional variance/entropy during sampling?

3. Does the $\Psi$-sampler consider resolving, or at least attempting to address, the significant numerical instability issue arising from the schedules in diffusion models as $\sigma_t$ approaches 0?

[1]. Li, S., et al., EVODiff: Entropy-aware Variance Optimized Diffusion Inference, NeurIPS 2025.

[2]. Ifriqi, T. B.,  et al., Entropy Rectifying Guidance for Diffusion and Flow Models, NeurIPS 2025.

[3]. Stancevic, D., et al.,  Entropic Time Schedulers for Generative Diffusion Models, NeurIPS 2025.

### Soundness
3

### Presentation
3

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
The paper propose predictor-correct samplers for discrete diffusion models. The authors:

1. First identity posteriors that sample from the marginals of a masked diffusion model. 
2. Next, the authors propose posteriors, that are a mixture of distributions, a predictor for un-masking, and another distribution to re-noise. 

Additionally, the authors also propose an efficient estimator for the curriculum learning objective of uniform state diffusion models. The authors note that:

1. There is a deterministic push-forward map between Gaussian and uniform state models. The denoiser can therefore be trained with continuous Gaussian noise. 
2. However, for each token, the curriculum learning objective in Sahoo et al 2025a requires sampling several high-dimensional Gaussian vectors, which can be expensive for large vocabulary sizes. 
3. The authors then propose a cheaper approximation that leads to reduced memory usage and faster training. 
4. Computing the noise for each position requires $|\text{Vocab}|$ many Gaussians, however the authors rely on order-statistics to sample from the $k$ largest values of a Gaussian vector. 

The authors then show that the superposition samplers combined with the diffusion duality model (Sahoo et al 2025a), a uniform state diffusion model, can match the performance of masked diffusion language models. The authors also show that using in-expensive approximations to compute the continual learning objective does not lead to a significant deterioration in performance.

### Strengths
The paper derives rigorous predictor-corrector schemes for both masked and uniform state diffusion models, as well as tractable approximations for the uniform state diffusion models.

### Weaknesses
See questions.

### Questions
1. To compute the test perplexity in table 2, how many $t$ samples do the authors use for each sample $x$. 
2. Can the authors also demonstrate the predictor-corrector methods they propose with existing pre-trained diffusion language models such as LLaDA and Dream.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors tackle the problem of fast sampling in discrete diffusion models and introduce a family of predictor corrector samplers that generalize over prior methods and apply to arbitrary noising processes. The authors then combine this framework with the Uniform state discrete diffusion models and show that the method outperforms baselines on text and image generation tasks. The authors also propose a curriculum learning strategy which improves upon the training efficiency of existing discrete diffusion models.

### Strengths
1. I like Figure 1 which conveys the main results of the paper convincingly.
2. The paper is well written in the sense that the background section is well formulated, the main contributions of the paper are well supported by empirical results.
3. The idea of formulating non-markovian forward processes for discrete diffusion models is quite interesting given its numerous applications in the context of continuous diffusion models in the form of DDIM.

### Weaknesses
I dont have a lot of concerns around the proposed method but rather a few suggestions for improving the presentation of the paper.

**Presentation Issues**

1. Is there a reason for using the psi notation to denote distributions throughout the paper? We can probably get rid of notations and denote distributions using their standard notations like p(.) or q(.) like other works in the literature.

2. In general, a lot of intuition is missing around the sampler design in Section 3. It is not clear, why can we expect the non-markovian design of the forward process to yield more efficient samplers? Can the authors provide some intuition around the sampler design in this context? The formulation of the non-markovian diffusion forward processes in Section 3 reminds me of the DDIM formulation for continuous time diffusion models so maybe adding some intuition from the continuous time diffusion processes literature could be useful?

3. Similarly in Eq. 10 what is the intuition behind different terms in the psi-posterior? All of these intuitions can be clarified in the main text to improve readability and presentation. In the current form the section appears to be very dense in equations which hinders readability.

4. Can the authors provide some intuition of the Curriculum learning strategy proposed in Section 4?. The section seems very heavy on notation and it is hard to understand the high level idea on a first read.

5. The authors can also consider revising the title of the paper as it currently seems very opaque.

**Empirical Issues**

1. In Fig. 1 unlike the MDLM and Duo samplers, the performance of the MDLM + ReMDM baseline and the proposed method keeps improving. Do the authors have updated results on how these curves saturate? I guess the cost of sampling beyond 4k steps is probably too high from a practical standpoint but it could still be worth exploring it from a research standpoint.

2. Can the authors report the FID score comparisons in Fig. 3. Im curious how discrete diffusion models compare to continuous time models on image quality.

### Questions
See weaknesses

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
This work proposes two improvements for uniform discrete diffusion models. The main contribution is a new family of predictor–corrector samplers (called psi-samplers) based on non-Markovian “psi-posteriors.” 

In more detail, they first define an x-conditional version that is a convex combination of the forward process and x-conditioned reverse posterior. The true reverse posterior then turns out to be a convex combination of q_{s|t} and q_{0|t} for 0 < s < t < 1. The reverse posteriors preserve the same marginals as standard diffusion while allowing adjustable stochasticity during sampling. This leads to empirical gains over ancestral sampling for uniform-state diffusion models, particularly in text and image generation tasks. Unlike standard samplers that plateau with number of steps, Psi-samplers are shown to continue improving with more steps (perhaps could be interpretted as a type of Gibbs sampling in the regime where steps > dims). 

A second contribution includes a practical efficiency optimization for a "curriculum learning" step from prior work, used during training. By approximating the vocabulary-wide averaging operation with a sparse top-k estimate, the authors reduce training time and memory usage while maintaining comparable perplexity and downstream accuracy. Altogether the combination of the train-time efficiency gain plus the new samplers give efficiency, perplexity, and sample quality gains on a number of tasks relative to a number of baselines.

The sampling idea is novel and presented cleanly, while the curriculum result is an engineering improvement, still meaningful, but not quite clearly defined or motivated besides the literal algorithm of the proposed computation. 

Overall, a solid technical contribution with good empirical validation. I view the sampling advance as the core insight and the training speedup as a perhaps useful but secondary contribution, at least given the amount of explanation/exploration of the underlying method being optimized.

### Strengths
- clear motivation for why sampling needs improvement in discrete diffusion: standard samplers can "over-commit" and cannot "self-correct" without further hacks, sometimes including adversarial training or other approximations.

- the psi-sampler formulation is general and recovers previous a few predictor–corrector methods as special cases (though please careful to not say "all cases in the literature", you never know with so many papers coming out daily, I suggest "that the authors are aware of")

- broad experimental coverage: language modeling, QA, and image generation for tasks + ablations on the \kappa_t hyperparameter and time-scheduling of where to use the psi-posterior.

- consistent empirical setups with appropriate baselines (Duo, MDLM, ReMDM) drawing on the Sahoo, Lou, etc... works.

- demonstrates measurable improvement in both FID/inception (images) and (generative GPT2) perplexity (on text) at fewer sampling steps, with nice gains in performance as NFE increases.

### Weaknesses
- The curriculum section assumes prior familiarity with Sahoo 2025a and gives little intuition for why that weighted-average operation helps training. Since you are devoting nearly a whole section to this method, I ask that you at least give a few sentences on how exactly the "curriculum" technique during training uses this average computation that you are approximating. Otherwise, the speedup could be relegated to an appendix section (I think "curriculum" is used 23 times in main text without being defined).

- experiments, while complete, lack interpretation, or at least attempts at providing intuition: results are shown, but, e.g., there is no discussion of why low-noise $\kappa_t$ regimes work best or what happens mechanistically when $\kappa_t$ is too small or too large

- the connection between psi-samplers and the NELBO s not explored, which might provide extra insight on what's going on.

- somewhat minor: when describing t_off, t_on, etc... it's possible that t is overloaded. Are you describing a function, t_for_model_input(t_real)? If so could you distinguish in symbols? It was a bit hard to read phrases like "t is constant in [t_off, t_on]".

### Questions
- there is some emphasis on the non-markovianness of the samplers. It would be curious to see where exactly this non-markovianness really matters, for example by seeing which conditioning values or time steps are correlated.

- could $\kappa_t$ be learned or adapted dynamically during sampling rather than chosen?

- the paper says psi-samplers improve with increasing NFEs while ancestral ones plateau. Why exactly does extra noise continue to help after many steps? If you run it for a while (steps >> dims/length), does it ever settle? Or if not, is there some relationship to Gibbs sampling?

- It seems like the psi posterios could be used for purely continuous diffusion models too. Has something like this been done in that setting? What would be things to watch out for?  More generally, is there any such trick in other areas of sampling that motivates this approach?

- when $\kappa_t < 1$, the sampler is said to be “noisier.” How is this related to the variance schedule $\alpha_t$? Is there any sense in which $\kappa_t$ values index an implied different set of $\alpha_t$'s? 

- why does the largest gains from psi-posteriosr occur at low noise levels and $\kappa_t$ close to 1? Is this due to the denoiser’s inductive (architectural) bias? how the mixture interacts with the diffusion schedule? some special property of the sequence of psi posteriors over time? something else?

- could the authors visualize or quantify how psi-sampling trajectories differ qualitatively from ancestral ones?

### Soundness
3

### Presentation
3

### Contribution
3
