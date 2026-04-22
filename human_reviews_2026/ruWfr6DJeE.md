# DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Diffusion models have emerged as powerful generative priors for high-dimensional inverse problems, yet learning them when only corrupted or noisy observations are available remains challenging. In this work, we propose a novel method for training diffusion models with Expectation-Maximization (EM) from noisy data. Our proposed method, DiffEM, utilizes conditional diffusion models to reconstruct clean data from observations in the E-step, and then uses the reconstructed data to refine the conditional diffusion model in the M-step. Theoretically, we provide monotonic convergence guarantees for the DiffEM iteration, assuming appropriate statistical conditions. We demonstrate the effectiveness of our approach through experiments on various image reconstruction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a method to train diffusion models only using corrupted/noisy observation based on expectation maximisation (EM). The idea of using EM to train diffusion models using corrupted observation is not novel
 and has been done in prior work [1],[2] (as the authors also point out in the introduction). However, previous work only trains an unconditional diffusion models and uses heuristic methods to create posterior samples for 
 the E-step. In contrast, the authors make use of conditional diffusion models for the posterior sampling step. 

[1] Rozet et al. "Learning Diffusion Priors from Observations by Expectation Maximization"

[2] Bai et al. "An Expectation-Maximization Algorithm for Training Clean Diffusion Models from Corrupted Observations"

### Strengths
- I think it is a good idea to make use of conditional diffusion models instead of heuristic approximations for posterior sampling in the E-step. In particular, in the context of non-linear forward operators $A$, where many of the heuristic methods often fail.

- The paper is well-written and easy to follow

### Weaknesses
- I have a few small issues with the formatting: page 2 (line 88) the subsection title "1.1 Preliminaries" is way to close to the figure subtitle, the text in Figure 1 is hard to read (small black text on a blue background) 

- In your experiments you choose two degradation processes: random masking and gaussian blur. In both of these setting the heuristic conditional sampling method often perform quite good. It would be interesting to see how your approach works under non-linear 
degradation operators?

- For both datasets (Cifar-10 and CelebA) you also could compare the "prior reconstruction" performance against a classical diffusion model, trained on the clean dataset. Such a comparison would be interesting to show the upper limit of performance.

### Questions
- In Algorithm 1 you state that you also output an unconditional model trained on the last sampled dataset $D_X^{(K-1)}$. It has been observed that training generative models on the output of another generative 
  model can degrade the quality a lot, see e.g. [3], [4]. Do you also observe something similar in your experiments?
- How many samples do you need to train the unconditional model? 
- Would it be possible to use classifier-free guidance to both train the conditional and unconditional model at the same time? 
- EM does depend a lot on the initialisation, how do you choose the initial model?  
- The noise for the Cifar-10 experiments seems to be really small ($sigma_Y^2 = 1e-6$). What happens for larger, more realistic noise? 
- How does the **unconditional samples** of CelebA look like? Do you observe less diversity compared to traditional unconditional diffusion models trained on the clean CelebA data?

[3] Alemohammad et al. "Self-Consuming Generative Models Go MAD"

[4] Shumailov et al. "The Curse of Recursion: Training on Generated Data Makes Models Forget"

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
This paper proposes to learning from corrupted data using the EM algorithm.

### Strengths
The proposed method can learn data distribution from corrupted data

### Weaknesses
1. Essential baselines/references not compared or discussed thoroughly. Including [1], [2]
2. The EM methods are not new, and used to train diffusion models with measurements. The novelty in this paper is just that replacing the posterior sampling with a conditional model if I am understanding it correctly.


[1] An Expectation-Maximization Algorithm for Training Clean Diffusion Models from Corrupted Observations

[2] Learning Diffusion Priors from Observations by Expectation Maximization

### Questions
The EM methods are not new, and used to train diffusion models with measurements. Training a conditional diffusion model usually means that it may need retraining for different measurements, and generalize not that well to OOD data. Could author discuss about this scenarios and analyze whether there is performance drop on OOD measurements or data?

### Soundness
2

### Presentation
2

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
This paper proposes an expectation-maximization scheme for training diffusion models from so-called corrupted data. The idea is to utilize conditional diffusion models to implement the E-step as opposed to prior work (which use diffusion posterior sampling for E step). The paper provides some theoretical claims and empirical results to support these claims.

### Strengths
The paper provides a reasonable approach to avoid the issues related to E-step in prior art. The idea here is to use conditional diffusion models -- albeit this requires an extra training burden. I appreciate that the authors discussed this explicitly. There are some theoretical results which can be further utilized in future work for a theoretical understanding of similar schemes.

### Weaknesses
The paper contains many confusing parts as will be discussed in the questions. The terminology, presentation, and notation is not often very clear.

The novelty of the work is very limited: It merely replaces the E-step in prior work with something more expensive, by training dedicated samplers; I think the idea is a straightforward extension of prior work.

### Questions
1) Figure 1 contains a typo in the caption (reads: todo). I found this figure quite confusing at this stage of the paper. It contains lots of notation that the reader is unfamiliar with at this stage, and the figure is really not well done and clear. I would suggest getting rid of it, or replacing it with something much simpler and cleaner.

2) The mathematical setting is a bit confusing. The paper starts by talking about a prior $P_X^\star$ which is defined over latent variables. This is the standard data distribution in generative modelling setting - this is not clarified. 'Forward channel' which is easy to mistake with the 'forward process' is instead the *standard likelihood* in ML/AI papers, again suffering from obscure terminology. This is also called corrupted channel, but this is not an information theory paper. I strongly suggest you to align with standard terminology and avoid *forward channel* altogether, it is confusing.

3) After this, diffusion models are introduced, talking about some $p_0$ to be recovered. No connections are made between $p_0$ and $P_X^\star$. I think it would make sense to clarify this for the reader's benefit.

4) Section 2 details the latent variable model setting. Now what is $q_\theta(x)$? It is never introduced. What is the relationship to $p_0$ and $P_X^\star$? Is it yet another notation for the same thing (or its approximations)?

5) I think it is not clear in Section 2.2 how eq. (10) pops out in the M-step (you just say *we consider the following conditional score matching loss*). Can you give a full derivation of how E and M steps are derived in your case in Section 2.2? Does it come out of standard expectation of the joint log-likelihood? Or is it an ad-hoc loss being introduced here?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DiffEM, a method for training diffusion models from corrupted observations only by learning a conditional diffusion posterior within an EM framework. Instead of treating a pre-trained unconditional prior as a fixed prior and approximating the posterior at inference, DiffEM directly trains a conditional diffusion model $q_\theta(x|y)$ with conditional score matching (M-step), using reconstructions produced by the current model (E-step). The authors give approximate monotone-improvement and convergence guarantees and demonstrate superior results over Ambient-Diffusion and EM-MMPS on masked/blurred CIFAR-10 and CelebA, alongside a computation-time analysis.

### Strengths
Modeling the *posterior* with a conditional diffusion directly is conceptually clean and broadly applicable to any known forward channel $Q(y|x)$, avoiding ad-hoc posterior approximations. Training reduces to standard conditional score matching; architecture/hyperparameters are transparent and close to common U-Net baselines. The paper states an identifiability assumption and proves a linear-rate EM convergence result up to learning/discretization error terms, giving useful intuition on where progress comes from. The paper decomposes cost (Tinit, K·Tft, Tu) and shows DiffEM is more time-efficient than EM-MMPS under the reported setup; warm starting from an EM-MMPS prior further improves numbers.

### Weaknesses
1. Many real pipelines have *model mismatch* (unknown blur kernel/noise, non-linear camera pipeline). Robustness to misspecification isn’t evaluated.
2. The M-step trains on reconstructions produced by the current model; the paper could more directly quantify diversity/coverage over EM iterations (beyond IS/FID/FD).
3. Metrics focus on distributional quality (IS/FID/FD/FDDINOv2). For cases with available ground truth (e.g., synthetic masks/blur), additional perceptual/semantic alignment analyses would strengthen the case (even if PSNR/LPIPS is cautioned against for heavy corruption).
4. The identifiability and small-error conditions underlying the convergence claim are hard to check in high-dimensional image settings; practical diagnostics would help.

### Questions
1. If the training/inference $Q$ differs slightly from reality (kernel width, noise variance, mask distribution), how do metrics and monotonicity behave?
2. Since theory separates learning vs. discretization error, can you report performance vs. sampling steps/schedules for the conditional model at fixed training?
3. Under equal GPU-hour budgets, how do DiffEM and EM-MMPS trade off final quality? The current tables are helpful but not strictly compute-matched.
4. Can you show JPEG or other non-linear/non-differentiable channels to underline generality?

### Soundness
2

### Presentation
3

### Contribution
2
