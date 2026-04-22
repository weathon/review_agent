# Test Time Scaling of Diffusion Model via Flow Matching Corrector

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Scalable self-improving capability is a desirable property of generative models, as it enables performance improvement with additional computational resources rather than requiring more training data. Existing approaches typically rely on external reward signals to fine-tune generative models and improve the generation. In this paper, we propose the Scalable Self-Improving Correction (SSI-Corr) framework, which requires neither new training data nor external rewards. Instead, SSI-Corr trains a corrector that directly aligns the sample generation process with the target distribution. Our method is supported by theoretical analysis and scales effectively with available computational resources. In the experiment, we demonstrate that SSI-Corr improves FID scores by 27\% and 14.3 \% on a pre-trained unconditional CIFAR10 DDPM with ancestral and DDIM samplers respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SSI-Corr, a “self-improving” framework for diffusion models that inserts a flow-matching corrector at a chosen timestep  to transport samples from the sampler’s biased marginal  toward the target marginal . Practically, the corrector is trained with synthetic trajectories from the base sampler and noisy data pairs constructed from the original dataset; at inference, an ODE for the learned velocity field is solved at step . The authors claim test-time and training-time scalability and report CIFAR-10 FID improvements for DDPM (e.g., from 6.872 to 4.963 by roughly doubling total steps) and a 14.3% gain over DDIM. They provide an error decomposition and a KL bound arguing that reducing  at step  should improve the final sample distribution.

### Strengths
Clear formulation that decouples correction from the base sampler and can be slotted at arbitrary timesteps; the method is conceptually simple (one extra ODE solve). 

Theory-backed motivation: a clean error analysis (initial bias, score error, discretization) and a KL bound illustrating when the corrector helps; also a note that the  corrector is near-linear in expectation.

### Weaknesses
1. Limited scope of evaluation: only unconditional CIFAR-10 is shown. No higher-resolution datasets (e.g., ImageNet-64/256, CelebA-HQ) or text-to-image settings, making scalability/generalization claims hard to assess. 

2. Self-improvement framing is overstated: the method still uses the original dataset to construct targets ; it is not data-free (just no new data). The paper should temper the claim and discuss limitations when access to data is restricted. 

3. Baselines feel weak/incomplete: comparison is mainly to predictor–corrector (Langevin) and an initial-bias tweak; no head-to-head against modern fast samplers / correctors (e.g., DPM-Solver variants, UniPC, consistency models, or recent flow-matching refinements) under matched compute. The fairness of comparing “one-shot” correction at a single  to PC applied after every step is also not fully justified. 

4. Compute accounting is unclear: the principal gains come when total sampling steps roughly double; wall-clock impact of the extra ODE solve, solver tolerances, and memory footprint are not reported. Training cost vs. improvement (especially at large ) also lacks a thorough cost–benefit analysis.

### Questions
see weakness

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
4

### Summary
This paper introduces the Scalable Self-Improving Correction (SSI-Corr) method to address three intrinsic sources of error in diffusion models: initial distribution mismatch, score estimation error, and discretization error.
The proposed approach trains an additional model to bridge the gap between the true distribution (ground truth) and the model distribution (obtained from the base diffusion sampler) at a specific diffusion timestep $t$. This corrector model is trained using the well-known flow matching loss.
The paper theoretically establishes an explicit bound on the sampling error of this method and empirically demonstrates its effectiveness through experiments.

### Strengths
- The paper proposes a principled approach based on flow matching to mitigate key sources of error inherent to diffusion models.

- It is not restricted to a specific sampler and can be applied to any diffusion sampling process, regardless of the underlying architecture or parameterization.

- It provides strong theoretical foundations, clearly deriving the specific error bounds achieved by the proposed algorithm.

### Weaknesses
- The method introduces an additional ODE solving step, inevitably increasing the total sampling time.

Moreover, since the ODE solver uses the same architecture as the base score model, it is computationally heavy.

Even for a relatively simple setup such as CIFAR-10 with a lightweight score model, the proposed framework requires an additional model of equal size.

Thus, it is unclear whether the method can be scalably applied to large and powerful diffusion models [1,2]. [2] has cifar-10 experiment.

- Experiments are conducted only on small-scale datasets, and comparisons are made mainly with older baselines.

It remains uncertain whether the proposed method would still be effective when applied to larger datasets or more recent diffusion architectures beyond DDPM and CIFAR-10.

-----------
[1] Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

[2] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is Secretly a GAN Discriminator

### Questions
- In Figure 3(a), the FID score appears to increase as training progresses, suggesting a lack of convergence.

This raises concerns about whether the proposed method is actually improving the sample quality as intended.

The observed instability might be due to the non-optimal training of the corrector at specific timesteps, but a more detailed explanation would be helpful.

- The proposed method appears general enough to be applicable to distillation models as well. 

Do the authors have any plans for additional experiments or evaluations in such settings?

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
The authors propose to improve the generative modelling of diffusion models post-training by learning an auxiliary flow matching model that maps the generated distribution to the true data distribution. They train this auxiliary model in practice using the original training data and synthetic data generated by the diffusion model *at a specific noise level*.  At inference, the flow matching model is used at the aforementioned noise level to "correct" the generated samples, diffusion generation then continues as normal.

### Strengths
I believe the core idea is reasonable and may have potential in practical generative modelling applications with diffusion models. This sort of approach may be interesting to explore in test-time distribution shifting (e.g. adjusting the generative style of a frozen/black box model post-training).

### Weaknesses
The paper load for ICLR this year has been large, and so I have not been able to spend as much time as I would like on reviewing. I encourage the authors to correct any errors/misunderstandings I may have with regards to the paper. 

1. **Understanding the approach**
    1. The approach applies a very general interpolation between the generated and data distribution via random pairing at train time, which to me would potentially suggest large transformations during the correction step. However, the actual effect (Figure 1) seems to be visually imperceptible. The authors do not provide any insight on why this is the case. Why does the learnt velocity make only very small changes?
2. **Poor presentation**  
    1. Figures are out of order, e.g. Figure 4 appears in the text before Figure 3, leading to confusion when reading.
    1. Figures are poorly formatted with text inside figures being tiny and close to illegible.
    1. Writing is unpolished e.g. in the introduction, "current works ... either rely on additional reward models for guidance." 
3. **Weak experiments**
    1. The authors only benchmark on CIFAR-10. That is to say a single, small-scale (32x32) dataset. I am not saying that CIFAR is necessarily a bad benchmark for generative modelling, however, using *only* CIFAR-10 is not enough. This is especially problematic since the authors motivate their work from the perspective of test-time scaling, which is a concept that is primarily relevant in large-scale foundation model deployment, where relevant experiments should be at the scale of high-resolution text-to-image generation. 
    1. The lack of any FID-50K results makes it hard to compare the results to the literature.

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Scalable Self-Improving Correction (SSI-Corr), a drop-in framework for diffusion models that replaces generation–verification schemes with a flow-matching corrector trained to align the sampler’s marginal with the ground-truth marginal at a chosen timestep t. The authors provide a clean error decomposition and a bound showing that the final divergence is controlled by the corrector’s mismatch at t plus residual score/discretization terms accumulated after t. Empirically, SSI-Corr improves FID on a pre-trained CIFAR-10 DDPM by ~27% with the ancestral sampler and ~14.3% with DDIM, and exhibits both test-time and training-time scalability via the choice of correction timestep and solver steps.

### Strengths
- The proposed method is simple and easy to plug in to the current diffusion models.
- The introduction of flow matching to the self-improving corrector is new
- Several experiments are conducted to demonstrate the main claims
- Implementation and evaluation details are reported clearly enough to replicate the results.

### Weaknesses
- Incremental core novelty. The main technical part—using flow matching to train a corrector at a chosen timestep—leans on established techniques; much of the machinery (least-squares flows, OU marginals, sampler integration) is also adapted from prior work.
- The writing could be improved; for example, the discussion of training and testing scalability appears before these terms are clearly defined.
- The proposed corrector can overfit training data (the paper notes this qualitatively). Please quantify with train/val curves vs dataset size, and report generalization across seeds/samplers.
- The experiments only focus on CIFAR-10 dataset and DDIM samplers. It would be interesting to see how the corrector perform on other datasets/samplers.

### Questions
- In the data sampling process from the distribution p_t, why is Gaussian noise added, and how are its mean and variance determined?
- In Algorithm 1, the sampler time horizon T is never used. Should t/h-1 be T/h-1 ?

### Soundness
2

### Presentation
2

### Contribution
2
