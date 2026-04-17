# Controllable Diffusion via Optimal Classifier Guidance

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
The controllable generation of diffusion models aims to steer the model to generate samples that optimize some given objective functions. It is desirable for a variety of applications including image generation, molecule generation, and DNA/sequence generation. Reinforcement Learning (RL) based fine-tuning of the base model is a popular approach but it can overfit the reward function while requiring significant resources. We frame controllable generation as a problem of finding a distribution that optimizes a KL-regularized objective function. We present SLCD -- Supervised Learning based Controllable Diffusion, which iteratively trains a small classifier to guide the generation of the diffusion model. Via a reduction to no-regret online learning analysis, we show that the output from SLCD provably converges to the optimal solution of the KL-regularized objective. Further, we empirically demonstrate that SLCD can generate high quality samples with nearly the same inference time as the base model in both image generation and biological sequence generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Supervised Learning–based Controllable Diffusion, a KL-regularized reward maximization as optimal classifier guidance for a pre-trained diffusion model. SLCD trains a small time-dependent classifier by iteratively rolling in with the current guidance and rolling out with the prior to collect rewards, and uses a distributional estimator of the reward at each diffusion time to compute the optimal guidance score, which also enables test-time adjustment of the KL penalty. Theoretical analysis reduces learning to no-regret online learning. Empirically, the paper reports reward and trade-offs on image, 5’ UTR, and enhancer tasks.

### Strengths
1. The reduction to no-regret learning and the KL bound on the sampling distribution is an interesting theoretical perspective.

2. The roll-in/roll-out data aggregation to align train/test distributions for the classifier is motivated.

3. Finetuning-free with small inference overhead is a appealing goal to achieve.

### Weaknesses
1. Missing RL fine-tuning baselines are needed, such as DPOK and Direct reward backprop on diffusion, which are RL/fine-tuning approaches to reward-aligned diffusion. These represent the “train-the-model” side of the Pareto frontier the paper positions against, and without them the empirical claim of superiority over existing approaches is incomplete. 
In addition, discrete-guidance baseline [1] is missing, which is a guidance framework for discrete state-space DMs directly overlapping with your DNA/RNA settings.

2. DPS configuration is not fully specified and can be potentially unfavorable. The paper should examine stronger DPS variants, since SLCD is rooted in classifier guidance. 

3. The learning process requires substantial computation compared to the inference, which is not well-analyzed or quantified. This also makes the claim of "paying only a fixed, small inference cost" less convincing.

4. Assumptions in theory are strong and partially opaque.
The realizability (Assumption. 3), no-regret (Assumption. 4) and smoothness bound linking distributional error to score error (Assumption. 6) are non-trivial and central to Theorem 7. The paper would benefit from concrete instantiations (e.g., Lipschitz constants) and a finite-sample bound on N,M required for optimality.

5. Ablations on iteration count are light. Few ablation studies are provided to show the contributions of the key components. Fig. 3 shows reward improves with iterations, but doesn’t analyze variance, overfitting, or the trade-off vs. online dataset size. Also, the paper discretize rewards and learn a multi-class classifier, but there’s no ablation on binning strategy, calibration, or on alternative parametric families.

6. Fig. 4 notes images “move subjects to edges,” “simplify scenes” for compression. This looks like reward hacking; the paper should quantify semantic drift vs. reward gain (e.g., CLIP-sim, caption fidelity), not only FID, to support “stays close to base distribution” claims.

7. Code is not provided, and thus readers cannot verify exact reproducibility.

[1] Unlocking Guidance for Discrete State-Space Diffusion and Flow Models

### Questions
Could stronger DPS settings close the gap between classifier guidance baseline and the proposed method?

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
3

### Summary
This paper introduces Supervised Learning based Controllable Diffusion (SLCD), a new method for steering pre-trained diffusion models to generate samples that optimize a given reward function. The authors frame this as a KL-regularized optimization problem. The core idea is to iteratively train a lightweight classifier to guide the diffusion process. To address the covariate shift inherent in training such a classifier on offline data, SLCD employs an iterative data aggregation (DAgger-like) strategy, where the classifier is repeatedly retrained on new data generated by the guided model itself. The paper provides a theoretical analysis showing that this approach, via a reduction to no-regret online learning, converges to the optimal solution of the KL-regularized objective. Empirical results on image generation and biological sequence tasks show that SLCD can generate high-quality samples that achieve high rewards, maintains a good reward-FID trade-off, and adds negligible computational overhead at *inference* time.

### Strengths
The paper is well-written, and the core idea is presented clearly and intuitively. The formulation of controllable generation as a KL-regularized objective solved by an iterative classifier is easy to follow.

The approach of using a DAgger-style iterative training loop to mitigate covariate shift for a guidance classifier appears to be a novel contribution to this problem space. To my knowledge, this specific application is new.

The theoretical analysis that reduces the problem to no-regret online learning provides a solid foundation for the algorithm's convergence, which is a significant strength.

### Weaknesses
A primary concern is the fairness of the experimental comparison in Table 1. The paper compares SLCD against methods like Best-N, DPS, SMC, and SVDD. It is noted that these baselines are "training-free" or can use an off-the-shelf classifier (or value function) trained a single time on offline data. However, SLCD introduces a significant computational overhead through its *iterative* training and data generation loop (Algorithm 1). This iterative refinement is a form of training that the baselines do not have. The direct comparison of final sample quality (reward) may not be fair, as SLCD benefits from a much more intensive procedure to create its guidance model. The paper emphasizes its low *inference* cost but does not discuss the *training* cost of the guidance model itself, which is not required by most of the chosen baselines.

The experiments are missing a crucial baseline comparison. The paper motivates its approach by highlighting the covariate shift issue of standard classifier guidance (where a time-dependent classifier is trained offline). However, this standard classifier guidance method is never empirically compared against. Without this baseline, it is difficult to quantify how much of the performance gain is due to the novel iterative training scheme versus simply using any form of classifier guidance (even a "naive" offline one).

I currently rate this paper as reject due to this issue. If I missunderstand something, please clarify and I am happy to adjust the score if this issue is addressed.

### Questions
1. Algorithm 1 mentions a "validation" step (Line 224) to select the best classifier, $f^{\hat{n}}$, from the $N$ iterations. What is the detailed process for this validation? How is the validation set constructed, and what metric is used to determine the "best" model?
2. Related to the weakness section, could the authors provide clarification on the total computational cost (e.g., training time or GPU-hours for the *guidance model*) for SLCD? This would be necessary to fairly assess the trade-offs against methods like DPS, which only require a one-time offline training of their value function (usually considered as "training-free guidance" in literature).
3. Why was standard, time-dependent classifier guidance (trained offline on the prior distribution) omitted as a baseline in the experiments? Including this comparison would seem essential to empirically validate the paper's core claim that the iterative, on-policy data collection is necessary to overcome covariate shift.

### Soundness
2

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
This paper proposes Supervised Learning based Controllable Diffusion (SLCD), a classifier-guided diffusion method where the guidance signal is provided by a classifier trained to estimate step-wise reward values along the diffusion trajectory. A key challenge of classifier guidance is covariate shift between real-data distributions used for classifier training and intermediate latent distributions encountered during generation (inference). SLCD addresses this by training the classifier on samples drawn from intermediate diffusion steps, thereby aligning the classifier’s training distribution with its test-time usage. Experiments on image generation tasks and a DNA enhancement tasks demonstrates that SLCD yields improved controllability with negligible increase in inference cost.

### Strengths
- The method is supported by clear theoretical motivation and formal analysis explaining how SLCD reduces covariate shift compraed to prior classifier-guided approaches.
- The visualization and derivation in Figure 2 effectively illustrate the benefit of aligning classifier training distribution and inference distribution.
- The framework is conceptually simple, inference-time effective, compatible with existing diffusion models, and does not require modifying the base model architecture.

### Weaknesses
- Important considerations—including classifier architecture, hyperparameters, sampling strategy for intermediate steps, and reward definitions—are only provided in the supplementary material without directions in the main script. Some guidance should be included in the main paper.
- While classifier training is central to SLCD, the paper does not analyze how classifier quality influences controllability or generation performance (e.g., ablation over classifier capacity, generalization on prompts, or data size).
- Experimental validation on image tasks is limited to SD 1.5, which is no longer state-of-the-art. Evaluation on stronger models (e.g., SDXL or SD3) would strengthen the claim that SLCD generalizes broadly.
- The scaling parameter for classifier guidance plays a crucial role, but no discussion is provided regarding sensitivity or tuning strategy.

### Questions
- Does Equation (5) compute KL divergence over the conditional distributions or marginal distributions? If samples are collected under conditional generation in SLCD, is q_0 treated as a conditional distribution as well?
- Could the authors clarify the motivation for using the compression task as an image evaluation setting? What intuition links compression to controllable guidance?
    - Also, could the authors provide more details about the compression task?
- How are hyperparameters for classifier training choosen? What is the computational cost (both sample collection and classifier training)? Does the classifier generalize to unseen prompts?
- How is the scaling parameter chosen in practice? Is a single value applied across experiments? (such as different rows in Figure 4), or is it tuned per dataset/task?
- In Section 6.3, is FID computed between base SD outputs and SLCD outputs, or between generated outputs and real images?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper frames controllable generation as KL-regularized reward maximization and introduces SLCD, a fine-tuning-free method that learns an optimal classifier to guide a fixed diffusion model toward the target posterior $p(x)\propto q_0(x)\exp(\omega r(x))$.
SLCD combats the classifier covariate-shift problem by iteratively aggregating on-policy data from its own guided rollouts and trains the guidance via standard supervised learning on a discretized reward distribution.
The theory reduces performance to no-regret online learning: under certain assumptions, the guided sampler’s distribution converges in KL to the optimal solution, with an explicit bound given in Theorem 7.
Empirically, on image aesthetics/compression and DNA/RNA sequence design, SLCD attains higher rewards than training-free baselines at near-base inference cost.

### Strengths
1. Under some assumptions, the authors prove that the no-regret learning of $\log R^{prior}$ implies the closedness of the distribution generated by the learned guidance and the target tilted distribution.

### Weaknesses
1. The idea of this paper is quite similar to the stochastic optimal control analysis provided in the `adjoint matching` paper: 

    A. In `adjoint matching` (actually this is a well-known fact in control literature under the name `Logarithmic Transformations`), it is observed that the optimal value function can be expressed in terms of the uncontrolled prior process. By noting that the classifier $p(y=1 \mid x_t)$ is equivalent to $\exp(-V)$ in `adjoint matching`, one can recover (7) of this paper by using (16) of `adjoint matching` (with the running cost $f\equiv0$).

    B. The classifier $f^n$ in (9) is the same as the optimal control (17) in `adjoint matching`.

    C. Assumption 5 of this paper basically requires the OU process in the forward step of the prior model (pretrained DM) to converge to the equilibrium Gaussian distribution. In this context, the initial random variable $X_0$ and the terminal random variable $X_T$ in under the prior dynamics (pretrained DM) are actually almost independent. This means that the `initial value function bias problem`  in `adjoint matching` is **not** present and there is no need for memoryless noise schedule.

2. The no-regret assumption is rather strong.

### Questions
Please see the discussion above.

### Soundness
2

### Presentation
3

### Contribution
2
