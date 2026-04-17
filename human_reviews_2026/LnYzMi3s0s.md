# Understanding Sampler Stochasticity in Training Diffusion Models for RLHF

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Reinforcement Learning from Human Feedback (RLHF) improves pretrained generative models, and its sampling design is important for training reliable, high-quality models. In practice, stochastic SDE samplers promote exploration during training, while deterministic ODE samplers enable fast, stable inference; this creates a discrepancy in sampling stochasticity that induces a preference-reward gap. In this paper, we establish a non-vacuous bound on this gap for general diffusion models and a sharper bound for Variance Exploding (VE) and Variance Preserving (VP) models with (mixture) Gaussian data. Methodologically, we leverage the stochastic gDDIM scheme to attain arbitrarily high stochasticity while preserving data marginals, and we evaluate, under multiple preference rewards, the performance of RL algorithms (e.g., log-likelihood and group-relative policy variants). Our numerical experiments validate that reward gaps consistently narrow over training, and ODE sampling quality improves when models are updated using higher-stochasticity SDE training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the mismatch between stochastic and deterministic samplers during inference when finetuning diffusion models with RLHF. To that end, the authors define the improvement of finetuning as the expected reward difference between finetuned determinisitic ODE sampler and the pretrained model. Moreover, they define the reward gap as the difference between the expected reward obtained by finetuned ODE and SDE samplers.
The authors conduct theoretical analyses for Variance Exploding (VE) and Variance Preserving (VP) processes in a tractable setting for Gaussian and GMM target densities.
The empirical study focuses on two T2I settings: First, finetuning of Stable Diffusion 1.5 with DDPO, and second, FLUX1 with mixGRPO, where the authors present several observations.

As a disclaimer, I cannot make any statement about the significance of the theoretical results presented in this paper.

### Strengths
This paper sheds light on a timely and important question, i.e., how deterministic inference and stochastic finetuning are connected. The authors conduct both a theoretical analysis in a tractable setting as well as an empirical study.

### Weaknesses
The paper and results are hard to follow for someone who is not very familiar with RLHF for T2I (like myself). I think the authors should explain the experimental setup better (see Questions for some things that may need clarification).

For the remaining weaknesses, please see Questions.

### Questions
- Are the results for the SDE samplers in Table 1 produced with the same $\eta$ used for training? 
- What happens if we train with $\eta < 1$? 
- I'm a bit confused about Table 2: If I understand it correctly, $T$ is the time horizon of the dynamical system. However, in Table 2 it is a discrete quantity called 'steps'. I assume the authors refer to the number of diffusion steps as $T$ (?). Does it mean you use a fixed number of diffusion steps during training? 
- Regarding the observation: 'High Stochasticity Benefits Moderate Time Steps' -> Does this also hold for smaller or larger time steps?
- What is meant by $T=0$?
- Regarding Figure 3: How can the authors conclude 'indicating that image quality improves for both samplers'? From my understanding the reward gap does not give any insights on image quality but only on the relative performance between SDE and ODE sampling. Could the authors please clarify. 
- 'This is also consistent with our empirical observation that the performance of T2I generation deteriorates when fine-tuning with very large $\eta$' -> Would be good if the authors could reference the results that support this claim
- Regarding Theorem 3.2: Why did the authors not consider general shifted reward functions, i.e. $r(x) = -(x - \mu)^2$? This would help to understand how the shift of the reward function affects the theoretical result.
- Both Theorem 3.2 and Corollary 3.1 are given for very specific reward function. How transferable are these results?

Some minor comments:
- The quality of the figures 3 and 4 could be improved (minor).
- Line 282: The authors refer to Appendix xxx
- Line 375: contray

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the reinforcement learning from human feedback (RLHF) for text-to-image models. More specifically, examining the impact of feedback being given on stochastic draws from diffusion SDEs, whereas inference uses the corresponding deterministic ODEs.

### Strengths
- The problem formulation is somewhat original, and certainly of commercial relevance given the massive interest in text-to-image models.
- The writing and mathematical presentation is mostly clear.
- The introduction, background and related work sections are comprehensive and easy to follow.

### Weaknesses
**Tentatively incremental contributions**

I'll willingly admit I'm not an expert on RLHF and the literature which this paper builds upon. However, my impression is that the main results are direct consequences of other works - putting the novelty into question. However, I'm open to be convinced otherwise.

**Relevance of the theoretical results unclear**

First, I find it difficult to interpret the generality of the results in sections 3.1-3.3. Are they primarily pedagogical examples since they happen to be rare examples of where things can be computed in closed form, or are they of practical interest in themselves? 

Second, the general result in section 4 relies on dissipativity of f (Assumption 4.1.1). Why would that hold? (I believe conditions 2 can be fulfilled, especially if you use e.g. ReLU activations.)

**Minor things:**
- l.90-96 essentially repeat the preceding paragraph. I suggest to merge them.
- l.107-119 could be omitted, in my opinion.
- Most figures are too small, in particular the text within them. 
- l.195 "th prompt"
- l.200 "isotopic Gaussian"
- l.200 sure you can use Monte Carlo sampling, but won't the variance be huge?
- l.207 "std" and l.211 "clip", please use proper mathematical notation.
- l.226 in eq. (8), how do you evaluate KL - I assume you don't have access to the densities?
- l.232 and elsewhere, please use \text{} for descriptive subscripts like REF, SDE, ODE and italics for variables like $t$.
- l.313 I suppose $W_2$ means Wasserstein-2 distance?
- Table 1: you're most likely using too many significant digits. (Based on your plots and how noisy RL typically is)
- Figure 3: given the amount of noise, I think only the left-most figure would allow you to draw any conclusions.

### Questions
- As stated above, I see the practical relevance of the problem, but the *scientific* relevance is more questionable. Isn't the question of stochastic vs. deterministic more broadly applicable to, say, distributional reinforcement learning? I believe such a formulation would have a greater and more lasting impact.  
- Please motivate the relevance and validity of the theoretical results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the gap between SDE samplers and ODE samplers in RLHF fine-tuning of diffusion models. Theoretical bounds are derived and numerical experiments are conducted.

### Strengths
1. A theoretical bound is derived for SDE vs ODE samplers in fine-tuning of diffusion models.
2. Comprehensive numerical experiments are conducted to support the claims.

### Weaknesses
I have several major concerns:
1. I think the gap between SDE and ODE samplers is confusing to me. As they are equivalent, why is there a gap after fine-tuning. One should expect the same distribution of $ Y_T $ regardless which sampler is used. The authors should elaborate this point very carefully.
2. Theorem 4.1 is not that interesting as it looks. First, the proof seems to be incorrect. One inequality sign seems to be flipped when applying Young's inequality. Second, Assumption 4.1 (1) does not hold for VP SDE as shown in [1] unless you modify the OU process. Hence, I strongly suspect the bound in Theorem 4.1 is not informative.
3. The definition of $ Y_t^\theta $ is not clear. In particular, the authors should explain what is parameterized. Usually, fine-tuning starts with a given reference model, e.g., a pre-trained score network. In this case, choosing $ \theta $ as the optimal parameter associated with the optimal distribution is not correct; see [2]. This is because one has to find the optimal score network following the sampling dynamics instead of solving an unconstrained entropy-regularized optimization problem. Therefore, I do not think the results in Sections 3.1 and 3.2 are sound. 

[1] Tang, Wenpin, and Hanyang Zhao. "Contractive diffusion probabilistic models." arXiv preprint arXiv:2401.13115 (2024). \
[2] Han, Yinbin, Meisam Razaviyayn, and Renyuan Xu. "Stochastic Control for Fine-tuning Diffusion Models: Optimality, Regularity, and Convergence." Forty-second International Conference on Machine Learning.

### Questions
1. I suggest shortening the review in page 3 - 4
2. Line 282, "Appendix xxx"

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper considers the finetuning of diffusion models using RLHF. It addresses the "reward gap" between stochastic SDE samplers used in training and deterministic ODE samplers used at inference. The paper aims to characterize this gap theoretically and validate it empirically.

### Strengths
The paper's strengths are its theoretical contributions:
1. Formalizing the theory of the SDE-ODE reward gap in the continuous-time limit.
2. Developing sharp bounds for this gap for VE and VP models with Gaussian and Gaussian Mixture targets and providing a general bound for arbitrary distributions.

### Weaknesses
Despite its theoretical novelty, the paper suffers from fundamental weaknesses:
1. The paper's central flaw is that its theory is practically limited. The bounds apply only to continuous-time processes, ignoring the discretization error from using $N$-step samplers, which is the dominant error in practice. The authors even cite (Liang et al., 2025), which provides the tools for this analysis, but they fail to apply this to their own work (e.g., their Section 3.3), making the theoretical contribution less significant.
2. The experiments are insufficient. Using Stable Diffusion 1.5  as a primary testbed is not enough. While the paper uses FLUX.1, it compares it against SD 1.5 instead of the relevant SOTA benchmark, SDXL. This omission makes the empirical conclusions scientifically unsound.
3. The paper quality is far below ICLR standards. It contains uncorrected placeholders (e.g., "Appendix xxx" on line 288 ), not introduced bullet points in Section 1, and uninterpretable figures (e.g., Figures 3 and 4) that are low-resolution screenshots from wandb without even the legends.

### Questions
In addition to weaknesses, I have the following questions:
1. Can the authors provide a bound on the discretization error for their gDDIM sampler? How can we be sure this un-analyzed error does not dominate the stochasticity gap you have bounded?
2. You cite Liang et al. (2025). Why did you not apply this framework to your Gaussian Mixture analysis to provide a practical, end-to-end bound?

### Soundness
1

### Presentation
1

### Contribution
2
