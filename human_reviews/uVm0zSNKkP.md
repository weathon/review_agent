# Skip the Steps: Data-Free Consistency Distillation for Diffusion-based Samplers

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 3

## Abstract
Sampling from probability distributions is a fundamental task in machine learning and statistics. However, most existing algorithms require numerous iterative steps to transform a prior distribution into high-quality samples, resulting in high computational costs and limiting their practicality in time-constrained and resource-limited environments. In this work, we propose consistency samplers, a novel class of samplers capable of generating high-quality samples in a single step. Our method introduces a new consistency distillation algorithm for diffusion-based samplers, which eliminates the need for data or full trajectory integration. By utilizing incomplete sampling trajectories and noisy intermediate representations along the diffusion process, we efficiently learn a direct one-step mapping from any state to its corresponding terminal state in the target distribution. Moreover, our approach enables few-step sampling, allowing users to flexibly balance compute costs and sample quality. We demonstrate the effectiveness of consistency samplers across multiple benchmark tasks, achieving high-quality results with one-step or few-step sampling while significantly reducing the sampling time compared to existing samplers. For instance, our method is 100-200x faster than prior diffusion-based samplers while having comparable sample quality.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents a distillation framework based on consistency models, for general diffusion samplers (e.g., DDS, DIS).
Compared to consistency models for content generation, the most challenge in this setting is the absence of datasets (from the unnormalized target distributions).
To distill a pre-trained diffusion sampler, the authors first simulate training data from the prior distribution and then optimize the consistency distillation objective function, while some architectures are specially designed to reduce the computational cost.
The experiments support the claim of the faster sampling speed of consistency samplers, especially the one-step sampling quality.
This paper is clearly written and easy to follow, but I still have some questions and critiques as detailed below.

### Strengths
- This paper is clearly written and easy to follow.
- The problem considered in this paper is interesting and may inspire the following works. As far as I know, the authors are the first to consider distill the diffusion samplers which target at an unnormalized distribution.
- The proposed distillation approach seems straightforward but practical, and its performance is evaluated on several data sets.

### Weaknesses
- Minor contribution. I cannot see a significant difficulty in directly applying consistency distillation (Song,2023) to distilling pre-trained diffusion samplers. In my point of view, the most important ingredients of the distillation sampler, including the data-prediction matching idea, the distillation loss in Eq (9), and the sampling scheme, all come form the CM paper (Song, 2023). I admit there are several minor differences, e.g., the way to obtain the training data (N-step simulation from prior, instead of T-step), but these are natural choices with the absence of a dataset.
- The experiments are insufficient to demonstrate the superiority of distillation sampler. The target distributions in section 5.1 are all two-dimensional, and the $d$ for DW is not specified. Exploration on high-dimensional and complicated target distributions is necessary. Some examples include the $\phi^4$ model in (https://arxiv.org/abs/2402.10758) and the VAE model in (https://arxiv.org/pdf/2310.02679).
- The background spends much space in introducing the formulation and training objective of diffusion samplers, but this is not very related to the distillation problem, since a sampler is pre-given. I suggest to cut down the length of the background section.
- As the proposed method requires simulating the pre-given diffusion sampler for many steps, it naturally raises concerns about the computational cost in the training process. Adding more information on the computation time and memory would be better.

### Questions
- What does the $p_{target}$ mean in the equation of the Gaussian Mixture Model in section 5.1?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces "consistency samplers," a novel approach to generating high-quality samples from probability distributions in a single step, rather than requiring hundreds of iterative steps like traditional methods such as Markov Chain Monte Carlo (MCMC) and diffusion-based samplers. The key innovation is a consistency distillation algorithm that can learn to map directly from a simple prior distribution to a target distribution without needing access to training data or full sampling trajectories, making it much more computationally efficient.

The method works by distilling knowledge from pre-trained diffusion-based samplers, but does so efficiently by only using incomplete sampling trajectories and intermediate noisy representations. Rather than requiring full trajectory integration, the approach leverages probability flow ordinary differential equations (ODEs) to generate deterministic consecutive points for training. This allows the consistency sampler to learn the mapping between states while using roughly half the parameters of traditional diffusion-based samplers.

Through extensive experiments on multiple benchmark tasks, including Gaussian mixture models, double well potentials, ring distributions, and image data, the authors demonstrate that their method can achieve comparable or better sample quality compared to traditional approaches while being 100-200 times faster. They also provide theoretical guarantees showing that under certain conditions, the learned consistency sampler can approximate the true consistency function with arbitrary accuracy as the step size of the ODE solver decreases. This represents a significant advance in making sampling methods more practical and efficient for real-world applications.

### Strengths
The proposed method is novel, and is the first algorithm to propose to accelerate generation for unnormalized sampling problems. Further, this paper also gives a practical solution based on consistency models to achieve one-step generation. This provides flexibility to balance between speed and quality through optional refinement steps. The training doesn't require training data. It also avoids the need for full trajectory integration during training. It also provides theoretical guarantees for the method's convergence.

### Weaknesses
The algorithm is not entirely clear to me; for the setting of sampling from a given unnormalized density function, there is no appearance of the target density in Algorithm 1. I guess the whole algorithm might be first train with DDS method, and do consistency training based on the pretrained DDS model? 

Another drawback is from the experiment side. Have you tried problems with 100+ / 1000+ dimensions like log Gaussian Cox process? Now most experiments are only in 2 dimensions.

### Questions
None.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Sampling of high-quality samples in time-constrained settings is a challenging problem. Consistency Models (CMs) allow for faster training and efficient sampling of diffusion models from complex probability distributions. The work builds on top of CMs and proposes a new class of samplers called Consistency Samplers (CS). CS distills representations from intermediate sampling steps along the diffusion process in order to learn a one-step mapping from any sampling step to the terminal step in the target distribution. This bypasses the need to generate and integrate complete sampling trajectories. CS enable one-step and multi-step sampling in order to trade-off computational cost for sample quality. Empirical evaluation on a range of probability distributions demonstrates faster samping when compared to prior diffusion-based samplers.

### Strengths
* The paper is well written and easy to understand.
* The work presents intuitive explanations with a theoretical result quantifying the gap betwen the learned and true consistency samplers.

### Weaknesses
* **Data-free sampling:** The paper claims that the proposed consistency sampler is data-free. However, the claim is not well supported as similar to Consistency Models (CMs) [1], the sampler is initialized at $x_{0} \sim p_{prior}$ which is a data sample. If I understand correctly, the key difference between CS and CM is that the sampler is executed for shorter run (i.e- terminated early) instead of running it upto full T sampling steps. While this requires less data in practice, the sampling process in itself cannot be cateogrized as data-free.
* **Similarity to CM:** A main concern from the paper is its contribution over the previously proposed CMs. It is unclear as to what is the additional contribution over CMs. CS terminates the sampling steps early and distills representations from intermediate samples. This can be inferred as a special case of CM sampling with shorter sampling steps. Even in the case of multi-step sampling, differences remain unclear as the training and sampling process remain exactly similar.
* **Comparison with CM:** While the authors compare CS to a range of prior diffusion sampling baselines, a clear comparison with CM is missing. Since the sampler directly borrows its training, distillation and short sampling steps from CMs, it would be a fair comparison to evaluate CS with CMs for shorter runs. Does CS provide high fidelity samples when compared to CMs? Can CS identify more modes early on when compared to CMs? What speedup and additional advantages does CS offer over prior CMs? A direct comparison with CMs will help answer these questions.
* **Benchmarks:** While the work presents empirical evaluation on a range of datasets, experiments are limited to small toy tasks with limited complexity and number of modes. CMs demonstrate benefits and performance improvements over prior methods when executed in higher dimensions with complex multi-modal data samples. Could the authors comment on how the sampler can be executed with high dimensional data such as images (CIFAR, ImageNet)? Experiments are not required, but an intuitive understanding of the sampler and its applicability to other tasks would benefit the paper as these modalities require samplers to be conditioned on data and run for longer sampling steps.

[1]. Song, Yang, et al. "Consistency models." arXiv preprint arXiv:2303.01469 (2023).

### Questions
Please refer to weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This work considers pre-trained diffusion-based models. Rather than simulating from the controlled diffusion -- which is costly, especially for fine time discretisations -- the authors propose to amortise this problem by learning so called "consistency" mappings which map samples from any point along the trajectory to the target distribution.

### Strengths
1. **Clarity.** The writing is mostly clear and concise and thus relatively easy to read.

2. **Originality.** To my knowledge, this method is novel but I am not an expert on diffusion-based models.

3. **Correctness.** The method overall seems justified (but note the remarks about Theorem 1 below).

### Weaknesses
**Main comments:**

1. **Computational cost of training the consistency function.** If I understand Table 1 correctly, the claimed "100--200$\times$" speedup does not account for the cost of training the consistency function $f_\theta$. If this is correct, then I think the claims of speedups must be substantially softened. More generally, it would seem to me that the proposed methodology requires simulating a large number of trajectories according to the probability-flow ODE from Equation 8. However, once we have sampled a large number of trajectories according to (8), aren't we already done? I mean, then we already have samples from the target, no? [yes, Algorithm 1 only uses potentially partial trajectories but the computational complexity is the same].

2. **Assumptions in Theorem 1.** Do we not need additional assumptions on $f_\theta$ in Theorem 1? It seems to me that the stated assumptions would be satisfied for /any/ Lipschitz function $f_\theta$? For instance, wouldn't the assumptions hold for $f_\theta \equiv \mathrm{const}$ being some constant function? Or are there additional assumptions buried in the terminology "consistency sampler"/"consistency mapping"? If so, these should be stated here more clearly.

3. **Numerical examples are low-dimensional.** The method is applied only to fairly simple examples which all seem to be two-dimensional (except for the "double-well" example for which I did not find a mention of the dimension). I think there should be higher-dimensional examples.



**Other comments:**

* Is there a difference between "consistency sampler" and "consistency function" (e.g., in Theorem 1 or Section 4.5)? The former is never defined. Likewise, the term "consistency mapping of the PF ODE defined by the control $u$" in Theorem 1 is never defined.



**Typos:**

* L117: Grammar in "$T$-step MCMC transition"

* Eq. 1--3: Is "$\mathbb{w}$" missing a subscript?

* Eq. 8: missing punctuation

* Algorithm 2: Should $n$ be $N$ instead?

* Bibliography: In many places, proper nouns are not capitalised.

### Questions
1. How realistic is the Lipschitz assumption in Theorem 1? Can you give examples in which this is verifiable?

2. What was the dimension $d$ in the double-well example?

### Soundness
2

### Presentation
3

### Contribution
2
