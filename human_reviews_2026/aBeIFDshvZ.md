# Diffusion Alignment as Variational Expectation-Maximization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Diffusion alignment aims to optimize diffusion models for the downstream objective. While existing methods based on reinforcement learning or direct backpropagation achieve considerable success in maximizing rewards, they often suffer from reward over-optimization and mode collapse. We introduce Diffusion Alignment as Variational Expectation-Maximization (DAV), a framework that formulates diffusion alignment as an iterative process alternating between two complementary phases: the E-step and the M-step. In the E-step, we employ test-time search to generate diverse and reward-aligned samples. In the M-step, we refine the diffusion model using samples discovered by the E-step. We demonstrate that DAV can optimize reward while preserving diversity for both continuous and discrete tasks: text-to-image synthesis and DNA sequence design. Our code is available at https://github.com/Jaewoopudding/dav.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper develops an approach to “Diffusion Alignment as Variational Expectation-Maximization” (DAV) that alternates between two complementary phases: the E-step, which is essentially an exploration step that aims to discover diverse and high-reward samples from the variational posterior; and the M-step, “amortization”,  which refines the diffusion model using samples identified from the E-step.

### Strengths
The DAV approach is built on solid technical foundations as outlined in \S4. For instance, the E-step uses gradient-based guidance and importance sample to enhance the exploration. The M-step minimizes a mode-covering objective that incentivizes the covering of all diverse modes generated in the E-step. 

The combined E-M steps iteratively refine the model towards a multi-modal aligned distribution; and this overcomes problems like over-optimization and mode collapse that often arise in RL.

### Weaknesses
The M-step involves two variations, in addition to the standard DAV objective in (10), there’s a variation,  DAV-KL in (11). From the experimental results in Table 1, it’s not clear which one to use and when, except perhaps the ad hoc trial and error.

### Questions
Can the author(s) shed some light on how to choose the value of the KL-coefficient \lambda in (11), which is meant to control “the trade-off between aligning with the expert policy and pre-serving the pretrained model”? In particular, is the value \lambda robust or not with respect to downstream applications?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces DAV, a novel framework for fine-tuning pre-trained diffusion models. The authors motivate the work by claiming to address reward over-optimization.

### Strengths
1. This work formulates diffusion alignment as an iterative Variational Expectation-Maximization, which appears to be a new and interesting theoretical lens for diffusion fine-tuning.2. 
2. The proposed DAV framework enjoys broad applicability. It can accommodate both continuous and discrete settings.
3. The presentation of this work is easy to follow.
4. The empirical results show DAV and DAV-KL enjoy superiority over multiple strong baselines, such as DDPO, DRaFT.

### Weaknesses
1.  The main comparison in Figure 3, which plots aesthetic reward against diversity/naturalness, is confusing and potentially incomplete. The reported performance of the RL-based if they are properly trained with "suitable" KL penalty (which might be non-trivial to choose). This raises questions about the optimal tuning of these baselines. 
2. Furthermore, the analysis in Figure 3 omits purely inference-time methods, which are often competitive in image experiments.
3. As noticed by the authors, this method has non-negligible computation costs. The E-step involves substantial "additional test-time computation" through gradient-guided search. In large-scale diffusion model finetuning, DDPO already takes much time to converge (compared to the fastest direct propagation). It is important to quantitatively present the added training-time overhead of DAV relative to DDPO.

### Questions
1. For the results presented in Figure 3, do the images for each algorithm come from a single training run, or are they gathered over multiple runs? If from a single run, please report the standard deviation rather than only the mean.
2. For discrete finetuning, usually it's straightforward to test both DNA and RNA sequences. Can the authors provide explanations on why only the DNA enhancer is tested?
3. See other questions above

### Soundness
2

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
The authors propose DAV, an alignment algorithm for diffusion models via variational expectation maximization. In the E-step, DAV employs test-time search algorithms to generate samples from the reward-weighted posterior distribution. In the M-step, the diffusion model is fine-tuned using the samples from the E-step. The authors demonstrate the effectiveness of DAV on both the continuous text-to-image generation task and the discrete DNA sequence design.

### Strengths
- The idea to formulate diffusion alignment as a variational expectation-maximization problem is interesting.
- The paper is well-written, and the theory and method are well-motivated.
- Experiments showcase the effectiveness of DAV.

### Weaknesses
- The searching algorithms lead to computational overhead. Therefore, a fairer comparison with the baselines should also take the computational cost into account. For example, it would be helpful to compare model performance under the same computational budget and the performance scaling curve of TR2-D2 as the computation increases.
- The value of 3-mer correlation for DNA sequence design is significantly lower than those reported by baselines, e.g., in DRAKES paper the value is 0.887, much higher than DAV (0.397), while in table 2 and figure 5, it is only 0.229. Also, the target and naturalness are two competing properties, and one can get a higher value of one property by sacrificing the other via hyperparam tuning or using different training epochs. Does DAV have Pareto optimal performance compared to baselines? 
- The E-step can lead to an inaccurate estimation of the posterior distribution, due to the limited sample size and the value estimation error in the test-time algorithms. How does this affect the M-step optimization, and is the model robust to a suboptimal posterior distribution (e.g., with fewer samples or inaccurate samples)?

### Questions
Please refer to the **weaknesses** section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a diffusion alignment method (DAV) that alternates between test-time searches as Expectation steps and online refinement of the diffusion model as Maximization steps. Specifically, the diffusion alignment problem is formulated as a soft RL objective, whose evidence lower bound is optimized an EM algorithm. In the E-steps, DAV draws posterior samples given a reward-tilted distribution; while in the M-steps, DAV distills the sampled trajectories into the diffusion model. Experimental results show the effectiveness of DAV compared to existing RL and direct preference optimization methods for both continuous and discrete diffusion models.

### Strengths
- The paper offers a fresh perspective by aligning diffusion models with the EM algorithm. I especially appreciate this idea because the multi-round iterative alignment could potentially help in settings where the reward is costly or intractable to evaluate—for example, when it requires human evaluation or expensive wet-lab experiments.
- The proposed method is accompanied by rigorous derivations and theoretical guarantees.

### Weaknesses
- Experiments only include one example for continuous diffusion and one for discrete diffusion. The case for generality would be stronger with additional tasks (e.g., compressibility or prompt alignment as in DDPO). Moreover, some of the recent methods are not included or discussed as well, such as DSPO[1], DanceGRPO[2].
- The EM algorithm may be substantially more expensive than the methods it is compared to (e.g., DRaFT or DDPO), given the test-time search required in each expectation step. However, there is currently no analysis on the runtime or convergence speed of DAV.

[1] Zhu et al. "DSPO: Direct Score Preference Optimization for Diffusion Model Alignment", ICLR 2025.

[2] Xue et al. "DanceGRPO: Unleashing GRPO on Visual Generation", arXiv: 2505.07818.

### Questions
- For the discrete diffusion model alignment, I am curious of how DAV compare to test-time sampling algorithms such as [3,4], which also consider the same DNA enhancer task, and also alignment methods designed for discrete diffusion models (e.g., [5,6]).

[3] Li et al. "Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding", arXiv: 2408.08252.

[4] Chu et al. "Split Gibbs Discrete Diffusion Posterior Sampling", NeurIPS 2025.

[5] Borso et al. "Preference-Based Alignment of Discrete Diffusion Models", ICLR 2025 Bi-Align Workshop.

[6] Zhu et al. "LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models", arXiv: 2505.19223.

### Soundness
2

### Presentation
2

### Contribution
3
