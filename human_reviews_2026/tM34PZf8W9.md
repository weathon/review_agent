# Reinforcement Learning with Discrete Diffusion Policies for Combinatorial Action Spaces

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Reinforcement learning (RL) struggles to scale to large, combinatorial action spaces common in many real-world problems. This paper introduces a novel framework for training discrete diffusion models as highly effective policies in these complex settings. Our key innovation is an efficient online training process that ensures stable and effective policy improvement. By leveraging policy mirror descent (PMD) to define an ideal, regularized target policy distribution, we frame the policy update as a distributional matching problem, training the expressive diffusion model to replicate this stable target. This decoupled approach stabilizes learning and significantly enhances training performance. Our method achieves state-of-the-art results and superior sample efficiency across a diverse set of challenging combinatorial benchmarks, including DNA sequence generation, RL with macro-actions, and multi-agent systems. Experiments demonstrate that our diffusion policies attain superior performance compared to other baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a novel framework to address the challenge of scaling Reinforcement Learning (RL) to large, discrete, and combinatorial action spaces using discrete diffusion models as the policy representation. The method frames the policy update as a distributional matching problem where an expressive diffusion model is trained to replicate a target distribution derived from Policy Mirror Descent (PMD). While the combination of expressive generative models like diffusion models with RL is a highly relevant research direction, the current submission suffers from fundamental weaknesses concerning algorithmic novelty, sample efficiency, and experimental validation in truly large-scale or combinatorial settings. Specifically, the on-policy nature of the training presents a major sample efficiency bottleneck that is neither justified nor quantitatively compared against standard methods. Furthermore, the core discrete diffusion adaptation appears straightforward, and the experimental environments are not sufficiently demanding to validate the paper's central claims. I recommend rejecting this paper.

### Strengths
Originality: The application of discrete diffusion models to directly model policies in a discrete/combinatorial RL setting is a novel and timely area of exploration. This represents an attempt to leverage the strong expressive power of modern generative models for complex policy representations.

Quality: The overall framework is logically constructed, and the connection between the policy update and Policy Mirror Descent (PMD) provides a clear theoretical grounding for the target distribution definition. The paper is generally well-written and easy to follow.

Clarity: The implementation details of the discrete diffusion process and the overall training loop, while basic, are presented clearly.

Significance: A successful integration of diffusion models that genuinely scales RL to vast combinatorial action spaces would be highly significant. However, the current submission does not convincingly demonstrate this scalability.

### Weaknesses
Poor Sample Efficiency of On-Policy Training: The decision to employ an on-policy training scheme is a critical limitation. Training highly parameterized generative models like diffusion models is computationally intensive and typically requires many gradient steps. The paper fails to provide any quantitative comparison of the sample efficiency (e.g., performance vs. number of environment steps) against standard on-policy or, more importantly, off-policy RL baselines on common benchmarks. Given the high cost of data collection in real-world scenarios, this lack of efficiency evidence is a major roadblock to the method's practical adoption.

Limited Algorithmic Novelty in Diffusion Adaptation: The discrete diffusion implementation appears to be a direct, simple translation of the standard Denoising Diffusion Probabilistic Model ($\text{DDPM}$) for discrete data (e.g., a variant of Categorical $\text{DDPM}$) to the RL domain. The paper lacks a substantive discussion on:

The specific technical challenges and adaptations required to make discrete diffusion stable and effective as a policy within an RL loop.

Why this diffusion formulation is superior in performance or expressiveness compared to other generative policy models (e.g., VAEs, normalizing flows) specifically for the RL task.

Unsubstantiated Claims on Combinatorial Spaces and Missing Baselines: Despite the title, the algorithm seems restricted to fixed, discrete action spaces. Its generalizability to truly combinatorial action spaces (e.g., generating sets, sequences, or permutations of arbitrary length/size) is not demonstrated. Furthermore, the paper entirely omits necessary comparisons with methods used for similar tasks:

There is no discussion or comparison with techniques that handle large action spaces via differentiable approximations, such as the Gumbel-Softmax trick, especially in its continuous formulation, which could offer insights into policy smoothness.

Lack of Advanced Diffusion Techniques and Objective Optimization: The paper relies on the standard Evidence Lower Bound ($\text{ELBO}$) for the diffusion objective. This choice is often suboptimal for computational efficiency and performance in modern diffusion models. The authors should explore and discuss more advanced alternatives:

The possibility of replacing the full $\text{ELBO}$ with a simpler and more computationally efficient loss function, such as a $\text{KL}$ divergence or a noise-matching loss (similar to the simplified objective in standard continuous $\text{DDPMs}$).

The potential for integrating techniques based on Tweedie's theorem or score matching to accelerate the reverse sampling process, which is the core of policy inference and execution. This omission suggests a basic implementation that does not leverage recent advances in the diffusion modeling literature.

Insufficient Experimental Scale: The experimental environments are not adequately challenging to support the paper's claims about handling large and complex action spaces. To truly validate the method's utility, experiments must be conducted on tasks with significantly higher dimensionality and action-space size. A strong validation would require testing the approach on larger-scale discrete environments, such as a high-dimensional, large grid-world variant of Frozen Lake (e.g., $64\times64$ or higher, perhaps with dense observations), following the spirit of scaled-up discrete $\text{DQN}$ research [1].

[1] Stochastic Q-learning for Large Discrete Action Spaces. 2024 ICML

### Questions
Efficiency Justification: Can the authors provide a rigorous comparison of the sample efficiency (environment steps) between the proposed on-policy diffusion method and a well-tuned off-policy baseline (e.g., $\text{DQN}$ or $\text{SAC}$) on a common benchmark like the discrete control tasks? What is the core algorithmic reason for choosing an on-policy approach over a more data-efficient off-policy formulation, given the computational cost of training the diffusion model?

Combinatorial vs. Discrete: Please provide a clear definition of the "combinatorial action spaces" tested. If the algorithm is applicable to truly combinatorial problems (e.g., generating a set of resources, ordering a sequence of deliveries), can the authors provide an experiment on a problem instance that is not simply a large, flat discrete space, and elaborate on how the diffusion chain structure is modified for that combinatorial output?

Objective Function Selection: Did the authors experiment with or consider replacing the $\text{ELBO}$ loss function with a simpler noise-matching objective (e.g., $\text{KL}$ or $\text{MSE}$) for the diffusion model training? If not, what is the theoretical or empirical justification for using the full $\text{ELBO}$ instead of a simpler, more computationally tractable surrogate loss common in $\text{DDPM}$ literature?

Sampling Acceleration: Have the authors investigated methods for accelerating the reverse sampling process, such as leveraging techniques based on Tweedie's theorem or incorporating techniques from fast sampler literature? Sampling efficiency is critical for policy deployment, and a discussion of this is highly relevant.

Scaling Experiments: To address the weakness of experimental scale, could the authors confirm if they have tested the approach on a significantly larger state/action space environment, such as a high-dimensional, large-scale grid-world task (e.g., a $64\times64$ grid with dense state observations) where the benefits of a highly expressive policy would be more pronounced?

### Soundness
2

### Presentation
2

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
This paper addresses the problem of RL with large, combinatorial discrete action spaces. The authors propose a new framework, RL-D^2, which uses a discrete diffusion model as the policy parameterization. The authors reframe the policy improvement step as a distributional matching problem. They leverage policy mirror descent (PMD) to define a stable target policy distribution and derive two practical loss functions. The framework is evaluated across three distinct and challenging domains: DNA sequence generation, online RL with long-horizon macro-actions in Atari, and cooperative multi-agent RL (MARL) in Google Research Football. The results show that the diffusion-based policies achieve state-of-the-art performance, demonstrating superior scalability and sample efficiency compared to baselines.

### Strengths
1. The authors demonstrate the effectiveness of the proposed method across three fundamentally different domains.
2. The empirical results are very strong. The method achieves state-of-the-art performance in all three domains.
3. The proposed idea of decoupling the RL objective from the representation learning is novel.

### Weaknesses
1. The core methodology (Section 4) is conceptually dense. A flowchart/figure illustrating the data flow of a single policy update would have vastly improved the paper's clarity and made the central contribution much easier to understand.
2. Figure 1 left is of low quality.
3. More explanations are needed on the baseline selection protocol.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
This work introduces RL-D2, a novel online RL framework that employs discrete diffusion models as expressive policies for large combinatorial action spaces.

By framing policy improvement as distributional matching to a stable target derived from policy mirror descent (PMD), the method decouples RL optimization from representation learning, yielding robust training dynamics. 

The authors propose forward and reverse KL objectives, with practical approximations for the intractable likelihood ratio in the reverse KL case, and introduce on-policy diffusion learning to align training with the policy’s generative process.

Extensive experiments across DNA sequence optimization, macro-action RL in MinAtar/Atari, and multi-agent football demonstrate state-of-the-art performance and superior efficiency compared to baselines.

### Strengths
- RL-D2 creatively combines discrete diffusion models with online RL via PMD-guided distributional matching, introducing on-policy diffusion learning and practical KL approximations

- Theoretical derivation  and thorough experiments deliver consistent SOTA results with strong efficiency gains

- Well-structured, with clear preliminaries, intuitive FKL/RKL distinctions, and effective figures/tables

### Weaknesses
- PMD Normalization Error: Eq. (3) incorrectly writes the denominator as Z(s)−1 instead of Z(s)

- FKL vs. RKL: A crucial missing experiment is an ablation study comparing FKL and RKL on the same benchmark.

### Questions
The use of K-step macro-actions implies an open-loop execution policy. In temporally sensitive environments like Atari, a single non-optimal action in the generated sequence $a = (a_1, ..., a_K)$ could be amplified, as the resulting state $s_i$ may render the rest of the planned sequence $(a_{i+1}, ..., a_K)$ suboptimal or even catastrophic. A standard 1-step, closed-loop policy $\pi(a_t|s_t)$ avoids this by re-evaluating the state at every step.

Could the authors justify the choice of this open-loop macro-action framework, which seems highly susceptible to compounding errors, over the seemingly more robust 1-step closed-loop approach for these environments? How does the framework mitigate this significant risk?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper tackles RL in very large, combinatorial discrete action. It proposes RL-D2: parameterize the policy with a masked discrete diffusion model and perform policy improvement by matching the diffusion policy to the analytic Policy Mirror Descent (PMD) target distribution. The authors (i) define the PMD target $ \pi_{MD} $ and frame improvement as minimizing a divergence between $ \pi_\theta $ and $ \pi_{MD} $, deriving practical forward-KL (FKL) and reverse-KL (RKL) objectives, and (ii) introduce “on-policy diffusion learning” to align the diffusion training distribution with inference.

### Strengths
The proposed method provides a clear treatment of both FKL (“mean-seeking”) and RKL (“mode-seeking”) updates with diffusion-specific practicalities (ELBO-based bound for FKL; IS-ratio approximations for RKL), which is a thoughtful design space exploration. 

The PMD foundation and the discrete diffusion setup are explained succinctly; the paper states contributions plainly and situates them against AR models and prior discrete diffusion work. 

If validated, RL-D2’s combination of stability (via PMD), expressivity (via diffusion), and efficiency (non-autoregressive sampling; fewer diffusion steps) could make combinatorial RL far more practical in macro-action planning and MARL. Gains over baselines across 3 domains bolster impact.

### Weaknesses
RKL inherits PMD’s guarantees, but the paper cannot compute exact likelihoods for diffusion policies and uses ELBO-based or single-step ratios. The bias factor $\Gamma$ in the ELBO-ratio estimator is acknowledged but unanalyzed; conditions under which this bias is small (or controlled) are unclear. A finite-sample or asymptotic analysis, or at least diagnostics correlating η̂ with true ratios in toy settings, would strengthen the claim of “strong theoretical guarantees” in practice. 

Football comparisons use an AR transformer baseline; it would be more convincing to include or justify against strong MARL baselines (e.g., MAPPO-style, value-decomposition methods) with joint-action coordination, not only sequence modeling. As is, the diffusion-vs-AR conclusion might conflate architectural class with training details. 

The paper shows RL-D2 scales with macro length and notes DQN-Macro fails for long macros. However, it’s not fully clear that baseline architectures/hyperparameters are tuned comparably at larger action cardinalities, nor how replay/priority settings affect them. A more stringent hyperparameter search for macro baselines, plus reporting compute used per method at each macro length, would address confounds. 

Minor: Several figures/tables say “confidential intervals” instead of “confidence intervals”

### Questions
In practice, do you use the hard-KL dual update every iteration? What target-KL values worked across domains, and how sensitive is performance to this hyperparameter? Any instability modes?

You use FKL for DNA/macro-Atari and RKL for MARL. Can you provide guidance/heuristics for selecting FKL vs RKL (e.g., entropy/temperature schedules, action-space sparsity, reward multimodality), with ablations on a common domain? 

Could you include or justify against MAPPO/QMIX-style baselines with similar feature extractors?

### Soundness
3

### Presentation
3

### Contribution
3
