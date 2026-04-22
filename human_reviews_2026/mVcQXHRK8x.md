# Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Test-time alignment of large language models (LLMs) attracts attention because fine-tuning LLMs requires high computational costs. In this paper, we propose a new test-time alignment method called adaptive importance sampling on pre-logits (AISP) on the basis of the sampling-based model predictive control with the stochastic control input. AISP applies the Gaussian perturbation into pre-logits, which are outputs of the penultimate layer, so as to maximize expected rewards with respect to the mean of the perturbation. We demonstrate that the optimal mean is obtained by importance sampling with sampled rewards. AISP outperforms best-of-n sampling in terms of rewards over the number of used samples and achieves higher rewards than other reward-based test-time alignment methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new method for aligning large language models at test time without fine-tuning. It works by adding Gaussian perturbations to the model’s pre-logit outputs and using importance sampling to adaptively update the perturbation mean for maximizing expected rewards. The method achieves higher reward efficiency and better alignment than existing test-time approaches (for example, BestOfN).

### Strengths
1. The method seems novel, which uses the optimal formulation for the test-time alignment problem.

### Weaknesses
1. Lack of important literature review and baseline comparison on more recent test-time alignment papers, such as GenARM [1], PAD [2], which performs better than ARGS that is discussed in this paper. Moreover, GemARM also tries to maximize the value function, so it can be worthwhile to compare with it conceptually or mathematically.
2. The motivation of using the optimal control formulation is not clear. KL-constrained reinforcement learning is a commonly used framework, and it is not clear what the benefits are using the optimal control formulation discussed in this paper.
3. The method assumes that the pre-logits follow a Gaussian distribution. This does not seem to hold in practice.

[1] GenARM: Reward guided generation with autoregressive reward model for test-time alignment. In International Conference on Learning Representations, 2025.
[2] PAD: Personalized alignment at decoding-time. In International Conference on Learning Representations, 2025.

### Questions
What is the main benefits of using the optimal control formulation instead of the commonly used RL formulation?

### Soundness
2

### Presentation
1

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
This paper proposes a training-free method called Adaptive Importance Sampling on Pre-logits (AISP), a test-time alignment technique that leverages stochastic model predictive control. By injecting Gaussian noise into the pre-logits and optimizing the perturbation mean to maximize the expected reward, AISP eliminates the need for data collection or additional computation during training. Experimental results show that AISP outperforms Best-of-N and other reward-based test-time alignment methods.

### Strengths
- AISP eliminates the need for training and data collection in test-time alignment.
- The integration of adaptive importance sampling with model predictive path integral (MPPI) control is novel and well-motivated.
- The analysis of modeling pre-logits $z$ as Gaussian distributions and its connection to Best-of-N (BoN) is insightful.

### Weaknesses
- AISP introduces numerous hyperparameters, including the standard deviation $\sigma$, the softmax temperature $\lambda$, MPPI coefficient $\alpha$, number of iterations $k$, and window size $\tau$. This complexity limits the practicality of AISP. Moreover, the paper does not sufficiently analyze the sensitivity of these hyperparameters across different tasks and models, or their interactions with standard generation parameters such as temperature, top-$p$, and top-$k$ sampling.
- Although AISP shows promise, its improvements over existing methods (as reported in Table 1) appear relatively small improvement, particularly in terms of diversity and coherence.
## References
[1] Scaling Laws for Reward Model Overoptimization. ICML 2023.

[2] BOND: Aligning LLMs with Best-of-N Distillation. ICLR 2025.

### Questions
- A deeper discussion of the hyperparameters $\alpha$ and $\tau$ is needed. What specific roles do they play? Why not set $\tau$ equal to the full sequence length $T$? Is there a configuration of hyperparameters that remains robust across different models and settings?
- Since BoN demonstrates strong robustness under large sampling budgets in mitigating reward over-optimization [1, 2], it is important to examine whether AISP maintains better robustness in the Reward-KL regularization trade-off as the sampling budget 
$𝑘$ increases. Does AISP still outperform BoN under large budgets (e.g., $k=\{128,256\}$)? Between scaling the number of responses per iteration and the number of iterations, which factor is more vulnerable to reward over-optimization?
- Although the authors claim that AISP and BoN have comparable sequential and parallel computational costs, it's still necessary to include empirical measurements of inference time under identical sampling budgets. As AISP involves additional Gaussian sampling and iterative importance weight updates, quantifying this overhead is essential.
- While not really necessary, it would be informative to compare AISP with standard RLHF training methods such as PPO and DPO.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AISP, a training-free, test-time alignment method that injects Gaussian perturbations into the pre-logits over a fixed control window and updates the perturbation mean by adaptive importance sampling. The objective is cast via a free-energy bound, yielding an optimal pre-logit distribution proportional to $exp(r/\lambda)p(V)$, since this is intractable, the mean is iteratively estimated with weighted samples. The authors argue the Gaussian assumption is consistent with softmax classifiers, show that AISP reduces to best-of-N (BoN) as as $\lambda$→0, and report higher reward and GPT-4 win-rates than BoN, ARGS, and RE-Control on SHP/HH-RLHF.

### Strengths
1. This paper maps decoding-time reward maximization to sampling-based optimal control in pre-logit space; derivation via a free-energy lower bound is standard but cleanly presented.

2. The method is training-free and lightweight; also the adaptive importance sampling loop is easy to implement.

3. Empirical results show consistent reward and win-rate improvements over BoN with the same total samples.

### Weaknesses
1. The control-theoretic view and MPPI-style derivation are known; the main step is moving importance sampling to pre-logit trajectories with a Gaussian prior. The reduction to BoN for λ→0 underscores AISP as a structured BoN generalization to me rather than a new paradigm.

2. Tasks are preference datasets (SHP, HH-RLHF) with reward-model scoring; diversity/coherence sometimes degrade, and win-rate uses small paired samples. No tests on reasoning/code/math where long-horizon dynamics might stress the method.

3. AISP requires sequential updates; the paper does not report wall-clock vs. BoN under the same accelerator budget, nor scaling under constrained interleave with other requests.

4. The Gaussian model for pre-logits is only heuristically connected to softmax; no quantitative validation of this assumption.

### Questions
1. Can you quantify the Gaussian pre-logit assumption (per-token goodness-of-fit, across layers/models)? Any evidence that non-Gaussian priors would materially help/hurt AISP?  

2. Please report wall-clock and throughput (tokens/sec) vs. BoN for matched total samples under realistic GPU budgets? 

3. For RE-Control, what data/compute were used to train the value function, and how does AISP compare at equal wall-clock including that one-time cost?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose an inference-time alignment framework, AISP (Adaptive Importance Sampling on Pre-logits). The authors frame this alignment task as a sampling-based optimal control problem. The core idea is to apply a stochastic Gaussian perturbation to the pre-logits, i.e., the penultimate layer outputs at each decoding step. AISP then uses adaptive importance sampling to iteratively update the mean of this perturbation, effectively creating a control signal that explores the generation space and guides the model toward high-reward sequences. Experimental evaluations on standard benchmarks on HH and SHP show that AISP outperforms competitive baselines such as  BoN, ARGS, and RE-Control.

### Strengths
1. The paper is generally well written. The proposed AISP approach operates at inference time and therefore does not require training value functions, unlike RE-Control.

2. The authors provide detailed hyperparameter ablations, a KL-divergence analysis, and a thorough comparison of batched AISP with BoN.

3. The empirical evaluation is comprehensive, covering multiple base LLMs and reward models.

### Weaknesses
1. The performance improvement on HH-RLHF appears incremental, and in many cases, BoN outperforms AISP. With only two datasets, it is difficult to fully assess AISP’s empirical effectiveness. I recommend evaluating on additional datasets to more clearly demonstrate the gains.

2. [Minor] While the paper includes strong baselines, adding comparisons with the controlled decoding literature [1, 2] would further strengthen the experimental section.

[1] Mudgal, S., Lee, J., Ganapathy, H., Li, Y., Wang, T., Huang, Y., Chen, Z., Cheng, H.T., Collins, M., Strohman, T. and Chen, J., 2023. Controlled decoding from language models. arXiv preprint arXiv:2310.17022.

[2] Chakraborty, S., Ghosal, S.S., Yin, M., Manocha, D., Wang, M., Bedi, A.S. and Huang, F., 2024. Transfer q-star: Principled decoding for llm alignment. Advances in Neural Information Processing Systems, 37, pp.101725-101761.

### Questions
Please see weakness 1.

### Soundness
2

### Presentation
3

### Contribution
2
