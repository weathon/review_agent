# $\nabla$-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Scaling inference-time compute for Large Language Models (LLMs) has unlocked unprecedented reasoning capabilities.
However, existing inference-time scaling methods typically rely on inefficient and suboptimal discrete search algorithms or trial-and-error prompting to improve the online policy. In this paper, we propose $\nabla$-Reasoner, an iterative generation framework that integrates differentiable optimization over token logits into the decoding loop to refine the policy on the fly. Our core component, Differentiable Textual Optimization (DTO), leverages gradient signals from both the LLM’s likelihood and a reward model to refine textual representations. $\nabla$-Reasoner further incorporates rejection sampling and acceleration design to robustify and speed up decoding. Theoretically, we show that performing inference-time gradient descent in the sample space to maximize reward is dual to aligning an LLM policy via KL-regularized reinforcement learning. Empirically, $\nabla$-Reasoner achieves over 20% accuracy improvement on a challenging mathematical reasoning benchmark, while reducing number of model calls by approximately 10-40% compared to strong baselines. Overall, our work introduces a paradigm shift from zeroth-order search to first-order optimization at test time, offering a cost-effective path to amplify LLM reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces $\nabla$-Reasoner, a test-time differentiable reasoning framework for large language models (LLMs). It proposed Differentiable Textual Optimization (DTO) that performs gradient descent on logits. The paper provides some empirical experiments and theoretical analysis.

### Strengths
The idea sounds interesting. It is interesting to perform training on logits during test time since it has some challenges. The authors also proposed methods to reduce the heavy computational overhead, showing their awareness of practical utility.

### Weaknesses
My major concern is that **the experiment setting is considerably vague, incomplete and flawed** in many aspects.

1. **Lack of efficiency analysis**. This is a strong concern I have on this paper. Although the authors proposed methods to reduce the heavy overhead, the detailed analysis and experiments are missing. In Sec 5.1, “Efficiency” is measured solely by model calls, ignoring **wall-clock latency, FLOPs, and GPU memory**, which are the core components of efficiency. Intuitively it is very likely that the overhead would be significant since getting gradients would require at least 2x GPU memory, and the latency is likely substantial since it requires performing gradient descent back of forth. In addition, it is important to show whether the efficiency is strongly dependent on the choice of policy models and reward models, and the output sequence length. Lack of such analysis clearly undermines the soundness of the paper.

2. **Unfair comparison with test-time methods**. It is unfair to compare your methods with vanilla SC and BoN using both 8 rollouts. With tools like vLLM library, vanilla SC and BoN can be done flashly in one go, which is much more efficient than your proposed framework if the number of rollouts are the same. I suggest the authors align the budgets (wall-clock latency) with these methods to ensure fairness.

3. **Training-based baselines are not convincing**. In practice, we often apply SFT then RL, not to perform them separately. It's unfair to compare SFT and GRPO separately since they should be incorporated as a whole framework. The paper also did not demonstrate how the budget of the training-based baselines aligns with $\nabla$-reasoner.

4. **No comparison with other test-time training / inference-time adaptation baselines**. This is a serious issue since there are already so many test-time training methods, including but not limited to [1][2]. **None of these are compared with $\nabla$-reasoner.**

5. **Limited domain coverage**. All datasets are about math reasoning. It is unclear how the method performs on other domains like coding, general reasoning, open-ended QA, helpfulness and safety.

6. **Missing analysis of dependence on reward model quality**. The proposed method’s effectiveness appears highly sensitive to the quality and compatibility of the reward model (RM) used for gradient guidance. Appendix D shows that different RMs are applied for different policy LLMs. Why do you do that? Can your framework still work well when the quality of RM is not that good or the output of policy model is OOD for the RM? I suggest incorporating detailed experiments on the effect of applying different RMs across different tasks to show that whether your method is sensitive with the choice of RM or not.

7. **Missing details for reproducibility**. The paper incorporates many experimental details in Appendix D, which is good. However some details are still unclear, including hardware (which is crucial for efficiency analysis) and evaluation methods (rule-based evaluation or strict string matching?). The authors also did not provide their code.


In addition to the experiment settings, I also have concern on the theoretical analysis. Theorem 4.1 is not sound enough since there is a **clear gap between the theorem and actual methodology**. The Wasserstein gradient-flow theorem assumes continuous time, continuous space, infinitesimal steps, and Gaussian diffusion noise. However, actual DTO uses finite discrete steps and deterministic updates. Hence, the theory is **intuitive but not rigorous**, that could act as a motivational study but not a rigorous proof of the equivalence.

[1] Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback  (ICML 2025)

[2] Learning to Reason from Feedback at Test-Time  (ACL 2025 main)

### Questions
1. What do you mean in "it reduces costs by up to 40.2%" in Sec 5.1's cost analysis? How did you get the figure? Which baseline are you comparing with? Is it about wall-clock latency or FLOPs or something else? This claim is not supported by any concrete experiments in the paper.

2. As mentioned above, why do you use different RMs for different policy LMs? Would the performance degrade when the quality of RM is poor or the output of policy model is OOD for the RM?

3. Is the overhead significant especially when RMs are large or output sequence is long?

4. Can your theory be more aligned with the actual methodology? 

5. Could you provide some case studies, that is, concrete examples on what your framework is doing on some certain questions and models?

### Soundness
1

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
3

### Summary
This paper investigates expanding computation during inference. Existing methods tend to saturate in long-chain reasoning due to the influence of sparse and noisy rewards. The authors propose performing gradient descent on token logits in the sample space during decoding. Theoretically, DTO supports bidirectional gradient propagation along the sequence and establishes an equivalence to reinforcement learning–based training. Empirically, it outperforms baselines across multiple benchmarks, achieving comparable or superior reasoning accuracy at a lower computational cost.

### Strengths
1.The method demonstrates outstanding performance across multiple benchmarks. 
2.The authors made practical considerations, enabling the proposed method to integrate well with existing LLM inference acceleration infrastructures.
3.The authors derived gradients over discrete text and attempted to provide theoretical guarantees for DTO

### Weaknesses
1. The authors introduced a hyperparameter in the objective function but did not conduct experiments to analyze its impact. 
2.The proposed method heavily relies on the performance of the reward model, yet only one reward model was used in the experiments. We hope to see results under multiple reward models.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focues on test-time scaling to improve LLM's performance for reasoning task. Commonly, we can use a reward model to choose the best sampling from a pool of candidates (BoN). Such a method can be considered as zeroth-order search. In this paper, the authors propose a new iterative generation framework, ∇-Reasoner, that applies differentiable optimization over token logits into the decoding loop to redfine the policy with the signal from reward model. The Differentiable Textual Optimization (DTO) can be considered as a first-order optimization, used at test time. It utilizes the gradient signals from the LLM's likelyhood and reward model to refine the textual representations. 

The proposed method is theoretically justified, and empirically shows significantly better accuracy and efficiency than various strong baselines.

### Strengths
1. The paper is well-written, with a clear motivation. 
2. The proposed method, DTO, is interesting and novel, offering a new way based on gradient optimization to test-time scaling. And DTO is supported with theoretical justification.
3. Extensive experiments show a significant accuracy and efficiency improvement.

### Weaknesses
1. Lack of reward models. In Appendix D, I notice you apply different reward models for differetn policy models. May I ask why? Could you offer an ablation study for the choice of different reward models for the same policy model.
2. The improvement statement in the abstract is overstated.  The 20% accuracy improvement and 40% less computation are for different baselines, incuring some confusion.

### Questions
See weakness

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
3

### Summary
This paper introduces ∇-Reasoner, an inference-time reasoning framework that applies gradient-based optimization to refine LLM outputs during decoding. The core innovation is Differentiable Textual Optimization (DTO), which optimizes token logits via gradient descent using signals from both the LLM's likelihood and a reward model. The approach combines an objective that balances reward maximization with log-likelihood regularization (Eq. 2), employs the straight-through estimator for differentiability, and integrates rejection sampling to accept only improved tokens. To enhance efficiency, the authors introduce gradient caching, rollout trajectory reusing, and confidence/gradient-guided token selection. Theoretically, they establish that sampling from an RL-optimized LLM is equivalent to drawing from the reference LLM and refining via DTO's gradient flow (Theorem 4.1). Evaluated on mathematical reasoning benchmarks (MATH-500, AMC, AIME), ∇-Reasoner achieves over 20% accuracy improvement while reducing computation by approximately 40% compared to Best-of-N and Self-Consistency baselines, demonstrating superior test-time scaling efficiency.

### Strengths
Originality:
- The shift from zeroth-order search to first-order gradient-based optimization for test-time reasoning is conceptually appealing and well-motivated by Figure 1
- The theoretical connection between DTO and PPO via Wasserstein gradient flow (Theorem 4.1) provides an elegant unification of parametric and non-parametric inference perspectives
- The gradient decomposition (δ_prefix, δ_postfix, δ_reward) offers clear intuition about how DTO enables bidirectional information flow along sequences

Quality:
- The experimental evaluation spans multiple model families (Qwen-2.5, Llama-3.1) and benchmarks with consistent improvements
- The ablation study in Table 2 effectively demonstrates DTO's impact on rejection rates (reducing from 66% to ~30-40%)
- The cost-accuracy trade-off analysis (Figure 3, Figure 4) convincingly shows efficiency gains over sampling-based baselines
- The comparison includes both test-time methods (SC, BoN, ToT, RAP) and training-based methods (SFT, GRPO), providing comprehensive context

Clarity:
- The motivation is clearly articulated—gradient information provides richer directional guidance than scalar rewards
- Figure 2 and Algorithms 1-2 effectively communicate the overall framework
- The progression from basic formulation to acceleration techniques is logical and well-structured
- The gradient decomposition in Section 4 provides valuable interpretability

Significance:
- Addresses the fundamental limitation of sparse reward signals in test-time scaling
- The 40% cost reduction while maintaining or improving performance has practical value
- The theoretical framework connecting test-time and training-time optimization could inspire future work
- The method appears general enough to work with different reward models and base LLMs

### Weaknesses
Experimental Analysis:
- While the cost comparison uses "number of calls," actual wall-clock time could differ due to backward passes. Could you provide runtime measurements to complement the theoretical cost analysis?
- The comparison with RAP and ToT might not be entirely fair if those methods weren't given comparable computational budgets. Could you ensure all baselines use similar total compute?
- Table 1 shows ∇-Reasoner sometimes underperforms training-based GRPO (e.g., Qwen-2.5-7B on AMC: 51.5% vs 52.8%). This suggests potential limitations—could you characterize when training-time methods remain superior?

Implementation Details:
- The method requires modifying the decoding loop and maintaining gradients through generation, which could complicate integration into existing serving infrastructure. Have you explored compatibility with standard inference optimization (e.g., KV caching, speculative decoding)?
- The acceleration techniques are crucial for practicality but add complexity. It would be valuable to quantify each technique's individual contribution to overall speedup
- For the Llama-3.1 experiments, you note inability to evaluate on AIME due to model incapability. This suggests potential brittleness—how does the method degrade with weaker base models?

Comparison Fairness:
- The SFT baseline uses only 10K examples while GRPO uses 35K—this asymmetry makes it difficult to assess whether ∇-Reasoner truly matches training-based methods or is being compared against undertrained baselines
- For the "comparable performance with training-based methods" claim, some results show ∇-Reasoner trailing GRPO (e.g., AIME25 on Qwen-2.5-7B: 15.0% vs 16.7%)

### Questions
- Gradient quality with straight-through estimator: The straight-through trick provides biased gradients. Have you investigated whether this bias significantly impacts convergence, or explored alternative differentiable relaxations (e.g., Gumbel-softmax)?
- Reward model dependency: How does performance degrade with reward models of varying quality? Have you experimented with different reward model architectures or training strategies? Could process rewards provide better gradients than outcome rewards?
- Optimization landscape: Can you characterize the reward landscape's properties (smoothness, multimodality, local optima)? Does DTO sometimes get stuck in local optima, and if so, how might this be addressed?

### Soundness
3

### Presentation
3

### Contribution
3
