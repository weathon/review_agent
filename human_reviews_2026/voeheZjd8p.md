# Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Reinforcement learning (RL) finetuning has become a key technique for enhancing the reasoning abilities of large language models (LLMs). However, its effectiveness critically depends on the selection of training data. Recent advances underscore the importance of online prompt selection methods, which typically concentrate training on partially solved or moderately challenging examples under the current policy, thereby yielding more effective model updates. While significantly accelerating RL finetuning in terms of training steps, they also incur substantial computational overhead by requiring extensive LLM rollouts over large candidate batches to identify informative samples, an expense that can outweigh the finetuning process itself. To address this challenge, this work proposes Dynamics-Predictive Sampling (DPS), which online predicts and selects informative prompts by inferring their learning dynamics prior to costly rollouts. Specifically, we introduce a new perspective by modeling each prompt's solving progress during RL finetuning as a dynamical system, where the extent of solving is represented as the state and the transition is characterized by a hidden Markov model. Using historical rollout reward signals, we perform online Bayesian inference to estimate evolving state distributions, and the inference outcome provides a predictive prior for efficient prompt selection without rollout-intensive filtering. Empirical results across diverse reasoning tasks, including mathematics, planning, and visual geometry, demonstrate that DPS substantially reduces redundant rollouts, accelerates the training process, and achieves superior reasoning performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an online prompt selection method to improve the efficiency and the effectiveness of RLVR. The main motivation is to keep the informative prompts for training (those that yield both correct and wrong responses for the current policy). Instead of following previous work such as DAPO which may need sampling a big number of rollouts to obtain responses that have different rewards, this paper proposes a method named DPS to perform online Bayesian inference to estimate the state distributions. Three states represent, for a single prompt: all questions are incorrect, all are correct, and the rest. The prompts with the highest probabilities in the "partially solved" state will be selected for training.
For evaluation, the authors compared with several selection strategies, such as uniform sampling, history resampling (discard all prompts once the responses are all correct), and the "oracle" DS used in algorithms such as DAPO. The results on several reasoning benchmarks show the effectiveness and efficiency.

### Strengths
+ Compared to DS, this method achieves comparable performance yet with significantly fewer rollouts.
+ This paper addresses a key problem in RLVR about how to effectively select data.

### Weaknesses
+ In each training step, the state belief for each prompt in D is updated, which makes it less efficient than baseline sampling methods. This may be a problem when data scales, although the training data used in this work is usually relatively small-scale.
+ The authors did not show the method's sensitivity to the number of rollouts per prompt. It is unclear whether this method is still stable with a smaller number of rollouts.
+ The baselines are relatively naive and rule-based. More advanced methods, such as entropy/diversity/gradient-based selection, should be considered.
+ The training data and evaluation benchmarks are primarily math-related. The authors may consider general-domain tasks for evaluation.

### Questions
other suggestions: 

+ The models are Qwen-series. The authors are suggested to test the performance on a model in other model families.
+ The authors may need to clarify that "oracle" actually refers to finding all prompts that have both successful and failed responses, instead of a method that can achieve the best performance by training on sampled prompts.

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
2

### Summary
This paper proposes Dynamics-Predictive Sampling (DPS), a novel method to reduce the high computational cost of online prompt selection during the reinforcement learning (RL) finetuning of large reasoning models. The core problem is that existing methods, like Dynamic Sampling (DS), must perform extensive and costly LLM rollouts on large candidate batches to identify informative, "partially solved" training examples. DPS avoids this by modeling each prompt's solving progress as a dynamical system, specifically a hidden Markov model (HMM). This HMM tracks the prompt's latent "solving state" $z_{t}^{\tau}$ (e.g., fully unsolved, partially solved, or fully solved). Using historical reward signals, DPS performs online Bayesian inference to efficiently compute a predictive prior $\mu_{t}^{\tau,prior}(2)$, estimating the probability that a prompt is in the most informative "partially solved" state before any rollouts are generated. The main contributions are this HMM-based predictive framework and empirical results showing DPS substantially reduces redundant rollouts (e.g., achieving strong performance with less than 30% of DS's rollout budget), accelerates training, and achieves comparable or superior reasoning performance to costly oracle methods across diverse mathematics, planning, and visual geometry tasks.

### Strengths
The paper's primary strength is its direct and effective focus on improving the computational efficiency of active prompt selection for RL finetuning. The work is presented with good clarity, first identifying the high cost of existing online sampling methods (like DS) that rely on extensive rollouts, and then proposing a clear, lightweight alternative. In terms of originality, the paper offers a practical methodological refinement; rather than introducing a new sampling goal, it iteratively improves the mechanism by using an HMM to predict a prompt's solving state, thereby avoiding the need for costly pre-evaluation. The quality of this contribution is substantiated by a solid empirical evaluation across several reasoning tasks. The results consistently show that the proposed DPS method achieves performance comparable to the compute-intensive DS baseline while operating with a significantly reduced rollout budget, as detailed in the tables and efficiency graphs. The work's significance is, therefore, practical: it provides a viable method to achieve the benefits of dynamic data curation without the prohibitive computational overhead, making active sampling a more feasible option for practitioners.

### Weaknesses
* The necessity of the HMM framework is not fully justified, as the paper omits comparisons to simpler predictive baselines. A non-probabilistic heuristic, such as tracking an exponential moving average of reward variance for each prompt, might also identify "partially solved" items with similar efficiency gains and less modeling overhead.
* The claim of "negligible" computational overhead is only validated on relatively small datasets (e.g., 7.5k for MATH). The method's costs scale linearly with the total dataset size |D|, as it maintains and updates a separate HMM for every prompt, which could become a practical bottleneck for datasets with millions of examples.
* The Top-B selection strategy is purely exploitative, as it only samples prompts with the highest predicted probability of being in State 2. This greedy approach ignores sampling based on uncertainty (e.g., high entropy across the HMM state belief), which could be crucial for efficiently detecting state transitions and improving the model's robustness.

### Questions
Could you provide a justification for choosing an HMM framework over simpler, non-probabilistic heuristics? For instance, did you experiment with tracking a simple exponential moving average of reward variance for each prompt to identify "partially solved" items, and how would its performance and efficiency compare to DPS?

The Top-B selection strategy is greedy, exploiting prompts known to be in State 2. Did you consider alternative selection strategies that incorporate exploration, such as prioritizing prompts with high entropy in their predicted state distribution (i.e., high uncertainty), and how might that impact learning stability and final performance?

The definition of the solving states (unsolved, partial, solved) seems highly dependent on the number of samples, k=8. How does the model's performance and the HMM's state estimation change with different values of k (e.g., k=4 or k=16)?

You briefly mention that the framework could extend to dense or process-based rewards. Could you elaborate on how you would concretely map a continuous reward signal onto the discrete 3-state system? For example, would you use fixed thresholds, and how would you prevent this from re-introducing the observation-sparsity issues you noted in the ablation study on finer-grained partitions?

### Soundness
3

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
This paper introduces Dynamics-Predictive Sampling (DPS), a method that predicts the informativeness of training samples by modeling their learning progression as a Hidden Markov Model (HMM), thereby replacing costly LLM rollouts for sample selection. The method's primary strength is its computational efficiency, shifting the overhead from LLM inference to lightweight matrix operations. This claim is well-supported by comprehensive experiments across diverse reasoning tasks.
However, its limitations warrant attention. First, its three-state model is tailored for binary rewards, and its generality for tasks with denser reward signals (e.g., from process supervision) is unclear. Second, by maintaining independent dynamics for each prompt, the method may suffer from data sparsity issues, leading to unreliable estimates for infrequently sampled prompts and thus impacting its robustness.

### Strengths
- The core contribution lies in shifting the sample selection overhead from LLM inference costs, which are proportional to batch size, to computationally negligible matrix operations. Experiments convincingly show that DPS can match or exceed the performance of baselines with substantially lower computational resources, a critical factor for the practical adoption of RL fine-tuning.
- The study features a comprehensive experimental design across diverse and complex reasoning tasks (mathematics, planning, visual geometry), with thorough comparisons against key baselines (e.g., Uniform Sampling, Dynamic Sampling). The inclusion of extensive ablation studies enhances the credibility of the methodological design and provides good support for the core arguments.

### Weaknesses
- The proposed three-state model (unsolved, partially solved, solved) is highly effective for binary reward settings. It would be beneficial to discuss the framework's extensibility to tasks with denser reward signals, such as continuous scores from process supervision. Addressing questions like how the state space could be partitioned (e.g., fixed vs. dynamic boundaries) would provide valuable insight into the method's potential applicability to a wider range of problems.
- The model maintains independent transition dynamics for each prompt. This raises a concern about data sparsity: for infrequently sampled prompts, the estimated dynamics may be unreliable, which could compromise the accuracy of state predictions. It would strengthen the paper to explain how this potential issue is handled. Additionally, an analysis of what happens to persistently under-sampled, difficult problems—and whether their learning dynamics risk stagnation—would be a valuable addition.

### Questions
- The state observation depends on generating k responses per prompt, creating a trade-off between cost and observation noise. The paper uses k=8 but does not include a sensitivity analysis for this hyperparameter. An analysis or discussion on the impact of varying k—and how effectively the HMM can mitigate noise from smaller k values—would provide a clearer picture of the method's robustness.

### Soundness
2

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
2

### Summary
The paper introduces Dynamics-Predictive Sampling (DPS), an online prompt-selection framework that enhances reinforcement-learning (RL) finetuning of large reasoning models (LRMs). The core motivation is that dynamic sampling strategies (e.g., Dynamic Sampling (DS), History Resampling (HR)) improve RL finetuning efficiency by prioritizing informative prompts—typically those that are only partially solved—but at the cost of significant rollout overhead.
DPS aims to predict which prompts are likely to be informative before performing expensive LLM rollouts. It models each prompt’s solving progress as a Hidden Markov Model (HMM) whose latent states correspond to unsolved, partially solved, and fully solved stages. Using Bayesian updates of transition and emission probabilities, DPS infers prompt-specific state distributions online and selects prompts with the highest predicted probability of being partially solved.

The proposed method is lightweight and scalable, requiring only low-dimensional updates per prompt. Experiments on mathematical reasoning (MATH, AIME24, AMC23), numerical planning (Countdown), and visual geometry (Geometry3K) demonstrate that DPS.

### Strengths
The paper offers a new dynamical-systems perspective on data selection in RL finetuning, bridging prompt-level reward evolution with state-space modeling.

DPS eliminates the need for redundant LLM rollouts by predicting informative prompts, yielding 2–3× runtime reduction and 70 % rollout savings while maintaining accuracy parity with DAPO.

Comprehensive experiments across three reasoning domains and multiple model scales. Evaluation includes accuracy curves, confusion-matrix analyses of prediction reliability, and ablation studies (non-stationary decay λ, number of states, prior α₀) demonstrating robustness and interpretability.

### Weaknesses
Reward formulation dependency:

The approach currently depends on binary correctness rewards, which are straightforward in math-style benchmarks but not directly transferable to open-ended or process-based domains (e.g., code synthesis with partial correctness). The authors acknowledge this and suggest extending DPS to dense or step-wise rewards.

Simplistic selection criterion:

DPS uses top-B selection on predicted State 2 probabilities. More nuanced criteria (e.g., entropy-based uncertainty, expected information gain) could further enhance exploration-exploitation balance.

Scalability in very large datasets:

While the complexity analysis asserts negligible overhead, maintaining per-prompt HMM parameters might become costly when |D| ≫ 10⁶ unless sampling approximations are used.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2
