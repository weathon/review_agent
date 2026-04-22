# SCAR: Shapley Credit Assignment for More Efficient RLHF

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Reinforcement Learning from Human Feedback (RLHF) is a widely used technique for aligning Large Language Models (LLMs) with human preferences, yet it often suffers from sparse reward signals, making effective credit assignment challenging. In typical setups, the reward model provides a single scalar score for an entire generated sequence, offering little insight into which token or span-level decisions were responsible for the outcome. To address this, we propose Shapley Credit Assignment Rewards (SCAR), a novel method that leverages Shapley values in cooperative game theory. SCAR distributes the total sequence-level reward among constituent tokens or text spans based on their principled marginal contributions. This creates dense reward signals, crucially, without necessitating the training of auxiliary critique models or recourse to fine-grained human annotations at intermediate generation stages. Unlike prior dense reward methods, SCAR offers a game-theoretic foundation for fair credit attribution. Theoretically, we demonstrate that SCAR preserves the original optimal policy, and empirically, across diverse tasks including sentiment control, text summarization, and instruction tuning, we show that SCAR converges significantly faster and achieves higher final reward scores compared to standard RLHF and attention-based dense reward baselines. Our findings suggest that SCAR provides a more effective and theoretically sound method for credit assignment in RLHF, leading to more efficient alignment of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SCAR (Shapley Credit Assignment Rewards), which applies Shapley values from cooperative game theory to address the sparse reward problem in Reinforcement Learning from Human Feedback (RLHF). The authors model text generation as a cooperative game where text segments (tokens, spans, or sentences) are players, and use Shapley values to distribute the terminal reward across generation steps. The method is evaluated on three tasks: sentiment control, text summarization, and instruction tuning, showing faster convergence and higher rewards compared to sparse RLHF and attention-based baselines.

### Strengths
The paper addresses an important problem in RLHF and provides a theoretically motivated approach based on cooperative game theory. The experimental design covers multiple tasks with different response lengths, demonstrating the method's applicability across various text generation scenarios. The writing is clear and the paper acknowledges some limitations in the appendix. The use of adaptive segmentation (token/span/sentence level) shows practical awareness of computational constraints.

### Weaknesses
The fundamental weakness lies in the mismatch between the cooperative game framework and autoregressive generation. Tokens are not independent agents that can form arbitrary coalitions; they are outputs of a sequential decision process with strong temporal dependencies. The characteristic function v(S) defined by space-filling is not theoretically justified and may not satisfy basic properties of cooperative games like superadditivity or monotonicity. The paper does not verify whether reward models can actually provide meaningful scores for such artificially constructed partial sequences.
The experimental choices are problematic for a 2025 submission. Using models from 2019-2020 (GPT-2, OpenLLaMA) instead of modern architectures (Llama 3, Qwen 2.5, Gemma 2) limits the relevance of findings. The absence of code generation and mathematical reasoning tasks is particularly concerning, since these domains represent critical RLHF applications where objective rewards (test pass rates, answer correctness) make credit assignment evaluation more reliable. The authors' explanation that "rule-based models" are unsuitable actually exposes a fundamental limitation: the method fails when tokens have strong causal dependencies, which is precisely where credit assignment is most needed.
The evaluation lacks rigor. Statistical significance is not tested despite marginal improvements. Win rates of 54.9% in LLM-as-judge evaluation are barely above random. Table 1 reports implausible standard deviations (0.00 for some methods), suggesting incomplete experimental runs.

### Questions
Could the authors provide evidence that reward models can meaningfully score partial sequences constructed with space-filling? Specifically, what is the distribution of scores for such sequences compared to naturally generated text, and how does this affect the validity of v(S) as a characteristic function?
Why were code generation and mathematical reasoning tasks excluded from evaluation? These tasks have objective rewards and strong causal dependencies, making them ideal test cases for credit assignment methods. Can the authors demonstrate SCAR's effectiveness on these tasks, or explicitly characterize the task properties where the method is expected to work versus fail?
The improvements over ABC are modest, and it's unclear whether they stem from Shapley values' fairness properties or simply from being a different dense reward scheme. Could the authors provide ablation studies comparing Shapley-based allocation against other principled dense reward distributions (e.g., based on gradient norms, attention entropy, or learned value functions)?
How does the method perform with modern language models (Llama 3, Qwen 2.5)? The use of GPT-2 and OpenLLaMA significantly limits the practical relevance of the findings in 2025.
Given that Owen value approximation is used for tractability, does Theorem 3.1's optimality guarantee still hold? What is the approximation error, and how does it affect the final policy quality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses credit assignment in RLHF for aligning LLMs. It proposes SCAR, a method that uses Shapley-value-based attribution to assign token- or span-level contributions to a holistic reward. Experiments on IMDb TL;DR summarization and the Anthropic Helpful–Harmless (HH) dataset suggest effectiveness relative to baselines.

### Strengths
1. Introduces a principled, game-theoretic approach to reward attribution that is potentially more interpretable than heuristic token credit schemes.

2. Provides token/span-level granularity, which could help better and fast convergence.

3. Presents a clear, modular framework that can, be combined with standard RLHF pipelines.

### Weaknesses
1.	While Shapley values do not require independence per se, they assume a well-defined utility for any coalition of “players.” In LLMs, tokens are highly dependent and causal; the contribution of a token is context-sensitive and may change as later tokens are generated. 
The paper should discuss how SCAR accounts for this sequential dependence beyond treating tokens as interchangeable players.

2.	Exact Shapley computation requires evaluating the utility for arbitrary coalitions. Using the reward model as the utility function directly may lead to out-of-distribution inputs the reward model was never trained to score. This raises questions about calibration, faithfulness, and bias in marginal contribution estimates.

3.	The adoption of the Owen approximation (grouped Shapley) is not justified with a principled grouping scheme. The paper lacks theoretical or empirical guidance on how to partition sequences into groups, how sensitive results are to the grouping strategy, and whether groups align with linguistic or structural units.

4.	Experiments are limited to GPT-2 and relatively simple scenarios. The absence of larger models, diverse tasks, and stronger baselines makes it difficult to assess generality and practical impact.

5.	Shapley-based methods can be expensive. The paper should quantify sampling budgets, variance, convergence, and training overhead relative to standard RLHF.

### Questions
1. How do you validate that the reward model is a faithful utility proxy for arbitrary coalitions of tokens or spans? Are there calibration studies or sanity checks?

2. What is the grouping rationale for the Owen approximation? How are groups defined (e.g., sentences, clauses, semantic units), and how sensitive are results to different grouping heuristics?

3. What is the computational overhead versus standard RLHF? How does this scale with sequence length?

4. How robust are attributions across different base models and domains? Have you tested SCAR on larger model families (e.g., Qwen, LLaMA), code or math tasks?

5. Do SCAR attributions reduce known reward hacking patterns?

6. How stable are token/span attributions across random seeds, prompt variations, and paraphrases? Is there high variance that could undermine their reliability for training?

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
This paper proposes SCAR, a method based on the Shapley value in cooperative game theory, to address the reward sparsity problem in RLHF. SCAR fairly allocates sequence-level rewards to tokens or text segments (such as phrases and sentences) in the generated text, thereby providing dense reward signals without the need for additional training of a critique model or fine-grained human annotations. The main contributions of the paper are as follows:
Proposing a Shapley value-based credit assignment method with theoretical guarantees (e.g., efficiency, symmetry, linearity);
Reducing computational complexity through adaptive text segmentation (token-level, phrase-level, sentence-level) and Owen value approximation;
Theoretically proving that SCAR preserves the optimal policy (based on Potential-Based Reward Shaping);
Conducting empirical validation on three tasks—sentiment control, text summarization, and instruction tuning—showing that SCAR achieves faster convergence and higher final rewards compared to the standard RLHF, Uniform, and Attention-Based Credit (ABC) baselines.

### Strengths
Originality: Introducing the Shapley value into RLHF credit assignment represents an innovative integration of cooperative game theory and RLHF, distinct from heuristic methods that rely on attention weights (e.g., ABC).
Quality: The method is rigorously designed, encompassing theoretical analysis (policy invariance), efficient approximations (Owen value, adaptive segmentation), and comprehensive experiments (three tasks, multiple baselines, and statistical validation).
Clarity: The paper is well-structured with detailed descriptions of the method; figures and tables effectively aid understanding, and the appendix provides hyperparameters, proofs, and additional experiments.
Significance: It addresses the core issue in RLHF—credit assignment—can enhance the efficiency and stability of LLM alignment, and holds broad application potential.

### Weaknesses
Computational overhead: Despite reducing complexity through the use of Owen values and adaptive segmentation, Shapley value approximation still introduces significant computational costs (e.g., Token-level SCAR requires 48 GPU hours for the summarization task, compared to 7 hours for Span-level). It may remain impractical for extremely long sequences or large-scale LLMs.
Reward model assumptions: SCAR assumes that the reward model can provide meaningful scoring for partial sequences (Eq. 3). However, some reward models (such as rule-based models or those that only evaluate final answers) may not be applicable, limiting its generalizability.
Limitations in experimental scale: The experiments are based on smaller models (e.g., GPT-2 small, Pythia-1B) and have not been validated on larger LLMs (e.g., LLaMA-2 or GPT-3 scale), which may affect the generalizability of the conclusions.
Hyperparameter sensitivity: The selection of α (Shapley reward coefficient) may affect performance, but the paper does not provide a systematic sensitivity analysis or guidelines for its selection.

### Questions
Computational efficiency: What is the computational cost of SCAR for tasks involving generating long documents? Are there plans to develop more efficient approximation algorithms ?
Generalizability of the reward model: How would SCAR be adjusted if the reward model cannot reliably evaluate partial sequences? Has consideration been given to using generative reward models or multi-step evaluation?
Scalability: Are there plans to validate SCAR on larger-scale LLMs or more diverse tasks?
Hyperparameter optimization: Is the selection of α and segmentation strategies (token/phrase/sentence level) task-dependent? Can guidelines for the automatic selection of these hyperparameters be provided?
Comparison with contemporaneous work: What are the relative advantages of SCAR in terms of efficiency and effectiveness compared to the method by Koo et al. (2025)? Has a direct comparison been conducted?

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
SCAR proposes to use Shapely Values to get a dense token/span-level reward for RLHF. The properties of Shapely values enable SCAR to preserve the original optimal policy and facilitate faster convergence. The method further uses Owen value approximation to make the computation tractable. The evaluations are carried out on Sentiment Control, Summarization and Instruction tuning tasks

### Strengths
1.	The use of Shapley values to redistribute sparse rewards into dense rewards has a strong theoretical foundation. Further, the dense learning signal does not change the optimal policy.
2.	Adaptive segmentation and Owen value approximations make the approach tractable and suited for real-world use.
3.	Experiments show that SCAR achieves faster convergence and preference compared to the baselines.

### Weaknesses
1.	The proposed method focuses on a novel way to convert sparse reward into dense reward(s) and the experiments demonstrate the efficacy of dense rewards in PPO training. However, there are other RL methods, e.g. OREO (Wang et al., 2025)  and DQO (Liu et al., 2024), that explicitly formulate intermediate steps as an MDP and employ soft Q-learning to learn the policy. These methods can also use process level supervision.  A major concern with the experiment section is that they focus only on improving the learnability of PPO. Maybe they should also demonstrate that other methods such as DQO can also be improved using the proposed sparse to dense reward conversion.

             a. Wang, Chaojie, et al. "Q*: Improving multi-step reasoning for llms with deliberative planning", 2024
             b. Liu et al. et al. "Enhancing multi-step reasoning abilities of language models through direct q-function optimization." 2024.


2.	The comparisons with ABC is appropriate; however, it would strengthen the paper to include evaluations against more recent dense-reward methods, such as critic-based auxiliary models, learned dense reward models, and the contemporary works cited. While SCAR has advantages (e.g., no need for dense human annotations or training a separate reward model), comparing its performance with these methods is essential to contextualize its effectiveness. Some potential baselines to consider are the following:

         a. Chan, Alex J., et al. "Dense reward for free in reinforcement learning from human feedback." arXiv preprint arXiv:2402.00782 (2024).
         b. Yoon, Eunseop, et al. "Tlcr: Token-level continuous reward for fine-grained reinforcement learning from human feedback."  ACL Fndings (2024).
         c. Koo, Ryan, et al. "Learning explainable dense reward shapes via bayesian optimization." arXiv preprint arXiv:2504.16272 (2025).


3.	The reward model (RM) score alone provides limited insight into the model’s actual improvement. Including a human evaluation on at least a subset of the test data would have helped assess how well the reward model’s judgments align with human preferences.


4.	A qualitative comparison of model outputs would be valuable to illustrate the specific kinds of improvements or additional learning that the dense reward model provides over the baseline methods.

### Questions
Figure 4 indicates that for the summarization task, token-level dense rewards underperform compared to span-level dense rewards. It would be useful for the authors to discuss why this happens. Does it suggest that an optimal reward granularity exists—somewhere between token-level and sequence-level—to balance precision and contextual coherence? An ablation study varying the span sizes used for dense rewards could clarify this relationship and reveal how reward granularity affects learning stability and performance.

### Soundness
3

### Presentation
3

### Contribution
2
