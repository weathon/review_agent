# Preference-Based Process Reward Model for Robust Mathematical Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Process reward models (PRMs) have emerged as a promising approach to guide LLMs by providing step-wise supervision, but traditional methods often rely on heuristic search strategies like Monte Carlo Tree Search (MCTS), which introduce bias and limit generalization. In this work, we propose a reinforcement learning framework guided by a   Preference-Based Process Reward Model (PPRM) , which provides step-wise supervision to refine reasoning trajectories. We first employ MCTS to estimate and select chosen and rejected rollouts, thereby constructing a high-quality step-level dataset. Our PPRM is trained on Bradley-Terry loss function, which mitigates the bias introduced by the heuristic search strategies of MCTS by leveraging preference-based learning. To enable effective RL training with PPRM, we enhance  Group Relative Policy Optimization (GRPO)  by introducing a robust advantage estimator that better captures the structure of preference-based process reward model enabling stable and efficient policy optimization. Experimental results on ProcessBench and best-of-n strategy  demonstrate that our approach achieves  $2$-$3\%$ improvement in intermediate step accuracy compared to existing methods for complex reasoning processes, thereby improving the reasoning accuracy of the policy model across several key reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a heuristic to select preference pairs to train PRM. Meanwhile, it also modifies the advantage function of original GRPO to adapt to process reward settings.

### Strengths
- The paper theoretically shows that the preference-based PRM can achieve higher expected accuracy than the PRM trained by hard labels estimated by MCTS.

- They modify the GRPO advantage estimator to adapt to the preference-based RM.

### Weaknesses
> Paradoxical motivation

The paper argues the MCTS estimation can lead to inconsistent or suboptimal results, however, they still use MCTS with a heuristic metric (Eq.1) to annotate preference.



> Comparison Fairness & Comprehensiveness 

- Empirical aspects: **Models are trained by different datasets**, so the comparsion seems unfair. If trained by the same dataset, would the preference-based formulation still outperform baselines?

- Theoretical aspect: There are many more advanced theoretical framework trying to bypass the hard label estimation of PRM [1][2][3][4], however, the theoretical part only compare with the classical PRMs.

[1] Yuan, Lifan, et al. "Free Process Rewards without Process Labels." Forty-second International Conference on Machine Learning.

[2] Lu, Jianqiao, et al. "Autopsv: Automated process-supervised verifier." Advances in Neural Information Processing Systems 2024.

[3] Zhang, Zheng, et al. "Linking Process to Outcome: Conditional Reward Modeling for LLM Reasoning." arXiv preprint arXiv:2509.2657.

[4] Li, Wendi, and Yixuan Li. "Process Reward Model with Q-value Rankings." The Thirteenth International Conference on Learning Representations.

### Questions
> PRMs are also evaluated in beam-search tasks or RL tasks. Can PPRM also yield superior performance in these tasks?

> The current experiments are only conducted on a single model. Can PPRM perform consistently across different backbones?

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
The paper presents a novel reinforcement learning framework guided by a Preference-Based Process Reward Model (PPRM). The method first uses MCTS to select chosen and rejected rollouts. Then, Bradley-Terry loss function is used to mitigate bias in MC-value estimation by leveraging pairwise comparisons of reasoning trajectories. The method is trained using GRPO with an optimized advantage estimator to better captures the structure of preference-based process reward model. Experimental results show that the proposed PPRM improves performance on intermediate step accuracy and enhances the final policy model's performance compared to existing works, demonstrating the method's effectiveness.

### Strengths
1. The proposed method is described in sufficient detail and appears technically sound.

2. The experimental results are strong, demonstrating the effectiveness of the proposed method.

### Weaknesses
### Major Concerns:

1. Clarity of Motivation. The writing in the introduction and motivation sections is not sufficiently clear, which hinders the reader's understanding of the precise problem being solved.

- L51-L53: The logical connection between the sentence "Lightman et al. (Lightman et al., 2023) demonstrated the effectiveness of using human expert annotators..." and the following phrase "To address this..." is abrupt and unclear. It is not evident what specific problem or limitation "this" refers to.

- L14, L58-L61: A central motivation of the paper appears to be the mitigation of bias in MCTS. However, this bias is not clearly defined or explained at the beginning of the paper. A more comprehensive introduction to this problem is needed to properly contextualize the paper's contributions.

2. Insufficient Experimental Discussion. While the experimental results are strong, the discussion and analysis are insufficient. The paper does not adequately connect the empirical gains back to the central claims made in the motivation. Specifically, the authors should provide more detailed analysis to demonstrate how the proposed method successfully alleviates the "Limitations of PRM" that were introduced in lines 46-63.

3. The LLM Judger is not formally introduced (L190).

### Minor Issues:

- L42-L44, "While the Process Reward Model (PRM) offers a promising solution by providing step-wise reinforcement learning feedback." is a dependent clause and grammatically incomplete.
- L79, "more rob-ust reasoning"

- L209-L211, there appears to be a missing equation number

### Questions
The core methodology and the reported results are promising, and this work appears to be a valuable contribution. However, the paper's current lack of clarity in the motivation and the insufficient experimental discussion are significant concerns. In order to maintain my rating, I would like the authors to address the major concerns above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduced preference-based process reward model (PPRM). It leverages Bradley-Terry pairwise comparison to reduce bias in process reward modeling. PPRM combines this preference-based formulation with a modified GRPO, to use a preference-aware advantage estimator to stablize training and reduce variance. The experiments are conducted on ProcessBench and RL finetuning tasks and the results show 2-3% accuracy improvement over strong baselines.

### Strengths
1. This paper directly targets MCTS-induced heuristic bias in process supervision and this approach offers a clean conceptual and mathematical reformulation.
2. The proposed preference-based advantage estimator fits into GRPO well and it transforms pairwise rewards into smoother, variance-reduced advantage estimates.
3. Results span multiple reasoning datasets and evaluation setups and demonstrate consistent improvement

### Weaknesses
1. Although PPRM is introduced as de-biasing MCTS, it still relies on MCTS-generated trajectories to form chosen-rejected pairs. If MCTS itself samples biased reasoning paths, the BT formulation merely reweights rather than corrects them. A comparison using non-MCTS rollouts (e.g., temperature-based or random sampling) would clarify true robustness.
2.  The reported gains likely combine effects from both (i) preference training and (ii) the new advantage estimator. A clear ablation isolating these factors — plus sensitivity analysis on α, β, and pair length penalty — is essential to interpret where the real improvements come from.
3.  All benchmarks are math reasoning datasets. Since the claimed contribution is a general reward modeling framework, it’s unclear whether the approach generalizes to symbolic, scientific, or program synthesis reasoning tasks.
4. Both training and evaluation involve MATH and related datasets (GSM8K variants, OlympiadBench). The authors should clarify de-duplication and overlap handling, especially since Qwen2.5-Math models have been partially trained on similar corpora.
5. Some relevant studies are missing [1-2]

[1] Entropy-Regularized Process Reward Model

[2] GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning

### Questions
m/a

### Soundness
2

### Presentation
1

### Contribution
2
