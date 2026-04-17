# BRIM: Block-wise Return Induction Method for Sequence Knowledge Distillation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 4

## Abstract
Reinforcement Learning (RL)-based knowledge distillation (KD) is increasingly used to train language models for text generation. However, existing methods suffer from high variance caused by long action chains during sampling. To address this, we propose a novel block-wise return induction approach (called BRIM) that mitigates the high variance issue and stabilizes the training process. 
Our idea is to apply the Bellman Optimality Equation inversely to each $K$-step block segmented student's explored trajectories, and thus induce a total reward for all blocks from the teacher model, serving as the policy-gradient training signal.
Theoretical analysis shows that our BRIM reduces the variance of the gradient estimates, thus leading to improved RL optimization, especially when the student model size is large. Empirical evaluation on three text generation tasks demonstrates that our approach yields superior performance in both standard task metrics and large language model (LLM)-based evaluation, which suggests that our BRIM offers a promising direction for enhancing RL-based KD in LLM research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets the high-variance problem in RL-based sequence KD. It proposes BRIM, which applies an inverse Bellman optimality expansion over K-step blocks along student rollouts to construct an approximate return used in policy-gradient updates. The paper positions BRIM as a REINFORCE-with-baseline variant whose baseline arises from teacher Q-values, with a variance-reduction argument. In the experiment section, it shows empirical gains across datasets including benchmarks of summarization, translation, and math reasoning.

### Strengths
* The paper proposes a novel approach to reduce variance induced in teacher-student distillation, with clear intuition. 
* The method shows consistent empirical gains on T5 models compared to previous baselines across evaluation benchmarks.

### Weaknesses
* The presentation lacks clarity and could make the algorithm hard to follow. For example, the paper does not mention how Q value function is implemented in the teacher model. Please consider adding more clarifications about this part. 
* All experiments use T5 for both teacher and student, which limits conclusions about model-family generality. Results on decoder-only families (e.g., Llama, Qwen, Mistral) and cross-family teacher-to-student settings are needed, along with a sensitivity study to weaker/miscalibrated teachers.
* The current evaluation tasks (seq2seq tasks + GSM8K) do not include open-ended instruction following or multi-turn settings, where sequence-level RL variance is often most problematic. Evaluations on mainstream instruction-following benchmarks(e.g., MT-Bench [1], AlpacaEval 2.0 [2]) would better validate the method’s breadth.
* The method is motivated as a low-variance alternative to REINFORCE-style sequence RL, yet there are no results for (i) REINFORCE on task reward and (ii) PPO+GAE on task reward under identical budgets. Adding REINFORCE based RL and PPO as baseline could better validate the method's effectiveness 

[1] Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin et al. "Judging llm-as-a-judge with mt-bench and chatbot arena." Advances in neural information processing systems 36 (2023): 46595-46623.

[2] Dubois, Yann, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. "Length-controlled alpacaeval: A simple way to debias automatic evaluators." arXiv preprint arXiv:2404.04475 (2024).

### Questions
* In line 154-160, How's $q(s, a)$ defined for teacher model ? Do you train any separate critic or value function, please clarify for the implementation for the $Q$ value function of teacher model. 
* The current K-step return estimate might be closely related to the classical multi-step/bootstrapped estimators such as n-step return [1], TD($\lambda$) [2], could you elaborate on the difference between your method and existing methods to reduce the variance? 

[1] Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning (A3C). ICML 2016.

[2] Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning 3(1): 9–44.

### Soundness
3

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
4

### Summary
This paper proposes a novel method called BRIM (Block-wise Return Induction Method) to address the high-variance issue in Reinforcement Learning (RL)-based Knowledge Distillation (KD) for text generation, which is caused by long action sequences during sampling. The method segments the student model's generated trajectory into blocks of length K. By applying the inverse Bellman Optimality Equation to each block, it induces a block-wise cumulative reward from the teacher model, which serves as the training signal for policy gradient. Theoretical analysis demonstrates that this approach effectively reduces the variance of gradient estimates. Extensive experiments on three text generation tasks (summarization, machine translation, and arithmetic reasoning) validate its superior performance in both standard task metrics and LLM-based evaluation.

### Strengths
S1: BRIM applies the inverse Bellman equation for block-wise reward induction, effectively mitigating the high variance problem in long-sequence RL training, which is novel and well-grounded in theory.
S2: The paper proposes the method with a theoretical proof that it reduces gradient variance (Theorem 1) and analyzes the bias-variance trade-off, which significantly enhances the credibility of the approach.
S3: The method is systematically evaluated on three text generation tasks from different domains. The evaluation includes not only traditional metrics (e.g., ROUGE, BLEU) but also introduces an LLM-based assessment, verifying the method's generality and effectiveness.
S4: The varianc-bias trade-off trends observed experimentally match theoretical predictions.

### Weaknesses
W1: It seems that the main content of this work is a simplified estimation of the calculation of G_t in [1], so I hope to see more insights from the author about this work and further explanation of possible optimizations.
W2: Theorem 1 relies on the assumption that the (state, action, reward) tuples are independent and identically distributed across timesteps. However, in autoregressive text generation, such tuples are inherently correlated since each token depends on its preceding context. The paper only briefly mentions this issue; therefore, a more thorough discussion is needed to justify when this i.i.d. assumption can be approximately valid. For example, when the student policy closely matches the teacher’s distribution or when large-batch sampling mitigates correlation effects.
W3: The experimental results show that the optimal value of K is inconsistent across different tasks and datasets (e.g., 2, 4, 8, 16). I hope the author could provide a strategy for automatically selecting K, which could increase tuning costs in practical applications.
W4: Estimation bias in the derivation process of Eq 5: In addition to the bias pointed in line 134, the decoding sampling strategy also has an impact, unless greedy decoding is used. Moreover, as K increases, the bias accumulates further. What kind of impact does this produce? From Fig. 1, there does not seem to be a consistent effect.


[1] LLMR: Knowledge Distillation with a Large Language Model-Induced Reward

### Questions
Please refer to weakness.

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
3

### Summary
This paper proposes an improved RL approach called BRIM, which mitigates the high variance issue in knowledge distillation (KD). Specifically, the authors define the sum of rewards over consecutive steps by approximating that an optimal action is taken by a student policy. Based on this, they propose a K-step reward formulation for RL-based generation KD and update the model following the policy gradient formula. Experiments are conducted on various tasks, including XSum Summarization, Europarl EN-NL Translation, and GSM8K. The results demonstrate that BRIM with a larger K leads to a more stable training process and achieves better performance across these tasks.

### Strengths
* The paper is well-organized and clearly presented. 
* The proposed approach is simple yet effective in mitigating the variance issue in RL-based knowledge distillation, and is supported by theoretical analysis.
* BRIM demonstrates consistent improvements over the baseline methods across different tasks presented in Table 1.

### Weaknesses
* The evaluation metrics in Table 1 are primarily based on n-gram matching. Incorporating semantic-centric metrics, such as G-Eval, could further validate the effectiveness of the proposed approach.
* The study lacks human evaluation. Although LLM-as-a-judge was used as a surrogate, LLM judges are prone to various biases and may not accurately reflect genuine human preferences.
* The experiments were conducted exclusively on T5 models with fewer than 3B parameters. Extending the evaluation to other model families and a wider range of teacher-student sizes is necessary to assess the robustness and generalizability of BRIM.

### Questions
1. It would be valuable to include an analysis of BRIM in knowledge distillation scenarios where there is a growing capability gap between the teacher and student models.
2. It is recommended to add experiments with other backbone language models, along with an analysis of their scaling trends.
3. Human evaluation should be incorporated to complement the automated metrics.
4. The addition of semantic-based evaluation metrics and the 95% confidence intervals for the results in Table 1 is suggested. While the authors state that their results are statistically significant compared to each baseline, the improvements on the Europarl dataset appear quite marginal, and some values seem to be bolded incorrectly.

### Soundness
3

### Presentation
3

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
This paper proposes a Block-wise Return Induction Method (BRIM) for reinforcement learning (RL)-based knowledge distillation. BRIM mainly based on LLMR (Li et al. 2024). BRIM further introduces a K-step reward estimation. This paper evaluates BRIM on three benchmarks including XSum (for summarization tasks), Europarl (for translation tasks), and GSM8K (for mathematical reasoning tasks).

### Strengths
- S1. [Idea] This paper aims to advance RL-based knowledge distillation methods, and these approaches seem promising in training small language models.

### Weaknesses
- W1. [Novelty] BRIM mainly based on LLMR (Li et al. 2024). LLMR is a knowledge distillation method based on a reward function induced from large language models. Based on LLMR, BRIM further introduces a K-step reward estimation. However, the extension seems rather limited. Furthermore, it is unclear what the advantages of K-step reward estimation are. 

- W2. [Performance] According to Table 1, BRIM (26.38 on GSM8K) does not seem to have a significant difference in performance from LLMR (25.39 on GSM8K).

- W3. [Evaluation] This paper evaluates BRIM on three benchmarks including XSum, Europarl, and GSM8K. These are rather easy benchmarks in each task. I am not sure that BRIM works well on more complex benchmarks such as AIME2024 instead of GSM8K.

### Questions
- Q1. What are the advantages of K-step reward estimation of BRIM, compared to LLMR?

### Soundness
1

### Presentation
2

### Contribution
2
