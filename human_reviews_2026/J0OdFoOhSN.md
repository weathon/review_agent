# TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Generalizing deep reinforcement learning agents to unseen environments remains a significant challenge. One promising solution is Unsupervised Environment Design (UED), a co‑evolutionary framework in which a teacher adaptively generates tasks with high learning potential, while a student learns a robust policy from this evolving curriculum. Existing UED methods typically measure learning potential via regret, the gap between optimal and current performance, approximated solely by value‑function loss. Building on these approaches, we introduce the transition-prediction error as an additional term in our regret approximation. To capture how training on one task affects performance on others, we further propose a lightweight metric called Co-Learnability. By combining these two measures, we present Transition‑aware Regret Approximation with Co‑learnability for Environment Design (TRACED). Empirical evaluations show that TRACED produces curricula that improve zero-shot generalization over strong baselines across multiple benchmarks. Ablation studies confirm that the transition-prediction error drives rapid complexity ramp‑up and that Co‑Learnability delivers additional gains when paired with the transition-prediction error. These results demonstrate how refined regret approximation and explicit modeling of task relationships can be leveraged for sample-efficient curriculum design in UED. https://geonwoo.me/traced

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the problem of Unsupervised Environment Design (UED)  by improving the environment sampling strategy. For instance, it enhances the regret estimation metric by incorporating the model's transition-prediction error , and proposes a 'Co-Learnability' metric. These contributions are designed to efficiently sample environment tasks with high learning potential , and the experiments demonstrate that this method effectively improves efficiency.

### Strengths
1. The paper's core insight is introduced very clearly, allowing the reader to easily follow the authors' thought process and understand the proposed method. The writing quality is high.

2. The concept of "Co-Learnability" is novel and interesting, and the paper provides a helpful visual analysis to support it . This idea appears to be a promising direction for multi-task generalization in reinforcement learning and deserves further investigation.

3. The experimental analysis is both thorough and solid, supported by sufficient ablation studies and insightful visualizations.

### Weaknesses
1. The novelty of the contributions appears limited. The use of a transition-prediction (reconstruction) loss is a well-established technique in RL, and the method itself is an incremental improvement upon the existing frameworks.

2. The estimation of 'Co-Learnability' is relatively coarse. As it is highly dependent on the sampling of the batch $\mathcal{T}$, its stability is a concern. The justification for using ATPL as a proxy for the 'Future value gap' (term iii in Eq. 2) lacks sufficient theoretical backing or direct experimental validation; it appears to be a heuristic choice rather than a formally derived approximation.

3. The improvement in performance appears relatively limited, even though the sample efficiency has increased. Moreover, the experiments are conducted in relatively simple envs (MiniGrid and Bipedal Walker) , which calls into question the practical to more complex tasks .

### Questions
1. Although the authors attempt to use Equation (2) to illustrate their insight for introducing ATPL, there is no theoretical proof of the relationship between ATPL and term (iii) . I believe it would be more convincing if the paper could show the progression curve of regret during training, both with and without the ATPL component.

2. I have concern about the reconstruction loss. The performance improvement from ATPL stems from introducing dynamics modeling into the loop. Is it possible that introducing the dynamics model directly on the student agent (e.g., as an auxiliary loss) would be more direct and potentially yield even better results?

### Soundness
3

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
4

### Summary
The paper studies the unsupervised environment design (UED) problem, where a teacher adaptively generates tasks and a student learns a policy from them. The authors identify a limitation in existing UED methods: their learning potential (task difficulty) measures fail to capture the mismatch between learned and true dynamics when approximating regret. To address this, the authors refine regret estimation by augmenting the positive value loss with a transition-prediction error term. Based on the refined regret and a co-learnability metric (measuring how training on one task affects others), they propose a new curriculum strategy called TRACED. Empirical results on MiniGrid and BipedalWalker demonstrate the efficacy of TRACED, supported by thorough ablation studies and wall-clock time comparisons.

### Strengths
The paper is generally well written and provides a solid review of related work.

Proposes a novel and computationally efficient curriculum strategy for UED that achieves competitive performance without additional overhead.

Includes systematic empirical evaluation, with detailed ablation studies and reporting of wall-clock time.

### Weaknesses
Clarity of Section 3.2 could be improved: 
- co-learnability term in Eq. (7) requires one-step lookahead in the curriculum
- task-difficulty $(i, t)$ defined as the regret approximation of the task $i$ when last sampled before time $t$
- co-learnability $(i, t)$ computed when task $i$ was last drawn before time $t-1$
- Please clarify why the rank transform prevents outliers from dominating the sampling distribution.

Missing discussion of ZPD-based curriculum strategies (e.g., Florensa et al., 2018; Tzannetos et al., 2023).

SFL was excluded as a baseline due to a lack of advantage in MiniGrid, but in other environments, SFL has been reported to outperform PLR. Are experiments on those domains prohibitively expensive (if so, this is understandable)?

Florensa et al., 2018: Automatic Goal Generation for Reinforcement Learning Agents.

Tzannetos et al., 2023: Proximal Curriculum for Reinforcement Learning Agents.

### Questions
clarification question: the learned policy $\pi_\phi$ is not conditioned task/environment parameter $\theta$, but rather adapts through trajectory conditioning (i.e., non-stationary behavior), right?

Lines 268-269 mention post-convergence oscillations in long-horizon MiniGrid runs for TRACED and ACCEL. How is early stopping handled in practice-by evaluating on a validation set and stopping when returns plateau?

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
The paper proposes TRACED (transition-aware regret approximation with co-learnability for environment design), an unsupervised environment design framework based on not only positive value loss (PVL) but also average transition prediction loss (ATPL) and co-learnability. On MiniGrid and BipedalWalker, the method achieves strong performance with relatively few PPO updates.

### Strengths
By introducing an intuitive transition-prediction loss and co-learnability metric, the method yields strong empirical gains with inexpensive changes.

### Weaknesses
* While co-learnability is interesting and intuitive, the paper lacks theoretical analysis or guarantees.
* Despite its stochastic dynamics formulation, ATPL is computed based on deterministic predictions. For state–action pairs with high aleatoric stochasticity, the next state is inherently unpredictable, so ATPL can be large even when the future-value gap is negligible.

### Questions
Does training on tasks with high co-learnability at time $k$ continue to accelerate progress on other tasks at subsequent times? Could you quantify this temporal persistency of co-learnability? (e.g., reporting the covariance between $\mathrm{CoLearnability}_i(k)$ and $\mathrm{CoLearnability}_i(k+1)$)

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
4

### Summary
The authors propose TRACED, a novel curriculum selection criterion for Unsupervised Environment Design methods. TRACED consists of a novel, transition-aware regret approximation and adds a co-learnability term. They evaluate TRACED in standard UED benchmarks. They provide an ablation analysis over their proposed changes and additional analyses over the curricula differences between existing methods and TRACED. They claim to show improved performance over existing methods.

### Strengths
1. Overall, the writing is easy to follow and straightforward.
2. The transition-aware regret approximation is well motivated.
3. The notation is simple, and the definitions (Section 3.2) are easy to follow.
4. The figures are mainly well formatted and readable.
5. The authors provide fair computational comparisons and a wall-clock time comparison with most existing methods.

### Weaknesses
1. The authors do not rely on novel analysis tools as proposed by SFL [1], which would allow the authors to strengthen their claim that TRACED improves performance over existing methods. For example, the authors could have analysed CVaR to show robustness to worst-case levels. Similarly, SFL provides density maps, such as those in Figures 4 and 6 of the SFL paper, which would strengthen the claim of TRACED that they improve performance. It is also unclear to the reviewer why they do not compare their method with SFL. The authors state that SFL does not outperform PLR; however, this seems directly contradicted by the authors, as seen in Figure 7a and Figure 14b.
2. There appears to be some performance difference between PLR results reported at 10k in TRACED and SFT. In SFT, it appears that PLR can achieve a mean success rate of over 0.8 on the Minigrid evaluation set at 4k PPO update steps. In TRACED, the reported performance at 10k update steps is around 0.45. How can these performance differences be consolidated?


## Clarity
1. The authors could do a more thorough job of properly reporting statistically significant results and be more transparent in what they report. For example, Figure 7 present mean and standard deviation over clearly heavily skewed results, where standard deviation is not particularly meaningful, e.g., (12% +/- 28%). Again, median and IQR would have been more appropriate.

### Questions
1. Could you please clarify the error bars for each figure? For example, in Figure 5, what is the shaded area for the median or for the mean?
2. Could you please change the y-axis tick size in Figure 7?
3. Could you please clarify the performance differences between SFL's PLR and the author's implementation?
4. The reviewer typically tries to avoid suggesting extra experiments. However, given that MultiAgent JaxNav has been released for at least a year and it has been demonstrated that most existing methods perform worse than DR on JaxNav, I believe it's only fair to request an evaluation of TRACED on JaxNav against SFL and other existing methods. This is especially true if the authors claim to beat the "state-of-the-art" CENIE (line 67, 251, 475).
5. Could you clarify how the ATPL loss term is calculated? The reviewer is unsure if the LSTM that's used for the transition prediction is also the backbone model used to calculate the value function? Or is the LSTM transition prediction model a completely separate network and no weights are shared with he value function? The authors state in the appendix, line 1246, that they differ only in the LSTM design choice. Could they clarify what they mean?

To summarise, the paper is overall pretty solid, and the idea of including the trajectory loss to capture the distribution difference is definitely interesting. The paper may feel somewhat incremental, but that doesn't detract from its solid execution. While it's nice to know that the transition loss and co-learnability influence performance, it lacks a really in-depth explanation of why co-learnability or transition loss improves performance. The reviewer is looking forward to the rebuttal period and hearing back from the authors.

### Soundness
3

### Presentation
3

### Contribution
3
