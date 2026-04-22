# BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6, 4

## Abstract
Reinforcement finetuning (RFT) is a key technique for aligning Large Language Models (LLMs) with human preferences and enhancing reasoning, yet its effectiveness is highly sensitive to which tasks are explored during training. Uniform task sampling is inefficient, wasting computation on tasks that are either trivial or unsolvable, while existing task selection methods often suffer from high rollout costs, poor adaptivity, or incomplete evidence. We introduce \textbf{BOTS}, a unified framework for \textbf{B}ayesian \textbf{O}nline \textbf{T}ask \textbf{S}election in LLM reinforcement finetuning. Grounded in Bayesian inference, BOTS adaptively maintains posterior estimates of task difficulty as the model evolves. It jointly incorporates \emph{explicit evidence} from direct evaluations of selected tasks and \emph{implicit evidence} inferred from these evaluations for unselected tasks, with Thompson sampling ensuring a principled balance between exploration and exploitation for task selection. To make implicit evidence practical, we instantiate it with an ultra-light interpolation-based plug-in that estimates difficulties of unevaluated tasks without extra rollouts, adding negligible overhead. Empirically, across diverse domains and LLM scales, BOTS consistently improves data efficiency and performance over baselines and ablations, providing a practical and extensible solution for dynamic task selection in RFT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a unified Bayesian online task selection framework for reinforcement fine-tuning of large language models. Specifically, it builds on Bayesian inference using a Beta distribution to model the policy’s success rate, incorporating both explicit evidence from batch reward signals and implicit evidence from inter-task relationships to update the parameters. The authors provide theoretical justification for the posterior inference and analyze how parameters influence the learning dynamics. Empirically, they conduct extensive experiments demonstrating the effectiveness of the proposed method and the impact of its key hyperparameters in terms of multiple metrics, including effective task ratio, time-to-baseline, and best-so-far across domains and models.

### Strengths
1. The proposed method is technically sound and principled. Incorporating both explicit and implicit evidence within recursive Bayesian inference framework for RL online task selection is well-motivated.
2. The paper provides both theoretical and empirical justifications for the key hyperparameters $\lambda$ and $\rho$, which is commendable.
3. The experimental analysis is thorough, and the metric design is reasonable, showing consistent gains across multiple domains and model scales.

### Weaknesses
1. Given that the recursive Bayesian framework has been proposed in prior work, and that the proposed estimation of implicit evidence via interpolation with the reference models is relatively simple, the overall contribution beyond prior studies [1][2] appears incremental.
2. A major issue in the paper is the discussion of prior mixing or temporal discounting, i.e., the interpretation of $\lambda$, which is confusing.
- First, it is unclear why interpolation with the prior distribution $\pi_0$ is introduced in the Bayesian sequential update. While such interpolation might be beneficial for non-stationary processes, the paper should provide appropriate references or theoretical justification.
- Several statements regarding $\lambda$ are inconsistent and potentially contradictory, raising concerns about possible implementation errors.
(1) In Proposition 2 (lines 239–242), the effective sample size $n_t$ depends inversely on $\lambda$, implying that $n_t$ increases as $\lambda$ decreases. This contradicts the claim in lines 1139–1140 that smaller $\lambda$ leads to a smaller effective sample size and higher variance.
(2) In Section 4.3, lines 350–352 state that smaller $\lambda$ emphasizes recent evaluations while larger $\lambda$ focuses on accumulated history. However, lines 373–375 claim that $\lambda \rightarrow 0$ corresponds to short-term memory and $\lambda \rightarrow 1$ to long-term memory. These descriptions are unclear and inconsistent.
- Given the known distributional form, it would be helpful to include toy examples illustrating the effects of $\lambda$ and observation noise to clarify its role.
3. In general, the analysis would benefit from separating the discussion of uncertainty estimation and success-rate estimation, rather than conflating them. The paper should more clearly define and distinguish terms such as “memory,” “forgetting,” and “exploration.”
4. The argument for low computational cost of the interpolation method in Section 3.3 is not convincing. The method requires reference labels from two models across the entire training dataset, which limits its scalability and undermines its practicality for new models and domains.

[1] Qu, Yun, et al. "Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?." arXiv preprint arXiv:2507.04632 (2025).
[2] Sun, Yifan, et al. "Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay." arXiv preprint arXiv:2506.05316 (2025).

### Questions
1. Overall, based on Figure 3, when $\lambda$ is small, the sampled $p$ from the posterior estimation tends to be smaller than the ground truth, whereas when $\lambda$ is large, the sampled $p$ tends to be larger than the ground truth. Could you provide a consistent explanation for this behavior from both the mean and variance estimation perspectives?
2. I appreciate that the authors discuss the selection of the reference model and the potential bias in estimating $\mu_B$ in the appendix. How does the choice of reference model affect empirical performance? Is it robust? Is there any guidance or heuristic for selecting it in practice?
3. How is the prior distribution (i.e., $\alpha_0$ and $\beta_0$) for each sample initialized?
4. Could you report the correlation between the interpolated average success rate and the ground-truth average success rate across training rounds to demonstrate the accuracy of the implicit evidence estimation?

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
This paper proposes a principled Bayesian framework to improve task selection efficiency during RFT of large language models. Instead of uniformly sampling tasks, which wastes computation on trivial or unsolvable ones, BOTS maintains Bayesian posteriors over task difficulties that evolve with model capability. It fuses explicit evidence (direct evaluation results) and implicit evidence (inferred task difficulty via inter-task relationships) using generalized Bayesian updates, and employs Thompson sampling to balance exploration and exploitation. The authors further introduce an ultra-light interpolation plug-in that estimates unselected task difficulty without extra rollouts. Experiments across math, code, and logic domains, using Qwen models of different scales, show that BOTS consistently improves data efficiency and training speed with negligible computational overhead.

### Strengths
1. The paper presents an interesting and original idea of combining both explicit and implicit evidence for online task selection, and unifies them under a principled Bayesian framework.

2. The interpolation-based implicit evidence plug-in is novel and practical, enabling efficient estimation without additional rollouts.

3. The writing is very clear and well-organized, with a solid and detailed appendix that strengthens the paper’s technical quality.

4. The experimental section is thorough: it clearly introduces evaluation metrics, systematically analyzes the effects of ρ and λ, and then presents comprehensive main results demonstrating certain improvements.

### Weaknesses
1. A major issue lies in the analysis of λ. The statement at line 244 that “smaller values accelerate forgetting and increase posterior uncertainty” appears inconsistent with Equation (1) and Proposition 2. Based on the update rule, smaller λ should slow down forgetting and reduce posterior uncertainty due to larger effective sample size. Moreover, line 372 later describes λ → 0 as “long memory,” which directly contradicts the earlier statement. The paper should carefully revisit this part.

Relatedly, the use of terms such as forgetting, exploration, and exploitation is somewhat confusing. It is not always clear what exactly is being “forgotten” or what dimensions are being “explored” versus “exploited,” especially since these terms may differ from their usage in reinforcement learning. Clarifying these concepts would improve readability and conceptual precision.

Overall, I suggest the authors carefully revisit the analysis of how λ and ρ affect the algorithm. The current explanations are confusing and inconsistent, making it difficult to assess whether the paper’s claims about their impact are correct.

2. On the experimental side, the evaluation on math benchmarks is limited. Only MATH500 and AIME24 are used, which seems insufficient to demonstrate generality. Moreover, AIME24 may be too difficult for small models, leading to unstable results. For example, as shown in the appendix figures, the 1.5B model’s AIME24 accuracy remains near zero throughout training, making it difficult to draw meaningful conclusions from such results.

---

I would be open to adjusting my rating depending on how the authors address these issues.

### Questions
1. Regarding the interpolation-based implicit evidence plug-in in Section 3.3: (1) What happens if there are multiple candidate weak and strong models? Would the resulting interpolations be consistent across different reference pairs?
(2) I saw the empirical results in the appendix, but the reported correlation coefficients (around 0.5) seem rather low. Could you elaborate on why this level of correlation is sufficient to justify the estimator’s reliability?
(3) If part of a dataset is either too easy or too hard—so that both weak and strong models perform similarly high or low—would the interpolation still behave well? I understand you could in principle extrapolate, but is there any empirical evidence showing this works reliably?
(4) Also, while you claim efficiency due to pre-computed references in existing datasets, would new datasets still require additional one-time rollouts for the reference models? It would be helpful to discuss this limitation explicitly.

2. In Section 4.2, you state that implicit evidence “fails to recognize when tasks are fully mastered.” Could you provide further explanation or empirical evidence to support this claim?

3. (Minor) For the BSF metric, why do you report results at 25% and 50% of total training steps? Wouldn’t BSF at 100% be sufficient to summarize the final performance? 

4. (Minor) I suggest adding a complete pseudocode of the full algorithm in the appendix. This would help readers clearly follow the overall logic and implementation flow.

5. (Optional) Including error bars for key results—at least for the 1.5B model—would make the experimental analysis more convincing, if computational resources permit.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes BOTS, a Bayesian framework for online task selection in reinforcement finetuning (RFT). It maintains posterior estimates of task difficulty and uses Thompson sampling to select tasks near a target difficulty (in practice, $p^*=0.5$).  BOTS aims to balance data efficiency and adaptability by combining explicit evidence from direct rollouts and implicit evidence inferred through an interpolation-based plug-in. Experiments on the GURU benchmark with Qwen2.5 models demonstrate improved sample efficiency compared to uniform random and other baselines.

### Strengths
- Provides a principled Bayesian formulation of online task selection that generalizes prior methods.
- Fuses explicit and implicit evidence to improve stability and cold-start performance.
- Includes ablations on $\lambda$ (forgetting) and $\rho$ (evidence fusion), providing good insight into the method’s behavior.
- The interpolation-based implicit evidence is computationally efficient (<0.2% overall overhead).

### Weaknesses
- Experiments are limited to two LLM scales (1.5B and 7B) of Qwen2.5 on a single benchmark (GURU), raising concerns about generalizability.
- The reported improvements over baselines, appear modest and sometimes inconsistent across domains and target fractions and no confidence intervals or statistical tests are provided.
- The method assumes access to reference models for implicit evidence, the practicality of this in new domains is not fully addressed.

### Questions
- Were any statistical significance tests (e.g., across seeds) conducted for the reported improvements? Confidence intervals or variance would strengthen claims of robustness.
- Since $\lambda$ and $\rho$ control forgetting and evidence fusion and seem to balance early and long-term performance, have you considered adaptive or schedule-based versions that evolve over training rather than remaining fixed?
- Correct me if I'm wrong, but there seems to be a minor inconsistency in Section 3.4: the paper defines utility as the distance $\| p_k - p^* \|$ from the target difficulty but states that tasks with the highest utility are selected, which appears opposite to the stated goal of favoring tasks difficulty near $p^*$.

### Soundness
2

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
The paper proposes a unified framework for Bayesian Online Task Selection in LLM reinforcement finetuning, named BOTS. The method jointly incorporates explicit evidence the selected tasks and implicit evidence from unselected tasks, with Thompson sampling to pick tasks which is close to a target difficulty. The method's effectiveness is empirically demonstrated on several benchmarks.

### Strengths
1. The paper is well-written and easy to follow.

2. The proposed idea is interesting and novel.

3. The empirical results demonstrates the effectiveness of the proposed method.

### Weaknesses
1. The proposed method is only evaluated on binary rewards. It is not clear whether it is applicable on more complex tasks.

2. The comparisons are internal. Lack of comparisons with other alternatives.

3. There are some related work on task selection are missed:

[1] DATS: Difficulty-Aware Task Sampler for Meta-Learning Physics-Informed Neural Networks.

[2] Model predictive task sampling for efficient and robust adaptation

### Questions
In table 1, it seems that the proposed method is sensitive to the hyperparameter values and there is no consistent conclusion. Is there any common principle to set the hyperparameters according to the task?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of online task selection in reinforcement finetuning (RFT) for Large Language Models (LLMs). The authors introduce BOTS, a unified Bayesian framework that aims to actively select tasks for each step that are of appropriate difficulty at the current stage. BOTS models task difficulty using a Beta distribution posterior and adaptively updates its beliefs by fusing two sources of information: explicit evidence from direct evaluations of selected tasks and implicit evidence from unselected tasks. The framework includes a lightweight interpolation-based method to generate implicit evidence with negligible overhead. Task selection is performed via Thompson sampling to balance exploration and exploitation. Experiments across various domains and model scales show that BOTS improves data efficiency and model performance compared to random selection.

### Strengths
1. The studied online task selection problem is crucial for the LLM reinforcement finetuning problem, and BOTS's formulation of it as a unified Bayesian inference problem is elegant and well-suited for the task.
2. BOTS avoids the need for additional model rollouts during training, and the computational overhead is shown to be negligible ($\le 0.2\%$) while consistently showing performance gains over random selection, making the framework efficient and effective in adoption.
3. The idea of trying to balance explicit and implicit evidence is practical for stabilizing the effective sample size $n_t$, as proved in Proposition 2. And the authors show it is beneficial for cold start in Section 4.2.

### Weaknesses
1. Although quite efficient, the linear interpolation function in BOTS may be weak as an instantiation of $\tilde{p}(k, \mathcal{B}_t)$. Results in Figure 5 show that the Pearson correlation between estimated and empirical difficulties is weak, and the ROC AUC is also not good enough (especially for Qwen2.5-1.5B-Instruct). And there is no theoretical analysis to justify why a linear interpolation alone can adequately capture the complex information needed for task difficulty estimation.
2. The training time is not reported in the main results. Although BOTS generates the same number of rollouts and trains for the same number of steps compared to random sampling, the actively selected tasks usually generate longer rollouts that actually need more computational resources. Thus, a fairer comparison should let random sampling generate more rollouts per task to have a similar training time as BOTS.
3. The comparisons to prior work DOTS [1] are done via "within-framework" ablations (BOTS-DOTS) rather than against their original implementations (Section 4.4, Appendix B.6). While this is sufficient for case studies within the BOTS framework, it does not fairly compare the performance between BOTS and DOTS. A comparison between BOTS and the original DOTS is needed. Additionally, more related baselines should be included for comparison to highlight BOTS's advantage (e.g., Dynamic Sampling [2], SPEED [3]).
4. The framework mainly studies two important hyperparameters, $\lambda$ and $\rho$. The ablation studies show that the model's performance is sensitive to these choices (Section 4.2, 4.3). The paper does not provide a clear methodology for setting these hyperparameters beyond empirical sweeps, which could be computationally expensive for new applications. Additionally, the momentum coefficient $\gamma$ is a crucial hyperparameter for interpolation, but the impact of it is not discussed in the paper.
5. It seems BOTS mainly works on math problems; there's no significant performance gain compared to random sampling in other cases, especially on experiments using Qwen2.5-7B on Code and Logic. The generalization is somewhat limited.
6. Minor suggestions: providing an algorithm pseudocode can help readers understand the workflow of BOTS more quickly.

[1] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay.

[2] DAPO: An Open-Source LLM Reinforcement Learning System at Scale.

[3] SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning.

### Questions
1. This work seems to be an extension of MoPPS [4], with new consideration on the balance of explicit and implicit evidence. But I don't understand why it can additionally capture "cross-task relationship". In my understanding, BOTS follows the same Thompson Sampling strategy as MoPPS, where different tasks are all treated independently. I believe BOTS aims to synchronizingly update tasks' difficulty that are not sampled at the current stage, to get a more precise estimation of these tasks for later sampling. Please correct me if I'm wrong.
2. How are $\bar{p}^k_w$ and $\bar{p}^k_s$ computed? I'm confused according to the description in the main text. From Appendix E.3, it seems that it directly uses precomputed results, then it may have large errors since training details are quite different. Moreover, I think the ablation of reference model choices is necessary and is not adequate for future work. I wonder if better reference model choices can indicate better Pearson correlation and ROC AUC?
3. Why is the result of "Random" not reported in Figure 15/16?

[4] Can prompt difficulty be online predicted for accelerating RL finetuning of reasoning models?

### Soundness
2

### Presentation
3

### Contribution
3
