# VIPO: Value Function Inconsistency Penalized Offline Reinforcement Learning

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Offline reinforcement learning (RL) learns effective policies from pre-collected datasets, offering a practical solution for applications where online interactions are risky or costly. Model-based approaches are particularly advantageous for offline RL, owing to their data efficiency and generalizability. However, due to inherent model errors, model-based methods often artificially introduce conservatism guided by heuristic uncertainty estimation, which can be unreliable.  In this paper, we introduce VIPO, a novel model-based offline RL algorithm that incorporates self-supervised feedback from value estimation to enhance model training. Specifically, the model is learned by additionally minimizing the inconsistency between the value learned directly from the offline data and the one estimated from the model. We perform comprehensive evaluations from multiple perspectives to show that VIPO can learn a highly accurate model efficiently and consistently outperform existing methods. In particular, it achieves state-of-the-art performance on almost all tasks in both D4RL and NeoRL benchmarks. Overall, VIPO offers a general framework that can be readily integrated into existing model-based offline RL algorithms to systematically enhance model accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel offline model-based reinforcement learning method called VIPO that incorporates value inconsistency loss in the offline model learning, where two distinct values are learned from the same offline dataset by introducing dual usage of the offline data. Authors contend that the uncertainty estimation of the conventional offline model-based methods is hard to rely on when complex offline data is given to the agent. They further suggest a two-way value learning approach: learning a value solely learned from the offline data and another value function learned with the empirical transition dynamics model, supported by theoretical analysis and empirical observation. They conduct an empirical study with model-free and model-based offline RL baselines on the D4RL and NeoRL benchmarks. VIPO achieves superior performance across benchmarks, highlighting the lower model prediction error and higher empirical performance.

### Strengths
- The authors provide comprehensive explanations on connecting theoretical foundations and practical implementations.
- The authors rigorously provide supplementary proofs for theorems on value learning and the model gradient theorem.
- Experiments on D4RL show improved performance across model-free and model-based baselines.
- Experiments on NeoRL demonstrate an interesting perspective of VIPO when the given data is limited.

### Weaknesses
- The novelty of VIPO is limited. The main contribution relies on reducing uncertainty estimations to learn an accurate model by introducing an auxiliary value inconsistency loss into the original model learning. However, there is no guarantee that the learned model predicts reliable synthetic rollouts under out-of-support data in the current methodology. Besides, the value inconsistency loss largely depends on the assumption that the learned values precisely estimate ($V^\mu_d (s)$ and $V^\mu_m (s)$) approximate the true value(L171, L188), whereas the learned value function fails to predict accurate values when the given $(s,a)$ pair lies outside of the offline data in general [1]. Based on my conjecture, employing a surrogate objective with heuristics in Section 3.3 contributes to the empirical performance gains to some extent, while those observations benefit the task-specific problem formulation- MuJoCo locomotion tasks often exhibit tightly bounded states and actions.
- Experimental results present conflicting views from the perspective of the authors. In Table 1, MOPO outperforms VIPO-MOPO in five out of twelve cases when comparing the effect of planners, which contrasts with the authors' argument that VIPO-MOPO outperforms MOPO across all tasks in L353. Additionally, the y-axis ticks of Figure 2 are misaligned across baselines. While it is notable that VIPO demonstrates increasing uncertainty when the drop ratio increases, the absolute difference is marginal when the y-axis is scaled with each other.

[1] Levine, Sergey, et al. "Offline reinforcement learning: Tutorial, review, and perspectives on open problems." arXiv 2020.

### Questions
- Could you conduct additional experiments in more complex domains to verify that the empirical gains do not solely stem from the practical designs? Current tasks often contain narrow bounds in state or action spaces, which aligns with heuristics that numerical differences are trivial. For instance, the tabletop robotic manipulation benchmark [1] provides tasks with a wide-ranging state space (e.g., distance in Cartesian space from a hand to an object).
- Are there alternative ways to approximate the augmented model learning loss instead of replicating the same tuple for the next time-step tuple, which can be further extended to domains outside of locomotion tasks? I believe suggesting more comprehensive directions for implementing the model gradient theorem would significantly improve the novelty of the paper.
- What is the limitation of VIPO? Discussions on the potential drawbacks of the proposed method should be addressed.

[1] Mandlekar, Ajay, et al. "What matters in learning from offline human demonstrations for robot manipulation." arXiv 2021.

Minor problems
- L122: Maybe a typo? $P(s'|s,a) : \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$
- L170: What is the meaning of "densely sampled"? Does this sentence stand for a nearly full-coverage offline dataset?
- Table 1: Bold faces do not denote the best scores. (MOReL is the best in *hopper-r* and *walker2d-r*, VIPO-MOPO is the best in *hopper-m-e*)

### Soundness
3

### Presentation
3

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
This work proposes VIPO, a model-based offline reinforcement learning algorithm that incorporates value function inconsistency into the training loss of the dynamics model. The key idea is to update the model by minimizing the discrepancy between the value function learned from offline data and the value estimated from the learned dynamics. The authors further derive how to compute the model gradients with respect to the parameters of the proposed loss function. In the evaluation, VIPO outperforms several SOTA model-based offline RL algorithms on the D4RL MuJoCo and NeoRL benchmarks.

### Strengths
1. VIPO introduces a novel perspective on value function inconsistency, which has been largely overlooked in previous model-based offline RL works.

2. The paper is clearly structured, allowing readers to follow the overall flow and reasoning easily.

3. When comparing several algorithms, VIPO achieves SOTA performance.

### Weaknesses
1. As I understand it, MOBILE quantifies uncertainty through the inconsistency of Bellman estimations under an ensemble of learned dynamics models. VIPO leverages value function inconsistency as a self-supervised learning signal that directly guides the model training. Considering that these two algorithms are based on different conceptual perspectives, it may not be appropriate to present the combined performance of VIPO and MOBILE as the representative result of VIPO in the evaluation.

2. In fact, when comparing the performance between MOPO and VIPO-MOPO, there are several offline datasets (e.g., hopper-r, walker2d-m, halfcheetah-m-r, walker2d-m-r, etc.) where MOPO outperforms VIPO-MOPO. This raises uncertainty about whether the performance improvement of VIPO is truly significant or consistent across tasks.

3. In Figure 2, the uncertainty scales for MOPO and VIPO are presented on separate axes, making direct comparison between the two methods difficult.

### Questions
1. In Table 3, measuring model error only for a rollout length of 1 is somewhat limited. Since model rollouts are typically performed for horizons of 5 or more, comparisons over longer horizons would provide a more comprehensive evaluation.

2. How would the performance change if a different type of model uncertainty, other than the one used in MOPO, were employed? It would be interesting to see whether VIPO remains effective under alternative uncertainty estimation schemes.

### Soundness
3

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
2

### Summary
This paper introduces VIPO (Value Function Inconsistency Penalized Offline Reinforcement Learning), a novel model-based offline RL algorithm designed to significantly improve the accuracy and reliability of the learned dynamics model. The core motivation is that previous model-based methods rely on heuristic uncertainty estimation to enforce conservatism, which is often unreliable in practice.

### Strengths
1. The central idea is the innovative dual-usage of offline data to generate a self-supervised signal for model training. This contrasts fundamentally with previous methods that used the data only once to learn the model ensemble.

2. The derivation of the Model Gradient Theorem (Theorem 3.3) is a significant technical achievement, providing the analytical expression needed to compute the gradient of the complex augmented loss.

### Weaknesses
1. The calculation of the surrogate gradient (Eq. 11) relies on the practical assumption that the short sampling interval in continuous-control problems makes the state change over a single step numerically insignificant. This allows approximating $(s', a', r', s'')$ using the available single-step samples $(s, a, r, s')$. This is a heuristic approximation that introduces an unquantified error, and its effectiveness may degrade in environments with high-frequency dynamics or more complex state transitions.

2. The benchmark results show that VIPO outperforms COMBO and RAMBO, but the critical experiments demonstrating uncertainty reliability (Figure 2) and predictive capability (Table 3) only compare VIPO to the Original Loss (OL) Model (which is MOPO/MOREL/MOBILE's model objective). Including a direct comparison of the predictive capability against models trained using other uncertainty-based model training methods would have provided a more comprehensive validation.

### Questions
Since the main benefit is improved model accuracy over previous conservatism strategies, how does the predictive capability of the VIPO model (Table 3) compare directly against models trained using other prominent model-based conservatism strategies, such as the maximum pairwise difference or the adversarial approach? 


The core VIPO method uses the MOBILE planner (Algorithm 2). Could the authors include an ablation study that swaps the planner used in Algorithm 2 for a simpler, less aggressive one (e.g., a standard SAC planner without the uncertainty penalty $\beta\mathcal{U}(s,a)$ term) to isolate how much of the performance gain is attributable solely to the $\mathcal{L}_{vic}$-trained model versus the aggressive mobile-like policy optimization loop?

### Soundness
3

### Presentation
2

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
The paper introduces VIPO, a novel model-based offline reinforcement learning (RL) algorithm designed to address the inherent challenges of offline RL, such as overestimation of values and model uncertainty. By incorporating value function inconsistency into the model training process, VIPO improves the model's accuracy and its ability to generalize from limited data. The paper presents empirical results across multiple benchmarks (D4RL, NeoRL), showing that VIPO consistently outperforms previous methods, demonstrating its efficacy in learning accurate models from offline datasets.

### Strengths
1. The idea of value function inconsistency as a self-supervised loss to enhance model accuracy in model-based offline RL is novel. The methodology and experimental setup are clearly presented, with an appendix and code provided for additional details.
2. The paper presents a well-defined theoretical framework for the gradient of the augmented loss function, contributing to a deeper understanding of the algorithm’s mechanics.

### Weaknesses
1. My main concern lies in the insufficient experimental evaluation. The paper primarily evaluates the algorithm on simpler locomotion tasks, such as Walker and Hopper, which may not fully demonstrate its generalization capabilities. Test results on tasks that rely on visual input(V-D4RL), navigation tasks(Antmaze) or more complex manipulation tasks, such as Fetch environment, would significantly improve the robustness and generalization of the reported findings. 

2. Additionally, the paper lacks ablation studies, such as evaluating the impact of the number of ensemble models (N) or removing the value inconsistency penalty, which are essential for understanding the contribution of each component to the overall performance of the algorithm.

3. While the paper highlights the improved performance of VIPO, it lacks a detailed analysis of the computational cost and training time required by VIPO compared to existing algorithms. Including this information would provide a more balanced perspective on the algorithm’s practical utility, particularly for real-world applications.

### Questions
Please refer to the "Weaknesses" section, I will raise my score if my concerns are addressed.

### Soundness
3

### Presentation
3

### Contribution
3
