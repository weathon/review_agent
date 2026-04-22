# Efficient and Robust Behavior Policy Search for Online Off-policy Evaluation through Transition Gradients

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2

## Abstract
In reinforcement learning policy evaluation, classic on-policy methods often suffer from high variance when estimating policy performance. To mitigate this issue, *behavior policy search* has been proposed to learn data-collecting policies tailored to reduce online evaluation variance. However, these approaches do not account for uncertainties in the transition functions. In practice, simulator transitions often differ from real world due to modeling errors or approximation limitations. As a result, behavior policies trained in simulation may still yield high variance when
deployed in real environments, leading to costly reliance on real-world evaluation samples. In this work, we propose a double-loop gradient-based algorithm for learning behavior policies that are both efficient and robust to transition uncertainty. Theoretically, we derive novel transition-variance gradient expressions and establish global convergence guarantees for the algorithm. Numerically, we demonstrate that our method is less sensitive to transition perturbations than existing approaches, providing supportive evidence for its practical utility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a robust and efficient behavior policy search (BPS) framework for reinforcement learning (RL) policy evaluation under transition uncertainty. Traditional on-policy Monte Carlo evaluation suffers from high variance, and standard BPS methods typically assume fixed simulator dynamics, which can lead to poor robustness for real-world applications.

This paper proposes a novel minimax formulation, where the inner-layer (i.e. maximization) represents adversarial transition perturbations and the outer-layer (i.e. minimization) solves for a behavior policy that minimizes worst-case evaluation variance.
The paper derives closed-form transition-gradient expressions (on-transition and off-transition cases), designs an inner-loop adversarial transition optimizer, and proposes a double-loop robust gradient algorithm (DRVG) for variance-minimizing policy search. Theoretically, they prove global convergence under linear-softmax parameterization with an $O(1/\sqrt{n})$rate.

The paper then conducts experiments on Garnet MDPs and an inventory management task, which show that DRVG has lower variance and is more robust to adversarial transition perturbations than baselines including MC, Behavior Policy Gradient, and Robust On-policy Sampling.

### Strengths
**Originality**: good

1, The paper discusses behavior policy search from a minimax optimization point of view with transition uncertainty, which seems novel in the RL evaluation literature to me.

2, It introduces transition-gradient methods for variance objectives. This seems a direction not thorough explored previously.

3, The unification of variance reduction and robustness bridges off-policy evaluation and robust MDP theory, a meaningful conceptual contribution.

**Quality**: fine

1, Theoretically rigorous paper: multiple analytical gradient theorems are presented with proofs and convergence guarantees.

2, Empirical results, though limited in scale, consistently support the theoretical claims.

**Clarity**: good

1, The math of the paper is clear; definitions are presented without ambiguity,

2, key steps are summarized in theorem boxes and algorithms.

**Significance**: fine

1, Robust policy evaluation is crucial for sim-to-real transfer and safe RL; the proposed method contributes toward that.

2, The convergence guarantee for a variance-minimizing, nonconvex–nonconcave min-max problem is an important theoretical advancement.

### Weaknesses
There is no critical weakness detected. However, I do have the following concerns.

(1) Some assumptions seem a bit strong.
Coverage/bounded-ratio assumptions are strong and under-discussed. Convergence and smoothness rely on a uniform bound $C$ on quantities such as importance ratios, plus bounded features, etc. Otherwise the current analysis doesn’t apply. I am not sure if these assumptions are realistic.

(2) Lack of ablation / sensitivity studies

The role of the KL regularization coefficient $\eta$ is not empirically analyzed; varying $\eta$ could show the trade-off between robustness and over-conservatism.

The effect of inner-loop precision $\epsilon_i$ on performance and convergence is not reported.

(3) (minor weakness), Limited empirical experiment

Experiments are restricted to synthetic MDPs (Garnet) and a toy control problem (inventory management).

### Questions
(1) The global convergence theorem (Theorem 5) assumes a linear–softmax form and convex domain. Do the authors believe the same convergence behavior holds in the nonconvex case (either theoretically or empirically)? Would like to hear some comments.

(2) The proposed algorithm claims to achieve ‘robustness’. What is exactly the definition of ‘robustness’? Is it equivalent to ‘variance minimization’? 

(3) The entire paper focuses on variance minimization and assumes unbiasedness of the OPE estimator. I might miss something but why can we take unbiasedness for granted?


Thank you.

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
3

### Summary
The paper studies behavior policy search (BPS) for off-policy evaluation (OPE) under transition-model misspecification. It frames BPS as a min--max objective that seeks behavior policies whose evaluation variance is small even under adversarially perturbed dynamics within a KL-bounded ambiguity set. The authors derive transition-variance gradient formulas for both on-transition and off-transition forms, propose an inner-loop adversarial ascent over transition parameters and an outer-loop projected descent for the behavior policy and provide a global convergence guarantee under linear-softmax policies with bounded features. Experiments on Garnet MDPs and an inventory-management task suggest improved robustness (lower variance, less sensitivity to transition shifts) compared to baselines such as Monte Carlo, Behavior Policy Gradient (BPG), and ROS.

### Strengths
- This paper develops a logically consistent double-loop optimization framework that integrates behavior policy search under transition perturbations, analytical gradient derivation, and global convergence analysis.

- Unlike conventional BPS methods assuming fixed transitions, the authors introduce an adversarial perturbation model to capture simulator–environment discrepancies, theoretically enhancing robustness to transition uncertainty and unifying OPE and RMDP principles.

- The derived gradient expressions are estimator-independent and combined with a KL regularizer to constrain adversarial deviations, ensuring both robustness and extensibility to advanced estimators or real-world deployment.

### Weaknesses
- A key theoretical inconsistency arises in the formulation of the proposed min--max objective, defined in Eq. (1)--(2) as $\min_{\theta \in \Theta} \max_{\omega \in \Omega} V_{H \sim p_{\omega}, \pi_{\theta}} [\mathrm{OPE}(\pi_e, \pi_{\theta}, H)].$ This objective is inherently nonconvex--nonconcave and generally non-differentiable with respect to both the policy parameter $\theta$ and the transition parameter $\omega$. The authors acknowledge this challenge (lines345--350), yet later claim a global convergence guarantee in Theorem 5. However, this guarantee is established only under a restrictive linear-softmax policy parameterization, which artificially convexifies the objective and simplifies the underlying problem.
- The theoretical results hinge on the linear-softmax policy assumption (lines358--361). While this setting is analytically convenient, it is limited to linear function approximation, a regime that has been extensively explored in earlier literature. In contrast, the experimental section employs neural network parameterizations (line452), which invalidate the convexity assumptions required for the claimed convergence. As a result, the global convergence guarantee does not strictly apply to the practical algorithm used in the experiments.
- Although the paper emphasizes theoretical robustness, the empirical evidence remains limited. The experiments focus mainly on small-scale settings such as Garnet MDP and inventory management. To convincingly demonstrate the robustness and scalability of the proposed double-loop DRVG algorithm under transition uncertainty, evaluations on more complex and high-dimensional environments (e.g., MuJoCo, Atari, or robotic simulators like Gazebo) would be valuable. Such experiments would better substantiate the claimed robustness of the minimax formulation.

### Questions
- Why does the paper adopt a standard MDP formulation instead of a more general POMDP framework, given that transition uncertainty or partial observability could naturally arise in the studied setting? 

- For better readability and referencing (e.g., in Section 5.2), it would be beneficial to formally present key conditions using a numbered `assumption` environment.

- The paper introduces the regularization parameter $\eta$ in the KL-penalized inner-loop objective, yet does not explain how $\eta$ is selected in practice. Could the authors provide intuition or a principled guideline for tuning $\eta$? For example, how sensitive is performance to this choice, and does it correspond to a trade-off between robustness and fidelity to the simulator dynamics?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of high-variance policy evaluation in reinforcement learning (RL), especially when there is a mismatch between simulated and real-world transition dynamics. The authors propose a novel robust BPS framework formulated as a minimax optimization: an adversarial transition model is introduced to maximize the evaluation variance while the behavior policy is simultaneously optimized to minimize it. The core contribution is a double-loop gradient-based algorithm that alternates between finding worst-case (variance-maximizing) transition dynamics and updating the behavior policy to counteract those dynamics. The paper derives analytical expressions for the gradient of the evaluation variance with respect to transition parameters and provides theoretical guarantees of convergence for both the inner (adversarial transition) loop and the overall procedure. Empirically, experiments on benchmark tasks demonstrate that the proposed method achieves significantly lower evaluation variance under perturbed transitions compared to baseline approaches, indicating improved robustness to model mismatch.

### Strengths
Building upon prior work, the paper makes a notable breakthrough by integrating behavior policy search with adversarial robustness for the first time, specifically addressing the issue of environmental dynamics uncertainty that previous methods have largely overlooked. The innovation lies in introducing an adversarial transition model to simulate worst-case scenarios, thereby enabling the training of behavior policies that maintain low variance across diverse and uncertain dynamics.

### Weaknesses
Despite the aforementioned merits, the paper still exhibits several limitations that warrant the authors’ attention and improvement:
1. The proposed double-loop adversarial optimization framework enhances robustness but introduces additional computational complexity and overhead. The inner-loop adversarial training (which searches for the worst-case transition) and the outer-loop policy update are nested, requiring the simulation of a large number of trajectory samples per iteration. This leads to substantial computational costs, making the method potentially inefficient or impractical in large-scale or high-dimensional environments, thus constraining its real-world applicability.
2. The robustness of the proposed method is defined with respect to a pre-specified transition perturbation model—that is, the adversarial process searches within a parameter space Ω for the most adverse transition probability distribution. However, if the actual environmental variations fall outside the assumed uncertainty set (e.g., structural model biases not captured by the perturbation assumption), the robustness guarantees may degrade. In other words, the degree of robustness depends critically on whether the uncertainty set adequately encompasses real-world dynamics, a factor that must be carefully calibrated in practical deployments.
3. The authors assume that the importance sampling ratios (the probability ratio between the target and behavior policies in policy evaluation) are uniformly bounded, and that the transition probability function is continuously differentiable and twice differentiable with respect to its parameters. While these assumptions are mathematically convenient for deriving gradient formulations and convergence proofs, they may not always hold in practice. Relaxing these assumptions or analyzing the algorithm’s behavior when they are violated would improve the generality and theoretical robustness of the proposed approach.
4. Although the experimental results demonstrate the effectiveness of the proposed method, the evaluation remains somewhat limited in scope and depth. Specifically, experiments are conducted only on two medium-scale environments—Garnet MDPs and a simplified inventory management problem. These domains are relatively small and structurally simple, and thus may not fully capture the diversity and complexity of real-world applications.
5. While the authors compare their method against standard on-policy Monte Carlo evaluation and two existing behavior policy optimization approaches (e.g., BPG by Hanna et al. and the method by Zhong et al.), the empirical comparison could be further strengthened. In particular, although the related work section mentions the closed-form offline behavior policy solution by Liu and Zhang (2024) and other robust evaluation techniques (Katdare et al., 2023; Voloshin et al., 2021), these baselines are not included in the experimental comparison.

In summary, the paper presents an innovative and well-motivated solution addressing two key challenges in policy evaluation—variance reduction and environmental uncertainty. The contributions are substantial, and the results largely support the main claims. Nevertheless, the aforementioned issues—namely, the algorithmic complexity and assumptions, the limited experimental scope, and the incompleteness of baseline comparisons—should be further clarified and addressed.

### Questions
null

### Soundness
2

### Presentation
2

### Contribution
2
