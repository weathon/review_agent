# Unconstrained Models as Constrained Problem Solvers: Duality-Driven Adaptation without Retraining

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
We present a novel extension of the forward-backward (FB) representation framework that enables zero-shot constrained reinforcement learning (RL) by embedding both reward and cost functions into a shared latent space. While existing FB methods excel in generalizing across rewards, they fail to account for constraints, a critical limitation in real-world applications where agents must satisfy varying cost budgets or safety requirements. Our approach overcomes this gap through a latent-space reparameterization grounded in Lagrangian duality, allowing efficient inference of constraint-aware policies without requiring any retraining at deployment. By leveraging a latent-space reparameterization grounded in Lagrangian duality, our method allows for efficient inference of constraint-aware policies. Extensive experiments on the ExORL benchmark demonstrate that our method achieves superior task performance while adhering to cost constraints, consistently outperforming prior FB-based and primal-dual baselines. These results highlight the effectiveness and practicality of latent-space constrained policy inference for scalable and safe RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper extends the forward-backward (FB) representation framework to enable zero-shot constrained reinforcement learning. The key contribution is reformulating constrained RL as a latent-space search problem where only the Lagrange multiplier needs to be identified, rather than jointly optimizing both policy and multiplier. The method leverages pre-trained FB models that were unaware of constraints during training and adds an online calibration procedure to handle discrepancies between model estimates and actual costs.

### Strengths
- The problem studied in this paper, i.e., zero--shot constrained RL, is of high practical significance.
- The paper proposes solutions to real-world challenges including discount factor mismatches and approximation errors.

### Weaknesses
1. The core contribution reduces to using $z_{\lambda}=z_r-\lambda z_c$ with existing FB models plus standard bisection search. While the insight is valuable, the technical contribution is somewhat incremental.
2. Theorem 1 seems like a straightforward corollary of Theorem 2 from (Touati & Ollivier, 2021).
3. It would offer more insight if the comparative evaluation contains baselines that use standard constrained RL agents from scratch.
4. The "zero-shot" claim seems like an exaggeration: the online calibration  process requires 30K samples. This is non-trivial compared to the number of samples used in pre-training (2M).

### Questions
1. Line 183: $d$ is not defined before this first usage.
2. Can the number of samples needed by online calibration be reduced?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to achieve zero-shot adaptation for different constraint configurations by using a forward-backward (FB) framework that models the optimal policies for a set of potential rewards. Specifically, it searches for an appropriate Lagrangian parameter based on value estimation, and this parameter, together with the reward latent and constraint latent, forms an index to constraint-satisfying optimal policies. In addition, the paper proposes using a few online samples to finetune the Lagrangian parameter to mitigate the gap between the real cost return and the cost value estimate. Experiments on ExORL benchmark datasets demonstrate the method’s ability to achieve constraint-satisfying policies through the FB framework and online calibration.

### Strengths
The motivation of using the FB framework to achieve zero-shot constraint adaptation is relatively novel and intuitively reasonable. Specifically, the potential of learning safe behaviors for various constraints even without explicit constraint functions during training is impressive.

### Weaknesses
- Descriptions and formulations should be further improved.
    - I recommend substantial revisions to Section 3, as many notations of FB are not clearly introduced. For example, the definitions of $ds’$ and $ρ$, the connection of $X$ in Eq.(4) with other formulations, and the operational details of $Q_r^* = Mr$, $Q_r^*(s,a) ≈ (F(s,a,·))B(·)^T r$, and $M ≈ FB$.
    - In addition, the implementation details for training $F$ and $B$ should be provided, or at least linked to the appendix for reference.
    - The $\overline F$ in Eq.(15) appears to differ from the one in Section 4.1, so this notation should be reclarified.
- The choice of baselines is limited and lacks comparisons with standard safe RL baselines. Several existing works [1–3] have addressed safe RL under various constraint thresholds, and these should be discussed and compared.
    
-  Although the paper claims zero-shot adaptation for both new cost functions and thresholds, the experiments only consider varying thresholds.
    
- The assumption of access to 3k online samples and their cost returns for fine-tuning is rather strong. While it intuitively improves performance for any algorithm that models diverse policies, it is demanding in practice and somewhat contradicts the zero-shot setting claimed in the paper.

[1] Liu, Zuxin, et al. "Constrained decision transformer for offline safe reinforcement learning." *International conference on machine learning*. PMLR, 2023.

[2] Lin, Qian, et al. "Safe offline reinforcement learning with real-time budget constraints." *International Conference on Machine Learning*. PMLR, 2023.

[3] Yao, Yihang, et al. "Constraint-conditioned policy optimization for versatile safe reinforcement learning." *Advances in Neural Information Processing Systems* 36 (2023): 12555-12568.

### Questions
- There are some environments specifically designed for constraint RL, such as Safety-Gym [1] and OSRL [2]. I believe they are also suitable for evaluating different thresholds or even different constraint functions. Could you clarify why these benchmarks were not used?
- I am a bit confused about the CRL_FB results in the table. While I understand that cost value estimation errors can lead to instability and constraint violations, why does the method exhibit extremely conservative behaviors? Could you elaborate on this point?
- I recommend reporting a plot of the relationship between $\lambda$ and $Q_c$, which could help verify their correlation.


[1] Ji, Jiaming, et al. "Safety gymnasium: A unified safe reinforcement learning benchmark." *Advances in Neural Information Processing Systems* 36 (2023): 18964-18993.

[2] Liu, Zuxin, et al. "Datasets and benchmarks for offline safe reinforcement learning." *arXiv preprint arXiv:2306.09303* (2023).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper extends forward–backward (FB) representations to constrained reinforcement learning by embedding reward and cost into a shared latent space and using a Lagrangian-duality reparameterization to enable zero-shot, constraint-aware policy inference without retraining. Specifically, author apply a bisection method to identify the appropriate  Lagrange multiplier. I think the proposed method is novel and have several questions about the experimental setup and specific implementation details, which I outline below.

### Strengths
1. This paper convert the Lagrangian dual optimization problem into a direct search over λ, which is simple and effective.
2. The paper is clearly structured, and I was able to grasp its content quickly.

### Weaknesses
1. The paper uses a bisection search over $\lambda$, but it does not prove that this procedure yields the optimal $\lambda$.  The bisection routine implicitly assumes that the surrogate cost  $\hat{C}_{\text{latent}}(\lambda) = \overline{F}(\cdot, z_\lambda)^\top z_c$  is monotone in $\lambda$, but the paper does not establish such monotonicity; policy switches can make  $\hat{C}_{\text{latent}}(\lambda)$ non-monotonic. 
2. The **Online Calibration** stage requires extra online sampling, whereas the baselines are evaluated without such an interaction pass. It makes the comparison not strictly fair. The author didn’t mention this point in experiments.

3. The experiments do not include a comparison against directly applying the original Lagrangian dual optimization. Consequently, Limitation 1 is not empirically substantiated.

### Questions
1. The proposed method is based on FB—which inherently relies on discounted returns—and thus does not fundamentally resolve Limitation 2. 
2. The algorithm’s performance without online calibration has not been evaluated.
3. The $\lambda$-search procedure lacks theoretical justification. The parameterization $z_{\lambda} = z_r - \lambda z_c$  constrains the policy space —and is this form essentially optimal?

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
The paper proposes a post-hoc way to impose budget (cost) constraints on a pre-trained forward–backward (FB) model at deployment time. It linearly combines reward and cost embeddings with a scalar $\lambda$ in latent space, searches $\lambda$ (e.g., via bisection) to meet a target budget, and uses a light online calibration step to correct the gap between latent and realized costs. Empirically, the method is evaluated on continuous-control tasks and compared against FB variants and a primal–dual baseline, showing a favorable performance–compliance trade-off.

### Strengths
Simple & practical: Turns constrained control into a 1-D search over  $\lambda$  without retraining; easy to bolt onto existing FB models.

Low deployment overhead: Only post-hoc inference + brief online calibration; attractive for real systems where retraining is costly.

Empirical coverage: Multiple tasks, budgets, and ablations (including a Hoeffding-style variant) help illustrate behavior across regimes.

Clear motivation: Addresses discount/representation mismatch between latent and realized costs and proposes a concrete calibration mechanism.

### Weaknesses
* **“Zero-shot” vs. online calibration:** The approach depends on non-trivial online interactions for calibration and ($\lambda$) selection. This weakens the “zero-shot” claim; please quantify how much those interactions matter (strict zero-shot vs. few-shot vs. your full setting).
* **Monotonicity/robustness of the ($\lambda$  $\mapsto$) cost curve:** The method assumes near-monotonic behavior to justify bisection, but function approximation noise can break this. The paper would benefit from conditions, diagnostics, and fallbacks for non-monotone cases.
* **Theory depth:** Current analysis mostly repackages Lagrangian ideas into latent space; tighter guarantees under approximation error and finite data would strengthen novelty.

### Questions
# Questions

1. **Positioning vs. BCORLE ((\lambda)).**
   BCORLE uses **(\lambda)-generalization** to reuse policies across different budgets in an offline coupon-allocation setting. Could you clarify where your method aligns or diverges—e.g., training regime (offline vs. pre-trained FB), reliance on online calibration, supported constraint types, and deployment-time adaptability?

2. **Stronger baselines.**
   Baselines feel a bit light—could you consider adding a stronger reference point such as **TREBI (ICML 2023)** for real-time budget constraints?

3. **Monotonicity for bisection.**
   Your approach implicitly assumes the realized cost is (approximately) monotone in ($\lambda$ ). Could you (i) state conditions under which this holds with function approximation and finite data; (ii) report how often violations occur, with representative cost–($\lambda$ ) plots and failure cases?

4. **($\lambda$ ) bounds & stopping.**
   How are search bounds and stopping criteria chosen across tasks? Is there a task-agnostic or adaptive rule, and how sensitive are outcomes to these choices?

5. **Generalization.**
   Does a (\lambda) tuned for one task/domain transfer to related tasks, or must calibration be repeated each time?

### Soundness
2

### Presentation
3

### Contribution
2
