# Constraint-guided Hardware-aware NAS through Gradient Modification

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Neural Architecture Search (NAS), particularly gradient-based techniques, has proven highly effective in automating the design of neural networks. Recent work has extended NAS to hardware-aware settings, aiming to discover architectures that are both accurate and computationally efficient. Many existing methods integrate hardware metrics into the optimization objective as regularization terms, which introduces differentiability requirements and hyperparameter tuning challenges. This can either result in overly penalizing resource-intensive architectures or architectures failing to meet the hardware constraints of the target device. To address these challenges, we propose ConNAS, a novel gradient-based NAS framework that enforces hardware constraints directly through gradient modification. This approach eliminates the need for differentiable hardware metrics and regularization weights. The novelty in ConNAS lies in modifying gradients with respect to architectural choices, steering the search away from infeasible architectures while ensuring constraint satisfaction. Evaluations on the NATS-Bench benchmark demonstrate that ConNAS consistently discovers architectures that meet the imposed hardware constraints while achieving performance within just 0.14% of the optimal feasible architecture. Additionally, in a practical deployment scenario, ConNAS outperforms handcrafted architectures by up to 1.55% in accuracy under tight hardware budgets. Our code is publicly available at https://gitlab.kuleuven.be/m-group-campus-brugge/distrinet_public/connas.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes to search for neural network architecture in a differentiable manner, guaranteeing hardware constraint compliance.
There are prior works that adopt hardware-related loss to add a hardware related objective.
However, using hardware loss as a penalty doesn't guarantee compliance with the hardware constraint.
Instead of using auxiliary loss for constraint, the paper proposes to modify the gradients of architecture parameters toward a direction for constraint compliance.

The novelty of the paper is computing the direction of the gradient that make the searched network comply with hardware constraint.

For a layer $l$, additional gradients of architecture parameters are achieved in the following manner.
- If all candidate operations comply with the constraint, the additional gradients are zero.
- If only some of them comply, split candidate operations into two sets according to the compliance.
For all combinations made by extracting one from both sets, a unit vector whose values are zero except for the indices of selected operations is initialized.
The absolute values of the indices are the same, while the sign of the value becomes positive if it complies with the constraint, negative if not.
And then, all combinations of the unit vectors are summed, and the result vector is normalized.
- If all candidate operations don't comply with the constraint, sort the operations according to the hardware metric and get all the combinations of two operations among them.
Then, follow the steps in the above.

With the proposed gradient modification, the well-performed network architectures that comply with the given constraint can be searched.

### Strengths
- Instead of adopting an auxiliary loss to search models that comply with the given constraint, the paper proposes to modify the gradients of architecture parameters according to the relative relation between arbitrary candidate operations. This idea is strong and valid.

- The performance of searched models is less affected by the hyperparameter that balances the gradient for performance and that for compliance. The hyperparameter doesn't have to be fine-tuned well, and it leads to reduce the search cost, too.

- The paper develops what it contends well, and thus it is easy to follow. Also, the paper shows the effectiveness of the proposal well, with various experiments.

### Weaknesses
- The paper addresses many prior works in the NAS field, but it doesn't address hardware constrained NAS works [1, 2], which are directly related to this paper.

- It is hard to understand Figure 1. Are the $G_{\alpha_l}$ and $-\nabla_{\alpha_l} \mathcal{L}(w, a)$ of each graph projecting the same vectors in different dimensions? If so, it is not clearly delivered.


[1] Nayman, Niv, et al. "Hardcore-nas: Hard constrained differentiable neural architecture search." International Conference on Machine Learning. PMLR, 2021.

[2] Hong, Deokki, et al. "Enabling hard constraints in differentiable neural network and accelerator co-exploration." Proceedings of the 59th ACM/IEEE Design Automation Conference. 2022.

### Questions
- Are the architecture parameters of all layers updated with the proposed gradient modification at once?
Or, a subset of layers is selected to update their architecture parameters with a single batch?

- The reviewer think that the proposal will be too complicated if there are multiple constraints.
Can the authors execute additional experiments with multiple constraints, and report how much the experiment time increases  and whether the searched networks comply with constraints well?

### Soundness
4

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
4

### Summary
The paper proposes CONNAS, a gradient-based NAS framework that finds the optimal feasible neural network architecture that satisfies a set of hardware constraints (number of parameters, FLOPS, and peak memory usage, etc.). The problem is defined as choosing the optimal subset of operations from a supernet by optimizing both the architecture weight and the operation weight. Unlike existing differentiable methods that include hardware regularization terms in the loss function, CONNAS directly modifies the gradient direction to favor the architecture that satisfies the hardware constraints. For each architecture weight, they find a normalized unit vector by aggregating all directionalities that point from candidates not satisfying the constraint, to the ones that do. Then, the vector is weighted and added to the gradient to push the update away from constraint-unsatisfying candidates, and towards the satisfying ones.  
The method is evaluated on two benchmark tasks, NATS-Bench-TS and NATS-Bench-SS, and against gradient-based NAS methods that use hardware-aware regularization terms. The authors show the searched architecture has better predictability whilst better conforming to the given hardware constraints.

### Strengths
**[S1]** Simple and modular mechanism for constraint enforcement.
CONNAS steers the architecture update using a geometrically defined direction dir_{C} computed from local candidate substitutions instead of crafting differentiable hardware surrogates and tuning the weights. In addition, CONNAS demonstrates a clean adaptation of CGGD to NAS without performance degradation.

**[S2]** Solid empirical evidence on multiple constraint types and search spaces
On NATS-Bench-TS/SS across CIFAR-10, CIFAR-100, and ImageNet16-120, CONNAS achieves a low relative error compared to the optimal feasible architectures while consistently satisfying constraints. In CSS-MEM, CIFAR-10 shows a 0.14% relative error, and strong results hold for PARAM and FLOPs as well, as shown in Table 2 and Figure 2. Furthermore, the paper reports an ablation on the rescale factor R, demonstrating robustness for moderate R in Appendix B.

**[S3]** Practicality in real scenarios on edge-ML cases.
By presenting experiments on induction-motor fault diagnosis under explicit MCU-level (STM32 family) budgets and within a large 1.94B-architecture 1D-conv search space, the paper shows consistent outperformance over handcrafted baselines under comparable budgets. Providing a practical scenario further strengthens the paper’s motivation for edge-ML applications.

**[S4]** Good writing
The overall presentation of the paper is good and easy to follow. The authors describe their approach in detail with well-defined terms.

### Weaknesses
**[W1]** Missing baselines and fairness should be stronger. 
The paper does seem to discuss existing solutions. In my search, it appears there are several existing works on hard-constrained NAS, and some of them are very similar to this work, including Hardcore-NAS [1], HDX [2], or TF-NAS [3]. I believe the authors should clearly discuss the difference and additional novelty over these, preferably with experimental comparison.

[1] Nayman, N. et al., "Hardcore-nas: Hard constrained differentiable neural architecture search." ICML, 2021.

[2] Hong, D. et al., “Enabling hard constraints in differentiable neural network and accelerator co-exploration”, DAC, 2022.

[3] Hu, Y. et al., "TF-NAS: Rethinking Three Search Freedoms of Latency-Constrained Differentiable Neural Architecture Search", ECCV, 2020 

**[W2]** The experiments lack comparison under varying difficulty of the constraints. The authors mention “The constraints are chosen such that about 50% of the architectures in each search space meet them”, which seems to be a rather easy bar to meet. In my opinion, methods like this would be much more valuable and convincing when the constraints are sufficiently rigorous that handcrafting or sampling methods fail to find a good candidate. Can the authors evaluate their method under different levels of constraint difficulty (e.g., 25%, 10%, or 5% of architectures satisfying the constraints) to better assess its robustness and effectiveness?

**[W3]** Theoretical fallbacks when moving CGGD to architecture parameters should be discussed more. 
The convergence statement in Sec3.3 relies on CGGD’s guarantee when R >1 and shortest-path directions yield convergence to the feasibility region. However, when mapping from \alpha to a discrete architecture via Gumbel Softmax annealing and the local replace-one-edge construction of dir_C, the resulting dir_C could violate the assumptions underlying the original convergence guarantee. A formal theorem should strengthen the claim.

**[W4]** Hardware-aware evaluation relies on analytical metrics not device measurements.
For NATS-Bench, hardware metrics are computed analytically, as shown in Appendix A, rather than measured on devices or by learned LUT regressors for a target platform. This condition reduces deployment realism and could introduce bias.

**[W5]** There is a lack of explanation behind how the manually designed baseline architecture was found. When the authors say “best manually designed model”, how was this model obtained? Did it go through a systematic search process? Is it just based on expert intuition? Appendix F only lists the outcome of such handcrafting and does not explain the process. To make the evaluation results more convincing, the authors should detail this process clearly in the paper.

### Questions
In general, please refer to weaknesses section. below are some relatively minor ones.
1. It seems figure 2 is not explained anywhere. Does the red dashed line denote 100% accuracy? Or is it a handcrafted or manually designed baseline? 
2. Since Fig. 2 excludes infeasible runs, could the authors provide the fraction of runs that reach feasibility within the budget, as well as the epoch at which feasibility is first reached?
3. Could the authors provide the overhead per epoch of computation dir_C relative to vanilla DARTS-like training, especially for large candidate sets per edge? Since there seems to be no analysis for the overhead on computation dir_C, it could make the paper stronger.
4. In algorithm 1, the paper sums and normalizes directions from all violated constraints. What happens when there are conflicting constraints, such as a case where memory and FLOPs push different edges in opposite directions? Is there any regulation regarding multi-constraint interactions?

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
4

### Summary
This paper proposes CONNAS, a gradient-based hardware-aware NAS method that enforces resource constraints through constraint-guided gradient modification rather than loss regularization. Instead of adding differentiable hardware penalties, CONNAS projects architecture gradients toward the feasible region, avoiding hyperparameter tuning of penalty weights.

Evaluations on NATS-Bench (CIFAR-10/100, ImageNet16-120) and a real STM32 MCU deployment show that CONNAS consistently meets hardware limits while achieving accuracy close to or better than regularization-based methods (ProxylessNAS, FBNet, TAS). It is robust to hyperparameters and effective under tight constraints.

Overall: An effective approach to enforce hardware constraints directly in differentiable NAS. The idea is empirically validated and practical for real deployment scenarios.

### Strengths
1. Clear and well-organized presentation. The paper is written in a clean, logical manner, making both the motivation and method easy to follow.

2. Experimental results on multiple benchmarks and a real hardware deployment convincingly demonstrate the effectiveness and robustness of the proposed CONNAS framework.

3. The idea of modifying gradients directly to satisfy hardware constraints, rather than learning or regularizing them, is unusual, providing a new methodological angle for hardware-aware NAS.

### Weaknesses
1. Outdated baselines and limited impact relative to a mature research field. Most compared baselines (ProxylessNAS, FBNet, TAS) are from 2019–2020, while recent NAS frameworks or one-shot supernet approaches are missing. Given that hardware-aware NAS is now a mature area, the lack of modern baselines (e.g., SPOS variants, DiffNAS, or once-for-all models) somewhat limits the significance and contemporary relevance of the work.

2. Directly modifying the gradients, although rare, in principle seems less elegant than learning-based methods. The authors argue that learning-based approaches require tuning the weights of additional penalty terms, but the proposed gradient modification also introduces extra hyperparameters, such as the rescale factor that determines the strength of constraint enforcement. Therefore, to me, the learning-based methods appear more elegant and principled overall.

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CONNAS, a constraint-guided gradient modification framework for hardware-aware neural architecture search (NAS). Instead of introducing differentiable hardware regularizers or manually tuned weighting factors, CONNAS modifies the gradients with respect to architectural parameters to directly enforce hardware constraints such as FLOPs, parameter count, and memory usage. The approach is evaluated on NATS-Bench across multiple datasets (CIFAR-10, CIFAR-100, and ImageNet16-120).

### Strengths
1. Clear motivation and practical focus: The paper identifies a well-defined challenge in hardware-aware NAS, balancing accuracy with strict resource constraints, and proposes a solution that directly targets constraint satisfaction.

2. Readable and well-organized writing: The manuscript is well-structured, and the figures and tables are clear, making the technical content accessible without ambiguity.

3. Comparison with relevant baselines: The evaluation includes established gradient-based NAS methods (e.g., ProxylessNAS, FBNet, TAS) re-implemented under a consistent setup, allowing fair comparisons.

### Weaknesses
1. Limited experiments: The experiments are only conducted on NATS-Bench, and only small-scale datasets are evaluated. This makes the generality and scalability of the proposed method unclear.

2. Technical novelty: Based on the method description, the gradient update mechanism is primarily derived from CGGD (Van Baelen & Karsmakers, 2023). More clarification on what is newly introduced in applying this approach to NAS is needed to justify the novelty of the contribution.

3. Lack of real hardware evaluation: The so-called hardware-aware metrics are not measured on actual hardware. FLOPs and the number of parameters are theoretical estimates, and the peak memory usage is computed by summing the input and output feature map sizes for each layer, without considering real deployment environments or memory behaviors on target devices.

4. Typos: “seperable” → should be “separable”

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
