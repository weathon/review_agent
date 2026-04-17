# Reinforcement Learning via Value Gradient Flow

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
We study behavior-regularized reinforcement learning (RL), where regularization toward a reference distribution (the dataset in offline RL or the base model in LLM RL finetuning) is essential to prevent value over-optimization caused by erroneous out-of-distribution extrapolation. Existing methods either rely on reparameterized policy gradient, which are difficult to scale to large generative models, or on reject sampling, which can be overly conservative when attempting to move beyond the behavior support. In this paper, we propose Value Gradient Flow (VGF), a scalable new paradigm for behavior-regularized RL. VGF casts behavior-regularized RL as an optimal transport problem that maps the reference distribution to the value-induced optimal policy distribution. We solve this transport problem via discrete gradient flow, where value gradients guide particles initialized from the reference distribution. Our analysis shows that VGF imposes regularization implicitly by controlling the transport budget. VGF eliminates explicit policy parameterization while remaining expressive and flexible, this enables adaptive test-time scaling by adjusting the transport budget. Extensive experiments demonstrate that VGF significantly outperforms prior methods, achieving state-of-the-art results on offline RL benchmarks (D4RL, OGBench) and LLM RL tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, an offline reinforcement learning framework is formulated as an optimal transport problem that transfers the actions induced by the behavior policy to those induced by the optimal policy.
This transport process is modeled as a particle-based gradient flow, and the resulting action update rule is analogous to Stein Variational Gradient Descent (SVGD).

In existing methods, a regularization term is typically introduced into the objective function, and its coefficient must be carefully tuned to balance the trade-off between maximizing the expected return and enforcing behavior regularization.
In contrast, the proposed method controls the degree of regularization through the transport budget. Specifically, the number of steps and the learning rate.

The proposed approach is evaluated on continuous control tasks in D4RL and OGBench, as well as on LLM fine-tuning tasks using TL;DR and Anthropic-HH datasets.
The experimental results demonstrate that the proposed method outperforms existing approaches on these benchmark tasks.
In addition, a toy example illustrates that the method can successfully model a multimodal action distribution.

### Strengths
- Formulating offline RL as an optimal transport problem is both interesting and appears to be novel.

- The experimental results clearly show the advantages of the proposed method in both continuous control and LLM fine-tuning tasks.

Although the idea of using the Wasserstein distance as a form of regularization has been explored in prior offline RL studies, I am not aware of previous work that learns a gradient flow from the behavior policy to the optimal policy.
In this regard, the proposed method is novel and conceptually appealing.

### Weaknesses
- The presentation is somewhat unclear in several places and would benefit from improvement.

- The paper lacks any discussion of computational cost.

The claim that “VGF removes the need to balance the optimization conflict between reward maximization and deviation penalties” is somewhat misleading. While it is true that the proposed method eliminates the need to manually tune the coefficient of an explicit regularization term, it still requires tuning the transport budget, which effectively serves a similar role. Thus, balancing between reward maximization and deviation control remains necessary. Since similar statements appear multiple times throughout the paper, the authors should carefully revise these passages to avoid confusion.

The action update rule in Equation (7) is analogous to SVGD. However, although Liu & Wang (2016) and Liu (2017) are cited, the term SVGD itself does not appear explicitly. Explicitly mentioning SVGD would make the connection clearer and help readers understand the proposed algorithm more easily.

Another potential weakness of the proposed method lies in its computational cost. Action generation using the flow-based behavior policy introduces additional overhead, and the gradient flow from the behavior policy to the optimal policy also incurs computational expense.
I expect that the proposed method would be significantly slower than TD3+BC and IQL, and possibly even slower than Diffusion-QL.
However, the JAX-based implementation may help alleviate some of these costs. Discussing these computational limitations would strengthen the overall presentation and contextualize the method’s contributions.

### Questions
- Please comment on the statement regarding the removal of the need to balance between reward maximization and deviation penalties. If you agree with my observation, please revise the paper accordingly.

- Why is SVGD not explicitly mentioned in the paper? Please elaborate on the connection between the proposed method and SVGD.

- How much training time does the proposed method require in practice? Please provide approximate wall-clock times and compare them against the baseline methods.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript proposes the Value Gradient Flow (VGF) for  behavior-regularized reinforcement learning. VGF formulates behavior-regularized RL as an optimal transport problem, guiding particles from the reference distribution (offline data or supervised fine-tuned policy) toward high-value regions via discrete gradient flow. The VGF framework demonstrates several merits including avoiding unnecessary restrictions on optimization problem, no need explicit policy parameterization and so on. Experiments on D4RL and other benchmarks demonstrate the superiority of the proposed method.

### Strengths
* It is appreciated that the authors combine the idea of optimal transport, a classic method in optimization, with RL situations. 

* The first strength of this manuscript is that it avoid optimization conflict between reward maximization and distribution penalty, mitigates the "deadly triad" problem in actor-critic algorithms, and enables more stable training.

* The idea of applying the latent variable in generative modeling is appealing. It combines the good merits in generative modeling (dimension reduction and re-parametrization), and applies well in exploration and exploitation.

### Weaknesses
1. We wonder whether the performance of VGF is limited by the reference distribution. When the reference distribution is skewed towards to suboptimal behaviors, how is the robustness of VGF?

2. Particle-based gradient flow solving requires maintaining multiple particles and function calculations, do the authors consider such cost in practice? Especially in high dimension scenarios

3. From the experiments, it seems the VGF still inferior to Guassian Policy or Diffusion policy in some D4RL, it is suggsted the authors provide additional explanations on this issue.

4. Although penalty coefficient tuning is unnecessary, key hyper-parameters such as training steps and temperature still require manual task-specific adjustment, lacking adaptive selection mechanisms. We encourage the authors do more trial in this field.

### Questions
See the weakness above

### Soundness
4

### Presentation
3

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
This paper proposes Value Gradient Flow, which solves the behavior regularized RL by initializing particles using the reference distribution and transporting the particles towards the target distribution, following the gradient of the value function. Compared to a vanilla value-guided MCMC, VGF also incorporates several other ingredients and techniques, such as the best-of-N refinement during evaluation and kernel-based affinity metrics. The evaluation of VGF is conducted on both offline RL and RLHF tasks. In Offline RL, VGF is used both for training and evaluation, while in RLHF, VGF is only used for test-time generation. The key findings are that 1) with proper hyperparameter configuration, VGF outperforms the flow-based actor critic method both on D4RL and OGBench; 2) On TL;DR and Anthropic-HH datasets, VGF seems to be efficient for alignment and outperforms baseline methods, including PPO, DPO by a large margin.

### Strengths
1. Instead of using a flow model to generate samples towards the target distribution, VGF instead first trains a flow model to initialize particles following a reference distribution and afterwards transports them towards high-valued areas. This approach is novel. 

2. Besides, the VGF framework seems to unify several existing practices. For example, using zero transportation steps corresponds to best-of-N sampling. Conversely, as the number of transportation steps approaches infinity, the VGF process effectively recovers gradient-based sampling methods, such as Langevin dynamics.

### Weaknesses
1. VGF incurs a higher inference cost than traditional flow-based methods. This is because, in addition to the initial sample generation from the reference distribution, VGF requires a computationally intensive iterative process to transport and refine these particles toward high-density regions.

2. Furthermore, VGF appears highly sensitive to its hyperparameters and can demonstrate inconsistent performance trends across different tasks. Consequently, applying VGF to a new problem often requires extensive, task-specific tuning to achieve optimal results.

### Questions
1. In Table 3, is VGF using particles that the SFT model initializes? 

2. Could the authors provide a runtime analysis of both VGF and other diffusion-based methods?

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
This paper focuses on the area of behavior-regularized reinforcement learning, specifically within the contexts of offline reinforcement learning (RL) and reinforcement learning with human feedback (RLHF). In response to the challenges of training instability and the difficulty of hyperparameter tuning in existing explicit constraint algorithms, the authors propose the Value Gradient Flow. This approach integrates particle-based gradient flow with value functions, utilizing multi-step gradient guidance to iteratively direct the policy towards regions with higher value distributions. Experimental evaluations on offline RL and RLHF datasets, including D4RL and OGBench, demonstrate that this method outperforms existing algorithms.

### Strengths
1. The motivation of the paper is clear and straightforward. The writing is fluent and easy to understand, effectively explaining how Value Gradient Flow is integrated into the behavior-regularized reinforcement learning problem.

2. In my view, introducing Particle-based Gradient Flow into the offline reinforcement learning domain and using gradient-based guidance to iteratively shift the behavior cloning action outputs towards higher value regions is an innovative approach. The related experiments also validate the effectiveness of the proposed method.

### Weaknesses
1. I suggest that the authors conduct experimental evaluations on more complex environments within the D4RL dataset (e.g., Adroit) as well as on the visual input V-D4RL, to better validate the generalization of the proposed method.

2. In Algorithm 1, the authors present the Q-network learning via TD with the iterated $a^{L\_{train}}\_N$. However, could this lead to overly optimistic learning for the Q-network? In my opinion,  $a^{L_{train}}_N$ is prone to producing out-of-distribution (OOD) actions. What would the results be if the Q-network were first learned using methods like CQL or IQL, and then directly tested?

3. Although the authors emphasize that VGF significantly reduces training costs, the paper lacks quantitative analysis, including but not limited to training time, inference time, and memory usage. Including these details would provide a more comprehensive evaluation of the entire algorithm.

4. I recommend adding a column to each main experiment table to show the results when  $L\_{test} = 0$, which corresponds to the best-of-N sampling method. This would better highlight the improvement brought by VGF, rather than being influenced by ensemble methods like best-of-N sampling or other factors.

5. The citation for Diffusion-QL on line 376 is incorrect.

### Questions
Please refer to the "Weaknesses" section, I will raise my score if my concerns are addressed.

### Soundness
3

### Presentation
3

### Contribution
3
