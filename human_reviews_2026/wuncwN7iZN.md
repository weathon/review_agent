# Flow Actor-Critic for Offline Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
The dataset distributions in offline reinforcement learning (RL) often exhibit complex and multi-modal distributions, necessitating expressive policies to capture such distributions beyond widely-used Gaussian policies. To handle such complex and multi-modal datasets, in this paper, we propose Flow Actor-Critic, a new actor-critic method for offline RL, based on recent flow policies. The proposed method not only uses the flow model for actor as in previous flow policies but also exploits the expressive flow model for conservative critic acquisition to prevent Q-value explosion in out-of-data regions. To this end, we propose a new form of critic regularizer based on the flow behavior proxy model obtained as a byproduct of flow-based actor design. Leveraging the flow model in this joint way, we achieve new state-of-the-art performance for test datasets of offline RL including the D4RL and recent OGBench benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Flow Actor-Critic, a new actor-critic method for offline RL, based on recent flow policies.

### Strengths
This paper proposes a new critic regularizer to achieve accurate Q-value estimation under OOD conditions, and provides extensive experimental validation. The paper is overall well-organized.

### Weaknesses
1. In the Introduction, the paper mentions that diffusion policies make policy optimization computationally heavy. How does the proposed method's computational complexity compare to that of diffusion-based policies? It is suggested that the author compares the hardware efficiency of the proposed method with existing methods using metrics such as wall-clock throughput, FLOPs, and VRAM usage.
2. In the Introduction, it is unclear why the highly expressive flow behavior model and accompanying density can solve the OOD problem.
3. The paper lacks theoretical proof for why the proposed method provides more accurate Q-value estimation under OOD conditions compared to existing methods.
4. The authors compare many baselines, but fewer methods from 2024 and 2025 are included. It is suggested to compare with more recent methods published in the last two years.
5. From Table 2 in the main text, the proposed method's advantages are not evident, as in many cases, its performance is worse than the baseline. More experimental results showing the advantages of the proposed method should be added or moved from the appendix to the main text.
6. It is suggested to include Q estimate results in the main text, comparing the Q-values of the proposed method with the baseline methods and the true Q-value to demonstrate the superiority of the proposed method.

### Questions
1. In the Introduction, the paper mentions that diffusion policies make policy optimization computationally heavy. How does the proposed method's computational complexity compare to that of diffusion-based policies? It is suggested that the author compares the hardware efficiency of the proposed method with existing methods using metrics such as wall-clock throughput, FLOPs, and VRAM usage.
2. In the Introduction, it is unclear why the highly expressive flow behavior model and accompanying density can solve the OOD problem.
3. The paper lacks theoretical proof for why the proposed method provides more accurate Q-value estimation under OOD conditions compared to existing methods.
4. The authors compare many baselines, but fewer methods from 2024 and 2025 are included. It is suggested to compare with more recent methods published in the last two years.
5. From Table 2 in the main text, the proposed method's advantages are not evident, as in many cases, its performance is worse than the baseline. More experimental results showing the advantages of the proposed method should be added or moved from the appendix to the main text.
6. It is suggested to include Q estimate results in the main text, comparing the Q-values of the proposed method with the baseline methods and the true Q-value to demonstrate the superiority of the proposed method.

### Soundness
2

### Presentation
3

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
The paper introduces Flow Actor-Critic (FAC) for offline reinforcement learning. The key idea is to pair a flow-based behavior proxy with a one-step flow actor and to use the proxy’s density to tell when an action looks in-distribution versus out-of-distribution. The critic is then penalized only in regions where the behavior density is low, reducing the usual overestimation on OOD actions without dampening values where the data actually supports them. The actor, meanwhile, is trained to maximize Q while staying close to the behavior proxy, which helps with stability and makes good use of the expressiveness of flows without resorting to costly multi-step sampling. Empirically, FAC delivers strong results across large offline RL suites (OGBench and D4RL), beating or matching strong Gaussian, diffusion, and prior flow baselines. The ablations support the main claims: when candidate action sets get larger (which typically increases OOD risk), FAC remains robust while variants without the density-aware critic degrade.

### Strengths
The paper is clearly written and well structured. The motivation is crisp, the method is introduced in a logical sequence, and the experiments are laid out so the key results are easy to grasp before diving into details. 

The core idea is clean and elegant. By using the behavior model’s density to decide when to be conservative, the critic only gets pushed down where actions look out-of-distribution and is left alone where the data provide strong support. Coupling this with a one-step flow actor keeps the approach expressive yet practical to optimize.

The empirical results are very strong. FAC matches or beats strong Gaussian, diffusion, and prior flow baselines across broad offline RL suites (OGBench and D4RL). The improvements are consistent rather than cherry-picked, and the gains on OGBench are clearly significant.

The ablation study is very insightful as well. "Effect of the critic penalization" section clearly shows why the density-aware critic matters. Variants without it degrade as OOD risk increases, while sweeps over key hyperparameters demonstrate robustness.

### Weaknesses
I don’t see any major weaknesses. The only area that feels under-explored is the threshold design used to decide when the behavior density is “low.” The paper offers two options—a dataset-wide constant and a batch-adaptive threshold—but stops short of examining how this choice affects robustness across tasks. It would strengthen the work to add a small, focused study on the threshold itself: for example, trying simple variants like scaling the base threshold by a weight; using per-batch quantile thresholds instead of a fixed cutoff; or annealing the threshold during training.

### Questions
This method relies on a learned behavior density to decide where to be conservative. What happens if that density is hard to estimate well, for example, the dataset is multi-modal with rare but good actions, or high-dimensional with sparse coverage? In those cases, could FAC wrongly mark good actions as low-density and hurt performance?

### Soundness
4

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
This paper proposes Flow Actor-Critic (FAC), a new offline reinforcement learning algorithm that leverages flow-based generative models to jointly regularize the actor and penalize the critic. Unlike previous flow policy methods that only used flow models for action sampling, FAC fully exploits the learned flow behavior proxy policy to estimate tractable behavior densities. These densities are then used to (1) identify out-of-distribution (OOD) regions for critic penalization and (2) constrain the actor through a one-step flow mapping toward the behavior support. The authors derive the modified Bellman operator under this framework and provide extensive empirical evaluation on OGBench and D4RL, demonstrating improved or state-of-the-art performance compared to Gaussian- and diffusion-based baselines.

### Strengths
Originality: The paper introduces a novel and coherent idea — using the same flow model both for actor regularization and critic penalization. This dual use of flow behavior density provides a principled way to directly detect OOD regions, addressing a key challenge in offline RL.

Technical quality: The method is conceptually sound and the derivation of the flow-based critic operator is clear. The integration of density-weighted Q penalization and flow-regularized actor optimization is well motivated.

Empirical validation: Experiments are extensive, covering 50 OGBench tasks and 23 D4RL tasks, with comparisons to strong baselines such as ReBRAC, IQL, CQL, IDQL, CAC, and FQL. FAC consistently achieves top or near-top results.

Significance: The approach bridges expressive policy modeling (via flows) and conservative value estimation in a unified framework. It represents a meaningful step forward in making expressive policies practically viable for offline RL.

### Weaknesses
Incremental over FQL: Conceptually, the work extends FQL by reusing the flow density for critic penalization rather than introducing an entirely new framework. Although the empirical gains are notable, the conceptual advance may be viewed as incremental.

Computational cost: Flow-based models are typically heavier than Gaussian or VAE-based policies, but the paper does not discuss computational trade-offs.

### Questions
How sensitive is FAC to the choice of ε and to inaccuracies in the flow-based density estimate? Could an adaptive or uncertainty-aware threshold improve robustness?

How does FAC compare computationally to diffusion-based methods (e.g., CAC, IDQL) and multi-step flow policies (e.g., IFQL)?

Can the authors provide quantitative results showing the quality of their flow density estimates versus ground-truth or surrogate densities?

How does FAC behave on datasets with limited coverage or severe distributional shift, where even flow models may fail to represent $\beta(a|s)$?

Could FAC be extended to online or offline-to-online fine-tuning, where exploration or policy improvement beyond the dataset is needed?

### Soundness
2

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
This paper proposes FAC (Flow Actor-Critic), a new offline RL algorithm that integrates flow-based policies into the actor-critic framework. The motivation is that flow-based networks have much more expressibility than the Gaussian-based counterparts. The proposed method achieves state-of-the-art performance on the D4RL and OGBench benchmarks.

### Strengths
- The preliminary experiments strongly show that flow-matching can better estimate the density of the behavior policy.
- The proposed method is intuitive and seems to be a nice approach to incorporate flow-matching to offline RL.
- The proposed method shows strong performance not only on the D4RL benchmarks but also on the more recent OGBench benchmarks.

### Weaknesses
- Since the proposed method relies heavily on the accuracy of the density estimation from flow matching, there is some possibility that the proposed method may not scale well to higher-dimensional environments like pixel-based ones.
- Importantly, the proposed method introduces many new hyperparameters (alpha, lambda, clipped double Q-learning, and epsilon), and those hyperparameters are tuned for each task. This is not a fair comparison with the baselines. For example, CQL and IQL use the same hyperparameter per domain (Mujoco, Adroit, ...).

### Questions
- In the experiments of Figure 1, how does flow matching work when T=50? 
- In Equation (8), there can be other designs for giving more weight to outlier actions, for example, exponentially increasing the weight for those actions. Could the authors explain why they chose this specific formulation?
- In L1261, the paper notes they chose the epsilon scheme according to the state coverage and the action diversity. Can the authors provide a clear definition of what the two terms mean?

### Soundness
3

### Presentation
3

### Contribution
3
