# Adaptive Scaling of Policy Constraints for Offline Reinforcement Learning

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6, 6

## Abstract
Offline reinforcement learning (RL) enables learning effective policies from fixed datasets without any environment interaction. Existing methods typically employ policy constraints to mitigate the distribution shift encountered during offline RL training. However, because the scale of the constraints varies across tasks and datasets of differing quality, existing methods must meticulously tune hyperparameters to match each dataset, which is time-consuming and often impractical. To bridge this gap, we propose Adaptive Scaling of Policy Constraints (ASPC), a second-order differentiable framework that automatically adjusts the scale of policy constraints during training. We theoretically analyze its performance improvement guarantee. In experiments on 39 datasets across four D4RL domains, ASPC using a single hyperparameter configuration outperforms other adaptive constraint methods and state-of-the-art offline RL algorithms that require per-dataset tuning, achieving an average 35\% improvement in normalized performance over the baseline. Moreover, ASPC consistently yields additional gains when integrated with a variety of existing offline RL algorithms, demonstrating its broad generality.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to balance the typical RL-BC trade-off in offline RL methods. Specifically, common offline RL objective can be abstracted as  \$\alpha L_{rl} + L_{bc}\$, forming a balance between $L_{rl}$ and $L_{bc}$, controlled by $\alpha$. This paper proposed a bi-level objective, which formulates $\alpha$ as learnable parameters to adaptively control the update rate of RL and BC objective, and thus avoiding any of them to dominate the other one, thus balancing these two terms. The idea and motivation of this paper is straightforward and simple. The evaluations on D4RL benchmarks demonstrate the effectiveness of the auto-tuned $\alpha$, based on its TD3+BC variants.

### Strengths
1. The idea is simple and straightforward.
2. This paper well-written and is easy to follow.
3. The proposed method is easy to be implemented.

### Weaknesses
1 `Novelty and Significance`
Automatically tuning the conservatism strength in offline RL is not a new research direction. Several existing methods have already addressed this problem in ways that are both simple and effective. Therefore, I would not consider the novelty or significance of this paper to be its main strength.

2 `Versatility of the Proposed Method`
The proposed approach is developed specifically on top of TD3+BC, which is now a relatively dated baseline. In fact, most offline RL methods can be abstracted as an RL + BC objective, including more recent diffusion- or flow-based approaches. However, the improvements in this paper are restricted to TD3+BC, and the additional loss function also appears tied to this specific algorithm, limiting its generalizability. Considering the already limited novelty, it would strengthen the contribution to derive a more versatile framework that can generalize to a broader range of offline RL approaches, rather than being confined to one.

3 `Dated Benchmark`
The D4RL benchmark has long been saturated, making it difficult to draw meaningful distinctions between methods. To strengthen the empirical soundness of the paper, it would be beneficial to include evaluations on more challenging benchmarks, such as OGBench.

### Questions
Please see weakness for details.

### Soundness
2

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
This paper presents ASPC, a novel adaptive tuning strategy for offline reinforcement learning via a bi-level parametric optimization framework. ASPC mainly focuses on policy-constraining offline RL methods that mitigate the distributional mismatch in offline RL by constraining a learned policy to stay close to the behavior policy. Specifically, ASPC adopts a meta learning approach for the adaptive scaling; the policy with the standard RL-BC-weighted objective is trained in the inner loop, and the regularization parameter $\alpha$ is updated by penalizing large drifts in both objectives in the outer loop. Theoretical proofs support the design of combined objectives for adapting $\alpha$, while extensive experiments demonstrate ASPC's outstanding performance across diverse tasks and datasets in the D4RL benchmark.

### Strengths
- The paper is clearly written to support the novelty of ASPC.
- The authors provide an in-depth view of policy constraints baselines in offline RL, which helps in understanding the contribution of ASPC.
- Extensive experiments support the superiority of ASPC across heterogeneous environments and datasets in the D4RL benchmark, including MuJoCo, Maze2d, AntMaze, and Adroit.
- Meticulously designed ablation studies resolve potential questions regarding ASPC.
- Comparing with SOTA baselines in adaptive regularization (e.g., A2PR, wPC) strengthens the unique contribution of this paper.

### Weaknesses
- The contribution of ASPC is limited to only policy-constraining offline RL baselines. While the baselines used in this paper (e.g., TD3+BC, IQL, and CQL) still maintain strong performance across popular offline RL benchmarks, other branches in offline RL have suggested numerous algorithms that outperform such baselines. For instance, offline model-based RL [1,2] or offline RL with generative models [3,4] prove remarkable performance over denoted baselines in the D4RL benchmark. If ASPC can be extended to other methods that employ the RL-BC-weighted objective, the novelty of this paper would be further reinforced.

[1] Sun Y, Zhang J, Jia C, Lin H, Ye J, Yu Y. Model-Bellman inconsistency for model-based offline reinforcement learning. ICML 2023.

[2] Rigter M, Lacerda B, Hawes N. Rambo-rl: Robust adversarial model-based offline reinforcement learning. NeurIPS 2022.

[3] Hansen-Estruch P, Kostrikov I, Janner M, Kuba JG, Levine S. Idql: Implicit q-learning as an actor-critic method with diffusion policies. arXiv 2023.

[4] Wang Z, Hunt JJ, Zhou M. Diffusion policies as an expressive policy class for offline reinforcement learning. arXiv 2022.

### Questions
- Why is it hard to extensively fine-tune the scale of the regularization constraint (L48)? Could you provide more intuitive examples or implications?
- How the learning rate of $\alpha$ affects the performance across experiments? In Figure 6-(a), both wPC and ASPC show large differences in AntMaze and Adroit by adding an extra layer and normalization. I wonder how the learning rate, which determines the magnitude of a step taken by the gradient descent, affects the performance of ASPC in those tasks.

### Soundness
4

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
4

### Summary
This paper proposes an adaptive, dynamically adjusted hyperparameter setting scheme using a meta-learning approach. The paper also derives a lower bound on policy improvement and constructs corresponding loss functions to tune hyperparameters accordingly, ensuring stable performance gains. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. Introducing a meta-learning framework to hyperparameter tuning in offline RL is novel. To identify a suitable alpha, the authors obtain new parameters via an inner loss and evaluate their performance using an outer loss, enabling adaptive adjustment of this hyperparameter.

2. The experimental evaluation is extensive and convincingly supports the method’s effectiveness. The paper reports results on 39 tasks, showing performance advantages. Moreover, the trajectory of alpha during training aligns with intuition.

### Weaknesses
1. According to the algorithm's description and Equations (6) and (7), the outer losses L1 and L2 evaluate the new, updated policy π_θ˜ using the Q-function from before the policy's "inner update." In Actor-Critic frameworks, the accuracy of a Q-function is tightly coupled with the policy it evaluates. Using a "stale" Q-function relative to the new policy introduces bias, making the optimization target of the meta-objective itself imprecise. This weakens the reliability of the theoretical foundation for adaptively tuning the hyperparameter α via meta-learning.
2. In Appendix A.1, the derivation in Equation (15) (lines 681-682) does not strictly follow the rules of inequality manipulation. The authors' application of the Cauchy-Schwarz inequality after the reverse triangle inequality (|a+b| ≥ ||a|-|b||) is not mathematically sound, as the substitution step within the absolute value operation is invalid. This error invalidates the derived lower bound for ∆L_BC and thus undermines the foundation of the proof for the mutual constraint relationship between ∆L_BC and (∆Q)².
3. The paper's central theoretical contribution—the single-step performance guarantee (Theorem 4.4)—claims that the algorithm ensures monotonic policy improvement. The theoretical precondition for this guarantee is that the Q-value gain ∆Q must be no less than a complex penalty term Φ (as per Proposition 4.3). However, the algorithm's actual implementation optimizes an outer loss (L_outer) that applies soft penalties via L2 and L3 to ∆Q and an approximation of Φ, rather than enforcing a hard constraint. Minimizing a weighted sum of these penalties does not mathematically guarantee that the hard condition ∆Q ≥ Φ will be met at every update step. Consequently, while the algorithm may tend towards stable updates, the claim that it guarantees monotonic performance improvement is an overstatement that is not rigorously supported by the provided theory.

### Questions
see above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Adaptive Scaling of Policy Constraints (ASPC), a second-order differentiable framework for offline reinforcement learning that automatically adjusts policy constraint scales to avoid per-dataset hyperparameter tuning, and it outperforms baselines on different D4RL tasks.

### Strengths
1. ASPC automatically adjusts policy constraint scales via a second-order differentiable framework, eliminating the need for  hyperparameter weight tuning.
2. ASPC incurs only minimal computational overhead compared to previous methods while maintaining strong performance across diverse tasks and datasets.

### Weaknesses
1. Although no hyperparameter in the objective is needed, the user may still need to choose the update intervals between inner and outer updates.
2. The average performance improvement of ASPC mainly comes from the Maze2d tasks, while its performance still underperforms in AntMaze and Adroit compared to other baselines. 
3. More elaborations on the three outer updates loss should be provided.

### Questions
1. Will the choices of different outer update intervals differ from tasks?
2. The three different outer update objectives, L1, L2, and L3, show different influence on the performance degradation in different tasks shown in Figure 6. For example, L3 is critical for AntMaze while seems not valuable for Adroit. Can you provide more insights on this observation? Can we choose some of the objectives for specific task optimization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ASPC, a bi‑level, second‑order method that learns the policy‑constraint scale $\alpha$ in offline RL. The inner loop is TD3+BC; the outer loop updates $\alpha$ with a composite loss that encourages Q‑value improvement while penalising large Q and BC‑loss shifts. On 39 D4RL datasets with a single hyperparameter setting, ASPC attains the best overall average, with particularly strong gains on Maze2D and consistent performance on MuJoCo; AntMaze/Adroit results are competitive but not always the highest. Ablations show (i) dynamic $\alpha$ matters more than picking a single good fixed value, and (ii) both L2 and L3 regularisers contribute, in a domain‑dependent way. The approach is practical and targets a pressing source of brittleness in offline RL.

### Strengths
- Addresses a central practical problem with a clean bi‑level solution
- Broad, careful experiments across 39 datasets; single‑config results are compelling
- Good diagnostics: $\alpha$ learns to down‑weight RL on expert data and up‑weight it on noisy data
- Useful, although idealised, theory; ablations that clarify when each loss term helps; modest runtime overhead

### Weaknesses
- Theoretical guarantees depend on assumptions and a particular L3 construction; they guide design more than they certify behaviour in deep RL
- Performance is not uniformly best on AntMaze/Adroit; understanding and closing this gap would strengthen the story
- Method depends on a stronger critic; without it, ASPC can fail 
- Generality beyond TD3+BC is argued but not demonstrated empirically

### Questions
- Can you show ASPC plugged into a value‑based method (IQL/CQL) without major surgery, even on a subset of D4RL? A small table would help establish generality.
- How sensitive is performance to the exact form and scaling of L3? Can you share one plot per domain? 
- Which parts of the monotonic‑step proof most clearly break in deep‑RL practice? Is there an empirical check to show the intended effect holds? 
- The sketch in App. A.3 mentions an initialisation where the Q‑gradient dominates BC. How sensitive is ASPC to the initial 
$\alpha$ and to early‑stage critic noise? 
- L3 uses sup‑norms. Do rare, high‑error samples dominate $\alpha$ updates?

### Soundness
3

### Presentation
4

### Contribution
3
