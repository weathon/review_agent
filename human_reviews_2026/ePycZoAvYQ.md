# Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Offline goal-conditioned reinforcement learning remains challenging for long-horizon tasks. While hierarchical approaches attempt to address this by decomposing tasks into high-level subgoals, most existing methods rely on two-level architectures with separate networks, leading to fundamental limitations: they generate only a single intermediate subgoal, leading the low-level policy to act without awareness of the final goal when misled by erroneous subgoals, and prevent end-to-end optimization due to separate training objectives. We discover a novel solution to these challenges through chain-of-thought reasoning from large language models. Building on this insight, we introduce the Chain-of-Goals Hierarchical Policy (CoGHP), a new framework that reformulates hierarchical control as autoregressive sequence generation within a single unified architecture. CoGHP generates a sequence of latent subgoals and the primitive action in a single forward pass, where each subgoal acts as a "reasoning token" encoding intermediate decision-making. To implement this chain-of-thought approach in hierarchical RL, we pioneer the use of the MLP-Mixer architecture. This design enables efficient cross-token communication through simple feedforward operations and captures consistent structural relationships essential for hierarchical reasoning. Experimental results on challenging navigation and manipulation benchmarks demonstrate that CoGHP consistently outperforms strong baselines, demonstrating its effectiveness for long-horizon offline control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Chain-of-Goals Hierarchical Policy (CoGHP), a framework for offline goal-conditioned RL. Instead of using separate high- and low-level networks, CoGHP generates multiple latent subgoals and the final action within a single forward pass. In particular, it adopts an MLP-Mixer backbone and demonstrates its advantage over transformer backbone. Experiments on navigation and manipulation benchmarks show that CoGHP consistently outperforms strong baselines like HIQL, especially on long-horizon tasks.

### Strengths
1. Reformulating hierarchical RL as autoregressive sequence generation within a single unified network is a neat and interesting idea that simplifies architecture design and enables end-to-end optimization.
2. The proposed CoGHP consistently outperforms strong baselines across both navigation and manipulation tasks. 
3. The finding that an MLP-Mixer backbone outperforms the commonly used Transformer architecture is interesting.

### Weaknesses
1. The connection with chain-of-thought reasoning from LLMs is weak and a stretch. It is not convincing to consider the subgoal token as reasoning token. Framing offline RL as a sequence modeling problem has a well established literature. It is more appropriate to discuss the connection with this line of work such as DecisionTransformer and Trajectory Transformer.
2. The authors argue that generating the subgoal sequence in a reverse order is better. However, there is no corresponding ablation study to support that claim. Also, ideally, the subgoal sequence should be optimized jointly. In this sense, alternative formulations such as bidirectional generation or iterative refinement (e.g., diffusion-based planning) could be more principled.
3. The authors claim that the framework can handle other forms of subgoal representation. However, this generality seems non-trivial. Incorporating abstract or semantic subgoals would likely require additional modalities or dataset features. The authors should further elaborate on the additional modifications or assumptions needed to make such accommodations.

### Questions
1. The empirical results clearly show that the MLP-Mixer backbone outperforms the Transformer variant. Could the authors further elaborate on why this is the case? Which aspects of the MLP-Mixer architecture contribute to this advantage? In particular, could the authors clarify what is meant by “stepwise procedural consistency", and why is this property important for hierarchical RL in the proposed framework? 
2. Does the proposed framework reduce to HIQL when the planning horizon is one? An ablation on the number of predicted subgoals (planning horizon) would help clarify how much of the performance gain arises from multi-step subgoal prediction versus the unified architecture.

### Soundness
3

### Presentation
2

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
This paper introduces Chain-of-Goals Hierarchical Policy (CoGHP), a method for offline goal-conditioned reinforcement learning that the authors mention draws inspiration from chain-of-thought reasoning in large language models to generate sequences of latent subgoals and actions using a unified MLP-Mixer architecture. The authors claim that the approach addresses limitations in traditional two-level hierarchical RL by enabling end-to-end training and multiple subgoal generation in reverse order (farthest to nearest). The main contributions include adapting MLP-Mixer for RL sequence modeling with a causal mixer, using advantage-weighted regression with a shared value function, and demonstrating performance gains on OGBench navigation (pointmaze, antmaze) and manipulation (cube, scene) tasks.

### Strengths
CoGHP demonstrates solid empirical performance on long-horizon tasks vs HIQL, highlighting benefits of multi-subgoal generation for navigation. The unified MLP-Mixer architecture enables end-to-end training, reducing the fragmentation in separate network methods like HIQL, and ablations confirm the causal mixer's role in complex settings.

### Weaknesses
1. It seems like the chain-of-thought inspiration is superficial rather than intuitive or practical: latent subgoals are opaque embeddings, not explicit reasoning steps, making the LLM analogy more rhetorical than substantive. The LLM analogy is conceptually motivating but technically superficial. The real contributions are (a) unified autoregressive generation and (b) end-to-end training. Also, there is no ablation to test forward subgoal generation, leaving the reverse-order claim unverified. 

2. The baselines are limited to OGBench standards, which might question robustness to other domains. 

3. The limitations like distribution shift vulnerability are acknowledged but untested, with no OOD experiments despite offline RL's emphasis on generalization. 

4. Computational overhead (training/inference time) is unreported, hindering practical assessment against baselines.

### Questions
1. Why generate subgoals from farthest to nearest? Can you provide an ablation comparing this to nearest-to-farthest ordering on antmaze-giant and cube-triple to justify the design.

2. How does CoGHP handle OOD goals? Can you provide evaluations on success rates on goals outside the dataset distribution, as offline RL often faces unseen targets.

3. Please provide comparisons of runtime and parameters to HIQL: Report training time, inference latency, and FLOPs for fair efficiency claims.​

4. Does CoGHP generalize to diverse robotic manipulation tasks beyond the curated OGBench setup? Like MetaWorld Benchmark (Robotics).

### Soundness
2

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
3

### Summary
The paper proposes Chain-of-Goals Hierarchical Policy, a unified policy that generates a short sequence of latent subgoals followed by the final action in a single forward pass. The key idea is to treat subgoals as reasoning tokens, produced autoregressively so that later predictions condition on earlier ones while remaining aware of the original goal. The architecture uses an MLP-Mixer backbone with a learnable causal token mixer, trained end to end with a shared goal-conditioned value function and advantage-weighted objectives. Experiments on OGBench navigation and manipulation tasks report sizable gains over strong offline goal-conditioned RL baselines, with ablations suggesting benefits from the Mixer backbone and the causal mixer.

### Strengths
* The paper targets a pain point in offline goal-conditioned RL for long horizons and argues for a cohesive alternative to two-level hierarchies. Formulating hierarchical control as autoregressive subgoal generation inside one network is conceptually neat and practically appealing, since it preserves access to the final goal and allows gradients to flow through all stages.

* Empirically, the method shows strong results across diverse domains. Notably, it improves success on difficult giant mazes and complex manipulation sequences where multiple intermediate decisions matter. These gains are consistent with the claim that multi-step intermediate guidance helps long-horizon tasks.

* The ablations are helpful. Replacing the Mixer with a Transformer hurts on the hardest tasks, and removing the causal mixer degrades performance further, which supports the specific architectural choices rather than attributing wins to generic capacity.

### Weaknesses
* The novelty is partly architectural refactoring and framing. The chain-of-thought analogy is evocative, but the subgoals are supervised by fixed k-step future states from trajectories. This is closer to structured imitation with value-based weighting than to learned reasoning.
The methodology introduces potential train–test mismatch. Training uses teacher forcing with ground-truth subgoals, while inference relies on the model’s own subgoal predictions. The paper acknowledges teacher forcing but does not quantify error accumulation or compare against scheduled sampling or consistency regularizers designed for autoregressive rollouts.

* Some comparisons and analyses feel underdeveloped. The claim that Transformers are less suitable is asserted and supported by a single ablation, but hyperparameter parity and capacity normalization are not deeply probed. It is also unclear how sensitive results are to the choice of k for subgoal extraction, to the advantage temperature, and to the exact weighting between subgoal and action losses. The experiments report strong headline numbers, yet each environment evaluates only five predefined state–goal pairs and success rates are averaged across eight seeds, which makes the statistical picture somewhat narrow. More per-task breakdowns or success-vs-horizon plots would improve confidence.

* On the representation side, most subgoals are encoded future states. The paper briefly notes that other subgoal types would fit, but the only concrete visual evidence comes from decoding to coordinates for antmaze. Demonstrations of learned abstract subgoals or skills, or at least richer visualizations across tasks, would better support the generality claim.

### Questions
* How robust is performance to the choice of H and the spacing k used to sample supervision subgoals from trajectories? A sensitivity sweep that varies H and k together would help establish whether improvements persist beyond the selected settings.

* Can the authors quantify exposure bias from teacher forcing? For example, report success when the policy is rolled out with its own predicted subgoals but trained without teacher forcing, or include scheduled sampling. A plot of success vs rollout depth of predicted subgoals would clarify compounding error.

* What is the effect of the causal mixer relative to simpler positional encodings or strictly triangular masking without learnable mixing? An ablation that replaces the causal mixer with fixed lower-triangular averaging could isolate the benefit of learnability.

* How fair and capacity-matched are the Transformer baselines? Please provide layer counts, parameter totals, token dimensions, and training curves for Mixer vs Transformer across tasks to rule out under-tuning.
   
* Do results hold when evaluating on larger sets of randomly sampled state–goal pairs and under distribution shift in goals? Reporting confidence intervals across many goals, as well as success vs geodesic distance to goal, would strengthen the empirical case.
   
* Can the method operate with non-state subgoals, such as latent skills or semantic waypoints, and still retain advantages? A small-scale experiment with learned discrete subgoals would substantiate the generality claim.

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
4

### Summary
This work introduces the framework Chain-of-Goals Hierarchical Policy (CoGHP). Inspired by chain-of-through reasoning in large language models, this framework formulates hierarchical control problems with autoregressive sequence. In a single forward pass, the framework generates a sequence of latent subgoals each encoding an intermediate decision-making, followed by the primitive action. 
The work uses MLP-Mixer backbone with a learnable causal token mixer to capture consistent structural relationships. Experiments on long-horizon offline control tasks reflects non-trivial improvements over strong baselines. Ablation study shows that the benefits is from the mixer backbone.

### Strengths
* Originality: The idea of taking inspiration from Chain-of-Thought to reformulate HRL tasks with autoregressive sequence within a single unified network is novel. The work also shows that this formulation allows for end-to-end optimization, which is practical and efficient. 

* Significance: The experiments show that an MLP-Mixer backbone consistently performs better than stronger baselines across various tasks that uses Transformer architecture. The experiment benchmark is also complex enough where it is often necessary to have multiple intermediate decisions. These results demonstrate that multi-step intermediate guidance helps long-horizon tasks. 

* Quality and clarity: The ablations shows that the MLP-Mixer and the causal mixer is crucial, as replacing them decreases performance.

### Weaknesses
* Using the term chain-of-thought is a little misleading because while it is valid as an inspiration, in practice it doesn't seem to be directly influencing the architecture. The subgoals are not explicit interpretable reasonings like their counterparts in LLMs, rather they are embeddings supervised by fixed step future states. The work could perhaps make it more clear that their contribution is focused on proposing a unified autoregressive generation and highlight that this allows for end-to-end optimization, and discuss more on the relationship to the line of work that frame offline RL as sequence modeling problem. 

* The ablation study feels under-explored: there could be more context discussing things like hyperparameter parity, how selecting different k affects the extracted subgoals. In general, this paper could benefit from more quantitative results that focuses on different tasks. The work could also benefit from adding an OOD experiments and discuss generalization as part of offline RL's focus. The claim that generating subgoals in reverse order also feels under-explored and lacks a clear and concrete supporting result.

### Questions
1. The reason behind generating subgoal in the order of farthest to nearest is not very clear. Is it really better compared to generating it the other way around, or is it just easier for the architecture?

### Soundness
3

### Presentation
3

### Contribution
2
