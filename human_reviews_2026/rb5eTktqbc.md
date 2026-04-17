# Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
We introduce a sequence-conditioned critic for Soft Actor--Critic (SAC) that models trajectory context with a lightweight Transformer and trains on aggregated $N$-step targets. Unlike prior approaches that (i) score state--action pairs in isolation or (ii) rely on actor-side action chunking to handle long horizons, our method strengthens the critic itself by conditioning on short trajectory segments and integrating multi-step returns without the need of importance sampling (IS). The resulting sequence-aware value estimates capture the critical temporal structure for extended-horizon and sparse-reward problems. On multiple benchmarks, we further show that freezing critic parameters for several steps makes our update compatible with CrossQ's core idea, enabling stable training without a target network. Despite its simplicity, a 2-layer Transformer with $128$--$256$ hidden units and a maximum update-to-data ratio (UTD) of $1$, the approach consistently outperforms standard SAC and strong off-policy baselines, with particularly large gains on long-trajectory control. These results highlight the value of sequence modeling and $N$-step bootstrapping on the critic side for long-horizon reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This article extends the work in the literature [1] by introducing a sequence-conditioned critic for SAC that models trajectory context with a light-weight Transformer and trains on aggregated N-step returns. Empirical results on standard online RL benchmarks show that the proposed method sometimes outperforms other algorithms.


[1]Qiyang Li, Zhiyuan Zhou, and Sergey Levine.Reinforcement learning with action chunking.arXiv  preprintarXiv:2507.07969,2025.

### Strengths
1. The paper is overall well-organized and easy to follow. 
2. The proposed transformer critic is intuitive and is supported by several theoretical results.

### Weaknesses
1. The improvement of the algorithm in the experiments is not significant for a basic benchmark, Mujoco.
The authors claim that an action mask is needed for T–SAC and may degrade performance with high trajectory variability.
T-SAC is better than other methods in Meta-World tasks, however, there is higher trajectory variability in Meta-World tasks than that in Mujoco.
2. The authors of literature [1] claim that we might desire a final optimal Markovian policy, the exploration problem can be better tackled with non-Markovian and temporally extended skills, and that action chunking offers a very simple and convenient recipe for obtaining this. 
*However,* they did not provide any theoretical explanation for the aforementioned phenomenon. They also did not offer any intuitive examples to demonstrate the effectiveness of action chunking in MDP tasks. The same problems are shown in this paper.
3. I think the major contribution of this work is the proposed causal Transformer. Thus, in addition to the performance gain by inserting the causal Transformer into SAC, it would be good to discuss other metrics that can directly evaluate the "quality" or "informativeness" of representations learned in the causal Transformer. Such metrics might include Centered Kernel Alignment and Mutual Information Neural Estimation.
4. The contribution is ambiguous. An alternative interpretation of the benefit from action chunking and T-SAC is that the model is simply given a richer input feature set, which allows a powerful function approximator (like a deep neural network) to better fit a complex value or policy function. 
5. The new method should be compared with QC, which is proposed in [1].


*If my concerns and questions are all addressed, I will raise the score.*

### Questions
1. What is the subset of MDPs that have the property that the Transformer-based action chunking can be provably useful?
This is an important question. As you can see, the proposed method does not perform well in Mujoco.
2. Could you discuss the computational cost and scalability of the proposed method in detail, especially as the sequence length increases? 
3. Could you provide a more detailed analysis of how your Transformer-based action chunk gain was observed in your experiments? In particular, in addition to the comparison of rewards in the paper, is there an experiment that more intuitively demonstrates the significant improvement in sample efficiency?

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
This paper proposes T-SAC (Transformer-based Soft Actor-Critic), which enhances standard SAC by introducing a sequence-conditioned critic that models short trajectory segments via a lightweight Transformer and trains with aggregated N-step returns—without importance sampling. The approach enables long-horizon credit assignment while keeping the actor update one-step. A simple critic-freezing schedule further removes the need for target networks. Experiments on various benchmarks show consistent improvements over SAC, CrossQ, and other baselines, demonstrating that sequence modeling on the critic side can improve sample efficiency and stability in long-horizon control.

### Strengths
- The idea of chunking the critic rather than the actor is novel and conceptually clean, offering a new perspective on temporal abstraction in off-policy RL.

- The removal of importance sampling through prefix-conditioned targets is a practical and effective simplification.

- The paper combines Transformer critics, multi-horizon learning, and critic freezing in a coherent design that improves stability without target networks.

- Experiments are broad and rigorous, with thorough ablations verifying each design component.

- Writing is clear, structured, and technically sound.

### Weaknesses
- The novelty is largely architectural; theoretical justification is limited, and the benefits of critic-side chunking lack deeper analysis.

- All results use low-dimensional state inputs; no experiments on visual or partially observable tasks.

- Statistical significance of gains and computational cost comparisons are not deeply analyzed.

### Questions
1. How sensitive is T-SAC to the choice of N-step horizon and freezing interval?

2. Would the approach extend to visual or partially observable environments?

3. Could actor and critic share a Transformer backbone for further efficiency?

4. How does T-SAC compare computationally to CrossQ and TOP-ERL at scale?

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
This paper addresses the challenge of long-horizon credit assignment in standard step-based reinforcement learning. It introduces T-SAC, an algorithm that replaces the standard MLP critic in Soft Actor-Critic (SAC) with a lightweight Transformer. This new critic conditions on short sequences of actions from the replay buffer and is trained on N-step returns across multiple horizons without requiring importance sampling, using a gradient averaging scheme to ensure stability. On long-horizon and sparse-reward control benchmarks like Meta-World ML1 and Box-Pushing, T-SAC is shown to achieve superior sample efficiency and final performance compared to strong off-policy baselines.

### Strengths
1, T-SAC significantly outperforms strong off-policy baselines such as SAC and CrossQ on the Meta-World ML1 benchmark and the FANCYGYM Box-Pushing tasks. Its 58% success rate on the sparse-reward Box-Pushing variant is particularly notable, clearly demonstrating the effectiveness of its long-horizon reasoning capabilities.

2, While N-step returns are known to suffer from high variance as n increases, the proposed gradient averaging technique enables stable training for horizons as long as n=16. This challenges the conventional wisdom of using small n and provides a robust method for learning long-term dependencies.

3, The paper provides a clear justification for its architectural choices through rigorous ablation studies. It demonstrates the critical importance of the self-attention and causal mask components in the Transformer critic and shows its superiority over recurrent backbones like GRU and LSTM.

### Weaknesses
1, The performance of T-SAC is not uniform across all benchmarks. On Gymnasium MuJoCo, it performs worse than the standard SAC baseline on Hopper and Walker2d.The authors attribute this to the need for an "action mask" on tasks with "high trajectory variability", which points to a limitation in the method's generality.

2, The critic conditions on multi-step action sequences (at, ..., at+n-1) generated by the current, often random, policy. Early in training, these sequences are noisy and suboptimal, potentially introducing significant variance into the critic's learning target. This may slow down the initial convergence speed compared to standard critics that condition only on a single (s, a) pair. This effect could be particularly pronounced in dense-reward settings like mujoco, where the benefit of a clean, immediate one-step TD target is high.

3, The proposed critic-parameter freezing schedule (Sec 4.4), which enables training without a target network, is only demonstrated on locomotion tasks. The authors admit it is unstable in sparse-reward settings, where the paper's main results are achieved using conventional Polyak averaging. This makes the target-free contribution a secondary point with limited applicability.

### Questions
1, Could you please elaborate on the specific role and necessity of the "action mask" mentioned as the cause for performance degradation on "high trajectory variability" tasks like Hopper and Walker2d ? Experimental result(e.g. with action mask vs without action mask) would be helpful to understand your claim.

2, The paper's core mechanism is averaging gradients rather than targets. The result shows that naive target averaging fails in sparse-reward settings (Fig. 10b, App. E). Could you provide more intuition as to why averaging gradients is more effective at preserving the sparse reward signal?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes T-SAC (Transformer-based Soft Actor-Critic), an off-policy RL method whose critic is a lightweight causal Transformer trained on short trajectory segments with aggregated (N)-step returns. Unlike standard SAC, which evaluates single state–action pairs, T-SAC conditions the critic on action prefixes $(s_t, a_t,\ldots,a_{t+n-1})$ and predicts prefix-conditioned values for multiple horizons without importance sampling. The method averages gradients across N-step targets and introduces a parameter-freezing schedule for the critic in locomotion tasks. Experiments on Meta-World ML1, Box-Pushing, and Gymnasium MuJoCo show improved sample efficiency, especially on long-horizon and sparse-reward tasks; e.g., 96.8% success on Box-Pushing (dense) and solving most ML1 tasks within ~5M interactions at UTD=1.

### Strengths
1. The causal Transformer predicts prefix-conditioned values aligned with realized action prefixes, which plausibly improves temporal credit assignment. Using a “non-soft” critic with entropy regularization on the policy side preserves the standard SAC actor objective and simplifies the critic target, improving implementability.
2. Gradient-level averaging is supported by Lemma 3 and Theorem 5, establishing $\mathrm{Var}[\nabla_\psi \bar L] < \sigma_w^2$ under equicorrelation assumptions. The paper analyzes target-side variance reduction for both reward and bootstrap components, providing bounds $1 \le R_\gamma(N) < 4$ and conditions for reducing bootstrap variance. The IS-free formulation avoids high-variance importance ratios by conditioning on realized prefixes and is clearly derived from the N-step return objective.
3. The evaluation spans 57 tasks across Meta-World ML1, Gymnasium MuJoCo, and Box-Pushing. Gains are pronounced on difficult tasks (e.g., 96.8% on Box-Pushing (dense), outperforming step-based baselines ≤85%) and on multi-phase ML1 tasks such as Assembly, Disassemble, Hammer, and Stick-Pull.

### Weaknesses
1. The core idea—Transformer critic trained with N-step returns—resembles TOP-ERL; the manuscript acknowledges this but does not sufficiently sharpen the technical distinctions. The claim of “bridging step-based and episodic regimes” appears overstated: the policy remains step-based while sequence conditioning is confined to the critic. Several components (causal Transformer, parameter freezing, N-step averaging) draw on prior art, but the paper does not isolate which design choices constitute the principal contribution.
2. Comparisons with traditional off-policy N-step methods using IS are empirical only; formal connections to prior analyses are not developed.
3. Reported instabilities on Box-Pushing-Sparse lead to a fallback on soft targets, limiting generality. The action-mask requirement and performance variability on Ant/Hopper/Walker2d suggest fragility and sensitivity to dynamics/constraints.
4. T-SAC trails CrossQ on Hopper and Walker2d; the explanation “action mask needed may degrade performance” lacks systematic investigation. On Box-Pushing-Sparse, success is 58%, below TOP-ERL’s 70%; the claim of being “competitive under sparse feedback” does not reconcile this gap.
5. No ablations on freeze length $K$ or reuse factor $N_c$ are provided to delineate stability regions.
6. Equation (1) defines $G^{(n)}$ with $V_\phi(s_{t+n})$ but does not clarify whether this is a soft or standard value; later (Sec. 4.3) it states the critic estimates the standard (non-soft) action-value. The notation should reflect this distinction.
7. All benchmarks are simulated continuous-control tasks; the paper does not evaluate discrete action spaces, stronger partial observability (beyond action history), or real-robot settings that motivate the approach.

### Questions
1. Can T-SAC be evaluated on discrete-action domains (e.g., Atari) or more strongly partially observable settings?
2. Do the benefits of sequence-conditioned critics carry over to image-based ML1 or partially observed Box-Pushing variants?
3. How does T-SAC compare with Transformer-based policies and offline RL approaches (e.g., Decision Transformer)?
4. For GRU/LSTM critics, how extensive was the hyperparameter and architecture search (depth, hidden size, normalization, teacher forcing, sequence length)?
5. Have you tested robustness under observation/action noise, stochastic terminations, or mixed-policy replay buffers? This seems particularly relevant given the “off-policy without IS” choice.
6. What aspects of the design (e.g., window sampling, bootstrap lag, prefix length) most limit performance in sparse-reward settings, and could targeted modifications (e.g., auxiliary returns, adaptive horizons) mitigate the observed instability?

### Soundness
3

### Presentation
3

### Contribution
2
