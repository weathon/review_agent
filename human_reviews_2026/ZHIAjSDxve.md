# Beyond Markov Assumption: Improving Sample Efficiency in MDPs by Historical Augmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Under the Markov assumption of Markov Decision Processes (MDPs), an optimal stationary policy does not need to consider history and is no worse than any non-stationary or history-dependent policy. Therefore, existing Deep Reinforcement Learning (DRL) algorithms usually model sequential decision-making as an MDP and then try to optimize a stationary policy by single-step state transitions. However, such optimization is often faced with sample inefficiency when the causal relationships of state transitions are complex. To address the above problem, this paper investigates if augmenting the states with their historical information can simplify the complex causal relationships in MDPs and thus improve the sample efficiency for DRL. First, we demonstrate that a complex causal relationship of single-step state transitions may be inferred by a simple causal function of the historically augmented states. Then, we propose a convolutional neural network architecture to learn the representation of the current state and its historical trajectory. This representation learning compresses the high-dimensional historical trajectories into a low-dimensional space to extract the simple causal relationships from historical information and avoid the overfitting caused by high-dimensional data. Finally, we formulate Historical Augmentation Aided Actor-Critic (HA3C) algorithm by adding the learned representations to the actor-critic method. The experiment on standard MDP tasks demonstrates that HA3C outperforms current state-of-the-art methods in terms of both sample efficiency and performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether leveraging historical information can improve performance in deep reinforcement learning (DRL), even under the Markov property assumption of MDPs. The authors design a representation learning architecture that abstracts historical state information using CNN-based models. Building on this, they propose the HA3C algorithm, which integrates these history-augmented representations into a TD3-style off-policy framework. Experimental results show that HA3C outperforms several off-policy DRL algorithms, including TD7, on Mujoco continuous control tasks.

### Strengths
- Provides a theoretical attempt to analyze the benefit of history augmentation.

- Demonstrates promising empirical results in Mujoco environments.

### Weaknesses
1. Motivation

The necessity of history augmentation under the Markov property assumption is not convincingly justified. The paper claims that “the causal function in Fig. 1(b) can be simpler than the causal function in Fig. 1(a)” and that “historical information can simplify complex causal relationships in MDPs,” but these statements are unclear.

- Could the authors explain these two claims in easier words?

- Do the authors believe that the lack of history augmentation (under the Markov property assumption) is the main bottleneck limiting the sample efficiency of off-policy DRL algorithms?

- Why can’t similar benefits be achieved through careful neural network design without explicitly augmenting history in DRL?

- Can the authors provide a simple empirical demonstration showing that history augmentation improves representation learning, apart from Figure 5 (which only reflects final return performance)?

&nbsp;

2. Literature Review

- If the authors believe that history augmentation is crucial for improving sample efficiency, the related work section should include prior studies that address sample efficiency improvements in DRL.

- Several existing works have incorporated causal inference into (deep) RL. The authors should compare HA3C with these studies, both in terms of (i) methodological differences and (ii) the contexts or issues being addressed.

&nbsp;

3. Organization and Writing

The writing and organization can be improved. In particular, the motivation should be strengthened, as discussed above. Additionally, many experimental results are deferred to the Appendix, but key results should appear in the main paper.

The experimental results in the Appendix include fewer baselines than those in the main text, please clarify this inconsistency.

&nbsp;

4. Clarifications Needed

- Can this approach also be effective in POMDP settings? If so, can the authors provide supporting experiments?

- Did the authors compare HA3C with conventional DRL algorithms (e.g., TD3 or TD7) using the same number of stacked input frames ($k$)?

- In Figure 3, how is the CNN used to process $k$-multiple input frames?

- Is the architecture in Figure 3 applicable to discrete action spaces?

- What is the distinction between fixed encoders and target encoders in Section 4.2?

- Can the authors provide the ablation with $k=2, 3$?

&nbsp;

5. Additional Questions

- Could the algorithm be extended with an adaptive adjustment mechanism for $k$?

- Can this architecture be adapted for online DRL algorithms such as PPO?

### Questions
Please see Weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents HA3C, a deep reinforcement learning algorithm designed to tackle the critical issue of sample inefficiency in Markov Decision Processes  with complex transition dynamics. The central premise is that augmenting the current state with historical information can simplify the learning of these complex causal relationships. To achieve this without suffering from high-dimensional inputs, HA3C employs a Convolutional Neural Network to compress the state trajectory into a concise, low-dimensional representation. This learned historical context is then integrated into a powerful actor-critic framework, leading to significant improvements in both sample efficiency and final performance across a suite of challenging continuous control benchmarks.

### Strengths
1. This paper is well-organized. 
2. The paper's central contribution rests on its compelling and counter-intuitive argument for leveraging history in MDPs. This core premise is rigorously validated through comprehensive empirical evidence.
3. Experimental results show promising results on MuJoCo and DMC tasks.

### Weaknesses
1. On several MuJoCo tasks, such as HalfCheetah, the performance improvement over TD7, while positive, may not be statistically significant when considering the reported standard deviations. The authors should provide a more detailed analysis of the statistical significance of their results.
2. The paper's core premise that history simplifies future prediction is not critically examined for its limitations. The authors' own finding that long histories introduce noise (for k=24) suggests potential failure modes, yet there is no broader discussion of which environmental properties might render historical information detrimental.
3. The theoretical justification for why the proposed method reduces sample complexity is underdeveloped. The paper provides an intuitive motivation but lacks a formal analysis (e.g., sample complexity bounds) to explain the mechanism, positioning the contribution primarily as a strong empirical finding rather than a new theoretical framework.
4. The algorithm's performance is highly sensitive to the historical window size k, a crucial hyperparameter that requires manual, task-specific tuning. The absence of a principled method for selecting k limits the algorithm's practicality and makes it difficult to apply to new environments without extensive tuning.

### Questions
1. What mechanisms within HA3C are designed to prevent the model from overfitting to spurious correlations within historical trajectories, and can you formalize the conditions under which history might become detrimental to learning?
2. Instead of a fixed k, have the authors considered dynamic architectures, such as using an attention mechanism, that would allow the agent to adaptively learn the relevant temporal dependencies for a given task or state?

### Soundness
3

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
The paper proposes historical augmentation to address sample inefficiency in DRL for complex MDPs. The manuscript is well-structured and easy to follow, with clear motivation (e.g., Fibonacci sequence examples), coherent theoretical-algorithmic design (Theorem 4.1, CNN-based representation learning, HA3C), and comprehensive baseline comparisons on MuJoCo and DMC tasks.

### Strengths
1. Clear Problem Definition: The paper effectively highlights the sample inefficiency of existing DRL methods under complex causal relationships in MDPs, even when the Markov assumption holds.
2. Theoretical and Algorithmic Coherence: Theorem 4.1 (existence of a k-order stationary deterministic policy) provides theoretical justification for historical augmentation, while the CNN-based representation learning (compressing high-dimensional historical trajectories to avoid overfitting) and HA3C algorithm (integrating TD3, historical augmentation, and checkpoints) form a coherent technical pipeline.
3. Comprehensive Baseline Comparisons: Validates performance against 5 SOTA algorithms and extends to HA3C-SAC, demonstrating cross-algorithm/task adaptability.

### Weaknesses
1. Missing key baseline: No comparison with raw concatenation of current and historical states (a standard POMDP practice). Without this, it’s unclear if HA3C’s gain comes from historical information or CNN compression—add this baseline to validate representation learning value.
2. Unreported parameters: Table 1 lacks network parameter counts for HA3C and baselines. HA3C’s extra encoders may increase capacity; report parameters to rule out "parameter overcapacity" as a performance driver.
3. Predictive loss is not new. The Section 4.1 prediction loss (predicting future state representations) replicates prior work (e.g., SPR, ICLR 2021). Explicitly compare with SPR or add ablations for unique components (e.g., pooling) to highlight insights.
4. Unjustified focus on causal inference: In practical RL scenarios, the core goal is task solving, not explicit causal relationship inference. Neural networks may implicitly learn causal patterns (or not) as long as the task is solved—there is no evidence that explicit causal understanding improves state representation. The paper fails to justify why causal inference is necessary for the proposed method.

### Questions
The term "complex MDP" is actually not very specific. The paper provides no clear criteria for identifying which MDP scenarios benefit from causal understanding, making it hard to generalize the method’s applicability. In what RL scenario does causal understanding benefit most?

### Soundness
2

### Presentation
3

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
The paper is motivated by the observation that a stationary Markov policy utilising single-step state transitions can often result in inefficient reinforcement learning, especially when the underlying causal relationships of state transitions are complex. To address this, the authors propose to augment the state with historical information, and hence learn a policy of the historically augmented state. To this end, they use a CNN to learn a compressed low-D representation of the high--D historical trajectory, which only focuses on the underlying causal relationships and avoids the overfitting due to having high-D data. The authors introduce a new algorithm, HA3C, which is a variant of the A3C method that additionally uses the learned representation. The experimental study demonstrates improved sample efficiency and performance.

### Strengths
- The paper has strong motivation and intuition - we indeed expect that including historical information can result in better sample complexity and more efficient reinforcement learning, because it can better capture long-term causal relationships in the transition function.
- The proposed framework is theoretically sound.
- The experiments on various benchmarks show strong performance compared to the baselines. Overall, the authors provide strong evidence in favour of using history-augmented policies.

### Weaknesses
- The novelty is not significant in my personal view. Self-predictive historical RL representations have been investigated in prior work. Check for instance the recent ICLR 2024 paper "Bridging state and history representations : Understanding self-predictive RL" by Ni et al. The idea of employing encoders together with an L2 loss or even a probabilistic f-divergence metric has been already studied in the prior literature (see, e.g., Section 4.1 of aforementioned paper and references therein). So, I feel the representation learning part covered in the current paper is not particularly new or novel. 
- Parts of the theory are already known, to the best of my understanding. Assume for instance Theorem 4.1, for whose proof the authors allocate quite some space. Isn't it a fundamental result in MDP theory that "For any MDP, there exists an optimal Markov (stationary + deterministic) policy that achieves the same or higher expected return as any history-dependent policy."? In that case, isn't it obvious that a history-dependent policy will be as good as the best possible policy on the entire history? I mean, $s_{k, t}$ includes $s_t$ and possibly many more states, so it will be at least as good as using $s_t$ only; but the latter can already result in an optimal policy. To me, Theorem 4.1 looks extremely obvious, unless the authors were trying to same something different.
- Theorem 4.2 seems correct, but I feel what would really be interesting would be to include some theory on how exactly history-dependent policies can accelerate reinforcement learning. Theorem 4.2 shows convergence for the modified Bellman operator, but seems a rather obvious result.

### Questions
- How do the authors position their work compared to prior works on self-predictive historical RL representations? Please check for instance references in paper mentioned above. 
- Why do the authors include a proof for Theorem 4.1? Isn't this a completely obvious result for first-order MDPs?
- Why do the authors limit their approach to A3C? Is there a particular reason why they decided to focus on the A3C variant? Can't the proposed framework be integrated with any RL algorithm in principle? I feel this would indeed be a strength of this work.

Overall, I feel this is a sound framework with solid performance, but there are various things that must be clarified, in particular with respect to prior work.

### Soundness
3

### Presentation
3

### Contribution
2
