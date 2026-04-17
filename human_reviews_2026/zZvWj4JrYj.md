# SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Training expressive flow-based policies with off-policy reinforcement learning is notoriously unstable due to gradient pathologies in the multi-step action sampling process. We trace this instability to a fundamental connection: the flow rollout is algebraically equivalent to a residual recurrent computation, making it susceptible to the same vanishing and exploding gradients as RNNs. To address this, we reparameterize the velocity network using principles from modern sequential models, introducing two stable architectures: Flow-G, which incorporates a gated velocity, and Flow-T, which utilizes a decoded velocity. We then develop a practical SAC-based algorithm, enabled by a noise-augmented rollout, that facilitates direct end-to-end training of these policies. Our approach supports both from-scratch and offline-to-online learning and achieves state-of-the-art performance on continuous control and robotic manipulation benchmarks, eliminating the need for common workarounds like policy distillation or surrogate objectives. Anonymized code is available at \url{https://anonymous.4open.science/r/SAC-FLOW}

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes to learn flow policies using RL. Unlike previous work in behaviour cloning that learns flow fields that match the demonstration trajectories, this paper allows the flow field to be optimised against an explicit objective function. Previous work in RL for flow fields has experienced problems with gradient instability --- this paper shows that using a GRU or transformer to stabilise the gradients, and show learning improvement relative to baselines on a series of benchmarks.

### Strengths
Overall this paper proposes an interesting and novel modification to learning control flow fields. The paper is clearly written and the paper is careful to provide each step of SAC for flow fields, including the the proposed modifications. The authors show performance on both from-scratch problems and offline-to-online performance.

The experimental results show that the technique performs well on several baselines.

### Weaknesses
There are not significant weaknessess to the paper.

However, the gradient stabilisation that the authors propose, whether it is Flow-G or Flow-T, is (I think) essentially a form of regularisation. In eliminating the high variance motion of the gradient, the gradient is biased, and this may harm the learning. And we see that while both Flow-T and Flow-G generally outperform the baselines (except for Robomimic-Can and Cube-Double-Task2, where no techniques succeed), neither one is a clear winner across all benchmarks. This absence of a clear winner shows that the specific bias of the GRU or the Transformer is in fact limiting performance in some cases. It would be helpful to have some insight into how to control this bias, and whether or not the bias could be controlled, for instance by slowly widening the reset gate of the GRU or the cross-attention of the transformer.

I was also surprised that the GRU gradient norms are so much smaller for Flow-G and than for Flow-T (Figure 6). It would be good to understand how sensitive the performance is to the specifics of the GRU and transformer.

Given that this is a fairly modest step above flow-based RL, it would be good to have some understanding of the limitations of the technique. It is not clear if there are constraints on the reward function. Are there kinds of problems where this technique will not succeed?

### Questions
How sensitive the performance is to the specifics of the GRU and transformers?
Are there kinds of problems where this technique will not succeed?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces **SAC Flow**, an off-policy reinforcement learning algorithm for **flow-based policies**.
It reinterprets the multi-step flow rollout as a **residual RNN**, linking instability in prior methods to recurrent gradient issues.
To stabilise training, the authors reparameterise the **velocity network** using sequential architectures: **Flow-G** (GRU-gated) and **Flow-T** (Transformer-decoded).
A **noise-augmented rollout** allows tractable entropy computation in SAC.
Experiments on **MuJoCo**, **OGBench**, and **Robomimic** report state-of-the-art sample efficiency over existing diffusion- and flow-based baselines, with ablations assessing gradient stability and robustness.

### Strengths
* **Novel conceptual framing:** The sequential-model reinterpretation of flow rollouts (residual RNN ≈ flow ODE) is insightful and could influence future flow-based RL research.
* **Architectural innovation:** Flow-G and Flow-T represent principled design choices inspired by GRUs and Transformers, addressing stability through gating and pre-norm residuals.
* **Algorithmic completeness:** The integration into SAC with a noise-augmented rollout is well-motivated and yields a self-contained training procedure.
* **Empirical breadth:** Evaluations cover diverse benchmarks (MuJoCo, OGBench, Robomimic) and both from-scratch and offline-to-online regimes.
* **Reproducibility:** Appendices provide detailed architectures and training algorithms.

### Weaknesses
**Summary:**
The paper contains a **promising conceptual contribution** — interpreting flow-based policies as sequential models and borrowing stability mechanisms from RNN/Transformer design — and achieves competitive empirical performance.
However, the **central motivation (gradient instability in naïve SAC-Flow)** is **not convincingly evidenced**, and several key figures are inconsistent or uninformative.
The experimental section needs stronger causal and statistical support to meet ICLR’s bar for empirical soundness.

### **Weaknesses**

#### **1. Insufficient evidence for the claimed instability**

The paper’s central claim — that “Directly backpropagating through the K-step action sampling is often unstable” — lacks quantitative support.
Even the ablation on velocity network parameterizations (Fig. 13a) shows gradient explosion only near the end of training, while Fig. 6(a) implies immediate collapse if I'm not mistaken.
These two plots contradict each other, and neither connects gradient norms to learning-curve failure.
Without aligned gradient-and-performance analysis, the claim of severe instability remains unproven.
So, I felt that aggregated plot of Fig. 6(a) is therefore potentially misleading and should be clarified or replaced with joint plots of gradient magnitude and return.

#### **2. Ambiguous or uninformative results (Figs. 4g,h)**

Panels (g) *Robomimic-Can* and (h) *Cube-Double-Task2* show all methods failing equally.
The authors acknowledge this only briefly as motivation for the offline-to-online setup (p. 7), but these subfigures add no evaluative value.
They should either be removed or replaced with quantitative evidence (e.g., normalized success rates, exploration metrics) illustrating why SAC Flow fails on sparse rewards.

#### **3. Lack of aggregate evaluation across environments**

Per-task curves make it difficult to judge overall performance.
Following **Agarwal et al., NeurIPS 2021**, the authors should report **aggregate IQM returns** with **95 % bootstrap confidence intervals** and **probability-of-improvement** or **rank-based summaries** across all benchmarks.
This would clarify whether SAC Flow-T and -G consistently outperform baselines rather than winning selectively.
* Agarwal et al., NeurIPS 2021 — “Deep Reinforcement Learning at the Edge of the Statistical Precipice.”

#### **4. Missing causal evidence for performance gains**

The connection between gradient stabilization (Flow-G/T) and improved returns is asserted but not verified through controlled experiments isolating that factor.
So, it's related to my first point but without a quantitative baseline showing naïve SAC-Flow failure, it is unclear whether the gains stem from the proposed architectures or from implementation details.

### Questions
see weakness

### Soundness
2

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
4

### Summary
The paper tackles instability when training flow-based policies with off-policy RL by viewing the K-step rollout as a deep residual RNN and proposing two stabilizing architectures that resemble a GRU-style gated cell and a Transformer decoder.

### Strengths
- The problem is meaningful and of great importance to flow policies applied to RL
- The proposed approach provides new perspectives to solve the problem, and the resulting solution is clean and simple
- Empirical results demonstrate the strong performance of their approach, mostly matching or exceeding the state of the art

### Weaknesses
- Line 159 “Off-policy methods (e.g., SAC, TD3) are highly sample-efficient (Mambelli et al., 2024), yet directly backpropagating through the K-step action sampling is often unstable, especially for large K (Park et al., 2025).” — I think this core technical challenge deserves a more serious presentation with necessary math beyond verbal explanations. 
- The current presentation of Flow-G/T assumes that the architectures are natural choices inherent from modern improvement over RNNs. However, more mathematical details, e.g., showing the Jacobians after the proposed fixes, might strengthen the authors' argument
- Figure 4: I am a bit confused about the results presented in Figure 4. For example, why is the reward for PPO so low, and looks like it just barely learns anything? The results do not match (are not even close to) common open source implementations, such as the ones reported in CleanRL (https://docs.cleanrl.dev/rl-algorithms/ppo/#experiment-results_2). Also, SAC does not seem to solve Ant-v4. That said, can the authors report how they came to the chosen hyperparameter configuration for all the algorithms?
- Minor:
    - “State-of-the-art sample efficiency and performance.” — This is not counted as a separate contribution, but more of the evaluation results of the first two contributions
    - Color scheme chosen in a number of figures, e.g., Figs. 4 and 7, is visually indistinguishable

### Questions
- Line 132 “yet a single unimodal Gaussian cannot capture inherently multimodal action distributions, a limitation that is especially harmful in long-horizon robotic control” —  Actions being unimodal and the horizon of the tasks seem to be two rather independent things, can the authors elaborate more on this point?
- Line 398 “Among all experiments, the sampling steps of flow-based policies are set to 4, and the denoising steps of diffusion policies are set to 16. More details of the experimental setting are described in Appendix C and Appendix D.” — How are the 4 steps and 16 steps determined respectively

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper extends SAC to flow matching policies. In order to do so, the authors realize that flows can be expressed as residual RNNs, showing the same problems with gradient instabilities. To counter-act this effects, the authors introduce a GRU like flow architecture (FLOW-G) and a transformer like flow architecture (FLOW-T). In ordert to compute the entropy for SAC, the authors show that the flow can be transformed into a stochastic flow using Gaussian noise at each time step and we can take the gradient of the path derivative of the action trajectories to compute the gradient of the entropy. Both approaches are benchmarked on online RL and offline-2-online settings and show competetive performance.

### Strengths
- The idea of using time-series architectures to improve the instabilities of flow-matching gradients in RL is novel and promising
- the paper presents the first max-ent framework for flow matching
- benchmarks seem to be competitive, yet I have some doubts regarding some evaluations (see weaknesses)

### Weaknesses
- The relation to existing methods is unclear: By introducing the noise augmented rollouts, the approach is very similar to diffusion based methods such as DIME. In DIME, they introduce a lower bound to the entropy which consists of the noise process and reverse noise process path. In this paper, only the reverse noise path is considered fro the gradient, this is the only difference that I can see (except for the architectures). However, while the derivation of using the gradient of the path measure for the entropy seems to be correct, it is unclear why the path measure should be used as additional reward in SAC. This is just done without any further justification. In contrast, the lower bound introduced by DIME provides a much more principled derivation also for the critic update (Showing that the critic update optimizes a clearly defined objective, which is not the case here) while resulting in almost the same algorithm. 
- The relation to  DIME needs to discussed in much more detail. As far as I can see it, the paper can be seen as an extension of DIME with more sophisticated architectures (GRU or transformer instead of residual RNN) and the missing forward noise process. However, it is unclear whether this modification benefits or harms the performance.  
- I do not believe the reported results for some baselines are correct. DIME is performing much better then the reported results (even though it is evaluated on the benchmark versions v3 instead of v4, but I do not believe that this makes a substantial difference). DIME performs competitive to much stronger baselines than standard SAC such as BRO or SimbaV2, also on harder tasks than presented in this paper. While these baselines use Gaussian policies, they should be added to evaluate if the reported results are really SOTA performance.

### Questions
- The authors should investigate why DIME is not performing well for them. Maybe the v4 of the benchmark version requires different hyper-parameters
- Please explain the algorithmic differences to DIME in much more detail (by adding stochasticity to the flow integration steps your process basically becomes a diffussion process, at least the generative process is the same (except for the parametrization)). 
- Please also elaborate on why the noise augmented rollout preserves the same marginal densities. This needs further justification or a reference where this has been shown.

### Soundness
2

### Presentation
3

### Contribution
2
