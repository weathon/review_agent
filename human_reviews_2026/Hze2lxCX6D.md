# Expressive Value Learning for Scalable Offline Reinforcement Learning

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Reinforcement learning (RL) is a powerful paradigm for learning to make sequences of decisions. However, RL has yet to be fully leveraged in robotics, principally due to its lack of scalability. Offline RL offers a promising avenue by training agents on  large, diverse datasets, avoiding the costly real-world interactions of online RL. Scaling offline RL to increasingly complex datasets requires expressive generative models such as diffusion and flow matching. However, existing methods typically depend on either backpropagation through time (BPTT), which is computationally prohibitive, or policy distillation, which introduces compounding errors and limits scalability to larger base policies. In this paper, we consider the question of how to develop a scalable offline RL approach without relying on distillation or backpropagation through time. We introduce Expressive Value Learning for Offline Reinforcement Learning (EVOR): a scalable offline RL approach that integrates both expressive policies and expressive value functions. EVOR learns an optimal, regularized $Q$-function via flow matching during training. At inference-time, EVOR performs inference-time policy extraction via rejection sampling against the expressive value function, enabling efficient optimization, regularization, and compute-scalable search without retraining. Empirically, we show that EVOR outperforms baselines on a diverse set of offline RL tasks, demonstrating the benefit of integrating expressive value learning into offline RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents a generative offline RL algorithm, which avoids policy distillation and backpropagation through time during policy optimization. Experiments on OGBench validate the effectiveness of the proposed method.

### Strengths
- EVOR enables inference-time policy extraction without relying on model distillation or backpropagation through time.
- The proposed approach outperforms baselines in various benchmarks.
- This paper provides both empirical and theoretical analyses.

### Weaknesses
- The authors argue that one-step models are difficult to scale to larger base policies (e.g., VLAs) or real-world tasks. However, no corresponding experiments on VLA or real-world settings are provided to support this claim.
- The paper includes only a single baseline (Q-chunking) and lacks comparisons with other competitive methods, such as FQL [1].
- The evaluation is limited to OGBench. It would be convincing to include results on other benchmarks, such as D4RL [2].

References:

[1] Park et al. "Flow Q-Learning", ICML, 2025.

[2] Fu et al. "D4RL: Datasets for Deep Data-Driven Reinforcement Learning", arXiv preprint arXiv:2004.07219, 2020.

### Questions
Have you compared the training time and the number of parameters with non-generative offline methods such as CQL [1]?

Reference:

[1] Kumar et al. "Conservative Q-Learning for Ofﬂine Reinforcement Learning", NeurIPS, 2020.

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
This paper proposes EVOR (Expressive Value Learning for Offline Reinforcement Learning), a scalable offline RL framework that integrates expressive generative models (flow matching) into both the policy and the value function.
The authors identify a key bottleneck in recent diffusion- and flow-based offline RL: Existing methods either rely on backpropagation through time (BPTT), which is computationally expensive, or on policy distillation, which introduces compounding errors and limits scalability to larger base policies (e.g., vision-language-action models).
EVOR addresses this by: 1. Learning expressive value functions via flow-based temporal difference (TD) learning, avoiding Gaussian approximations 2. Performing inference-time policy extraction through rejection sampling against the expressive learned value function, enabling scalable optimization and regularization without retraining. 3. Leveraging inference-time compute scaling  and regularization. EVOR thus provides a unified framework for scalable, expressive, and compute-efficient offline RL, avoiding BPTT and distillation entirely. Empirically, EVOR significantly outperforms the baseline Q-Chinking (QC) across 25 tasks in the OGBench suite (antmaze, cube manipulation, scene play, and pointmaze). It also demonstrates strong scaling properties with increased inference compute and robustness to hyperparameters.

### Strengths
The paper addresses a timely and underexplored issue: how to scale offline RL with expressive generative models without incurring the computational and stability issues of BPTT or distillation. While diffusion and flow models have been extensively used for policy modeling, they have not been effectively integrated into value learning. EVOR’s insight, that flow-based value functions can provide a scalable and expressive mechanism for both regularization and inference-time optimization, is conceptually elegant and fills an important gap.
The theoretical formulation is solid and well-grounded, and this clarity and generality make EVOR theoretically appealing and reproducible.
Unlike prior work that focuses on expressive policies, EVOR emphasizes expressive value functions, a neglected component in scaling offline RL. This focus on value expressivity represents a genuine conceptual advance and could influence future RL architectures.
EVOR achieves consistent improvements across many tasks, outperforming QC-1 and QC-5 baselines. Importantly, EVOR uses the same training hyperparameters across all environments.

### Weaknesses
Although the authors justify focusing on QC to isolate “value learning expressivity,” it would be more compelling to include additional baselines such as IQL and CQL. 
All results are on simulation-based OGBench tasks. Given the paper’s claim of “scalability,” evaluation on higher-dimensional or real-robot tasks (e.g., Pi0, RT-1 datasets) would strengthen the impact. Even a discussion on computational resource scaling (training time, memory vs. QC) would help quantify efficiency.
While the ablations on inference-time parameters are thorough, there is no study isolating the contribution of flow-based TD learning itself versus standard TD learning with a Gaussian critic. A simple comparison of “EVOR without flow-based TD” would help clarify whether expressivity, or the training dynamics, drive the observed improvements.
The derivation of the optimal regularized Q-function (Eq. 16) assumes deterministic transitions, which rarely hold in realistic robotics. Although the authors later extend to stochastic settings (Section 3.2), a discussion of approximation error or theoretical relaxation would improve rigor.

### Questions
1.	It would be better if the authors add ablations on flow-based TD learning, compare to Gaussian critics or classification-based Q-updates (Farebrother et al., 2024) to show explicit benefit.
2.	It would be better if the authors include wall-clock efficiency metrics to show training/inference time versus QC and policy distillation methods.
3.	It would be better if the authors discuss potential integration with policy gradients.  The discussion section (p. 9) hints at combining EVOR with reparameterized gradients; elaborating on this could position EVOR as a foundation for hybrid methods.

### Soundness
3

### Presentation
2

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
This paper proposes EVOR, a novel flow-matching-based offline reinforcement learning algorithm that scales to data, model, and inference-time computation without enforcing the backpropagation through time or policy distillation. EVOR introduces expressive policy and value learning via flow-based temporal difference learning with the distributional Bellman operator and the reward-to-go approach. Furthermore, EVOR enables inference-time-scalable offline RL by leveraging rejection sampling with the learned optimal value function and searching the test-time hyperparameters (e.g., $\tau$ temperature) without further training. Experiments highlight the contribution of expressive value learning against the baseline with flow matching policy learning in OGBench.

### Strengths
- The paper is well structured and sounds coherent.
- EVOR demonstrates strong performance on diverse tasks in OGBench.
- The authors provide empirical observations on EVOR's inference-time scaling property by carefully designing ablation studies.

### Weaknesses
- The novelty of EVOR is marginal. The theoretical analysis of expressive flow matching policy learning strictly depends on Theorem 2.2 in Zhou et al. While it is noticeable that EVOR suggests a novel expressive value learning by introducing a flow-based TD objective with reward-to-go, experimental results on verifying the performance of EVOR on diverse offline RL tasks are insufficient to claim the practicality of EVOR. I appreciate the main results on comparing QC with EVOR and the additional ablation studies on inference-time scaling, while I believe further experiments on differentiating EVOR with state-of-the-art offline RL methods (specifically, offline RL with generative models) would support the novelty of this paper.
- It is unclear how EVOR scales to different axes of scalability: the data and model in offline RL. The authors introduce three heterogeneous axes for scaling offline RL in the introduction: the data, model, and computing. However, the primary results in the paper mainly consider the inference-time scaling property. Offline trajectories often involve multi-modal, sub-optimal behaviors from unknown policies, which makes it hard for the model to learn such diverse behaviors with limited scalability. While OGBench provides diverse tasks with thoroughly collected sub-optimal behaviors, comparing only the performance of EVOR in single tasks may overshadow the true scalability of the data and model.

### Questions
- Could you provide additional results with popular baselines in OGBench? As the authors denote in the paper, comparing with baselines trained with standard offline RL objectives may escape the scope of this paper. However, comparing with diffusion-based (or flow-based) policy learning [1, 2] (representatives for BPTT) or policy distillation methods [3] (representatives for distillation) would provide an in-depth view of scalable offline RL with generative models.
- Could you visualize how EVOR learns complex behaviors from the dataset using flow matching TD objectives and policy extractions? For instance, Diffusion Policy [4] demonstrates strong multi-modal capability by visualizing modes in trajectory levels.

# Minor problems (do not affect the score)
- There are several typos:
- - In L86, does not learn require $\rightarrow$ does not require
- - In L91, the Q-function learned $\rightarrow$ the learned Q-function
- - In 199, samples from in $\rightarrow$ samples from
- The Related Work section is duplicated in the Appendix. I appreciate the extended related work section in the Appendix, but offline RL with generative models and inference-time scaling in offline RL are exactly the same as the section in the main paper.

[1] Park S, Li Q, Levine S. Flow q-learning. arXiv 2025.

[2] Hansen-Estruch P, Kostrikov I, Janner M, Kuba JG, Levine S. Idql: Implicit q-learning as an actor-critic method with diffusion policies. arXiv 2023.

[3] Ding Z, Jin C. Consistency models as a rich and efficient policy class for reinforcement learning. arXiv 2023.

[4] Chi C, Xu Z, Feng S, Cousineau E, Du Y, Burchfiel B, Tedrake R, Song S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research 2025.

### Soundness
3

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
4

### Summary
This paper introduces EVOR (Expressive Value Learning for Offline Reinforcement Learning), a new scalable offline RL framework that integrates expressive generative value functions with expressive policies. Existing methods that use diffusion or flow-matching policies in offline RL often rely on backpropagation through time (BPTT) or policy distillation, both of which limit scalability.

EVOR avoids both by:
1. Learning a regularized, expressive Q-function using flow matching, resulting in an optimal solution to the KL-regularized offline RL objective.
2. Performing inference-time policy extraction using rejection sampling guided by the expressive value function.
3. Supporting inference-time scaling and regularization by adjusting the number of sampled actions and softmax temperatures.

Empirical results on OGBench (25 tasks across 5 environments) show consistent improvements over Q-Chunking (QC-1 and QC-5) baselines.

### Strengths
- Clear motivation and problem formulation. The authors clearly identify the scalability bottlenecks in prior generative-policy offline RL methods—specifically the reliance on BPTT or distillation. They motivate EVOR as a natural step forward: integrating expressive flow-based modeling into the value function, not just the policy (Sec. 1–3).
- Elegant inference-time optimization mechanism. EVOR’s rejection-sampling-based policy extraction (Algorithm 2, Eq. 22) allows scalable test-time optimization without retraining. The mechanism directly leverages the expressive Q function and supports compute scaling via N_\pi and temperature parameters (τ_R, τ_Q), which are empirically validated. EVOR’s inference-time scaling shows performance increasing monotonically with compute, aligning well with scaling-law intuition in modern generative RL systems. This feature enhances real-world deployability in compute-flexible settings.
- Empirical results show benefit. Table 1 and Table 2 demonstrate consistent improvement on all five OGBench environments, with particularly strong gains on difficult manipulation and navigation tasks.

### Weaknesses
- Limited novelty compared to prior works. The construction of the value function in EVOR essentially follows the same principle as "Q♯: Provably Optimal Distributional RL for LLM Post-training", which estimates Q^*(x,a) by sampling reward-to-go values and applying a softmax/log-sum-exp aggregation. The only difference is that EVOR learns the reward-to-go distribution via flow matching, but this idea is already very similar to TD-Flow (Farebrother et al., 2025) and Intention-Conditioned Flow Occupancy Models (ICFOM), which also learn value functions from offline data using flow-based velocity fields. Hence, the claimed contribution is incremental rather than conceptually novel.
- Restricted to KL-regularized RL formulation. EVOR is derived specifically under the KL-regularized RL objective, which is known to suffer from over-conservatism due to the KL divergence (penalizing exploration beyond the dataset distribution). This limitation may constrain the method’s ability to generalize to other regularizers (e.g., support-based ones) or to handle cases where mild extrapolation beyond the offline data is beneficial.
- Dependence on rejection sampling for policy extraction. The learned Q(s,a) is only utilized through rejection-sampling-based policy extraction, which makes the method heavily reliant on the quality of the offline dataset. When the offline data are suboptimal or sparse in high-value regions, rejection sampling cannot meaningfully exceed the dataset performance, since all sampled actions originate from the base policy. This limits EVOR’s applicability in low-quality or diverse-data regimes, where gradient-based policy improvement or explicit policy learning may be preferable.
- Comparative scope is narrow. Experiments are restricted to QC baselines and only on OGBench datasets. Other recent scalable offline RL approaches (e.g., Flow Q-Learning) are missing. Since EVOR’s main claim is “scalability,” benchmarking against state-of-the-art methods beyond QC would strengthen the empirical case.

### Questions
Since the paper emphasizes Scalable Offline Reinforcement Learning, could the authors further validate the effectiveness of Expressive Value Learning on the D4RL MuJoCo datasets? Compared with OGBench, the behaviors in D4RL datasets are typically more suboptimal and diverse, making them a strong testbed for scalability and robustness. Would Expressive Value Learning help improve performance under such more challenging, low-quality offline data distributions? This seems like a very meaningful way to verify the scalability claims of the proposed method.

### Soundness
2

### Presentation
3

### Contribution
2
