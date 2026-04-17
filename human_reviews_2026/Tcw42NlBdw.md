# Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Value decomposition is a core approach for cooperative multi-agent reinforcement learning (MARL). However, existing methods still rely on a single optimal action and struggle to adapt when the underlying value function shifts during training, often converging to suboptimal policies. To address this limitation, we propose Successive Sub-value Q-learning (S2Q), which learns multiple sub-value functions to retain alternative high-value actions. Incorporating these sub-value functions into a Softmax-based behavior policy, S2Q encourages persistent exploration and enables $Q^{\text{tot}}$ to adjust quickly to the changing optima. Experiments on challenging MARL benchmarks confirm that S2Q consistently outperforms various MARL algorithms, demonstrating improved adaptability and overall performance. Our code is available at https://github.com/hyeon1996/S2Q.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes S2Q, a new framework for cooperative MARL under CTDE settings. This paper retains and learns from suboptimal actions so that the system can better adapt when the optimal joint policy changes during the training. Instead of relying on single best action, S2Q successively trains multiple sub-value functions that each caputure distinct high-value joint actions. By combining them through softmax-based behaviour policy, the method encourages persistent exploration and rapid adjustment to new optima. Experiments on SMAC, SMAC-Comm, and GRF shows outperformance to baseline algorithm.

### Strengths
1. Originality: The paper presents an original idea of retaining suboptimal actions through the proposed S2Q framework. This introduces a new perspective in value decomposition MARL by modeling multiple high-value modes instead of a single optimal action, addressing a long-standing issue of adaptability under shifting optima.
2. Quality: The technical formulation is solid and well-motivated, supported by a clear derivation of objectives and loss functions. The experimental setup is comprehensive, evaluating the method on multiple challenging benchmarks and performing ablations to validate each component.
3. Clarity: The paper is clearly written, with well-organized sections, intuitive figures, and a motivating example (the payoff matrix) that effectively conveys the main idea. The methodology and algorithm are clearly explained and reproducible, with good visual aids to guide understanding.
4. Significance: The contribution is significant for cooperative MARL. It addresses a key limitation in existing value decomposition approaches, poor adaptability when optimal actions shift. The framework’s generality and empirical robustness indicate potential impact on future MARL research.

### Weaknesses
1. Limited theoretical depth. While the paper provides a solid conceptual and empirical contribution, the theoretical analysis of why successive sub-value learning guarantees adaptability or improved convergence remains shallow. The paper lacks formal results on convergence properties, stability, or value approximation bounds for S2Q, which would strengthen its scientific aspect.
2. Incremental novelty relative to prior frameworks. Although the idea of retaining suboptimal actions is interesting, the implementation may be viewed as an extension of WQMIX with auxiliary sub-networks and a Softmax exploration mechanism. The paper could better clarify how S2Q fundamentally differs from or improves upon multi-head or ensemble-based exploration methods beyond heuristic layering.
3. Computational efficiency discussion. The paper acknowledges increased computation and memory overhead but only qualitatively. A quantitative analysis comparing GPU memory, or training wall-clock time to QMIX/WQMIX would make the practicality argument more convincing, especially for scaling to large agent teams.
4. Clarity of communication variant. The role of the encoder–decoder communication module is somewhat underexplained. It is unclear how much of the improvement in SMAC-Comm tasks comes from the communication architecture itself versus the S2Q mechanism. A clearer disentanglement or comparison to standalone communication baselines would improve interpretability.
5. Missing experiment details. Although the codebases are described in the Appendix individually, it seems that the performance in Figure 5 of the QMIX algorithm originated from pymarl2 does not achieve the performance stated in RIIT[1]. A detailed training parameters or any modifications are better to be discussed.

[1] Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning

### Questions
Based on the weaknesses described above, I have some questions for the authors.

1. Could the authors provide a more formal justification or theoretical insight into why maintaining multiple sub-value functions helps track shifting optima more effectively?
2. Are there any guarantees on the convergence or stability of S2Q, particularly when sub-value suppression is applied iteratively? A sketch of theoretical grounding could greatly strengthen the contribution.
3. Could the authors provide a direct comparison or at least a discussion to clarify whether S2Q’s performance gains come from retaining suboptimal modes or simply from having additional representational capacity?
4. The paper mentions that the computational overhead is “moderate.” Could the authors quantify this or training time ratio relative to QMIX or WQMIX?
5. How does the overhead scale with the number of sub-value networks K and the number of agents N? This would help assess the method’s practicality in larger-scale MARL tasks.
6. Have the authors considered testing in deliberately non-stationary environments, such as mixed multiple opponent strategies in SMAC-Hard environment,to validate S2Q’s adaptability?
7. In SMAC-Comm results, how much improvement arises from the encoder–decoder communication mechanism versus the S2Q framework itself?
8. The paper positions S2Q as addressing the issue of “shifting optima.” Could the authors better relate this to other known issues such as non-stationarity, policy co-adaptation, or value landscape drift in MARL? Clarifying this link may help highlight the conceptual novelty and general applicability of S2Q.

If all the concerns are well-addressed, I would consider raising the score. Thank you.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a method to overcome value function shifts during training in CTDE MARL, named S2Q. The proposed method learns a set of several sub-value functions, where each aims to identify different suboptimal actions.

### Strengths
This paper is well written and provides a very extensive set of experiments and ablations. The results are consistently strong across very diverse environments. While apparently simple, i find clever the idea to surpress the optimal actions in the calculations of subsequent value functions and the performances show big improvements in a range of tasks.

### Weaknesses
The authors could have provided a deeper analysis of the scalability of the proposed method, since it requires sequential computations using sub-networks. I.e, since there is a mixing for each Q, up until what point can k scale? In the communication encoder-decoder module in figure 3, the authors could have provided a better description of the architecture of these modules. 

Please find below some more specific questions.

### Questions
1. Could the authors describe another more practical scenario aside from the provided matrix games where the optimal action shift might happen? For example in the experimented environments such as SMAC, when could such shift happen?
2. i find interesting the sudden boost around 1M timesteps in figure 6 for the task bane_vs_hM; have the authors explored why this is seen only in this specific environment, instead of a more linearly increasing curve?
3. how can you find the best k? is there any guarantees (theoretical maybe) that for higher k the variance and learning instability will increase (as mentioned in lines 454-455)?
4. considering the sub value functions are learned in a sequential manner, how does the proposed method relate to the higher order q-learning approach presented in [1]?

[1] https://arxiv.org/abs/2304.13383

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
4

### Summary
The paper proposes Successive Sub-value Q-learning (S2Q) for cooperative MARL under CTDE. Instead of committing to a single estimated-optimal joint action, S2Q learns a sequence of monotonic, value-decomposition subnetworks that each focuses on a different high-value mode by suppressing actions selected by earlier subnetworks in its TD objective. To synchronize the index k during training, the authors use an encoder–decoder that reconstructs the global state and approximates the Softmax distribution. Experiments on SMAC Hard+, GRF, SMAC-Comm, and SMACv2 report higher win rates and faster convergence than strong baselines. Ablations suggest the Softmax selection and successive learning are key. Compute overhead is reported as modest while reaching 50% win rate substantially faster than QMIX.

### Strengths
1. The toy matrix-game illustrates why retaining information about nearby high-value actions can help when the optimum shifts. S2Q operationalizes this via a suppressed TD objective and Softmax-guided behavior policy. The algorithmic presentation (Alg. 1; eqs. (B.2–B.5)) is easy to follow. 

2. Results span SMAC Hard+, GRF, SMAC-Comm (with a “-Comm” variant), and SMACv2, showing consistent gains and faster learning, not only final win rates. The compute table quantifies overhead.

3. Removing Softmax selection or randomizing $k$ materially hurts performance. Analysis of $K$ and temperature $T$ is informative. Sensitivity of suppression $\alpha$ and weighting $w_c$ appears in the appendix.

4. Default evaluation needs no communication relying on $Q_0^{sub}$, which is attractive compared to methods that require message passing at inference.

### Weaknesses
1. The paper claims theoretical/empirical analyses, but no formal result is provided to justify that minimizing the modified TD with the suppression term reliably extracts distinct top-k modes under the IGM constraint or preserves contraction/stability properties. A small lemma would strengthen the case.

2. S2Q learns an encoder–decoder to approximate $P_t$ and reconstruct $s$, which provides additional supervision such as cross-entropy and reconstruction. Several non-communication baselines do not leverage comparable auxiliary signals, raising comparability questions. A variant without the encoder–decoder, or with matched auxiliaries for baselines would calibrate the gain attributable to successive sub-values and auxiliary training.

3. While $K$ and $T$ are studied, they’re analyzed mainly on 6h vs 8z. It would be helpful to report these sweeps across more tasks (e.g., MMM2, GRF academy 4v3) and include a “no-comm” ablation that samples $k$ independently to show the necessity of synchronization, and an oracle $P_t$ using the true Softmax to bound performance. 

4. In Eq. (2)/App. B.2, the target for previously selected actions is reduced by $\alpha Q^*$, which can change the scale and sign of TD targets. There is no analysis of potential instability or bias this introduces, particularly with function approximation. Clarifying normalization or clipping strategies, and why the scheme does not collapse would help. In addition, this method shares the similar idea with [1], it would be better to compare with this baseline.

[1] Wan, L., Liu, Z., Chen, X., Wang, H., & Lan, X. (2021). Greedy-based value representation for optimal coordination in multi-agent reinforcement learning. arXiv preprint arXiv:2112.04454.

### Questions
1. Can you formalize guarantees (even under simplified assumptions) that successive suppression + weighting (B.5) yields distinct top-k action sets, or at least that $Q_0^{sub}$ is not harmed by the auxiliary subnetworks? A proposition about bias introduced by the suppression term would be valuable.

2. What happens if you remove communication during training and have agents sample k independently, or conversely provide baselines with a matched auxiliary (state reconstruction)? Please include a “no-comm” S2Q variant and, if possible, an oracle $P_t$ variant to bound gains.

3. You show $K=2,T=0.1$ works broadly (Tab. E.2). Could you include per-map sweeps (or at least MMM2 and academy 4v3) to show consistency and to better justify the choice of $K$?

4. Since the paper cites “MARL as sequence modeling”, could you comment on (or include) a representative sequence-model baseline and discuss compatibility?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Successive Sub-value Q-learning (S2Q), a framework that successively learns multiple subvalue functions to retain information about alternative high-value actions. By incorporating these sub-value functions into a Softmax-based behavior policy, S2Q encourages persistent exploration and enables Qtot to adjust quickly when the optimal action changes. Experimental results show that S2Q outperforms other recent MARL methods on the StarCraft II Multi-Agent Challenges and Google Research Football.

### Strengths
- Dynamic value functions

    S2Q overcomes the limitation that conventional methods do not explicitly track suboptimal actions. When the optimal action changes, S2Q can immediately leverage the corresponding sub-value function and guide Q^{tot} to adapt. 

- Introducing communication during training

    S2Q explicitly executes tracked suboptimal actions with priority determined by a Softmax distribution P_t over their Q^{∗} values, thereby enabling exploration of a wider range of spaces than conventional ϵ-greedy exploration.

### Weaknesses
- Old Benchmarks 

    The StarCraft Multi-Agent Challenge (SMAC) (Samvelyan et al., 2019) is an old benchmark. It is advised to report the experimental results on the recently proposed SMAC-Hard benchmark [1].

    [1] SMAC-Hard: Enabling Mixed Opponent Strategy Script and Self-play on SMAC, arXiv:2412.17707.

### Questions
Figure 1 demonstrates changes in the payoff matrix. Could you provide concrete examples of how the value function dynamically evolves in SMAC-Hard or GRF environments?

### Soundness
3

### Presentation
3

### Contribution
3
