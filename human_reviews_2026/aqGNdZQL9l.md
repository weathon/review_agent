# Decoupled Q-Chunking

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Bootstrapping bias problem is a long-standing challenge in temporal-difference (TD) methods in off-policy reinforcement learning (RL). Multi-step return backups can alleviate this issue but require delicate importance sampling to correct their off-policy bias. Recent work has proposed to use chunked critics, which estimate the value of short action sequences ("chunks") rather than individual actions, enabling unbiased multi-step backup. However, extracting policies from chunked critics is challenging: policies must output the entire action chunk open-loop, which can be sub-optimal in environments that require policy reactivity and also challenging to model especially when the chunk length grows. Our key insight is to decouple the chunk length of the critic from that of the policy, allowing the policy to operate over shorter action chunks. We propose a novel algorithm that achieves this by optimizing the policy against a distilled critic for partial action chunks, constructed by optimistically backing up from the original chunked critic to approximate the maximum value achievable when a partial action chunk is extended to a complete one. This design retains the benefits of multi-step value propagation while sidestepping both the open-loop sub-optimality and the difficulty of learning policies over long action chunks. We evaluate our method on challenging, long-horizon offline goal-conditioned benchmarks and show that it reliably outperforms prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the bootstrapping bias problem by proposing a novel method to decouple the chunk length of the critic from that of the policy, allowing for more efficient multi-step return backups. The proposed algorithm optimizes the policy against a distilled critic, retaining the benefits of multi-step value propagation while avoiding the challenges of open-loop sub-optimality and long action chunk learning.

### Strengths
1. The paper is clearly written and easy to follow, with well-structured explanations.
2. The authors effectively discuss the scenarios in which action chunking Q-learning is preferable over standard n-step return learning.

### Weaknesses
1. My main concern is that, by comparing the performance and curves of DQC and QC-NS, QC-NS—without decoupling the action chunking of the Q-value from that of the policy—achieves nearly the same performance as DQC, except for the cube-quadruple-100M task.
2. How much does the performance degrade if the expectile update in Eq. (24) is replaced with a quantile update?
3. Why does setting $N=128$ not lead to better performance?
4. In most algorithms, the batch size is typically set to 256, but in this paper, it is set to 4096. Could the authors conduct an ablation study to justify this choice?
5. Could the proposed method be evaluated on standard offline RL benchmarks such as D4RL?
6. There appear to be some minor errors, such as two consecutive "the"s in the first line of page 2, and potential mistakes regarding $\tau_b$ and $\tau_d$ in Algorithm 1. The authors should carefully proofread the manuscript.

### Questions
Please see weakness

### Soundness
3

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
4

### Summary
This paper introduces DQC, that builds upon Q-chunking but decouples the critic’s chunk length from the policy’s chunk length. It aims to address bootstrapping bias from TD backups and open-loop policy inefficiency in off-policy TD learning. Experiments on the OGBench suite show that DQC outperforms Q-chunking and other baselines.

### Strengths
1. This paper derives explicit bias and near-optimality bounds, offering theoretical insights of when action-chunking methods succeed or fail.

2. Empirical performance of DQC is nice.

### Weaknesses
1. Theorem 4.6 is somewhat idealized. It depends on the open-loop consistency assumption. It is hard to hold in the practice for realistic offline datasets, especially in long-horizon settings, in such cases $\epsilon_h$ may not be small, and thus the errors scale and then the bound can become vacuous. Could the authors provide an empirical analysis on how big is $\epsilon_h$ in the OGBench datasets?
2. Theorem 4.6 does not seem to cover the decoupled critics. Is this bound still valid when the two critics are trained with different horizons and objectives?

### Questions
1. The distilled partical critic $Q_\psi^P$ is not Bellman-consistent with the long-horizon cirtic $Q_\phi$. $Q_\psi^P$ is trained to approximate an optimistic projectiion of $Q_\phi$ rather than its bellman target, thus there's an inherent objective mismatch between the two critics. Will it introduce systematic bias or training instability? Especially, when in non-markovian or sparse-reward settings, the optimistic assumption might not hold. Could the authors clarify why this training remains stable and effective in practice? It would be helpful to see a quantitive analysis showing how closely $Q_\psi^P$ tracks $Q_\phi$.

2. How the key hyperparameters were chosen, for example, $h$ and $h_a$? How were these values selected in practice—through tuning, fixed ratios, or heuristics? And how sensitive is the method to their choice?

3. What is the computational overhead of maintaining two critics with different horizons?

4. In Algorithm, it introduces two expectile parameters, $\tau_d$ and $\tau_b$ for the two critics. However, I do not find how the two factors are selected or tuned in Table 4. Expectile regression is known to be sensitive to $\tau$. It would amplify overestimation bias when using a large $\tau$. Are $\tau_b$ and $\tau_d$ the same across envs, or task-specific? Did the authors observe significant sensitivity in performance when varing $\tau_b$ and $\tau_d$?

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
This paper proposes Decoupled Q-Chunking (DQC), an offline RL algorithm that mitigates bootstrapping bias in long-horizon learning by evaluating long action chunks while predicting only short ones. By decoupling the policy and critic chunk lengths, DQC eases policy learning without sacrificing the benefits of multi-step value propagation. Theoretically, it formalizes the open-loop bias in chunked TD backups and proves when chunked critics outperform standard $n$-step returns. Empirically, DQC consistently outperforms prior state-of-the-art methods—including SHARSA—on the most challenging OGBench tasks.

### Strengths
1. The core idea of decoupling the policy and critic chunk sizes is novel, effectively addressing a known trade-off in multi-step Q-learning to get "the best of both worlds."
2. The paper provides deep theoretical backing for Q-learning with action chunking, formally identifying and quantifying bias, and proving the conditions under which the approach is superior.
3. It demonstrates superior, state-of-the-art results on challenging long-horizon tasks, significantly outperforming previous methods on the OGBench benchmark.
4. The work includes comprehensive experiments, comparisons to relevant baselines, and ablation studies that strongly support the authors' claims.

### Weaknesses
1. The approach still suffers from the inherent bias of open-loop value evaluation in action chunking and lacks a mechanism to actively correct it.
2. Its theoretical guarantees rely on a strong "open-loop consistency" assumption for the offline dataset, which may not hold in many real-world scenarios, limiting the generality of the claims.
3. The use of a fixed, global chunk size for both the policy and critic is a limitation, as the optimal action horizon might vary depending on the state.
4. The framework introduces additional components (e.g., two Q-networks, best-of-N sampling) and hyperparameters, increasing computational overhead and the practical difficulty of tuning and deployment.

### Questions
in weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work improves on the concept of chunked critics in the context of off-policy RL.
The chunk length of the critic and policy is decoupled by optimizing the policy against a distilled critic, allowing the policy to output shorter chunks.

### Strengths
- Well-written and detailed theoretical investigation.
- Strong and robust results.

### Weaknesses
- Some minor grammatical errors (e.g., l. 26, 182)
- There is no discussion of the computation overhead compared to the baseline methods (e.g., from maintaining the additional distilled critic). Since the performance gap to the baselines is substantial, providing a short intuition should suffice.

### Questions
- Notably, the only experiment where the proposed method is outperformed is by the $NS$ approach in the puzzle-4x6-1B environment. Can you provide an explanation/intuition of why this is the case? Are there certain environmental characteristics where you would expect your approach to perform worse than computationally simpler methods?

### Soundness
4

### Presentation
3

### Contribution
3
