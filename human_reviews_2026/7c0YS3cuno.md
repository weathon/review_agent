# MOBODY: Model-Based Off-Dynamics Offline Reinforcement Learning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
We study off-dynamics offline reinforcement learning, where the goal is to learn a policy from offline source and limited target datasets with mismatched dynamics. Existing methods either penalize the reward or discard source transitions occurring in parts of the transition space with high dynamics shift. As a result, they optimize the policy using data from low-shift regions, limiting exploration of high-reward states in the target domain that do not fall within these regions. Consequently, such methods often fail when the dynamics shift is significant or the optimal trajectories lie outside the low-shift regions. To overcome this limitation, we propose MOBODY, a Model-Based Off-Dynamics Offline RL algorithm that optimizes a policy using learned target dynamics transitions to explore the target domain, rather than only being trained with the low dynamics-shift transitions.  For the dynamics learning, built on the observation that achieving the same next state requires taking different actions in different domains, MOBODY employs separate action encoders for each domain to encode different actions to the shared latent space while sharing a unified representation of states and a common transition function. 
We further introduce a target Q-weighted behavior cloning loss in policy optimization to avoid out-of-distribution actions, which push the policy toward actions with high target-domain Q-values, rather than high source domain Q-values or uniformly imitating all actions in the offline dataset.  We evaluate MOBODY on a wide range of MuJoCo and Adroit benchmarks, demonstrating that it outperforms state-of-the-art off-dynamics RL baselines as well as policy learning methods based on different dynamics learning baselines, with especially pronounced improvements in challenging scenarios where existing methods struggle.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
**Summary.** This work studies how to tackle the problem of off-dynamics offline RL. More precisely, off-dynamics offline RL aims to learn a policy using offline data from a source domain and a much smaller amount of data from a target domain. Main challenge limits exploration and adaptation to the target's optimal trajectories when the dynamics shift is significant. To alleviate this issue, this paper introduces MOBODY that explicitly learns an unified state representation and separate action encoders for source and target domains. They further introduce a Q-weighted behavior cloning technique to bias the policy toward high-value actions instead of naive action imitation. The method achieves consistent improvements over strong baselines on MuJoCo and Adroit benchmarks with different types and severities of dynamics shift.

---
**Review summary.** Overall, the reviewer thinks that the paper is clearly written, well-motivated, and timely, especially in the context of sim-to-real transfer. The idea itself is conceptually simple, intuitively reasonable, and appears to be practical to implement. Although the work is not theoretically deep, it is coherent and sufficiently justified as a framework-oriented contribution. However, the experiments section leaves several open questions and limitations that need further clarification. Therefore, the reviewer assigns an initial score of 6 and plan to revisit this rating after the authors address the concerns and questions raised in this review.

### Strengths
**Writing**
- The paper is very readable and logical from motivation to formulation to algorithm and experiments.
- Figures and tables are generally informative; the method and notation are consistent.

---
**Methodology**
- The central idea, learning shared latent state and transition functions with domain-specific action encoders, is simple and well-motivated.
- The architecture (shared $\phi_E$, $\phi_T$; separate $\psi_{\text{src}}$, $\psi_{\text{trg}}$) directly targets the off-dynamics challenge, that is, achieving the same actions but different dynamic transtion.
- The introduction of a Q-weighted BC is reasonable, encouraging the policy to favor actions with high target-domain value while mitigating overfitting to source data.

---
**Experiments**
- The experiments are extensive, covering MuJoCo with diverse transition (e.g., gravity/friction/kinematic/morphology shifts) and Adroit tasks with multiple difficulty levels.
- Ablation studies demonstrate the impact of both the cycle transition loss and Q-weighting.
- Computational cost, hyperparameter sensitivity, and reproducibility are reported in detail.

### Weaknesses
**Writing**
- Figures 3–4 summarize results but are under-analyzed. differences and error ranges are not interpreted in the text. Please discuss more why and how come.
- Algorithm 1 and 2 include many moving parts, such as DARA regularization and reward modeling, that are not clearly integrated into the paper.
 
**Methodology**
- It is not fully clear how the encoder losses (Eq. 4) interact within the overall objective (Eq. 6). Are $L^{\text{src}}{\text{rep}}$ and $L^{\text{trg}}{\text{rep}}$ averaged or weighted differently?
- The paper should clarify whether $p(s'|s,a)$ is assumed deterministic. This strongly affects the validity of the VAE-style cycle transition loss and the interpretation of uncertainty.
- The cycle transition loss (Eq. 5) is creative, but its formal grounding is weak: the gradient-stopping mechanism and KL structure are under-explained, and it is unclear how it avoids mode collapse or degenerate representations.
- The target-domain encoder $\psi_{\text{trg}}$ is trained with very limited data. Overfitting and instability risks should be analyzed or mitigated via regularization or weight sharing.
- Target Q-weighted BC is empirically validated but lacks theoretical analysis; no formal bound or convergence argument is provided, and the technique is a relatively mild variation of AWR.

**Experiments**
- The reviewer concerns baseline selection and fairness.
    - Most baselines (IQL, TD3-BC, DARA, BOSA, MOPO) are strong but somewhat outdated.
    - More recent or structurally related methods (e.g., BPRL, LEO, Koopman Q-learning, or model-based domain adaptation approaches) are missing.
- In several tasks, simpler methods, IQL, outperform off-dynamics-specific methods, i.e., DARA, BOSA. This discrepancy deserves explicit discussion.
- Only two dynamics domains are considered. It is unclear how MOBODY generalizes to multiple or continuous domains.
- High variance in results, for example, Walker2d-Gravity 5.0: $46.05 \pm 20.73$, suggests instability that is not analyzed.
- There is no qualitative evidence supports cross-domain transfer.

**Limitations**
- MOBODY assumes that a shared transition mapping ($\phi_E$, $\phi_T$) exists across domains. This is a strong assumption; when dynamics are structurally divergent, the approach may fail.
- The approach relies on uncertainty regularization from model ensembles, but this is treated superficially. 
- There is no analysis of behavior under extreme dynamics mismatch, where shared representations might be misleading.
- Need to more discuss and survey about prior connections.

**Miscellaneous**
- Eq. (3): inconsistent summation notation and formatting.
- Eq. (6): unclear aggregation of $L_{\text{rep}}$ terms.
- Line 165: justification for adding $z_s$ to $\psi(z_s,a)$ is weak (“for simplicity” is insufficient).
- Line 126–127: references needed for conservative reward regularization (e.g., MOPO, COMBO).
- In several places, “off-dynamics RL” should be replaced with “off-dynamics offline RL” for accuracy.

**References**
- M. Weissenbacher et al. Koopman Q-learning: Offline Reinforcement Learning via Symmetries of Dynamics. ICML 2022.
- J. Eo, et al. The impact of dataset on offline reinforcement learning performance in UAV-based emergency network recovery tasks. IEEE Comm. Let. 2023.
- M. Uehara, et al. Pessimistic model-based offline reinforcement learning under partial coverage. ICLR 2022.
- B. Hao, et al. Bootstrapping fitted q-evaluation for off-policy inference. ICML 2021.
- G. An, et al. Uncertainty-based offline reinforcement learning with diversified q-ensemble. NeurIPS 2021.
- D. Lee, et al. Temporal Distance-aware Transition Augmentation for Offline Model-based Reinforcement Learning. ICML 2025.

### Questions
- In Table 1, IQL often outperforms specialized methods such as DARA and BOSA. Could the authors explain why this occurs?
    - Have the authors tried using AWR style policy extraction for DARA or BOSA to see whether the observed gap stems from the extraction mechanism rather than the dynamics modeling?
- For Ablation 1, would it be possible to combine alternative dynamics models, like COMBO or RAMBO, with the proposed Q-weighted BC to isolate the contribution of the policy regularizer?
- For Ablation 2, could the authors replace the Q-weighted cloning loss with an AWR to directly compare different policy extraction approaches?
- BOSA appears conceptually compatible with MOBODY’s dynamics learning. Have the authors tested whether integrating the two (BOSA + MOBODY dynamics) yields complementary effects or insights into the efficacy of Q-weighted behavioral cloning?
- Given the extremely small target dataset, have the authors observed any instability or collapse in the target encoder $\psi_{\text{trg}}$?
- How does MOBODY perform when the amount of target-domain data decreases further, e.g., below 1 K samples?
- How many ensemble models are used for uncertainty quantification? 
    - Is the performance sensitive to ensemble size or the uncertainty penalty $\beta$?
- The reviewer wonders what the aurhos predict will happen if we consider a representation based on the TD information of each domain.
    - A. Zhang, et al. Learning Invariant Representations for Reinforcement Learning without Reconstruction. ICLR 2021.
    - C. Zheng, et al. Contrastive Difference Predictive Coding. ICLR 2024.
    - D. Lee, et al. Temporal Distance-aware Transition Augmentation for Offline Model-based Reinforcement Learning. ICML 2025.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the "off-dynamics offline reinforcement learning" (ODRL) problem, where the goal is to learn a target-domain policy using a large source-domain dataset and a very small target-domain dataset, which have different dynamics. The authors propose MOBODY (Model-Based Off-Dynamics Offline RL), an algorithm that learns a model of the target dynamics to generate synthetic data, allowing the policy to explore high-reward regions that may be in high-shift areas, which existing methods often fail to do.

### Strengths
1. The paper correctly identifies a key weakness of current ODRL methods (e.g., DARA, BOSA): their conservatism, which stems from data filtering or reward penalization, prevents them from exploring high-reward states that may exist outside of low-dynamics-shift regions.
2. The central idea of using separate action encoders while sharing state and transition functions is a clever and intuitive way to leverage the large source dataset to learn shared structural knowledge while still capturing the domain-specific differences with the small target dataset.

### Weaknesses
1. The paper fails to cite or compare against a relevant class of transfer learning and offline RL methods. The related work discusses reward regularization (DARA) and data filtering (VGDF), but it omles a large body of work on policy regularization, specifically methods that use state-distribution matching or regularization. For example, State Regularized Policy Optimization (SRPO) and similar approaches that explicitly constrain the policy's state-visitation frequency to be close to a trusted distribution (like the target dataset) are highly relevant. Such methods provide an alternative and potentially more direct way to handle the dynamics shift than the complex model-based approach proposed here.
2. While the dynamics learning architecture is interesting, the policy learning side feels incremental and like a complex combination of existing ideas.
- The "enhanced target data" buffer is a simple-enough combination of target data, model rollouts (a standard model-based RL technique, e.g., MOPO), and augmented source data (using the existing DARA method).
- The "target Q-weighted behavior cloning loss" (Eq. 8) is presented as a novel contribution but is a minor variant of advantage-weighted regression, which the paper itself cites as an inspiration (IQL). The paper provides no theoretical or strong empirical justification (e.g., a direct comparison) for why its specific formulation ($\exp (Q / \operatorname{avg}|Q|) \times(\pi(s)-a)^2$) is superior to the more established and better-understood formulations used in methods like IQL.
3. The final MOBODY algorithm is highly complex. It stitches together a VAE-style dynamics learner with an ensemble for uncertainty, the full DARA framework for reward augmentation, and a custom Q-weighted policy optimizer. This high number of "moving parts" makes the method difficult to reproduce, tune, and analyze, and it obscures which components are providing the most benefit.
4. The motivation for the Q-weighted loss is purely intuitive. There is no analysis of its properties. Why use an exponential weight? Why normalize by the average absolute Q-value? How sensitive is this to the Q-value estimates, which are known to be noisy? A more principled derivation or analysis is needed to justify this specific design over simpler, existing alternatives.

### Questions
1. Could you elaborate on why methods that directly regularize the policy's state-visitation distribution (like SRPO) were not considered as baselines, given that they directly address the problem of distribution shift under new dynamics?

2. The Q-weighted BC loss (Eq. 8) is very similar to IQL's advantage-weighted regression. What is the performance of your method if you simply replace your loss with the IQL policy loss? This would clarify whether the specific formulation of Eq. 8 is a key part of your contribution.

3. How much of the performance gain comes from the model-based rollouts versus the superior dynamics-learning architecture? (i.e., what is the performance of MOPO if it uses your dynamics model?)

### Soundness
3

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
3

### Summary
The paper proposes MOBODY, a model-based approach to off-dynamics offline reinforcement learning, addressing the setting where the source and target environments share rewards but differ in dynamics. Unlike prior methods that penalize or exclude high-shift samples, MOBODY leverages both limited target data and abundant source data through separate action encoders and a shared latent transition model. A cycle-transition loss is introduced to stabilize representation learning, while a target-Q-weighted behavior cloning loss guides the policy toward high-value target actions. Experimental results on MuJoCo and Adroit benchmarks show that MOBODY consistently outperforms existing off-dynamics and model-based baselines, particularly under large dynamics shifts.

### Strengths
- The paper introduces a clear and well-motivated model-based framework for off-dynamics offline reinforcement learning. Unlike prior work that relies on reward regularization or data filtering, it directly learns the target dynamics, offering a fresh and practical way to handle dynamics mismatch in offline settings.
- The paper introduces MOBODY, which shows solid and consistent improvements over existing baselines on several off-dynamics offline RL benchmarks.
- This paper provides a comprehensive empirical evaluation across both MuJoCo and Adroit environments, covering multiple types of dynamics shifts—gravity, friction, kinematics, and morphology—at varying levels of severity.
- The paper is well written, with clear organization and logical flow throughout. In particular, the results are effectively visualized—figures and tables are well designed, easy to interpret, and enhance the overall clarity and impact of the presentation.

### Weaknesses
- Lack of theoretical analysis supporting the proposed approach. The paper does not provide any convergence guarantees, error bounds, or stability analysis, which are important for understanding the conditions under which the algorithm is expected to perform reliably [1-2]. Given the complexity of the model and the presence of domain shifts, some theoretical insights—even in simplified settings—would significantly enhance the rigor and credibility of the work.
- While MOBODY generally outperforms existing baselines, some of the reported improvements appear marginal or potentially within the range of statistical noise—for example, in the Walker2d-Friction-2.0 setting, where the gains over baselines are minimal. Moreover, in a few scenarios, MOBODY underperforms relative to other methods. The paper would benefit from a more detailed analysis or discussion of these cases to help contextualize performance variance and identify potential limitations of the approach.
- All experiments are conducted in dense-reward environments, which may not fully capture the challenges faced in sparse-reward settings. It would strengthen the paper to include an analysis of MOBODY's performance in sparse environments, where exploration and generalization are typically more difficult [3-4].
- Although the method is described as operating with "limited target data," it still assumes access to 5,000 transitions, which may not reflect truly scarce or low-resource settings. It remains unclear how MOBODY would perform when target data is extremely limited, noisy, or when the dynamics mismatch is highly non-smooth or discontinuous. Evaluating the sensitivity of the model under such challenging conditions would help clarify the robustness and practical applicability of the approach.
- The proposed framework introduces multiple components—such as separate action encoders for the source and target domains (ψ_src, ψ_trg) and a VAE-based cycle transition module—which likely increase both training time and memory usage. However, the paper does not provide any runtime, training cost, or sample-efficiency comparisons against baseline methods, making it difficult to assess the practical overhead of the approach.
- While the paper compares MOBODY against several strong baselines, it overlooks a class of return-conditioned or return-based methods that have also demonstrated competitive performance in off-dynamics offline RL settings [5]. Including representative algorithms from this category—such as decision transformer variants or return-augmented models—would provide a more comprehensive evaluation and help clarify whether the proposed approach offers advantages beyond standard model-based and reward-regularized baselines.

### Questions
- Could the authors provide any theoretical guarantees (e.g., convergence, stability, or error bounds) for MOBODY, even under simplified assumptions? How does the model behave under known pathological dynamics mismatches?
- In settings where MOBODY's improvements over baselines are small (e.g., Walker2d-Friction-2.0), or where it underperforms, what factors contribute to this variance? Could the authors provide an analysis or hypothesis for these outcomes?
- How does MOBODY perform in sparse-reward environments, where exploration is more challenging and informative transitions are rare? Could the authors include results or discussion in such settings?
- How sensitive is MOBODY to the quantity of target data? Have the authors evaluated performance when fewer than 5,000 transitions are available?
- Several return-conditioned or return-based algorithms (e.g., Decision Transformers, RADT, or return-augmented models) have shown promising results in off-dynamics offline RL. How does MOBODY compare conceptually and empirically to such return-based approaches?

## Reference
- **[1]** Miyaguchi, Kohei. "Worst-case offline reinforcement learning with arbitrary data support." Advances in Neural Information Processing Systems 37 (2024): 61481-61512.

- **[2]** Brandfonbrener, David, et al. "When does return-conditioned supervised learning work for offline reinforcement learning?." Advances in Neural Information Processing Systems 35 (2022): 1542-1553.

- **[3]** Cen, Zhepeng, et al. "Learning from sparse offline datasets via conservative density estimation." arXiv preprint arXiv:2401.08819 (2024).

- **[4]** Rengarajan, Desik, et al. "Reinforcement learning with sparse rewards using guidance from offline demonstration." arXiv preprint arXiv:2202.04628 (2022).

- **[5]** Wang, Ruhan, et al. "Return augmented decision transformer for off-dynamics reinforcement learning." arXiv preprint arXiv:2410.23450 (2024).

### Soundness
3

### Presentation
3

### Contribution
3
