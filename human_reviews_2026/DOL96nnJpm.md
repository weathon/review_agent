# Offline Reinforcement Learning Through Trajectory Clustering and Lower Bound Penalisation

- Decision: Reject
- Scores: 4, 4, 4, 4, 6

## Abstract
In this paper, we propose a new framework for value regularisation for offline reinforcement learning (RL). While most previous methods evade explicit out-of-distribution (OOD) region identification due to its difficulty, our method explicitly identifies the OOD region, which can be non-convex depending on datasets, via a newly proposed trajectory clustering-based behaviour cloning algorithm. With the obtained explicit OOD region, we then define a Bellman-type operator pushing the value in the OOD region to a tight lower bound while operating normally in the in-distribution region. The value function with this operator can be used for policy acquisition in various ways. Empirical results on multiple offline RL benchmarks show that our method yields the state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a value-penalized offline RL algorithm. It first identifies OOD actions (for computing the target value), i.e., whether $a' \in OOD(s')$ using a clustering method that addresses the challenges posed by heterogeneous datasets. It then applies a modified Bellman operator that replaces OOD values with lower-bound values.

The paper identifies OOD actions in heterogeneous datasets using a clustering algorithm, which appears to be directly borrowed from existing meta-RL literature, specifically variBAD [1].

The second focus of this work is the penalization of these OOD values (while the standard Bellman operator is applied to in-distribution data). The penalization is achieved by setting the target value to: $V(s) - r_{max} + r_{min} - \gamma K_V K_P D(a,a')$

The experimental section mainly focuses on MuJoCo tasks. Given that the main contribution of this paper lies in the theoretical justification, I believe it is not strictly necessary to include comparison results on other tasks.

[1] Luisa M. Zintgraf, Sebastian Schulze, Cong Lu, Leo Feng, Maximilian Igl, Kyriacos Shiarlis, Yarin Gal, Katja Hofmann, and Shimon Whiteson. VariBAD: Variational Bayes-Adaptive Deep RL via Meta-Learning. JMLR 22 (2021): 1–39.

### Strengths
The analysis in Section 4.2 is excellent. It proves not only the convergence of the customized Bellman operator (γ-contraction) but also the optimality of the resulting policy.

### Weaknesses
W1 Novelty of Section 4.1.
This section describes the method for identifying OOD actions in (more realistic) heterogeneous datasets, which seems to constitute about half of the paper’s contribution. However, this component appears to be directly borrowed from existing meta-RL literature, particularly variBAD [1]. It would be the AC’s responsibility to verify the novelty and contribution of this part to the community.

W2/Q1 Organization and clarity.
The organization of the paper is somewhat confusing. The main focus is clearly the theoretical analysis of the proposed penalized Bellman operator. However, I found it difficult to connect the theoretical formulation to its practical implementation. The related concerns are listed in the next section.

### Questions
How are the values of $K_V$ and $K_P$ in Equation (7) determined when computing the target value for OOD actions in the experiments?

### Soundness
2

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
2

### Summary
This paper proposes a new framework for value regularisation in offline reinforcement learning (RL) that explicitly identifies out-of-distribution (OOD) regions, rather than approximating them indirectly as prior methods do (e.g., CQL, SVR, or mild regularisation approaches).

Following are the key ideas:

1. Trajectory clustering for OOD identification: The authors develop a trajectory-level clustering algorithm that models the data as arising from multiple unknown behaviour policies. Using a meta-learning–style latent variable model inspired by VariBAD but modified with state-space layers (S5) and a VQ-VAE structure, the method assigns trajectories to clusters representing different behaviour policies. This explicit clustering allows identification of in-distribution (ID) and OOD action sets, even when the dataset is heterogeneous or multi-modal.

2. Lower Bound Penalisation (LBP): Once the OOD region is determined, the method introduces a Bellman-type operator that penalises critic Q-values of OOD actions by regressing them towards a tight lower bound of the optimal value function Q*. The lower bound is derived under Lipschitz continuity assumptions on the transition and value functions, ensuring that the penalised values remain conservative but not overly pessimistic. In-distribution actions are left unpenalised, addressing the over-regularisation issue of prior methods like CQL.

### Strengths
1. Novel integration of trajectory clustering into offline RL: The idea to use trajectory-level clustering to infer multiple behaviour policies is  novel, and goes beyond existing techniques such as density-based OOD estimation methods. Authors provide a new practical approach for explicitly identifying in-distribution and OOD action regions, which is novel for offline RL. While theoretical grounding may be limited, the design idea is fresh and potentially useful.

2. Empirical rigor and diversity of experimental evaluation: Algorithm is tested across both standard D4RL benchmarks and synthetically mixed datasets (combining random, medium, and expert trajectories). The reported results consistently outperform CQL, TD3+BC, and other baselines, showing strong empirical consistency.

### Weaknesses
Though the idea of estimating OOD regions is novel, I feel that the novelty of the paper is somewhat limited and results not surprising.

1. The paper assumes that clustering over trajectories can reliably separate distinct behaviour policies. However, there is no theorem or formal guarantee that the latent clusters correspond to true behavioural modes or that the learned clusters align with distinct data-generating policies. Without such theoretical validation, the separation between "in-distribution" and "out-of-distribution" regions is empirically plausible but not provable, which weakens the foundation.

2. Missing quantitative link between theoretical lower bound and empirical robustness: It is claimed that the proposed lower-bound penalisation leads to improved robustness and sample efficiency.
However, no analytical relation or empirical metric is provided to connect the derived lower bound to the actual generalisation behaviour or to OOD performance metrics. The validation is empirical and qualitative.

### Questions
1. Assumption 1 seems hard to justify. Variations in action space do not depend upon state value. However, we know that in many commonly encountered systems, the dynamics (and hence most likely differential of probabilities) do depend upon the current state.

2. Similarly, Assumption 2 the r.h.s. bound does not depend upon \beta.

3. Please point out explicitly the formal guarantee that the latent clusters correspond to true behavioural modes or that the learned clusters align with distinct data-generating policies.

4. Please point out the quantitative link between theoretical lower bound and empirical robustness.

I would be happy to increase score if 3. and 4. above are addressed.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a framework for explicit OOD action identification in offline RL and a new lower‑bound value regularizer to penalize Q-values only outside the estimated in‑distribution (ID) set.

### Strengths
Most value‑regularization methods avoid explicit OOD delineation; this paper defines ID via analytically computed HDRs (Gaussian) and unions across clusters. The closed‑form HDR for Gaussians and the link to confidence regions are clear and practical.

Formulating trajectory clustering with a discrete latent "task" and training an amortized posterior (VQ‑VAE + S5 encoder) is a nice adaptation of variBAD‑style ideas to offline RL. The ablation in Table 4 and extended Table 6 shows the modeling choices (transition on, reward off) matter and supports the claim.

The derived Q-lower bound is tight and connects to Wasserstein Lipschitzness of dynamics and Lipschitz. The contraction‑mapping based results for the penalized operator are careful and readable.

Misc. Notes:
The paper offers competitive empirical results, informative figures, and the presentation + organization are sufficient.

### Weaknesses
The tightness and usefulness of $Q_{\beta}^{LB}$ hinge on $K_V, K_P$. I could not find how these are estimated or tuned in practice; yet they appear in the penalty target throughout (Sec. 4.2 p.5–6; Appx. C.3 p.22–23). Without principled estimation or sensitivity analysis, the regularizer may effectively collapse back to $r_{min} / 1 - \gamma$. Please describe how $K_V, K_P$ are chosen, (ii) include an ablation showing performance vs. $K_V, K_P$.

The use of extra datasets to set $r_{min}, r_{max} may break comparability. Clarify whether competing baselines also used cross‑dataset reward ranges; if not, redo with per‑dataset ranges or discuss fairness.

Table 2 appears to omit the “halfcheetah‑medium‑expert” score for the paper's method (Ours) and does not report an Average for Ours, although other methods have an average row. This impedes a balanced comparison.

Stage III samples OOD actions by drawing from a large hypercube in $tanh^{-1}$ space and rejecting ID points. In higher‑dimensional actions this can be extremely inefficient; no acceptance‑rate analysis or runtime is provided.

The theory assumes each trajectory is generated by a single stationary behavior, but many offline datasets (e.g., medium‑replay) arise from non‑stationary policies within a trajectory. The authors acknowledge this (Limitations section), yet the clustering encoder uses action‑less sequences, which may struggle when rewards are noisy and state overlaps across behaviors are large. Please (i) evaluate on truly mixed‑policy trajectories (e.g., replay‑style) with known switches, (ii) compare to encoders that also see actions.

### Questions
How are $K_V, K_P$ set in practice? Are they tuned per task, fixed across tasks, or estimated from data (e.g., via empirical Lipschitz upper bounds)? Please include a sensitivity figure.

Did all baselines also compute $r_{min}, r_{max}$ across the union of datasets, or only from the target dataset? If not, could you rerun with per‑dataset ranges?

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
3

### Summary
The authors point out that many existing offline RL methods sidestep explicit OOD detection and instead rely on indirect conservatism (e.g., value or policy penalties), which can break down when the logged data come from a multi-modal behavior distribution. To address this, the paper introduces a trajectory-based behavior modeling approach that clusters trajectories and uses the resulting per-cluster behavior policies to explicitly characterize the in-distribution and OOD region. On top of this, the authors derive a Bellman-style operator that replaces or pushes down Q-values for OOD actions using a theoretically motivated lower bound, thereby reducing overestimation in unsupported regions. Empirically, they evaluate on standard D4RL benchmarks and on a constructed heterogeneous dataset, reporting improvements over representative offline RL baselines.

### Strengths
1. Many offline RL methods do not explicitly determine which state–action pairs are OOD, but instead apply global/indirect conservatism, which can inadvertently penalize valid but low-frequency actions when the data happen to be heterogeneous or multi-modal.

2. This paper proposes a method to prevent overestimation of the Q-values for OOD state–action pairs in offline RL. Instead of directly penalizing them, the authors first identify the OOD regions and then introduce a new Bellman operator to push down the values of the state–action pairs in those regions.

3. The paper is well structured and easy to follow

### Weaknesses
1. Problem setup not fully validated. The paper’s main motivation is that existing OOD-penalization methods can fail on heterogeneous / multi-modal datasets. However, the authors also note in Section 5.1 that many of the standard offline-control benchmarks they use (including several D4RL tasks) are effectively closer to uni-modal at the state–action level. Since the paper does not quantify how heterogeneous the actual benchmarks are, it is hard to tell how often the proposed clustering-based OOD identification is really needed.

2. The positive-definite covariance assumption is strong. Around line 163 the paper assumes that, for every state  $s$, the behavior policy’s covariance $\Sigma(s)$ is positive definite. For offline data, especially when some states are rarely visited or when actions concentrate on a low-dimensional manifold, this is not guaranteed. The paper should at least discuss how to handle rank-deficient / ill-conditioned covariances.

3. Theory relies on strong global continuity assumptions. The main Bellman-type result depends on 1-Wasserstein continuity of the dynamics in the action space and Lipschitz continuity of the value in the state space. While these assumptions are intuitively reasonable (“nearby actions/states behave similarly”), they are fairly strong and may not hold in environments with mode switches or highly non-smooth rewards. The paper should provide evidence or discussion about how broadly these assumptions apply in common offline RL benchmarks.

4. Missing key ablations. The method critically depends on correctly identifying the number of behavior modes $K$ and on the quality of the trajectory-based behavior model, but there is no ablation on mis-specified  $K$, on different HDR levels, or on simpler clustering baselines. This makes it difficult to assess robustness.

5. Complexity vs. gains is unclear. Compared with prior offline RL methods, the proposed approach introduces extra components (sequence/trajectory modeling, VQ-VAE, per-cluster behavior estimation) to enable explicit OOD detection, but the reported performance improvements over strong baselines are modest. A clearer comparison of computational/training complexity versus returned performance would strengthen the practicality claim.

6. Limited evidence for real/practical settings. Because OOD identification here is tied to the learned VAE-style behavior model and is only evaluated on MuJoCo-style (and partly synthetic/heterogeneous) datasets, it is not clear how well the approach would scale to larger or noisier real-world logs, where behavior is messier and coverage is sparser. A discussion or experiment in a more realistic domain would be helpful.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel framework for value regularization in offline Reinforcement Learning (RL) that directly addresses the challenge of Out-of-Distribution (OOD) value overestimation by explicitly identifying the OOD region and imposing a tighter lower bound penalty.

### Strengths
1. While previous methods avoid identifying the OOD region directly, this work introduces a methodology to explicitly define it, acknowledging that this region can be non-convex. The core mechanism uses the Highest Density Region (HDR) concept, a rigorous statistical tool, to set an adaptive likelihood threshold for OOD status, which is a highly original approach in RL.

2. The use of a specialized trajectory clustering algorithm, which reformulates the clustering problem as a task inference problem akin to meta RL (variBAD), is highly innovative. This approach is tailored to heterogeneous datasets and utilizes sequence modeling (S5 architecture) to learn effective trajectory representations, avoiding the information loss associated with simple averaging used in prior work.

3. The paper provides formal theoretical results, including proving the existence of a unique fixed point for the proposed penalized Bellman optimality operator.

### Weaknesses
1. The entire framework is built upon a fundamental assumption that may not hold true in real-world scenarios. The work assumes that each trajectory in the dataset was sampled from a single, uni-modal behavior policy.

2. The adaptive two-phase training paradigm used to automatically determine the number of clusters is a practical heuristic. It relies on a manually set threshold for removing clusters, adding another layer of tuning that is not theoretically derived.

### Questions
1. The selection of the confidence level $\alpha$ for the HDR is crucial yet left unspecified beyond a general principle. How sensitive is the final policy performance (e.g., average normalized score) to the choice of $\alpha$? Can the authors provide an ablation study on $\alpha$ across different D4RL domains? This is vital for showing the method's practical robustness.

### Soundness
3

### Presentation
2

### Contribution
3
