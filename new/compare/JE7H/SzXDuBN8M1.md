---
job_id: cd71f0c7-cdc7-4352-8c07-36e7ab6a4e1f
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: SzXDuBN8M1.pdf
paper: TD-JEPA: Latent-Predictive Representations for Zero-Shot Reinforcement Learning
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper is squarely about self-supervised / latent-predictive representation learning for zero-shot reinforcement learning, fully within ICLR’s core scope.

## Minimum Quality
Pass ✅. The paper is complete (abstract, introduction, related work, method, theory, experiments, conclusion), technically nontrivial, written in English, with substantial experiments and theory. No obvious fatal methodological or evaluation flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. I do not see any attempts at manipulating automated reviewing or hidden prompts in the provided content.

---

# Expected Review Outcome:

## Summary

The paper proposes TD-JEPA, a temporal-difference (TD) latent-predictive framework for learning state and task representations, policy-conditioned multi-step predictors, and a family of policies from offline, reward-free data, enabling zero-shot optimization for arbitrary downstream rewards in the span of the learned task encoder. 

At the core is a TD-based latent-predictive loss (Eq. 7 / 9) that approximates successor features of multiple policies using off-policy transitions, with an asymmetric variant that learns separate state encoder $\phi$ and task encoder $\psi$. 

The authors provide a theoretical analysis in a linear/tabular setting, connecting their MC and TD latent-predictive losses to low-rank factorization of successor measures and to policy evaluation error bounds, and empirically evaluate on 65 tasks across ExoRL and OGBench (proprioceptive and visual), where TD-JEPA matches or improves upon strong zero-shot baselines such as FB, HILP, RLDP, BYOL-* and BYOL-$\gamma$-*. 


## Strengths

1. **Conceptual contribution: TD latent-predictive representations for multi-policy successor features.**  
   The main idea of combining JEPA-style latent prediction with TD learning to approximate policy-conditional successor features in latent space is clear and compelling. Equations (5)–(9) articulate a natural progression from MC latent prediction of successor features ($\mathcal{L}_{\text{MC-JEPA}}$) to a TD version ($\mathcal{L}_{\text{TD-JEPA}}$) that is estimable from offline data, and then to an asymmetric variant with separate state and task encoders. This bridges the empirical practice of self-predictive losses (BYOL-style) with the successor-measure factorization viewpoint of FB / ICVF in a fairly clean way.

2. **Nontrivial theoretical analysis tying JEPA-style losses to successor-measure approximation.**  
   Section 4 contains substantive theory. In particular:
   - The successor-measure loss $\mathcal{L}_{\mathrm{SM}}$ in Eq. (10) formalizes the goal of learning a multilinear factorization $M^{\pi_z}\approx \phi T_z \psi^\top$.  
   - Theorem 1 shows that, under Assumptions A1–A3, the optimal latent-predictive MC predictors and the optimal $\mathcal{L}_{\mathrm{SM}}$ predictors coincide and that the gradients w.r.t. $\phi,\psi$ match, yielding orthogonal projections $\Pi_\phi M^{\pi_z}\psi$ and $\Pi_\psi M^{\pi_z}\phi$.  
   - Theorem 3 provides a similar “gradient matching” result in the TD case, showing that optimizing the TD-JEPA loss (Eq. 19) corresponds to optimizing forward/backward TD losses for the successor measure (Eqs. 11–12), with oblique projections $\tilde{\Pi}_{\phi,z},\tilde{\Pi}_{\psi,z}$.  
   - Theorem 4 upper-bounds the worst-case policy evaluation error (over rewards) by $\mathcal{L}_{\mathrm{SM}}$, tying representation learning directly to value approximation quality.  
   Together, these results are nontrivial and extend prior self-predictive theory beyond single-policy, one-step settings, in a way that is at least mathematically consistent and interesting.

3. **Clear, end-to-end zero-shot RL instantiation and algorithm.**  
   Algorithm 1 spells out the full TD-JEPA training pipeline, including:
   - two TD latent-predictive losses $\widehat{\mathcal{L}}_{\mathrm{TD-JEPA}}(\phi,T_\phi,\psi)$ and $\widehat{\mathcal{L}}_{\mathrm{TD-JEPA}}(\psi,T_\psi,\phi)$ (Eq. 9 form),
   - orthonormality regularizers $\widehat{\mathcal{L}}_{\mathrm{REG}}(\phi),\widehat{\mathcal{L}}_{\mathrm{REG}}(\psi)$,
   - an actor loss $\widehat{\mathcal{L}}_{\mathrm{actor}}$ that encourages $\pi_z$ to be optimal for reward $\psi^\top z$.  
   The connection from zero-shot reward inference via linear regression on $\psi$ to selecting $\pi_{z_r}$ is explicit and operational. The pseudocode in Listing 1 further clarifies practical details such as target networks and gradient flow.

4. **Strong and broad empirical evaluation with competitive baselines.**  
   Table 1 covers 13 datasets (DMC/DMCRGB, OGBench/OGBenchRGB) and a significant number of tasks, with baselines including Laplacian, FB, HILP, RLDP, BYOL-*, BYOL-$\gamma$-*, and ICVF*. On the key pixel-based suites, DMCRGB and OGBenchRGB, TD-JEPA achieves the best or tied-best average scores in most settings; for example:
   - DMCRGB avg: TD-JEPA $628.8 \pm 5.5$ vs BYOL-$\gamma^*$ $582.4 \pm 9.8$ and RLDP $525.7 \pm 13.3$.
   - DMC avg: TD-JEPA $661.2 \pm 6.3$ slightly above FB $648.2 \pm 4.1$ and other baselines.  
   The probability-of-improvement heatmaps in **Figure 2** provide a more global comparison, indicating TD-JEPA is among the most consistently strong methods across suites, with clear advantages over contrastive baselines in visual regimes.

5. **Insightful ablations that dissect what matters.**  
   Several ablations are well designed and actually informative:
   - **Figure 3 (left)** compares normalized performance between BYOL*, BYOL-$\gamma^*$, and TD-JEPA. It supports the claim that modeling multi-step, **policy-conditional** dynamics, rather than only the behavioral policy’s dynamics, is generally more beneficial, especially when data is not strongly expert-like.  
   - **Figure 3 (right)** quantifies the performance gain of asymmetric TD-JEPA over a symmetric variant (single encoder $\phi$). While gains are not uniform, the figure shows that separate state and task encoders are helpful on average, especially in more complex visual domains.
   - The symmetric contrastive vs symmetric latent-predictive comparison in Appendix D.2 (Table 3 and **Figure 5 (right)**) indicates that even with the same architecture and symmetry, the non-contrastive TD-JEPA-style loss yields better performance, particularly from pixels.
   - Architectural ablations in **Figure 7** highlight how encoder / predictor depth affects zero-shot performance, with a coherent story about giving encoders higher capacity and keeping predictors shallow.

6. **Fast adaptation / reuse of representations is convincingly demonstrated.**  
   **Figure 4** shows normalized performance when fine-tuning zero-shot solutions (TD-JEPA and FB) offline and online with TD3, compared to learning from scratch. The blue/yellow “frozen” curves show that keeping the representation fixed often yields almost the same learning curve as full fine-tuning, suggesting that the learned $\phi$ (and $\psi$) are already well aligned with the downstream value functions. Additional plots in **Figure 6** further explore variants (only encoders loaded, only convolutional weights frozen, OGBench tasks), adding credibility to the claim that TD-JEPA learns useful general-purpose control representations.

7. **Qualitative visualizations that connect geometry to dynamics.**  
   **Figure 8** and **Figure 9** depict cosine similarities between successor-feature predictions and goal embeddings in the antmaze-$\ell n$ task as a function of agent position, comparing TD-JEPA to BYOL*, BYOL-$\gamma^*$, HILP, RLDP, etc. These plots do a nice job of visually linking the learned latent geometry to shortest-path distances and directed goal-reaching behavior. **Figure 10** comparing PCA projections of TD-JEPA’s conv features vs DINO-v2’s for cube-single nicely illustrates that TD-JEPA focuses on control-relevant aspects (end-effector, cube) rather than generic segmentation.

8. **Reasonably thorough related work discussion and positioning.**  
   Sections 5 and A provide a detailed comparison to FB, HILP, RLDP, BYOL-$\gamma$, ICVF, VIP, MR.Q, and broader work on pre-trained vision models and unsupervised RL. The paper is clear about being a multi-policy, off-policy, non-contrastive successor-measure learner, and distinguishes itself from single-policy latent-predictive theory (Tang et al., Lawson et al.) and from value-based goals like VIP / ICVF.


## Weaknesses

1. **Mismatch between theory and the practical TD-JEPA algorithm.**  
   The theoretical results in Section 4 are stated for a simplified, *tabular, action-free, linear* setting with symmetric kernels and orthonormal features, while the main algorithm in Algorithm 1 is fully nonlinear, action-conditional, and uses asymmetric encoders. Several gaps are left unaddressed:
   - Theorem 1 and 3 assume $\phi^\top \phi = \psi^\top \psi = I$ and uniform state distribution (A1–A2) and symmetric $P^{\pi_z}$ (A3). The paper notes these can be relaxed (App. C), but the main theorems are not restated in the relaxed form, and there is no concrete statement that holds under realistic MDPs (asymmetric transition kernels) and non-orthonormal features. The gradient matching argument in App. C is quite abstract and never connected back to the exact nonlinear parameterization used in Algorithm 1.  
   - The algorithm uses *action-conditioned* predictors $T_\phi(\phi(s),a,z)$, while the theory reverts to “action-free predictors” $T_{\phi,z}$ in Section 4. This simplification is understandable, but it is not analyzed how the action-conditioning affects the correspondence to successor measures $M^{\pi_z}$ in Eq. (10). The mapping $P^{\pi_z}$ versus $P(s'|s,a)$ is explained briefly, but the impact on the learnable factorization $M^{\pi_z}\approx \phi T_z \psi^\top$ is not quantified.
   - Theorem 2’s non-collapse guarantee assumes an idealized, continuous-time limit where predictors are always at their optimum (Eq. 17), which is far from how TD-JEPA is trained in practice. Yet, the discussion implies that “if predictors are trained at a faster rate” then covariance is preserved. There is no empirical or theoretical examination of how sensitive TD-JEPA is to this idealized separation-of-timescales assumption; in practice $\phi,\psi,T_\phi,T_\psi$ are all updated with similar learning rates.

   This theory–practice gap does not invalidate the method, but the formal guarantees are less directly applicable to the deployed algorithm than the main text suggests.

2. **Assumption of symmetric dynamics is strong and rarely satisfied.**  
   The core theorems (Theorem 1 and Theorem 3) rely on the assumption that, for all $z$, the Markov kernel $P^{\pi_z}$ is symmetric (Assumption A3). This is quite restrictive and almost never true in realistic control tasks (e.g., continuous control with contact dynamics, navigation in a maze with directionality). While App. C sketches a generalized density-based loss and gradient matching under “relaxed assumptions,” the main takeaways in the paper are built on the symmetric-kernel case.  
   It would help to at least provide a precise theorem in the non-symmetric case (with oblique instead of orthogonal projections), even if the constants are worse, or an explicit statement that, in general MDPs, TD-JEPA optimizes *some* well-defined surrogate for the successor measure but not necessarily $\mathcal{L}_{\mathrm{SM}}$.

3. **Limited ablation on the necessity of bidirectional TD-JEPA vs one-sided objective used in practice.**  
   The theory in App. C.3 derives a “forward-backward” variant with TD losses $\mathcal{L}_{\phi}$ and $\mathcal{L}_{\psi}$ (Eqs. 26–27) that more closely match the density-based forward/backward Bellman residuals, requiring backward-in-time sampling. However, the practical TD-JEPA algorithm in Algorithm 1 uses only a *forward* TD loss for each direction (Eqs. 9 and its symmetric counterpart).  
   The paper notes this variant cannot easily be optimized off-policy, but there is no empirical probing of whether the simpler forward-only TD-JEPA materially differs in behavior. For example, an ablation comparing:
   - the current TD-JEPA loss,
   - a pure forward TD loss approximating $\mathcal{L}_{\text{fw}}$ (Eq. 11) in the original state space, and  
   - a bi-directional version approximating both $\mathcal{L}_{\text{fw}}$ and $\mathcal{L}_{\text{low}}$  
   would sharpen the empirical justification for the specific loss structure chosen.

4. **Optimization and stability concerns are only partially addressed.**  
   The algorithm involves a *highly coupled* system: two encoders, two TD predictors (each taking $\phi(s)$ or $\psi(s)$, action, and latent $z$), a policy $\pi_z$, and target networks for all four representation modules. Some choices are made heuristically (orthonormal covariance regularizers, EMA coefficients, sampling of $z$, FlowQ-style offline corrections for OGBench).  
   While App. E gives implementation details, there is limited discussion of:
   - Sensitivity to $\lambda$ (the regularization coefficient) and to the exact form of the orthonormal regularizer. Table 6 shows very different optimal $\lambda$ ranges across algorithms and suites, but the paper does not analyze why TD-JEPA prefers certain ranges (e.g., $[0.1,1]$ in OGBench nav vs $[1,10]$ in OGBench manipulation) nor whether training is brittle outside those ranges.  
   - The effect of TD bootstrapping hyperparameters, such as $\gamma$ and EMA rates, on stability and representation quality.  
   Given the complexity of the training dynamics, some additional robustness experiments (e.g., varying $\gamma$, or removing the actor loss to see if representation quality survives) would strengthen confidence that the method is not overly fragile.

5. **Experimental emphasis is on averages; per-task behavior and failure modes are underexplored.**  
   Although Table 1 is dense and includes per-domain means and standard errors, the main narrative focuses heavily on suite averages (“DMCRGB (avg), DMC (avg), OGBenchRGB (avg), OGBench (avg)”), which can obscure important structure:
   - In several OGBench tasks, TD-JEPA is not clearly best and sometimes worse than specialized baselines. For example, in OGBenchRGB: cube-single, TD-JEPA ($67.8 \pm 3.67$) is below BYOL-$\gamma^*$ ($76.4 \pm 3.24$); in antmaze-me, TD-JEPA ($0.2 \pm 0.2$) underperforms BYOL-$\gamma^*$ ($3.2 \pm 1.98$) and even the plain Laplacian baseline occasionally.  
   - In proprioceptive OGBench, TD-JEPA’s advantage is not uniform: e.g., antmaze-mn: TD-JEPA $50.4 \pm 3.72$ vs FB $73.0 \pm 2.72$ and HILP $83.6 \pm 2.63$.  
   These nuances are visible in Table 1 and in the probability-of-improvement plots in **Figure 2**, but the main text does not dig into *why* TD-JEPA struggles in some expert-like, narrow-support settings and whether those failures are systematic (e.g., sensitivity to offline coverage, to task encoding dimensionality, or to FlowQ regularization). A more explicit analysis of such failure modes would be valuable.

6. **Some design choices are justified empirically but not conceptually.**  
   Several key design decisions seem driven by empirical tuning rather than principled arguments:
   - The decision to have separate conv encoders for state vs task representations (App. E.2) is later relaxed with the statement that sharing convs did not significantly change performance, but no numbers are given.  
   - The dimensionality $d_\psi = 50$ is fixed across all suites, though the tasks in DMC vs OGBench differ greatly in complexity. There is no ablation on how $d_\psi$ affects zero-shot coverage of reward spaces or successor-factorization quality.  
   - The parameterization of the actor is quite specific (SVG-style, twin networks for TD targets, FlowQ-like offline correction in OGBench), and it is unclear whether simpler policy updates (e.g., plain greedy with respect to $T_\phi$) would work as well.
   These may be reasonable pragmatic choices, but the paper’s framing as a principled successor-measure factorization method would benefit from more discussion of their conceptual roles.

7. **Theoretical treatment of multi-policy coverage is limited.**  
   A central motivation is to learn representations that capture the successor measures of *multiple* policies $\{\pi_z\}_{z\in\mathcal{Z}}$ from *off-policy* data. However:
   - The theory does not analyze how the choice of the latent set $\mathcal{Z}$ (e.g., how $\pi_z$ are initialized and updated) impacts the factorization $M^{\pi_z}\approx \phi T_z \psi^\top$. There is no formal notion of diversity or coverage of the policy family, nor conditions guaranteeing that the span of $\psi$ is rich enough to parameterize “all rewards of interest.”  
   - In practice $\mathcal{Z}$ is sampled from a mixture of task embeddings $\psi(s)$ and random hypersphere vectors (App. E.4), but this choice is not justified beyond prior work (Touati et al.). It is unclear if the resulting set of $\pi_z$ meaningfully covers the behavior space induced by the offline dataset, particularly in low-coverage expert settings.

8. **Mathematical clarity: some notational issues and under-specified aspects.**  
   There are several places where the notation is imprecise or confusing:
   - On Page 3, there is a clear LaTeX/typo: “a policy-dependent predictor $T_{\phi}:=\mathcal{D}\in\mathcal{S}\times\mathcal{D}$ $\mathbb{R}^{d_{\phi}}\times\mathcal{A}\times\mathcal{Z}\to\mathbb{R}^{d_{\phi}}$” which seems garbled. It should likely be $T_\phi:\mathbb{R}^{d_\phi}\times\mathcal{A}\times\mathcal{Z}\to\mathbb{R}^{d_\phi}$.  
   - Equation (7) for $\mathcal{L}_{\text{TD-JEPA}}$ uses $z\sim\mathcal{Z}$ and $a'\sim\pi_z(\cdot|s')$, but the role of the behavior policy underlying $\mathcal{D}$ is not made explicit; off-policy corrections (if any) are not discussed. In practice, the algorithm seems to simply plug in $a'$ from the current $\pi_z$ regardless of behaviour, which is an implicit form of off-policy TD.  
   - In Theorem 4, the statement $\omega_r := (\psi^\top\psi)^{-1}\psi^\top r$ assumes $\psi^\top\psi$ invertible, which holds under orthonormality, but the theorem then maximizes over all $r\in\mathbb{R}^{\geq}:\|r\|_2\le 1$ without relating this to $\mathcal{R}_\psi$. It might be cleaner to bound only over $r$ in the span of $\psi$.

   These are not fatal, but they make some steps harder to parse and weaken the narrative of a fully rigorous theoretical framework.

9. **Minor but relevant: missing direct connection to recent offline RL from videos.**  
   Given the stated motivation in Appendix A around large, multi-task, reward-free datasets (internet videos, teleoperation, etc.), it would be natural to discuss more directly recent work on robotic offline RL from internet videos via value-function pretraining (e.g., Bhateja et al., 2023). That line uses offline, mostly unlabeled video with some action annotations to pretrain value-based RL agents. A comparison could clarify to what extent TD-JEPA would be competitive or complementary in such settings, especially since both aim at extracting reusable control representations from heterogeneous data.

Overall, the paper is technically rich and empirically strong, but the main limitations lie in the tightness of the theoretical connection to the practical algorithm, the strength of assumptions in theorems, and somewhat underdeveloped analysis of failure modes and design trade-offs. None of these are fatal, but they do pull the work down from “spotlight-level” rigor to “strong poster” territory.


## Potentially Missing Related Work

1. **C. Bhateja et al., “Robotic Offline RL from Internet Videos via Value-Function Pre-Training”, 2024.**  
   - **Why it is directly related:** This work tackles learning generalizable control/value functions from large-scale offline internet videos, a setting very close to the paper’s stated long-term goal of using reward-free, multi-task datasets for robotic control. It provides another approach to extracting reusable control representations from unlabeled or weakly labeled data, via value-function learning rather than latent-predictive successor features.  
   - **Where to cite/discuss:**  
     - In Section 5 (“Related Work”), under “Other representation learning methods for RL” or “Applications of Unsupervised RL”, comparing how TD-JEPA’s reward-free successor-factorization differs from value-function pretraining and whether it would handle similar data regimes (heterogeneous internet videos with sparse or noisy action annotations).  
     - Possibly in Appendix A where the authors discuss real-world robotics applications and in-the-wild data.

If the authors already consider extensions to large-scale video datasets, directly positioning against this line of work will help clarify TD-JEPA’s role relative to value-function pretraining approaches.


## Questions

1. **Practical sensitivity to predictor–encoder training timescales.**  
   The non-collapse guarantee in Theorem 2 critically assumes that predictors are trained to optimality at each step (Eq. 17). In the actual implementation, how many gradient steps are used on $T_\phi$ and $T_\psi$ per encoder update, and how sensitive is performance to this ratio? Could the authors provide an ablation where the predictor is updated, say, $\{1, 5, 10\}$ times per encoder update to see if covariance collapse or degradation in zero-shot performance appears?

2. **Impact of the symmetric-kernel assumption (A3).**  
   While App. C hints at how one can relax A3 by using adjoints $\Xi_z^*$ and density-based losses, the main theorems in Section 4 still depend on symmetry. Can the authors provide a *concrete* theorem statement (even in an appendix) for the non-symmetric case that applies to TD-JEPA’s practical setting, or at least quantify how the bias in approximating $M^{\pi_z}$ scales with the asymmetry of $P^{\pi_z}$?

3. **Failure modes on specific OGBench tasks.**  
   In Table 1, TD-JEPA underperforms FB/HILP in OGBench (proprio) antmaze-mn and on some manipulation tasks (e.g., cube-single in RGB). Could the authors investigate and explain whether this is due to:  
   (a) the FlowQ-style offline regularization,  
   (b) insufficient coverage in $\mathcal{Z}$ (too few or too similar policies),  
   (c) overly tight orthonormal regularization on $\psi$, or  
   (d) something else (e.g., actor training instability)?  
   Some diagnostic plots (e.g. reward-projection error in $\mathcal{R}_\psi$, or distribution of learned $z_r$) would be helpful.

4. **Necessity of separate state and task encoders.**  
   Figure 3 (right) shows a gain from using separate $\phi$ and $\psi$, but the symmetric variant often performs reasonably well. Could the authors comment on when they expect the symmetric variant to suffice in practice (e.g., low-dimensional proprioceptive control vs complex vision-based tasks), and whether there are concrete cases where forcing a shared encoder demonstrably hurts successor-factorization quality (e.g. by measuring $\mathcal{L}_{\mathrm{SM}}$ or a proxy)?

5. **Effect of task-encoder dimensionality and orthonormality.**  
   $d_\psi$ is fixed to 50 everywhere, with orthonormal regularization to avoid collapse. Have the authors experimented with significantly larger or smaller $d_\psi$ (e.g., 16, 128) and observed saturation or breakdown in zero-shot performance? Does Theorem 4’s bound suggest any practical guidance on $d_\psi$ relative to the number of downstream tasks or inherent reward complexity?

6. **Off-policy bias in TD-JEPA updates.**  
   Equation (7)/(9) uses $(s,a,s')\sim\mathcal{D}$ and $a' \sim \pi_z(\cdot|s')$ with no explicit importance weighting. The online/offline TD literature (e.g., Precup et al., 2001) is quite clear that off-policy TD with function approximation can be biased or even divergent. Do the authors observe any divergence or instability in practice? Would simple importance sampling or Retrace-like corrections change the behavior, at least in ablation?


## Flag For Ethics Review

- No ethics review needed.  


## Details Of Ethics Concerns

N/A. The work uses standard continuous-control and goal-conditioned RL benchmarks without sensitive human data or explicit deployment claims; there are no obvious fairness, privacy, or safety concerns beyond those already common in RL research.


## Soundness Rating

3: good.  
The method is well motivated, the empirical evaluation is thorough and carefully executed, and the theoretical analysis is mathematically nontrivial, though somewhat idealized relative to the practical algorithm and based on strong symmetry and orthonormality assumptions. No fatal flaws are evident, but there is a noticeable theory–practice gap.


## Presentation Rating

3: good.  
The paper is generally well written and structured, with clear equations (despite a few notational glitches), helpful figures (especially Figures 1–4, 8–10), and comprehensive appendices. However, some notation is sloppy in places, and the interplay between the main text theorems and the more general appendix treatment could be made clearer.

## Contribution Rating

4: excellent.  
The combination of TD-based latent prediction with successor-measure factorization across multiple policies, the extension of latent-predictive theory to multi-policy TD settings, and the strong empirical results in challenging pixel-based zero-shot RL benchmarks together constitute a substantial and timely contribution to representation learning for control.

## Overall Rating

8: Accept, good paper (poster).  
The paper advances both the theory and practice of self-predictive representation learning for zero-shot RL, providing a conceptually clean TD-based latent-predictive objective, nontrivial analysis linking it to successor-measure factorization and value approximation, and solid empirical gains over strong baselines, especially from pixels. The main weaknesses are the reliance on idealized assumptions in the theory and an underdeveloped analysis of certain empirical failure modes, but these do not undermine the core ideas or results. I recommend acceptance as a strong poster.

## Reviewer Confidence

4: confident.  
I am familiar with latent-predictive/self-predictive RL, successor features, and zero-shot RL, and I carefully checked the key equations and theorems. Some parts of the very general gradient-matching framework in Appendix C are intricate enough that subtle issues might remain, but they do not affect my overall assessment.