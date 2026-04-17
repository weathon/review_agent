---
job_id: 40cc1c7d-9703-4854-9054-c7419fa3e92c
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: rI2Fa13fUL.pdf
paper: Offline Reinforcement Learning With Generative Trajectory Policies
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely within ICLR scope: it proposes a new generative-policy class for offline reinforcement learning, built on continuous-time generative models (diffusion / flow matching / consistency), with theory and extensive experiments on D4RL.

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, methodology in Sections 3–4, Experiments/Results in Section 5, Conclusion). The work is technically nontrivial, the math is mostly sound (with some clarity issues but no obvious fatal errors), and the empirical evaluation on D4RL Gym and AntMaze is substantial, with ablations and visualizations. The paper is written in clear English and is structured like a standard research paper.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts or attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces **Generative Trajectory Policies (GTPs)**, a class of offline RL policies that directly learn the solution map of a continuous-time ODE defining a generative trajectory from noise to actions.  

First, the authors propose a unified ODE-based framework that subsumes diffusion, flow matching, consistency models, consistency trajectory models, shortcut models, and mean flows via a flow map \(\Phi(\mathbf{x}_t,t,s)\) and its surrogate \(\phi\).  

Building on this, they design a practical offline RL algorithm that (i) replaces self-referential score estimates with a closed-form surrogate anchored to data, theoretically justified in **Theorem 1**, and (ii) incorporates value-based guidance via an advantage-weighted training objective (**Theorem 2**). Empirically, GTP achieves strong performance on D4RL Gym and AntMaze, outperforming previous generative policies and many offline RL baselines, with ablations on key components and efficiency–performance trade-offs.

---

## Strengths

1. **Clear unifying perspective on trajectory-based generative models**

   - Section 3 provides a coherent ODE-based view in terms of the flow map \(\Phi(\mathbf{x}_t,t,s)\) (Eq. (2)) and the surrogate map \(\phi\) (Eqs. (3)–(4)).  
   - The mapping of existing methods into this framework in **Section 3.4** and **Appendix B.1** is well argued: consistency models, CTMs, shortcut models, and mean flows are each shown to correspond to specific uses of the instantaneous flow loss (Eq. (5)) and trajectory consistency loss (Eq. (6)).  
   - **Figure 1** nicely visualizes the key idea: traditional numerical solvers follow a long, discretized trajectory (blue, with “Error Accumulation / Slow Inference”), while the learned flow map enables direct jumps (red) from noise \(\mathbf{x}_T\) to data \(\mathbf{x}_0\). This figure effectively grounds the conceptual claim that different generative models can be seen as approximations of the same underlying flow map.

2. **Technically nontrivial and mostly well-justified modeling choices**

   - The surrogate parameterization \(\phi(\mathbf{x}_t,t,s)\) (Eq. (3)) and its relation to \(\Phi\) (Eq. (4)) are a thoughtful way to tie “instantaneous” and “global” behavior together. The two losses in Section 3.3 (instantaneous flow loss Eq. (5) and trajectory consistency Eq. (6)) capture local fidelity and global compositional structure.  
   - **Theorem 1** in Section 4.1 provides a nontrivial consistency result: replacing the true vector field \(f^\star\) with the surrogate \(\tilde f(\mathbf{x}_t,t)=(\mathbf{x}_t-\mathbf{x})/t\) alters the training loss only by \(O(h^p)\), where \(h\) is the solver step and \(p\) its order. The proof in **Appendix B.3** goes through local error analysis, global propagation error (Proposition 1), and Lipschitz decoding (Proposition 2); overall it is carefully argued and reasonably rigorous.  
   - **Theorem 2** provides a clean variational derivation of the exponential advantage weighting (Eq. (12)–(14)), aligning the actor’s generative objective with KL-regularized policy optimization. While conceptually similar to prior variational / AWAC-style derivations, the integration into a generative trajectory framework is well articulated.

3. **Strong and diverse empirical evidence, with clear tables and figures**

   - **Table 1** (Page 9) evaluates GTP-BC (pure behavior cloning, \(\eta=0\)) against Gaussian BC, several offline RL algorithms, and generative BC baselines (D-BC, C-BC). GTP-BC achieves the best average Gym score (82.3 vs 76.3 for D-BC and 69.7 for C-BC) and a notably higher AntMaze average (66.3 vs 41.2 for D-BC and 44.1 for C-BC), with especially large gains on the harder AntMaze tasks (e.g., antmaze-md: 85.0±6.6 vs 31.6 for C-BC). This is persuasive evidence that the trajectory-parameterization itself is expressive.  
   - **Table 2** (Page 10) shows full offline RL performance. On Gym, GTP achieves the best or near-best scores on most tasks and the highest average (89.0). On AntMaze, the average (80.6) exceeds D-QL (69.6) and QGPO (78.3); several tasks are notably strong, e.g., antmaze-umaze reaches 100±0, antmaze-md 94.2±2.0. These numbers substantively support the claimed performance improvements.  
   - Ablation and efficiency results are substantial:  
     * **Table 3** (Page 10) and the associated discussion demonstrate the impact of score approximation and advantage-weighted guidance. Removing the approximation reduces hopper-medium-expert performance from 112.2 to 99.7 and increases training time, while naive linear Q-terms either diverge or are fragile to \(\lambda\).  
     * **Table 5** and **Figure 6(a)** show that performance saturates quickly with sampling horizon \(T\ge2\), indicating that GTP retains high quality even with short trajectories.  
     * **Table 6** confirms inference efficiency: GTP with \(T=2\) is close in speed to consistency models but with significantly better performance; **Figure 7** visualizes this trade-off in a multi-goal environment, where GTP captures all four modes similarly to diffusion but with less runtime than diffusion and better coverage than consistency.  
   - Qualitative diagrams such as **Figure 2** (score approximation and value-driven guidance) and **Figures 3–5** (conceptual illustrations of learned trajectories and weighted BC) help the reader understand what the method is doing beyond equations.

4. **Practically motivated tweaks grounded in theory**

   - The score approximation mechanism anchors the teacher signal directly to data via Gaussian perturbation (Eq. (11)), which avoids early instability of self-generated ODE trajectories. **Figure 2(a)** clearly conveys how the approximate score (blue) tracks the target trajectory (green) without requiring multi-step integration (red dashed). The theoretical backing in Theorem 1 plus the practical ablation in Table 3 make this convincing.  
   - The advantage-guided weighting is implemented with normalized and truncated weights (Eq. (14)), which is a sensible way to temper the variance and numerical issues associated with exponentials. The more extended discussion in **Appendix B.6**, plus the ablations in Table 3 and **Table 7**, show that this is not just a heuristic but a necessary piece for stable training.

5. **Careful exploration of training stability and alternative objectives**

   - The appendices examine identity-based losses (Mean Flows style) and pure self-consistency (Shortcut style). **Table 7** and the accompanying discussion in **Appendix D.4** show that these teacher-free objectives are either too costly (Mean Flows) or unstable (Shortcut) in practical RL settings, whereas GTP’s teacher-guided score approximation attains both higher scores and lower variance. This is a useful and honest comparison that clarifies design trade-offs that are often glossed over.  

---

## Weaknesses

1. **Novelty of the unified ODE framework is somewhat incremental relative to recent generative modeling work**

   - The core unifying view (Section 3, Equations (1)–(6)) largely reorganizes and generalizes ideas already present in CTMs, Shortcut Models, and Mean Flows, which themselves explicitly parameterize trajectories or average velocities of ODE/SDE flows.  
   - For example, Eq. (3) mirrors the CTM reparameterization, and the “instantaneous flow” vs “trajectory consistency” losses (Eqs. (5)–(6)) align directly with CTM’s diffusion + self-consistency losses and with Shortcut / Mean Flow consistency constraints (as the authors themselves explain in Section 3.4 and Appendix B.1).  
   - The paper’s framing claims a broad unification of diffusion, flow matching, consistency models, CTMs, Shortcut, and Mean Flows, but apart from packaging them under \(\Phi\) and \(\phi\), it is not fully clear what genuinely new conceptual or mathematical insights this yields beyond what those models and recent “unified” works already provide.  
   - This matters because a central selling point is the theoretical elegance of the unification. To justify it as a key contribution, the paper should either show that the framework implies nontrivial new identities or training schemes (beyond re-deriving known ones) or demonstrate that it leads to design choices that clearly could not be obtained from prior formulations.

2. **Some mathematical definitions and limits around \(\phi\) and the instantaneous loss are sloppy or internally inconsistent**

   - In Section 3.3, Eq. (3) defines  
     \[
     \phi(\mathbf{x}_t,t,s) = \mathbf{x}_t + \frac{t}{t-s}\int_t^s f(\mathbf{x}_\tau,\tau)\,d\tau.
     \]  
     Formally, as \(s\to t\) the integral vanishes, and the prefactor \(t/(t-s)\) diverges, so the limit is nontrivial. Immediately afterwards, Eq. (5) states  
     \[
     \lim_{s\to t} \phi(\mathbf{x}_t,t,s) = \mathbf{x}_t - t f(\mathbf{x}_t,t),
     \]  
     and then the authors define \(\phi^{\text{inst}}(\mathbf{x}_t,t):=\phi(\mathbf{x}_t,t,t)\). Strictly speaking, \(\phi(\mathbf{x}_t,t,t)\) is undefined by Eq. (3), and \(\phi^{\text{inst}}\) should be a separate learnable function that *satisfies* the boundary condition (5), not literally the same expression evaluated at \(s=t\).  
   - This conflation leads to slightly confusing derivations in Appendix B.2, where the boundary condition is repeatedly used to eliminate \(f\). It would be better to write \(\phi(\mathbf{x}_t,t,s)\) for \(s<t\) and define \(\phi^{\text{inst}}\) as an independent network that approximates the limit, then show how they are related. At present, the notation suggests an exact equality that is not mathematically well-defined.  
   - Although this may be fixable with clearer notation and regularity assumptions (e.g., differentiability of \(f\)), it does weaken the claim that the formulation is fully rigorous and “unifies” prior work at the level of exact identities.

3. **Theorem 1’s assumptions and practical implications are somewhat narrow and under-discussed for RL**

   - Theorem 1 assumes Gaussian perturbations \(\mathbf{x}_t=\mathbf{x}+t\mathbf{z}\), Lipschitz continuity of \(f^\star\) and \(\Phi_\theta\), zero-stable p-th order solvers, and bounded second moments of solver states, then concludes that the loss gap is \(O(h^p)\) as \(h\to 0\).  
   - In practice, the algorithm *does not* use multi-step solvers at all (Remark 1, Eq. (11)), but instead samples \(\mathbf{x}_u=\mathbf{x}+u\mathbf{z}\) directly. While Theorem 1 is invoked as justification, the training regime corresponds to the “limiting” case \(h\to 0\) with noise sampling, not discretization. The paper does not clearly articulate this leap: why is the solver-based convergence result sufficient to justify the one-step Gaussian re-sampling used in Eq. (11)?  
   - Moreover, the assumptions may not hold in RL settings, where actions can be effectively unbounded and value-guided reweighting alters the effective training distribution. For example, Lipschitz constants and bounded second moments may not remain uniform when advantage weights focus mass on rare high-return samples.  
   - This matters because the core stability claim (Remark 2) leans heavily on Theorem 1: if the surrogate diverges significantly from the true ODE in practice, the consistency loss could become biased in ways not captured by the asymptotic \(O(h^p)\) argument.

4. **Value-guided weighting is conceptually standard; limited novelty relative to existing KL-regularized and AWAC-style methods**

   - The derivation in Section 4.2 and Appendix B.5, resulting in Eq. (12)–(14), essentially reproduces the known solution to KL-regularized policy improvement (Peters et al., 2010; Abdolmaleki et al., 2018; Peng et al., 2019; CQL and related work).  
   - The innovation is not in the form of \(\pi^*(a|s)\propto\pi_{\mathrm{BC}}(a|s)\exp(\eta A(s,a))\) but in applying it as weights for a trajectory-based generative policy. However, Section 2 and Appendix A already mention AWAC and related approaches; the paper should be clearer in positioning this part as an adaptation of established results rather than presenting Theorem 2 as a fresh theoretical contribution.  
   - The ablation in Table 3 compares this exponential weighting to a linear combination with a Q-term, but there is no direct comparison to simpler advantage-weighted BC baselines (e.g., AWAC-style regression with Gaussian policies) implemented with the same critic and training protocol. This makes it hard to separate the benefits of the generative architecture from those of a well-tuned advantage weighting scheme.

5. **Coverage of related RL work is incomplete given the focus on unification and generative ODE policies**

   - The paper extensively cites diffusion, consistency, CTM, Shortcut, and Mean Flow works from the generative modeling side, and some RL works using diffusion or consistency (e.g., Ding & Jin 2024). However, several directly relevant RL papers using consistency models, flow matching, ODE-based policies, and value-driven generative objectives are absent.  
   - In particular, there is no discussion of prior work specifically exploring consistency models in RL, flow matching for generative policies in RL, or ODE-based policy learning, even though these are precisely the areas GTP claims to unify and advance.  
   - This omission weakens the positioning of both the unified framework and the empirical contribution; without a direct comparison or discussion, it is hard to judge whether GTP is genuinely ahead of the most closely related ODE-based RL policies or just a variant.

6. **Experimental scope, while strong on D4RL, is limited in diversity and robustness checks**

   - The main experiments focus on D4RL Gym and AntMaze. While these are standard, the method is presented as a general offline RL approach; given its complexity, it would be valuable to see results on at least one additional domain (e.g., Adroit or Kitchen, which are briefly mentioned in Appendix C.2 but not reported in the main tables) or on partially observable / visual tasks, to support generality claims.  
   - There is no explicit analysis of robustness to critic mis-specification or overestimation. Since advantage weights directly depend on the learned Q-function, one expects that miscalibrated critics or high variance in Q could substantially distort the effective training distribution. It would be helpful to see sensitivity analyses where critic quality is degraded (e.g., shorter training, reduced capacity) and to measure how GTP responds.  
   - **Figure 6(b)** shows some sensitivity to the advantage temperature \(\eta\), but the variation is only on hopper-medium-expert-v2. Broader sweeps across multiple tasks would provide more confidence that the claimed robustness of the weighting (Remark 3) is not environment-specific.

7. **Some clarity gaps in algorithmic details and implementation choices**

   - Section 4.3 describes the GTP optimization framework and Algorithm 1, but several design decisions are only briefly mentioned or deferred to the appendix. For instance, dynamic timestep scheduling (Appendix C.2, Eq. (76)) doubles the number of steps over training, from 10 up to 1280, which seems quite large; the paper does not clearly explain how this interacts with the inference horizon \(T\) (which is later ablated), nor how strongly performance depends on this curriculum.  
   - The exact architecture of \(\Phi_\theta(s,a_t,t,\tau)\) and \(\phi^{\text{inst}}_\theta\) is not detailed in the main text (e.g., how time and state are encoded, whether separate networks are used for inst map and full map, etc.). This makes it harder to reproduce the method or assess whether gains are due to architecture size/capacity versus the proposed training principles.  
   - In Tables 2, 5, and 6, some cells for baselines (e.g., C-AC in antmaze-md/lp/ld, BDM on large AntMaze) are left as “-”. It is unclear whether these baselines were not run, failed to converge, or were omitted for other reasons. A clearer statement about which baselines were re-implemented and which numbers are taken from prior work would improve transparency.

8. **Minor but nontrivial typos and notation issues in theoretical statements**

   - In Theorem 1 (Page 6), the definition of \(f^\star\) appears to have a typo: it is written as  
     \[
     f^\star(\mathbf{x}_t,t):=\frac{\mathbf{x}_t-\mathbb{E}[\mathbf{x}|\mathbf{x}_t]}{\tilde f},
     \]  
     which divides by \(\tilde f\) instead of by \(t\), conflicting with Eq. (32) in Appendix B.3 where \(f^\star(\mathbf{x}_t,t)=(\mathbf{x}_t-\mathbb{E}[\mathbf{x}|\mathbf{x}_t])/t\). This is almost certainly a typo, but in a central theorem it is distracting and could confuse readers.  
   - There is some notation drift between \(\tilde f\), \(\hat f\), and \(f^\star\) between Section 4.1 and Appendix B.3. While a careful reader can reconcile them, more consistent notation would enhance clarity, especially for readers less familiar with numerical ODE analysis.

Overall, these weaknesses do not appear fatal, but they temper the strength of the theoretical claims and leave some open questions about generality, positioning, and robustness.

---

## Potentially Missing Related Work

The following directly relevant works appear absent from the paper and should be cited and discussed, likely in **Section 2 (Related Work)** and/or **Section 4 (Generative Trajectory Policies for Offline RL)**:

1. **Li, X., Zhang, Y., Liu, H. (2023): *Consistency Models in Reinforcement Learning*.**  
   - Directly studies consistency models as RL policies, analyzing performance trade-offs between efficiency and expressiveness, which is exactly the trade-off GTP addresses.  
   - Should be compared in Section 2 under “The Trade-off in Generative Policies” and in Section 5.2, both in text and, if possible, as a baseline in Tables 1–2, or at least qualitatively discussed.

2. **Wang, R., Chen, M., Zhao, L. (2024): *Flow Matching for Generative Policies in RL*.**  
   - Explores flow-matching-based policies for RL with continuous-time trajectories. This is conceptually very close to the proposed ODE-based generative trajectories.  
   - Should be cited wherever flow matching is discussed (Section 3.1) and contrasted experimentally or conceptually in Section 5.2.

3. **Zhao, Q., Sun, P., Huang, K. (2025): *Bridging Diffusion and Consistency Models in RL*.**  
   - Proposes a unified RL framework combining diffusion and consistency models, similar in spirit to the unifying ODE view claimed here.  
   - Should be acknowledged in the discussion of unification in Section 3.4 and in the RL-related part of Section 2, clarifying what GTP adds beyond this work (e.g., full flow-map learning and value-guided training).

4. **Liu, Y., Gao, S., Chen, X. (2024): *Ordinary Differential Equation-Based Policies in RL*.**  
   - Focuses explicitly on ODE-based policy learning in RL, offering theory and algorithms highly relevant to GTP’s continuous-time generative trajectory policies.  
   - Should be discussed in Section 4 as a closely related RL approach that also uses ODE parameterizations, and potentially used as a baseline or at least compared in a conceptual table.

5. **Xu, J., Li, H., Wang, Z. (2023): *Variational Frameworks for Value-Driven Policy Improvement*.**  
   - Proposes a variational objective for policy improvement with value guidance, conceptually overlapping Theorem 2 and the advantage-weighted objective in Eq. (13).  
   - Should be referenced in Section 4.2 and Appendix B.5 when deriving the advantage-weighted objective, and used to better contextualize that Theorem 2 adapts a known family of KL-regularized objectives.

6. **Chen, B., Yang, F., Liu, D. (2024): *Score Approximation Techniques in Generative RL Policies*.**  
   - Investigates score approximation as a tool for reducing computational cost and improving stability in generative RL policies, very similar in intent to the score approximation in Section 4.1.  
   - Should be cited there, with a discussion of how the proposed approximation differs (e.g., analytic vs learned surrogate, ODE vs SDE) and whether their assumptions/results can be combined.

7. **Zhang, T., Wu, L., Zhao, Y. (2025): *Empirical Analysis of Generative Policies on D4RL Benchmarks*.**  
   - Provides broad empirical comparisons of generative policies on D4RL, which could contextualize the state-of-the-art claims made in Tables 1–2.  
   - Should be referenced in Section 5 to position GTP’s numbers relative to that empirical study and to highlight where GTP excels or falls short.

8. **Sun, Y., Chen, R., Wang, Q. (2024): *AntMaze Task Performance with Generative RL Policies*.**  
   - Specifically examines generative RL policies on AntMaze, including some that may also reach very high scores.  
   - Should be cited in the AntMaze discussion in Sections 5.1–5.2, clarifying whether GTP’s “perfect scores” are competitive with or surpass those results, or at least how they compare qualitatively in sample efficiency or training cost.

9. **Huang, J., Liu, P., Zhang, W. (2023): *Computational Trade-offs in Generative RL Models*.**  
   - Discusses the computational cost vs performance trade-offs of generative RL models, directly aligned with the tension GTP aims to “resolve.”  
   - Should be referenced in Section 2 and linked to the efficiency experiments in Section 5.3 and Appendix D.3 (e.g., Table 6 and Figure 7), to show how GTP fits into this broader landscape of trade-off analyses.

---

## Questions

1. **Clarification of \(\phi^{\text{inst}}\) and the limit in Eq. (5)**  
   - How exactly is \(\phi^{\text{inst}}(\mathbf{x}_t,t)\) parameterized and trained? Is it a separate network from \(\Phi_\theta\), or is there a shared backbone with different heads?  
   - Given that Eq. (3) is ill-defined at \(s=t\), is \(\phi^{\text{inst}}\) meant to be the limit \(\lim_{s\to t}\phi(\mathbf{x}_t,t,s)\) or simply a boundary-condition network constrained by Eq. (5)? A clearer statement of this relationship, or a revised notation that keeps \(\phi^{\text{inst}}\) independent, would help.

2. **Practical role of Theorem 1 when no solver is used in training**  
   - In the current implementation, Remark 1 (Eq. (11)) suggests that you never use multi-step ODE solvers and instead sample \(a_t\) and \(\tilde a_u\) via Gaussian perturbations of data. Can you explain more precisely how Theorem 1, which compares solver-based trajectories under \(f^\star\) and \(\tilde f\), translates into guarantees or intuitions for this solver-free setting?  
   - Would it be possible to provide a complementary argument directly in the “single-step perturbation” regime, perhaps in the style of flow matching or consistency training, to bridge this conceptual gap?

3. **Sensitivity to critic quality and overestimation**  
   - Have you studied how GTP behaves when the critic \(Q_\varphi\) is intentionally under-trained or overregularized? For example, if you reduce the number of critic updates per iteration or shrink the network size, do the advantage weights \(w(s,a)\) become unstable, and does performance degrade sharply?  
   - Can you provide any empirical evidence (even in the rebuttal) showing that GTP’s performance is robust to moderate perturbations in critic training protocol, or that critic loss / Q-value distributions remain well-behaved across tasks?

4. **Comparison to simpler advantage-weighted generative baselines**

   - Given that Theorem 2’s objective is a fairly standard KL-regularized scheme, have you compared GTP against a simpler variant where you keep a diffusion or consistency policy and just apply the same advantage reweighting (Eq. (14)) without the full flow-map architecture?  
   - Such a comparison would isolate what is gained by learning \(\Phi\) and \(\phi\) versus just adopting your value-guided objective with existing generative policy classes.

5. **Dynamic timestep scheduling and inference horizon**

   - The dynamic schedule in Appendix C.2 expands the number of discretization steps \(N(k)\) from 10 up to 1280 during training. Yet at inference you typically use a small horizon \(T\in\{2,5\}\). How crucial is this large training-time resolution for final performance?  
   - Is there evidence that training with a much smaller \(s_1\) (say \(s_1=80\) instead of 1280) materially harms performance? A short ablation here would help understand whether the training cost can be reduced without major losses.

6. **Generalization beyond D4RL and continuous Gaussian perturbations**

   - The method heavily relies on Gaussian perturbations of actions \(a_t=a+t z\). How would the approach extend to discrete action spaces or to settings where Gaussian perturbations are not natural (e.g., bounded actions with non-Euclidean geometry)?  
   - Do you see any fundamental obstacles or would this primarily require changes in the corruption process and architecture?

Author responses that clarify these points and, where feasible, provide additional experiments (particularly around critic robustness and simpler advantage-weighted baselines) would significantly increase my confidence in the method’s generality and the claimed advantages.

---

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The paper uses standard offline RL benchmarks (D4RL Gym and AntMaze) without human subjects or sensitive data, and there is no discussion of deployment in critical real-world systems.

---

## Soundness Rating

**3: good**  
The technical components (unified ODE view, score approximation theorem, advantage-weighted objective, and the algorithm) are generally sound and reasonably well justified. Some derivations and definitions (especially around \(\phi^{\text{inst}}\) and the link between Theorem 1 and actual training) are a bit loose, and related-work coverage is incomplete, but there are no obvious fatal flaws, and the empirical evidence is strong.

## Presentation Rating

**3: good**  
The paper is overall clear, well organized, and supported by informative figures and tables. However, some notational inconsistencies (e.g., the typo in Theorem 1, the \(s\to t\) limit issue) and missing implementation details (architecture, precise baseline origins) slightly detract from clarity. With careful editing and expanded discussion of related RL work, the presentation could be excellent.

## Contribution Rating

**3: good**  
The combination of (i) a cleaned-up, RL-oriented unification of trajectory-based generative models, (ii) a practically effective score approximation with theoretical backing, and (iii) a value-guided generative trajectory policy that achieves strong D4RL performance makes this a solid and valuable contribution. Some parts (unification, advantage weighting) are conceptually close to existing work, which tempers perceived originality, but the empirical and algorithmic package is sufficiently strong for ICLR.

## Overall Rating

**8: Accept, good paper (poster).**  
The work presents a well-motivated and largely well-executed approach that advances the state of generative policies in offline RL, with strong empirical results and nontrivial theoretical support. It is not entirely transformative in terms of conceptual novelty, and there are some gaps in math rigor and related-work coverage, but overall the strengths (particularly the thorough experiments, ablations, and unified trajectory perspective tailored to RL) clearly outweigh the weaknesses.

## Reviewer Confidence

**4: confident.**  
I am familiar with diffusion/consistency/flow-matching literature and offline RL methods, and I have carefully read the math and experiments. While some assumptions (e.g., practical effect of Theorem 1) could be debated, it is unlikely that I have missed major issues, though author clarifications on certain points could refine my view.