---
job_id: 38a7d805-fc23-4699-bdb0-25ebc989654c
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: O6n4dgTBBL.pdf
paper: Stabilizing Gradient Descent via Second-Order Control-Theoretic Dynamics
main_score_norm: 0.2
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies the stability of gradient descent via continuous-time and control-theoretic analysis, which falls squarely under non-convex optimization and learning theory, clearly within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is in English and has all required sections: Abstract, Introduction, Related Work, Methodology (Sections 3–6), Experiments (Section 7), Results/Discussion (Sections 7–8), and Conclusion. While there are significant technical and conceptual issues (detailed in the review), they do not rise to the level of desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden instructions, prompt injection, or manipulative content targeting automated reviewers is apparent in the provided main paper text.

---

# Expected Review Outcome:

## Summary

The paper analyzes gradient descent (GD) from a continuous-time perspective by viewing gradient flow as a dynamical system, then deriving a second‑order ODE for the parameter trajectory. Using local linearization, the authors relate the Jacobian spectrum of the resulting first‑order state-space system to the eigenvalues of the Hessian and claim that GD dynamics are Lyapunov stable only under strong convexity but unstable in merely convex or concave cases. To address this, they introduce a linear controller in the second‑order dynamics, leading to a modified gradient update that is theoretically argued to yield locally asymptotically stable dynamics and empirically tested on low‑dimensional synthetic loss functions.

## Strengths

1. **Clear dynamical systems framing and linearization analysis.**  
   Sections 3 and 4 give a reasonably clear derivation of the second‑order ODE from gradient flow, \(\ddot{\theta} = -H(\theta)\dot{\theta}\) (Equation (2)), followed by the state‑space formulation (Equation (3)) with state \(\mathbf{z} = [\theta; x]\). The Jacobian computation and block structure of \(J(\mathbf{z})\) are explicit, and the eigenvalue expression \(\det(\lambda^2 I + \lambda H)\) at equilibrium is nicely tied to Theorem 1. This provides a tractable and transparent continuous‑time lens on GD dynamics.

2. **Connection between Hessian curvature and Jacobian spectrum.**  
   Theorem 2 and the accompanying discussion in Sections 4.2.1–4.2.3 provide a systematic classification (strongly convex, convex but not strongly convex, concave) and relate these to the eigenstructure of the lifted system. While not surprising to experts in dynamical systems, it is pedagogically useful to see how strongly convexity yields Lyapunov stability and how degeneracies in the Hessian lead to problematic Jordan blocks, particularly in the convex but not strongly convex case.

3. **Use of a quadratic eigenvalue problem and control‑theoretic tools.**  
   In Section 5, the controlled dynamics introduce a Jacobian whose eigenvalues solve a quadratic matrix polynomial \(Q(\lambda) = \lambda^2 I + \lambda (H + K_2) + K_1\). Leveraging Lemma 4 (Tisseur & Meerbergen, 2001) to ensure all eigenvalues have negative real parts when \(I \succ 0\), \(H+K_2 \succ 0\), and \(K_1 \succ 0\) is a clean control‑theoretic argument that, in the continuous‑time model, ensures local asymptotic stability. The translation into Table 1 (Page 5), contrasting curvature assumptions and stability guarantees between vanilla GD and the controlled system, gives a compact conceptual summary.

4. **Figures effectively illustrate qualitative behavior of trajectories.**  
   - **Figure 1** (Page 2) contrasts GD and controlled GD trajectories on \(L(\theta) = 2\theta_1^2 + 0.5\theta_2^2\). The zig‑zagging and apparent edge‑of‑stability behavior under standard GD versus the more regular path under controlled GD visually support the claimed stabilization effect.  
   - **Figure 2** (Page 8) extends this to strongly convex ellipse, convex sphere, and quartic losses, showing 3D surface plots (subfigures (a)–(c)) and training curves (subfigures (d)–(f)) where GD either oscillates or diverges while CGD converges for the chosen settings. These visualizations, especially the 3D trajectory plots, are helpful for intuition.

5. **Some empirical support for robustness to hyperparameters and larger learning rates.**  
   In Section 7.1, the authors vary \(k_1=k_2\) among \(\{0.05,0.1,0.2\}\) in CGD and show in Figure 2(d)–(f) that all settings converge, while standard GD is unstable or slow. Section 7.2 and **Figure 3** analyze the convex sphere loss near the classical stability threshold \(\eta = 2/\text{sharpness} = 1\). For \(\eta \in \{0.99,1.0,1.01\}\), the loss curves show standard GD slowing, oscillating, or diverging while CGD remains stable. These toy experiments do provide some concrete evidence that the proposed modification can tolerate somewhat larger step sizes than standard GD in the tested cases.

## Weaknesses

1. **Conceptual mismatch between continuous‑time analysis and discrete GD claims.**  
   The entire theoretical development is based on gradient flow ODEs; the key system is \(\dot{\theta} = -\nabla L(\theta)\) (Equation (1)) and its derived second‑order ODE \(\ddot{\theta} = -H(\theta)\dot{\theta}\) (Equation (2)). However, the Introduction and claims throughout the paper are phrased for *discrete* gradient descent with step size \(\eta\), including references to the sharpness bound \(\eta < 2/L\) and the "edge of stability". Theorem 2 and the curvature‑based classification are explicitly about the continuous‑time, linearized system (Section 4.2), not about the discrete‑time map \(\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)\). The paper repeatedly states or implies conclusions such as "even if the learning rate \(\eta\) is properly bounded by \(\eta < 2/\lambda\), gradient descent can still be unstable" (Page 2, bullet 1), but this is not actually proved: the theorems never introduce \(\eta\) or analyze the spectrum of the discrete update matrix. Instead they show that the *continuous‑time* lifted system in \((\theta, \dot{\theta})\)-space is not asymptotically stable in the Lyapunov sense when the Hessian is only positive semidefinite. This is not the same as divergence of discrete GD iterates or training loss, and the paper does not clearly bridge this gap. The limitation is acknowledged only briefly in Section 8, but the main narrative overstates implications for discrete GD.

2. **Derivation of the modified gradient (Equation (5)) is mathematically incorrect.**  
   The central step connecting the controlled second‑order ODE (Equation (4)) to the practical update rule is flawed. Starting from
   \[
   \ddot{\theta}' = \ddot{\theta} + u = -H(\theta)\dot{\theta} - K_1\theta - K_2 \dot{\theta},
   \]
   the paper proceeds to "integrate" to obtain (Equation (5)):
   \[
   \frac{d\theta'}{dt} = \int \ddot{\theta}' dt = \frac{d\theta}{dt} - \frac{1}{2}K_1 \theta^2 - K_2 \theta,
   \]
   with \(\theta^2\) defined elementwise. This is incorrect on multiple levels:
   - If \(u = -K_1\theta - K_2\dot{\theta}\), then
     \[
     \int u\,dt = -K_1\int \theta(t)\,dt - K_2 \theta(t) + C,
     \]
     not \(-\frac{1}{2}K_1\theta(t)^2 - K_2 \theta(t)\). The latter would require treating \(\theta\) as the integration variable, not time, and even then would mix up \(\int \theta\,d\theta\) with \(\int \theta\,dt\).  
   - The indefinite integral must include a constant of integration, which is ignored.  
   - There is no justification for equating \(\int \ddot{\theta}\,dt\) with \(\dot{\theta}\) *plus* a closed‑form function of \(\theta\) alone without solving an ODE. In general, \(\ddot{\theta}' = f(\theta,\dot{\theta})\) cannot be integrated in closed form to yield \(\dot{\theta}'\) as a simple algebraic function of \(\theta\).
   Consequently, Equation (5) does **not** correctly follow from Equation (4), and the proposed discrete Algorithm 1, which uses
   \[
   g_t = \nabla_\theta L(\theta_t) - K_1 \theta_t^2 - K_2 \theta_t,
   \]
   is not mathematically grounded in the preceding control design. This is a major flaw: the theoretical stability guarantees are derived for a different continuous‑time system than the one that is actually being discretized and used in experiments.

3. **Questionable stability conclusions for convex but not strongly convex losses.**  
   Section 4.2.2 (Page 5) claims that if \(L\) is convex but not strongly convex, then the lifted system is unstable because the algebraic multiplicity of \(\lambda=0\) exceeds its geometric multiplicity, implying a Jordan block of size greater than \(1 \times 1\). The reasoning is incomplete and, as stated, dubious:
   - For \(H \succeq 0\) with at least one zero eigenvalue, the characteristic polynomial is \(\prod_i \lambda(\lambda + \lambda_i)\). If exactly one \(\lambda_i = 0\), then \(\lambda=0\) indeed has multiplicity at least \(n+1\). However, the dimension of the nullspace of \(J\) is not derived; the paper asserts without calculation that geometric multiplicity is "strictly less than" algebraic multiplicity. This needs explicit computation of eigenvectors for \(\lambda=0\), similar to the calculation done in Section 4.2.1 for the strongly convex case.  
   - Even if a nontrivial Jordan block exists at \(\lambda=0\), this gives polynomially growing modes in continuous time, but that does not automatically translate to divergence of \(\theta(t)\) from the optimizer; the state \((\theta,\dot{\theta})\) can have marginal or drift behavior while \(\theta\) itself still converges or remains bounded. The text claims "solutions grow linearly over time" without actually solving the ODE or showing how \(\theta(t)\) behaves.  
   - Empirically, **Figure 2(b, e)** for \(L(\theta) = \theta_1^2 + \theta_2^2\) uses \(\eta = 0.995\) which is extremely close to, but below, the discrete stability limit. GD in that setting converges slowly but is linearly stable (and the loss in Figure 2(e) indeed decreases). The "instability" referred to in the text is thus ambiguous and not consistent with standard discrete‑time theory.
   Overall, the convex‑but‑not‑strongly‑convex instability narrative is neither rigorously proved nor aligned with classical understanding of gradient flow, which is known to converge for convex but not strongly convex functions under mild conditions.

4. **Control design assumptions are unrealistic and not operational in practice.**  
   Definition 4 requires selecting \(K_1 \succ 0\) and \(K_2\) such that \(H(\theta) + K_2 \succ 0\) for all \(\theta\). This is critical because Lemma 4 demands \(C \succ 0\) and \(K \succ 0\) in the quadratic eigenvalue problem \(Q(\lambda) = \lambda^2 M + \lambda C + K\). However:
   - For general nonconvex neural network losses, \(H(\theta)\) can have arbitrarily negative eigenvalues; ensuring \(H(\theta) + K_2 \succ 0\) for **all** \(\theta \in \mathbb{R}^d\) with a fixed \(K_2\) is infeasible unless one bounds \(\|H(\theta)\|\) globally and takes \(K_2\) extremely large, which is not discussed.  
   - Remark 2 suggests empirically choosing \(K_1=\mu I\) and "choosing \(K_2 \succ -H(\theta)\) for all \(\theta\)", which is circular: one cannot know \(-H(\theta)\) for all \(\theta\) in advance, nor compute it in high dimensions. In experiments, the authors simply use small scalar values \(k_1=k_2\) (e.g., 0.01 or 0.2), which almost certainly do not satisfy the stated theoretical condition globally.  
   As a result, Theorem 3's conclusion that the system is "locally asymptotically stable regardless of curvature" is predicated on assumptions that are neither verifiable nor enforced in realistic settings. The disconnect between these strong assumptions and the empirical implementation significantly weakens the theoretical relevance.

5. **Very limited and non‑representative empirical evaluation.**  
   Experiments are restricted to three 2‑D toy objectives (ellipse, sphere, quartic) with hand‑picked learning rates and hyperparameters (Sections 7.1 and 7.2). There are no experiments on actual neural networks, no higher‑dimensional problems, and no stochasticity. For a method positioned as "Controlled Gradient Descent for Neural Network Training" (Algorithm 1), this is severely inadequate:
   - There is no evaluation on even simple logistic regression, linear regression with ill‑conditioned Hessians, or small MLPs, let alone modern deep networks.  
   - There are no comparisons against simple baselines with similar computational cost, such as momentum, Nesterov acceleration, gradient clipping, or adaptive step‑size methods like AdaGrad/Adam that are often used to stabilize training and allow larger effective step sizes.  
   - No quantitative tables summarize performance (e.g., convergence speed, max step size before divergence) across a grid of learning rates; all evidence is via plots. The only table, **Table 1** (Page 5), is a *theoretical* summary, not empirical.  
   For ICLR standards, the experimental evidence is too weak to assess practical utility or robustness.

6. **Missing and under‑positioned related work in control‑theoretic and ODE views of optimization.**  
   The Related Work section (Section 1.1) focuses almost exclusively on edge‑of‑stability, sharpness, and SGD‑as‑SDE perspectives. It misses several bodies of work that have already used control theory, Lyapunov analysis, and continuous‑time ODEs to analyze and *design* optimization algorithms, many of which are directly relevant: Lyapunov analyses of momentum, IQC‑based design, high‑resolution ODEs for accelerated methods, and variational frameworks for Nesterov‑style dynamics. This absence makes the contribution look less original than it might be and prevents a nuanced comparison (details listed in the "Potentially Missing Related Work" section below).

7. **Ambiguous interpretation of the "second‑order" formulation and its necessity.**  
   The second‑order system arises trivially by differentiating gradient flow with respect to time, \(\ddot{\theta} = -H(\theta)\dot{\theta}\). This is not the usual physical second‑order system (like mass‑spring‑damper) where the state is position and velocity, driven by a potential. Here, the "acceleration" depends on the velocity via the Hessian. The paper does not justify why working with this particular second‑order formulation is more informative than directly analyzing the first‑order gradient flow (which has well‑studied stability properties under convexity/PL conditions). The controller \(u\) is then added at the acceleration level and converted back to an ad hoc first‑order update via the flawed integration (Equation (5)). As a result, the claimed "second‑order control‑theoretic dynamics" do not convincingly lead to a useful or interpretable algorithm.

8. **Multiple issues in exposition and technical precision.**  
   There are several smaller but cumulative clarity and correctness problems:
   - Theorem 2's statement on Page 4 has a typo: “unstable if the loss function \(L\) is convex but not strongly concave” should be “concave” or "not strongly convex"; as written it is nonsensical.  
   - In Equation (3), the Jacobian block \(-\sum_{i=1}^n x_i \frac{\partial H}{\partial \theta_i}\) is not further analyzed; then, in Section 4.2 the authors abruptly drop this term at equilibrium. While correct, there is no explicit argument that it vanishes when \(\dot{\theta}=x=0\).  
   - Throughout, the notation oscillates between bold and non‑bold for vectors and matrices, and some equations have minor inconsistencies (e.g., \(\nabla_L^2\) instead of \(\nabla^2 L\)).  
   - Section 7 states that standard GD "diverges" on the strongly convex ellipse with \(\eta = 0.5\), yet **Figure 1(c)** actually shows the loss monotonically decreasing for both GD and CGD, with essentially identical curves. The narrative in the text should be reconciled with the behavior shown in the figure.

9. **Theoretical contribution is incremental relative to known results on gradient flow and convexity.**  
   Once the derivations are stripped of errors, much of the analytical content is a restatement of classical results: strong convexity implies positive definite Hessian (Lemma 1), convexity implies positive semidefinite (Lemma 2), and concavity implies negative semidefinite (Lemma 3). The new aspect is the specific lifting to a 2n‑dimensional system and observing eigenvalues \(0\) and \(-\lambda_i(H)\). However, the use of local linearization and eigenvalue conditions for Lyapunov stability is textbook Khalil/Perko material. Given the major flaw in translating this to a concrete algorithm (Equation (5)), the net theoretical advance is modest.

Given the foundational error in Equation (5), the strong unproven claims about discrete GD, and the weak empirical validation, I do not think the submission in its current form meets ICLR standards.

## Potentially Missing Related Work

Below are directly relevant works that should be discussed, contrasted, and, where appropriate, used as conceptual baselines, ideally in Section 1.1 and in a more expanded related‑work section:

1. **Su, W., Boyd, S., & Candès, E. (2014). "A Differential Equation for Modeling Nesterov's Accelerated Gradient Method: Theory and Insights."**  
   - Relevance: Develops a continuous‑time ODE that models Nesterov’s method and analyzes stability and convergence; very close in spirit to modeling discrete optimization via ODEs.  
   - Suggestion: Compare in Sections 1.1 and 3, emphasizing how the proposed second‑order dynamics differ from classical Nesterov ODEs and whether the controller can be interpreted as modifying inertial or damping terms.

2. **Wibisono, A., Wilson, A. C., & Jordan, M. I. (2016). "A Variational Perspective on Accelerated Methods in Optimization."**  
   - Relevance: Provides a variational and Lagrangian framework for accelerated methods using continuous‑time dynamics.  
   - Suggestion: Section 5 currently mentions a "variational interpretation" of the controller but does not actually connect to existing variational frameworks; citing and contrasting with this work would clarify what is genuinely new.

3. **Shi, B., Du, S. S., & Jordan, M. I. (2019). "Understanding the Acceleration Phenomenon via High-Resolution Differential Equations."**  
   - Relevance: Uses high‑resolution ODEs to analyze optimization dynamics; strong methodological parallel.  
   - Suggestion: Discuss in Sections 3–4, as it also refines gradient‑flow models to better capture discrete behavior, which is precisely the missing bridge in this paper.

4. **Wilson, A. C., Recht, B., & Jordan, M. I. (2016). "A Lyapunov Analysis of Momentum Methods in Optimization."**  
   - Relevance: Uses Lyapunov functions to analyze stability and convergence of momentum methods from a control-theoretic standpoint.  
   - Suggestion: Cite in Sections 2 and 5, and contrast the proposed controller with classical momentum/Polyak updates and their stability properties.

5. **Lessard, L., Recht, B., & Packard, A. (2016). "Analysis and Design of Optimization Algorithms via Integral Quadratic Constraints."**  
   - Relevance: Pioneering work that explicitly frames optimization algorithms as dynamical systems and uses control‑theoretic tools (IQCs) to design and analyze them.  
   - Suggestion: Section 1.1 should acknowledge this line of work and clarify how the proposed controller differs from IQC‑based design; it is currently absent but highly relevant.

6. **Scieur, D., Bach, F., & d'Aspremont, A. (2018). "Regularized Nonlinear Acceleration."**  
   - Relevance: Introduces a regularized approach to accelerating gradient methods by modifying updates; conceptually related to designing modified dynamics for better behavior.  
   - Suggestion: Compare in Sections 5–6, especially regarding how their regularization affects stability and step‑size robustness versus the proposed \(K_1, K_2\) terms.

7. **Krichene, W., Bayen, A. M., & Bartlett, P. L. (2015). "Accelerated Mirror Descent in Continuous and Discrete Time."**  
   - Relevance: Analyzes accelerated methods in both continuous and discrete time, bridging the gap that is currently missing in this paper.  
   - Suggestion: Discuss in Section 3–6 how to rigorously connect continuous‑time guarantees to discrete updates and what this implies for the proposed approach.

8. **Ghadimi, S., & Lan, G. (2016). "Accelerated Gradient Methods for Nonconvex Nonlinear and Stochastic Programming."**  
   - Relevance: Addresses nonconvex and stochastic settings with accelerated methods; directly related to the paper’s motivation of stabilizing training in general nonconvex settings.  
   - Suggestion: Cite in Introduction and describe how CGD compares to or complements these methods in nonconvex regimes.

9. **Ochs, P., Chen, Y., Brox, T., & Pock, T. (2014). "iPiano: Inertial Proximal Algorithm for Nonconvex Optimization."**  
   - Relevance: Uses inertial (second‑order) dynamics along with proximal steps for nonconvex optimization, with convergence guarantees.  
   - Suggestion: Position the proposed second‑order control‑based method relative to iPiano in Section 5, particularly regarding stability guarantees versus nonconvexity.

10. **Attouch, H., Peypouquet, J., & Redont, P. (2016). "Fast Convergence of Inertial Dynamics and Algorithms with Asymptotic Vanishing Damping."**  
    - Relevance: Studies inertial dynamics with vanishing damping and their fast convergence; again, a second‑order dynamical‑systems view of optimization.  
    - Suggestion: Discuss in Section 5 as related work on stabilizing and accelerating second‑order dynamics, and clarify how the proposed fixed controller compares to vanishing damping schemes.

Incorporating and discussing these works would significantly improve the paper’s positioning and clarify what is actually new relative to a substantial existing literature on control‑theoretic and ODE‑based analysis of optimization.

## Questions

1. **Correct form of the controlled update:**  
   Can you provide a rigorous derivation from Equation (4),
   \[
   \ddot{\theta}' = -H(\theta)\dot{\theta} - K_1\theta - K_2 \dot{\theta},
   \]
   to a first‑order update rule for \(\dot{\theta}'\) that justifies Equation (5)? In particular:
   - How do you justify \(\int u \, dt = -\tfrac{1}{2} K_1 \theta^2 - K_2 \theta\) given that \(\theta=\theta(t)\) is time‑dependent?  
   - What initial conditions and constants of integration are being used?  
   A corrected and explicit derivation may substantially change the form of Algorithm 1.

2. **Formal link between continuous‑time and discrete‑time stability.**  
   Your theorems concern the eigenvalues of the Jacobian of the continuous‑time system. What precise assumptions or theorems do you invoke to argue that these results extend to discrete GD with a finite step size \(\eta\)? For example:
   - Can you characterize the discrete‑time Jacobian of the lifted map \((\theta_t, \dot{\theta}_t) \mapsto (\theta_{t+1}, \dot{\theta}_{t+1})\) corresponding to Algorithm 1?  
   - Are there step‑size constraints under which your continuous‑time stability guarantees imply discrete‑time Lyapunov or asymptotic stability?

3. **Feasibility of ensuring \(H(\theta) + K_2 \succ 0\).**  
   In Definition 4 and Remark 2, how do you propose to guarantee \(H(\theta)+K_2 \succ 0\) for all \(\theta\) in practice when \(H\) is high dimensional and possibly highly nonconvex?
   - Are you assuming a global upper bound on \(\|H(\theta)\|\)?  
   - If not, is the stability result meant only to apply in a local neighborhood where the Hessian is bounded? Clarifying this could make Theorem 3 more realistic and better scoped.

4. **Behavior on higher‑dimensional and realistic problems.**  
   Have you tested CGD on:
   - Moderately high‑dimensional quadratic problems (e.g., \(d=100\)) with ill‑conditioned Hessians?  
   - Simple neural networks (e.g., 2‑layer MLP on MNIST or CIFAR‑10)?  
   Evidence from such experiments, possibly comparing with momentum and Adam, would greatly help assess whether your stabilization idea meaningfully improves training in practical settings.

5. **Interpretation of "instability" in convex cases.**  
   In Section 4.2.2 and Section 7.1, what precise notion of instability are you using for convex losses like \(L(\theta) = \theta_1^2 + \theta_2^2\)?  
   - Are you referring to Lyapunov instability of the lifted state \((\theta,\dot{\theta})\), or divergence of \(\theta(t)\) from the optimizer, or oscillations in the discrete iterate sequence?  
   - Could you clarify this in the text and possibly add a figure decomposing \(\theta_1\), \(\theta_2\), and \(\dot{\theta}\) over time to support your claim?

Addressing these questions with precise mathematics and additional experiments could substantially change my assessment.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The continuous‑time linearization and control‑theoretic argument for the *idealized* controlled ODE are mostly sound under strong assumptions, but the derivation of the implemented algorithm (Equation (5) and Algorithm 1) is mathematically incorrect, and the link from continuous to discrete dynamics is not established. Empirical evidence is too limited to compensate.

## Presentation Rating

2: fair.  
The high‑level narrative and many derivations are readable, and figures (especially Figures 1–3) communicate qualitative behavior. However, key derivations are incorrect or hand‑wavy, several statements are misleading or imprecise, related work is incomplete, and some terminology around stability and curvature is ambiguous.

## Contribution Rating

1: poor.  
While the perspective of applying a controller to GD dynamics is potentially interesting, the theoretical bridge from the controlled ODE to the actual update rule is broken, and the experimental validation is minimal and toy‑level. Relative to extensive prior work on ODE‑based and control‑theoretic analysis/design of optimization algorithms, the incremental contribution is limited in its current form.

## Overall Rating

2: Reject, not good enough.  
The paper explores a reasonable idea and has some pedagogical value, but there is a serious mathematical flaw in the derivation of the proposed algorithm, a loose connection between continuous‑time stability results and discrete GD behavior, and extremely limited experiments. Substantial reworking of both theory (fixing Equation (5) and clarifying assumptions) and experiments (adding realistic problems and baselines) would be needed for this to reach ICLR main‑track standards.

## Reviewer Confidence

4: confident.  
I am familiar with dynamical systems, control‑theoretic analyses of optimization, and gradient‑flow‑based modeling, and I carefully checked the key derivations (especially Equations (2)–(5) and the use of Lemma 4). While I may have missed some subtleties, I am confident that the integration step leading to the practical update and the claimed implications for discrete GD stability are problematic as written.