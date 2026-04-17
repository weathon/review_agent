# Wasserstein Policy Gradient: Implicit Policies, Entropy Regularization and Linear Convergence

- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
We revisit Wasserstein Proximal Policy Gradient (WPPG) for continuous control in infinite-horizon discounted reinforcement learning. By projecting the iterate of Wasserstein proximal gradient onto a parametric policy family with respect to the Wasserstein distance, we derive a new WPPG update that eliminates the need for policy densities or score functions. This makes our method directly applicable to implicit stochastic policies. We prove a linear convergence rate for the WPPG iterate under entropy regularization and a log-Sobolev condition on the policy class, for both exact and approximate value function estimates. Empirically, our algorithm is simple to implement and achieves competitive performance on standard benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new formulation of Wasserstein Policy Proximal Gradients with entropy regularization. The update rule can be applied to optimizing both explicit policies and implicit polices. The paper presents the convergence analysis when the update rule is applied to per-state maximization under exact and inexact Q estimate.  The paper also evaluates different variants of the proposed method (approximate version) on continuous control tasks.

### Strengths
### Originality
It’s novel to see that the proposed WPPG formulation can be applied to two different types of policies: the explicit parameterization with density functions and the implicit parameterization as state-to-action mapping. The empirical evaluation shows the method can training performant policies for both cases. 

### Quality and clarity
The paper is well-written and easy to understand. All notations and equations are clearly defined and explained. The proposed methods have been rigorously studies from both the theoretical perspective and the empirical evaluation. The paper provides sufficient review on relevant literatures and clearly points out the difference and the gap.

### Weaknesses
### Novelty and Significance
The theoretical results are similar to those in existing publications, including Lan (2021) and Song et al. (2023). Also, these results are built on the tabular-like policy update rule, see Eq. (8) and (11), both of which differ greatly from the optimization of parameterized policies (as specified in Eq. (7)). Specifically, Eq. (8) and (11) are per-state policy update without any policy parameterization. Consequently, the linear convergence rate is applicable only to the exact per-state policy update, not even to the policies with implicit or explicit parameterization. 

The empirical results do not show that the proposed WPPG and its variants are more performant than existing methods on the selected benchmark tasks. In fact, the WPPG and WPPG-I have the similar performance as the SAC on most learning environments. This is sensible since the WPPG and its variants leverage the same entropy regularization as used in SAC. This performance similarity is on the condition that the entropy regularization coefficient was fixed for SAC (“SAC is evaluated with entropy coefficient self-tuning disabled”), which might imply that the SAC could be more performant. The paper should at least present one learning environment that can clearly show the benefits of using WPPG.

### Questions
1. what’s a log-Sobolev condition on the policy class? It was mentioned in both Abstract and Conclusion but not explained in the main texts. 

2. Line 236 – 238 is repetitive, same as the lines 233 – 235

3. Line 267, typo “assuption”

4. For Eq. (7), it only specifies the optimization for the implicit policies. What about the explicit policies? 

5. As WPPG is derived based on the Q function, why the advantage function is then used in the formulation? Line 190.

### Soundness
3

### Presentation
4

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
This paper introduces Wasserstein Proximal Policy Gradient (WPPG), a novel framework for policy optimization in reinforcement learning that performs proximal updates directly under Wasserstein geometry rather than the usual KL-based geometry. The proposed WPPG eliminates dependence on policy densities, enabling the optimization of implicit stochastic policies via gradient updates over action samples. Moreover, the authors derived linear convergence guarantees for both exact and approximate value functions. Empirically, the proposed WPPG outperforms standard baselines such as PPO, SAC, and WPO on MuJoCo benchmarks.

### Strengths
1. The paper provides a clear convergence proof of the proposed WPPG, adapting transport-information inequalities and entropy regularization.
2. The proposed WPPG enables learning in settings where policy density is intractable. And the projection-based update is simple yet powerful, avoiding the explicit computation of the log-density.
3. Empirically, WPPG-I consistently outperforms all baselines (PPO, SAC, WPO) on MuJoCo tasks.

### Weaknesses
1. The experiments are restricted to standard MuJoCo tasks. It would strengthen the paper to include additional environments (e.g., stochastic or discontinuous environments, sparse rewards) to demonstrate the robustness and generality of WPPG.
2. The computational overhead of introducing Wasserstein gradient flow is unclear. The paper briefly reports wall-clock time (Table 7) but does not provide detailed complexity comparisons (e.g., gradient computation cost per update vs. KL-based methods).
3. The effects of $\tau$ and latent dimension are demonstrated only on Humanoid. Moreover, additional ablation experiments on step size $\eta$, sample number K, critic depth, or replay buffer size would provide a clearer view of WPPG’s stability and sensitivity.
4. There are some typos that require further calibration, e.g., a duplicate paragraph appears at the beginning of Section 4.

### Questions
1. Can you provide a deeper theoretical or empirical comparison between the proposed WPPF and WPO? Since both methods use Wasserstein geometry but differ in projection (Wasserstein vs. KL). For example, whether WPPG achieves tighter bounds or better conditioning than WPO in continuous control?
2. Since WPPG optimizes actions using Wasserstein projection, the per-update cost may differ from KL-based methods. Can you provide empirical estimates of the complexity of WPPG updates relative to SAC/PPO/WPO?
3. In your convergence analysis, the log-Sobolev and $T_{2}$ assumptions are strong. Could the analysis be extended to weaker conditions?
4. Have you tested WPPG or WPPG-I on tasks beyond MuJoCo, such as stochastic environments (e.g., AntMaze, POMDPs, or hybrid control), to assess its broader applicability?
5. Can the approximate realizability parameter $\delta$ be quantified empirically, or does it vanish asymptotically as network capacity increases?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers policy optimization in distribution space, with specific attention to KL vs Wasserstein distance metrics. They propose Wasserstein Proximal Policy Gradient (WPPG), a policy optimization approach which eliminates needing access to policy densities or score functions and is readily applicable to optimizing implicit stochastic policies. They provide a linear convergence proof under entropy regularization and a log-Sobolev condition. They then evaluate WPPG and its implicit variant (WPPG-I) empirically.

### Strengths
* The paper provides good theoretical contributions in both WPPG's derivation and analysis, which—as far as I was able to verify—appear correct. For each main assumption at the start of 4.1, they provided details on how they might be met.

* The resulting algorithm is relatively easy to implement, which is not always the case when dealing with the Wasserstein metric. The sole dependence on the action-value gradient broadens its applicability to implicit policies and crucially opens a lot of avenues for further exploration.

* The paper is relatively well written and easy to follow, situates itself well among prior work, ad provides good motivation for considering the Wasserstein metric (e.g., respecting the geometry of the action space, intuitions behind earth-moving distance, etc.).

* The empirical evaluation provided relevant ablations to better understand the algorithm and its sensitivities.

### Weaknesses
* The empirical evaluation was relatively limited in that it only considered the MuJoCo suite. This isn't a huge concern though if the paper's primary contributions are theoretical.

* In the derivation of the Wasserstein projection, the shared-latent coupling is first-order optimal under small $\eta$. However, the convergence theorem lower bounds $\eta \geq \frac{1}{\gamma \lambda \tau}$, making the derivation and analysis appear at odds.

Minor which did not impact my review:

* The intro paragraph to Section 4 was duplicated.
* "under the geometry of optimal transport. (Pfau et al., 2025)." -> "under the geometry of optimal transport (Pfau et al., 2025)."

### Questions
* Can the authors comment on the seeming conflict between the derivation ideally having $\eta$ be small and the analysis requiring $\eta$ be sufficiently large? What are the consequences for $\eta$ being too large?

* In the empirical evaluation, what exactly is being presented? It claims to present results over 10 independent evaluation runs—does this suggest that there was one learning run, and at regular intervals, the current policy was saved and run 10 times to compute its average return? Or is this the average of 10 independent learning runs?

* While I appreciate the comparison in B.4 where the impact of single- vs. double-Q in WPPG was explored, is there any reason why WPO + double-Q was not tried? From these ablations, this had a very dramatic effect on WPPG's performance, that WPO might similarly be on par with SAC if it were to also maintain two action-value functions?

* In the empirical evaluation, why were the shaded regions chosen to represent one standard deviation? This is a measure of variation and not confidence, the latter of which is more relevant for making claims about differences in performance. While this can be used to compute a standard error, it would be more informative to present the standard error or confidence intervals directly.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a projected wasserstein proximal policy gradient method and analyzed its global convergence behavior under various assumptions on the policy class. Numerical experiments accompanied the theoretical framework proposed.

### Strengths
I find the motivation to use Wasserstein metric in the proximal policy gradient update to be convincing.

### Weaknesses
Main concerns:

- Related work: Proximal methods can be interepreted as soft-version of trust region methods. Wasserstein trust region policy optimization methods are considered already a few years back in Terpin, A., Lanzetti, N., Yardim, B., Dorfler, F., & Ramponi, G. (2022). Trust region policy optimization with optimal transport discrepancies: Duality and algorithm for continuous actions. Advances in Neural Information Processing Systems, 35, 19786-19797. Please discuss and compare with this related work. 

- Assumptions: there are quite a few assumptions on the coverage and smoothness of policy class. While there are some accompanying discussions next to the assumptions, I am wondering if the authors can provide a concrete example of policy class parametrization therein that satisfy all of the assumptions without vague wording like "designing the neural network to have sufficient smoothness and non-degeneracy."

Minor points:

- There seems to be some latex issues when composing the theorem environment. Instead of showing the theorem environment, it shows plain text with [linear convergence]. Please fix these.

- Line 267: "assuption"

### Questions
See previous section

### Soundness
3

### Presentation
1

### Contribution
3
