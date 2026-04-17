---
job_id: 00cf0ca3-cbab-4edc-b959-047248a3ecff
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: klB3AvQZqJ.pdf
paper: Constraint-Aware Reward Relabeling for Offline Safe Reinforcement Learning
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely on offline safe reinforcement learning, with representation/value learning, constrained optimization, and safety, which are core ICLR topics.

## Minimum Quality
Pass ✅.  
The paper is in English and has all major sections: Abstract, Introduction, Related Work, Method (Sections 4–5), Experiments/Results (Section 6 and Appendix B), and a brief Summary/Conclusion (Section 7). The methodology is nontrivial, experiments are substantial, and there are no glaring fatal theoretical or experimental mistakes that would justify an automatic desk reject.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any text attempting to instruct or manipulate automated reviewers, nor hidden prompts or similar artifacts.

---

# Expected Review Outcome:

## Summary

The paper tackles offline safe reinforcement learning (OSRL) under CMDP constraints using only pre-collected datasets. It proposes CARL (Constraint-aware Reward Relabeling), a wrapper around standard offline RL algorithms where a cost critic is estimated via off-policy evaluation and then used to relabel rewards in each mini-batch: transitions whose estimated cost-to-go exceeds a user-specified budget receive a large negative penalty, while others retain their original reward.  

The authors present a theoretical formulation based on state-action-wise constraints, show an equivalence between the constrained CMDP (under strong assumptions) and an unconstrained penalty-based objective (Theorem 1), and empirically evaluate CARL primarily with TD3-BC (and secondarily with IQL) on the DSRL benchmark suite, reporting strong safety and reward trade-offs, especially under small cost budgets.

---

## Strengths

1. **Clear, simple algorithmic idea with strong practical appeal.**  
   The core mechanism in Equation (5) and Algorithm 1 (Page 6) is extremely straightforward: estimate a cost Q-function $Q_c^{\pi}$, then relabel rewards in each batch by replacing $r$ with $-V_{\max}$ (or $-R_{\max}$ in practice) whenever $Q_c^{\pi}(s,a) > \kappa$. This makes CARL easy to implement on top of existing offline RL libraries and attractive for practitioners who want safety without dealing with dual variables or complex generative models.

2. **Reasonable theoretical framing of state-action-wise constraints.**  
   Section 4’s formulation (Equation (2)) that enforces $Q_c^{\pi}(s,\pi(s)) \le \kappa$ for all states is a stronger and, for many safety-critical domains, more natural specification than the standard expectation constraint in Equation (1). The reduction to an unconstrained maximization of the penalized reward (Equation (3)) and the equivalence in Theorem 1 help conceptually justify the reward relabeling objective.

3. **Strong empirical safety performance across many benchmarks.**  
   Table 1 (Page 8) is a central piece of evidence: across 19 DSRL tasks with stringent budgets ($\kappa=5$ for Bullet and $\kappa=10$ for Safety Gym), CARL is safe (normalized cost $\le 1$) on all Bullet tasks and on 8/11 Safety Gym tasks. This contrasts sharply with many baselines (e.g., COptiDICE, CDT, CCAC) that often blow through the budget. Importantly, CARL usually maintains competitive or near-best normalized reward among methods that are actually safe (blue bold entries), which is precisely the regime OSRL cares about.

4. **Robustness to varying cost budgets and to unsafe-only data.**  
   - Figure 2 (Page 9) illustrates how CARL scales rewards up as the budget increases from 10 to 40/80 on several Safety Gym tasks, while keeping normalized cost under 1, even in settings (CarCircle2) where CAPS and CCAC remain unsafe (Table 6, Page 18). This supports the claim that CARL does not collapse into over-conservative behavior as the budget relaxes.  
   - Figure 3 (Page 9) is particularly interesting: when trained solely on unsafe trajectories, the red CARL-generated trajectories shift mass into the safe region (left of the dashed cost line) with competitive rewards in AntCircle, BallCircle, and AntVelocity. This shows that reward relabeling can effectively “recycle” unsafe data rather than discarding it, which is a practically important feature in real-world datasets where unsafe behavior is common.

5. **Backbone-agnostic design validated across two different offline RL algorithms.**  
   CARL is evaluated with both TD3-BC and IQL backbones. Table 2 (Page 8) shows that wrapping IQL with CARL yields safe policies with rewards roughly on par with TD3-BC across several Bullet and Velocity tasks, despite the substantial architectural differences between TD3-BC and IQL. This supports the claim that CARL’s logic is decoupled from specific algorithmic details.

6. **Empirical analysis of OPE noise and penalty variants.**  
   Appendix B gives unusually detailed ablations for an OSRL paper. Table 3 (Page 15) shows how performance degrades as synthetic noise is injected into the cost critic’s safe/unsafe decisions: CARL remains safe up to roughly 20% random flips and only breaks badly around 50%, suggesting some robustness to miscalibrated $Q_c$. Table 4 (Page 16) with random penalties further demonstrates that the learned safety behavior is not just an artifact of indiscriminately punishing transitions, but indeed depends on the structure learned by $Q_c$. Table 5 (Page 17) examines the effect of harsher penalties using $V_{\max}$ vs $R_{\max}$, highlighting trade-offs between reward and conservatism.

7. **Good experimental coverage and reporting.**  
   The main experiments span Bullet-Safety-Gym, Safety-Gymnasium, and velocity-constrained Mujoco tasks (Ant, HalfCheetah, Swimmer). Additional results with larger budgets, unsafe-only training data, alternative backbones, and more seeds (Table 9, Page 20) provide a fairly comprehensive empirical picture. The environment visuals in Figure 4 (Page 20) help clarify the domains and nature of safety constraints.

---

## Weaknesses

1. **Limited conceptual and theoretical novelty beyond classical penalty-based control.**  
   At a conceptual level, CARL is “reward shaping by large penalties based on estimated cost-to-go,” which is very close to standard penalty-based safe RL and to existing CMDP Lagrangian methods where $r' = r - \lambda c$ (discussed by the authors in Section 3, Page 3). The main difference is that CARL uses an estimated $Q_c^{\pi}(s,a)$ to gate a fixed extreme penalty $-V_{\max}$ / $-R_{\max}$ rather than a tunable multiplier.  
   - Theorem 1 (Page 4) essentially formalizes that, under the assumption that we *know* $Q_c^{\pi}$ exactly and choose the right penalty, safe policies solving the pointwise-constrained problem are equivalent to those solving the unconstrained penalized problem. This is a rather standard “big penalty enforces constraints” argument and does not bridge the significant conceptual gap to the approximate, function-approximated, offline setting actually used.  
   - The iterative policy improvement sketch (Equation (4), Page 4) is a straightforward policy evaluation → reward relabeling → policy improvement loop. The paper does not provide new convergence guarantees or analysis under approximation, which is where the hard technical questions lie.  
   Overall, the paper’s primary contribution is a careful empirical study of a well-known idea instantiated for OSRL, rather than a fundamentally new optimization or modeling principle.

2. **Theorem 1 is built on unrealistic assumptions and does not inform the practical algorithm.**  
   Theorem 1 assumes:
   - Existence of an optimal safe policy $\tilde{\pi}^*$ for Problem (2) and  
   - Access to *exact* $Q_c^{\pi}$ and $V_{\max} = R_{\max}/(1-\gamma)$,  
   then shows equivalence between Problems (2) and (3). However:  
   - In practice, CARL uses an approximate cost critic learned from a fixed dataset and almost always uses $R_{\max}$ extracted from data, not $V_{\max}$, as the penalty (Section 6.2, Page 7 and Table 5 in the appendix). This breaks the assumptions of the theorem.  
   - The theorem says nothing about how approximation, function approximation error, or dataset coverage affect safety. The algorithm actually deployed (Algorithm 1, Page 6 with $M=K=1$ updates and learned $Q_c$) could, in principle, oscillate or converge to unsafe local optima. The authors themselves acknowledge “theoretical convergence guarantees are unclear” (Page 6) but do not attempt even partial analysis (e.g., monotonicity conditions, bounds on constraint violation under bounded $Q_c$ error, etc.).  
   As a result, the theoretical section feels decoupled from the algorithmic practice and primarily serves as motivation rather than as a rigorous underpinning.

3. **Safety guarantees in the approximate OPE setting are entirely empirical, not principled.**  
   CARL’s core safety mechanism hinges on the cost critic $Q_c^{\pi}$. Misestimating $Q_c^{\pi}$ in low-data or out-of-distribution regions can cause either:  
   - under-penalization of truly unsafe actions (leading to constraint violations), or  
   - over-penalization of safe-but-rare actions (resulting in overly conservative policies).  
   Appendix B.1 (Table 3, Page 15) attempts to probe robustness by *synthetically* flipping safe/unsafe decisions with fixed probabilities. While this is helpful, it does not capture realistic structured errors from FQE like extrapolation bias, biased coverage, or correlated errors across states. Because CARL acts on a binary threshold $Q_c^{\pi}(s,a) \le \kappa$, it is particularly sensitive around the decision boundary. No analysis or experiments consider situations where $Q_c^{\pi}$ is systematically biased (e.g., optimistic in sparse-cost regions).  
   Without at least some bound relating $Q_c$ error to expected constraint violation, it is hard to reason about when CARL can be trusted in truly safety-critical applications. The current message is “it seems to work on these benchmarks,” which is useful but falls short of the paper’s strong rhetoric about safety.

4. **Mathematical and notation issues, plus lack of precision in some key definitions.**  
   A few examples that matter for clarity and rigor:
   - Page 2: “Similarly $V_{c}^{\pi}(s)$ and $Q_{r}^{\pi}(s,a)$ denote the cost state- and ation-value functions respectively.” That should presumably be $Q_{c}^{\pi}$ not $Q_{r}^{\pi}$, and “action-value” is misspelled. This is minor but sloppy and confusing.  
   - Equation (3), Page 4, defines $r_{\pi}(s,a):=1_{\{Q_{c}^{\pi}(s,a)\leq\kappa\}}\cdot r(s,a)-1_{\{Q_{c}^{\pi}(s,a)>\kappa\}}V_{\max}$, but the text immediately refers to maximizing $V_{r_s}^{\pi}$, suggesting inconsistent notation ($r_{\pi}$ vs $r_{s}$).  
   - In Theorem 1’s proof, the step  
     $$V_{r_{\pi^*}}^{\pi^*}(s)=-V_{\max}+\mathbb{E}_{\pi^*,P}\left[\sum_{t=1}^{\infty}\gamma^t r_{\pi^*}(s_t,a_t)\right] < 0 < V_{r}^{\tilde{\pi}^*}(s)$$  
     relies on the assumption that subsequent rewards under $\pi^*$ are bounded above by $V_{\max}$, but this is not spelled out. Also, the last equality $V_{r}^{\tilde{\pi}^*}(s)=V_{r_{\pi^*}}^{\tilde{\pi}^*}(s)$ conflates $r$ and $r_{\pi^*}$; under the theorem’s assumptions it is true for safe policies, but this should be stated clearly.  
   - Algorithm 1’s use of $\mathcal{A}_{\mathrm{OPE}}$ and $\mathcal{A}_{\mathrm{OPO}}$ (Page 5–6) abstracts away crucial details: what are the exact targets for FQE, what is the bootstrapping horizon, how are terminal costs handled, and how is the policy $\pi$ used in OPE (e.g., importance weighting vs direct TD)? These choices directly affect the learned $Q_c$ and thus safety.

5. **Methodology under-specified in a few important ways.**  
   While Appendix C.2 (Table 10, Page 21) provides some hyperparameters, several important implementation details are missing or only lightly stated in the main text:
   - FQE details: The paper mentions “fitted Q evaluation (FQE)” (Page 7) but omits the loss function, target definition, sampling strategy, and how terminal states / truncated episodes are treated. Different FQE implementations can yield noticeably different bias/variance trade-offs.  
   - CARL penalty choice: The main experiments use $R_{\max} = \max_{(s,a,r)} r$ from the offline data (Section 6.2), but it is not fully clear whether this is per-environment, per-dataset split, or across all tasks. Also, the link between $R_{\max}$ and the theoretical $V_{\max}$ is hand-waved.  
   - Thresholding: No discussion is given on how to deal with stochastic costs or whether a margin is used around $\kappa$ (e.g., penalizing when $Q_c^{\pi} > \kappa - \epsilon$ to compensate for estimation variance). As a result, the CARL rule in Equation (5) is quite brittle in principle.

6. **Baseline coverage is good but not fully up to date and unevenly tuned.**  
   The baseline suite in Table 1 is strong (BC-Safe, CPQ, COptiDICE, CDT, CAPS, CCAC, FISOR), and the paper also includes BEAR-Lag and BCQ-Lag comparisons in Table 5. However, several directly relevant recent works that combine safety with generative modeling or reward/safety relabeling are not included or discussed, for example:
   - Diffusion-regularized offline safe RL focusing on reward–safety balance (see missing related work list below), which seems very close in spirit to CARL but with a generative regularizer rather than deterministic reward relabeling.  
   - Reward-relabeled offline RL variants (e.g., optimal-transport-based offline imitation, survival instinct / robustness to corrupted rewards, hindsight relabeling in offline settings).  
   Moreover, it is not entirely clear whether each baseline was tuned fairly under the strict $\kappa = 5/10$ regimes; e.g., FISOR is designed for zero-violation but is evaluated primarily at fixed settings. The paper would benefit from explicitly documenting the hyperparameter tuning protocol for baselines, particularly the Lagrangian-based ones known to be sensitive.

7. **Evaluation focuses heavily on “is $C_{\text{norm}} \le 1$” without deeper analysis of safety–reward Pareto fronts.**  
   The binary notion of safety “normalized cost $\le 1$ or not” is consistent with DSRL, but it does not reveal how far a method is from the boundary. For example, many CARL entries in Table 1 have cost $\ll 1$; it is unclear whether slightly relaxing the penalty could yield noticeably higher reward with small additional cost. Similarly, some baselines with $C_{\text{norm}} \approx 1.1$ are labeled simply as unsafe, but might be desirable in practice compared to ultra-conservative CARL solutions.  
   Figure 2 partially addresses this as budgets vary, but it only compares CARL against CAPS and CCAC, not CPQ/FISOR/others, and only on a handful of tasks. A more systematic visualization of reward vs cost trade-offs for all methods (akin to the scatter plots in Figure 3 but per method rather than per trajectory) would provide a much clearer picture of how CARL compares on the full Pareto frontier.

8. **Some experimental design choices could mask weaknesses.**  
   - For the small-budget regime, CARL is sometimes compared to baselines that are not explicitly designed for very low $\kappa$ (the authors note FISOR as the only prior method targeting “small budgets”). It would be helpful to show whether simply tuning Lagrangian multipliers or penalties more aggressively could make methods like CPQ or COptiDICE competitive in this stricter regime.  
   - The hard-filtering variant in Table 8 (Page 19) is constructed as a strong negative control; unsurprisingly, throwing away all predicted-unsafe transitions leads to poor performance. But this is a strawman compared to more sophisticated feasibility-based baselines (e.g., those approximating safe regions via generative models or classifiers). It supports CARL vs naïve filtering but not necessarily vs the best alternative design choices.

9. **Scope of environments is still fairly narrow and simulator-bound.**  
   All experiments are in physics simulators (PyBullet and Mujoco) with hand-designed cost functions. This is standard in safe RL work, but given the paper’s emphasis on robustness and strong safety, one might hope to see at least one domain with more structured OOD generalization or richer constraint structure (e.g., real-world logs, safety in discrete structured tasks). As it stands, it is unclear how well CARL’s cost critic and reward relabeling generalize to more complex datasets where safety is not strongly correlated with simple geometric boundaries.

---

## Potentially Missing Related Work

Below are works that seem directly relevant to CARL’s setting or mechanism and do not appear in the references:

1. **Guo et al., “Reward-Safety Balance in Offline Safe RL via Diffusion Regularization”, 2025.**  
   - Relevance: This work explicitly studies balancing reward and safety in offline RL using diffusion-based regularization, clearly overlapping the problem CARL addresses.  
   - Suggested integration: It should be discussed in Section 3 (Offline Safe RL) alongside FISOR, CDT, CAPS, OASIS, and CCAC, with a comparison on how CARL’s deterministic penalty relabeling differs from diffusion-based feasibility shaping. If re-implementable, it would also be a natural additional baseline in Table 1.

2. **Luo et al., “Optimal Transport for Offline Imitation Learning”, 2023.**  
   - Relevance: This paper uses optimal transport to perform reward relabeling / cost-aware matching in offline imitation learning, which is conceptually close to CARL’s use of relabeled rewards to shape behavior using offline data.  
   - Suggested integration: It should be cited in Related Work when discussing penalty-based and reward-relabeled offline approaches, and a short discussion added in Section 4 or 5 pointing out the conceptual similarities and differences (e.g., CARL’s binary gating vs OT’s continuous matching).

3. **Li et al., “Survival Instinct in Offline Reinforcement Learning”, 2023.**  
   - Relevance: This work analyzes offline RL under misspecified or corrupted rewards, which is tightly connected to CARL’s reliance on relabeling rewards using a learned cost critic.  
   - Suggested integration: It should be mentioned in Section 3 as related to robustness to incorrect reward labels and in Appendix B.1 when discussing OPE noise robustness, as their analyses and empirical findings could shed light on CARL’s behavior under reward relabeling errors.

4. **Yu et al., “Self-Supervised Imitation for Offline Reinforcement Learning with Hindsight Relabeling”, 2022.**  
   - Relevance: Hindsight relabeling for offline RL is directly analogous to CARL’s relabeling mechanism, albeit with different goals (goal-conditioned success vs safety).  
   - Suggested integration: Add to Related Work under offline RL methods that change rewards post hoc. A short discussion contrasting CARL’s cost-based safety relabeling with hindsight success relabeling would help contextualize CARL in a broader “reward editing” literature.

5. **Yu et al., “How to Leverage Unlabeled Data in Offline Reinforcement Learning”, 2022.**  
   - Relevance: While not about safety, this paper is about using unlabeled trajectories and different reward structures in offline RL. CARL’s use of unsafe trajectories as penalized examples is conceptually related to extracting signal from data with incomplete or undesirable reward signals.  
   - Suggested integration: Cite and briefly discuss in Section 3 or Appendix A’s extended discussion on leveraging unsafe/off-budget data, clarifying how CARL’s relabeling plays a similar role in converting “bad” data into useful learning signal.

---

## Questions

1. **On the practical penalty magnitude and its connection to Theorem 1.**  
   You use $R_{\max}$ (max per-step reward from the dataset) as the penalty in the main experiments, while Theorem 1 assumes $V_{\max} = R_{\max}/(1-\gamma)$. Could you provide a more principled rationale or empirical sweep that justifies $R_{\max}$ as “large enough but not too large”? In particular, are there tasks where using an intermediate penalty between $R_{\max}$ and $V_{\max}$ yields a better reward–cost trade-off?

2. **Behavior under systematic cost critic bias rather than random label flips.**  
   The noise ablation (Table 3) injects i.i.d. flips in safety decisions. Have you examined scenarios where $Q_c^{\pi}$ is systematically biased, for example due to lack of coverage in dangerous parts of the state-action space (e.g., always underestimating costs in regions never visited by behavior policy)? If not, could you sketch how you might construct such a setting and what you expect CARL to do?

3. **Clarification of FQE details.**  
   Please specify the exact FQE update used in Algorithm 1:  
   - What TD target do you use for $Q_c$? Is it $c + \gamma \mathbb{E}_{a' \sim \pi(\cdot|s')} Q_c(s',a')$ or something else?  
   - How do you handle terminal states and truncated episodes in the DSRL datasets?  
   - Are you using target networks, double estimators, or other variance-reduction tricks for $Q_c$?

4. **Sensitivity to the threshold $\kappa$ and possible margin use.**  
   In Equation (5), the decision is $Q_c^{\pi}(s,a) \le \kappa$ or not. Did you experiment with adding a margin (e.g., only penalizing if $Q_c^{\pi}(s,a) > \kappa + \delta$) to compensate for variance in $Q_c$? If so, what did you observe? If not, can you comment on why a strict threshold is preferable?

5. **Comparisons to more recent diffusion-regularized or relabeling-based OSRL methods.**  
   There are recent offline safe RL approaches that balance reward and safety via diffusion or generative regularization (beyond FISOR). Could you comment on how CARL compares algorithmically and empirically to these methods, and whether you anticipate CARL could be combined with such generative components (e.g., using diffusion for candidate actions but still applying CARL relabeling)?

6. **Reward–cost Pareto analysis.**  
   Would it be possible to add plots showing the Pareto frontier of reward vs normalized cost for CARL and at least one or two strong baselines (e.g., CAPS, CCAC) by varying an internal knob (e.g., penalty magnitude or Lagrangian weight)? This would help clarify whether CARL is confined to a single extreme safety operating point or actually gives a good trade-off across budgets.

Author responses that clarify these points, add missing baselines/related work, or provide additional analyses could positively impact my assessment.

---

## Flag For Ethics Review

No ethics review needed.  

---

## Details Of Ethics Concerns

N/A. The work uses standard simulated benchmarks with no human subjects, no sensitive data, and no obviously harmful application domain beyond generic autonomous control. The aim is to improve safety, not to bypass it.

---

## Soundness Rating

3: good.  
The algorithm is conceptually sound and empirically validated across many benchmarks, and the math is mostly correct at the level it is claimed. However, theoretical guarantees in the approximate setting are absent, and some derivations / notational details are sloppy or only loosely aligned with the actual implementation.

---

## Presentation Rating

3: good.  
The paper is generally well written, with clear motivation, structure, and figures (e.g., Figure 1’s oscillation visualization, Figure 2’s budget scaling, Figure 3’s trajectory scatter plots, and Figure 4’s environment screenshots). Some notation errors and under-specified implementation details reduce clarity but are fixable.

---

## Contribution Rating

2: fair.  
The main contribution is a carefully executed empirical study of a very simple penalty-based wrapper for offline safe RL, along with a conceptual reframing via state-action-wise constraints. While the empirical results are strong and the simplicity is attractive, the conceptual and theoretical novelty over standard penalty/Lagrangian methods is limited, and the paper does not provide principled guarantees in the realistic approximate OPE regime.

---

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The work presents a clean, practical idea that seems to work impressively well on a range of OSRL benchmarks, with strong safety performance and a very low implementational barrier. However, the conceptual novelty is modest, the theoretical treatment does not extend to the approximate offline setting actually used, several important implementation details are underspecified, and the positioning relative to closely related recent work on diffusion/relabeled offline safe RL is incomplete. With these caveats addressed and the related work strengthened, I could see this being a solid paper, but in its current form it sits just below what I would expect for ICLR main-track acceptance.

---

## Reviewer Confidence

4: confident.  
I am familiar with offline RL and safe RL, carefully inspected the math and experiments, and cross-checked the claimed contributions against the presented results, though I did not re-implement the method.