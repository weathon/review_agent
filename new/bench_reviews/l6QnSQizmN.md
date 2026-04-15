## Summary
This paper studies online RL in non-stationary **context-driven** environments where an observed exogenous context changes over time and induces changing dynamics. It proposes **Locally Constrained Policy Optimization (LCPO)**, an on-policy method that uses past samples only to **anchor** policy outputs on out-of-distribution old contexts via a KL constraint, rather than replaying them for off-policy optimization. Empirically, LCPO is evaluated across modified Gymnasium control tasks and a real systems benchmark, and generally outperforms the authors’ online baselines while approaching a prescient offline upper bound.

## Strengths
- **The paper targets a meaningful and fairly specific setting that many prior continual/non-stationary RL methods do not handle cleanly:** observed, exogenous, potentially smooth and non-piecewise-stationary context processes. The paper is explicit about this scope in §2 and contrasts it against task-label / CPD assumptions in §1 and §4.1.
- **The core idea is specific and conceptually clean:** use old data to *constrain policy drift* on context-distant samples rather than to optimize on stale returns. This is more than generic replay; Eq. (1) and §4.2 define a targeted KL anchoring objective over \(W(B_a, B_r)\), which is a sensible way to try to preserve behavior on prior contexts while remaining on-policy on current data.
- **The gridworld in §4.1 is genuinely informative, not just illustrative.** It clearly demonstrates the intended mechanism: standard A2C forgets old-context behavior, tabular A2C avoids this because updates are localized by state-context row, and LCPO partially recreates this locality with function approximation.
- **The evaluation spans both simulated control and a real systems task with production traces.** In particular, the straggler mitigation environment with Microsoft production workloads is a useful demonstration that the method is not only tested on toy drift processes.
- **Two practical ablations are valuable:** sensitivity to OOD threshold (§5.2) and to buffer size (§5.3). The finding that performance remains strong even with relatively small buffers is practically useful, and the paper appropriately notes the likely dependence on context complexity.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper’s central catastrophic-forgetting claim is only directly demonstrated on the toy gridworld, not on the main benchmarks.**  
  On the major experiments, the paper reports lifelong/normalized return (§5.1, Fig. 3, Table 1), which mixes together forgetting, adaptation speed, exploration quality, and optimization stability. That is enough to show LCPO is a strong method in the tested setup, but not enough to isolate *catastrophic forgetting mitigation* as the reason. Since the paper’s framing throughout the abstract/introduction is specifically about combating CF, the lack of direct forgetting diagnostics on the main tasks is a substantial evidential gap.

- **The headline claim that LCPO is “on-par” with a prescient offline agent is overstated relative to the presented evidence.**  
  The main text supports a weaker statement: LCPO is *the closest online method* to the prescient baseline. That is visible in Fig. 3a and stated in §5.1 (“LCPO maintains a lead over baselines, is close to the best-performing prescient policy”). But “on-par” in the abstract is too strong, especially since Table 1 still shows noticeable gaps to prescient on the systems task. This is a claim-calibration issue rather than a method flaw, but it matters because it overstates the empirical conclusion.

- **The evaluation setup substantially narrows the paper’s scope, especially the move to discrete-action versions of control benchmarks.**  
  The paper states: “Gym environments were modified to accept discrete action space policies, as even prescient policies struggled to learn stable continuous space policies in the presence of contexts” (§5). This is an important limitation. Since several benchmark domains are originally continuous control tasks, the results do not establish LCPO in standard continuous-action online RL settings. This is not a fatal flaw given the paper’s actual experiments, but it materially limits significance and external validity.

- **The method depends critically on having a meaningful context-space OOD metric, and the paper likely understates how central this requirement is.**  
  The paper argues that LCPO “only requires an OOD detector,” but in practice its success hinges on access to context representations where simple distances are meaningful: L2 distance on wind vectors in Gym and Mahalanobis distance on workload/context features in systems (§5, §5.2). The paper does ablate threshold choice, which is good, but detector quality is clearly a core ingredient rather than an incidental implementation detail. This weakens the broad practical claim that OOD is simply an easy drop-in substitute for task labels.

- **The comparisons support “works well in this benchmark suite,” but not as strongly the broader class-level conclusions drawn about alternative approaches.**  
  For example, §5.1 repeatedly interprets weak baseline performance as evidence that off-policy RL, CPD, or rehearsal methods are broadly brittle. Some of that is plausible, but the paper’s own setup—especially discretized control, a specific online protocol, and tuned OOD-aware LCPO—does not fully justify broad conclusions about entire method families. The empirical claim that LCPO outperforms the tested baselines is supported; the broader methodological generalizations should be toned down.

### Minor
- **The paper does not sufficiently isolate which part of the LCPO package is doing the work.**  
  The method combines OOD-selected anchor samples, a KL constraint, conjugate-gradient/TRPO-style optimization, and a line-search step that also enforces recent-batch stability (§4.2). The current ablations on threshold and buffer size are useful, but they do not disentangle whether the gains come primarily from OOD-based anchoring, from the trust-region-like stabilization, or from the combination.

- **Normalized return CDFs are useful summaries, but they obscure absolute gaps and make the main evidence harder to interpret.**  
  The paper defines normalized return per environment/trace so that 0/1 correspond to the minimum/maximum among agents (§5.1). This is acceptable as a compact aggregate measure, especially since the appendix reportedly contains fuller tables, but in the main paper it weakens interpretability of the magnitude of improvement and the true distance to prescient performance.

- **The paper’s practical scope is narrower than the introduction may suggest because it requires observed exogenous context.**  
  This is actually acknowledged in §2, §3, and the conclusion, so it is not a hidden flaw. Still, it is an important limitation: LCPO does not address latent or action-dependent non-stationarity.

- **Computational cost is only lightly characterized.**  
  The paper reports that LCPO is about \(1.5\times\) as demanding as A2C and provides wall-clock results in the appendix (§5.1), which is helpful. But because the method uses conjugate gradient, OOD sampling, and line search, a more explicit breakdown would make practical deployment tradeoffs clearer.

### Trivial
- **A more direct main-paper discussion of when the anchor constraint is active would help interpretation.**  
  The current text explains how LCPO skips the constraint if insufficient OOD samples are found (§4.2), but does not summarize how often this happens across environments.

## Nice-to-Haves
- Add direct forgetting metrics on the main benchmarks, e.g., periodic evaluation on previously seen context bands or backward-transfer/retention curves.
- Add at least one ablation comparing OOD-selected anchors against random old samples or all-buffer anchors, to verify that “local” anchoring is the key ingredient.
- Present absolute returns more prominently in the main paper alongside normalized summaries.
- Clarify the practical recipe for selecting the OOD threshold in a new domain, beyond the current sensitivity plots.
- If feasible in future versions, test a continuous-action variant; if not, discuss more concretely what blocks such an extension.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“No PPO baseline.”**  
  Removed. While PPO would be a natural extra comparator, its absence alone is not a substantive flaw given the already broad baseline suite including A2C, TRPO, SAC, DDQN, MBPO, CLEAR, PT-DQN, EWC, OGD, and MBCD. This is more of a benchmark wish list item than a core weakness.

- **“Lack of theoretical guarantees for the neural-network setting.”**  
  Soft-removed / weakened. For this paper’s empirical RL setting, the lack of a nontrivial function-approximation theorem is not a decisive flaw. The paper is primarily an algorithmic/empirical submission, and demanding a full guarantee here is outside normal standards. It would strengthen the work, but its absence should not be treated as a major deficiency.

- **“Warm-up period of 6M steps undermines online learning.”**  
  Removed as stated. The paper does include a warm-up period in the benchmark protocol (§5), but this is part of the experimental setup applied across methods, not evidence that the work is “not online RL.” A sensitivity study could be useful, but the criticism as framed is overstated.

- **Generic complaints that the baselines are unstable / sensitive because the paper says so.**  
  Removed in that form. The valid point is narrower: the paper should avoid broad class-level conclusions from this setup. It would be wrong to claim the baseline results are invalid simply because some methods are sensitive.

- **Any suggestion that cited benchmarks/models/tools may be unavailable or unverifiable.**  
  Removed per instruction.

## Novel Insights
The paper’s most compelling contribution is not merely “on-policy RL with replay-like machinery,” but a sharper design principle: in non-stationary contextual RL, stale data may be more valuable as **behavioral anchors** than as direct optimization targets. That is a useful conceptual reframing. However, the current evidence only partially closes the loop: the toy example strongly supports this mechanism, while the main benchmarks mostly show that the full LCPO package is effective. The most important next step is therefore not another benchmark, but demonstrating on the main tasks that old-context competence is indeed being retained rather than simply relearned more efficiently.

## Suggestions
- Replace “on-par with a prescient agent” with a more precise claim such as “closest among online methods to the prescient upper bound.”
- Add direct retention/forgetting evaluations on the main benchmarks to substantiate the paper’s core CF framing.
- Add an ablation for random-anchor or all-buffer-anchor constraints to verify that OOD-locality matters.
- Make the scope limitation around observed exogenous context and context-metric availability more explicit in the introduction/abstract.
- Include more prominent absolute-return plots or tables in the main paper, not only normalized summaries.
- Expand the discussion of the discrete-action benchmark modification and what would be required to extend LCPO to continuous-action settings.

## Score and Decision
**Novelty:** Moderate. The ingredients are familiar, but the way they are combined for online non-stationary context-driven RL is clean and meaningful.  
**Technical soundness:** Reasonably sound as an empirical algorithm paper, but the causal claim about catastrophic forgetting is not fully established on the main tasks.  
**Empirical support:** Good breadth, but not fully diagnostic of the central mechanism; some claims are stronger than the evidence.  
**Significance:** Moderate. The problem is important and the method looks useful, but the restricted setting (observed exogenous contexts, simple OOD metrics, discrete-action experiments) limits broader impact.  
**Clarity:** Good overall, with a particularly effective motivating example and a solid limitations section.

Relative to calibration examples, this looks stronger than typical clear rejects that lack convincing experiments, but weaker than accepted continual/non-stationary RL papers that either establish their mechanism more directly or have broader/cleaner empirical support. I view it as **borderline reject**: promising idea, useful results, but still too much claim inflation and not enough direct evidence for the central CF mechanism for ICLR’s bar.

MY FINAL SCORE: <pineapple>5.9</pineapple>
MY FINAL DECISION: <orange>Reject</orange>