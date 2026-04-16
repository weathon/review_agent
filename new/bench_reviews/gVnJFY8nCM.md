The paper mentions: "Also, we conduct experiments upon the same MuJoCo environments with IL prior policies, whose results are summarized in Table 10 in Appendix F.1."

Let me now check the dynamics training data requirements more carefully. The paper says:
- MuJoCo: dynamics training uses "a substantially smaller dataset" (2K steps)
- GTS: "~2,000 laps" for dynamics training, "~100 laps" for fine-tuning

The paper also says in the abstract: "customize a given prior policy on new performance metrics in few-shot and even zero-shot online settings, given access to the prior action distribution alone"

This is somewhat misleading since "zero-shot" still requires a pre-trained dynamics model. But the paper does discuss this in Section 3.2.

Let me now evaluate the harsh critic's points systematically:

**Point 1 (Max-entropy prior assumption)**: This is a legitimate concern. The RQL framework the paper builds on does assume the prior is a maximum-entropy optimal policy. GT Sophy 1.0's training details are unknown. However, the paper uses log π as a proxy in practice and shows it works empirically. The theoretical guarantee may not hold exactly, but the practical effectiveness is demonstrated. This is a valid concern but not fatal — the method works empirically even if the assumption is not perfectly satisfied.

**Point 2 (Unfair comparison with baselines)**: The paper explicitly notes that Full-MPPI and Guided-MPPI have privileged information. The claim that Residual-MPPI outperforms them despite having less information is meaningful. The failure of these baselines in GTS does highlight that having full reward information but no good action prior is insufficient, which makes the Residual-MPPI contribution more notable. However, the concern about Guided-MPPI being underspecified and the failure of baselines in GTS is partially valid — it's hard to draw strong conclusions from non-functional baselines.

**Point 3 (Zero-shot/few-shot framing)**: This is a valid concern. "Zero-shot" in the paper means zero additional policy training, but still requires a trained dynamics model. The abstract says "given access to the prior action distribution alone" which is technically incomplete since a dynamics model is also needed. However, the paper does discuss dynamics learning explicitly, so this is more of a framing/terminology issue than a fundamental one.

**Point 4 (Uncertainty/variability)**: Standard deviations are reported. The paper does have appendix ablations on planning parameters. The concern about statistical significance is somewhat generic.

Now let me compose the final review.</think>Now I have a comprehensive understanding of the paper and the reviewer perspectives. Let me compile the final review.

## Summary

Residual-MPPI proposes integrating the Residual Q-Learning (RQL) framework into Model Predictive Path Integral (MPPI) control to enable online policy customization for continuous control. Given a prior policy π and an add-on reward r_R, the method uses log π as a surrogate for the unknown original reward, enabling online planning that balances prior task performance with new objectives—without access to the original reward function or critic. Experiments span MuJoCo benchmarks (with SAC and IL priors) and the Gran Turismo Sport simulator, where the method successfully customizes the champion-level GT Sophy 1.0 agent to reduce off-course driving.

## Strengths

- **Well-motivated and practically relevant problem formulation.** Online policy customization—adapting a deployed policy to new objectives without retraining—is a genuinely useful capability. The RQL framework provides a principled theoretical basis, and extending it from discrete actions (Residual-MCTS) to continuous actions via MPPI fills a clear gap. The key insight that log π encodes original task information in a theoretically motivated way is elegant.

- **Demonstrated scalability on a challenging, real-world-grade domain.** Customizing GT Sophy 1.0—a champion-level racing agent—to reduce off-course steps from ~93 to ~4-9 while preserving competitive lap times is a compelling result. The GTS environment is a realistic, high-fidelity simulator, making this a significant step beyond standard MuJoCo benchmarks.

- **Comprehensive ablation structure in MuJoCo.** The six MPPI variants (Prior, Greedy, Full, Guided, Valued, Residual) plus two RL baselines effectively isolate the contribution of each design choice. The finding that log π carries long-horizon information beyond simple regularization is well-supported by the Guided-MPPI vs. Residual-MPPI comparison, particularly in the Ant environment where orthogonal reward structures expose Greedy-MPPI's limitations.

- **Significant data efficiency over RL-based customization.** Residual-MPPI requires ~2K environment steps (MuJoCo) or ~2K laps (GTS) for dynamics training, compared to 4M steps or 80K laps for Residual-SAC to achieve comparable or worse performance. This is a practical advantage for deployment scenarios.

## Weaknesses

### Major:

- **The core theoretical assumption is unverified in the most important experimental setting.** The RQL equivalence—replacing ωr with ω′log π in the augmented MDP—holds only if the prior policy is an optimal maximum-entropy policy for some underlying reward. For the MuJoCo priors (SAC-trained), this assumption is approximately satisfied. For GT Sophy 1.0, the paper has no access to its training configuration, reward structure, or temperature, making it unlikely that GT Sophy satisfies this condition. The method still works empirically in GTS, which suggests that the approach is robust to violations of this assumption, but the paper frames the result as if the theoretical guarantee holds. This gap between theory and practice should be addressed honestly: the paper should characterize Residual-MPPI as a heuristic that is *inspired* by the RQL framework when applied to priors that don't satisfy the maximum-entropy optimality assumption, rather than presenting it as a direct instantiation with formal guarantees.

- **GTS evaluation has limited viable baselines, weakening the strongest empirical claim.** In the most important experiment (Table 2), Full-MPPI, Guided-MPPI, and Greedy-MPPI all fail to complete a single lap, and Valued-MPPI is infeasible. This leaves Residual-MPPI without a meaningful planning-based comparison in GTS—the domain where the paper's claims are most bold. While the failure of baselines is itself informative (showing that naive MPPI variants struggle in complex domains), it also means the GTS results primarily demonstrate that Residual-MPPI *can* work, not that it is *better* than carefully tuned alternatives. The paper should acknowledge this limitation and discuss what additional baselines could be constructed for GTS (e.g., trajectory optimization with the prior as a warm-start, or constrained MPPI formulations).

- **Dynamics model dependence is under-quantified, creating tension with the "zero-shot" framing.** The abstract claims the method works "given access to the prior action distribution alone," but all experiments require a pre-trained dynamics model trained on thousands of environment transitions. "Zero-shot" means zero additional policy training, but the method is not zero-shot in terms of environment interaction. The paper should be transparent about this: the method requires (1) a prior policy and (2) a dynamics model. The dynamics model quality directly determines planning performance (Section 3.2 acknowledges compounding errors), but no experiment varies dynamics model accuracy or training data budget, leaving the reader unable to assess how robust the method is to imperfect dynamics.

### Minor:

- **Modest differentiation from Greedy-MPPI on simpler tasks.** In HalfCheetah (total reward 1936.2 vs. 1939.9) and Hopper (7363.0 vs. 7367.0), Residual-MPPI and Greedy-MPPI perform nearly identically. The advantage of the log π term primarily manifests in tasks where the add-on and basic rewards are orthogonal (Ant) or in complex domains (GTS). This limits the generality of the claim that the log π term is always beneficial.

- **No computational cost analysis.** MPPI requires K forward rollouts per timestep, and Residual-MPPI additionally evaluates log π under the prior. No wall-clock timing or real-time feasibility analysis is provided, which matters for the "online" claim—especially in the latency-sensitive GTS environment.

- **Limited sensitivity analysis for ω′ in the main text.** The key trade-off hyperparameter between prior retention and add-on optimization is not analyzed in the main paper (deferred to Appendix F). Given that ω′ is the primary tuning knob for customization, understanding its sensitivity is central to practical adoption.

- **Proposition 1's conditions are loosened without analysis.** Proposition 1 requires γ=1 and infinite noise variance, but the algorithm uses γ<1 and finite Σ. The paper acknowledges this is "heuristic" but does not analyze how performance degrades as these conditions are violated—a relevant concern since γ directly affects the discounting of the log π term in the scoring function.

### Trivial:

- None worth listing.

## Nice-to-Haves

- Analysis of dynamics model prediction error over the planning horizon and its correlation with planning performance degradation.
- Experiments with sparse or delayed add-on rewards beyond the dense reward settings tested.
- A trajectory-level visualization showing how behavior smoothly transitions as ω′ varies from preserving the prior to optimizing the add-on task.
- Real-time timing results for the GTS deployment to validate practical feasibility.

## Removed Points

- **"Comparison with Full-MPPI and Guided-MPPI is structurally unfair."** The harsh critic argues these baselines are weak comparisons. However, the paper explicitly designed these baselines to have *more* information (full reward, value function) than Residual-MPPI, making the comparison conservative and favorable to the baselines. If Residual-MPPI outperforms methods with privileged access, that is a valid finding. The asymmetry favors the baselines, not the proposed method.

- **"Zero-shot/few-shot claims are overstated."** While the terminology could be clearer, the paper does discuss dynamics learning explicitly in Section 3.2 and distinguishes "zero-shot" (no online dynamics fine-tuning) from "few-shot" (with fine-tuning). The data efficiency advantage (2K vs 80K laps in GTS) is a real practical contribution regardless of labeling.

- **"GT Sophy is proprietary/unknown training."** The paper explicitly acknowledges this limitation and treats it as a strength—showing that the method works even without access to the reward or critic. This is the core use case for policy customization.

- **"No confidence intervals or hypothesis tests."** Single-run evaluation with means ± std is the norm in this community (all referenced calibration papers use the same). Requesting additional statistical rigor beyond community standards is scope creep.

## Novel Insights

The most interesting insight, supported by empirical evidence but underexplored theoretically, is that log π functions not just as a regularizer keeping actions near the prior, but as a *proxy for long-horizon value* beyond the planning horizon. The comparison between Residual-MPPI and Guided-MPPI in Ant (where Guided-MPPI has full reward access but still underperforms) suggests that the prior policy's action distribution encodes information about the original task that a finite-horizon planner with full reward information cannot capture—because the planner's horizon is too short to reason about long-term consequences. This finding has implications beyond policy customization: it suggests that using pre-trained policy distributions as action priors in model-based RL/planning can convey more useful information than simply conditioning on expert actions, particularly for tasks requiring long-horizon reasoning.

## Suggestions

- Reframe the theoretical claims honestly: when the prior doesn't satisfy the maximum-entropy optimality assumption exactly (as in GT Sophy), Residual-MPPI should be described as a principled heuristic inspired by RQL rather than a theoretically guaranteed solution. An empirical analysis of how robust the method is to violations of this assumption would strengthen the paper significantly.
- Add dynamics model quality ablations (vary training data size, measure rollout error vs. planning performance) to quantify the method's dependence on model accuracy.
- Include ω′ sensitivity analysis in the main text—this is the most important hyperparameter and deserves upfront discussion.
- Report wall-clock timing per planning step for each environment to substantiate the online feasibility claim.

## Score and Decision

**Calibration comparison:**

- **H-GAP** (Accept Spotlight, scores 8/8/6): Novel trajectory model + MPC planning for humanoid control with zero-shot transfer. Strong experimental results, clear methodology. More novel architecture contribution than Residual-MPPI.
- **BMPC** (Accept Poster, scores 6/8/8/6): Systematic improvement of TD-MPC2 via expert imitation. Thorough evaluation across 28 environments. Engineering-focused but solid. Comparable level of novelty to Residual-MPPI—both combine existing frameworks in a useful way.
- **M³PC** (Accept Poster, scores 6/8/8/6): Test-time MPC with pretrained trajectory model. Similar "planning with a pre-trained model" theme, stronger theoretical grounding.
- **Q-Adapter** (Accept Poster, scores 6/6/6/6): Applies RQL framework to LLM customization—directly comparable motivation (policy customization via residual framework), but in a different domain.
- **L-MBOP-E** (Reject, scores 3/3/5/3): Latent model-based offline planning. Similar model-based planning contribution but with fundamental methodological issues (unclear extrinsic policy training, insufficient baselines). Residual-MPPI is clearly stronger.
- **Multi-Objective Preference Control** (Reject, scores 5/6/6/6): Multi-objective RL with preference conditioning. Overclaims relative to evidence, unclear methodology. Residual-MPPI has stronger empirical grounding.

Residual-MPPI presents a meaningful algorithmic contribution (integrating RQL with MPPI for continuous control) with compelling results on a challenging domain (GT Sophy). Its weaknesses—unverified theoretical assumptions in the main showcase, limited viable baselines in GTS, and under-quantified dynamics dependence—are significant but do not undermine the core contribution. The method demonstrably works on a real-world-grade problem and offers genuine practical value. Compared to accepted papers at similar score levels (BMPC, M³PC, Q-Adapter), Residual-MPPI has comparable novelty and strong empirical results on a harder domain, but weaker theoretical grounding and more caveats around its flagship experiment. It falls below the H-GAP and M³PC level due to the theoretical gap and limited GTS baselines, but above the rejected L-MBOP-E and Multi-Objective Preference Control.

**Score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>