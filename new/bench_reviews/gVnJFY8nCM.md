Now I have enough context. Let me carefully analyze the paper and reviewers' claims to produce my consolidated review.

Key things to verify from the paper:

1. **Structural mismatch claim (Harsh Critic #1):** The critic claims the paper doesn't justify replacing r with ω' log π + r_R. Let me check: The paper says in Sec. 2.1 that "finding the maximum-entropy policy solving the full MDP problem [...] is equivalent to solving an augmented MDP problem M^aug = (X, U, ω' log π(u|x) + r_R, p)." This is from Li et al. (2024). The paper then says in Sec 3.1 that Proposition 1 establishes MPPI can approximate the max-entropy optimal policy, and then "we can then derive Residual-MPPI straightforwardly by defining the evaluation function S^aug_{x0}(U) in MPPI as [Eq 6]" which uses ω' log π + r_R. So the logic is: (1) RQL shows the combined task is equivalent to an augmented MDP with reward ω' log π + r_R, and (2) MPPI can solve this augmented MDP. The RQL equivalence relies on π being an optimal max-entropy policy. The harsh critic is right that this assumption may not hold exactly in practice (learned SAC policies), but the paper does acknowledge this indirectly through Proposition 1's conditions. However, this is a legitimate concern about the gap between theory and practice.

2. **Unfair baselines claim (Harsh Critic #2):** The critic claims Full-MPPI and Guided-MPPI are set up to fail. However, the paper explicitly states: "except for Greedy-MPPI, the remaining MPPI baselines have access to the underlying reward or value function of the prior policy. These baselines show that Residual-MPPI is still the ideal choice, even with privileged access to additional reward or value information." This is a valid experimental design: giving baselines MORE information and still showing Residual-MPPI is competitive. Per my instructions, I should REMOVE criticisms about unfair comparison where the asymmetry favors the baseline. The Full-MPPI having no prior policy structure is actually how it would work - the point is that having the prior policy's log-likelihood is more useful than having the prior reward directly. The critic is wrong that this is "unfair" to the baselines - rather, it's actually giving the baselines MORE resources (true reward) and showing Residual-MPPI still does well.

However, the concern that these baselines might not be well-tuned is partially valid. In GTS, Full-MPPI, Guided-MPPI, and Greedy-MPPI all "Failed" to complete a lap - the paper doesn't explain tuning details. But the design of giving baselines privileged information is fine.

3. **Dynamics model validation (Harsh Critic #3):** This is a valid concern - no quantitative dynamics model evaluation is provided.

4. **Greedy-MPPI as strawman (Harsh Critic #2, subpoint 3):** Per the instructions, this is actually a useful ablation showing the importance of the log π term. The critic calls it a "straw-man that deliberately contradicts the problem definition," but Greedy-MPPI serves as a clear ablation to demonstrate what happens without log π. This is a legitimate ablation, not an unfair comparison.

5. **ω' sensitivity analysis (Neutral #5, Spark):** Valid concern - no systematic sweep of ω' is provided.

6. **Computational cost (Neutral #3, Spark):** Valid concern.

7. **Restrictive assumptions (Neutral #1):** γ=1 and infinite variance are acknowledged in the paper. Valid concern but the paper is aware.

8. **Similar results to Greedy-MPPI in some tasks (Neutral #2):** Valid observation.

Now let me think about the overall assessment.

The paper makes a meaningful contribution: extending policy customization from discrete (RQL+MCTS) to continuous control (RQL+MPPI). The GTS application with GT Sophy 1.0 is impressive. The key theoretical concern is that using log π as a surrogate for the unknown prior reward relies on the prior being an optimal maximum-entropy policy, which is idealized. However, this is the same limitation shared by RQL itself (Li et al., 2024), and the paper acknowledges limitations. The empirical results, while with some limitations in baseline tuning (especially in GTS), do show that the method works effectively in practice.

Comparing to calibration papers:
- BMPC (Accept Poster, scores 6,8,8,6): Combining MPPI with policy learning, incremental novelty, good results on DMControl. Similar profile to Residual-MPPI.
- MPC² (Accept Poster, scores 5,6,8,6): Hierarchical MPPI for high-dimensional control, incremental engineering contribution but works well.
- M³PC (Accept Poster, scores 6,8,8,6): Test-time MPC with pretrained trajectory model, similar space.
- Policy Decorator (Accept Poster, scores 8,5,8): Online refinement of policies with residual learning, very similar problem space.
- DLPA (Reject, scores 5,5,5,10,6): Applying existing methods to PAMDP with limited novelty.
- Planner-Guided Policy Optimization (Reject, scores 6,3,1,6): MPPI+RL for off-road driving, similar theme but weaker.

The paper has a meaningful contribution (extending policy customization to continuous control online, demonstrated on GT Sophy 1.0), reasonable empirical results, and a clear theoretical motivation. The main weaknesses are: (1) gap between theoretical assumptions and practice, (2) limited dynamics model validation, (3) limited sensitivity analysis for ω', (4) some baseline comparisons could be stronger. These are similar to weaknesses in accepted papers like BMPC and MPC². The paper seems comparable to the lower end of accepted papers (like MPC² at scores 5,6,8,6) or around the borderline.

I'd place this around 5.5-6, which is borderline. The key question is whether the theory-practice gap and the limited novelty (combining two existing frameworks) are too significant. I think the practical contribution (demonstrating on GT Sophy 1.0) and the clean formulation of continuous policy customization are sufficient for a borderline accept, but the theoretical concerns keep it from being clearly above the bar.

Actually, reconsidering more carefully: the harsh critic's concern about the theoretical gap is quite significant. The paper claims that Residual-MPPI "solves the augmented MDP" and "approximates the optimal customized policy online," but this claim relies on π being an optimal max-entropy policy. In practice, learned SAC policies are not exactly optimal, and the paper provides no analysis of how suboptimality affects performance. Moreover, in GTS, they use GT Sophy 1.0 which was trained with complex rewards and regulations - it's unclear how well its policy log-likelihood serves as a proxy for the original task reward. This does undermine the central theoretical narrative.

On the other hand, the empirical results are strong, and the GTS demonstration is noteworthy. The method clearly works in practice, even if the theoretical justification is incomplete.

Overall, I think this is around a 5.5-6. The paper has real contributions (problem formulation, practical algorithm, strong application) but also real weaknesses (theory-practice gap, limited ablations, some empirical questions). Compared to borderline-accepted papers like MPC², this has a similar profile.

Wait, let me reconsider the unfair baseline criticism more carefully. The instructions say: "REMOVE criticisms that are factually wrong or misunderstand the paper." The harsh critic claims Full-MPPI is not given prior policy structure, but the paper explicitly says Full-MPPI uses "no prior on the MDP of the full task" - this is by design, it's a different method (standard MPPI on the full reward). Giving Full-MPPI access to the prior policy would make it Guided-MPPI, which is already a separate baseline. So the comparison structure is actually well-designed: Full-MPPI (standard MPPI), Guided-MPPI (MPPI with prior sampling + full reward), Valued-MPPI (Guided-MPPI + value function), vs Residual-MPPI (MPPI with prior sampling + log π reward). Each baseline adds or removes components to show what matters. The comparison is actually quite systematic.

The critic's claim that "Full-MPPI fails catastrophically because it's badly tuned" may be partially true, but it's also a well-known fact in the MPC literature that standard MPPI without a good initialization/prior often fails in high-dimensional continuous control. This is actually one of the core motivations for using a prior policy. So Full-MPPI failing is expected and informative, not just "badly tuned."

Similarly, Greedy-MPPI is a valid ablation. It's not a "straw-man" - it's what you'd get if you naively tried to do policy customization by just optimizing the add-on reward while sampling from the prior. The paper shows this fails in GTS, which is an important finding.

So the "unfair baselines" criticism from the harsh reviewer is largely invalid per my instructions (remove unfair comparison criticisms when the asymmetry favors the baseline), and the Greedy-MPPI ablation is legitimate.

Let me finalize my assessment. Given:
- Real contribution: extending policy customization to continuous control online planning
- Strong practical demonstration on GT Sophy 1.0
- Clean formulation connecting RQL to MPPI
- But: theoretical gap between idealized assumptions and practice
- Limited ω' sensitivity analysis
- Limited dynamics model validation
- No computational cost analysis
- Incremental algorithmic novelty (combining two existing frameworks)

Score: 5.5 (borderline, leaning toward weak accept given the practical value and GT Sophy demonstration, but the theoretical gap and limited novelty hold it back)

Actually, let me reconsider. The theoretical gap is significant but the paper never claims to have a complete theoretical guarantee - it acknowledges limitations. And the practical results are compelling. The novelty is moderate but the application is impactful. Comparing to similar papers:
- BMPC (6.25 avg): Stronger novelty, but in a more incremental space
- MPC² (6.25 avg): Similar incremental engineering, accepted
- M³PC (7.0 avg): More novel formulation, accepted
- Policy Decorator (7.0 avg): Similar problem (online refinement), simpler method, accepted
- DLPA (5.2 avg): Rejected, limited novelty

I think this paper falls around 5.5-6, comparable to DLPA but with a more impressive application (GTS) and clearer motivation. It's at the borderline.

Let me settle on 5.5 with a weak accept recommendation, but I could see this going either way.</think>Now let me produce my final consolidated review based on careful analysis of the paper and all reviewer inputs.

## Summary

The paper proposes Residual-MPPI, an online planning algorithm that extends the Model Predictive Path Integral (MPPI) framework to customize pre-trained continuous-control policies at execution time. By incorporating the log-likelihood of a prior policy as a reward surrogate (derived from the Residual Q-learning framework of Li et al., 2024) alongside an add-on reward, the method achieves zero-shot and few-shot policy customization without requiring access to the original training reward or value function. Experiments on MuJoCo benchmarks and on customizing the champion-level GT Sophy 1.0 racing agent in Gran Turismo Sport demonstrate the approach's effectiveness.

## Strengths

- **Well-motivated and practically important problem**: Online policy customization—adapting a pre-trained policy to new requirements at execution time without retraining or accessing the original reward—is clearly defined and has significant practical implications for robotics deployment (Sec. 1).

- **Clean integration of RQL with MPPI**: The derivation connecting maximum-entropy policies to MPPI's optimal distribution via Proposition 1 provides a principled foundation for using log π as a reward surrogate. The algorithm naturally leverages the prior policy both for sampling initialization and for encoding original task information.

- **Comprehensive baseline comparisons in MuJoCo**: Seven baselines including MPPI variants with privileged information and RL-based approaches make the comparison informative. The design of giving Full-MPPI, Guided-MPPI, and Valued-MPPI access to the true rewards or value function—more information than Residual-MPPI has—makes it a strong experimental design that strengthens rather than weakens the claims.

- **Compelling GTS application**: Customizing GT Sophy 1.0—a champion-level racing agent—is a strong demonstration of scalability and practical relevance. The few-shot variant reducing off-course steps from 93 to 4.4 with only ~2k laps of dynamics data, versus Residual-SAC needing 80k laps for a much more conservative policy, is a striking efficiency result (Table 2).

- **Clear ablation on the log π term**: Greedy-MPPI (which removes log π) serves as a meaningful ablation that demonstrates the necessity of this component. Its failure in GTS—where the agent simply stops driving to avoid off-course penalties—confirms that the log π term is not just regularization but a theoretically grounded component of the joint optimization (Sec. 5.2, Fig. 3).

## Weaknesses

### Fatal

None.

### Major

- **Gap between theoretical assumptions and practical deployment**: The core theoretical result (Proposition 1 and the derivation of S^aug) requires the prior policy π to be an optimal maximum-entropy policy for the original MDP with γ=1 and infinite noise variance (uniform prior). In practice, priors are standard SAC policies with finite training and unknown suboptimality, and MPPI uses γ<1 with finite covariance Σ. The paper acknowledges γ=1 and infinite variance as approximations (Sec. 3.1) but provides no theoretical or empirical analysis of how deviations from these idealized conditions affect the quality of policy customization. Most critically, there is no analysis of how suboptimality in the prior (e.g., GT Sophy 1.0, which was trained with complex multi-objective rewards and regulations) propagates through the log π term. If log π does not faithfully encode the original task reward, the method may trade off the basic task in uncontrolled ways. This does not invalidate the approach—heuristically, a well-trained prior's log π often serves as a reasonable proxy—but it means the paper's claim to solve the "policy customization" problem in the formal sense of Li et al. (2024) is not fully established.

- **No quantitative dynamics model evaluation**: The method's practical viability depends heavily on having an accurate learned dynamics model in the state-action regions visited by the customized policy. Despite introducing multi-step error training, exploration noise, and online fine-tuning, the paper provides no quantitative assessment of dynamics model accuracy (no prediction error tables, no comparison of zero-shot vs. few-shot dynamics quality). This is a significant omission for an MPC-based method where model errors compound over the planning horizon.

### Minor

- **Limited sensitivity analysis for ω'**: The weight ω' is the primary hyperparameter controlling the trade-off between retaining the prior behavior and optimizing the add-on reward. No systematic ablation or Pareto frontier varying ω' is presented, making it unclear how sensitive results are to this choice or how a practitioner should set it.

- **No computational cost analysis**: MPPI requires evaluating K sampled trajectories over horizon T at every time step, each requiring both a dynamics rollout and a log π query. The paper claims "online" customization but provides no wall-clock timing, making it difficult to assess real-time feasibility, especially for the GTS environment.

- **Similarity to Greedy-MPPI in some MuJoCo tasks**: In HalfCheetah, Swimmer, and Hopper, Residual-MPPI and Greedy-MPPI achieve nearly identical total rewards (Table 1). The advantage of Residual-MPPI primarily manifests in Ant (where add-on and basic rewards are orthogonal) and GTS (where the add-on reward is sparse/constraining). Characterizing when log π provides critical advantages versus when simple sampling from the prior is sufficient would strengthen understanding of the method's operating regime.

- **No failure case analysis**: The paper only shows successful customization. Understanding when the method fails—e.g., when the add-on reward strongly conflicts with the basic task, when the dynamics model is poor, or when the prior has low entropy—would clarify its practical limits.

### Trivial

- Algorithm 1, line 15 appears to compute unnormalized weights (dividing by λ rather than by η), which may be a typesetting inconsistency with Eq. (3)–(4) but does not affect correctness if the subsequent normalization in line 18 is applied.

## Nice-to-Haves

- Systematic variation of ω' with a Pareto frontier between basic and add-on reward, which would demonstrate controllability of the customization trade-off.
- Quantitative dynamics model error analysis (prediction error vs. horizon, zero-shot vs. few-shot comparison).
- Analysis of how prior policy quality/suboptimality affects customization performance (e.g., using priors at different training checkpoints).
- Comparison with alternative online model-based planners (CEM, PETS) modified with the same RQL objective.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Unfair baseline comparisons (Harsh Critic #2)**: The reviewer argues that Full-MPPI and Guided-MPPI are "set up to fail" and used as evidence of Residual-MPPI's superiority. However, per the paper's own design, these baselines are given *more* information (true reward, value function) than Residual-MPPI. If they fail despite having privileged access, that is a valid comparison that strengthens Residual-MPPI's case. The instructions say to remove criticisms about unfair comparisons when the asymmetry favors the baseline. Additionally, Full-MPPI failing without a prior initialization is a well-known issue in the MPC literature and is one of the core motivations for using a prior.

- **Greedy-MPPI as a "strawman" (Harsh Critic #2, subpoint 3)**: The reviewer calls Greedy-MPPI a "straw-man that deliberately contradicts the problem definition." In fact, Greedy-MPPI is a natural and informative ablation: it shows what happens when you use prior sampling *without* the log π term. Its failure in GTS (where the agent simply stops to satisfy the on-course constraint) is a substantive finding that validates the role of log π. This is standard ablation design, not a strawman.

- **Reproducibility concern about GTS (Human Finder #6)**: The reviewer notes that GTS is a proprietary environment. Per the instructions, we do not flag cited environments as unavailable or unverifiable.

- **Missing related works (Spark, Neutral)**: The instructions say not to mention missing related works since we cannot verify their existence or relevance.

- **Formatting and notation nitpicks (Harsh Critic)**: Minor notation inconsistencies are removed per the instructions on formatting/style nitpicks.

## Novel Insights

The key insight that bridges the two reviewer perspectives is that Residual-MPPI's practical efficacy may be more attributable to the heuristic virtue of using a well-trained policy's log-likelihood as a "soft constraint" during sampling-based planning than to the formal policy customization guarantee claimed under idealized conditions. The paper frames this as solving an augmented MDP, but the practical success likely stems from log π acting as a powerful implicit prior over good action sequences—encoding not just reward information but also the dynamics-exploiting structure learned during prior training. This distinction matters because it reframes the contribution from "theoretically grounded policy customization" to "practically effective prior-guided MPPI," which is still valuable but makes different claims. The GTS results, where the prior policy is complex and potentially suboptimal (aggressive racing with off-course exploits), actually provide indirect evidence that the method works beyond the idealized maximum-entropy setting—though the paper misses the opportunity to explicitly demonstrate this through controlled experiments with impaired priors.

## Suggestions

1. **Add an ablation varying ω'** across multiple values (e.g., ω' ∈ {0.1, 0.5, 1.0, 2.0, 5.0}) and plot the resulting Pareto frontier between basic and add-on reward. This would both validate the method's controllability and provide practical guidance for ω' selection.

2. **Report dynamics model prediction error** (e.g., multi-step rollout MSE) for both zero-shot and few-shot models, quantifying how much online fine-tuning actually improves model quality and how that correlates with planning performance improvements.

3. **Address the theory-practice gap explicitly**: Even an empirical study showing that Residual-MPPI with a suboptimal prior (e.g., undertrained SAC checkpoints) degrades gracefully, or that varying γ and noise variance has predictable effects, would substantially strengthen the connection between Proposition 1 and the practical algorithm.

4. **Report per-step computation time** for both MuJoCo and GTS experiments, since real-time feasibility is central to the "online customization" claim.

## Evaluation Axes

- **Originality**: Moderate. The core idea—substituting log π into the MPPI objective via the RQL augmented reward—is a natural but well-motivated extension. The application to continuous policy customization (extending Li et al.'s discrete-space work) is incremental but meaningful.

- **Importance of research question**: High. Online policy customization without access to original training details is practically important for robotics and deployment.

- **Claims well supported**: Partially. The empirical results are convincing on the tested environments, but the theoretical claim of "solving the augmented MDP" relies on idealized assumptions not validated in practice, and key baselines fail without sufficient analysis of why.

- **Soundness of experiments**: The MuJoCo experiments are reasonable with good baseline coverage. The GTS experiment is ambitious but limited by the absence of dynamics model evaluation and the categorical failure ("*Failed*") of several baselines without diagnostic analysis.

- **Clarity**: Good. The paper is well-structured with clear algorithm presentation and informative figures.

- **Value to research community**: Moderate-to-good. The formulation of continuous online policy customization and the GT Sophy demonstration have practical value, even if the theoretical contribution is limited.

## Score and Decision

**Calibration**: BMPC (MPPI+policy learning, Accept Poster, avg ~7) is more incremental in novelty but has cleaner experiments and stronger baselines. MPC² (hierarchical MPPI, Accept Poster, avg ~6.25) has similar profile—heuristic engineering with practical results. M³PC (test-time MPC with pretrained models, Accept Poster, avg ~7) has cleaner theory. Policy Decorator (residual policy for online refinement, Accept Poster, avg ~7) addresses a very similar problem with simpler method but stronger novelty framing. DLPA (MPPI for PAMDP, Reject, avg ~5.2) has limited novelty. The Planner-Guided Policy Optimization paper (Reject, avg ~4) also combines MPPI with policy learning but with weaker results.

Residual-MPPI sits between DLPA (rejected for limited novelty) and BMPC/M³PC (accepted for solid practical contributions). The GT Sophy demonstration is a stronger practical result than typical in this space, but the theoretical gap and limited ablations (no ω' sweep, no dynamics evaluation, no timing) place it below clean accept papers. This is a borderline paper with real practical merit but insufficient depth in analysis.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>