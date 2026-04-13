=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary
This paper studies a meaningful and underexplored setting: learning an offline RL policy when rewards are unavailable but preference feedback can be queried. The core proposal, Sim-OPRL, uses a learned transition model to generate simulated trajectory pairs for preference elicitation, combining pessimism over transition uncertainty with optimism over reward uncertainty. The paper also provides a formal problem setup and theoretical analyses for both querying from the offline buffer (OPRL-style) and querying model-generated rollouts.

## Strengths
- **The paper identifies and formalizes a genuinely important problem that is not just standard offline RL or standard preference-based RL.** Section 3 explicitly defines the offline preference elicitation setting, where one has only logged trajectories plus a limited preference-query budget, and frames the goal relative to an offline optimum rather than the unattainable fully interactive optimum. This is a concrete contribution beyond merely combining existing components.
- **The central algorithmic idea is principled and well aligned with the problem structure.** Sim-OPRL uses pessimism for transition uncertainty and optimism for preference acquisition, which is exactly the tension one would expect in this setting. The ablation in Figure 2 is specific and supportive: removing pessimism in rollouts or output policy hurts performance, matching the intended mechanism rather than just showing aggregate gains.
- **The paper contributes theory for both offline-buffer querying and model-rollout querying, rather than presenting only an empirical heuristic.** In particular, Theorem 5.1 gives a bound for OPRL-style querying and Theorem 6.1 gives a bound for Sim-OPRL-style querying. Even if the comparison between them is not as clean as the paper sometimes suggests, the attempt to characterize sample complexity in this hybrid offline/preference-learning setting is substantive.
- **The empirical results consistently show that the proposed query strategy improves over the direct offline-querying baselines actually evaluated.** Across all five environments in Figure 1 / Table 2, Sim-OPRL beats both OPRL-Uniform and OPRL-Uncertainty in preference efficiency, and the gap is especially notable in Sepsis and HalfCheetah-Random.
- **The paper surfaces a practically interesting application-specific benefit of simulated queries.** In Section 7, the authors note that in sensitive domains such as healthcare, querying experts about synthetic trajectories rather than real patient trajectories may be preferable. This is a concrete, domain-relevant advantage of the approach.

## Weaknesses
###: Fatal

### Major:
- **The main theoretical comparison between Theorem 5.1 and Theorem 6.1 is not clean enough to fully justify the paper's stronger claims of an “improved theoretical guarantee.”**  
  This concern is real on reading the paper itself. In Section 6.1, the paper explicitly decomposes the error as
  \[
  V^{\pi^*}-V^{\hat{\pi}^*}
  \le (V_{T,R}^{\pi^*}-V_{\hat T_{\inf},R}^{\pi^*})
   + (V_{\hat T_{\inf},R}^{\pi^*_{\text{offline}}}-V_{\hat T_{\inf},\hat R_{\inf}}^{\pi^*_{\text{offline}}}),
  \]
  i.e., the reward term is analyzed under the **learned pessimistic model** and for \(\pi^*_{\text{offline}}\), not under the true dynamics and \(\pi^*\). By contrast, Section 5 analyzes the OPRL reward term with a concentrability factor \(C_R(\mathcal F_R,\pi^*)\). So while Theorem 6.1 does remove \(C_R\), it does so after shifting the object being controlled. That does not make the theorem useless, but it means the paper overstates the result when it says Sim-OPRL has “improved theoretical guarantees” or “eliminates the reward concentrability term” in a way that reads like a direct apples-to-apples strengthening for the original offline problem. The theory is suggestive, but not as definitive as the presentation implies.
- **There is a substantial theory-to-practice gap, and it matters here because the contribution hinges on calibrated optimism/pessimism.**  
  The theoretical development in Sections 4–6 relies on confidence sets \(\mathcal T,\mathcal R\), exact worst-case optimization in Eq. (3), and the candidate policy set \(\Pi_{\text{offline}}\) from Algorithm 2. The practical method in Section 6.3 replaces these with bootstrap ensembles, disagreement-based uncertainties, penalized rewards in Eq. (5), and one-policy-per-reward-model style approximations. The authors do acknowledge that Section 6.3 is a “possible implementation strategy,” but the experiments therefore validate a heuristic inspired by the theory rather than the analyzed algorithm itself. For many RL papers this would be acceptable; here it is more consequential because the claimed advantage of Sim-OPRL is specifically about how optimism and pessimism are balanced.
- **The empirical case is good for showing improvement over OPRL, but not broad enough to support the paper's broader claims about scalability or advancing the state of the art in offline preference-based RL.**  
  The main experiments cover five environments, but only one standard continuous-control D4RL task appears in the main paper, and much of the deeper analysis is concentrated on StarMDP. The comparisons are primarily against OPRL variants plus an online upper bound (PbOP). That is enough to support the narrower claim that Sim-OPRL is better than direct offline-buffer querying in these settings, but not the stronger claims in Section 7 and the conclusion about broad practical scalability or state-of-the-art advancement.
- **The method's dependence on transition-model quality and offline coverage is central, but the paper does not clearly delineate when Sim-OPRL should be preferred over simpler OPRL-style querying.**  
  To the authors' credit, this issue is not hidden: Section 6.2 explicitly states that performance depends on coverage of the optimal policy, and Figure 3a/3b shows performance degrading when the offline dataset is small or poorly covering. In fact, the paper itself states, “A more accurate transition model should therefore require fewer preference samples,” and Figure 3 shows that when coverage is poor enough, Sim-OPRL's advantage shrinks substantially. This is an important limitation of practical significance, and the paper should sharpen the operational takeaway: when is the extra modeling machinery worth it, and when is it not?

### Minor
- **Some core definitions and notation are sloppy enough to hinder interpretation.**  
  Definition 3.1 appears misstated: it says “\(V_{\pi,R}^{\pi} - V_{\pi,R}^{\pi^*} \le \epsilon\),” which is likely a sign/order/notation error for comparing \(\hat\pi\) and \(\pi^*\). Algorithm 1 line 9 uses \(d_\pi^\pi\), which is inconsistent with the earlier notation \(d_T^\pi\) or \(d_\pi^\tau\). These are not mere style issues because Definition 3.1 is the central correctness criterion.
- **The role and definition of \(\pi^*_{\text{offline}}\) / \(\epsilon_T\) are not fully precise in the main text despite being central to the framing.**  
  Section 3 says \(\pi^*_{\text{offline}}\) is the “optimal offline policy estimated based on the offline data, with access to the true reward function,” but that leaves ambiguity about whether this is under the pessimistic model, support constraints, or some other offline restriction. Since much of the later argument pivots on \(\pi^*_{\text{offline}}\), this should be crisper in the main paper.
- **The paper does not explain some theorem-level notation changes clearly enough.**  
  For example, \(\kappa\) is defined differently in Theorem 5.1 and Theorem 6.1 (\(1/\sigma^2(r)\) vs. \(1/\sigma(r)\)), and the theorem statements overload names such as \(C_R\) as both a concentrability term and a constant in nearby text. These may be correct, but the presentation makes the results harder to trust and parse.
- **The practical implementation imposes a narrower reward structure than the general setup, but this is only implicit.**  
  Section 4 presents a trajectory-level reward/preference formulation, whereas Section 6.3 implements \(\hat R(s,a)\) and then aggregates over rollouts. That is a reasonable restriction, but it should be stated plainly as a modeling assumption rather than left implicit.
- **The use of synthetic preferences only leaves open robustness questions about real human feedback noise and misspecification.**  
  The whole paper assumes Bradley-Terry preferences. For an algorithmic paper this is an acceptable starting point, and I would not require human studies here, but a more explicit discussion of sensitivity to inconsistent or non-Bradley-Terry human feedback would strengthen the practical interpretation.

### Trivial

## Nice-to-Haves
- Expand the continuous-control evaluation beyond HalfCheetah-Random to better support scalability claims and the theory's dependence on dataset quality.
- Add a clearer practical guideline for when Sim-OPRL is likely to outperform OPRL, based on dataset size/coverage and transition-model quality.
- Provide some analysis of computational overhead, since Sim-OPRL maintains transition, reward, and policy ensembles and performs rollout-based query selection.
- Include a visualization or case study of queried trajectory pairs, showing how Sim-OPRL's selected comparisons differ from OPRL's and why they are more informative.
- Discuss sensitivity to preference model misspecification or annotation noise, even if only in a short robustness experiment or limitations paragraph.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Missing comparison to specific related methods such as FREEHAND.**  
  This may be a reasonable suggestion for future work, but per the review policy I should not criticize the paper for omitting particular external baselines/related works that I cannot independently verify as required comparisons.
- **Comparison to non-preference offline RL with hand-designed rewards.**  
  This is outside the paper's stated problem setting. The paper studies reward learning from preferences when rewards are unavailable; requiring a baseline that assumes hand-crafted rewards is scope creep rather than a core flaw.
- **Simulation-only evaluation as a reproducibility/existence concern.**  
  It is fair to note that the experiments use simulated preference feedback rather than humans, but not fair to elevate this into a claim that the setup is unverifiable or that cited environments/tools are questionable. The paper never claims a human-subjects evaluation.
- **Pure formatting/parser issues.**  
  Some notation artifacts may partially stem from PDF extraction. I retained only those notation issues that materially affect interpretation, and removed mere formatting noise.

## Novel Insights
The most useful way to interpret this paper is not as having fully “solved” offline preference elicitation, but as identifying a concrete regime in which model-based synthetic querying can convert a hard coverage problem in reward-learning space into a transition-modeling problem. That reframes the bottleneck: Sim-OPRL is attractive not because it universally dominates offline-buffer querying, but because it can spend preference budget near plausible high-value policies when the learned dynamics are already good enough. In that sense, the paper's real contribution is less the theorem-level removal of \(C_R\) in isolation, and more the articulation of a practical tradeoff between transition-model quality and preference-query efficiency.

## Suggestions
- **Temper the theoretical claims.** Rephrase “improved theoretical guarantees” and “eliminates the reward concentrability term” to make clear that Theorem 6.1 controls a different reward-estimation object under the learned pessimistic model.
- **Tighten the formalism in Section 3 and Algorithm 1.** Fix Definition 3.1, clarify the precise meaning of \(\pi^*_{\text{offline}}\) and \(\epsilon_T\), and clean up the trajectory-distribution notation.
- **Be more explicit about the theory-practice gap.** State clearly which parts of the practical implementation are heuristic approximations to the analyzed method, and discuss what can fail when ensemble uncertainty is miscalibrated.
- **Strengthen the empirical positioning.** Narrow the claims to “better than OPRL-style offline querying” unless more diverse large-scale experiments are added.
- **Add a practical decision rule.** Based on Figure 3 and Section 6.2, summarize when Sim-OPRL is likely worthwhile versus when transition uncertainty makes direct offline querying more competitive.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0, 6.0]
Average score: 6.8
Binary outcome: Accept
