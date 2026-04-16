## Summary
This paper proposes Residual-MPPI, an online planning method for continuous-control policy customization that uses a prior policy’s log-likelihood together with an add-on reward inside MPPI. The goal is to adapt a deployed policy to new requirements without retraining the policy itself, and the paper demonstrates this on MuJoCo tasks and, notably, on customizing GT Sophy 1.0 in Gran Turismo Sport for safer route selection.

## Strengths
- **Addresses a practically important problem with a clean formulation.** The paper targets online customization of deployed continuous-control policies under new requirements, which is a meaningful robotics/autonomy problem where full retraining is often impractical.
- **The core method is simple and operationally appealing.** Using the prior policy both as a sampler/initializer and via a `log π` reward term is a natural way to preserve prior behavior while optimizing add-on objectives.
- **Good empirical scope, especially the GTS case study.** Beyond standard MuJoCo tasks, the paper demonstrates customization of GT Sophy 1.0 in a challenging racing simulator, which is a substantially more compelling test than toy continuous-control alone.
- **Baseline design is generally thoughtful.** The paper includes prior policy, multiple MPPI variants, and RL baselines, including privileged variants with access to reward/value information, which helps isolate what the `log π` term contributes.
- **Evidence for strong interaction efficiency relative to RL fine-tuning.** In both MuJoCo and GTS, Residual-MPPI appears much more interaction-efficient than Residual-SAC, especially in the racing setup where Residual-SAC needs vastly more data to become viable.

## Weaknesses

###: Fatal
- None. The paper is a real contribution with meaningful experiments; the main issues are overclaiming and incomplete analysis rather than a collapse of the entire contribution.

### Major:
- **The theoretical justification is materially weaker than the paper’s rhetoric suggests.** In Sec. 3.1, Proposition 1 connects MPPI to a maximum-entropy policy only under restrictive conditions: deterministic dynamics, `γ = 1`, terminal value `V*`, and effectively uniform action-sequence sampling via “Gaussian noise with infinite variance.” The actual method in Algorithm 1 and Eq. (6) uses finite covariance, discounted rewards, learned dynamics, and no terminal soft value. So the paper does **not** really establish that the implemented Residual-MPPI is a principled solver for the augmented MDP in the same strong sense claimed; rather, it provides motivation for a heuristic approximation. This matters because the paper repeatedly frames the method as theoretically grounded, even calling the `log π` incorporation “theoretically sound.”
- **The main-text algorithm appears incorrect/incomplete as written.** In Algorithm 1, line 13 defines a normalizer `η`, but line 15 defines the sample weights without using it, and line 18 then updates the action sequence with apparently unnormalized weights. If this is just a transcription mistake, it is still a substantive one because it affects understanding and reproducibility of the core update rule.
- **The empirical claims of broad superiority are overstated relative to Table 1.** The MuJoCo results support that Residual-MPPI is competitive and sometimes clearly better, but not that it consistently “outperforms” Greedy-MPPI or is the “ideal choice.” In HalfCheetah, Swimmer, and Hopper, Residual-MPPI is effectively tied with Greedy-MPPI on total reward; the clearest gap is Ant. Likewise, Valued-MPPI is competitive on preserving the basic task. The paper’s discussion in Sec. 4.2 is stronger than what the numbers justify.
- **No computational/runtime analysis is provided for an online planner.** Since the method is explicitly positioned as an online execution-time customization approach built on sampling-based MPPI, omitting wall-clock cost, planning frequency, or real-time feasibility is a significant omission. This is especially important for the GTS setting, where practical deployability is part of the appeal.
- **Dependence on dynamics-model quality is central but insufficiently characterized.** The paper acknowledges this in Sec. 3.2 and Sec. 7, but there is no systematic analysis of how model error affects customization quality, when zero-shot breaks down, or how much few-shot fine-tuning recovers. Given that the method requires a learned/plausible dynamics model, this is a key practical limitation.

### Minor
- **The evaluation only partially matches the paper’s broader “retain prior properties” framing.** In practice, retention is evaluated mostly through basic-task reward in MuJoCo and lap time in GTS. That is reasonable, but narrower than the paper’s wording about preserving the prior policy’s “properties.” Some direct behavioral similarity analysis would have made this claim better supported.
- **The abstract/introduction somewhat overstate plug-and-play simplicity.** The abstract says customization is possible “given access to the prior action distribution alone,” but the method section clearly also requires a dynamics model (`F` in Algorithm 1), and in practice the paper relies on learned dynamics and sometimes online fine-tuning. Similarly, the claim that the method “eliminates the need for additional policy training” is true for the policy, but not for the dynamics model.
- **Sensitivity of key hyperparameters is underexplored in the main paper.** The balance parameter `ω'` is central to the customization trade-off, yet there is no substantive main-text analysis of how to set it. The same applies, to a lesser extent, to planning horizon, sample count, and temperature.
- **Few-shot evidence is incomplete across domains.** The paper discusses both zero-shot and few-shot Residual-MPPI, but the main MuJoCo table reports zero-shot only, while GTS highlights few-shot improvement. A more complete matrix would better support the few-shot story.

### Trivial
- **Some explanatory claims could be phrased more carefully.** For example, the discussion that `log π` “serves as a proxy of the prior policy’s Q-function” is directionally intuitive under maximum-entropy assumptions, but stronger than what the paper itself rigorously establishes.

## Nice-to-Haves
- A direct study of when Greedy-MPPI is sufficient versus when the `log π` term becomes crucial would sharpen the practical takeaway, especially since the benefit is dramatic in Ant/GTS but modest in several MuJoCo tasks.
- More diagnostics in GTS—e.g., where remaining off-course steps occur on the track, or segment-wise failure analysis—would better illuminate the method’s remaining limitations.
- A behavioral closeness metric to the prior policy, beyond reward proxies, would better align evaluation with the paper’s customization framing.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaints about omitted baselines/related work.** Several reviewer comments ask for comparisons to additional methods (e.g., TD-MPC2, MBPO-style methods, other model-based planners). Per instruction, I do not treat missing related works as a weakness here.
- **Criticism that some GTS baselines are unavailable because of access constraints.** The paper explicitly states in Sec. 5.1 that for GT Sophy they only have access to the policy network, so Valued-MPPI cannot be run. This is not a valid weakness in itself.
- **Generic sim-to-real criticism.** The paper already discusses sim-to-real limitations in Sec. 7. Asking it to solve real-world transfer is outside the stated scope; at most this is a future direction, not a core flaw.
- **Claim that “zero-shot” is invalid because a dynamics model is trained.** The paper’s usage is clearly “zero-shot policy customization” / no customization-time policy training, not no prior learning of any component. The terminology could be clarified, but the criticism in its strong form would be misleading.
- **Strong criticism of deterministic-dynamics assumptions as invalidating the paper.** Proposition 1 indeed assumes deterministic transitions, but the paper uses this as a limited theoretical bridge and then demonstrates the method empirically with learned models. This weakens the theory claim, but does not by itself invalidate the empirical paper.

## Novel Insights
The most important synthesis is that this is best viewed as a **strong empirical systems/planning paper with a weaker-than-advertised theory wrapper**. The practical contribution is real: using the prior policy as both sampling prior and objective proxy enables an effective customization mechanism, and the GTS demonstration is genuinely impressive. But the paper would be substantially stronger if it framed Residual-MPPI as a motivated approximation to the augmented-MDP objective rather than as a near-direct theoretically justified solver. The gap between what the experiments establish (“useful, scalable, interaction-efficient customization”) and what the theory language implies (“principled solution of the augmented MDP”) is the central issue.

## Suggestions
- Recast the theoretical claims more modestly: explicitly state that Proposition 1 is a motivating bridge under restrictive assumptions, while the implemented Residual-MPPI is an approximation.
- Fix Algorithm 1 in the main paper so the weight normalization is unambiguous and matches the implementation exactly.
- Add a runtime/planning-cost analysis for MuJoCo and GTS, including per-step inference/planning latency.
- Add a focused sensitivity analysis for `ω'`, since it is central to the behavior trade-off and practical usability.
- Temper empirical language in Sec. 4.2 from “outperforms/ideal choice” to “competitive and clearly advantageous in settings where preserving prior behavior matters over long horizons.”
- Include a more direct analysis of dynamics-model quality, ideally by varying training data or comparing zero-shot vs few-shot in a controlled MuJoCo setting.
- Add a behavioral-retention metric or visualization to support the stated goal of preserving prior-policy properties, not only reward trade-offs.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Moderate. The integration of RQL-style customization with MPPI is natural rather than conceptually radical, but still meaningful.  
- **Importance:** High. Online customization of deployed control policies is valuable for robotics/autonomy.  
- **Claims support:** Mixed. The empirical utility is supported; the stronger theory and superiority claims are not fully supported.  
- **Experimental soundness:** Good overall, with a standout GTS demonstration, but missing runtime and deeper model-quality analysis.  
- **Clarity:** Generally clear, though Algorithm 1 has a concerning specification issue and some claims are overstated.  
- **Community value:** Solid. The GTS case study and practical framing make this useful to the robotics/planning community.

**Calibration against retrieved human-reviewed papers:**  
- Compared to **M³PC** (`/home/wg25r/review_agent/human_reviews/inOwd7hZC1.md`, accepted, scores 6/8/8/6), this paper is similar in having a practically appealing MPC-style idea and good experiments, but weaker on completeness because it lacks runtime analysis and overstates its theory more. So I place it somewhat below that acceptance level.  
- Compared to **Off-Road Autonomous Driving via Planner Guided Policy Optimization** (`/home/wg25r/review_agent/human_reviews/uaKBM9sGEm.md`, rejected, scores 6/3/1/6), this paper is stronger: the method is cleaner, the empirical evidence is more convincing, and the GTS showcase is better aligned with the stated contribution.  
- Compared to **Equivariant TD-MPC** (`/home/wg25r/review_agent/human_reviews/vl3F3s8OMg.md`, rejected, scores 3/3/6/5), this paper is again stronger empirically and clearer in practical value, though it shares some theory-to-practice mismatch issues.  
- Compared to **Q-Adapter** (`/home/wg25r/review_agent/human_reviews/WLSrq1254E.md`, accepted with uniform 6s), this paper feels in a similar broad quality band: meaningful contribution with clear practical value, but not a slam-dunk due to missing analyses and some overclaiming.

Overall, I view this as **borderline but slightly below accept**: a good and useful paper whose practical contribution is real, but whose theory framing and evaluation claims need tightening.

**Score: 5.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>