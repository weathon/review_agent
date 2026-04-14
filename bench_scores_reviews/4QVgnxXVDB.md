## Summary

3CIL proposes a causality-inspired conditional imitation learning framework for autonomous driving that decomposes learning into a representation stage and a policy stage. The representation model combines future image reconstruction (causal direction), action residual prediction (capturing Δa_t from ŝ_t), and a Rank-N-Contrast supervised contrastive loss (anti-causal direction) to address causal confusion and spurious correlations. A sample-weighting mechanism based on residual prediction error then guides the policy network to emphasize rare and challenging scenarios. The method is evaluated in CARLA across six scenarios including unseen towns and weather/traffic/camera shifts.

---

## Strengths

- **Coherent translation of causal intuitions into concrete objectives.** The three traits (T1–T3) and their corresponding losses (ℒ_fo, ℒ_ar, ℒ_RNC, sample weighting) form a internally consistent design story. Each loss component has a clear motivation tied to a specific failure mode (future-reconstruction for state inference, residual prediction for breaking the a_{t-1} shortcut, contrastive loss for representation alignment). This level of structured motivation is above average for IL papers at this venue.

- **Empirically strong across most seen and unseen scenarios.** 3CIL achieves the highest accumulated reward in 5 of 6 scenarios, including unseen towns (Scenario 5) and unseen weather/camera configurations. It also achieves competitive or best collision rates in 3 of 6 settings. Compared to specialized baselines like PALR and DIGIC, the improvements are substantial.

- **Specific insight about PALR's failure.** Section 4.2 correctly diagnoses that PALR's aggressive suppression of previous-action influence hurts driving performance because prior action is a genuine causal parent of current state — the approach eliminates information needed for state inference, trading low collision rate for high overall failure. This is a non-trivial observation that meaningfully contextualizes the baselines.

- **The choice to predict Δa_t rather than a_t directly is specifically justified.** By predicting the action residual rather than the absolute action, the model avoids encoding a_{t-1} as a direct cue while still capturing its causal influence through state dynamics. This design decision is well-reasoned and non-obvious.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Scenario 6 failure is unexplained and undermines the robustness narrative.** In Table 1, 3CIL achieves only 195.53 reward in Scenario 6 (Town05, unseen) versus RAP's 447.44 — a ~56% shortfall. This is the paper's most direct test of generalization to novel map layouts, and 3CIL performs worse than even relatively simple baselines (Keyframe: 215.77, DIGIC: 409.88). The paper says "3CIL still maintains a robust driving strategy" and cites "5 of 6," but this failure is not discussed or analyzed at all. Given that generalization is the paper's central claim, this omission is a significant credibility problem. The authors should analyze what causes this collapse — is it the contrastive loss, the weighting scheme, or the frozen representation failing on novel road layouts?

- **The core causal confusion claim is never directly validated.** The paper's stated goal is to alleviate causal confusion and reliance on spurious correlations, but the experiments only report reward, collision rate, and speed. There is no intervention-based test (e.g., perturbing speed while holding the scene constant, or the copycat diagnostic probing whether the model's actions track ŝ_t vs. a_{t-1}). Without such evidence, the reader cannot know whether 3CIL actually relies less on spurious features or simply has better general representation quality. This matters because an auxiliary-task explanation for the performance gains would not require the causal framing at all.

- **No ablation from a shared base architecture.** The paper treats RNC and RAP (separate published methods with different architectures and hyperparameters) as ablations of 3CIL. While this is acknowledged in the text, it is not a valid substitution: differences in performance could reflect architectural gaps rather than the presence or absence of a particular objective. A proper ablation would start from 3CIL's architecture and remove ℒ_RNC, ℒ_ar, and the weighting term one at a time. The paper states a detailed ablation is in Appendix A.4, but this result is not visible in the main text and the community should be able to assess it directly.

### Minor

- **The observation function notation is internally inconsistent.** The paper writes (o_t, v_t) ~ F(o_t, v_t | s_{t+1}, a_{t-1}), but then says these variables are "recorded as the proxy of the state s_t observed by expert." If o_t is a proxy for s_t, conditioning on s_{t+1} is temporally awkward and the notation creates confusion for readers trying to follow the causal diagram in Figure 1b. The intended meaning (camera captures the result of prior state–action transition) may be correct, but the indexing should be clarified.

- **The mutual information claim for Eq. (2) is informal.** The paper states that the action residual prediction objective "maximize[s] the conditional mutual information I(ŝ_t, a_t | a_{t-1})" but Eq. (2) is a plain MSE loss. Minimizing MSE on a deterministic predictor is not formally equivalent to maximizing CMI; this is at best an intuition. Either a derivation should be provided (e.g., as a lower-bound argument) or the language should be softened to "proxies" or "encourages."

- **Sample-weighting hyperparameters lack sensitivity analysis.** The clipping range [-0.3, 0.3] and γ=6.67 are hard-coded. Since the final policy quality depends on this weighting, readers need to know whether performance is robust to these values or whether they were tuned specifically for CARLA. Even a brief two- or three-point sweep in the appendix would be valuable.

- **Collision rate in Scenario 3 is notably worse for 3CIL.** In Scenario 3, 3CIL achieves the best reward (420.38) but the worst collision rate (3.15‰) — substantially worse than CIL (1.35), PALR (4.18 is worse, but several baselines are clearly better). The paper does not address this specific tradeoff. For a safety-critical application, an explanation of why high reward coexists with elevated collision rate in this scenario is important.

- **The "causal direction" / "anti-causal direction" framing is used loosely throughout.** Terms like "anti-causal supervision" and "stable causal effects" are evocative but not formally defined. This is acceptable if the paper is clear it is using these as design metaphors, but the language in some places (e.g., the analogy between Eq. (4) and inverse probability weighting / doubly robust learning) risks misleading readers who expect these to be standard causal estimators. Eq. (4) is not an IPW estimator and lacks a propensity model or target distribution; the analogy should be presented as loose inspiration, not a methodological equivalence.

### Tiny

- The two-stage training (freeze G before training J) is not discussed in terms of its potential disadvantage — specifically, whether end-to-end finetuning might recover the Scenario 6 failure case. A brief justification or comparison would strengthen Section 3.3.

- The RNC loss for continuous multidimensional actions (steer + throttle/brake) uses d(a_k, a_i) without specifying how multi-component action distance is computed or whether scaling across action dimensions is handled. This detail matters for reproducibility.

---

## Nice-to-Haves

- **Direct causal confusion diagnostic.** Providing the conflicting-cue test (feed model with contradictory speed vs. visual signals and measure which dominates action) would be the most direct validation of the paper's central claim and would strongly differentiate it from papers that simply add auxiliary losses.

- **Qualitative analysis of Scenario 6 failures.** Side-by-side rollout comparisons showing where and why 3CIL fails in Town05 vs. where RAP succeeds would substantially clarify the Scenario 6 gap.

- **Visualization of high-weight samples.** Showing which training episodes receive high weight under Eq. (4) — and confirming these are genuinely rare or safety-critical scenarios rather than noisy labels — would validate the weighting mechanism's design rationale.

- **Representation space visualization.** A t-SNE/UMAP of ŝ colored by action or scenario type would provide evidence that the RNC loss actually structures the representation as claimed.

- **Comparison with simpler distribution-shift baselines.** A straightforward loss re-weighting by naive prediction error (without the causal framing) would help isolate whether the causal motivation adds anything beyond standard hard-sample mining.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Directly mapping (o_t, v_t) to a_t is inappropriate" is too strong (Harsh Critic).** The harsh critic argues this claim is unjustified because many successful driving policies do exactly this. However, the paper is making a specific argument about partial observability and the need to infer s_t: given the POMDP framing and the causal graph, the point is that without history aggregation the imitator lacks sufficient information. The paper is not claiming such a mapping never works; it is motivating the use of observation history. This is a reasonable design choice, not an overstatement.

- **Eliminating a_{t-1} from the representation model is questionable (Harsh Critic).** The critic notes that previous action is a genuine parent of current state in the causal graph. The paper explicitly acknowledges this tension and explains that Δa_t prediction indirectly captures the effect of a_{t-1} on s_t. This is addressed in the paper and the workaround is principled. The concern is therefore already handled.

- **No statistical significance / multiple seeds (Harsh Critic).** Single-run CARLA evaluation is standard in this sub-community. Requesting confidence intervals or multi-seed statistics for large CARLA benchmarks imposes a standard not applied to most related work. This is not a fair criticism under the paper's evaluation norms.

- **Novelty is "mostly in combination, not individual components" (Harsh Critic).** Technically true, but a well-justified combination that empirically outperforms individual components and provides a unifying conceptual framework is a legitimate contribution. This is not a meaningful weakness on its own.

- **"The paper should be evaluated on another benchmark or expert policy" (Harsh Critic).** Requesting validation on an entirely different benchmark is scope-expanding beyond the paper's stated contribution. CARLA is the standard for end-to-end driving IL evaluation. This is scope creep.

- **Generic strength about extensiveness of experiments** (Positive Reviewer) — removed as this applies to most papers with a comparable experiment count.

---

## Novel Insights

The most insightful observation across the three reviews — not made by the paper itself — is the asymmetry between the paper's two "partial ablations": RAP wins Scenario 6 despite being just one component of 3CIL, while 3CIL comprehensively wins five other scenarios. This pattern suggests that the full 3CIL objective may overfit its representation to training-town dynamics (through the combined supervision of ℒ_fo + ℒ_ar + ℒ_RNC on a fixed architecture) in a way that each individual objective does not. In other words, the three objectives may jointly constrain the representation too tightly for novel map layouts, whereas the individual losses leave more degrees of freedom for generalization. Investigating whether this is an optimization issue (conflicting gradients between ℒ_ar and ℒ_RNC), a capacity issue, or a data coverage issue in training towns would be a concrete scientific contribution worth pursuing.

---

## Suggestions

1. **Analyze and explain the Scenario 6 collapse.** Conduct a targeted investigation: run 3CIL rollouts in Town05, identify the failure mode (stuck, collisions, wrong turns), and correlate it with specific components. Run 3CIL minus the weighting term and 3CIL minus ℒ_RNC in Scenario 6 to isolate which component causes the regression relative to RAP.

2. **Move the ablation table from Appendix A.4 to the main text.** Replace one of the qualitative discussion paragraphs in Section 4.2 with the clean component ablation. This is the most important single piece of evidence for understanding what makes 3CIL work.

3. **Add a causal confusion diagnostic.** Implement a simple copycat/inertia probe: present the trained model with scenes where the action implied by visual context contradicts the action implied by the speed trend. Report which cue dominates for 3CIL vs. baselines. This directly validates the paper's core claim and is implementable within CARLA.

4. **Soften the MI and IPW language.** Change "maximize the conditional mutual information" in the main text to "encourages high conditional mutual information" and explicitly label the IPW analogy in Section 3.3 as motivational rather than formal.

5. **Provide a hyperparameter sensitivity sweep for γ and the bounding range.** A simple 3-point grid (γ ∈ {3, 6.67, 10}, bound ∈ {[-0.2, 0.2], [-0.3, 0.3], [-0.5, 0.5]}) in the appendix would demonstrate robustness of the weighting scheme.

6. **Clarify the action distance metric in RNC for multi-component actions.** State explicitly how d(a_k, a_i) is computed (e.g., L2 of [steer, throttle, brake] with or without normalization) and discuss sensitivity to action scaling.

---

## Evaluation

**Originality:** Moderate-to-good. The individual components (RSSM, future reconstruction, action residual prediction, supervised contrastive loss) are all prior work, but the specific combination under a causal framing with residual-error-driven sample weighting is novel and non-obvious. The paper's contribution is more synthesis than invention, but the synthesis is principled.

**Importance of research question:** High. Causal confusion in imitation learning for autonomous driving is a genuine and well-documented problem. The paper addresses a real failure mode with practical safety implications.

**Claims well-supported:** Partially. Performance claims in 5 of 6 scenarios are supported by Table 1. However, the core causal confusion claim — that 3CIL actually relies less on spurious correlations — is never directly tested. The Scenario 6 failure is also not addressed, which weakens the robustness claim.

**Soundness of experiments:** Adequate but with gaps. The CARLA setup, six-scenario evaluation, and diverse baselines are appropriate. The missing proper ablation (from shared base) and the unexplained Scenario 6 result are the main soundness concerns. Single-run evaluation is standard and acceptable for this community.

**Clarity of writing:** The paper is clear at a conceptual level. The causal diagram and loss functions are presented in a consistent logical order. The main clarity issue is the informal use of causal language (MI claims, IPW analogy) that risks misleading technically precise readers.

**Value to the research community:** Moderate-to-high for the imitation learning and autonomous driving communities. The method is practical, the code contribution (combining RSSM with RNC and residual prediction) is directly usable, and the analysis of PALR's failure mode is insightful. The results would be more valuable if the Scenario 6 failure were explained and the causal story were empirically verified.

**Contextualization relative to prior work:** Good. The paper engages meaningfully with causal confusion (de Haan et al.), copycat/inertia (Codevilla et al., Wen et al.), and relevant representation learning works (PALR, Premier-TACO, DIGIC). The positioning of why prior approaches are insufficient is specific and largely accurate.