## Summary
Puppeteer is a hierarchical world model framework for visual whole-body humanoid control using a 56-DoF simulated humanoid. A low-level tracking world model (TD-MPC2) is pretrained on MoCap data to track end-effector reference motions; a high-level puppeteering world model is then trained from scratch per downstream task, taking visual observations and outputting end-effector commands for the frozen tracker. The paper contributes a new 8-task benchmark, demonstrates competitive performance against flat MBRL baselines, and introduces naturalness evaluation via a 51-participant user study showing 97.8% preference for Puppeteer over TD-MPC2.

---

## Strengths

- **Naturalness evaluation with quantitative and human-preference grounding:** The paper explicitly evaluates motion naturalness both via a user study (n=51, 97.8% preference) and via proxy metrics (torso height, episode length in Table 1). Unlike the majority of humanoid RL papers that collapse evaluation to task reward only, this dual evaluation reveals qualitative failure modes of flat RL — such as TD-MPC2 "rolling" up stairs — that would otherwise be hidden in aggregate return numbers. This is a specific and nontrivial contribution to evaluation methodology.

- **Single reusable tracking model replacing thousands of per-clip policies:** A single 5M-parameter tracking world model trained on 836 MoCap clips is frozen and reused across all 8 downstream tasks. This directly contrasts with MoCapAct's ~2600 individually trained policies and achieves downstream task learning in ≤3M steps vs. ~150M in prior work. The architectural economy is not just efficiency-for-its-own-sake but enables a clean task-agnostic motion prior.

- **Hierarchical planning ablation with stark quantitative evidence:** Figure 8 shows normalized scores of 43.0 (planning at both levels), 6.4 (planning at low level only), and 2.9 (planning at high level only) vs. 1.6 (no planning). This is among the most decisive ablation results in the paper — the 7× gap between single-level and full hierarchical planning powerfully validates the architecture's core claim.

- **Zero-shot generalization to out-of-distribution gap lengths:** The method, trained on gaps [0.1, 0.4]m, achieves non-trivial performance on gaps up to 1.2m (3× the training range), while TD-MPC2 degrades immediately outside training distribution. This is a concrete and measurable generalization benefit attributable to the MoCap prior embedded in the hierarchy.

- **Task suite contribution:** The 8-task visual humanoid benchmark with a 56-DoF model fills a real gap acknowledged even by concurrent work (HumanoidBench uses non-visual observations and a less expressive embodiment). Releasing both environments and code makes the contribution reproducible and community-usable.

---

## Weaknesses

- **Inaccurate "without any reward design" claim:** The abstract, Figure 1 caption, and Section 3 repeatedly state the method requires no reward design. This is demonstrably inaccurate: the tracking stage uses the hand-crafted reward function from Hasenclever et al. (2020), and all 5 visual tasks use a forward linear velocity reward. What the paper actually achieves is avoiding *task-specific style reward shaping or adversarial objectives*, which is a real and meaningful contribution — but misstating it as "no reward design" misrepresents the approach. This inaccuracy propagates throughout the paper and should be corrected to something like "without adversarial reward terms, skill-specific reward shaping, or manually designed gait incentives."

- **User study does not isolate the source of naturalness improvements:** The naturalness user study compares Puppeteer only against flat TD-MPC2, but two other hierarchical baselines (SAC + LL TD-MPC2, DreamerV3 + LL TD-MPC2) also share the same frozen tracking agent and thus inherently produce motions driven by the same MoCap prior. If the naturalness gain is primarily a property of the shared tracking agent rather than of the puppeteering world model specifically, the user study as designed does not distinguish these hypotheses. Including at least one hierarchical baseline in the user study is necessary to cleanly attribute the naturalness improvement.

- **Temporal abstraction (k) presented as a key feature but never used:** Section 3.2 introduces k (low-level steps per high-level step) as "a hyperparameter that allows us to trade strong motion prior for control granularity," implying it is a significant tunable design feature. However, k=1 is used in all experiments without any ablation or empirical motivation. At k=1, the hierarchy provides no temporal abstraction. The paper's own Section 5 acknowledges this ("does not rely on... temporal abstraction for task learning") but only in the related work, not in the methods section where k is introduced. This framing is misleading — the real contributions of the hierarchy are modality separation and data-source separation, which should be described more accurately from the outset.

- **Naturalness metrics in Table 1 are evaluated on a single task only:** The quantitative naturalness proxies (episode length and torso height) are reported only for the *gaps* task. The claim that these "strongly support" the user study results, which covers multiple tasks, is not justified by a single-task analysis. Given that the paper runs 10 seeds and the data is presumably available, reporting these metrics aggregated across all 8 tasks would substantially strengthen this claim.

- **User study methodology lacks sufficient detail:** Several confounds are not addressed: (1) it is unspecified whether clip presentation order was randomized; (2) it is unspecified whether evaluators were blinded to method identity; (3) task performance is not controlled — if Puppeteer visibly succeeds at a task while TD-MPC2 visibly fails, "naturalness" preference is confounded with competence preference. The 97.8% figure is striking, but its interpretation requires that these confounds be addressed or acknowledged.

- **Visual task reward diversity is limited:** All 5 visual tasks reward forward linear velocity only. The suite is therefore a visual locomotion benchmark rather than a true "whole-body control" benchmark — the agent never needs to navigate backwards, manipulate objects, stop, reach, or change directions. The authors acknowledge this in the limitations, but the gap between the paper's "visual whole-body control" framing and the actual task diversity is significant enough to warrant more prominent acknowledgment in the main paper (not only in the conclusion).

---

## Nice-to-Haves

- **Sensitivity analysis for α (termination loss coefficient):** The termination head is described as a novel contribution, but no ablation is provided for α. A brief sensitivity plot or range would validate robustness of this design choice.

- **Command trajectory visualization:** A visualization of the end-effector reference trajectories generated by the puppeteer agent (smoothness, kinematic feasibility, temporal coherence) would provide meaningful interpretability. It would clarify whether naturalness emerges primarily from smooth command synthesis or from the low-level tracker's motion prior.

- **Sensitivity to offline/online sampling ratio:** The paper explicitly notes "we did not experiment with other ratios" for the 50%/50% pretraining mixture. A brief sensitivity analysis (e.g., 25/75, 75/25) would either confirm robustness or reveal an important hyperparameter.

- **Vision ablation on visual tasks:** Demonstrating that removing visual input degrades performance on visual tasks (hurdles, gaps, stairs) would close the loop on the "visual whole-body control" claim and confirm that the agent is actually using visual observations rather than relying purely on memorized trajectories.

- **Discussion of sim-to-real feasibility:** Not required for a simulation paper, but even a brief analysis of what architectural elements would or would not transfer (third-person RGB, 64×64 rendering, MoCap distribution shift) would significantly increase the paper's relevance to the robotics community.

- **Low-level fine-tuning ablation:** Allowing gradient updates to the frozen tracking agent during downstream task training would test whether "frozen reuse" is a practical convenience or a performance constraint.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: No comparison to AMP/ASE even in non-visual settings.** The paper is explicitly scoped to *visual* whole-body control, and the paper explicitly states these methods do not support visual observations. Demanding a non-visual motion quality comparison is scope creep. The paper's closest contribution — naturalness without adversarial rewards — is validated by the user study within its own setting.

- **Harsh Critic: Figure 8 high-level pretraining inconsistency.** The critic claimed the figure showed "scratch matches or exceeds pretrained," but the figure description clearly states "The Pretrained line shows higher performance, reaching approximately 40%." The critic appears to have misread the figure; the text ("benefits substantially from finetuning") and figure are consistent. Removed.

- **Harsh Critic: JEPA-style label is self-citation for branding.** LeCun is a co-author; using "JEPA-style" as a descriptor for their own joint-embedding prediction framework is appropriate attribution, not branding. The technical connection to Grill et al. (2020) joint-embedding loss is explicit in Section 2. Removed as a substantive criticism, though a brief technical clarification in the text would be helpful.

- **Harsh Critic: The "first to evaluate naturalness" claim is too strong.** The paper hedges with "to the best of our knowledge," which is standard academic phrasing and appropriate. AMP/ASE optimize for naturalness but do not conduct an explicit user study or propose naturalness metrics. The claim is defensible. Removed.

- **Positive Reviewer: Limited visual resolution (64×64).** Single-run 64×64 RGB is standard for MBRL simulation benchmarks (DreamerV3, TD-MPC2). Requesting higher resolution or resolution ablations is not a standard expectation for this setting. Moved to nice-to-have level at most; not a core weakness.

- **Positive Reviewer: Contextualization of sample efficiency.** The paper already explicitly notes the difficulty differential (visual vs. proprioceptive) in the comparison with MoCapAct. The concern is addressed; the addressal is reasonable.

- **Spark Finder: Termination head novelty may be overclaimed.** While DreamerV3 handles episodic tasks, the specific problem of *planning with noisy predicted termination signals and applying soft truncation* is sufficiently specific that the novelty claim is defensible. The critic's concern is legitimate but does not materially undermine the contribution.

---

## Novel Insights

The most genuinely novel observation synthesized across the three reviews concerns the *source* of naturalness improvement in hierarchical MoCap-based control. The paper presents two intertwined factors — (1) the MoCap motion prior embedded in the tracking agent, and (2) the higher-level world model's ability to generate smooth end-effector commands — as jointly responsible for naturalness. However, because the user study only compares against flat TD-MPC2 (which has neither), and because both hierarchical baselines (which share the tracking agent) are excluded from naturalness evaluation, the paper cannot currently attribute what fraction of naturalness comes from the prior alone versus the command quality of the puppeteering agent. This decomposition would have direct implications for future work: if the tracking agent alone drives naturalness (which seems plausible given the 97.8% preference is hard to attribute to command smoothness alone), the correct conclusion is that MoCap priors dominate and the high-level architecture is a performance mechanism, not a naturalness mechanism. This distinction would meaningfully reframe the paper's claims.

---

## Suggestions

1. **Revise "no reward design" language throughout:** Replace with "without adversarial reward terms, task-specific gait incentives, or skill-specific reward shaping" to accurately characterize the method's actual departure from prior work.

2. **Include at least one hierarchical baseline in the user study:** Add SAC + LL TD-MPC2 or DreamerV3 + LL TD-MPC2 to the naturalness comparison to determine whether naturalness improvement is attributable to Puppeteer's world model or to the shared tracking agent alone.

3. **Report Table 1 naturalness metrics across all tasks, not just gaps:** Episode length and torso height should be aggregated across the full benchmark to substantiate "strongly support the user study results."

4. **Foreground the actual nature of the hierarchy in Section 3.2:** Clearly state from the outset that k=1 is used in all experiments and that the hierarchy's contribution is modality/data-source separation rather than temporal abstraction. Reserve k>1 as a described but untested extension.

5. **Report user study methodology in main paper:** State explicitly whether clip order was randomized, whether evaluators were blinded, and how task performance differences between clips were handled in instructions to participants.

---

## Evaluation

- **Novelty:** Solid incremental. Applying dual TD-MPC2 world models in a hierarchical hierarchy with end-effector command interface and MoCap pretraining is a meaningful architectural contribution, though each component (hierarchical RL, MBRL, MoCap tracking) is established. The evaluation framework for naturalness is more novel than the architecture.
- **Technical soundness:** Good, with two notable issues: the "no reward design" claim needs correction, and the k=1-always choice needs honest framing.
- **Empirical support:** Strong for task performance (10 seeds, 8 tasks, clear ablations). The user study is compelling in magnitude but has design gaps regarding confounds and baseline coverage. The naturalness quantitative analysis is underpowered (single task).
- **Significance:** High. The paper introduces a benchmark, a reusable motion prior framework, and evaluation standards for naturalness — all of which are useful to the community regardless of whether Puppeteer itself becomes the dominant method.
- **Clarity:** Good overall, but the k=1 inconsistency and the reward language are sources of confusion that would benefit from revision.

The paper makes a genuine and useful contribution to visual humanoid control and deserves publication after addressing the user study baseline gap and the reward language; the core results are real and the benchmark is a lasting contribution. The weaknesses are fixable and do not invalidate the central empirical finding.

MY FINAL SCORE: <pineapple>6.8</pineapple>