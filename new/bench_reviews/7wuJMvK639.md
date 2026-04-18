Now I have a good understanding of the paper and the calibration papers. Let me now synthesize my final review.

Key calibration references:
- Universal Humanoid Motion Representations (8,8,8 → Accept spotlight): Strong method for humanoid control with motion priors, good results
- H-GAP (8,8,6 → Accept spotlight): Similar problem (56-DoF humanoid control with MoCap data, MPC planning), strong results
- FB-CPR (6,8,6,6 → Accept poster): Zero-shot humanoid control, moderate novelty but clear empirical contribution
- HuWo (6,5,6,3 → Reject): Humanoid locomotion with world model but limited contribution
- HumanoidOlympics (6,5,3,5 → Reject): Benchmark paper with limited method contribution
- Pixel-based Hierarchical Policies (1,3,3,3 → Withdrawn/Reject): Very weak study paper, limited scope
- TD-MPC2 (8,8,8,8 → Accept spotlight): The backbone algorithm, very strong

The Puppeteer paper is somewhere between HuWo (reject) and H-GAP/FB-CPR (accept). Let me assess the core strengths and weaknesses.

Key issues I need to carefully evaluate:

1. **"No reward design/no skill primitives" framing** - The harsh critic claims this is misleading. Let me check: The paper says "without any reward design or skill primitives" in the abstract. But Section 3.1 says they use "the reward function from Hasenclever et al. (2020)" for the tracking agent. And the tracking agent's interface (end-effector commands) functions similarly to a skill primitive. However, the paper is distinguishing itself from approaches that need per-task reward engineering (adversarial motion priors, etc.) and explicit skill primitives. For the *downstream tasks*, the reward is just forward velocity - no reward design. And the "skill" is a general tracker, not hand-crafted per-task skills. The claim is somewhat overblown but has a kernel of truth - the point is that they don't need task-specific reward design or skill primitives, even if the tracking agent does use a structured reward during pretraining. This is a fair criticism but not fatal - the framing should be more precise.

2. **Baseline comparison** - For visual tasks, SAC and DreamerV3 don't work at all. The meaningful comparison is essentially TD-MPC2 flat vs. TD-MPC2 hierarchical. This is a real limitation.

3. **Naturalness evaluation** - Only compared to TD-MPC2. This is valid but the user study preference is so overwhelming (97.8%) that even if the comparison is narrow, the signal is strong.

4. **k=1 means no temporal abstraction** - This is a good observation. The paper claims "hierarchical" but with k=1 and same-frequency agents, it's really just a different action space representation.

5. **Termination handling** - The method is underspecified (no exact formula).

6. **Limited scope (locomotion only)** - Acknowledged by authors.

Let me now write the consolidated review.</think>## Summary

Puppeteer proposes a hierarchical world model for visual whole-body humanoid control. A low-level tracking agent is pretrained on MoCap data (via TD-MPC2) to follow end-effector reference commands, and a high-level "puppeteer" agent learns to output those commands from visual observations for downstream tasks. Both agents use planning with learned world models. The method produces motions strongly preferred by humans over flat TD-MPC2 (97.8% preference in a 51-person study) while achieving comparable task returns on 8 tasks with a 56-DoF simulated humanoid.

## Strengths

- **Clean and effective hierarchical design**: Training a single TD-MPC2 tracking model on all 836 MoCap clips (vs. ~2600 per-clip policies in MoCapAct) and reusing it across all downstream tasks is a genuine efficiency gain. The architecture is simple—both levels use the same TD-MPC2 backbone—and the ablations in Figure 8 clearly show that planning at both levels is critical (no planning: 1.6 normalized score; full planning: 43.0).

- **Compelling naturalness results**: The 97.8% human preference over TD-MPC2 is a very strong signal. Even though the comparison is only against TD-MPC2, the magnitude of the preference (only 0.4% preferring TD-MPC2) makes it hard to attribute to noise. Qualitative results (stair-climbing gait vs. rolling) effectively illustrate the reward-hacking problem that the MoCap prior prevents.

- **New challenging benchmark**: The 8-task suite for 56-DoF visual humanoid control fills a real gap—no prior benchmark combines this embodiment complexity with visual observations. The release of code and environments will benefit the community.

- **Practical termination handling**: Adding a termination prediction head to the world model (Section 3.3) is a simple but necessary extension for episodic MDPs in world model planning, and the paper empirically shows it is important for stability.

- **Useful ablations**: The pretraining ablations (offline vs. online data, number of MoCap clips), the hierarchical planning ablation, the high-level pretraining result, and the zero-shot generalization result (Figure 9) provide actionable insights for future work on high-dimensional world-model control.

## Weaknesses

### Major

- **Framing of "no reward design or skill primitives" is overstated**: The abstract and introduction repeatedly claim the method works "without any reward design or skill primitives." Yet the tracking agent uses a carefully engineered reward function from Hasenclever et al. (2020) that encodes tracking error, contact terms, etc. (Section 3.1: "We label all transitions using the reward function from Hasenclever et al. (2020)"). The low-level tracking agent's interface (end-effector position commands) also functions akin to a learned skill primitive. What the paper actually demonstrates is that downstream task rewards can be simple (forward velocity) because the MoCap prior is absorbed upstream—this is a meaningful contribution, but it is not "no reward design." A more precise framing (e.g., "no per-task reward engineering") would strengthen rather than weaken the paper.

- **Narrow baseline comparison for visual tasks, which are the main novelty**: For the 5 visual tasks, only TD-MPC2 and TD-MPC2+high-level-SAC/DreamerV3 are effective baselines. SAC and DreamerV3 fail entirely. The hierarchical baselines (SAC/DreamerV3 as high-level with TD-MPC2 as low-level) perform poorly, but it is unclear if they are genuinely well-tuned for this setting or whether the failure is architectural rather than algorithmic. No comparison is made to methods that also incorporate MoCap priors (e.g., adversarial motion priors, MoCapAct-style tracking + visual high-level). This makes it hard to disentangle whether the gains come from the hierarchical world-model design, the MoCap prior, or simply from using TD-MPC2 at both levels.

### Minor

- **k=1 eliminates temporal abstraction, weakening the "hierarchical" framing**: The paper explicitly sets k=1 (Section 4.1), meaning both agents act at the same frequency. The hierarchy is purely representational (different action spaces), not temporal. The paper describes temporal abstraction as a "feature" of the approach (Section 3.2) but provides no experiments varying k, leaving this claimed benefit unvalidated.

- **Naturalness evaluation compares only against TD-MPC2**: The user study only shows Puppeteer vs. flat TD-MPC2. TD-MPC2 is a reward maximizer with no naturalness prior, so it predictably learns exploitative gaits. While the preference signal is overwhelming, comparing against at least one method that also incorporates motion priors (e.g., an AMP-style reward variant) would more convincingly attribute the naturalness to the specific design rather than to simply "having a MoCap prior at all."

- **Termination handling is underspecified**: Section 3.3 introduces a termination prediction head and "cumulative weighting (discount) of termination probabilities" for soft truncation, but does not give the explicit formula. For a planning algorithm, the exact form matters for reproducibility and correctness.

- **Task suite is locomotion-only**: All 8 tasks evaluate walking, running, jumping, and navigating terrain. The "whole-body control" framing implies broader capabilities (manipulation, multi-contact), which the tasks do not test. The authors acknowledge this in Section 6.

### Trivial

- The 50%/50% offline/online data mix ratio for tracking pretraining is set without tuning beyond the ablation showing it matters. This is a minor design choice that others replicating the method would need to explore.

## Nice-to-Haves

- An ablation varying k (the ratio of low-level to high-level steps) to validate or invalidate whether temporal abstraction helps in this setting.
- Comparison to a MoCapAct-style low-level policy (even on proprioceptive tasks stand/walk/run) to substantiate the "orders of magnitude more efficient" claim with like-for-like counts.
- Richer naturalness metrics (e.g., joint acceleration smoothness, foot-slip metrics, Fréchet distance to MoCap trajectories) to complement the coarse proxies in Table 1.
- Analysis of what the puppeteer actually learns from visual observations (e.g., adversarial visual perturbation study).

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"No real-robot/sim-to-real evaluation"**: The paper is explicit about operating in simulation and makes no real-robot claims. Demanding real-robot evaluation for a simulation benchmark paper is scope creep. The 56-DoF CMU Humanoid is a standard MuJoCo testbed.

- **"Observability of visual features" (what does the agent use from vision?)**: This is a nice-to-have analysis, not a fundamental weakness. Many RL papers successfully use visual observations without decomposing what features the agent learns.

- **"User study population (students) may carry biases"**: This is a standard study design. The 97.8% preference is so overwhelming that minor demographic biases would not change the conclusion.

- **"Missing comparison with HumanoidBench"**: The paper cites HumanoidBench (Sferrazza et al., 2024) and explains why direct comparison is not feasible—different embodiment, no visual observations in HumanoidBench. This is a reasonable design choice.

- **"Inference-time latency of MPC planning"**: The paper already reports training times (12+4 days on a single 3090). Real-time control latency concerns are standard for any MPC method and not unique to this work.

- **"Not comparing to MoCapAct/DeepMimic directly due to vision incompatibility is partially dodge — compare on proprioceptive tasks"**: This has some merit but the paper already refrains from claiming superiority over those methods on their own terms. The methods are sufficiently different in setting (state vs. visual observations, different training budgets) that a forced comparison would be misleading rather than illuminating.

## Novel Insights

The paper demonstrates a surprisingly clean result: consolidating MoCap-based tracking from thousands of per-clip policies into a single world model, and then using that model as a reusable low-level controller for a higher-level visual planner, produces motions that humans overwhelmingly prefer over end-to-end TD-MPC2—while achieving comparable task returns. The core finding is not that hierarchy solves the control problem better (flat TD-MPC2 actually matches or exceeds Puppeteer on reward), but that the MoCap-grounded action space is what prevents reward hacking and yields natural gaits. This reframes the value proposition: the contribution is less "hierarchical RL" and more "structuring the action space via a MoCap prior enables natural behavior without task-specific reward engineering."

## Suggestions

- Reframe the main claim from "no reward design or skill primitives" to "no per-task reward engineering" and acknowledge the MoCap tracking reward explicitly in the framing. This makes the contribution more precise and harder to dismiss.
- Add at least one more baseline on the naturalness comparison (e.g., a variant with an adversarial motion prior added to TD-MPC2's reward) to disentangle the MoCam prior's contribution from the hierarchical design's contribution.
- Vary k in at least one ablation to empirically characterize the role of temporal abstraction in this architecture.
- Provide the explicit formula for the termination-aware value truncation during planning in an appendix to aid reproducibility.

## Score and Decision

**Calibration**: I compared this paper against several related works:
- **H-GAP** (Accept spotlight, scores 8/8/6): Also uses MoCap data for 56-DoF humanoid control with MPC planning, achieves strong results, and introduces a generalist planner. Puppeteer similarly addresses 56-DoF humanoid control with MoCap data, but with a more restrictive evaluation of naturalness.
- **Universal Humanoid Motion Representations** (Accept spotlight, scores 8/8/8): Learns universal motion representations for humanoid control. Puppeteer has less novelty in the motion representation (uses standard end-effector targets) but adds visual observation which those papers don't.
- **FB-CPR** (Accept poster, scores 6/8/6/6): Zero-shot humanoid control with behavioral foundation models, moderate novelty but clear empirical contribution. Puppeteer has stronger naturalness evidence but weaker baseline comparisons.
- **HuWo** (Reject, scores 6/5/6/3): Humanoid locomotion with world model but limited contribution and questionable novelty. Puppeteer is clearly stronger—better method, better evaluation, new benchmark.
- **TD-MPC2** (Accept spotlight, scores 8/8/8/8): The backbone algorithm Puppeteer builds upon. Puppeteer's contribution is the hierarchical application, not the RL algorithm itself.

Puppeteer is a solid contribution that would benefit from more honest framing and stronger baselines for the visual tasks. It is clearly above papers like HuWo and HumanoidOlympics (which were rejected), and has a meaningful (if narrower than claimed) contribution. However, the overclaiming on "no reward design/skill primitives" and the thin baseline comparison for the visual tasks (the paper's main novelty) hold it below the level of H-GAP or Universal Humanoid Motion Representations. I place it in the range of FB-CPR (6/8/6/6, Accept poster)—a solid empirical contribution with useful insights but with notable framing and evaluation gaps.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>