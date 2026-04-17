Now let me carefully verify key claims from the reviewers against the actual paper text:

The harsh critic raises several important issues. Let me verify each one:

1. **Unfair baseline comparison on DexGraspNet**: The paper (Section 3.2) states: "Notably, the baseline methods do not randomize object initial positions, whereas our method is trained and tested with a large reset region of 50 cm × 50 cm." The paper then argues this makes DemoGrasp's setting harder. However, it also acknowledges translation invariance helps DemoGrasp. The comparison IS asymmetric, but the paper is transparent about this.

2. **Vision-based sim-to-real details**: The paper mentions "Flow-Matching (Lipman et al., 2022) policy with action chunking for imitation learning" and "finetune a pre-trained ViT (Dosovitskiy et al., 2021) encoder." Details are referenced to "Appendix E" but the appendix is not included in the provided text.

3. **Cross-embodiment training**: The paper says "we evaluate DemoGrasp on various robotic hands without hyperparameter tuning" and the results show per-hand results. It does not explicitly state whether it's per-hand training or joint.

4. **Single demonstration universality**: Table 9 tests 4 demonstrations (small/large × top/side). The paper claims "a single demonstration" which is literally true—you only need one—but the universality claim could be overstated.

5. **RL algorithm**: Not specified in the main text. Referenced to Appendix E presumably.

6. **The claim about "first to grasp small thin objects"**: This is stated as "to our knowledge, the first"—a qualified claim.

Let me now calibrate scores. Looking at comparison papers:
- ResDex (similar topic, DexGraspNet, sim-only): scores 6,8,6,8 → Avg ~7, Accept Poster
- BiDexHD (dexterous manipulation, teacher-student, no real robot validation of core claim): scores 5,5,5,6 → Avg ~5.25, Reject
- ThinShellLab (strong benchmark paper, real-world but limited): scores 8,8,8,8,8 → Accept Spotlight
- ManiBox (grasping, weak baselines, limited novelty): scores 3,6,8,6 → Avg ~5.75, Reject

DemoGrasp has a genuinely novel and elegant formulation, strong empirical results in simulation and real world, and broad evaluation. Its main weaknesses are: (1) the unfair DexGraspNet comparison, (2) some overclaiming around "universal" and "single demonstration," and (3) limited analysis of failure modes. These are important but don't undermine the core contribution—which is substantial.

## Summary

DemoGrasp reformulates universal dexterous grasping as a single-step MDP where an RL policy learns to edit a single demonstration trajectory (adjusting wrist SE(3) transform and hand joint angle deltas), then replays the edited trajectory in simulation. This drastically reduces exploration complexity and eliminates the need for complex reward shaping. A vision-based policy is subsequently trained via flow-matching imitation learning on successful rollouts for sim-to-real transfer. The method achieves 95% success on DexGraspNet, generalizes across six unseen datasets and multiple robotic hand embodiments, and demonstrates robust real-world grasping of 110 unseen objects including small/thin items.

## Strengths

- **Clean and effective problem reformulation**: The single-step MDP via demonstration editing (Eqs. 1–2) is a genuinely elegant insight that directly addresses the exploration challenge in high-dimensional dexterous grasping. Table 8 confirms that even replaying the unedited demonstration achieves 75.3% success, validating the demo as a strong prior.

- **Strong and broad empirical results**: 95.2% on DexGraspNet (state-based), 84.6% average success across six unseen datasets with six different hand embodiments, and 86.5% real-world success on 110 unseen objects. The 71.1% on small/thin objects and 95.3% on normal-sized objects are practical advances.

- **Simplicity of reward design**: The binary reward (Eq. 3) with collision-randomization trick is significantly simpler than prior methods' multi-term dense rewards, and the paper convincingly shows this simplicity stems from the single-step MDP formulation rather than being coincidental.

- **Cross-embodiment transfer without re-tuning**: The method applies to five-fingered hands, a three-fingered gripper, and a parallel gripper without hyperparameter tuning, demonstrating genuine generality.

- **Comprehensive ablations**: Table 8 (action space components), Table 9 (demonstration quality), and Table 7 (training set scaling) provide solid evidence for the method's robustness and scalability.

## Weaknesses

### Major:

- **Asymmetric baseline comparison on DexGraspNet**: In Table 1, DemoGrasp is trained and tested with 50cm × 50cm position randomization, while baselines (UniDexGrasp, UniDexGrasp++, UniGraspTransformer) are not. The paper argues this makes DemoGrasp's setting *harder*, but acknowledges that the demonstration-replay mechanism is translation-invariant, meaning spatial randomization actually *benefits* DemoGrasp's exploration. Without re-running baselines under identical conditions (or at least evaluating DemoGrasp *without* randomization), the headline 4–5% improvement over UniGraspTransformer cannot be confidently attributed to the method rather than the evaluation protocol. This does not invalidate DemoGrasp's strong performance, but it weakens the "state-of-the-art" claim on this benchmark specifically.

- **Limited exploitation of dexterous hand capabilities**: Table 8 shows that adding hand DoF editing to wrist-only editing yields only +2% improvement (94.22% → 96.24% on the training set), suggesting the method largely learns where to position a nearly fixed grasp shape rather than truly exploiting multi-finger dexterity. While the paper honestly reports this, it raises questions about the approach's scalability to tasks requiring diverse finger configurations (e.g., precision pinch, finger gaiting, in-hand manipulation). The "universal dexterous grasping" framing implies more dexterity utilization than the method actually achieves.

- **Absence of failure mode analysis**: The paper reports ~5% failure on DexGraspNet and ~15% on unseen datasets but never analyzes what object properties or grasp scenarios cause failure. Without understanding whether failures are systematic (e.g., specific geometries that the editing parameterization cannot handle) or random, the "universality" claim is not well supported. Is the ceiling imposed by the linear interpolation structure of the hand pose editing (Eq. 2), or by perception gaps, or something else?

### Minor:

- **Open-loop nature of the RL policy**: The single-step formulation commits to an entire trajectory edit before execution. The paper mentions "regrasp behaviors" in real-world deployment (Figure 7), but this comes from the vision-based imitation policy, not the RL policy itself. The paper does not clearly explain how mid-trajectory corrections work during deployment, and there is no analysis of how often the edited trajectory fails initially but is recovered.

- **Unspecified RL algorithm and key implementation details**: The paper does not state which RL algorithm is used (likely PPO, based on reference [Schulman et al., 2017]), nor key hyperparameters for it or the vision-based policy training. These details are deferred to Appendix E, which is not included. While not a fatal issue for a conference submission, it makes the core technical contribution harder to evaluate.

- **Cross-embodiment training regime is ambiguous**: It is unclear whether a separate policy is trained per hand embodiment or a single universal policy handles all. The language ("we evaluate DemoGrasp on various hands") suggests per-hand training, which would make the "universal across embodiments" framing less impactful—it is universal in the sense that the *method* transfers, not that a single policy handles all hands.

- **Object rescaling for Omni6DPose and ModelNet40**: These datasets contain larger objects that are randomly scaled to 6–15cm for testing. This rescaling likely reduces OOD-ness by making test objects more similar to training objects. The paper should acknowledge this or test with original scales.

## Nice-to-Haves

- **Failure mode analysis**: A brief analysis of failure cases (e.g., what object geometries fail, or whether failures cluster in certain shape categories) would strengthen the universality claim and guide future work.

- **Real-world comparison with prior work**: While challenging to implement, comparing with at least one prior sim-to-real dexterous grasping method on the same hardware would significantly strengthen the practical advantage claim.

- **Multi-step RL baseline with the same binary reward**: Showing that standard multi-step RL fails with the simple binary reward would more directly validate the core claim that the single-step formulation (not just the reward) enables effective learning.

## Removed Points

- **"Vision-based sim-to-real details are underspecified"**: The spark reviewer and harsh critic claim the vision-based pipeline is a "black box" with missing details. While the main text is brief on this (referencing Appendix E), the key design choices—flow-matching with action chunking, ViT encoder, domain randomization—are stated. For a conference paper, deferring implementation details to an appendix is standard practice. This is not a structural weakness but a presentation choice.

- **"No baseline comparison for the vision-based policy"**: The spark reviewer requests comparisons against diffusion policy, BC Gaussian, etc. This is a nice-to-have but not required. The paper's contribution is the RL formulation; the vision-based policy is a standard teacher-student distillation. Requesting a full ablation of imitation learning methods on the *learned* data is scope creep.

- **"Language-conditioned results are under-specified"**: Table 4 shows cluttered-scene and language-conditioned results in both sim and real. The language-conditioned result is a nice extension, not a core claim. Requesting full details on language encoding and instruction generation is beyond scope.

- **"Comparison with CMA-ES or Bayesian optimization"**: The spark reviewer suggests comparing the single-step RL with direct optimization methods. This is an interesting question but the single-step nature of RL here (essentially a policy mapping observations to actions) is different from black-box optimization—and RL naturally handles stochastic dynamics and generalizes across objects, which sampling-based methods would not do as efficiently.

- **"Real-world results limited to one embodiment"**: All real-world experiments use the Inspire Hand + FR3 arm. This is a valid observation, but the paper primarily validates the method's cross-embodiment generality in simulation. Real-world deployment for every embodiment is not realistic for a single paper.

- **"The collision-disabling hack lacks safety analysis"**: The paper clearly explains that randomly disabling collision detection in half the environments during training allows minimal table contact for flat objects, and the reward structure encourages collision-free grasps (reward=1) over collision-tolerant ones (reward=0.5). Testing in the real world shows collision-free grasps for normal objects, and appropriate contact for thin objects. The safety concern is valid in principle but the empirical results address it directly.

- **"Success criterion is lenient"**: The paper uses a 10cm lift threshold and hand-keypoint distance <12cm. While reasonable to request a stricter criterion, the real-world evaluation adds a hold-for-2-seconds requirement, which is more stringent than the simulation criterion. This is standard for grasping benchmarks.

- **"Overclaiming 'first' to grasp small thin objects"**: The claim "to our knowledge, the first" is properly qualified with a hedge. It is a reasonable claim given the paper's experimental evidence, not an unsubstantiated absolute.

## Novel Insights

The key insight—that a single successful grasp demonstration encodes sufficient structure (approach trajectory, close-and-lift dynamics) to serve as a strong prior for universal grasping via parameterized editing—deserves emphasis. The empirical confirmation in Table 8 that even unedited replay achieves 75.3% (with wrist translation only) reveals just how much of the grasping problem is solved by the demonstration prior, and how much remains for RL (primarily SE(3) positioning rather than finger coordination). This reframes universal dexterous grasping as primarily an "approach-and-position" problem rather than a fulltrajectory learning problem, which has broader implications for how we think about dexterous manipulation skill acquisition.

## Suggestions

- Re-run the DexGraspNet evaluation without spatial randomization (or provide an additional column with this setting) to enable a direct apples-to-apples comparison with published baselines. Even a single sanity check would strengthen the SOTA claim substantially.

- Add 1–2 paragraphs of failure mode analysis: categorize the ~5% DexGraspNet failures and ~15% unseen-dataset failures by object geometry or grasp type, and discuss which failures stem from the editing parameterization vs. other factors.

- Clarify in the text whether cross-embodiment results come from per-hand or joint training, and whether each hand requires its own demonstration trajectory.

- Moderate the language: replace "universal" with "broadly generalizing" or similar in the abstract and introduction, given the limited dexterity exploitation and tabletop-only evaluation.

## Score and Decision

Calibration:
- **ResDex** (similar topic, DexGraspNet SOTA, simulation-only, no real robot, accepted as poster at ~7): DemoGrasp is clearly stronger—real-world experiments, simpler method, broader evaluation.
- **DexTrack** (dexterous manipulation, teacher-student, real robot but less robust, accepted as poster at ~6.75): DemoGrasp has stronger empirical results and a cleaner formulation.
- **BiDexHD** (rejected at ~5.25): Much weaker—no real experiments, jerky policies, limited novelty.
- **ThinShellLab** (accepted spotlight at 8): DemoGrasp is somewhat less polished as a systems contribution but has comparable empirical breadth.

DemoGrasp has a genuinely novel and effective formulation, extensive simulation experiments, real-world validation on 110 objects, and compelling cross-embodiment transfer. The weaknesses (asymmetric baseline comparison, limited dexterity utilization, missing failure analysis) are real but do not undermine the core contribution. The paper is above the acceptance bar but has sufficient issues to prevent a very high score.

MY FINAL SCORE: <pineapple>7</pineapple>
MY FINAL DECISION: <orange>Accept</orange>