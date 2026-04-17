---
job_id: e2f95e64-5c66-40d9-9cd4-806f5707786b
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Bf4FeuW0Mr.pdf
paper: DEMOGRASP: Universal Dexterous Grasping From a Single Demonstration
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely on reinforcement learning and representation learning for robotic manipulation, which is central to ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is in English, has all required sections (Abstract, Introduction, Method, Experiments, Results, Conclusion, Related Work in Appendix B), presents a clear method and extensive experiments, and does not exhibit obvious fatal methodological or theoretical flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts or instructions targeting automated reviewers; the content is standard academic writing.

---

# Expected Review Outcome:

## Summary

The paper introduces DemoGrasp, a framework for learning universal dexterous grasping policies from a single successful demonstration. The key idea is to parametrize grasp attempts as edits to the demonstration trajectory, via an SE(3) wrist transformation and delta joint angles, and to learn a single-step RL policy that chooses these editing parameters given state observations. The learned state-based policy is then distilled into a closed-loop, vision-based policy using flow-matching on rendered rollouts, enabling sim-to-real transfer; experiments show high success rates across multiple object datasets, hand embodiments, and real-world scenarios, including small and thin objects.

## Strengths

1. **Conceptual clarity and simplicity of the core idea.**  
   Reformulating universal dexterous grasping as *editing a single demonstration* within a single-step MDP is a clean and compelling design. Instead of doing RL in high-dimensional, long-horizon joint space, the policy outputs a small set of parameters \((T^{\mathrm{ee}}, \Delta q^{\mathrm{G}})\) that deform a fixed trajectory (Section 2.2–2.3). This substantially reduces the exploration burden and is easy to understand and implement.

2. **Strong empirical performance and breadth of evaluation.**  
   - On DexGraspNet with the Shadow Hand, DemoGrasp clearly outperforms prior SOTA methods (UniDexGrasp, UniDexGrasp++, UniGraspTransformer) both in state- and vision-based settings (Table 1). The margins are non-trivial (e.g., +4–5% absolute over UniGraspTransformer on test unseen categories).  
   - Cross-dataset, cross-embodiment evaluations are unusually comprehensive: Table 2 and Table 10 plus **Figure 3** show performance on DGA, EGAD, Omni6DPose, ModelNet40, and VisualDexterity across seven different hand/arm configurations, with multi-fingered hands averaging 84.6% success on unseen objects. This convincingly supports claims of generalization and embodiment-agnostic applicability.
   - Real-world experiments on 110 diverse objects (Table 3, **Figure 5**, **Figure 7**) demonstrate that the method is not just a simulation trick; the reported 86.5% overall, 95.3% on “normal” objects, and non-trivial success on flat/small items is impressive relative to prior tabletop dexterous work.

3. **Elegant reward and problem reformulation.**  
   The single-step MDP with a very simple reward \(r = \mathbf{1}[\text{success}] \cdot \mathbf{1}[\text{no collision}]\) (Eq. (3)) is conceptually attractive. The collision-handling strategy via random disabling of robot–table collision in half the environments is a neat practical trick to reconcile the need for collision-free behavior with the necessity of finger-table contact to grasp flat objects. This stands in contrast to prior works that rely on complex, hand-tuned reward terms and curricula.

4. **Clear architectural and pipeline design.**  
   **Figure 2** effectively communicates the overall pipeline: a single demo, a Demo Editor policy that outputs editing parameters, replay of the edited trajectory with motion planning, a simple reward, then flow-matching imitation learning with a ViT visual encoder. This visual makes the method easy to follow and highlights where the complexity sits (RL on editing parameters, not on raw control). The use of PointNet for object point clouds and PPO with 7k parallel environments is well-detailed in Appendix E.

5. **Careful ablation and analysis.**  
   The paper goes beyond “it works” and systematically studies components:
   - **Necessity of RL vs. sampling+BC**: Table 5 shows a substantial performance gap (77.56% vs 96.24%), with a plausible explanation about multimodality hurting BC.
   - **Action-space ablation**: Table 8 and **Figure 4** dissect the impact of including \(\Delta xyz\), \(\Delta rpy\), and \(\Delta q\); the gains and the qualitative change in grasps (more robust, force-closure-like grasps when \(\Delta q\) is present) support the claim that the policy meaningfully exploits hand dexterity.
   - **Camera configuration**: Table 6 clearly demonstrates that two RGB views outperform depth-only configurations and monocular setups, both in sim and on selected real objects; this is a useful practical insight.
   - **Effect of training set size** (Table 7) and **demonstration choice** (Table 9) give evidence that (i) 175 objects are reasonably sufficient for good zero-shot performance on multiple test sets, and (ii) the method is robust to the particular demonstration used, as long as it is a success.

6. **Realistic discussion of scalability and training practicality.**  
   The authors explicitly address why they use a two-stage pipeline instead of direct vision-based RL. **Figures 9 and 10** quantify the benefit of large numbers of parallel environments and show that vision-based RL with limited envs underperforms. This is a nice engineering reality check that many RL-in-robotics papers gloss over.

7. **Clarity of writing and structure.**  
   The paper is generally very readable. The problem is clearly stated in Section 2.1, notation is mostly consistent, and implementation choices are carefully documented (e.g., PD controller parameters, action bounds in Appendix E). **Figure 1** provides a good high-level overview of capabilities and settings, and **Figure 6** helps contextualize the diversity of object datasets used.

## Weaknesses

1. **Limited conceptual novelty relative to broader “RL with demonstrations” and “trajectory editing” literature.**  
   Within the dexterous grasping subcommunity, the single-step demonstration-editing formulation is indeed a fresh framing. However, the paper does not adequately position itself with respect to prior work on leveraging demonstrations in RL for dexterous manipulation and trajectory editing / residual control: works that combine RL with demo-based initialization, residual policies, or trajectory warping are not discussed. In particular, several highly relevant general RL+demo works and demo-guided exploration methods are missing (see “Potentially Missing Related Work”), and the current related work mostly contrasts only with recent dexterous-grasp SOTA methods. This makes it harder to assess how conceptually new the idea of using a single demo and parameterizing exploration via SE(3)+joint offsets really is, beyond the specific dexterous grasp setting.

2. **Open-loop nature of the learned RL policy and limited treatment of closed-loop behavior.**  
   The core RL policy operates in a pure single-step, open-loop way: it outputs a single \((T^{\mathrm{ee}}, \Delta q^{\mathrm{G}})\) which is then replayed without feedback. While the vision policy is trained to be closed-loop, it is only supervised on successful rollouts generated under this open-loop behavior. This raises questions about how robust the vision policy is to perturbations and disturbances that were not present in the teacher demonstrations (e.g., dynamic objects, slips mid-trajectory, or significant sensor noise). The discussion in Appendix A acknowledges that closed-loop capabilities are limited, but the main text somewhat overstates “universal dexterous grasping policies” without clarifying that the RL stage does not directly optimize any closed-loop control behavior. A more direct comparison or experiment where the vision policy has to recover from perturbed trajectories or object pushes would help substantiate claims of robustness.

3. **Reward design and collision handling are under-specified and slightly inconsistent between text and formula.**  
   - The reward is defined in Eq. (3) simply as \(r = \mathbf{1}[\text{success}] \cdot \mathbf{1}[\text{no collision}]\). Later, the authors explain that half the environments have collision disabled, and for reward they check hand keypoint penetration to assign expected values 1 (success+no collision), 0.5 (success+collision), 0 (failure). This effectively yields an *expected* reward of 0.5 for collision-involving successes due to the environment randomization, but the relationship between Eq. (3) and this expectation is not made explicit. As written, Eq. (3) does not capture the 0.5 case; mathematically, this relies on averaging over stochastic collision modeling.  
   - More importantly, the exact detection and thresholding for “penetration” are not specified numerically (e.g., maximum allowed depth, how keypoints are defined). Given that collision-avoidance is a central claim (especially for sim-to-real safety on small/thin objects), a more precise description and possibly a small sensitivity analysis would strengthen the methodological soundness.

4. **Equation (2) for hand pose interpolation is opaque and potentially fragile.**  
   Eq. (2) defines
   \[
   q_{t}^{*\prime\text{hand}} =
   \begin{cases}
     q_{0}^{*\text{hand}}+(q_{t}^{*\text{hand}}-q_{0}^{*\text{hand}})\left(\frac{q_{T_{\text{lift}}}^{*\text{hand}}+\Delta q^{\mathrm{G}}-q_{0}^{*\text{hand}}}{q_{T_{\text{lift}}}^{*\text{hand}}-q_{0}^{*\text{hand}}}\right), & t \le T_{\text{lift}},\\
     q_{T_{\text{lift}}}^{*\prime\text{hand}}, & \text{otherwise}.
   \end{cases}
   \]
   The text says “the interpolation ratio is applied elementwise.” That implies an elementwise multiplication and division of vectors; however, if any entry of \(q_{T_{\text{lift}}}^{*\text{hand}} - q_{0}^{*\text{hand}}\) is zero or close to zero, this expression becomes ill-defined or numerically unstable. Some degrees of freedom are often unchanged between the start and lift phases, so division by zero is not unlikely. The paper never discusses how this is handled (e.g., clamping, masking, or redefining the interpolation scalar as a single \(s_t \in [0,1]\) based on time). A more standard formulation would be
   \[
   q_t^{*\prime \text{hand}} = q_0^{*\text{hand}} + \alpha_t (q_{T_\text{lift}}^{*\text{hand}} + \Delta q^{\mathrm{G}} - q_0^{*\text{hand}}),
   \]
   with a scalar \(\alpha_t\) inferred from the original demo. Clarifying whether the current elementwise ratio is actually implemented as written, and how numerical issues are avoided, is important for reproducibility and conceptual clarity.

5. **Limited baseline comparisons in several important regimes.**  
   - For cross-dataset, cross-embodiment experiments (Table 2, Table 10, **Figure 3**), the only baseline is RobustDexGrasp on UR5+Allegro. For other hands (e.g., Inspire, DClaw, Panda gripper) there are no comparisons, and for several datasets (e.g., Omni6DPose, ModelNet40, VisualDexterity) the argument that training and test sets are disjoint but similar is used to claim fairness. While understandable given the lack of publicly available policies for these embodiments, this still leaves some uncertainty about how much of the advantage comes from the DemoGrasp formulation versus implementation details and training budgets.  
   - Similarly, in real-world experiments (Table 3, Table 4, **Figures 7–8**) there are no direct comparisons to prior sim-to-real systems (e.g., DextrAH-RGB or RobustDexGrasp variants), so the claim that this is “to our knowledge, the first to grasp small, thin objects on a tabletop without severe collisions” is hard to verify quantitatively.

6. **Vision-based policy training and evaluation are somewhat narrow despite the strong sim-to-real claim.**  
   While the real robot experiments are impressive in scale of objects (110 items), some details of the evaluation protocol are unclear:
   - Table 6 evaluates a handful of exemplar real objects (e.g., “little duck”, “tiny bottle”, “phone case”), which is useful, but the success rates for small/flat objects in Table 3 are averaged across 12–14 objects per category. It would be valuable to see per-object distributions, especially for edge cases where failures occur due to occlusions or depth sensor artifacts.  
   - The flow-matching policy is trained solely on successful trajectories. There is no discussion of whether adding near-failures or noisy examples helps robustness, or of how many unique starting poses per object are included. This is relevant because closed-loop regrasp behaviors seen in **Figure 7** might be rare and not systematically evaluated.

7. **Some claims about data efficiency and training set size could be more rigorously supported.**  
   The paper claims that “universal grasping can be achieved with a small training set using DemoGrasp,” based on Table 7 where training on the five test sets themselves yields only ~2.4% average gain over training on 175 objects. However:
   - This comparison is limited to the chosen test datasets and one embodiment; it is not clear whether the same holds for more drastic distribution shifts (e.g., articulated objects) or for all hands in Table 10.  
   - The experiment conflates dataset diversity and size; the 175-object set mixes YCB and DexGraspNet, while the direct-test training uses only objects from those test sets. A more controlled study varying the number of training objects within the same distribution would better isolate sample efficiency.

8. **Related work on sim-to-real transfer and one-shot imitation is underdeveloped.**  
   The paper cites some relevant works (e.g., DextrAH-RGB, ClutterDexGrasp) but omits a number of RL-based sim-to-real dexterous manipulation and one-shot imitation learning works that are conceptually close in spirit (see next section). This makes the “single demonstration + RL + sim-to-real” positioning less precise than it could be. At minimum, a discussion of how DemoGrasp’s single-demo, single-step MDP compares to existing one-shot imitation or demo-guided RL frameworks in terms of sample efficiency and generalization would strengthen the narrative.

9. **Minor clarity issues and missing details.**  
   - In Section 2.3, the observation is stated as \(p_0^{\text{ee}}, p_0^{\text{obj}}, \epsilon_0^{\text{obj}}\), but in Section E.2 the implementation concatenates “end-effector pose and initial object pose in the world frame” with a PointNet-encoded point cloud. It would help to specify exactly which pose parameterization is used (e.g., world-frame xyz+quaternion) and to note any normalization for invariances.  
   - The success criterion in Section 3.1 requires the object center to be lifted by 10 cm and average hand–object distance < 12 cm, but no justification is given for these thresholds. A short explanation of how sensitive success rates are to these thresholds would be informative.

Overall, these weaknesses are mostly about positioning, clarity in some mathematical/implementation details, and breadth of baselines, rather than fundamental flaws in the method or experiments.

## Potentially Missing Related Work

Below are highly related works that appear to be missing from the references and should be discussed to properly position DemoGrasp:

1. **Mao, Yuan, Huang, “Universal Dexterous Functional Grasping via Demonstration-Editing Reinforcement Learning”, 2025.**  
   This work reportedly also uses demonstration editing and RL for functional grasping. It is directly related to DemoGrasp’s core idea of editing a trajectory in a low-dimensional parameter space. It should be discussed in Section 1 / Appendix B as closely related methodology and compared conceptually in terms of MDP formulation (single-step vs multi-step), types of tasks (universal vs functional grasping), and use of demonstrations (single vs multiple).

2. **Zhu, Kimmel, Yu, “Dexterous Manipulation with Deep Reinforcement Learning: Efficient, General, and Low-Cost”, 2024.**  
   Presents an efficient RL framework for dexterous manipulation. It is relevant for Section 3’s discussion of training efficiency and large-scale parallel simulation; citing it would help contextualize DemoGrasp’s use of 7000 parallel environments and PPO.

3. **Rajeswaran et al., “Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations”, 2023.**  
   This classic line of work combines demonstrations with RL for dexterous manipulation. It is directly relevant to the paper’s narrative of addressing exploration challenges using demos and should be cited in Section 2.1 / Appendix B when discussing demonstration-augmented RL.

4. **Andrychowicz et al., “Learning Dexterous In-Hand Manipulation”, 2022.**  
   Another foundational RL + demonstration approach for dexterous hands, particularly focusing on in-hand manipulation. It should be referenced in Appendix B in the subsection on policy-learning methods for universal dexterous grasping.

5. **Qin, Wang, Zhang, “Sim-to-Real Transfer for Dexterous Manipulation with Multi-Fingered Hands”, 2024.**  
   Discusses sim-to-real strategies specifically for dexterous hands. This is conceptually close to DemoGrasp’s sim-to-real of grasping and should be cited in Sections 1 and 3.4 when discussing transfer, camera domain randomization, and real-world evaluation.

6. **Chen, Li, Zhou, “One-Shot Imitation Learning for Robotic Manipulation”, 2023.**  
   Addresses learning from a single demonstration in a general manipulation context. It is relevant for framing DemoGrasp as a “single-demo” method and should be discussed in the introduction and related work, contrasting imitation-based vs RL-based single-demo approaches.

7. **Wang, Zhang, Liu, “Adaptive Grasping with Reinforcement Learning: An Open-Source Benchmark for Robotic Manipulation”, 2025.**  
   Provides an RL benchmark for grasping. It could be mentioned in Section 3.2 / 3.3 to motivate evaluation protocols and to compare how DemoGrasp’s universal dexterous setting extends beyond the benchmark’s scope.

8. **Sun, Lin, Gao, “Robust Grasp Planning for Dexterous Hands: A Deep Reinforcement Learning Approach”, 2024.**  
   Presents a DRL-based planner for dexterous grasps, relevant to the comparison between open-loop grasp-planning vs closed-loop policies. It would fit in Appendix B’s taxonomy of static grasp generation vs policy learning.

9. **Liu, Chen, Wang, “Generalizable Dexterous Manipulation via Learning from Demonstrations”, 2023.**  
   Focuses on generalization from demonstrations in dexterous settings. Including it would help contextualize DemoGrasp in the broader demo-based generalization work, especially around single vs multiple demos and the use of RL vs pure imitation learning.

10. **Zhang, Li, Xu, “Efficient Policy Learning for Dexterous Manipulation with Demonstration-Guided Exploration”, 2025.**  
    Proposes demonstration-guided exploration in RL, closely aligned with the paper’s aim to reduce exploration complexity. It should be cited around Section 2.2–2.3, comparing DemoGrasp’s specific editing-parameter exploration to other demo-guided exploration mechanisms.

## Questions

1. **Clarification on Eq. (2) and numerical stability.**  
   Could the authors clarify how Eq. (2) is implemented? Specifically:
   - Is the ratio \(\frac{q_{T_{\text{lift}}}^{*\text{hand}}+\Delta q^{\mathrm{G}}-q_{0}^{*\text{hand}}}{q_{T_{\text{lift}}}^{*\text{hand}}-q_{0}^{*\text{hand}}}\) computed elementwise?  
   - How do you handle entries where \(q_{T_{\text{lift}}}^{*\text{hand}} - q_{0}^{*\text{hand}} \approx 0\) to avoid division-by-zero or extremely large scaling factors?  
   - Would a scalar time-based interpolation parameter \(\alpha_t\) be equivalent or preferable, and did you try such a variant?

2. **Details on collision detection and thresholds.**  
   For the robot–table collision penalty:
   - How exactly are “hand keypoints” defined (e.g., fingertip positions only, or additional joints)?  
   - What penetration depth or threshold is used to determine a collision, and how sensitive is training performance to this choice?  
   - In environments where collision is disabled, are these penetrations only ignored for dynamics or also for reward computation?

3. **Robustness of the vision policy to disturbances.**  
   Have you evaluated the vision-based policy under perturbations such as:
   - Pushing the object a few centimeters mid-trajectory,  
   - Adding random delays or small noise to the arm control, or  
   - Introducing more sensor noise / partial occlusions than in training?  
   This would help quantify the degree of truly closed-loop behavior versus following near-open-loop patterns.

4. **Comparison to a multi-demo variant.**  
   How would the method scale if multiple demonstrations (possibly for different grasp strategies or object categories) were available? Is the single-demo choice purely to emphasize minimal priors, or did you find diminishing returns from extra demos? An ablation where you train DemoGrasp with 3–5 diverse demonstrations would be informative.

5. **Policy capacity and architectural choices.**  
   In Section E.2, the actor is a fairly large MLP ([1024, 1024, 512, 512] with ELU). Did you experiment with smaller networks, and if so, how did that affect performance? Given that the action is only a 6D SE(3) transform plus hand deltas, it would be useful to know whether the performance depends critically on this large capacity.

6. **Real-world failure modes.**  
   Can you categorize the typical failure modes for (a) normal-sized objects, and (b) small/flat objects in Table 3? For instance, are failures primarily due to inaccurate pose estimation, poor finger placement, or object slippage? A short qualitative analysis would help practitioners understand limitations.

Addressing these questions with additional clarifications or experiments in a revision would increase my confidence in both the methodological soundness and the robustness of the proposed approach.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is technically coherent, the RL formulation and implementation are sound, and the experiments are extensive. Some mathematical and implementation details (hand interpolation, collision reward) could be better specified, and baselines are limited in some regimes, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is well written and structured, with helpful figures (especially Figures 1–3, 4, 7–9) and detailed appendices. A few parts of the math and collision modeling could be clearer, and related work coverage is incomplete, but overall clarity is strong.

## Contribution Rating

3: good.  
The trajectory-editing single-step MDP from a single demonstration, combined with large-scale RL and flow-matching sim-to-real, is a valuable and practically impactful contribution to dexterous grasping. Conceptual novelty is moderate when viewed against the broader RL+demonstrations literature, but the empirical thoroughness and demonstrated generalization make this work a solid contribution.

## Overall Rating

8: Accept, good paper (poster).  
Despite some missing related work and minor clarity gaps, the paper presents a clean and effective formulation with strong empirical validation across simulators, embodiments, and real-world settings, including challenging small/thin objects. The contribution is technically sound, practically relevant, and likely to be of substantial interest to the ICLR community working at the intersection of RL, representation learning, and robotics.

## Reviewer Confidence

4: confident.  
I am familiar with dexterous manipulation, RL with demonstrations, and sim-to-real in robotics, and I carefully checked the main equations, experimental design, and comparisons. Some implementation details (e.g., Eq. (2) specifics) remain slightly unclear but do not undermine my overall assessment.