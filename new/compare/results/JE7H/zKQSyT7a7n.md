---
job_id: cb753cc8-8c08-4bf0-a051-1d2dc4f16163
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: zKQSyT7a7n.pdf
paper: Visuo-Tactile World Models
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about multimodal representation learning and action‑conditioned world models for robotic planning, which is fully aligned with ICLR’s scope (representation learning, generative models, robotics, planning).

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion/Conclusion). The work is methodologically nontrivial, written in clear English, and provides both quantitative and qualitative experiments on real robots. I see no fatal theoretical or experimental flaw that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any instructions targeting automated reviewers, hidden prompts, or other manipulative content in the provided main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces a multi‑task Visuo‑Tactile World Model (VT‑WM) that augments a vision‑only latent world model with tactile embeddings from Digit 360 fingertip sensors. The model uses pretrained encoders (Cosmos for RGB, Sparsh‑X for touch) and a transformer‑based action‑conditioned predictor trained with a mix of teacher‑forcing and short‑horizon autoregressive losses to forecast future visual and tactile latents. Experiments on several real‑robot, contact‑rich manipulation tasks show improved imagination quality (object permanence and causal compliance) and higher zero‑shot planning success rates compared to a vision‑only world model, as well as better data efficiency than a behavioral cloning baseline on a new insertion task.

## Strengths

1. **Clear and timely problem: contact‑aware world models for real robots.**  
   The paper targets a genuine pain point in current video‑based world models for robotics: hallucinations and contact ambiguity in occluded or contact‑rich scenes. The examples in **Figure 1** and **Figure 5** concretely illustrate common failure modes of vision‑only models (object disappearance, teleportation, severe deformations) and how tactile grounding mitigates them. This is well motivated and highly relevant.

2. **Solid, realistic experimental setup with a real dexterous hand and tactile sensors.**  
   Using an Allegro hand with four Digit 360 sensors on a Franka arm, plus an exocentric RGB camera, is ambitious and much closer to “proper” manipulation than simple parallel‑jaw setups. The multitask dataset (8 contact‑rich tasks, **Figure 9**, Appendix A.0.1) and the fact that models are trained on both successes and failures provide a convincing testbed for evaluating world models beyond toy domains.

3. **Well‑defined metrics for “world model physics” grounded in an external benchmark.**  
   The use of components of the World Consistency Score (Rakheja et al., 2025) to evaluate object permanence and causal compliance is a nice step toward principled evaluation. **Figure 4** (object permanence) and **Figure 6** (causal compliance) are not just cherry‑picked anecdotes; they show normalized Fréchet distances with 95% confidence intervals and paired t‑tests across five tasks. The consistent ∼33% and ∼29% relative improvements give quantitative backing to the claim that VT‑WM is more physically coherent.

4. **Contact‑aware world models improve real‑robot planning performance.**  
   The zero‑shot CEM‑based planning experiments in **Figure 8 (left)** are a strong point. Both VT‑WM and V‑WM achieve 100% success on reach‑only, kinematic tasks, but VT‑WM yields substantial gains on contact‑rich tasks: +10% (push fruits), +35% (reach & push), +31% (wipe cloth), +11% (stack cubes). The paired qualitative rollouts vs. real executions in **Figures 14–18** support the quantitative bar chart and illustrate *why* VT‑WM helps (e.g., establishing and maintaining contact, preserving object geometry).

5. **Data efficiency story compared to a strong BC baseline.**  
   The “place plate in dish rack” experiment in **Section 4.3** and **Figure 8 (right)** is a nice touch: with only 20 demos, fine‑tuned VT‑WM + CEM achieves 77% success vs. 22% for an ACT‑style BC policy. This speaks to the value of a multi‑task world model that reuses priors about contact dynamics, instead of learning a policy from scratch.

6. **Architecture and training are clear enough to reproduce at a high level.**  
   **Figure 3** gives a reasonably interpretable diagram of the Cosmos / Sparsh‑X encoders feeding a shared transformer with factorized spatio‑temporal attention and action cross‑attention. The loss formulation in **Equations (1)–(2)**, with explicit teacher‑forcing and sampled‑trajectory components, is straightforward and makes the training objective easy to understand.

7. **Qualitative visuo‑tactile rollouts convincingly show multimodal consistency.**  
   The reconstructions in **Figures 11–13** and the action‑controllability visualizations in **Figure 10** show that vision and touch latents evolve consistently under commanded motions and under ground‑truth action sequences. In particular, **Figure 13** (cube stacking) and **Figure 12** (table‑leg insertion) make it evident that VT‑WM not only predicts visual futures but also tactile contact patterns that align with which fingers are actually in contact.

## Weaknesses

1. **Positioning vs prior visuo‑tactile world or dynamics models is incomplete.**  
   The Related Work mostly contrasts to vision‑only world models and standalone tactile encoders. It mentions only one prior joint vision‑touch dynamics work (Zhang & Demiris, 2023) and then claims “little work on training world models with vision and touch”. Recent visuo‑tactile modeling papers that are directly on point are missing, for example:

   - Visuo‑tactile world modeling or predictive dynamics for contact‑rich tasks (e.g., OmniVTA‑style frameworks).  
   - ViTaSCOPE‑type implicit visuo‑tactile representations for object pose/contact estimation.  
   - Visuo‑tactile dynamics of deformable objects (e.g., VIRDO++‑style work).

   These are not obscure niche works; they are exactly in the visuo‑tactile predictive modeling space. Without comparing or even citing them, the claimed “first multi‑task visuo‑tactile world model” is overstated. This matters because it is unclear whether the main novelty is (i) contact‑aware world modeling per se or (ii) doing this at scale with foundation encoders and multi‑task data. The paper really needs to delineate what is new beyond “we plug Cosmos + Sparsh‑X into a transformer dynamics model and do CEM.”

2. **Core evaluation metrics rely heavily on a somewhat opaque pipeline; important details are missing.**  
   The object permanence and causal compliance metrics in **Section 4.1** are defined via CoTracker trajectories and normalized Fréchet distances, but several key details are unspecified:

   - How are keypoints chosen on each object? Fixed grid, manual selection, or automatic sampling?  
   - How is tracking through heavy occlusion handled? CoTracker will often either drop tracks or drift when the object disappears behind the hand, which could differentially favor one model depending on appearance changes.  
   - For object permanence, are lost tracks penalized as large Fréchet distances, or are they simply shorter trajectories? Is any re‑initialization or matching strategy used?

   Because these metrics underpin the central claims (33% and 29% improvements shown in **Figures 4 and 6**), the ambiguity weakens the evidential strength. This is not a trivial quibble; metrics that are sensitive to tracking failures can easily conflate visual fidelity with tracker robustness.

3. **Ablation and analysis of the tactile contribution are thin.**  
   The comparison is almost entirely between “V‑WM” (vision only) and “VT‑WM” (vision + touch). However, the architecture changes slightly when moving to VT‑WM (more tokens, different multimodal fusion). Several questions are left unexplored:

   - What happens if you give the model synthetic or noisy tactile channels? Does performance degrade gracefully, or does the model over‑rely on touch?  
   - Is the benefit mainly from additional tokens (capacity/regularization effect) or from genuinely different information? An ablation where tactile tokens are replaced by extra visual tokens or by random vectors would be very helpful.  
   - The tactile encoder Sparsh‑X is fine‑tuned, while Cosmos is frozen (Appendix A.1). Is some of the gain simply because one modality is actively adapted? A “frozen Sparsh‑X” ablation could clarify this.

   Without these, it is hard to isolate *how much* of the improvement is really about tactile information vs. other confounders in architecture and training.

4. **Planning setup and baselines are minimal, and CEM details are tuned somewhat informally.**  
   Planning uses a fairly simple latent‑L2 cost to the final visual latent and standard CEM, but:

   - No comparison is made to learning a goal‑conditioned policy (e.g., training ACT or another BC method to be goal‑conditioned), nor to using MPC with a classic dynamics model. The only planning baseline is “same CEM but with V‑WM”. While that is the central comparison, the lack of any more traditional controller weakens the general conclusion that VT‑WM is an effective planning substrate rather than merely “less bad than V‑WM”.  
   - **Algorithm 1** provides some hyperparameters (H=2s, P=36, N=10), but the paper does not discuss sensitivity: are the success rates in **Figure 8** robust to horizon length or population size? Were these tuned on the test tasks?  
   - The planner operates open‑loop in 2s chunks at relatively low control frequency, which makes it vulnerable to modeling error and unmodeled disturbances. This is acknowledged in Limitations but not quantified. It is unclear if a simple closed‑loop BC policy using *the same amount of data* as VT‑WM (multi‑task) would be competitive.

   While these do not invalidate the experimental findings, they limit the strength of the planning claims.

5. **Limited task diversity and generalization scope.**  
   All tasks are on a single tabletop scene with the same few objects and a single robot platform. The evaluation tasks in **Figure 8 (left)** and qualitative figures **14–18** are essentially the same families as the training dataset in **Figure 9**. The new “plate in dish rack” task in **Section 4.3** is certainly useful, but still uses the same robot, table, and sensing setup. No experiment tests transfer to novel objects with significantly different shapes or materials, or to tasks outside the contact regimes seen in training (e.g., deformable garments, compliant assemblies). Given that the abstract and introduction pitch VT‑WM as a “multi‑task visuo‑tactile world model,” the empirical story is still fairly narrow.

6. **Some mathematical and notational inconsistencies around the dynamics model.**  
   There are a few places where the notation is either inconsistent or misleading:

   - In **Section 3.2.1**, the predictor is described as estimating next‑step states with $(s_{k+1}, t_{k+1}) \sim P_\phi(s_k, t_k \mid a_k)$. This ignores conditioning on past states and actions, and the arguments to the conditional are reversed; presumably the intended form is  
     \[
       (s_{k+1}, t_{k+1}) \sim P_\phi(s_{k+1}, t_{k+1} \mid s_{1:k}, t_{1:k}, a_{1:k}).
     \]  
     or at least \(P_\phi(s_{k+1}, t_{k+1} \mid s_k, t_k, a_k)\). This may seem cosmetic, but for a world model paper the precise Markov vs. non‑Markov structure matters.
   - **Equations (1) and (2)** define \(L_{\text{teacher}}\) and \(L_{\text{sampling}}\) as simple L1 losses on latents. However, the text says the sampling loss uses H autoregressive steps, yet the summation index is ambiguous: in (2) the target \(s_{k+1}\) is written as if it comes from the original sequence of length \(T\), whereas during sampling the time indices differ. Clarifying the exact indices and whether the loss is normalized by T or H would help assess training stability and the relative impact of teacher vs. sampled terms.
   - There is no explicit statement of how the action chunks (grouped 5× at 30Hz, Section 3.2.2) align with the 6Hz vision and 0.16s tactile windows. It would be helpful to write the full mapping from continuous‑time actions to discrete indices in the equations.

   These issues do not suggest a fatal flaw but do signal that the mathematical formulation is not as tight as it could be.

7. **Quantitative reporting is visually focused; explicit tables would aid interpretation and reproducibility.**  
   Most key results are in bar plots (**Figures 4, 6, 8**), often with only relative improvements stated in text. There are no numerical results tables summarizing exact mean ± std/CI values or p‑values. For example, **Figure 8 (left)** shows success rates as bars, but precise success counts and confidence intervals are not tabulated, and **Figure 4** & **Figure 6** supply units on the axis but no numeric breakdown per task in tabular form. Having a results table with the exact normalized Fréchet distances, t‑statistics, and p‑values per task would substantially improve interpretability and facilitate comparison by future work. At present, readers must visually approximate from the plots.

8. **Task‑specific engineering choices are somewhat under‑discussed.**  
   Several nontrivial practical decisions could influence results but are only briefly mentioned:

   - Tactile input consists of 0.16s windows, two frames per sensor at 30–60Hz; why this exact horizon and sampling rate?  
   - Cosmos is kept frozen while Sparsh‑X is fine‑tuned (Appendix A.1); was freezing vision encoder necessary or just convenient?  
   - CEM cost is a pure L2 distance in vision latent space with no explicit tactile term; this seems at odds with the emphasis on tactile grounding. The paper argues that touch acts indirectly via improved rollouts, but some empirical evidence (e.g., comparing to a variant that uses tactile goal embeddings or a multi‑step cost) would strengthen the story.

   These are not “bugs” but make it harder to disentangle what aspects of the system are generic vs. heavily tuned to this particular setup.

## Potentially Missing Related Work

1. **Zheng, Y., Gu, S., Li, W., “OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation,” 2026.**  
   This work appears to address visuo‑tactile world modeling for contact‑rich manipulation in a very similar spirit. It should be discussed in **Section 2 (Related Works)** as a directly comparable visuo‑tactile world model, and the experimental section should at least conceptually compare methodological differences (e.g., architecture, training objective, planning usage).

2. **Lee, J., Fazeli, N., “ViTaSCOPE: Visuo-Tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation,” 2025.**  
   While focused on in‑hand estimation rather than general world models, this paper introduces an implicit visuo‑tactile representation specifically designed to encode contact and pose. It is highly relevant to **Section 3.1** and **3.2.1**, where VT‑WM is argued to capture object pose and contact state from touch; it should be cited and contrasted there.

3. **Wi, Y., Zeng, A., Florence, P., “VIRDO++: Real-World, Visuo-tactile Dynamics and Perception of Deformable Objects,” 2022.**  
   This work deals with visuo‑tactile dynamics and perception for deformable objects in the real world. It is relevant as a visuo‑tactile dynamics model and should be discussed in **Section 2** as prior work on multimodal dynamics modeling, even though VT‑WM focuses on rigid and semi‑rigid objects. A short discussion of how VT‑WM could or could not extend to deformables would be valuable.

## Questions

1. **Details of the CoTracker‑based metrics.**  
   Could the authors clarify exactly how keypoints are initialized and tracked for the object permanence and causal compliance metrics? Specifically:  
   - How many keypoints per object and how are they selected?  
   - How are occlusions handled (e.g., track termination, interpolation, re‑initialization)?  
   - How are differing sequence lengths or dropped tracks reconciled when computing normalized Fréchet distances?  
   A precise description or pseudocode would increase confidence that the reported ∼33% and ∼29% gains are robust to tracking artifacts.

2. **Quantitative ablations on tactile contribution.**  
   Are there any experiments where tactile tokens are replaced by random noise, frozen Sparsh‑X features, or additional visual tokens with matched dimensionality? This would help rule out confounding factors (e.g., benefits from more tokens or more training parameters) and better isolate the true information gain from tactile sensing.

3. **Sensitivity of planning performance to CEM hyperparameters and horizon.**  
   Have the authors explored different horizons (e.g., 1s vs 2s vs 3s) or particle counts in **Algorithm 1**? Do the relative gains of VT‑WM over V‑WM in **Figure 8 (left)** remain consistent if you change these settings, or are the numbers sensitive to tuning?

4. **Why not incorporate tactile into the planning cost?**  
   The paper currently uses an L2 cost only on the *visual* latent, while tactile is used purely for improving rollouts. Have you tried augmenting the cost with a term involving tactile latents, for goals where the desired contact state is known (e.g., grasped vs. not‑grasped)? If so, how did it affect planning quality and robustness?

5. **Generalization to novel objects / scenes.**  
   Do the authors have any preliminary evidence on how VT‑WM behaves when the same tasks are performed with different objects (e.g., differently shaped fruits, a different cube, or another cloth)? Even a qualitative example would help understand whether the tactile grounding learned here is specific to the current objects or somewhat object‑agnostic.

6. **Data efficiency vs multi‑task BC.**  
   In **Section 4.3**, VT‑WM is compared to a single‑task BC policy trained only on plate‑insertion demos. Do the authors have any insight into how a multi‑task BC model trained on the same 8 tasks + 20 new demos would perform? This is mentioned in Limitations but seems central to the “data efficiency” claim.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The technical approach (latent dynamics model with visuo‑tactile inputs, mixed teacher‑forcing and sampling losses, CEM‑based planning) is standard but competently executed, and the real‑robot experiments are generally well designed. Some methodological details (metrics pipeline, ablations, exact conditional structure) are under‑specified, and the evaluation remains narrow in scope, but I do not see fatal flaws.

## Presentation Rating

3: good.  
The paper is generally clear, with well‑chosen figures (especially **Figures 1, 3–8, 10–18**) and a logical structure. A few mathematical notational issues and missing experimental details detract slightly from clarity, and the Related Work needs better coverage of visuo‑tactile modeling literature.

## Contribution Rating

3: good.  
The contribution is meaningful and useful: a multi‑task visuo‑tactile world model that concretely improves object permanence, causal compliance, and planning success rates on a real robot. The conceptual novelty is moderate (combining strong pretrained encoders with a transformer world model), but the empirical demonstration in real contact‑rich tasks and the data‑efficiency comparison with BC are valuable to the community.

## Overall Rating

6: marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper presents a solid and timely integration of tactile sensing into latent world models, supported by both quantitative and qualitative real‑robot results, and it addresses a real deficiency of current vision‑only models. At the same time, the positioning vs. closely related visuo‑tactile dynamics work is incomplete, ablations are limited, and the evaluation scope is somewhat narrow. I lean to a positive recommendation because the empirical evidence for better contact‑aware imagination and planning is convincing, but I expect significant revision in related work discussion and analysis to reach the standard implied by the current claims.

## Reviewer Confidence

4: confident.  
I am familiar with world models, visuo‑tactile sensing, and model‑based planning, and I have carefully read the equations, figures, and experimental descriptions. Some implementation details (e.g., exact CoTracker setup, tuning of CEM) are not fully specified, so I leave a bit of room for clarification, but I am reasonably confident in the assessment above.