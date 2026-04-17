# Learning to Grasp Anything By Playing with Random Toys

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4, 4

## Abstract
Robotic manipulation policies often struggle to generalize to novel objects, limiting their real-world utility. In contrast, cognitive science suggests that children develop generalizable dexterous manipulation skills by mastering a small set of simple toys and then applying that knowledge to more complex items. Inspired by this, we study if similar generalization capabilities can also be achieved by robots. Our results indicate robots can learn generalizable grasping using randomly assembled objects that are composed from just four shape primitives: spheres, cuboids, cylinders, and rings. We show that training on these "toys" enables robust generalization to real-world objects, yielding strong zero-shot performance. Crucially, we find the key to this generalization is an object-centric visual representation induced by our proposed detection pooling mechanism. Evaluated in both simulation and on physical robots, our model achieves a 67% real-world grasping success rate on the YCB dataset, outperforming state-of-the-art approaches that rely on substantially more in-domain data. We further study how zero-shot generalization performance scales by varying the number and diversity of training toys and the demonstrations per toy. We believe this work offers a promising path to scalable and generalizable learning in robotic manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The main contribution of this paper consists of the introducing a data-efficient, object-centric learning framework—called LEGO (LEarning to Grasp from tOys)— that enables robots to generalize grasping skills to unseen real-world objects by training only on randomly assembled synthetic toys, built from only 4 primitive shapes. It gets its inspiration from an analogy between human cognitive development and robotic skill acquisition.
As many other papers in the field, it emphasizes the critical role of an object-centric perception model, through a detection pooling (DetPool) mechanism.

### Strengths
* Paper clearly written, well structured ans easy to understand, with good intuitive explanations
Elegant, data-efficient learning setup leveraging primitive-based object construction.
* The proposed method shows strong empirical performance with minimal data (the data efficiency clain is well demonstrated empirically).
* I appreciate the experimental coverage including some ablations and, more importantly, multiple robot platforms and real-world experiments!

### Weaknesses
* The approach is overall heuristic (based on an analogy with child development) and does not formally support that toy-based training will yield to generalizable features.
In robotic manipulation, it is well known that geometry is not enough: mass distribution, friction/contact forces, affordances, texture are also very important factors to deal with when we want to generalize to a wide class of objects. All these aspects are not sufficiently covered in the paper and it is likely to explain the drop in performance on complex or non-rigid objects.
* Dependence on accurate segmentation, as it is based on reliable object masks (obtained via SAM2). In real world settings, due to lightning conditions and other occlusion factors, this could not be the case.
* The Object-centric representation (DetPool) is not really new. It could be useful to add (and compare with) two recent papers: https://arxiv.org/abs/2505.11563 and https://arxiv.org/abs/2503.11565.

### Questions
In the experimental section, you are often referring to a pre-defined grid when testing objects multiple times. Could you give more information about that pre-defined grid?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies “cross-object generalization in grasp learning.” Inspired by developmental psychology, the authors train a grasping policy using only four geometric primitives (sphere, cube/rectangular prism, cylinder, and ring) that are randomly combined into “toys,” and achieve zero-shot generalization to real, unseen objects. The core idea is a “detection pooling” method called **DetPool** to build a **target-centric** visual representation: during visual encoding, a target segmentation mask constrains attention so object patches do not attend to background patches; the model then pools over the object patches to obtain a compact representation. A policy network takes this representation together with proprioception and predicts an action sequence (the overall framework is dubbed **LEGO**). Experiments show that training on only a few hundred random “toys” with limited demonstrations attains strong zero-shot grasp success on the real YCB object set. Compared to common pooling baselines (CLS token, mean pooling, attention pooling), DetPool yields markedly better generalization and scales well as data increases. Against several large vision-language-action baselines, LEGO is more robust under small-data and domain-shift settings; it also achieves leading or competitive success rates on a real Franka platform and a humanoid dexterous-hand platform, supporting the “target-centric representation + few demonstrations” path as a data-efficient solution.

Main contributions:

1. Proposes a “toy-driven” grasp learning paradigm and the **LEGO** framework, showing that random combinations of a few geometric primitives suffice to support generalization to complex real-world objects.
2. Introduces **DetPool**, which explicitly constructs a target-centric representation during visual encoding, significantly improving zero-shot and out-of-distribution generalization.
3. Provides systematic evaluation and scaling analyses in both simulation and real robots (including Franka and a dexterous-hand platform), demonstrating consistent gains across data scale, object diversity, and model size.

### Strengths
Originality
1.Proposes a “toy-driven” paradigm that composes a few geometric primitives to approximate real-world shape complexity, avoiding reliance on large annotated object corpora.
2.Introduces DetPool to build a target-centric representation during visual encoding (mask-constrained attention + pooling), offering a simple, effective object-centric alternative to CLS/mean/attention pooling.
Quality
Comprehensive evaluation: sim-to-real comparisons, strong VLA baselines, and ablations/scaling on pooling, data scale/diversity, and model size.
Clarity
 Coherent structure from motivation to analysis, clearly explaining why target-centric encoding and primitive-based training support generalization.
Significance
1. Demonstrates zero-shot grasping on complex real objects without heavy real-data/annotation costs, improving practical feasibility.
2.Provides concrete evidence for object-centric representations in embodied manipulation, nudging the field toward “structural priors + target-centric encoding.”
3. Cross-embodiment results suggest portability across manipulators/dexterous hands/mobile settings, indicating broad impact.

### Weaknesses
1. **Mask dependence and insufficient robustness**
   *Current setup:* GT masks in simulation, SAM2 in the real system; this may inflate DetPool’s gains.
   *Specific gaps:* No sensitivity curves of “mask quality → performance,” and no weak/no-prompt controls (box-only, no-mask).
   **Suggestion:** Use automatic (noisy) masks in both sim and real; plot performance vs IoU corruption, under/over-segmentation, and occlusion levels; add box-only and no-mask variants to quantify DetPool’s intrinsic benefit.

2. **Evaluation too idealized; extrapolation radius unclear**
   *Current setup:* Single-object, regular poses; the dexterous-hand setting uses only 13 objects with ~50% average success and high variance.
   *Specific gaps:* Missing stratified stress tests for clutter, multi-object scenes, heavy occlusion, look-alike distractors, with attribute-bucketed results.
   **Suggestion:** Build layered benchmarks and report attribute-bucketed success and failure attributions; provide “shift strength (viewpoint/material/scale) → success” curves to characterize the effective operating radius.

3. **Insufficient ablations; key claims are questionable**
   *Claim:* DetPool is the key to generalization and outperforms mean/attn/CLS pooling by 22–48%.
   *Specific gaps:*

   * All pooling variants share the same architecture/training recipe, but there’s no evidence they aren’t simply overfitting under limited data.
   * Missing **no-pooling** baseline using full-image features (full-image ViT).
   * No analysis of **SAM2** mask quality on performance (robustness to mask noise unclear).
   * SOTA baselines (e.g., OpenVLA-OFT 7B) overfit in small-data regimes; the paper doesn’t show LEGO remains superior at larger scales.
     **Suggestion:** Add data-regime sweeps (few→many), include a full-image no-pooling baseline, conduct mask-quality sensitivity analyses, and report matched-compute/parameter learning curves to verify whether DetPool’s advantage persists with more data.

### Questions
1. Authors use GT masks in simulation and SAM2 masks in the real setup. Could this amplify DetPool’s gains? Please unify to automatic (noisy) masks in both sim and real, report performance as a function of mask quality (e.g., IoU corruption, under/over-segmentation, occlusion levels), and include “box-only” and “no-mask” variants to quantify DetPool’s pure benefit under weak or absent prompts.
2. How is DetPool materially different from existing mask/ROI-guided object-centric representations (e.g., ROI pooling, mask-guided token selection, segmentation feature fusion)? Please compare against these closest alternatives under a fair hyperparameter grid and explain where DetPool’s advantage comes from (attention constraint vs pooling choice).
3. Your policy uses history length C=16 and predicts K=16 future actions. Please provide modular ablations where you hold the controller fixed and swap the visual representation (and vice versa). Also include latency/frame-rate vs success curves to show the gains are due to DetPool rather than increased temporal capacity or compute.
4.On Franka you use ~250 toys and ~1,500 successful demos; on the dexterous hand ~500 demos are needed. Please provide real-world scaling plots (toys count × diversity × demos → success), identify the “minimal viable toy set” and demo thresholds.
5. Quantify geometric similarity between training toys and target objects (e.g., Chamfer/SDF/voxel features), bucket by similarity, and report success with error bars. When targets diverge strongly from the primitives, does DetPool still generalize? Diagnose failure sources (grasp candidate generation, gripper aperture, approach strategy, etc.).
6.When mask boundaries are biased, have holes, or stick to background, how distorted is the representation? Do you need boundary refinement or multi-scale token aggregation to stabilize features? Please provide sensitivity analyses and any corrective post-processing that helps.

If the authors can address these questions convincingly during rebuttal, I am inclined to raise my score.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a method LEGO (LEarning to Grasp from tOys), to grasp diverse objects by learning to grasp random, synthetically generated toys composed of primitive 3D geometries (spheres, cuboids, cylinders, and rings). For each toy, a random number of primitives of random sizes are fused in random positions and orientations. In total, 250 such toys are generated in simulation as well as 3D printed in the real world. Grasping data is primarily collected via teleoperation except for single primitives, which are collected with motion planning.

A neural network is trained with behavior cloning on the grasp demonstrations. The architecture consists of a Vision Transformer (ViT) for extracting visual features. Authors use a pretrained MVP. SAM 2 is used to get a segmentation mask of the object of interest. Attention is masked between object and background patches using the segmentation mask. Finally, features in object patches are mean-pooled to get a vision feature. This masking mechanism and pooling is coined Detection Pooling (DetPool).

The vision feature is concatenated with proprioception to get a single token at a time step. Multiple tokens from the history of timesteps are passed to a transformer policy to predict the next action token.

Results, both in simulation and real robot, show competitive performance, even when compared to the state-of-the-art VLAs.

### Strengths
**Positive result**
The authors show results in simulation as well as the real world on 2 robot platforms (Franka robot and Unitree H1-2), comparing their method against state-of-the-art VLAs (both scratch and finetuned). In simulation and Unitree H1-2, the proposed method outperforms all baselines. In real Franka, the proposed method outperforms all baselines except finetuned $\pi_0-FAST$

**Extensive ablations**
1. As they increase no. of toys, performance increases initially but quickly saturates with no/minila difference beyond 25 toys
2. As they increase no. of demonstrations, performance increases
3. As they increase model size, performance improves and then saturates. 86M is the sweet spot for their setting
4. Sphere is the most important primitive shape as removing it leads to significant drop in performance. Ring has the least impact
5. Toy complexity (defined by no. of primitives): 2 primitives contributes the most. Likely due to the level of complexity of toys in the test set

### Weaknesses
**Doesn't capture desired grasp location**
Many objects have desired grasp location, e.g. a cup should be grapsed by it's handle & a vertical cylindrical object is more stably grasped sideways instead of top-down. The proposed method finds a grasp location but doesn't guarantee grasp at desired grasp location since grasping is learnt on random toys.

**Doesn't capture physical properties of objects**
Some objects may have delicate parts necessitating that the robot doesn't apply too much force which grasping. As such, it's a stretch to claim that the method can grasp any object as per the title. A better wording would be - diverse real world objects or something similar.

**Constraints imposed by masking**
This makes it more difficult to extend to complex manipulation tasks involving multiple objects or occlusion

**Minor comments**
1. Line 191: Grammar - "details our including preliminaries"
2. Line 197: Definining visual observations as $i_{1:T}$ is incorrect since at $t=1$, visual observation is $i_{1-C+1:1}$ which includes time $t<1$ which is outside the $1:T$ range. Suggestion - We denote visual observation at time $t$ as $i_t$ and proprioceptive state as $s_t$. Observation at time $t$ consists of $i_{t-C+1:t}, s_{t-C+1:t}$.
3. Line 425: "Effect of Number and Diversity of Demonstrations" - Diversity can come from grasp style variation and not just object variation. Suggestion - Effect of Number Demonstration and Number of unique toys
4. Table 4 and 5: Label the entries. No. of demonstration and success rate.
5. In conclusion, it's not entirely correct to claim that the method outperforms baseline VLAs since for Franka finetuned $\pi_0 FAST$ significantly outperforms.
6. Line 403: Missing citation of ManiSkill
7. Line 425: Missing details on test objects, no. of evals, etc

### Questions
1. What was the reason for choosing absolute instead of delta joint prediction?
2. Line 307: "our unique 250 toys". Line 341: "We 3D print the 250 toys with the highest". If 250 is the total no. of toys, how many of them were selected for Franka experiment?
3. What is the performance trend when trained between 1 and 25 toys?
4. What were the primary failure modes? What's preventing higher success rates, say 90%, 99% or even 100%?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes training robotic grasping policies exclusively on randomly assembled “toys” made from four shape primitives, then evaluating zero-shot on real objects. A detection pooling mechanism (DetPool) is introduced to induce object-centric visual representations. Experiments in simulation and on two real robot platforms (a Franka with a parallel gripper and a humanoid with dexterous hands) show strong transfer, including notable gains over large pretrained vision-language-action models under several settings. Ablations study the role of toy diversity, number of demonstrations, model size, and primitive/complexity composition.

### Strengths
- Clear and compelling motivation with an interesting set of experiments
- Training only on random compositions of four primitives is a nice way to study object-generalization. 
- The procedure for generating and 3D-printing toys is well specified and reproducible.
- Simple, modular representation idea with strong empirical effect:
- Results in simulation and a real Franka setup on 64 YCB objects, and a humanoid with dexterous hands
- The study disentangles the effects of number of unique toys versus demonstrations per toy, showing stronger gains from more demonstrations once a small toy set is present (Figure 4, left). This is a useful, actionable finding for data collection planning.
- Detailed reporting of implementation and experiment setup in the appendix

### Weaknesses
- Inconsistency/ambiguity in DetPool’s implementation and mechanism:
  - Section 4.3 states that the object mask is used to enforce no attention between object and non-object tokens inside the vision transformer and then mean-pool object tokens. Appendix E, however, describes patchifying, passing through transformer blocks, and then using the mask only at pooling time to select spatial features. This distinction matters: masked self-attention changes the encoder’s computation; masked pooling does not. The paper should explicitly reconcile which variant is used in each experiment (simulation vs. real) and in Table 1’s ablations, and whether both variants were tried and compared head-to-head.
- Mask generation pipeline is split across sections, creating avoidable ambiguity:
  - Section 5.1 states SAM2 for real and ground-truth masks for simulation, while Appendix E explains the two-stage real pipeline (Faster R-CNN boxes → SAM2 masks). The pipeline is present, but unifying its description in the main text would reduce confusion and make the experimental setup easier to follow.
- Unfair evaluation of baselines:
  - LEGO has access to object masks (ground-truth in simulation, detector→SAM2 pipeline in real), whereas competing VLAs (OpenVLA-OFT, π0-FAST) are not given any object cues (e.g., crop/box/mask). Since the paper’s main contribution is about the value of object-centric features, this mismatch likely affects the performance gap. A fair comparison should provide comparable object-centric inputs (or report variants with/without such cues for all methods).
- Mixed messaging on “outperforming state-of-the-art”:
  - The abstract claims outperforming state-of-the-art approaches trained with much more in-domain data. In Table 2, a fine-tuned π0-FAST achieves higher real-world success than LEGO on the Franka setting. The paper discusses this, but the abstract/overall positioning would benefit from qualifying the claim or toning down these claims.
- Evaluation protocol mismatches and thresholds:
  - In simulation, the success threshold for OpenVLA is relaxed to 0.15 m (vs. 0.3 m for others) due to gripper reopening (Appendix H.1). Although documented, thresholds should be unified or the relaxation carefully justified and accompanied by sensitivity analysis.
  - Sensor and control mismatches exist across methods (e.g., camera views, absolute vs. delta joint control). While some of these are inherited from baseline codebases, the paper should quantify or mitigate their impact.
- Vision encoder training status unclear and parameter counts potentially misleading:
  - The paper uses a ViT-L MVP encoder (Section 5.1), but it is not explicitly stated whether it is frozen or fine-tuned in each setting. Reporting whether gradients flow to the encoder is important for reproducibility and for fair comparison to large VLAs that train all vision stacks.
- Missing or limited analyses that would strengthen the generalization claim:
  - Failure-mode and robustness analyses (lighting changes, heavy clutter, occlusion, distractors with similar colors/shapes, heavy texture) are limited. Given the object-centric focus, stress tests around segmentation failures (mask noise, false positives/negatives) would be highly informative.
  - An oracle-cropping or oracle-bounding-box control for baselines would isolate the contribution of precise object localization versus the transformer architecture itself.

**To the authors:** I find the motivation behind this work compelling and see potential for a strong contribution. I’d be glad to reconsider my evaluation if you can effectively address the issues highlighted in the review and outline a concrete plan to strengthen the experimental validation.

### Questions
1. Why only consider Transformer architectures here? It seems to me this could be applicable to CNNs as well?
2. Can you clarify precisely which variant of DetPool is used in your main experiments — masked self-attention plus pooling, or masked pooling only. Were they both compared head-to-head?
3. Since LEGO uses explicit object masks while other baselines do not, did you evaluate variants of the baselines with comparable object-localization inputs (e.g., crops or masks)? If not, how do you justify the fairness of these comparisons?
4. Is the ViT-L MVP encoder frozen or fine-tuned in each experiment? If partially fine-tuned, could you specify which layers receive gradients?
5. The model is described as having 86M parameters. Does this exclude the ViT-L encoder? Could you report both total and trainable parameter counts for transparency?
6. Why is the success threshold for OpenVLA relaxed relative to other methods? Did you perform sensitivity analyses to confirm that this does not alter comparative trends?
7. Some baselines use different control modes and camera setups. Can you quantify or comment on how much these differences might affect performance comparisons?
8. Have you tested how performance degrades under realistic mask noise, detection failures, or segmentation errors? Since DetPool relies on accurate masks, this seems crucial for generalization. 
9. Could you include analyses under more challenging conditions, such as heavy clutter, occlusion, lighting variation, or distractor object, to better support claims of robust object generalization?
10. Have you compared DetPool against simpler alternatives such as crop-and-encode or mask-as-input-channel baselines (e.g., Shariar et al. 2025) to isolate the contribution of attention masking itself? 
11. The abstract claims outperforming state-of-the-art approaches, though π0-FAST surpasses LEGO in one setup. Could you clarify or qualify which baselines and settings support this claim?


**References**
1. Shahriar, F., Wang, C., Azimi, A., Vasan, G., Elanwar, H. H., Mahmood, A. R., & Bellinger, C. (2025). General and Efficient Visual Goal-Conditioned Reinforcement Learning using Object-Agnostic Masks. arXiv preprint arXiv:2510.06277.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces LEGO, demonstrating that robots can acquire robust general-purpose grasping skills by learning from a simple set of objects composed of just four basic shape primitives: spheres, cuboids, cylinders, and rings. 

A key contribution of this paper is the use of a detection pooling mechanism to learn a critical object-centric visual representation, enabling the policy to generalize to a wide range of real-world objects in a zero-shot manner.

### Strengths
- The paper executes a significant number of real-world robot experiments, which is excellent.

- The idea of using randomly assembled objects that are composed from just four shape primitives—spheres, cuboids, cylinders, and rings is insightful.

### Weaknesses
- The detection pooling proposed in this paper shares similar ideas with those widely applied in object-centric manipulation. The authors should discuss these related works and clearly articulate the differences between them and their own approach. For instance, the following works:

[1] Learning Generalizable Manipulation Policies with Object-Centric 3D Representations

[2] Transferring foundation models for generalizable robotic manipulation

[3] DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping

- Another major shortcoming lies in the paper's evaluation. First, the paper only evaluates the grasp task. Evaluating other contact-rich manipulation tasks is necessary, as this is central to distinguishing the proposed method from pure grasping approaches like Anygrasp. Second, the comparison made against models such as FAST and OpenVLA-OFT is unfair, as the evaluation task is a single-task setup that does not require language understanding or reasoning.

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
