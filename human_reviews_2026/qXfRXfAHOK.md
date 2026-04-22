# PA3FF:Learning Part-Aware Dense 3D Feature Field For Generalizable Articulated Object Manipulation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Articulated object manipulation is essential for various real-world robotic tasks, yet generalizing across diverse objects remains a major challenge. A key to generalization lies in understanding functional parts (e.g., door handles and knobs), which indicate where and how to manipulate across diverse object categories and shapes. Previous works attempted to achieve generalization by introducing foundation features, while these features are mostly 2D-based and do not specifically consider functional parts. When lifting these 2D features to geometry-profound 3D space, challenges arise, such as long runtimes, multi-view inconsistencies, and low spatial resolution with insufficient geometric information. To address these issues, we propose \textbf{Part-Aware 3D Feature Field (PA3FF)}, a novel dense 3D feature with part awareness for generalizable articulated object manipulation. PA3FF is trained by 3D part proposals from a large-scale labeled datasets, via a contrastive learning formulation. Given point clouds as input, PA3FF predicts a continuous 3D feature field in a feedforward manner, where the distance between point feature reflects the proximity of functional parts: points with similar features are more likely to belong to the same part. Building on this feature, we introduce the \textbf{Part-Aware Diffusion Policy (PADP)}, an imitation learning framework aimed at enhancing sample efficiency and generalization for robotic manipulation. We evaluate PADP on several simulated and real-world tasks, demonstrating that PA3FF consistently outperforms a range of 2D and 3D representations in manipulation scenarios, including CLIP, DINOv2, and Grounded-SAM, achieving state-of-the-art performance. Beyond imitation learning, PA3FF enables diverse downstream methods, including correspondence learning and segmentation task, making it a versatile foundation for robotic manipulation.  Project page: https://pa3ff.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that robots handle new objects better when they reason about the parts that matter for action—like handles, buttons, and lids—rather than whole objects. The authors introduce a 3D “part-aware” feature field that turns a point cloud into dense features where points on the same functional part look alike, and those features are tied to plain-language part names. They then use this representation in a diffusion policy that conditions control on the named part, letting the robot plan motions directly from the 3D scene. Because the features are native to 3D, they’re more consistent across viewpoints and make part boundaries clearer. In experiments spanning simulation and eight real-world tasks, the method outperforms strong 2D-feature and 3D-policy baselines, particularly when generalizing to unseen objects and states. The same features also enable point-to-point correspondence and unsupervised part segmentation, making the approach a broadly useful backbone for part-centric perception and manipulation.

### Strengths
The paper introduces a 3D-native, part-aware representation that’s aligned with language and plugs cleanly into a diffusion control policy, leading to strong generalization across unseen objects, states, and tasks. The evaluation is thorough—covering simulation and eight real-world tasks with clear five-way generalization splits—and shows sizable gains over both 2D-lifted and 3D baselines. Beyond control, the same features enable point correspondence and unsupervised part segmentation, and ablations clarify why the 3D-native design outperforms view-lifted alternatives.

### Weaknesses
1. The work has extensive evaluation on robot experiment, but lack of quantitative evidence on the feature field quality.
2. Runtime/latency: PADP runs at ~4.23 FPS vs. DP/DP3 at ~12 FPS, limiting high-frequency control. 
3. Dependence on part supervision & external text embeddings: training leans on labeled parts and SigLIP part-name embeddings; baselines may not use comparable supervision.

### Questions
Here are more feature splatting paper to cite:
LERF: Language Embedded Radiance Fields (ICCV 2023)
Feature Splatting: Language-Driven Physics-Based Scene Synthesis and Editing (ECCV 2024)
M3: 3D-Spatial Multimodal Memory (ICLR 2025)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Summary:
The paper proposes PA3FF, a 3D-native, part-aware dense feature field learned directly from point clouds, and PADP, a diffusion policy that uses PA3FF as a frozen perception backbone with language and robot-state conditioning. Together they target articulated-object manipulation with better sample efficiency and generalization to unseen objects, showing superior results to prior 2D/3D features plus diffusion-policy baselines in both simulation and real-world tasks, and also enabling downstream uses like correspondence and part segmentation.

Main contributions:
1.PA3FF: a part-aware 3D feature field that enforces within-part feature coherence and between-part separability, trained with contrastive objectives and text alignment to functional part names.
2.PADP: a diffusion policy built on the frozen PA3FF backbone, conditioned on language cues and robot state, improving sample efficiency and cross-instance and cross-category generalization.
3.Strong empirical gains over representative baselines (e.g., CLIP/DINOv2 features and DP/GenDP families) across multiple generalization splits in simulation and eight real-world articulated-object tasks.
4.Versatility of the learned representation, supporting additional perception tasks such as shape correspondence and part segmentation.

### Strengths
### Quality

* Pipeline is reasonably complete: pretrained 3D backbone, contrastive representation learning, language conditioning, and diffusion policy, with corresponding ablations.
* Experiments cover simulation and a modest set of real tasks, include cross-instance/category splits, and compare against common 2D/3D feature and diffusion baselines with generally consistent gains.

### Clarity

* Problem framing is clear: focus on functional-part consistency to address articulated-object manipulation bottlenecks.
* Exposition is structured (representation → policy), with losses and inputs described in layers; implementation details are sufficient for high-level replication.

### Significance

* Potential to reduce instance-specific engineering and data needs, especially under shifts from unseen objects or deformations.

### Weaknesses
1. Insufficient novelty (core issue)
   The paradigm—“part-aware dense 3D representation + language prompts + diffusion policy”—reads as a combination/tuning of existing components (NDF-style dense correspondence, DP3/GenDP-style 3D-aware diffusion, ULIP-style language–3D alignment). The manuscript does not present an indispensable conceptual increment (new inductive bias/new representational property/new problem formulation); current differences are mainly in implementation and loss engineering.
   — Suggestion: Use a “conceptual comparison + ablation proof” to pinpoint your **single unique idea**: show that removing that idea (e.g., the part-consistency term or specific field structure) causes a **significant** drop, and provide head-to-head results against the strongest nearby baselines (DP3/GenDP/NDF variants).

2. Lacking verifiable “necessity evidence” for the representation claim
   You claim “part-aware” beats a “generic 3D semantic field,” but there is no counterfactual under matched supervision and budget to show the advantage comes from the representation itself rather than backbone scale or the pretraining data distribution.
   — Suggestion: Under the **same backbone and training budget**, swap only “part-aware field ↔ generic semantic field,” and report cross-task/cross-object gains with significance tests.

3. Unclear scope of the language module’s contribution
   Conditioning/alignment on part names is not novel, and there is no degradation/robustness quantification (synonyms, hierarchical terms, noisy/wrong labels, no-language variant) to show language is a **key** driver rather than a cosmetic add-on.
   — Suggestion: Provide curves of **language-noise strength → performance**, and report **per-task/per-part** marginal contributions.

4. Heavy reliance on strong pretrained backbones; insufficient factorized ablations
   A large portion of the gains may come from PTv3/Sonata-style pretraining; current ablations do not sufficiently disentangle backbone capacity from your objectives/structure.
   — Suggestion: Run **full-factorial ablations** (backbone type × with/without large-scale pretraining × with/without language alignment, contrastive losses, part supervision, structural tweaks).

### Questions
1. Please state the paper’s **single indispensable conceptual increment** (not implementation or loss details) and explain what **new inductive bias/representational property** it introduces.
2. Please provide a **conceptual comparison table** contrasting NDF / DP3 / GenDP / ULIP / *this work* , and mark which elements are **first introduced** by this paper.
3. Please report ablations that **remove the key new component(s)** (e.g., part-consistency loss, specific field structure, alignment mechanism): do all primary metrics **drop significantly**? Include statistical significance.
4. Under **identical sensing inputs, action space, number of demonstrations, and training budget**, run **head-to-head** comparisons against the strongest nearby baselines (DP3/GenDP/NDF variants). If the method still does not win, explain how the claimed “novelty” stands.
5. Please provide a **correlation analysis from field quality to performance**: e.g., within-part coherence, cross-view consistency versus success rate, with correlation coefficients and visualizations.
6. Please add **degradation and robustness** studies for the language component:
   * Synonym substitution (handle/knob/grip), hierarchical terms (door handle vs. handle);
7. How is the vocabulary constructed and disambiguated? How do you handle **same-name different parts** or **cross-category semantic drift**? Please provide **error cases and proportions**.
8. Please provide a **full-factorial ablation**:
   {Backbone: PTv3 / Point Transformer / others} ×
   {With/without Sonata or equivalent large-scale pretraining} ×
   Report results in **both simulation and real** settings.

**If the authors can satisfactorily address these questions, I would raise my score.**

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel Part-Aware 3D Feature Field (PA3FF), which is trained via contrastive learning to integrate 3D geometric priors and functional part awareness, addressing the challenge of limited generalization in articulated object manipulation. Building upon this feature, the authors introduce the Part-Aware Diffusion Policy (PADP), an imitation learning framework , and demonstrate superior performance over existing 2D and 3D representation baselines in both simulated and real-world tasks.

### Strengths
1. Clarity and Novelty of Motivation: The paper proposes PA3FF as a novel 3D-native feature representation, directly addressing challenges faced by lifting existing 2D foundation features to 3D space, such as long runtime, multi-view inconsistencies, and low spatial resolution. PA3FF explicitly incorporates functional part awareness, which is crucial for generalizable articulated object manipulation.
2. Demonstrated Generalization Capability: The proposed PADP framework is claimed by the authors to achieve superior performance over baselines in both simulated and real-world environments. It demonstrates notable robustness, particularly in handling unseen objects, spatial generalization, and environment generalization tasks.
3. Methodological Completeness: The proposed approach features a complete multi-stage learning framework: 1) leveraging the pre-trained Sonata model to extract 3D geometric priors; 2) employing contrastive learning to fuse a geometric loss and a semantic loss, thereby enhancing feature part-awareness and semantic alignment ; and 3) integrating PA3FF into a diffusion policy for action generation.

### Weaknesses
1. The architectural modification of the PTv3 backbone (removing down sampling layers and stacking additional Transformer blocks) is a core engineering contribution. However, the paper lacks sufficient quantitative details (e.g., parameter count, FLOPs, precise layer configuration) and a dedicated gain analysis for these changes. This absence of detailed exposition and architectural diagrams prevents readers from adequately assessing its contribution to the final performance and hinders the reproducibility of the research.
2. The real-world experiments are evaluated with only 10 trials per task. This low number of evaluations in robotics may not provide sufficiently high statistical reliability to convincingly support the "state-of-the-art" claims. Furthermore, restricting the ablation study to a single task ("Put in Drawer")  severely weakens the proof of generality for component contributions across a broad range of tasks and different generalization types (e.g., OI, OC).
3. PADP exhibits a significant drawback in inference speed compared to baselines like DP3 (4.23 FPS vs. 12.7 FPS).  Although PADP achieves higher success rates, this over 60% reduction in inference speed has not been adequately justified as a necessary trade-off (i.e., whether a  10~20% success rate gain, which does not guarantee deterministic success, is worth the real-time cost). In real-time robotic control requiring high-frequency feedback or integration into large policy frameworks, this performance-efficiency trade-off may reduce its practical applicability.

### Questions
1. Generalization Source and Action Semantics Decoupling: The PartInstruct benchmark tests generalization across various factors, including Object State (OS), Object Instance (OI), Part Combination (TP), Task Category (TC), and Object Category (OC). Please confirm if the model is trained only on the Training Set data. If so, how does PADP or its language encoding module achieve generalization over action direction semantics (e.g., generalizing from *forward* to *backward* action prediction as seen in Figure 12)? This requires explaining the policy's mechanism for understanding and decoupling non-object-related semantics in the language instruction.
2. Precise Role of Language Embedding in Feature Aggregation: The paper states that the "semantic embedding of the task-critical part name" is used as the CLS token in the Transformer encoder to guide aggregation. Concurrently, Language Instruction is shown as an input in Figure 2, Stage III. Please clarify which specific text input (task-critical part name vs. full language instruction) is input to the PERCEPTION part for embedding, and how it relates to the language information input to SigLip during PA3FF training.
3. Definition of Real-World Evaluation Metrics: Table 2 reports Train/Test success rates for real-world tasks. Given the experiment statement "Each task is evaluated with 10 trials under randomized initial conditions", please explicitly define: Does the Train success rate represent testing under randomized initial conditions using the exact objects and scenes used for training? And does the Test success rate represent testing under randomized initial conditions using unseen object instances or environmental changes? Clarifying these definitions is crucial for interpreting the real-world generalization performance.
4. Feature Robustness and Cross-Topology Consistency: For the same object category with significantly different spatial morphologies, can PA3FF effectively cluster and align features? For example, regarding the faucets with topologically distinct structures shown in Figure 6, please provide a quantitative assessment of PA3FF features' cross-topology consistency/transferability between them. This is needed to more fully demonstrate the robustness limits of the part-aware features.
5. Completeness of the PADP Framework Flow: The overall flow of the PADP framework remains insufficiently clear. Please provide a detailed description of the training process for a new task, explicitly detailing like which point cloud data requires manual labeling,how the data flows in the framework, and what information needs to be unified.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
In this work, the authors consider the task of learning representations for articulated objects which are useful in downstream manipulation tasks. Specifically, the authors propose a procedure to pre-train a neural network to map 3D point clouds to part-aware features, with two different contrastive supervision techniques (spatial and semantics). They build on top of Sonata (PTv3 pre-trained), but make some architectural modifications to enable higher-resolution representations. These representations are then used in several downstream manipulation tasks. The authors show compelling results on simulated & real tasks.

### Strengths
* Their proposed architecture, pre-training, and modification are all sensible & principled approaches to handling object-level features at high-resolution
* The results do improve over SOTA considerably
* There are extensive ablations showing how different parts of the system contribute to performance, as well as qualitative visualizations of feature representations.
* The paper is well-written

### Weaknesses
* It’s unclear whether the comparison with DP3 is completely valid (e.g. Sonata + DP3); the authors should clarify the difference between DP3 and the various ablations (e.g. where/when SigLip are included, architectures, etc.). It would help the authors cleanly show that 1) the point cloud architecture change and 2) the pre-training compared to DP3 make a major difference on the task (right now it’s just difficult to tell from the details of the paper).
* It’s unclear how much the spatial vs. semantic components actually make a difference. In ablations, the contrastive pretraining (feature refinement) is not broken down by whether the spatial or semantic components make the difference
* Details on architecture / training are a bit sparse

### Questions
* Lots of details of training / architecture are omitted - despite being pointed to Appendix A I didn’t see much there. Particularly interested in specific architectural modifications, fine-tuning/pre-training technique when using the pre-trained Sonata weights.
* Why is it called a field instead of a representation? Seems to be per-point features, like querying other points in space don’t seem to be possible without altering the representation?

### Soundness
3

### Presentation
3

### Contribution
3
