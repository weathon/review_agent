# 3D-aware Disentangled Representation for Compositional Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Vision-based reinforcement learning can benefit from object-centric scene representation, which factorizes the visual observation into individual objects and their attributes, such as color, shape, size, and position.
While such object-centric representations can extract components that generalize well for various multi-object manipulation tasks, they are prone to issues with occlusions and 3D ambiguity of object properties due to their reliance on single-view 2D image features. 
Furthermore, the entanglement between object configurations and camera poses complicates the object-centric disentanglement in 3D, leading to poor 3D reasoning by the agent in vision-based reinforcement learning applications.
To address the lack of 3D awareness and the object-camera entanglement problem, we propose an enhanced 3D object-centric representation that utilizes multi-view 3D features and enforces more explicit 3D-aware disentanglement.
The enhancement is based on the integration of the recent success of multi-view Transformer and the prototypical representation learning among the object-centric representations.
The representation, therefore, can stably identify proxies of 3D positions of individual objects along with their semantic and physical properties, exhibiting excellent interpretability and controllability.
Then, our proposed block transformer policy effectively performs novel tasks by assembling desired properties adaptive to the new goal states, even when provided with unseen viewpoints at test time.
We demonstrate that our 3D-aware block representation is scalable to compose diverse novel scenes and enjoys superior performance in out-of-distribution tasks with multi-object manipulations under both seen and unseen viewpoints compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a 3D-aware object-centric representation of scenes for training Reinforcement Learning (RL) agents that exhibits improved generalization capabilities to unseen compositions of object-level attributes in goal-conditioned robotic manipulation tasks. The representation is based on the 3D object scene representation transformer (OSRT) and incorporates the block-slot attention module for further decomposition of slots into object-level factors. The representation is evaluated on Clevr3D and a novel simulated robotic manipulation dataset and improves over OSRT in various metrics. The paper additionally proposes a policy architecture for a goal-conditioned RL task that leverages the object-level factorization of the representation to achieve better zero-shot transfer to unseen compositions/variations of object-level attributes compared to baselines.

### Strengths
**Overview**
- Novelty and contribution are clear.
- Self-contained and clear method description.
- Learned representation facilitates human interpretability and controllability. 
- Good empirical results in terms of transfer to unseen test tasks.
- Thorough study of the different method components.
- Detailed and helpful figures.
- Detailed Appendix.

I am positive about raising my score if at least some of the concerns I have raised in the Weaknesses/Questions sections are addressed.

### Weaknesses
**Overview**
- Abstract and Intro are not clear and logic is hard to follow.
- Assumes access to agent and background masks.
- The Block Transformer relies on human annotation of blocks within the proposed block-slot representation (post representation training but prior to RL policy training).
- Incorporates multiple assumptions about the specific task in the experiment within the RL Block Transformer architecture, which may limit its wider applicability.

**Abstract**

After reading the paper and understanding it, I still find the abstract very confusing. It uses terms which I would not consider as common knowledge such as “shared concept memory” and some misleading phrases such as “The representation, therefore, can stably identify and track 3D trajectories”---reading this I would assume the representation operates on sequences of images (i.e., videos) but it is a multi-view image representation if I am not mistaken. I suggest making the abstract more “abstract” to appeal to the general reader.

**Introduction**

The two key challenges presented in the introduction for “fully utilizing object-centric representations” (lines 52-53) should be backed-up by citations or by results in this work but these are not referenced. This paragraph reads more like—“these two aspects of our method are what is missing”---but the challenges and their causes are not clearly described.

**Assumptions and Limitations**

*Agent and background masks*: this assumption makes the overall method no longer unsupervised. That in itself is not a problem but could introduce a limitation which stems from reliance on humans to e.g., define what is considered the “background” as opposed to being learned from data. It seems that the method strongly relies on the assumption that these masks can be obtained with high fidelity which may make the overall pipeline brittle when this is not the case. The authors claim that these masks can be obtained using foundation models. If so, why not obtain the entire object-centric representation using foundation models? Is your representation-learning method applicable without the dedicated background and agent slots?

*Identifying the meaning of latent block attributes*: your method seems to produce human-interpretable disentanglement in the latent slot representations of scenes considered in this work, which is impressive and a positive outcome in my opinion. That said, your RL policy architecture relies on the human interpretation of the role of each block, which is another form of supervision that can make scaling difficult and introduce brittleness.

*Strong assumptions about a specific task embedded within the policy architecture*: The proposed Block Transformer is designed for a very specific kind of task which limits its wider applicability. The authors “suppose the task is to relocate objects with the matched attributes into the goal position” (lines 242-243), which is not just treated as an example but determines the processing of state and goal slots. The block-wise cross-attention only allows matching objects to attend to each other which is a strong inductive bias that limits its expressivity. What if the task requires reasoning about the relationships of multiple goal objects? What if the task is to bring all the objects to the location of a single predefined object? Every change in the task/reward incurs a change to the architecture, which is not a desirable property. Judging by the results in Figure 11, this limited expressivity seems to not be beneficial even for the task it was designed for, resulting in lower sample efficiency compared to the EIT with the same representation.

Limitations of the approach should be discussed in the paper, possibly in the Conclusion section.

**Experiments**

*3D Block Slot Representation*: it is not clear how much of the improvement in the metrics you considered stem from the additional supervision versus the additional block structure. Experiments evaluating OSRT with the mask objective would help shed light on this matter.

*Goal-conditioned RL*: As I see it, the strong assumptions about the task embedded within the architecture are the main reason for the improved performance on the unseen test tasks. Since the matching mechanism is not learned, these tasks do not require much generalization in the learning sense, would the authors agree with this statement? Thus, although these findings are interesting, they could be considered narrow. I find the results in Figure 11 much stronger and I suggest they be moved to or at least discussed in the main text. The explicitly 3D-aware representations result in significant gains in sample efficiency compared to the 2D representation when paired with the EIT, which points to the fact that they facilitate better in-distribution generalization of the RL policy. Please see my suggestion under **Questions**.

*Misc*:
- How many seeds were used for the experiments?
- How many objects are present in the RL environments? It is not explicitly stated in the paper.
- Are the OOD colors also unseen during the representation model training or just the RL agent?
- I suggest highlighting results that are within a STD from the best performing method in Table 2.

### Questions
**Suggestion**

In my eyes, the main contribution of the paper is the 3D-aware object-centric representation and its applicability for training RL agents that generalize. This view is based on the experimental results, but is not highlighted by the structure and sentiment of the paper (other than the title).
The following suggestion is structural and semantic, and is of course left to the discretion of the authors, i.e., it will not affect my recommendation for acceptance/rejection:

Center the contribution around the 3D block-slot representation, and present the block transformer policy as an *example* of how this representation can facilitate design of task-specific architectures (although I do not think policy architectures should be task-specific). I would move the results in Figure 11 to the main text and discuss how with a fixed policy architecture:
1. 3D-aware representations can improve in-distribution generalization: evidence to that is the sample efficiency of OSRT and your representation compared to DLP on ID.
2. 3D-aware representations with factored attribute structure (i.e., yours) can improve attribute-level compositional generalization: evidence to that is the performance of yours compared to OSRT in CG and CG (same color).
3. Maybe discuss why OSRT performs better on OOD generalization with the same architecture.

**Questions**

- How much does the representation rely on knowing the amount of objects/slots in advance?
- How stable is training the block slot representation in terms of disentanglement of objects to distinct slots and factors to distinct blocks? Does the desired factorization emerge consistently across training runs?
- There are no details about the attribute matching mechanism in the block Transformer. Can the authors provide further details on how this is done?
- Are the authors planning to release code or take other measures for reproducibility purposes?
- A major benefit of the 3D aware representation in my opinion is that it may be view-agnostic. Have you considered training the policy when randomizing the viewpoints (maybe with some restrictions on their directions such that they will capture sufficiently different angles of the scene)? Demonstrating view-generalization capabilities can have a very large impact on the robotics community since vision-based control methods are largely not robust even to slight variations in camera angles to my impression.

### Soundness
3

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
This paper proposes a 3D block-slot representation for vision-based reinforcement learning (RL), targeting improved compositionality, interpretability, and 3D-awareness in multi-object manipulation domains. The method integrates a multi-view transformer encoder with a novel block-slot attention mechanism, decomposing object representations into attribute-level blocks (e.g., color, shape, position) and separating active (objects), passive (background), and agent entities. The paper introduces a block transformer policy leveraging these structured representations for goal-conditioned RL, aiming to boost generalization in out-of-distribution and compositional scenarios. Experimental results are presented on Clevr3D and IsaacGym3D, highlighting gains in object decomposition, disentanglement, and RL performance compared to strong scene-centric and object-centric baselines.

### Strengths
1. **Explicit 3D Disentanglement**: The proposed 3D block-slot attention mechanism, as described in Section 2.1.3, provides structured disentanglement of object attributes into interpretable blocks (e.g., shape, color, size, position), enabling clear alignment with compositionality goals. This is well illustrated in Figure 3 and further analyzed in Figure 8 and Figure 9, which show that certain blocks consistently encode specific semantics (e.g., shape, color).
2. **Separation of Agents, Foreground, Background**: The method distinguishes between active (object), passive (background), and agent entities, as detailed in Section 2.1.2, with dedicated auxiliary objectives enforcing slot assignment and improving downstream policy stability. Figure 6 demonstrates visually superior decomposition versus OSRT, where the agent and background are reliably isolated.
3. **Goal-Conditioned Block Transformer Policy**: The block transformer policy (Section 2.2, Figure 2 and Figure 5) leverages block-level cross-attention for precise goal conditioning, explicitly matching object attributes in current and target scenes. Table 2 quantitatively supports the claim that this block-level policy bolsters generalization, especially in compositional and OOD tasks.

### Weaknesses
1. **Limited Diversity of Environments**  
   The evaluation is restricted to Clevr3D and Isaac Gym 3D environments. Expanding the experiments to include additional benchmarks would provide stronger empirical evidence and reinforce the paper’s claims of effectiveness.

2. **Incomplete Baseline Comparisons**  
   The work compares performance only against earlier object-centric models such as DLPv2 and OSRT. However, more recent and effective object-centric world models [1], [2] should also be considered. Even though these operate in 2D, their inclusion would clarify whether the 3D extension genuinely enhances object-centric policy quality.  
   Additionally, comparisons with non-object-centric 3D baselines [3] and visual 2D non-object-centric RL approaches [4], [5] would help assess whether object-centric representations offer advantages for policy learning.

3. **Insufficient Ablation and Sensitivity Analysis**  
   The paper introduces several architectural and training design choices that are not thoroughly examined.  
   - The choice to apply block-slot attention only to foreground slots (Section 2.1.3), while using vanilla slot attention for agent and background, is motivated by intuition but lacks empirical validation. A comparison among different configurations—such as applying block-slot attention to all slots, a mixed setup, or using only vanilla slot attention—would strengthen the justification. Similarly, the auxiliary mask losses in Equation 4 are not ablated.  
   - Key hyperparameters such as the number of slots, blocks (Table 3), and prototypes are chosen without detailed diagnostic analysis. The absence of sensitivity studies makes it difficult to assess reproducibility, robustness, and generalization of the proposed method.


[1] Ferraro et al. “FOCUS: Object-Centric World Models for Robotics Manipulation.” 2023. [https://arxiv.org/abs/2307.02427](https://arxiv.org/abs/2307.02427)


[2] Mosbach et al. “SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels.” 2024. [https://arxiv.org/abs/2410.08822](https://arxiv.org/abs/2410.08822)

[3] Shim et al. “SNeRL: Semantic-aware Neural Radiance Fields for Reinforcement Learning.” 2023. [https://proceedings.mlr.press/v202/shim23a.html](https://proceedings.mlr.press/v202/shim23a.html)

[4] Hafner et al. “Mastering Diverse Domains through World Models.” 2023. [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

[5] Zhou et al. “DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning.” 2024. [https://arxiv.org/abs/2411.04983](https://arxiv.org/abs/2411.04983)

### Questions
1. See weaknesses section
2. Could you share insights or figures analyzing the impact of number of slots, blocks and prototypes on both decomposition and RL quality?
3.  It is unclear whether the permutation of slots between the current and goal images influences the policy’s performance. Does the model explicitly handle slot alignment or permutation, or could mismatched slot ordering negatively impact policy learning?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a 3-D block-slot encoder–decoder architecture that fuses a multi-view Object Scene Representation Transformer (OSRT) with a block-slot decomposition module inspired by SysBinder. Each scene is encoded into a set of latent “blocks,” each corresponding to an object slot that factorises attributes such as shape, colour, size, and position. Two auxiliary supervision losses constrain fixed background and agent slots.
On top of this representation, a Block Transformer (BT) policy performs block-wise cross-attention between current and goal states for goal-conditioned manipulation.

### Strengths
1. The paper correctly identifies that prior slot-based encoders struggle with 3-D occlusion and pose entanglement, and that OSRT-like models lack attribute factorization. The system architecture is easy to follow, and ablations are reasonably thorough.

2. The combination of OSRT’s volumetric scene reasoning with SysBinder-style factorized slots is a clever, pragmatic design. The explicit background and agent slots make the representation interpretable and stable across timesteps.

3.  On both Clevr3D and IsaacGym3D, the model shows clear quantitative gains in decomposition (FG-ARI and D/C/I) and control success. The Block-Transformer (BT) policy demonstrably improves compositional generalization, especially in out-of-distribution color/shape settings.

### Weaknesses
1. The method’s conceptual innovation is modest. The encoder is largely a direct hybrid of OSRT (multi-view scene transformer for volumetric representation) and SysBinder (slot factorisation via attribute prototypes). The Block Transformer (BT) is a small architectural tweak of EIT, with cross-attention operating at the block level instead of the pixel/feature level. This makes the contribution primarily engineering integration rather than algorithmic advancement. The claimed novelty of “3D-aware block-slot” reduces to combining volumetric rendering with slot decomposition.

2. A major conceptual gap is the reliance on explicit segmentation masks for two dedicated slots: Eq. (2) defines auxiliary losses requiring binary masks for the background and agent (manipulator). Without such supervision, it’s unclear whether the model can autonomously discover these roles. This violates the unsupervised or weakly-supervised assumption common in object-centric learning, where slots are meant to emerge purely from reconstruction signals. In real-world robotics, such masks are rarely available. In unlabeled multi-object scenes, the model might not allocate the correct number of slots or separate the manipulator from movable objects.

3. Several notational and definitional issues appear: In Eq. (1), the attention-weight mask loss compares per-ray weights 
𝑤 with 2D masks 𝑚; the mapping from ray to pixel is unspecified, making the loss potentially ill-posed.

In Eq. (3), λ_bg and λ_ag are introduced but not analyzed, their effect on disentanglement or convergence is unknown.
The model description doesn’t clarify how 3D latent “blocks” correspond to volumetric regions or voxels; thus, the claim of 3D-awareness is more architectural than mathematical.

4. Does not currently release its implementation or code repository.

### Questions
1. How does performance degrade if only a single camera is available at test time? Can the model lift monocular views to 3-D reliably?

2. Please report inference latency per frame and total compute (encoder + BT) versus DLPv2 and OSRT baselines.

3. If background/agent masks are unavailable, does the model still allocate them automatically or does performance collapse?

### Soundness
2

### Presentation
3

### Contribution
3
