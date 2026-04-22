# SceneAdapt: Scene-aware Adaptation of Human Motion Diffusion

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Human motion is inherently diverse and semantically rich, while also shaped by the surrounding scene. However, existing motion generation approaches fail to generate diverse motion while simultaneously respecting scene constraints,  since constructing large-scale datasets with both rich text-motion coverage and precise scene interactions is extremely challenging. In this work, we introduce \textbf{SceneAdapt}, a framework that injects scene awareness into text-conditioned motion models by leveraging disjoint scene–motion and text–motion datasets through two adaptation stages: inbetweening and scene-aware inbetweening. The key idea is to use motion inbetweening, learnable without text, as a proxy task to bridge two distinct datasets, thereby injecting scene-awareness. In the first stage, we introduce keyframing layers that modulate motion latents for inbetweening while preserving the latent manifold. In the second stage, we add a scene-conditioning layer that injects scene geometry by adaptively querying local context through cross-attention. Experimental results show that \textbf{SceneAdapt} effectively injects scene awareness into text-to-motion models, and we further analyze the mechanisms through which this awareness emerges. Code and models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on HSI and aims to solve the issue of modeling motion semantics and scene-awareness simultaneously. This paper introduces SceneAdapt, a framework that injects scene awareness into text-conditioned motion models by leveraging disjoint scene–motion and text–motion datasets through two adaptation stages. Comprehensive experiments have proven the effectiveness of SceneAdapt.

### Strengths
This paper innovatively introduces motion inbetweening into the model to generate scene-aware human motion. The paper is well-developed and provides extensive analysis and visualization results.

### Weaknesses
1. The distinction between HSI(Human-Scene Interaction) and scene-aware text-to-motion generation is not clearly stated in this paper.
2. This paper only reports performance on the dataset constructed by the author, and does not conduct experiments on other public datasets.

### Questions
1. Why is MDM trained for the task of motion inbetweening instead of directly using the motion inbetweening model?
2. What impact will the absence of Stage 0 have on scene-aware text-to-motion generation performance?
3. Why didn't the author use an autoregressive motion generation model to complete the motion inbetweening task without adding additional modules?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
# Summary
The paper proposes SceneAdapt, a two-stage adaptation pipeline that injects 3D scene awareness into a pretrained text-conditioned motion diffusion model (MDM) using disjoint text–motion and scene–motion datasets.   Stage 1: Insert Context-aware Keyframing (CaKey) layers that modulate only keyframe latents to learn motion inbetweening while preserving the pretrained latent manifold. Stage 2 (scene-aware inbetweening): Freeze CaKey and add scene-conditioning cross-attention layers that query voxel ViT patch embeddings so each frame attends to local scene context; 
Once done, during the Inference, one can generate scene-aware motion from text + scene without keyframes, trading off semantics vs geometry via two CFG scales.

### Strengths
# Strengths
- Simple, plausible idea: Using inbetweening as a proxy to bridge two datasets is intuitive and effective, avoiding costly text–scene–motion triplets. 
- Controllable trade-off: Dual CFG scales for text and scene provide a clear knob to balance semantics and geometry.
- The writing is generally ok but with some problems (See the next section).
- Authors promise code/model release.

### Weaknesses
# Weaknesses
- The idea is not very novel. Keyframe-based motion gen and attention-based adaptor have been largely explored.

- Visual comparisons can be clearer; collisions are still visible. In the “selected_blinded” materials, please consider adding method-name overlays on every clip. Also, it's better to make them into one single video. Meanwhile, despite improvements, there are some noticeable penetrations in many cases in the video.

- Overstated positioning in related work. The claim that prior work addresses “either semantics or scene-awareness in isolation” is too strong… Recent efforts *do* combine text and scene: 1) Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance, CVPR 2024. 2) Generating Human Interaction Motions in Scenes with Text Control, ECCV 2024. 3) Generating Human Motion in 3D Scenes from Text Descriptions, CVPR 2024. 4) LaserHuman: Language-guided Scene-aware Human Motion Generation in Free Environment, 2024... The contribution should be framed as *how* SceneAdapt avoids triplets and *how* it injects scene priors efficiently.

- Evaluation-set construction may bias outcomes. The test set pairs HumanML3D text–motion with random TRUMANS start poses/trajectories after SDF-based filtering. This can inadvertently favor or disfavor certain prompts and should be discussed as a limitation (correct me if I am wrong).

- Writing. Several typos/grammar issues (e.g., “exsisting,” “well-adpated,” “indicies,” “sacraficing,” “Qaulitative”)...; terminology for the scene-conditioning module (“SceneCo/ScenCo”) is inconsistent. Also, “et, as these datasets omit scene context, such models remain blind to spatial constraints and cannot generate motions that interact plausibly with the environment” - this is because they are targeting text-to-motion generation…

### Questions
1. Could you report **penetration-depth distributions** (e.g., percentiles or max/min) in addition to means? Table 1 (p.7) averages may hide long tails with severe MMP. I did see some penetration cases in the submitted video.

2. Robustness to scene representations: How does performance change with **sparser or noisier reconstructions** (e.g., raw point clouds / TSDF / meshes) vs voxel ViT patches?

### Soundness
3

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
3

### Summary
This paper introduces SceneAdapt, a two-stage adaptation framework that effectively injects scene awareness into a pretrained text-conditioned motion diffusion model by leveraging motion inbetweening as a proxy task, enabling the generation of semantically rich and physically plausible human motions without requiring costly text-scene-motion triplets.

### Strengths
**Practical Problem Formulation**
- Solves the critical problem of injecting scene awareness into pretrained text-to-motion models
- Avoids costly collection of text-scene-motion triplets by leveraging existing disjoint datasets

 **Well-Designed Two-Stage Framework**
- **Stage 1**: Learns motion inbetweening while preserving the pretrained model's capabilities
- **Stage 2**: Focuses exclusively on scene awareness building on a stable motion foundation
- Clear separation of concerns prevents catastrophic forgetting

 **Novel CaKey Layer Design**
- **Sparse modulation** only affects keyframe latents, preserving the generative manifold
- **Context-aware** through diffusion timestep and self-attention inputs
- Superior to standard adaptation methods like LoRA for this specific task

 **Advanced Scene Conditioning**
- Uses **patch-wise cross-attention** instead of global scene vectors
- Enables dynamic, localized interaction with scene geometry over time
- More physically plausible than previous global conditioning approaches

### Weaknesses
**Missing Critical Comparison with Triplet Datasets**
- **Core Issue**: Fails to compare against models trained on existing **text-scene-motion triplet datasets**
- **Consequences**: Cannot determine if performance stems from:
  - The **adaptation method itself**, OR
  - Simply **inheriting advantages** from larger text-motion pre-training data
- **Relevant References**:
  - `[HUMANISE]` Wang et al., "Humanise: Language-conditioned Human Motion Generation in 3D Scenes", NeurIPS 2022
  - `[LaserHuman]` Li et al., "LaserHuman: Language-guided Scene-aware Human Motion Generation", arXiv 2024

**Overly Narrow Definition of "Scene Awareness"**
- **Core Issue**: Equates scene awareness primarily with **collision avoidance metrics** (CFR, MMP, JCR)
- **Consequences**: Shows no capability for **positive, goal-oriented interactions**

**Unverified Proxy Task Justification**
- **Core Issue**: Entire pipeline relies on unsubstantiated assumption that **motion inbetweening** is optimal proxy for scene awareness
- **Consequences**: No theoretical/empirical evidence why temporal interpolation enables spatial reasoning

The adaptation paradigm cannot be properly evaluated without fair comparisons against models trained on existing triplet datasets.

### Questions
**Question 1:** The evaluation set is constructed by randomly pairing HumanML3D texts with TRUMANS scene trajectories, creating semantically mismatched pairs (e.g., "a person does a cartwheel" in a narrow hallway). How can we trust that the reported R-Precision and FID scores accurately reflect model performance when the ground truth pairs themselves are semantically implausible? Would results on a curated, semantically consistent subset tell a different story?

**Question 2:** Given the existence of text-scene-motion triplet datasets like HUMANISE and LaserHuman, why wasn't SceneAdapt compared against models trained end-to-end on such data? Without this comparison, how can we determine whether your performance advantage comes from the adaptation method itself, or simply from leveraging a larger and richer text-motion pre-training dataset?

**Question 3:** Your method demonstrates competence in collision avoidance but shows no capability for positive object interactions (e.g., sitting on chairs, grabbing cups). Do you consider your current approach fundamentally limited to passive obstacle avoidance, or can it be extended to support goal-directed interactions? What architectural changes would be needed to handle such "functional" scene awareness?

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
This paper proposes a two-stage adaptation framework named SceneAdapt, designed to inject scene awareness into pre-trained text-to-motion diffusion models. The core idea is to cleverly leverage "motion inbetweening" as a text-free proxy task to bridge two disjoint datasets: a large-scale text-motion dataset and a smaller scene-motion dataset. In the first stage, the model learns inbetweening via a novel CaKey layer; in the second stage, it learns to perform inbetweening within a scene via a SceneCo layer, thereby acquiring scene awareness. Ultimately, the model is capable of generating motions that are both consistent with the text description and physically plausible within the given scene.

### Strengths
1.  Leveraging "motion inbetweening" as a proxy task to decouple and bridge two heterogeneous datasets is a very clever idea. It provides an effective new paradigm for multi-modal model fusion in data-scarce scenarios.
2.  The method directly confronts the bottleneck of data acquisition by proposing a solution that does not rely on expensive triplet-annotated data. This makes it more feasible to imbue existing large-scale motion generation models with scene awareness.
3.  The designs of the CaKey and SceneCo layers are well-targeted. Specifically, the CaKey layer's use of sparse modulation to preserve the latent space of the pre-trained model and the SceneCo layer's use of cross-attention to focus on local scene features both demonstrate sound design principles.

### Weaknesses
The evaluation protocol is biased: Due to the lack of a real-world test set, the authors construct an evaluation set by randomly matching text-motion pairs with scenes. This is a significant flaw. This approach can generate a large number of illogical test cases (e.g., testing the instruction "walks up the stairs" in a room with no stairs). Consequently, the current evaluation metrics primarily measure the model's geometric collision avoidance capabilities when performing an arbitrary action in an arbitrary scene, rather than its ability to generate a reasonable action in a logically matched scene. This diminishes the persuasiveness of the evaluation results.

### Questions
1.  Regarding the validity of the evaluation set: How do the authors address the potential for logical mismatches arising from the "random matching" of the evaluation set? For instance, when the text description is entirely unrelated to the scene content, how should we interpret the model's performance and the meaning of metrics like R-Precision?
2.  Regarding the potential bias of inbetweening: Is it possible that using motion inbetweening as the core proxy task might inadvertently bias the model towards generating smooth, continuous motions, thereby affecting its ability to generate more explosive or abrupt actions?

### Soundness
3

### Presentation
3

### Contribution
2
