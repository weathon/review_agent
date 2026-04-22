# iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Video Virtual Try-On (VVT) aims to seamlessly replace a garment on a person in a video with a new one. While existing methods have made significant strides in maintaining temporal consistency, they are predominantly confined to non-interactive scenarios where models merely showcase garments. This limitation overlooks a crucial aspect of real-world apparel presentation: active human-garment interaction. To bridge this gap, we introduce and formalize a new challenging task: Interactive Video Virtual Try-On (Interactive VVT), where subjects in the video actively engage with their clothing (e.g., pulling a hem or unzipping a jacket). This task introduces unique challenges beyond simple texture preservation, including: (1) resolving the semantic ambiguity of interactions from standard pose information, and (2) learning complex garment deformations from video where interactive moments are sparse and brief.
To address these challenges, we propose \textbf{iTryOn}, a novel framework built upon a large-scale video diffusion Transformer. iTryOn pioneers a multi-level interaction injection mechanism to guide the generation of complex dynamics. At the spatial level, we introduce a garment-agnostic 3D hand prior to provide fine-grained guidance for precise hand-garment contact, effectively resolving spatial ambiguity. At the semantic level, iTryOn leverages global captions for overall context and time-stamped action captions for localized interactions, synchronized via our novel Action-aware Rotational Position Embedding (A-RoPE). Furthermore, we design an action-aware constraint loss to stabilize training and focus the learning process on these critical interactive frames. To facilitate research and evaluation, we construct VVT-Interact, the first large-scale dataset for this task. Extensive experiments demonstrate that iTryOn not only achieves state-of-the-art performance on traditional VVT benchmarks but also establishes a commanding lead in the new interactive setting, marking a significant step towards more dynamic and controllable virtual try-on experiences.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Given a video and a garment image, the goal of this paper is to change the garment in the input video to match the input garment image, while maintaining realistic human-garment interactions.

The authors introduce several contributions:

(1) Multi-level interaction injection mechanism: The authors leverage a 3D hand prior (HaMeR model (Pavlakos et al., 2024)) and global captions for video segments, along with Action-aware Rotational Positional Embedding.

(2) Constraint loss: Intensifies supervision around interactions in a video, since these are more rare than non-interactive video clips.

(3) VVT-Interact: The first dataset for interactive video try-on

### Strengths
Originality: 
- This paper addresses the unique challenge of VVTO with interaction.
- The paper introduces a novel benchmark for evaluating interactive VVTO

Quality:
- The folding/movement of the garment fabrics looks very realistic with respect to the material of the input garment.
- Compared to related methods, iTryOn has much more realistic garment/person interactions

Clarity:
- The authors provide clear details about their implementation, training details, and choice of models
- Contributions are well-justified and ablated

Significance:
- Interactive VVTO is a key challenge that could greatly increase realism of try-on videos, if solved.

### Weaknesses
- There are some noticeable warping artifacts in the supplementary videos, especially near the garment/person boundary
- In most examples, the garments in the input image and input video are very similar in shape and fit. More examples should be provided where there is significant shape change (e.g. short to long sleeves, long to short garment, etc.), or this should be listed as a limitation of the method.
- Similarly, how does the method work if the interaction is implausible? For example, unzipping a shirt without a zipper? This should also be listed as a limitation of the method.
- VVT-Interact (based on the examples shown) is limited in diversity of appearance and body shapes
- The test dataset for VVT-Interaction s only 180 videos, which is only ~2% of the training videos. I think this  small scale limits how reliable it can be as a benchmark.
- In the comparisons in Figure 5, the iTryOn garment looks a bit over-saturated w.r.t. the input garment for both examples, while MagicTry-On seems to have better garment fidelity. Perhaps this should be addressed?

### Questions
- There does not seem to be much validation provided of the VLM used for extracting video annotations. Although such evaluation may be tricky, it seems essential to the method that the annotations are valid, so I would suggest finding some way to validate that the annotations are reliable, such as through human qa.
- In section 3.5 (lines 316-318), why is A-RoPE only applied to some keys, but all queries?
- Since A-RoPE is based on 1D-RoPE, I recommend adding a brief description about 1D-RoPE as a preliminary, for example.
- I am surprised that AC-Loss does not improve the performance more in the quantitative ablations (Table 3). Is this because the evaluation datasets are not focused on interactive video-clips? That is, if the evaluation dataset consisted only (or mostly) of interactive clips from the dataset, would AC-loss make a bigger improvement? In general, it would be interesting to see the benefit of each contribution specifically for handling interactive clips.
- Currently, the interactive motion is always replicated from the input video. Future work with this dataset could include editing or adding interaction to an input video using text annotations.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Interactive Video Virtual Try-On (Interactive VVT), targeting scenarios where subjects actively manipulate garments (e.g., pulling hems, unzipping), and highlights challenges in resolving interaction semantics and modeling sparse, complex deformations. 
It proposes iTryOn, a video diffusion Transformer with multi-level interaction injection: a garment-agnostic 3D hand prior for precise hand–garment contact, and global plus time-stamped action captions aligned via Action-aware Rotational Position Embedding (A-RoPE). 
An action-aware constraint loss further stabilizes training and emphasizes key interactive frames.
The authors also release the VVTInteract dataset and report state-of-the-art results on standard VVT and substantial gains in the interactive setting.

### Strengths
- The paper is well-structured, clearly written, and easy to follow.
- The problem addressed—interactive video virtual try-on—is timely and important for the VVT community.
- The authors introduce a new large-scale dataset that enables rigorous evaluation and future research on this task.

### Weaknesses
- In the practical use of the system, users only provide a reference video and a target clothing image. Fine-grained conditions such as 3D hand pose and human pose can be automatically detected by existing models. However, the acquisition of global/action prompts remains unclear; it is ambiguous whether these are also user inputs or obtained otherwise.
- The use of the term “interaction” in the paper is ambiguous. While it is commonly interpreted as interaction between the system and users, the paper refers to interaction between a person and clothing, which is unconventional and potentially confusing.
- The paper addresses the "how" of an interaction by introducing 3D hand pose as an additional condition, yet this approach lacks novelty. More importantly, it remains unclear why the interaction between the task and clothing can be sufficiently defined only by hand movements.
- The paper addresses the "what" the type of action by defining several interaction types using captions. However, the generalizability of this approach is questionable, as the proposed method may not scale to a broader range of interactions.
- The paper solves the "when" precise timing by ensuring that action prompts interact only with interaction frames and not with non-interaction frames. Nevertheless, in practical scenarios, it is not specified how interaction frames are accurately identified.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces iTryOn, a novel framework for Interactive Video Virtual Try-On that addresses the limitation of existing methods in handling active human-garment interactions. The authors formalize a new task where subjects actively engage with clothing (e.g., pulling hems, unzipping jackets). The framework is built upon Wan2.1-VACE with novel components including A-RoPE and an action-aware constraint loss.

### Strengths
1. The paper introduces Interactive VVT as a new and challenging task that bridges the gap between passive garment display and real-world e-commerce scenarios with active human-garment interactions.

### Weaknesses
1. **Limited Technical Innovation**: While the paper presents novel guidance mechanisms, the underlying diffusion transformer architecture closely follows existing designs (Wan2.1-VACE) without fundamental architectural innovations. The contributions are primarily in the conditioning and guidance layers rather than core generative modeling advances. In fact, the Wan2.1-VACE model itself can achieve video virtual try-on functionality. The author needs to explain this and provide performance comparisons.

2. **Insufficient Ablation Study**: The ablation study lacks detailed analysis of individual components. Critical hyperparameters like the A-RoPE separation scale (k=4) and action constraint loss weight (λ=0.5) are presented without sensitivity analysis or justification for these specific choices.

3. **Unfair Comparison**: Directly comparing iTryOn with the baseline methods may not be entirely fair in Tables 1-2 , given that iTryOn benefits from a substantial amount of extra training data. This raises uncertainty regarding whether the observed improvements stem from the added data or the unique architecture of iTryOn. To ensure a thorough and impartial assessment, it is recommended that the authors re-train their approach only using publicly accessible datasets like VVT and ViViD.

4. **Computational Efficiency**: Despite claims of parameter efficiency (2B vs competitors' 14B), the paper lacks detailed analysis of inference time, memory requirements, and computational complexity compared to baseline methods. The practical deployment feasibility remains unclear.

5. **Lack of specialized metrics for interaction fidelity**: Although the author introduced the new task of Interactive VVT, the evaluation metrics still rely on generic video/image metrics (such as VFID, SSIM). The authors fail to provide appropriate metrics to quantify the physical plausibility of garment deformations (e.g., distinguishing realistic stretching from visual coherence).
6. **Imbalanced dataset distribution**: Over 70% of samples belong to "Adjusting the collar" and "Adjusting the hem", leading to potential bias in model generalization across rare interaction types (e.g., "Other interactions" at 2.98%).

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a video tryon method where the human interacts with the garments noticeably. It is done by (1) modeling the hands explicity with 3D hand mesh model, and (2) a action-position-encoding to bias the generation on the motion rather than un-action words. They also introduce a dataset specifically for tryon with hands-cloth interaction. They showed improved results over prior state of art for video tryons.

### Strengths
1. Supporting human interaction with the garments is a important yet natural task for virtual tryon. The VVT-interact dataset is an important next step towards this goal.
2. The action aware semantic guidance is necessary to avoid the model from being biased by non-action words in the prompt.

### Weaknesses
1. The proposed method requires an existing video to run tryon. Depending on the motion of this video, there could be incompatiblity with the tryon garments,  say roll up sleeve motion when the garment is a short-sleeve, or unzip the jacket when the garment is a t-shirt. Ideally, we should be able to use text prompt to control the motion of the user in the source video such that we can select the garment that is compatible with the prompt.

2. The method uses explicit 3D hand mesh to condition try video generation model. This estimated hand model is often misaligned with the actual image. This is problematic because the video has to generate the hand that matches the rgb hand pixels of the exiting source video.

3. It was unclear why the existing clothing is not "delcoth", leading to bleeding problem seen in Fig. 3

### Questions
1. Line 200-201: looks like the action caption is determined by per-frame by a VLM. It is hard to determine action or motion with only 1 frame.
2. Vivid is a dataset with limited to no hands and cloth interaction. Why does the model outperform existing method on this data, as shown in Tab 2, especially with the smallest model capacity (2B)?

### Soundness
2

### Presentation
3

### Contribution
3
