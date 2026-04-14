# Zero-Shot Subject-Driven Video Customization with Precise Motion Control

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Recent advances in customized video generation have enabled users to create videos tailored to both specific subjects and motion trajectories. However, existing methods often require complicated test-time fine-tuning and struggle with balancing subject learning and motion control, limiting their real-world applications. In this paper, we present $\textbf{DreamCustomizer}$, a zero-shot video customization framework capable of generating videos with a specific subject and motion trajectory, guided by a single image and a bounding box sequence, respectively, and without the need for test-time fine-tuning. Specifically, we introduce reference attention, which leverages the model’s inherent capabilities for subject learning, and devise a mask-guided motion module to achieve precise motion control by fully utilizing the robust motion signal of box masks derived from bounding boxes. While these two components achieve their intended functions, we empirically observe that motion control tends to dominate over subject learning. To address this, we propose two key designs: $\textbf{1)}$ the masked reference attention, which integrates a blended latent mask modeling scheme into reference attention to enhance subject representations at the desired positions, and $\textbf{2)}$ a reweighted diffusion loss, which differentiates the contributions of regions inside and outside the bounding boxes to ensure a balance between subject and motion control. Extensive experimental results on a newly curated dataset demonstrate that DreamCustomizer outperforms state-of-the-art methods in both subject customization and motion control. The dataset, code, and models will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a video customization framework, DreamCustomizer, which can generate videos based on a specific subject and motion trajectory.
The framework incorporates two primary components: reference attention for subject learning and a mask-guided motion module for precise motion control.
Experiments demonstrate that DreamCustomizer outperforms state-of-the-art methods in both subject customization and motion control.

### Strengths
- The paper is well-written and the figures are properly plotted. Both of them make readers easy to understand the proposed method.

- The experiments are extensive, covering the quantitative and qualitative comparisons with the state-of-the-arts, human evaluation, testing under different conditioning scenarios, and ablation study.

- The authors provide a lot of generated videos, which can help the reader better understand the quality and the potential problems of the proposed method.

- The model is an optimization-free method, which does not require fine-tuning during inference stage.

- The motion control of the proposed model is precise and impressive.

### Weaknesses
- Low visual quality:
    - The generated videos have significant artifacts. Did the authors try a better and more recent base model, such as VideoCrafter2?
    - With this video quality, it is hard to judge the fidelity of subject customization.

- Weakness of reconstruction-based method:
    - The model adopts a reconstruction-based training strategy, which collects the input conditions (reference image and bounding boxes) from the video and learn to reconstruct the video with these inputs.
    - However, such training strategy is well-known by worse performance when the user inputs prompt and image from different sources.
    - For example, in the "a corgi is swimming" sample, the model seems to directly copy and paste the input standing corgi and make it floating on the water without introducing reasonable pose or appearance change.
    - Have the authors tried to solve this problem? Applying some image augmentations could alleviate the problem.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces DreamCustomizer to generate video with precise motion control and specific subject without the need for fine-tuning during inference. DreamCustomizer leverages "reference attention" and a "mask-guided motion module" to achieve accurate video customizations, controlled by a single subject image and bounding box sequences. The methodology uses masked reference attention and a reweighted diffusion loss to balance subject learning and motion control. The paper claims superior performance over state-of-the-art methods by a new dataset and extensive quantitative and qualitative evaluations.

### Strengths
1. The paper distinguishes and mitigates the challenge of motion control dominance by introducing masked reference attention and reweighted diffusion loss, successfully balancing subject fidelity with motion accuracy.

2. The newly curated, diverse dataset provides comprehensive annotations, facilitating training and evaluation for subject and motion control and supporting future research in video customization.

3. With extensive quantitative and qualitative evaluations, the paper demonstrates the effectiveness of DreamCustomizer against state-of-the-art methods.

### Weaknesses
1. While DreamCustomizer introduces elements like reference attention and reweighted diffusion loss, these techniques lack substantial novelty. Reference attention, for instance, has been well-studied in prior works such as "StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation," where it has been used to effectively maintain subject consistency. Similarly, the reweighted diffusion loss does not introduce a novel approach to balancing subject fidelity with motion control, as similar weighting techniques have been explored in generative models. 

2. DreamCustomizer requires bounding boxes as inputs for motion guidance, similar to control techniques seen in previous works like MotionBooth. This limitation reduces its capability for more complex and high quality video generation tasks and the proposed method is incremental to resolve this problem.

3. Moreover, the presented video quality is not convincing to demonstrate that these proposed components are instrumental in advancing controllable video generation.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a zero-shot approach to subject-driven video generation with controlled motion, showing potential for reducing test-time fine-tuning overhead. The reference attention mechanism and mask-guided motion control provide innovations over previous methods by enhancing the fidelity of subject appearance within controlled bounding boxes.

### Strengths
- The DreamCustomizer framework proposed in this paper does not require fine-tuning during inference, and can directly customize the target subject and motion trajectory in a zero-shot situation. 

- This generation framework that does not require fine-tuning improves the efficiency of video generation and is conducive to a wide range of practical applications.

- Users only need to provide the subject image and a set of bounding box sequences to generate a custom video, without the need for complex inference stage debugging.

### Weaknesses
- While DreamCustomizer claims tuning-free inference, the paper acknowledges the challenge of decoupling camera movement from object motion, leading to camera drift in certain contexts.

-  DreamCustomizer is designed for single-subject customization and does not handle multi-subject videos, which is highlighted as a limitation but without proposed extensions.

- The effectiveness of motion control relies heavily on the precision of bounding box annotations, making the system vulnerable to errors if box tracking is inconsistent.

- Missing some important previous works:

[1] MotionFollower: Editing video motion via lightweight score-guided diffusion. (2024).

[2] FaceChain-ImagineID: Freely crafting high-fidelity diverse talking faces from disentangled audio. CVPR 2024

[3] MotionEditor: Editing video motion via content-aware diffusion. CVPR 2024

[4] Combo: Co-speech holistic 3D human motion generation and efficient customizable adaptation in harmony.  (2024).

### Questions
- The paper mentions data filtering steps, but additional details on the refinement of bounding boxes and control signals could provide readers with greater insight into handling imperfect data in training.

- Given the limitations in complex motions and single-subject restriction, it's recommended that the paper discuss specific applications that could benefit from these features as they currently exist.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces DreamCustomizer, a video customization model capable of generating videos with a specific subject and motion trajectory. Technically, DreamCustomizer incorporates reference attention into the T2V model to learn the appearance information of the subject, and employs a mask-guided motion module to achieve motion control based on a sequence of bounding boxes. Additionally, the paper proposes masked reference attention and reweighted diffusion loss to emphasize the model's focus on learning from the subject input. Extensive experiments conducted on a newly curated dataset demonstrate the effectiveness of DreamCustomizer for subject customization and motion control.

### Strengths
S1: The paper introduces DreamCustomizer, a video generation method capable of controlling both the appearance of the generated subject and the motion trajectory. Additionally, it proposes two strategies, masked reference attention and reweighted diffusion loss, to enhance the model's focus on learning subject appearance. 

S2: DreamCustomizer constructs a dataset from WebVid-10M for training, enabling subject customization and motion control during inference without the need for fine-tuning.

S3: The paper presents good qualitative and quantitative results.

### Weaknesses
W1: The description of the dataset used for evaluation in Section 5.1 could benefit from further clarification. I would appreciate it if the authors could specify the total number of subject-BBox pairs. Additionally, the statement "we design 60 textual prompts for validation" raises a question for me: does this imply that there are only 60 pairs used for evaluation? It seems that each input pair would typically have a corresponding text prompt.

W2：I’m unsure if I missed this detail, but I would like to know whether Tables 2, 3, and 4 are evaluated on the same validation set. If so, I’m curious why the values for DreamCustomizer differ for the same metrics, such as CLIP-T and Temporal consistency, across the three tables.

W3：While the qualitative comparison indicates that DreamCustomizer achieves better mIoU and CD, Figure 10 shows that the model often generates camera movement rather than subject movement. This type of camera movement is not the intended form of motion in this paper and typically results in low CD values. My concern is that the metrics CD and mIoU may not accurately assess the appropriateness of the generated video’s motion (e.g., distinguishing between camera movement and subject movement), potentially reducing the credibility of these metrics.

### Questions
Q1: I hope the authors can address the concerns raised in the "Weaknesses" section.

### Soundness
3

### Presentation
2

### Contribution
2
