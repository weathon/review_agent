# PickStyle: Video-to-Video Style Transfer with Context-Style Adapters

- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
We address the task of video style transfer with diffusion models, where the goal is to preserve the context of an input video while rendering it in a target style specified by a text prompt. A major challenge is the lack of paired video data for supervision. We propose PickStyle, a video-to-video style transfer framework that augments pretrained video diffusion backbones with style adapters and benefits from paired still image data with source–style correspondences for training. PickStyle inserts low-rank adapters into the self-attention layers of conditioning modules, enabling efficient specialization for motion–style transfer while maintaining strong alignment between video content and style. To bridge the gap between static image supervision and dynamic video, we construct synthetic training clips from paired images by applying shared augmentations that simulate camera motion, ensuring temporal priors are preserved. In addition, we introduce Context–Style Classifier-Free Guidance (CS–CFG), a novel factorization of classifier-free guidance into independent text (style) and video (context) directions. CS–CFG ensures that context is preserved in generated video while the style is effectively transferred. Experiments across benchmarks show that our approach achieves temporally coherent, style-faithful, and content-preserving video translations, outperforming existing baselines both qualitatively and quantitatively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PICKSTYLE, a video style transfer framework that efficiently adapts pre-trained video diffusion models via low-rank style adapters and a novel guidance strategy, achieving superior results by leveraging paired image data and simulating temporal coherence.

### Strengths
The Core Innovations are as follows:

1. Efficient Low-Rank Adaptation: The integration of specialized style adapters into self-attention layers enables effective motion-style transfer while maintaining computational efficiency and strong content-style alignment.

2. Bridging Image-Video Domain Gap: A novel synthetic clip construction strategy using shared augmentations that simulate camera motion, effectively leveraging static image supervision for dynamic video stylization.

3. Factorized Classifier-Free Guidance: The proposed Context-Style Classifier-Free Guidance (CS-CFG) innovatively disentangles style and context control, ensuring precise style application without compromising video content integrity.

### Weaknesses
1. The proposed 2D motion simulation may lack generalizability for complex real-world videos involving significant 3D perspective changes. The current experiments, as shown in Figure 7, are primarily validated on relatively simple motions, leaving its performance on more complex scenarios unverified.

2. The decision to keep the text-video cross-attention module frozen is a potential limitation. If the base model was not exposed to certain style descriptions during its pre-training, the framework might struggle to establish a correct correspondence between novel textual style prompts and their visual manifestations.

### Questions
Could you clarify the design choice for C_null? Specifically, why was shuffling frame orders adopted instead of using a null context (empty set)? Furthermore, what is the rationale behind the specific form of the context direction, why don't use this formulation as the context direction: ϵ_cond - ϵ_theta(x_t, t; T, C_null), and were other alternative formulations explored or ablated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PickStyle, a video style transfer framework using diffusion models to render videos in a text-specified style while preserving content, overcoming the lack of paired video data. It achieves this by augmenting video diffusion backbones with low-rank style adapters and training on synthetic video clips created from paired images using simulated camera motion. Context–Style Classifier-Free Guidance (CS–CFG) is also introduced, which independently guides the style (text) and context (video) directions, resulting in superior, temporally coherent, and content-preserving style translations.

### Strengths
The paper presents end-to-end, feedforward video style transfer network.
The dataset construction strategy would be useful to the research community.

### Weaknesses
- All baseline methods are built on far inferior backbones (some are based on t2i backbone) and it’s not a surprise that PickStyle-based on VACE (WAN) beats the selected baselines. More recent baselines or any methods thats applied on the same WAN backbone is needed.
- The base model, VACE inherently cannot perform video style transfer? If so, can we see how PickStyle is improved compared to the original VACE backbone?
- Compared to the normal CFG, how much more computation overhead does CS-CFG incur in terms of both memory and time?
- My biggest concern is the lack of technical contribution. The adapter module is ControlNet-style network, also frequently adapted and used in recent DiT-based generation methods. The CFG with additional condition term (triangular CFG) is not new either. For example, SV3D or VideoJAM uses similar approaches.

### Questions
Please see the weakness above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
PickStyle introduces a video-to-video style transfer framework that preserves motion and context while rendering stylized frames from one of nine trained styles. The method’s primary innovation seems to be a context-style classifier-free guidance mechanism, allowing explicit control over content and style conditioning during diffusion. Additionally, a tunable noise initialization strategy enables improved temporal coherence and perceptual fidelity. The paper includes reasonable experiments demonstrate improvements over prior methods in both qualitative and quantitative metrics, including a standard battery of metrics across content, video quality, etc.

### Strengths
S1 The paper builds on the Wan2.1 generation backbone to include style adaptation and seems to be able capture nine different styles in a way that generalizes from image-pairs to video.  The paper makes reasonable technical innovations to accomplish this that seem original and relevant, such as CS-CFG which creates a tunable trade-off between fidelty and stylization (although this trade-off seems not analyzed in the paper).

S2 The paper includes a motion augmentation strategy that enables the use of image pairing as training data.

S3 Results seem compelling and meaningful analyses are included.  It seems clear that PickStyle is best at being able to match the style prompt, at least according R Precision score (although it is curious that this score only uses one frame from the video).  It also seems that the video quality aspects are strong.

### Weaknesses
W1 The key aspects of the proposed method seem to the LoRA adapters that modulate the attention to capture style, the CS-CFG method and the noise initialization.  Yet, none of these are really thoroughly analyzed in any way.  For example although Fig. 8 captures one example of where CS-CFG helps, the interplay between $t_\text{guide}$ and $c_\text{guide}$ is not studied.  Similarly, we have no evidence about to what degree the context-based initialized is necessary.  Hence it is impossible to actually assess whether the technical innovations align with the observed results improvements, or if it is from other sources (e.g., the different data, the augmentation approach, etc.)

W2 It is not clear whether the comparisons are fair.  Considering this paper creates a composite dataset with nine styles, have the other methods to which the paper compares been retrained on this dataset?  The paper does not sufficiently describe this critical point.

W3 The approach to augment the paired image samples with some motion to generating pair training videos is not well described in the text and therefor hard to analyze.  It would seem, for example, that the types of augmentations used are not able to capture realistic motions in video resulting from 3D content and perspective effects.  This implies that perhaps the datasets used and results shown, however compelling they may be, may not be indicative of utility on more general video. 

W4 It seems that PickStyle is the most computationally expensive of the methods evaluated.


Minor things
- The manner in which the references are typically cited, e.g., "VACE Jiang et al. (2025)" is not proper, at least not for this style of including the author name.  These should be in parenthesis or better incorporated directly into the text.  VACE by Jiang et al. (2025) or VACE (Jiang et al. 2025).

### Questions
Q1 What would happen if multiple style prompts were given as input?  What would happen if an out of set style prompt were given?

Q2 What are the limitations of applying this to video?  Is there any reason to expect degradation for longer videos, for example?

Q3 Is the dataset created here publicly available?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces PICKSTYLE, a video-to-video style transfer framework leveraging diffusion models with context-style adapters. The method aims to preserve motion and context while translating videos into diverse styles, using paired still image data and synthetic motion augmentation for training. A novel Context–Style Classifier-Free Guidance (CS–CFG) mechanism is proposed to independently control style and context during generation. The approach is evaluated against several baselines, showing improvements in temporal coherence, style fidelity, and perceptual quality across multiple metrics and styles.

### Strengths
1) The qualitative and quantitative results demonstrate that PICKSTYLE achieves superior style transfer, temporal stability, and perceptual quality compared to existing baselines. 
2) The paper provides extensive quantitative and qualitative comparisons with multiple baselines, covering a wide range of styles and metrics.

### Weaknesses
1) The paper is difficult to follow in several key sections. The training procedure, especially how style and content consistency are achieved, is not clearly explained. The technical details of the model architecture and training pipeline are scattered and could benefit from a more structured presentation. 
2) The manuscript does not sufficiently highlight the core technical differences that make PICKSTYLE outperform baseline methods. The related work section is shallow, mostly listing existing approaches without deep analysis or positioning of the proposed method’s unique contributions. 
3) The training dataset is selectively curated, focusing on a limited set of styles (e.g., Anime, Pixar, Clay, LEGO, etc.) and synthetic Unity3D renderings. There is little discussion or evidence regarding the model’s ability to generalize to styles not covered in the training data, raising concerns about robustness and applicability.
4) It is unclear whether baseline methods were trained or fine-tuned on the same dataset as PICKSTYLE. Without this information, the fairness of the comparisons and the claimed superiority of the proposed method are questionable.

### Questions
Refer to the weakness part

### Soundness
2

### Presentation
1

### Contribution
2
