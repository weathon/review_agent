# Cross-ControlNet: Training-Free Fusion of Multiple Conditions for Text-to-Image Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8, 4

## Abstract
Text-to-image diffusion models achieve impressive performance, but reconciling multiple spatial conditions usually requires costly retraining or labor intensive weight tuning.
We introduce Cross-ControlNet, a training-free framework for text-to-image generation with multiple conditions.
It exploits two observations: intermediate features from different ControlNet branches are spatially aligned, and their condition strength can be measured by spatial and channel level variance.
Cross-ControlNet contains three modules: PixFusion, which fuses features pixelwise under the guidance of standard deviation maps smoothed by a Gaussian to suppress early-stage noise; ChannelFusion, which applies per channel hybrid fusion via a consistency ratio gate, reducing threshold degradation in high dimensions; and KV-Injection, which injects foreground- and background-specific key/value pairs under text-derived attention masks to disentangle conflicting cues and enforce each condition faithfully.
Extensive experiments demonstrate that Cross-ControlNet consistently improves controllable generation under both conflicting and complementary conditions, and further generalizes to the DiT-based FLUX model without additional training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
* The paper proposes Cross-ControlNet, a training-free framework for fusing multiple conditional branches in T2I generation.
* It is built upon two key observations: the spatial alignment across ControlNet branches and the variance-based condition strength.
* The framework introduces three modules:
  * PixFusion for pixel-level feature fusion guided by Gaussian-smoothed variance maps,
  * ChannelFusion for adaptive hard, soft fusion based on channel-wise consistency ratios,
  * KV-Injection for foreground-background disentanglement using text-derived attention masks.
* Without any additional training, Cross-ControlNet achieves robust controllable generation under both complementary and conflicting conditions.
* It outperforms existing training-free methods such as MaxFusion, AnyControl, and Uni-ControlNet, and generalizes to DiT-based models.

### Strengths
* The method is training-free, making it practical and computationally efficient for multi-condition control.
* The variance-guided fusion offers an intuitive yet mathematically grounded mechanism for balancing control strength and spatial coherence.
* The KV-Injection module elegantly leverages textual attention maps to isolate and refine foreground and background regions.
* The paper provides comprehensive ablations that demonstrate the contribution of each module.
* Quantitative results show clear improvements over baselines.

### Weaknesses
* The framework is sensitive to several hyperparameters, but their selection rationale is largely empirical.
* The approach inherently depends on pre-trained single-condition ControlNets, making it less flexible when new modalities are introduced.
* Using multiple ControlNet branches may increase inference latency and GPU consumption.

### Questions
* How stable is the variance-based fusion under time-dependent noise variations during the diffusion process?
* While generalization to DiT-based models is promising, how does cross-branch fusion behave in transformer-based architectures?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This submission introduces a novel method for combining conditioning signal from several ControlNets in a single diffusion model. The proposed approach uses two main observations: that different ControlNets are spatially aligned and that the strengths of individual conditions can be quantified via feature variances, that combined with KV-injection allows for accurate fusion of spatial conditions.

### Strengths
- There is a clear introduction to the main idea behind the submission in section 3.1. Presented observations are sound and provide great motivation for the proposed solution.

### Weaknesses
- The usability of the proposed technique is very limited to the specific task of fusing multiple spatial conditions for T2I models
- The proposed technique is a combination of several “add ons” that build on top of the standard controlnet, this slightly limits the novelty of the proposed method
- Main experiments are performed with quite old SD 1.5. While the qualitative results with FLUX are impressive, the submission lacks proper comparison with other approaches using this model.
- The experimental section focus mostly on a setup with clear separation between foreground and background which highly utilizes the KV-injection technique. It would be great to see some results with more complex compositions


Smal issue:
(Presentation) - Lines 60-64 in introduction are copies from the abstract. As I didn’t fully understand what authors meant in the abstract, it was not easier when presented with exactly the same sentence for the second time.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Cross-ControlNet, a training-free framework for fusing multiple spatial conditions in text-to-image diffusion models. It introduces three modules—PixFusion (pixel-wise fusion guided by smoothed spatial variance), ChannelFusion (channel-wise adaptive fusion using a consistency ratio), and KV-Injection (foreground–background disentanglement via attention masks), to handle both conflicting and complementary conditions.

### Strengths
1. PixFusion uses Gaussian-smoothed spatial standard deviation to guide pixel-wise selection, suppressing early-stage noise as validated by Fig. 3b showing cleaner fused variance maps. ChannelFusion and KV-Injection are also resonable modules.

2. Qualitative results in Fig. 4 show faithful preservation of both teddy bear pose and train window structure, where baselines fail to reconcile conflicting cues. Other qualitative results also performs better than other baslines.

3. The framework can transfer to DiT-based FLUX without modification, producing sharp, controllable images, demonstrating architecture agnosticism.

### Weaknesses
1. No inference time or memory usage is reported despite combining multiple ControlNet branches

2. The claim that ChannelFusion is applied only in the final layer (Sec. 4.1) is not justified; no ablation tests layer-wise placement (e.g., middle vs. final layer).

### Questions
Please see the weakness.

### Soundness
2

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
4

### Summary
The paper presents Cross-ControlNet, a training-free framework designed for multi-condition text-to-image generation. The framework integrates multiple spatial conditions without retraining, using three main components: PixFusion, ChannelFusion, and KV-Injection. These modules allow for robust feature fusion while handling both conflicting and complementary conditions, with demonstrated improvements in controllable generation. The method improves performance over existing models, notably in generating high-quality images under complex, multi-condition prompts.

### Strengths
1. The paper presents an innovative solution to the challenge of multimodal conditional image generation by introducing Cross-ControlNet, which uses a combination of PixFusion, ChannelFusion, and KV-Injection to enhance feature fusion and address the issues of noise sensitivity and high-dimensional fusion degradation. 
2. The framework is compatible with existing models (e.g., DiT-based architectures), demonstrating its generalization potential and ability to adapt to different underlying network backbones.
3. The experiments are comprehensive, covering both quantitative metrics (e.g., mIoU, MSE) and qualitative results. The method consistently outperforms state-of-the-art methods, such as MaxFusion, Multi-ControlNet, and AnyControl, particularly under conflicting conditions.

### Weaknesses
1. The model’s architecture, which combines multiple ControlNet branches, leads to significant increases in memory consumption and inference time. This makes it challenging to deploy Cross-ControlNet for high-resolution image generation or real-time applications where computational efficiency is critical.

2. While the method works well under conflicting and complementary conditions, extreme conflicting signals may still cause artifacts, such as foreground-background misalignment or color bleeding. The paper could further explore ways to mitigate these extreme cases.

### Questions
A more concise explanation of why these components are crucial for multimodal consistency could strengthen the paper's readability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper explores training-free methods to combine multiple ControlNet conditions via fusing ControlNet outputs of multiple branches. This is done through either PixFusion or ChannelFusion. They also use KV-Injection, to separate the foreground/background more clearly to avoid conflicting requirements.

### Strengths
- Training-free method, so quick to test/implement for use
- Works on any architecture that ControlNet works upon
- Results show improvement over baselines on SD 1.5

### Weaknesses
- Quantitative results for only SD 1.5, though this may be due to ControlNet restrictions
- Combining the end product of multiple branches will increase latency, though no values are given as to the cost
- Could be good to see results without the use of the Gaussian Kernels - how much does this use contribute to the results?
- The paper assumes knowledge of ControlNet implementation - perhaps some time could be spent on introducing it since the paper relies on it entirely
  - I found the paper hard to follow in places and so was unsure of the exact methodology used - see questions

### Questions
Clarifications:
- How are the spatial-level variance maps calculated for PixFusion?
- For KV-Injection, the terms "background and foreground ControlNets" are introduced. Does this mean that KV-Injection only works for combinations of ControlNets that work on foreground/background, and not for two ControlNets that both only focus on the foreground?
- In equation (9), Q1 is not used. Is this correct?
- In Figure 3a, the x axis is referred to as the "depth of the feature layer", which I am taking to mean the depth of the block in the model, but the text referring to it talks about the dimensionality of the feature space, which does not seem to be the same thing?


Suggestions:
- My understanding for ChannelFusion is that each channel is over all the tokens. Could it not be that the channel would be more useful in some spatial areas than others for each ControlNet, especially if the nets do not overlap?
- For both PixFusion and ChannelFusion, the threshold is a binary choice between taking only one ControlNet, vs an averaging approach. Could a less hard boundary be used instead, such as an interpolation or sliding scale? Perhaps the lack of a hyperparameter in this way would avoid the threshold degredation?

### Soundness
2

### Presentation
2

### Contribution
2
