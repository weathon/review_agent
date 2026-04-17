# MATRIX: Mask Track Alignment for Interaction-aware Video Generation

- Decision: Accept (Poster)
- Scores: 2, 8, 8

## Abstract
Video DiTs have advanced video generation, yet they still struggle to model multi-instance or subject-object interactions. This raises a key question: How do these models internally represent interactions? To answer this, we curate MATRIX-11K, 
a video dataset with interaction-aware captions and multi-instance mask tracks. Using this dataset, we conduct a systematic analysis that formalizes two perspectives of video DiTs: semantic grounding, via video-to-text attention, which evaluates whether noun and verb tokens capture instances and their relations; and semantic propagation, via video-to-video attention, which assesses whether instance bindings persist across frames. We find both effects concentrate in a small subset of interaction-dominant layers. Motivated by this, we introduce MATRIX, a simple and effective regularization that aligns attention in specific layers of video DiTs with multi-instance mask tracks from the MATRIX-11K dataset, enhancing both grounding and propagation. We further propose InterGenEval, an evaluation protocol for interaction-aware video generation. In experiments, MATRIX improves both interaction fidelity and semantic alignment while reducing drift and hallucination. Extensive ablations validate our design choices. Codes and weights will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The current work investigates how the object interactions are represented inside the video diffusion transformers and proposes a training strategy to refine them. More concretely, it investigates the attention map inside the CogVideoX-5B-I2V model in terms of text/visual tokens interactions, and observe that there is some object grounding (some spatial visual tokens have higher attention scores to the object/subject text prompt) and interaction grounding (the interaction verb attends to both the object and the subject). The curate a dataset of 11K videos and annotate instance-level segmentation mask tracks to quantify this grounding. This allows to locate the layers where grounding is the highest, and also to evaluate the interaction quality later. To improve the interaction modeling, the authors propose extra regularizations on attention maps pushing them closer to the correctly grounded segmentation maps.

### Strengths
- The paper is well written and easy to follow, and the illustrations are high-quality
- The proposed dataset could be useful for simply fine-tuning to improve the human-object and human-human interactions
- The included attention visualizations could be useful for future references
- The overall topic is quite important to the community

### Weaknesses
- The fact that text tokens are grounded to visual tokens in attention to some extent is a trivial observation and dates back to StableDiffusion-V1 editing pipelines for UNet+Attention architectures. The same observation was made for convolution-free architectures as well (i.e., DiT/UViT). One of the recent references is https://arxiv.org/abs/2505.04320 where it's visualized for Flux. I do not think that it's a surprising observation that similar mechanics happens for video DiTs. I would be more surprised if it would be something opposite. That's why the analysis is not insightful.
- The paper uses outdated baselines (e.g., CogVideoX-5B) instead of the modern ones (e.g., Wan 2.1/2.2)
- The paper does not compare to the baseline of simply fine-tuning on their curated data. I believe it's a critical ablation
- The submission does not include any samples from the dataset, making it hard to assess its quality
- The overall visual quality is far behind modern open-source models (e.g. Wan, Hunyuan, etc).

### Questions
- Typo: "suject-object" (the first sentence in the abstract)

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper analyzes a common problem in video models, incorrect multi-object interactions. It identifies key concepts that allow analyzing two aspects of this problem (semantic grounding and propagation), and uses these concepts and their corresponding metrics to understand how these interactions are represented inside a typical DiT model. Based on this analysis and metrics, the paper introduces 1) a method for regularization (Matrix), 2) an evaluation protocol (InterGenEval), and  3) a dataset with 11k videos (Matrix-11k) with properly labeled interactions and instances.

The method for effective regularization is based on locating the layers that most impact on semantic grounding and propagation, and fine-tuning those using LoRA on the corresponding losses for those two concepts: Semantic Grounding Alignment and Semantic Propagation Alignment.

### Strengths
This paper is a strong work in all the important dimensions:

* originality: to the best of my knowledge this is novel work
* quality: the problem is analyzed from an interesting perspective, starting with an analysis that provides a new understanding on how the interactions are represented in particular layers of the network, and the solution (introducing new concepts, a regularization method, a dataset, and an evaluation benchmark) presented properly deals with the problem
* clarity: the paper is well written and easy to follow
* significance: this is where I doubted the most, because while the problem is important for video generation, it was unclear to me that per se it would justify the added complexity in the training system (even if based on something as lightweight as LoRA). However, the use of this method is also beneficial for overall quality and other important metrics, which is intriguing, impactful, and to me brings final justification to this additional complexity.

### Weaknesses
* There is no analysis about how this method impacts training time. Especially important would be to understand whether the benefits of fine-tuning are as strong as presented, when compared to a training run with the same amount of flops used (iso-FLOPS comparison)

* The dataset used for eval, based on only 60 synthetic and 60 real pairs appears too small. It would be useful to include analysis on what to expect, in terms of noise, from this dataset size

* While ablations for most important aspects of this work are provided, there's little justification for why Av2t  and Av2v are the best proxies for semantic grounding and propagation. Could one use At2v for the former, or even reuse the Av2t for the latter? What other alternatives were considered? While it's not reasonable to expect a completely exhaustive ablation for all possible alternatives, even a small explanation would improve on this.

* this work focuses on actions that require contact between different objects in the videos, such that the fine-tuning process only includes such actions in the dataset. This is a reasonable assumption for this particular paper (but as a minor improvement, it could be interesting to consider how the work could be expanded to better support contact-less interactions). However it would be useful to understand how this choice impacts other model properties, more specifically:
  * is the fine-tuned model more prone to generate contact actions even when provided with a contactless prompt?
  * minor: it would be interesting to understand, also, whether a fine-tuned model improves on contactless actions as well, even if those aren't included in the expanded loss

* similarly, since the primary source HOIGen is described to be mostly focused on human-based interaction,  it seems useful to ask what's the impact on non-human video generation

* minor: given the strong results using LoRA, the paper could benefit from some discussion about whether using the same method on large scale pretraining

### Questions
* What are your insights regarding which layers are the most influential? do you find that they are consistently placed around the same locations in different kinds of networks?

* Have you considered how general this method is; would you expect similar results in non-DiT based networks for example?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigate why t2v diffusion transformers struggle with multi instance, subject object interactions and proposes a targeted fix. The authors proposed MATRIX-11K, a dataset of ~11K videos with interaction-aware captions and multi instance mask tracks. And analyzed where interaction semantics emerge inside 3D full attention of video DiTs, finding that semantic grounding and semantic propagation concentrate in a small set of layers, and introduced the lightweight regularization that aligns attention in those layers to the mask tracks via SGA and SPA loss. On their benchmark, MATRIX improves interaction fidelity and reduces drift and duplication.

### Strengths
1. A deep and comprehensive clear failure analysis, and the MATRIX fixed that.
2. The data pipeline of the seg data generation is well designed, pragmatic and reproducible.
3. Simple, effective objective. The SGA and SPA losses are standard but cleverly applied to attention maps rather than pixels alone, brings lower computational cost and easy to adapt into Dit.

### Weaknesses
1. It mainly compared and analyse with CogVideoX, as this work is a 2025 work, it probably need to do more with the more recent models like wan and hunyuan.
2. 11K dataset may not that a good amount for the video generation tasks, especially for those larger and new models, the society may be benifit if the dataset could be larger
3. Like the other related work videojam or udpdiff, they used optical flow, seg, depth to help with the video generation, while in this work only segmentation is used, consider to include discussion on involving others would beneficial.

### Questions
see weakness

### Soundness
4

### Presentation
3

### Contribution
4
